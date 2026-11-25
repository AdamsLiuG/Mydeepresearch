"""Standalone HTTP Sandbox Sidecar executing restricted Python workloads in an isolated subprocess."""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from services.sandbox_executor import SandboxASTValidator
from services.sandbox_runner import (
    ALLOWED_MODULES,
    MAX_CODE_CHARS,
    MAX_INPUT_BYTES,
)

logger = logging.getLogger(__name__)
_TRACEPARENT_RE = re.compile(r"^[0-9a-fA-F\-]{10,128}$")
_RUNNER_PATH = Path(__file__).with_name("sandbox_runner.py")


class SidecarExecutionRequest(BaseModel):
    """Strict schema for sandbox sidecar HTTP executions."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1, max_length=65536)
    tables: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    timeout_seconds: float = Field(default=10.0, gt=0.1, le=30.0, allow_inf_nan=False)
    memory_mb: int = Field(default=512, ge=64, le=2048)


def _sanitized_traceparent(header_value: str | None) -> str | None:
    raw = str(header_value or "").strip()
    if raw and _TRACEPARENT_RE.fullmatch(raw):
        return raw
    return None


def _verify_sidecar_auth(
    request: Request,
    *,
    auth_secret: str,
    allow_unauthenticated: bool,
) -> None:
    if not auth_secret:
        if allow_unauthenticated:
            return
        raise HTTPException(
            status_code=503,
            detail="sandbox sidecar auth secret not configured",
        )
    header = str(request.headers.get("authorization") or "").strip()
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
    provided = parts[1].strip()
    if not hmac.compare_digest(provided.encode("utf-8"), auth_secret.encode("utf-8")):
        raise HTTPException(status_code=401, detail="unauthorized sandbox sidecar token")


def _subproc_env() -> dict[str, str]:
    env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    path_val = os.getenv("PATH")
    if path_val:
        env["PATH"] = path_val
    return env


def create_sidecar_app(
    *,
    auth_secret: str | None = None,
    allow_unauthenticated: bool = False,
    max_timeout_seconds: float = 15.0,
    max_memory_mb: int = 512,
    max_code_bytes: int = 65536,
) -> FastAPI:
    """Create the isolated FastAPI Sandbox Sidecar application."""
    resolved_secret = (
        str(auth_secret if auth_secret is not None else os.getenv("V4_SANDBOX_SIDECAR_AUTH_SECRET", "")).strip()
    )
    clamped_timeout = max(1.0, min(30.0, float(max_timeout_seconds or 15.0)))
    clamped_memory = max(64, min(2048, int(max_memory_mb or 512)))
    clamped_code_bytes = max(1024, min(MAX_INPUT_BYTES, int(max_code_bytes or MAX_CODE_CHARS)))
    validator = SandboxASTValidator()

    app = FastAPI(title="DeepResearch Sandbox Sidecar", version="1.0.0")

    @app.get("/v1/capabilities")
    def get_capabilities(request: Request, response: Response) -> dict[str, Any]:
        _verify_sidecar_auth(
            request,
            auth_secret=resolved_secret,
            allow_unauthenticated=allow_unauthenticated,
        )
        traceparent = _sanitized_traceparent(request.headers.get("traceparent"))
        if traceparent:
            response.headers["traceparent"] = traceparent
        return {
            "service": "sandbox-sidecar",
            "version": "v1",
            "languages": ["python"],
            "max_timeout_seconds": clamped_timeout,
            "max_memory_mb": clamped_memory,
            "max_code_bytes": clamped_code_bytes,
            "allowed_modules": sorted(ALLOWED_MODULES),
        }

    @app.post("/v1/executions")
    async def execute_workload(request: Request) -> JSONResponse:
        _verify_sidecar_auth(
            request,
            auth_secret=resolved_secret,
            allow_unauthenticated=allow_unauthenticated,
        )
        started = time.perf_counter()
        raw_body = bytearray()
        max_body_bytes = min(MAX_INPUT_BYTES, clamped_code_bytes + 256 * 1024)
        async for chunk in request.stream():
            raw_body.extend(chunk)
            if len(raw_body) > max_body_bytes:
                raise HTTPException(status_code=422, detail="sandbox sidecar request payload exceeds limit")
        try:
            decoded = json.loads(raw_body)
            payload = SidecarExecutionRequest.model_validate(decoded)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid sidecar payload: {exc}") from exc

        if len(payload.code.encode("utf-8")) > clamped_code_bytes:
            raise HTTPException(status_code=422, detail="sandbox code exceeds max byte limit")

        traceparent = _sanitized_traceparent(request.headers.get("traceparent"))
        response_headers = {"traceparent": traceparent} if traceparent else {}

        ast_check = validator.validate(payload.code)
        if not ast_check.ok:
            return JSONResponse(
                status_code=200,
                headers=response_headers,
                content={
                    "ok": False,
                    "backend": "sidecar",
                    "error_code": "ast_rejected",
                    "error": "; ".join(ast_check.errors),
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "tables_out": [],
                    "charts": [],
                    "timed_out": False,
                    "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                },
            )

        effective_timeout = min(float(payload.timeout_seconds), clamped_timeout)
        effective_memory = min(int(payload.memory_mb), clamped_memory)
        runner_payload = {
            "code": payload.code,
            "tables": payload.tables,
            "timeout_seconds": effective_timeout,
            "memory_mb": effective_memory,
        }
        try:
            payload_bytes = json.dumps(runner_payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"non-serializable sandbox input: {exc}") from exc
        if len(payload_bytes) > MAX_INPUT_BYTES:
            raise HTTPException(status_code=422, detail="serialized sandbox payload exceeds input limit")

        with tempfile.TemporaryDirectory(prefix="dr-sidecar-") as workdir:
            try:
                proc = subprocess.run(
                    [sys.executable, str(_RUNNER_PATH)],
                    input=payload_bytes,
                    capture_output=True,
                    timeout=effective_timeout + 1.0,
                    check=False,
                    shell=False,
                    cwd=workdir,
                    env=_subproc_env(),
                )
            except subprocess.TimeoutExpired:
                return JSONResponse(
                    status_code=200,
                    headers=response_headers,
                    content={
                        "ok": False,
                        "backend": "sidecar",
                        "timed_out": True,
                        "error_code": "timeout",
                        "error": "sandbox sidecar execution timed out",
                        "stdout": "",
                        "stderr": "",
                        "result": None,
                        "tables_out": [],
                        "charts": [],
                        "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    },
                )

        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        try:
            runner_output = json.loads(proc.stdout.decode("utf-8"))
        except Exception:
            runner_output = {}
        if not isinstance(runner_output, dict) or proc.returncode != 0:
            err_msg = (
                runner_output.get("error")
                if isinstance(runner_output, dict) and runner_output.get("error")
                else (proc.stderr.decode("utf-8", errors="replace").strip() or "sandbox runner failed")
            )
            return JSONResponse(
                status_code=200,
                headers=response_headers,
                content={
                    "ok": False,
                    "backend": "sidecar",
                    "timed_out": False,
                    "error_code": (
                        runner_output.get("error_code")
                        if isinstance(runner_output, dict) and runner_output.get("error_code")
                        else "runtime_error"
                    ),
                    "error": str(err_msg)[:500],
                    "stdout": "",
                    "stderr": proc.stderr.decode("utf-8", errors="replace")[:2000],
                    "result": None,
                    "tables_out": [],
                    "charts": [],
                    "duration_ms": duration_ms,
                },
            )

        return JSONResponse(
            status_code=200,
            headers=response_headers,
            content={
                "ok": bool(runner_output.get("ok")),
                "backend": "sidecar",
                "timed_out": False,
                "stdout": str(runner_output.get("stdout") or ""),
                "stderr": str(runner_output.get("stderr") or ""),
                "result": runner_output.get("result"),
                "tables_out": runner_output.get("tables_out") or [],
                "charts": runner_output.get("charts") or [],
                "error": runner_output.get("error"),
                "error_code": runner_output.get("error_code"),
                "duration_ms": duration_ms,
            },
        )

    return app


app = create_sidecar_app()
