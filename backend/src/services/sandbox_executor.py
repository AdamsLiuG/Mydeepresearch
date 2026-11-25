"""Controlled code sandbox executor with AST allowlist, Docker/Pyodide isolation, and chart sanitizer."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from services.memory_sanitizer import sanitize_injection_text
from services.sandbox_runner import (
    ALLOWED_MODULES,
    BANNED_NAMES,
    MAX_AST_NODES,
    MAX_CODE_CHARS,
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    _validate_ast_safety,
)

try:
    from hello_agents.tools import Tool, ToolParameter
except ImportError:  # pragma: no cover - lightweight test fallback
    class ToolParameter:  # type: ignore[override]
        def __init__(
            self,
            name: str,
            type: str,
            description: str,
            required: bool = True,
            default: Any = None,
        ) -> None:
            self.name = name
            self.type = type
            self.description = description
            self.required = required
            self.default = default

    class Tool:  # type: ignore[override]
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

logger = logging.getLogger(__name__)

_SOURCE_ID_REGEX = re.compile(r"T\d+-(?:S|M|C)\d+")
_DANGEROUS_CHART_TEXT_RE = re.compile(
    r"(?:<\s*script|javascript:|vbscript:|data:text/html|\bon(?:error|load|click|mouseover)\s*=|function\s*\(|=>|\beval\s*\(|new\s+Function\s*\(|document\.|window\.)",
    flags=re.IGNORECASE,
)
_MAX_CHART_SPEC_BYTES = 64 * 1024
_MAX_CHART_DEPTH = 12
_MAX_CHART_POINTS = 2000


class SandboxInputTable(BaseModel):
    """Structured table bound into sandbox variables with explicit source provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=64, validation_alias=AliasChoices("name", "table_name"))
    source_id: str | None = None
    source_ids: list[str] = Field(default_factory=list, max_length=64)
    columns: list[str] = Field(default_factory=list, max_length=128)
    rows: list[list[Any] | dict[str, Any]] = Field(default_factory=list, max_length=2000)
    records: list[dict[str, Any]] = Field(default_factory=list, max_length=2000)
    description: str = ""

    @field_validator("name")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned or not cleaned.isidentifier() or cleaned.startswith("_"):
            raise ValueError(f"invalid sandbox table name: {value!r}")
        if cleaned in {"tables", "tables_out", "charts", "result", "emit_chart"}:
            raise ValueError(f"reserved sandbox table name: {cleaned!r}")
        return cleaned

    @field_validator("source_id")
    @classmethod
    def _validate_source_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = (value or "").strip().strip("[]")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError(f"sandbox table must bind a valid source_id [Tn-(S|M|C)m], got: {value!r}")
        return cleaned

    @model_validator(mode="after")
    def _validate_table(self) -> SandboxInputTable:
        self.source_ids = list(dict.fromkeys(
            self._validate_source_id(sid) for sid in ([self.source_id] if self.source_id else []) + self.source_ids
        ))
        if not self.source_ids:
            raise ValueError("sandbox input table requires at least one source ID")
        if self.records and self.rows:
            raise ValueError("sandbox input table must provide one row representation")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("sandbox input table columns must be unique")
        return self


class ReportChartArtifact(BaseModel):
    """Sanitized visualization specification produced by the sandbox for downstream reports."""

    model_config = ConfigDict(extra="forbid")

    chart_id: str
    title: str
    library: str = "vega-lite"
    spec: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    caption: str = ""
    anchor_id: str = ""


class SandboxExecutionRequest(BaseModel):
    """Execution request submitted to SandboxExecutorService."""

    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(gt=0)
    code: str = Field(validation_alias=AliasChoices("code", "script"))
    title: str = Field(default="", max_length=120)
    tables: list[SandboxInputTable] = Field(
        default_factory=list, max_length=8, validation_alias=AliasChoices("tables", "input_tables"),
    )
    derived_from_source_ids: list[str] = Field(
        default_factory=list, max_length=64,
        validation_alias=AliasChoices("derived_from_source_ids", "input_source_ids"),
    )
    timeout_seconds: float | None = Field(default=None, gt=0.5, le=30, allow_inf_nan=False)
    purpose: str = ""
    emit_chart: bool = False
    chart_library: str | None = None
    traceparent: str | None = None

    @field_validator("chart_library")
    @classmethod
    def _validate_library(cls, value: str | None) -> str | None:
        if value not in {None, "none", "vega-lite", "echarts"}:
            raise ValueError("unsupported sandbox chart library")
        return value


class SandboxExecutionResult(BaseModel):
    """Structured result emitted by a sandbox backend or AST validation gate."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    backend: str
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    tables_out: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[ReportChartArtifact] = Field(default_factory=list)
    derived_from_source_ids: list[str] = Field(default_factory=list)
    execution_hash: str = ""
    duration_ms: float = 0.0
    error: str | None = None
    error_code: str | None = None
    timed_out: bool = False


@dataclass
class ASTValidationOutcome:
    """Result of static AST allowlist verification."""

    ok: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


class SandboxASTValidator:
    """Static AST allowlist gate blocking imports, dunder escapes, and dangerous builtins."""

    def __init__(
        self,
        *,
        allowed_modules: frozenset[str] = ALLOWED_MODULES,
        banned_names: frozenset[str] = BANNED_NAMES,
        max_code_chars: int = MAX_CODE_CHARS,
        max_ast_nodes: int = MAX_AST_NODES,
    ) -> None:
        self.allowed_modules = allowed_modules
        self.banned_names = banned_names
        self.max_code_chars = max_code_chars
        self.max_ast_nodes = max_ast_nodes

    def validate(self, code: str) -> ASTValidationOutcome:
        errors = _validate_ast_safety(
            code, allowed_modules=self.allowed_modules, banned_names=self.banned_names,
            max_code_chars=self.max_code_chars, max_ast_nodes=self.max_ast_nodes,
        )
        return ASTValidationOutcome(ok=not errors, errors=errors)


def _sanitize_spec_value(value: Any, *, depth: int = 0, points: list[int]) -> Any:
    if depth > _MAX_CHART_DEPTH:
        raise ValueError("chart spec exceeds max nesting depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("chart spec contains a non-finite number")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _DANGEROUS_CHART_TEXT_RE.search(value) or re.search(
            r"[<>]|(?:https?|file|ftp|image|data):|^\s*//", value, re.IGNORECASE,
        ):
            raise ValueError("chart spec contains unsafe markup or a resource URL")
        return sanitize_injection_text(value)
    if isinstance(value, list):
        points[0] += len(value)
        if points[0] > _MAX_CHART_POINTS:
            raise ValueError("chart spec exceeds point limit")
        return [_sanitize_spec_value(item, depth=depth + 1, points=points) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, raw_val in value.items():
            if not isinstance(key, str):
                raise ValueError("chart spec keys must be strings")
            key_lower = key.lower()
            if key_lower in {"href", "usermeta"}:
                continue
            if key_lower == "$schema":
                continue
            if key_lower in {"url", "__proto__", "constructor", "prototype", "expr", "calculate"}:
                raise ValueError("chart spec contains an unsafe key")
            if key_lower == "type" and raw_val in ("image", "html"):
                raise ValueError("chart spec contains an unsafe renderer type")
            cleaned[key] = _sanitize_spec_value(raw_val, depth=depth + 1, points=points)
        return cleaned
    raise ValueError("chart spec contains a non-JSON value")


def sanitize_chart_spec(
    raw_chart: dict[str, Any] | ReportChartArtifact,
    *,
    valid_source_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Validate and sanitize a chart artifact; return None if unsafe or invalid."""

    payload = (
        raw_chart.model_dump()
        if isinstance(raw_chart, ReportChartArtifact)
        else dict(raw_chart or {})
    )
    library = str(payload.get("library") or "vega-lite").strip().lower()
    if library not in {"vega-lite", "echarts"}:
        return None
    spec = payload.get("spec")
    if not isinstance(spec, dict) or not spec:
        return None
    try:
        serialized = json.dumps(spec, ensure_ascii=False)
    except Exception:
        return None
    if len(serialized.encode("utf-8")) > _MAX_CHART_SPEC_BYTES:
        return None

    try:
        cleaned_spec = _sanitize_spec_value(spec, points=[0])
    except ValueError as exc:
        logger.warning("dropped unsafe chart spec: %s", exc)
        return None

    sources = payload.get("source_ids")
    if sources is not None and not isinstance(sources, list):
        return None
    raw_sources = list(dict.fromkeys(str(s).strip().strip("[]") for s in sources or [] if s))
    if valid_source_ids is not None:
        filtered_sources = [sid for sid in raw_sources if sid in valid_source_ids]
        if not filtered_sources or len(filtered_sources) != len(raw_sources):
            return None
    else:
        filtered_sources = [sid for sid in raw_sources if _SOURCE_ID_REGEX.fullmatch(sid)]

    title = sanitize_injection_text(payload.get("title") or "数据图表").replace("<", "").replace(">", "")[:120]
    caption = sanitize_injection_text(payload.get("caption") or "").replace("<", "").replace(">", "")[:500]
    chart_id = re.sub(r"[^a-zA-Z0-9_-]", "-", str(payload.get("chart_id") or "chart-1")[:100]).strip("-") or "chart-1"
    anchor_id = re.sub(r"[^a-zA-Z0-9_-]", "-", str(payload.get("anchor_id") or chart_id)[:100]).strip("-") or chart_id

    return ReportChartArtifact(
        chart_id=chart_id,
        title=title,
        library=library,
        spec=cleaned_spec,
        source_ids=filtered_sources,
        caption=caption,
        anchor_id=anchor_id,
    ).model_dump()


class SandboxBackend(Protocol):
    """Protocol implemented by isolated Python execution backends."""

    backend_name: str

    def is_available(self) -> bool:
        ...

    def execute(
        self,
        *,
        code: str,
        tables: list[dict[str, Any]],
        timeout_seconds: float,
        memory_mb: int,
    ) -> dict[str, Any]:
        ...


class DockerSandboxBackend:
    """Hardened Docker container execution backend with strict namespace/capability drops."""

    backend_name = "docker"

    def __init__(
        self,
        *,
        image: str = "helloagents-sandbox:latest",
        cpus: str = "1.0",
        runner_path: str | None = None,
        docker_bin: str | None = None,
    ) -> None:
        self.image = (image or "helloagents-sandbox:latest").strip()
        self.cpus = str(cpus or "1.0").strip()
        self.docker_bin = docker_bin or (shutil.which("docker") or "docker")

    def is_available(self) -> bool:
        if not shutil.which(self.docker_bin):
            return False
        try:
            probe = subprocess.run(
                [self.docker_bin, "image", "inspect", self.image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
                check=False,
                shell=False,
            )
            return probe.returncode == 0
        except Exception:
            return False

    def build_command(
        self,
        *,
        container_name: str,
        workdir: str = "",
        memory_mb: int,
    ) -> list[str]:
        return [
            self.docker_bin,
            "run",
            "--rm",
            "--interactive",
            "--pull=never",
            "--name",
            container_name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--pids-limit=64",
            f"--memory={max(64, int(memory_mb))}m",
            f"--memory-swap={max(64, int(memory_mb))}m",
            f"--cpus={self.cpus}",
            "--user=65534:65534",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--log-driver=none",
            self.image,
        ]

    def execute(
        self,
        *,
        code: str,
        tables: list[dict[str, Any]],
        timeout_seconds: float,
        memory_mb: int,
    ) -> dict[str, Any]:
        container_name = f"ha-sbx-{uuid.uuid4().hex[:12]}"
        payload = {
            "code": code,
            "tables": tables,
            "timeout_seconds": timeout_seconds,
            "memory_mb": memory_mb,
        }
        cmd = self.build_command(container_name=container_name, memory_mb=memory_mb)
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload, ensure_ascii=True, allow_nan=False),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    [self.docker_bin, "rm", "-f", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                logger.warning("Docker sandbox cleanup failed for %s", container_name)
            return _backend_failure("timeout", f"docker sandbox timed out after {timeout_seconds}s")
        return _parse_backend_output(proc.stdout, proc.returncode, "docker")


def _backend_failure(error_code: str, error: str) -> dict[str, Any]:
    return {"ok": False, "timed_out": error_code == "timeout", "error_code": error_code, "error": error}


def _parse_backend_output(stdout: str, returncode: int, backend: str) -> dict[str, Any]:
    if len(stdout) > MAX_OUTPUT_BYTES or len(stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
        return _backend_failure("output_limit", f"{backend} sandbox output exceeds limit")
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict) and type(parsed.get("ok")) is bool and (returncode == 0 or not parsed["ok"]):
            return parsed
    except (ValueError, RecursionError):
        pass
    if returncode in (-signal.SIGKILL, 137):
        return _backend_failure("oom", f"{backend} sandbox was killed")
    return _backend_failure("runtime_error", f"{backend} sandbox exited with code {returncode}")


class PyodideSandboxBackend:
    """Node.js + Pyodide WASM fallback backend with session process-group kill on timeout."""

    backend_name = "pyodide"

    def __init__(
        self,
        *,
        pyodide_cache_dir: str = "",
        worker_path: str | None = None,
        node_bin: str | None = None,
        max_old_space_mb: int = 512,
    ) -> None:
        self.pyodide_cache_dir = (pyodide_cache_dir or "").strip()
        self.worker_path = worker_path or str(Path(__file__).with_name("pyodide_worker.mjs"))
        self.node_bin = node_bin or (shutil.which("node") or "")
        self.max_old_space_mb = max(128, int(max_old_space_mb or 512))

    def is_available(self) -> bool:
        return bool(
            self.node_bin
            and shutil.which(self.node_bin)
            and os.path.isfile(self.worker_path)
            and self.pyodide_cache_dir
            and Path(self.pyodide_cache_dir, "pyodide.mjs").is_file()
        )

    def build_command(self, memory_mb: int | None = None) -> list[str]:
        worker = Path(self.worker_path).resolve()
        heap_mb = self.max_old_space_mb if memory_mb is None else min(self.max_old_space_mb, max(16, memory_mb // 4))
        return [
            self.node_bin or "node",
            f"--max-old-space-size={heap_mb}",
            f"--wasm-max-mem-pages={max(16, (memory_mb or 512) - heap_mb) * 16}",
            "--experimental-permission",
            f"--allow-fs-read={Path(self.pyodide_cache_dir).resolve()}",
            f"--allow-fs-read={worker}",
            f"--allow-fs-read={worker.with_name('sandbox_runner.py')}",
            str(worker),
        ]

    def execute(
        self,
        *,
        code: str,
        tables: list[dict[str, Any]],
        timeout_seconds: float,
        memory_mb: int,
    ) -> dict[str, Any]:
        cmd = self.build_command(memory_mb)
        payload = json.dumps(
            {
                "code": code,
                "tables": tables,
                "timeout_seconds": timeout_seconds,
                "memory_mb": memory_mb,
                "pyodide_cache_dir": str(Path(self.pyodide_cache_dir).resolve()),
            },
            ensure_ascii=True,
            allow_nan=False,
        )
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            shell=False,
            env={"PATH": os.defpath, "LANG": "C.UTF-8"},
        )
        try:
            stdout, _ = proc.communicate(input=payload, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=2.0)
            except Exception:
                pass
            return _backend_failure("timeout", f"pyodide sandbox timed out after {timeout_seconds}s")
        return _parse_backend_output(stdout, proc.returncode, "pyodide")


class SandboxExecutorService:
    budget_provider: Callable[[], Any] | None = None

    """Orchestrates AST verification, per-task execution caps, backend selection, and hash provenance."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        backend_preference: str = "auto",
        timeout_seconds: float = 10.0,
        memory_mb: int = 512,
        max_executions_per_task: int = 4,
        docker_backend: SandboxBackend | None = None,
        pyodide_backend: SandboxBackend | None = None,
        ast_validator: SandboxASTValidator | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.backend_preference = (backend_preference or "auto").strip().lower()
        self.timeout_seconds = max(1.0, float(timeout_seconds or 10.0))
        self.memory_mb = max(64, int(memory_mb or 512))
        self.max_executions_per_task = max(1, int(max_executions_per_task or 4))
        self.docker_backend = docker_backend or DockerSandboxBackend()
        self.pyodide_backend = pyodide_backend or PyodideSandboxBackend()
        self.ast_validator = ast_validator or SandboxASTValidator()
        self._metrics = metrics
        self._lock = threading.Lock()
        self._task_execution_counts: dict[int, int] = {}
        self._execution_slots = threading.BoundedSemaphore(2)

    def restore_task_counts(self, tasks: list[Any]) -> None:
        with self._lock:
            for task in tasks:
                count = sum(
                    item.get("source_origin") == "sandbox"
                    for item in task.evidence_items if isinstance(item, dict)
                )
                self._task_execution_counts[task.id] = max(self._task_execution_counts.get(task.id, 0), count)

    def _inc_metric(self, name: str, delta: int = 1) -> None:
        if self._metrics is not None:
            try:
                self._metrics.increment(name, delta)
            except Exception:
                pass

    def _select_backend(self) -> SandboxBackend | None:
        pref = self.backend_preference
        if pref not in {"auto", "docker", "pyodide"}:
            return None
        if pref == "docker":
            return self.docker_backend if self.docker_backend.is_available() else None
        if pref == "pyodide":
            return self.pyodide_backend if self.pyodide_backend.is_available() else None
        if self.docker_backend.is_available():
            return self.docker_backend
        if self.pyodide_backend.is_available():
            return self.pyodide_backend
        return None

    def _compute_execution_hash(
        self,
        *,
        code: str,
        tables: list[dict[str, Any]],
        derived_ids: list[str],
    ) -> str:
        canonical = json.dumps(
            {
                "code": code.replace("\r\n", "\n"),
                "tables": tables,
                "derived_from_source_ids": sorted(derived_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def execute(
        self,
        request: SandboxExecutionRequest,
        *,
        valid_source_ids: set[str] | None = None,
    ) -> SandboxExecutionResult:
        """Validate AST, enforce task cap, execute in isolated backend, and sanitize charts."""

        started = time.perf_counter()
        if not self.enabled:
            return SandboxExecutionResult(
                ok=False,
                backend="disabled",
                error="code sandbox is disabled",
                derived_from_source_ids=list(request.derived_from_source_ids),
            )

        derived_ids = list(
            dict.fromkeys(
                [sid for table in request.tables for sid in table.source_ids]
                + [str(sid).strip().strip("[]") for sid in request.derived_from_source_ids if sid]
            )
        )
        if not derived_ids or any(
            not _SOURCE_ID_REGEX.fullmatch(sid) or "-C" in sid
            or (valid_source_ids is not None and sid not in valid_source_ids)
            for sid in derived_ids
        ):
            return SandboxExecutionResult(
                ok=False, backend="provenance_rejected", error_code="provenance_rejected",
                error="computation requires known original evidence source IDs",
                derived_from_source_ids=derived_ids,
            )

        ast_check = self.ast_validator.validate(request.code)
        if not ast_check.ok:
            self._inc_metric("sandbox_ast_rejections_total")
            return SandboxExecutionResult(
                ok=False,
                backend="ast_rejected",
                error_code="ast_rejected",
                error="; ".join(ast_check.errors),
                derived_from_source_ids=derived_ids,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        backend = self._select_backend()
        if backend is None:
            return SandboxExecutionResult(
                ok=False,
                backend="unavailable",
                error_code="backend_unavailable",
                error=f"no isolated sandbox runtime available for backend={self.backend_preference!r}",
                derived_from_source_ids=derived_ids,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        table_payloads = [tbl.model_dump() for tbl in request.tables]
        try:
            if len({tbl.name for tbl in request.tables}) != len(request.tables):
                raise ValueError("duplicate sandbox table names")
            if len(json.dumps({"code": request.code, "tables": table_payloads}, allow_nan=False)) > MAX_INPUT_BYTES - 4096:
                raise ValueError("sandbox input exceeds byte limit")
        except (ValueError, TypeError, RecursionError) as exc:
            return SandboxExecutionResult(
                ok=False, backend="input_rejected", error_code="input_rejected", error=str(exc)[:300],
            )

        with self._lock:
            used = self._task_execution_counts.get(request.task_id, 0)
            if used >= self.max_executions_per_task:
                return SandboxExecutionResult(
                    ok=False,
                    backend="quota_exceeded",
                    error=f"task {request.task_id} exceeded max sandbox executions ({self.max_executions_per_task})",
                    derived_from_source_ids=derived_ids,
                    duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
                )
            self._task_execution_counts[request.task_id] = used + 1

        budget = self.budget_provider() if self.budget_provider else None
        timeout = min(self.timeout_seconds, float(request.timeout_seconds or self.timeout_seconds))
        if budget is not None:
            timeout = min(timeout, budget.governor.remaining_timeout(
                stage=budget.stage, work_class="required",
            ))
            if timeout <= 0:
                budget.refuse("deadline_exceeded")
        try:
            if self._execution_slots.acquire(timeout=timeout):
                try:
                    dispatch = budget.attempt({"sandbox_calls": 1}) if budget is not None else nullcontext()
                    with dispatch as attempt:
                        self._inc_metric("sandbox_executions_total")
                        raw_res = backend.execute(
                            code=request.code,
                            tables=table_payloads,
                            timeout_seconds=min(timeout, attempt.timeout) if attempt is not None else timeout,
                            memory_mb=self.memory_mb,
                        )
                finally:
                    self._execution_slots.release()
            else:
                raw_res = _backend_failure("timeout", "sandbox execution queue timed out")
        except (OSError, ValueError, subprocess.SubprocessError):
            raw_res = _backend_failure("backend_unavailable", "sandbox runtime could not start")
        timed_out = bool(raw_res.get("timed_out"))
        if timed_out:
            self._inc_metric("sandbox_timeouts_total")

        if not raw_res.get("ok"):
            return SandboxExecutionResult(
                ok=False, backend=backend.backend_name, timed_out=timed_out,
                error_code=raw_res.get("error_code") or "runtime_error",
                error=raw_res.get("error") or "sandbox execution failed",
                derived_from_source_ids=derived_ids,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        sanitized_charts: list[ReportChartArtifact] = []
        valid_chart_sources = set(derived_ids) if derived_ids else valid_source_ids
        for raw_chart in (raw_res.get("charts") or [])[:8]:
            if not isinstance(raw_chart, dict):
                continue
            enriched = dict(raw_chart)
            if not enriched.get("source_ids") and derived_ids:
                enriched["source_ids"] = list(derived_ids)
            cleaned_chart = sanitize_chart_spec(
                enriched,
                valid_source_ids=valid_chart_sources,
            )
            if cleaned_chart is not None:
                if request.chart_library in (None, cleaned_chart["library"]):
                    sanitized_charts.append(ReportChartArtifact.model_validate(cleaned_chart))

        result_val = raw_res.get("result")
        tables_out = [t for t in (raw_res.get("tables_out") or []) if isinstance(t, dict)]
        exec_hash = self._compute_execution_hash(
            code=request.code,
            tables=table_payloads,
            derived_ids=derived_ids,
        )
        backend_name = str(getattr(backend, "backend_name", "sandbox") or "sandbox")
        return SandboxExecutionResult(
            ok=bool(raw_res.get("ok")),
            backend=backend_name,
            stdout=str(raw_res.get("stdout") or ""),
            stderr=str(raw_res.get("stderr") or ""),
            result=result_val,
            tables_out=tables_out,
            charts=sanitized_charts,
            derived_from_source_ids=derived_ids,
            execution_hash=exec_hash,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            error=(str(raw_res.get("error")) if raw_res.get("error") else None),
            timed_out=timed_out,
        )


def record_sandbox_result(
    store: Any, request: SandboxExecutionRequest, result: SandboxExecutionResult, *, tenant_id: str = "default",
) -> dict[str, Any]:
    result_text = sanitize_injection_text(json.dumps(result.result, ensure_ascii=False))
    stdout = sanitize_injection_text(result.stdout)[:400]
    summary = f"计算结果: {result_text}" + (f"\n输出: {stdout}" if stdout else "")
    charts = [chart.model_dump() for chart in result.charts]
    return store.record_computation_evidence(
        task_id=request.task_id,
        title=request.title,
        summary_text=summary,
        derived_from_source_ids=result.derived_from_source_ids,
        computation_artifact={
            "execution_hash": result.execution_hash,
            "backend": result.backend,
            "result": result.result,
            "tables_out": result.tables_out,
            "code": request.code,
            "input_tables": [table.model_dump() for table in request.tables],
            "stdout": result.stdout,
            "duration_ms": result.duration_ms,
        },
        structured_payload={"result": result.result, "tables_out": result.tables_out, "charts": charts},
        tenant_id=tenant_id,
        backend=result.backend,
    )


class RunPythonSandboxTool(Tool):
    """Execute bounded quantitative Python code in an isolated sandbox and register [Tn-Cm] evidence."""

    def __init__(
        self,
        *,
        sandbox_service: SandboxExecutorService,
        evidence_store: Any,
        config: Any | None = None,
        tenant_id: str = "default",
    ) -> None:
        super().__init__(
            name="run_python_sandbox",
            description=(
                "在隔离代码沙箱中执行量化计算/表格校验并生成可引用的 [Tn-Cm] 计算证据。"
                "输入 JSON 需包含 task_id、title、code、derived_from_source_ids，以及可选 tables。"
            ),
        )
        self._sandbox_service = sandbox_service
        self._evidence_store = evidence_store
        self._config = config
        self._tenant_id = tenant_id

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task_id", type="integer", description="任务 ID"),
            ToolParameter(name="title", type="string", description="计算验证标题"),
            ToolParameter(name="code", type="string", description="受限 Python 计算脚本（需赋值 result 变量）"),
            ToolParameter(
                name="derived_from_source_ids",
                type="array",
                description="上游支撑证据 source_id 列表，如 ['T1-S1']",
            ),
            ToolParameter(
                name="tables",
                type="array",
                description="绑定到沙箱的结构化输入表（每张表含 name/source_id/columns/rows）",
                required=False,
            ),
            ToolParameter(
                name="emit_chart",
                type="boolean",
                description="是否输出可视化图表",
                required=False,
            ),
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        if self._config is not None and hasattr(self._config, "resolved_sandbox_effective_enabled"):
            if not self._config.resolved_sandbox_effective_enabled():
                return json.dumps(
                    {
                        "ok": False,
                        "error": "code sandbox is disabled for the active request/skill pack",
                    },
                    ensure_ascii=False,
                )

        try:
            raw = dict(parameters)
            raw["title"] = sanitize_injection_text(raw.get("title") or "沙箱定量计算")[:120]
            for key in ("derived_from_source_ids", "input_source_ids"):
                if isinstance(raw.get(key), str):
                    raw[key] = [value.strip() for value in raw[key].split(",") if value.strip()]
            for key in ("tables", "input_tables"):
                if isinstance(raw.get(key), str):
                    raw[key] = json.loads(raw[key])
            req = SandboxExecutionRequest.model_validate(raw)
        except (ValueError, TypeError, RecursionError):
            return json.dumps(_backend_failure("input_rejected", "invalid sandbox request"))

        valid_task_sources: set[str] | None = None
        if hasattr(self._evidence_store, "list_task_evidence"):
            items = self._evidence_store.list_task_evidence(req.task_id)
            valid_task_sources = {
                str(item.get("source_id") or "").strip()
                for item in items
                if item.get("source_id") and (
                    not getattr(self._evidence_store, "web_extraction_enabled", False)
                    or self._evidence_store.is_eligible_for_facts(item["source_id"])
                )
            }

        exec_result = self._sandbox_service.execute(req, valid_source_ids=valid_task_sources)

        payload: dict[str, Any] = {
            "ok": exec_result.ok,
            "backend": exec_result.backend,
            "result": exec_result.result,
            "tables_out": exec_result.tables_out,
            "charts": [c.model_dump() for c in exec_result.charts],
            "derived_from_source_ids": exec_result.derived_from_source_ids,
            "execution_hash": exec_result.execution_hash,
            "error": exec_result.error,
            "error_code": exec_result.error_code,
            "timed_out": exec_result.timed_out,
        }

        if exec_result.ok and hasattr(self._evidence_store, "record_computation_evidence"):
            record = record_sandbox_result(self._evidence_store, req, exec_result, tenant_id=self._tenant_id)
            payload["source_id"] = record.get("source_id")
            payload["evidence"] = record

        return json.dumps(payload, ensure_ascii=False)
