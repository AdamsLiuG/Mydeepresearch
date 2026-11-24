"""Controlled code sandbox executor with AST allowlist, Docker/Pyodide isolation, and chart sanitizer."""

from __future__ import annotations

import ast
import hashlib
import html
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.memory_sanitizer import sanitize_injection_text
from services.sandbox_runner import (
    ALLOWED_MODULES,
    BANNED_NAMES,
    MAX_AST_NODES,
    MAX_CODE_CHARS,
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
_MAX_CHART_SPEC_BYTES = 256 * 1024
_MAX_CHART_DEPTH = 12


class SandboxInputTable(BaseModel):
    """Structured table bound into sandbox variables with explicit source provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_id: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
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
    def _validate_source_id(cls, value: str) -> str:
        cleaned = (value or "").strip().strip("[]")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError(f"sandbox table must bind a valid source_id [Tn-(S|M|C)m], got: {value!r}")
        return cleaned


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

    task_id: int
    code: str
    title: str = ""
    tables: list[SandboxInputTable] = Field(default_factory=list)
    derived_from_source_ids: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    purpose: str = ""
    emit_chart: bool = False


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
        errors: list[str] = []
        if not isinstance(code, str) or not code.strip():
            return ASTValidationOutcome(ok=False, errors=["code must be a non-empty string"])
        if len(code) > self.max_code_chars:
            return ASTValidationOutcome(
                ok=False,
                errors=[f"code exceeds max length ({len(code)} > {self.max_code_chars})"],
            )

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return ASTValidationOutcome(
                ok=False,
                errors=[f"syntax error at line {exc.lineno}: {exc.msg}"],
            )

        node_count = 0
        for node in ast.walk(tree):
            node_count += 1
            if node_count > self.max_ast_nodes:
                errors.append(f"AST node budget exceeded ({node_count} > {self.max_ast_nodes})")
                break
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in self.allowed_modules:
                        errors.append(f"disallowed import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if node.level != 0 or root not in self.allowed_modules:
                    errors.append(f"disallowed import-from '{node.module}'")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") or node.attr in {"_subclasses__"}:
                    errors.append(f"disallowed dunder attribute access '.{node.attr}'")
            elif isinstance(node, ast.Name):
                if node.id in self.banned_names:
                    errors.append(f"disallowed name '{node.id}'")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.banned_names:
                    errors.append(f"disallowed builtin call '{node.func.id}()'")

        return ASTValidationOutcome(ok=not errors, errors=errors)


def _sanitize_spec_value(value: Any, *, depth: int = 0) -> Any:
    if depth > _MAX_CHART_DEPTH:
        raise ValueError("chart spec exceeds max nesting depth")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _DANGEROUS_CHART_TEXT_RE.search(value):
            raise ValueError(f"chart spec contains unsafe script/URI content: {value[:80]!r}")
        return html.escape(sanitize_injection_text(value), quote=False)
    if isinstance(value, list):
        return [_sanitize_spec_value(item, depth=depth + 1) for item in value[:500]]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for raw_key, raw_val in list(value.items())[:200]:
            key = str(raw_key)
            key_lower = key.lower()
            if key_lower in {"href", "usermeta"}:
                continue
            if key_lower == "url":
                raise ValueError("external url references are forbidden in chart specs")
            if key_lower == "data" and isinstance(raw_val, dict) and "url" in {
                str(k).lower() for k in raw_val
            }:
                raise ValueError("external data.url is forbidden in Vega-Lite specs; use inline values")
            if key_lower in {"formatter", "symbolsize"} and isinstance(raw_val, str):
                if _DANGEROUS_CHART_TEXT_RE.search(raw_val):
                    raise ValueError(f"unsafe JS callback in '{key}'")
            cleaned[key] = _sanitize_spec_value(raw_val, depth=depth + 1)
        return cleaned
    return html.escape(sanitize_injection_text(str(value)), quote=False)


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
        cleaned_spec = _sanitize_spec_value(spec)
    except ValueError as exc:
        logger.warning("dropped unsafe chart spec: %s", exc)
        return None

    raw_sources = [str(s).strip().strip("[]") for s in (payload.get("source_ids") or []) if s]
    if valid_source_ids is not None:
        filtered_sources = [sid for sid in raw_sources if sid in valid_source_ids]
        if not filtered_sources:
            return None
    else:
        filtered_sources = [sid for sid in raw_sources if _SOURCE_ID_REGEX.fullmatch(sid)]

    title = html.escape(sanitize_injection_text(payload.get("title") or "数据图表"), quote=False)
    caption = html.escape(sanitize_injection_text(payload.get("caption") or ""), quote=False)
    chart_id = re.sub(r"[^a-zA-Z0-9_-]", "-", str(payload.get("chart_id") or "chart-1")).strip("-") or "chart-1"
    anchor_id = re.sub(r"[^a-zA-Z0-9_-]", "-", str(payload.get("anchor_id") or chart_id)).strip("-") or chart_id

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
        self.runner_path = runner_path or str(Path(__file__).with_name("sandbox_runner.py"))
        self.docker_bin = docker_bin or (shutil.which("docker") or "docker")

    def is_available(self) -> bool:
        if not shutil.which(self.docker_bin) or not os.path.isfile(self.runner_path):
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
        workdir: str,
        memory_mb: int,
    ) -> list[str]:
        return [
            self.docker_bin,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--pids-limit=64",
            f"--memory={max(64, int(memory_mb))}m",
            f"--cpus={self.cpus}",
            "--user=65534:65534",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            "-v",
            f"{self.runner_path}:/opt/sandbox/sandbox_runner.py:ro",
            "-v",
            f"{workdir}:/workspace:ro",
            self.image,
            "python",
            "/opt/sandbox/sandbox_runner.py",
            "/workspace/input.json",
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
        with tempfile.TemporaryDirectory(prefix="ha_sandbox_") as workdir:
            input_path = Path(workdir) / "input.json"
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.chmod(input_path, 0o444)
            cmd = self.build_command(
                container_name=container_name,
                workdir=workdir,
                memory_mb=memory_mb,
            )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds + 2.0,
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                try:
                    subprocess.run(
                        [self.docker_bin, "rm", "-f", container_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0,
                        check=False,
                        shell=False,
                    )
                except Exception:
                    pass
                return {
                    "ok": False,
                    "timed_out": True,
                    "error": f"docker sandbox timed out after {timeout_seconds}s",
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "tables_out": [],
                    "charts": [],
                }

        raw_out = (proc.stdout or "").strip()
        if raw_out:
            try:
                parsed = json.loads(raw_out.splitlines()[-1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {
            "ok": False,
            "error": (proc.stderr or "").strip() or f"docker sandbox exited with code {proc.returncode}",
            "stdout": raw_out[:4000],
            "stderr": (proc.stderr or "")[:4000],
            "result": None,
            "tables_out": [],
            "charts": [],
        }


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
            and os.path.isdir(self.pyodide_cache_dir)
        )

    def build_command(self) -> list[str]:
        return [
            self.node_bin or "node",
            f"--max-old-space-size={self.max_old_space_mb}",
            self.worker_path,
        ]

    def execute(
        self,
        *,
        code: str,
        tables: list[dict[str, Any]],
        timeout_seconds: float,
        memory_mb: int,
    ) -> dict[str, Any]:
        cmd = self.build_command()
        payload = json.dumps(
            {
                "code": code,
                "tables": tables,
                "timeout_seconds": timeout_seconds,
                "memory_mb": memory_mb,
                "pyodide_cache_dir": self.pyodide_cache_dir,
            },
            ensure_ascii=False,
        )
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            shell=False,
        )
        try:
            stdout, stderr = proc.communicate(input=payload, timeout=timeout_seconds + 2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.communicate(timeout=2.0)
            except Exception:
                pass
            return {
                "ok": False,
                "timed_out": True,
                "error": f"pyodide sandbox timed out after {timeout_seconds}s",
                "stdout": "",
                "stderr": "",
                "result": None,
                "tables_out": [],
                "charts": [],
            }

        raw_out = (stdout or "").strip()
        if raw_out:
            try:
                parsed = json.loads(raw_out.splitlines()[-1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {
            "ok": False,
            "error": (stderr or "").strip() or f"pyodide worker exited with code {proc.returncode}",
            "stdout": raw_out[:4000],
            "stderr": (stderr or "")[:4000],
            "result": None,
            "tables_out": [],
            "charts": [],
        }


class SandboxExecutorService:
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

    def _inc_metric(self, name: str, delta: int = 1) -> None:
        if self._metrics is not None:
            try:
                self._metrics.increment(name, delta)
            except Exception:
                pass

    def _select_backend(self) -> SandboxBackend | None:
        pref = self.backend_preference
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
        result: Any,
        tables_out: list[dict[str, Any]],
    ) -> str:
        canonical = json.dumps(
            {
                "code": code,
                "tables": tables,
                "result": result,
                "tables_out": tables_out,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
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
                [t.source_id for t in request.tables if t.source_id]
                + [str(sid).strip().strip("[]") for sid in request.derived_from_source_ids if sid]
            )
        )
        if valid_source_ids is not None:
            ungrounded = [sid for sid in derived_ids if sid not in valid_source_ids]
            if ungrounded:
                return SandboxExecutionResult(
                    ok=False,
                    backend="provenance_rejected",
                    error=f"derived_from_source_ids contain unknown sources: {', '.join(ungrounded)}",
                    derived_from_source_ids=derived_ids,
                )

        ast_check = self.ast_validator.validate(request.code)
        if not ast_check.ok:
            self._inc_metric("sandbox_ast_rejections_total")
            return SandboxExecutionResult(
                ok=False,
                backend="ast_rejected",
                error="; ".join(ast_check.errors),
                derived_from_source_ids=derived_ids,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        backend = self._select_backend()
        if backend is None:
            return SandboxExecutionResult(
                ok=False,
                backend="unavailable",
                error=f"no isolated sandbox runtime available for backend={self.backend_preference!r}",
                derived_from_source_ids=derived_ids,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
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

        self._inc_metric("sandbox_executions_total")
        timeout = min(self.timeout_seconds, float(request.timeout_seconds or self.timeout_seconds))
        table_payloads = [tbl.model_dump() for tbl in request.tables]
        raw_res = backend.execute(
            code=request.code,
            tables=table_payloads,
            timeout_seconds=max(1.0, timeout),
            memory_mb=self.memory_mb,
        )
        timed_out = bool(raw_res.get("timed_out"))
        if timed_out:
            self._inc_metric("sandbox_timeouts_total")

        sanitized_charts: list[ReportChartArtifact] = []
        valid_chart_sources = set(derived_ids) if derived_ids else valid_source_ids
        for raw_chart in raw_res.get("charts") or []:
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
                sanitized_charts.append(ReportChartArtifact.model_validate(cleaned_chart))

        result_val = raw_res.get("result")
        tables_out = [t for t in (raw_res.get("tables_out") or []) if isinstance(t, dict)]
        exec_hash = self._compute_execution_hash(
            code=request.code,
            tables=table_payloads,
            result=result_val,
            tables_out=tables_out,
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


class RunPythonSandboxTool(Tool):
    """Execute bounded quantitative Python code in an isolated sandbox and register [Tn-Cm] evidence."""

    def __init__(
        self,
        *,
        sandbox_service: SandboxExecutorService,
        evidence_store: Any,
        config: Any | None = None,
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

        task_id = int(parameters.get("task_id") or 0)
        if task_id <= 0:
            raise ValueError("task_id is required")
        code = str(parameters.get("code") or "")
        title = sanitize_injection_text(parameters.get("title") or "沙箱定量计算")
        raw_sources = parameters.get("derived_from_source_ids") or []
        if isinstance(raw_sources, str):
            raw_sources = [s.strip() for s in raw_sources.split(",") if s.strip()]
        derived_ids = [str(s).strip().strip("[]") for s in raw_sources if str(s).strip()]

        raw_tables = parameters.get("tables") or []
        if isinstance(raw_tables, str):
            raw_tables = json.loads(raw_tables)
        tables = [
            SandboxInputTable.model_validate(tbl)
            for tbl in (raw_tables or [])
            if isinstance(tbl, dict)
        ]

        valid_task_sources: set[str] | None = None
        if hasattr(self._evidence_store, "list_task_evidence"):
            items = self._evidence_store.list_task_evidence(task_id)
            valid_task_sources = {
                str(item.get("source_id") or "").strip()
                for item in items
                if item.get("source_id")
            }

        req = SandboxExecutionRequest(
            task_id=task_id,
            code=code,
            title=title,
            tables=tables,
            derived_from_source_ids=derived_ids,
            emit_chart=bool(parameters.get("emit_chart", False)),
        )
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
            "timed_out": exec_result.timed_out,
        }

        if exec_result.ok and hasattr(self._evidence_store, "record_computation_evidence"):
            snippet_parts = [f"计算结论: {json.dumps(exec_result.result, ensure_ascii=False)}"]
            if exec_result.stdout:
                snippet_parts.append(f"输出: {exec_result.stdout.strip()[:400]}")
            if exec_result.derived_from_source_ids:
                snippet_parts.append(f"上游来源: {', '.join(exec_result.derived_from_source_ids)}")
            record = self._evidence_store.record_computation_evidence(
                task_id=task_id,
                title=title,
                summary_text="\n".join(snippet_parts),
                derived_from_source_ids=exec_result.derived_from_source_ids,
                computation_artifact={
                    "execution_hash": exec_result.execution_hash,
                    "backend": exec_result.backend,
                    "result": exec_result.result,
                    "tables_out": exec_result.tables_out,
                    "code": code,
                    "duration_ms": exec_result.duration_ms,
                },
                structured_payload={
                    "result": exec_result.result,
                    "tables_out": exec_result.tables_out,
                    "charts": [c.model_dump() for c in exec_result.charts],
                },
                backend=exec_result.backend,
            )
            payload["source_id"] = record.get("source_id")
            payload["evidence"] = record

        return json.dumps(payload, ensure_ascii=False)
