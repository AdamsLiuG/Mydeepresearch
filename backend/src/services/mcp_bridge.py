"""Production MCP client bridge, circuit breaker, SSRF guard, and ScopedToolRegistry."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlsplit

import httpx
import yaml

from metrics import RequestTrace, metrics_registry
from services.evidence import _is_public_ip

try:
    from hello_agents.tools import Tool, ToolParameter, ToolRegistry
except Exception:  # pragma: no cover - test harness stubs
    class ToolParameter:  # type: ignore[override]
        def __init__(
            self,
            *,
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

    class ToolRegistry:  # type: ignore[override]
        def __init__(self) -> None:
            self._tools: dict[str, Any] = {}
            self._functions: dict[str, dict[str, Any]] = {}

        def register_tool(self, tool: Any) -> None:
            self._tools[tool.name] = tool

        def get_tool(self, name: str) -> Any | None:
            return self._tools.get(name)

        def list_tools(self) -> list[str]:
            return list(self._tools.keys()) + list(self._functions.keys())

        def get_tools_description(self) -> str:
            return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values()) or "暂无可用工具"

        def execute_tool(self, name: str, input_text: str) -> str:
            if name in self._tools:
                return self._tools[name].run({"input": input_text})
            return f"错误：未找到名为 '{name}' 的工具。"

logger = logging.getLogger(__name__)


class MCPTransportType(str, Enum):
    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class MCPToolAnnotationsContract:
    """MCP ToolAnnotations contract: missing readOnlyHint strictly defaults to False (Fail-Closed)."""

    read_only_hint: bool = False
    destructive_hint: bool = True
    idempotent_hint: bool = False
    open_world_hint: bool = True

    @property
    def is_safe_readonly(self) -> bool:
        return self.read_only_hint is True

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> MCPToolAnnotationsContract:
        if not isinstance(raw, dict):
            return cls()
        readonly = raw.get("readOnlyHint")
        destructive = raw.get("destructiveHint")
        idempotent = raw.get("idempotentHint")
        open_world = raw.get("openWorldHint")
        return cls(
            read_only_hint=(readonly is True),
            destructive_hint=(True if destructive is None else bool(destructive)),
            idempotent_hint=(idempotent is True),
            open_world_hint=(True if open_world is None else bool(open_world)),
        )


@dataclass
class MCPServerConfig:
    """Configuration for a registered MCP server loaded from YAML or direct objects."""

    server_id: str
    transport: MCPTransportType = MCPTransportType.STDIO
    enabled: bool = True
    endpoint_url: str | None = None
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env_allowlist: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    trust_tier: Literal["curated_private", "internal_metrics", "external_mcp"] = "curated_private"
    tenant_scoped: bool = True
    allow_cross_request_archive: bool = False
    connect_timeout_seconds: float = 5.0
    tool_timeout_seconds: float = 10.0
    pool_size: int = 4
    max_retries: int = 1
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 60.0
    allow_write_hitl_tools: list[str] = field(default_factory=list)
    in_process_handler: Callable[[dict[str, Any], dict[str, str]], dict[str, Any] | None] | None = None


@dataclass(frozen=True)
class MCPToolSpec:
    """Cached metadata snapshot for a single MCP tool."""

    server_id: str
    tool_name: str
    qualified_name: str
    description: str
    input_schema: dict[str, Any]
    annotations: MCPToolAnnotationsContract

    @property
    def is_readonly(self) -> bool:
        return self.annotations.is_safe_readonly


@dataclass
class MCPCallResult:
    """Normalized MCP invocation result envelope."""

    ok: bool
    server_id: str
    tool_name: str
    latency_ms: float
    structured_payload: dict[str, Any] | None = None
    text_blocks: list[str] = field(default_factory=list)
    normalized_evidence: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    pending_approval: dict[str, Any] | None = None


class MCPHitlApprovalRequired(Exception):
    """Control-flow exception pausing execution until a human approves a write MCP call."""

    def __init__(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments_digest: str,
        arguments_preview: dict[str, Any] | None = None,
        task_id: int | None = None,
        approval_id: str | None = None,
        destructive_hint: bool = True,
    ) -> None:
        self.approval_id = approval_id or f"mcp_{uuid.uuid4().hex[:12]}"
        self.server_id = server_id
        self.tool_name = tool_name
        self.arguments_digest = arguments_digest
        self.arguments_preview = dict(arguments_preview or {})
        self.task_id = task_id
        self.destructive_hint = bool(destructive_hint)
        super().__init__(
            f"MCP HITL approval required for {server_id}:{tool_name} (approval_id={self.approval_id})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "server_id": self.server_id,
            "tool_name": self.tool_name,
            "arguments_digest": self.arguments_digest,
            "arguments_preview": dict(self.arguments_preview),
            "task_id": self.task_id,
            "destructive_hint": self.destructive_hint,
            "status": "awaiting_approval",
        }


def _host_or_ip_in_allowlist(target: str, allowlist: set[str]) -> bool:
    if not target or not allowlist:
        return False
    norm = target.strip().lower().rstrip(".")
    if norm in allowlist:
        return True
    try:
        ip_obj = ipaddress.ip_address(norm)
    except ValueError:
        return False
    for entry in allowlist:
        try:
            net = ipaddress.ip_network(entry, strict=False)
            if ip_obj in net:
                return True
        except ValueError:
            continue
    return False


def _validate_mcp_endpoint_url(
    url: str,
    *,
    allow_insecure_http: bool = False,
    allowed_private_hosts: set[str] | None = None,
) -> str:
    """Block non-HTTPS unless explicitly enabled and enforce DNS/IP SSRF protection."""

    cleaned = str(url or "").strip()
    parsed = urlsplit(cleaned)
    scheme = (parsed.scheme or "").lower()
    allowed_schemes = {"https", "http"} if allow_insecure_http else {"https"}
    if scheme not in allowed_schemes:
        raise ValueError(f"blocked MCP endpoint scheme: {scheme or 'empty'}")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError("MCP endpoint hostname is required")

    allowset = {item.strip().lower().rstrip(".") for item in (allowed_private_hosts or set()) if item.strip()}
    if _host_or_ip_in_allowlist(hostname, allowset):
        return cleaned

    try:
        ip_obj = ipaddress.ip_address(hostname)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        if not _is_public_ip(hostname):
            raise ValueError(f"blocked non-public MCP endpoint IP {hostname}")
        return cleaned

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"failed to resolve MCP endpoint hostname {hostname}: {exc}") from exc

    resolved_ips = {str(info[4][0]) for info in infos if info and info[4]}
    if not resolved_ips:
        raise ValueError(f"no IP addresses resolved for MCP endpoint: {hostname}")
    for ip_str in resolved_ips:
        if _host_or_ip_in_allowlist(ip_str, allowset):
            continue
        if not _is_public_ip(ip_str):
            raise ValueError(f"blocked non-public MCP endpoint IP {ip_str} for {hostname}")
    return cleaned


def mint_scoped_tenant_token(
    *,
    secret: str | None,
    tenant_id: str,
    server_id: str,
    request_id: str,
    scope: str = "mcp:read",
    ttl_seconds: int = 120,
    now: float | None = None,
) -> str:
    """Mint a short-lived HMAC-SHA256 scoped token propagating tenant isolation across processes."""

    if not secret:
        return ""
    ts = float(now if now is not None else time.time())
    payload = {
        "tenant_id": str(tenant_id or "default").strip() or "default",
        "server_id": str(server_id).strip(),
        "request_id": str(request_id).strip(),
        "scope": str(scope).strip() or "mcp:read",
        "iat": int(ts),
        "exp": int(ts + max(5, int(ttl_seconds or 120))),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    sig_bytes = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig_bytes).decode("ascii").rstrip("=")
    return f"v1.{payload_b64}.{sig_b64}"


def verify_scoped_tenant_token(
    *,
    token: str,
    secret: str,
    expected_server_id: str | None = None,
    expected_tenant_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Validate an HMAC-SHA256 scoped tenant token using constant-time comparison."""

    parts = str(token or "").strip().split(".")
    if len(parts) != 3 or parts[0] != "v1":
        raise PermissionError("invalid scoped tenant token format")
    payload_b64, sig_b64 = parts[1], parts[2]
    expected_sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    if not hmac.compare_digest(sig_b64, expected_sig):
        raise PermissionError("invalid scoped tenant token signature")
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception as exc:
        raise PermissionError("malformed scoped tenant token payload") from exc
    if not isinstance(claims, dict):
        raise PermissionError("invalid scoped tenant token payload")
    current_ts = float(now if now is not None else time.time())
    if float(claims.get("exp") or 0) < current_ts:
        raise PermissionError("scoped tenant token expired")
    if expected_server_id and str(claims.get("server_id") or "") != expected_server_id:
        raise PermissionError("scoped tenant token server mismatch")
    if expected_tenant_id and str(claims.get("tenant_id") or "") != expected_tenant_id:
        raise PermissionError("scoped tenant token tenant mismatch")
    return claims


def compute_arguments_digest(arguments: dict[str, Any] | None) -> str:
    canonical = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _matches_single_use_approval(
    approved_token: dict[str, Any] | None,
    *,
    server_id: str,
    tool_name: str,
    arguments_digest: str,
) -> bool:
    if not isinstance(approved_token, dict):
        return False
    if str(approved_token.get("status") or "") != "approved_once":
        return False
    if str(approved_token.get("server_id") or "") != server_id:
        return False
    if str(approved_token.get("tool_name") or "") != tool_name:
        return False
    expected = str(approved_token.get("arguments_digest") or "")
    return bool(expected) and hmac.compare_digest(expected, arguments_digest)


class _MCPEventLoopRunner:
    """Dedicated background event-loop thread bridging sync workers with async MCP transports."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = Event()
        self._closed = False
        self._thread = Thread(target=self._run_loop, name="mcp-async-reactor", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def submit(self, coro: Awaitable[Any], *, timeout: float) -> Any:
        if self._closed:
            raise RuntimeError("MCP reactor is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        try:
            return future.result(timeout=max(0.05, float(timeout or 10.0)))
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"MCP coroutine exceeded {timeout:.2f}s") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
        except Exception:
            pass


class MCPCircuitBreaker:
    """Thread-safe three-state circuit breaker: CLOSED -> OPEN -> HALF_OPEN (single probe)."""

    def __init__(self, *, failure_threshold: int = 5, recovery_seconds: float = 60.0) -> None:
        self._threshold = max(1, int(failure_threshold or 5))
        self._recovery_seconds = max(0.05, float(recovery_seconds or 60.0))
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_probe_inflight = False

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def allow_request(self) -> tuple[bool, CircuitState]:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True, self._state
            if self._state == CircuitState.OPEN:
                if (time.monotonic() - self._opened_at) >= self._recovery_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_probe_inflight = True
                    return True, CircuitState.HALF_OPEN
                return False, CircuitState.OPEN
            if not self._half_open_probe_inflight:
                self._half_open_probe_inflight = True
                return True, CircuitState.HALF_OPEN
            return False, CircuitState.HALF_OPEN

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._half_open_probe_inflight = False

    def record_failure(self) -> None:
        with self._lock:
            self._half_open_probe_inflight = False
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                return
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()


class _StdioSession:
    """Persistent newline-delimited JSON-RPC 2.0 session over a child subprocess."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._cfg = config
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._req_id = 0
        self._initialized = False

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        cmd = [*self._cfg.command, *self._cfg.args]
        if not cmd:
            raise RuntimeError(f"stdio MCP server '{self._cfg.server_id}' missing command")
        env: dict[str, str] = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        for var in self._cfg.env_allowlist:
            if var in os.environ:
                env[var] = os.environ[var]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cfg.cwd or None,
            env=env,
        )
        self._initialized = False

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            await self._ensure_started()
            if not self._initialized and method != "initialize":
                await self._send_rpc(
                    "initialize",
                    {
                        "protocolVersion": "2025-03-26",
                        "clientInfo": {"name": "deepresearch-mcp", "version": "1.0.0"},
                        "capabilities": {},
                    },
                )
                await self._notify("notifications/initialized", {})
                self._initialized = True
            resp = await self._send_rpc(method, params)
            if method == "initialize":
                await self._notify("notifications/initialized", {})
                self._initialized = True
            return resp

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        line = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}, ensure_ascii=False)
        self._proc.stdin.write((line + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _send_rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._req_id += 1
        rid = self._req_id
        payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()
        while True:
            raw_line = await self._proc.stdout.readline()
            if not raw_line:
                raise RuntimeError(f"MCP stdio server '{self._cfg.server_id}' closed stdout unexpectedly")
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict) and parsed.get("id") == rid:
                if "error" in parsed and parsed["error"] is not None:
                    err = parsed["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"MCP JSON-RPC error: {msg}")
                return parsed.get("result") or {}

    async def close(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._initialized = False


class MCPServerPool:
    """Connection pool per MCP server supporting in-process, stdio, and Streamable HTTP."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._cfg = config
        self._pool_size = max(1, int(config.pool_size or 4))
        self._sem: asyncio.Semaphore | None = None
        self._stdio_sessions: list[_StdioSession] = []
        self._stdio_rr = 0
        self._http_client: httpx.AsyncClient | None = None
        self._http_req_id = 0

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._pool_size)
        return self._sem

    async def send_jsonrpc(
        self,
        *,
        method: str,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        sem = self._get_sem()
        async with sem:
            if self._cfg.in_process_handler is not None:
                self._http_req_id += 1
                rpc_req = {"jsonrpc": "2.0", "id": self._http_req_id, "method": method, "params": params}
                reply = self._cfg.in_process_handler(rpc_req, dict(headers or {}))
                if not isinstance(reply, dict):
                    return {}
                if "error" in reply and reply["error"] is not None:
                    err = reply["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"MCP JSON-RPC error: {msg}")
                return reply.get("result") or {}

            if self._cfg.transport == MCPTransportType.STDIO:
                if len(self._stdio_sessions) < min(self._pool_size, 2):
                    self._stdio_sessions.append(_StdioSession(self._cfg))
                idx = self._stdio_rr % len(self._stdio_sessions)
                self._stdio_rr += 1
                return await self._stdio_sessions[idx].request(method, params)

            return await self._post_streamable_http(method=method, params=params, extra_headers=headers or {})

    async def _post_streamable_http(
        self,
        *,
        method: str,
        params: dict[str, Any],
        extra_headers: dict[str, str],
    ) -> dict[str, Any]:
        if not self._cfg.endpoint_url:
            raise ValueError(f"streamable_http MCP server '{self._cfg.server_id}' missing endpoint_url")
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=float(self._cfg.connect_timeout_seconds or 5.0),
                    read=float(self._cfg.tool_timeout_seconds or 10.0),
                    write=float(self._cfg.tool_timeout_seconds or 10.0),
                    pool=float(self._cfg.connect_timeout_seconds or 5.0),
                ),
                follow_redirects=False,
            )
        self._http_req_id += 1
        rid = self._http_req_id
        merged_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **self._cfg.headers,
            **extra_headers,
        }
        payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        resp = await self._http_client.post(self._cfg.endpoint_url, json=payload, headers=merged_headers)
        resp.raise_for_status()
        content_type = str(resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in content_type:
            last_obj: dict[str, Any] = {}
            for line in resp.text.splitlines():
                stripped = line.strip()
                if stripped.startswith("data:"):
                    raw_data = stripped[5:].strip()
                    if not raw_data or raw_data == "[DONE]":
                        continue
                    candidate = json.loads(raw_data)
                    if isinstance(candidate, dict):
                        last_obj = candidate
            parsed = last_obj
        else:
            parsed = resp.json()

        if isinstance(parsed, dict) and parsed.get("error") is not None:
            err = parsed["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"MCP HTTP error: {msg}")
        return parsed.get("result") or {} if isinstance(parsed, dict) else {}

    async def close(self) -> None:
        for session in self._stdio_sessions:
            await session.close()
        self._stdio_sessions.clear()
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


class MCPBridgeManager:
    """Central gateway managing MCP server pools, SSRF guards, Fail-Closed annotations, and circuit breakers."""

    def __init__(
        self,
        config: Any,
        *,
        servers: list[MCPServerConfig] | None = None,
    ) -> None:
        self._config = config
        self._lock = Lock()
        self._servers: dict[str, MCPServerConfig] = {}
        self._pools: dict[str, MCPServerPool] = {}
        self._breakers: dict[str, MCPCircuitBreaker] = {}
        self._tool_specs_cache: dict[str, tuple[float, list[MCPToolSpec]]] = {}
        self._reactor: _MCPEventLoopRunner | None = None

        loaded_servers = list(servers) if servers is not None else self._load_servers_from_config(config)
        for server_cfg in loaded_servers:
            self.register_server(server_cfg)

    @staticmethod
    def _load_servers_from_config(config: Any) -> list[MCPServerConfig]:
        if not bool(getattr(config, "mcp_enabled", False)):
            return []
        raw_path = str(getattr(config, "mcp_config_path", "") or "").strip()
        if not raw_path:
            return []
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            return []
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed loading mcp_servers.yaml path=%s error=%s", path, exc)
            return []
        entries = payload.get("servers") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return []

        result: list[MCPServerConfig] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("server_id") or "").strip()
            if not sid:
                continue
            transport_raw = str(entry.get("transport") or "stdio").strip().lower()
            transport = (
                MCPTransportType.STREAMABLE_HTTP
                if transport_raw == "streamable_http"
                else MCPTransportType.STDIO
            )
            cwd_raw = str(entry.get("cwd") or "").strip() or None
            if cwd_raw and not Path(cwd_raw).is_absolute():
                cwd_raw = str((path.parent.parent / cwd_raw).resolve(strict=False))
            result.append(
                MCPServerConfig(
                    server_id=sid,
                    transport=transport,
                    enabled=bool(entry.get("enabled", True)),
                    endpoint_url=str(entry.get("endpoint_url") or "").strip() or None,
                    command=[str(x) for x in (entry.get("command") or []) if str(x).strip()],
                    args=[str(x) for x in (entry.get("args") or []) if str(x).strip()],
                    cwd=cwd_raw,
                    env_allowlist=[str(x) for x in (entry.get("env_allowlist") or []) if str(x).strip()],
                    headers={str(k): str(v) for k, v in (entry.get("headers") or {}).items()},
                    trust_tier=str(entry.get("trust_tier") or "curated_private"),  # type: ignore[arg-type]
                    tenant_scoped=bool(entry.get("tenant_scoped", True)),
                    allow_cross_request_archive=bool(entry.get("allow_cross_request_archive", False)),
                    connect_timeout_seconds=float(entry.get("connect_timeout_seconds", 5.0) or 5.0),
                    tool_timeout_seconds=float(entry.get("tool_timeout_seconds", 10.0) or 10.0),
                    pool_size=int(entry.get("pool_size", 4) or 4),
                    max_retries=int(entry.get("max_retries", 1) or 1),
                    circuit_failure_threshold=int(entry.get("circuit_failure_threshold", 5) or 5),
                    circuit_recovery_seconds=float(entry.get("circuit_recovery_seconds", 60.0) or 60.0),
                    allow_write_hitl_tools=[
                        str(x) for x in (entry.get("allow_write_hitl_tools") or []) if str(x).strip()
                    ],
                )
            )
        return result

    def register_server(self, cfg: MCPServerConfig) -> None:
        with self._lock:
            self._servers[cfg.server_id] = cfg
            self._pools[cfg.server_id] = MCPServerPool(cfg)
            self._breakers[cfg.server_id] = MCPCircuitBreaker(
                failure_threshold=cfg.circuit_failure_threshold,
                recovery_seconds=cfg.circuit_recovery_seconds,
            )
            self._tool_specs_cache.pop(cfg.server_id, None)

    def _get_reactor(self) -> _MCPEventLoopRunner:
        with self._lock:
            if self._reactor is None:
                self._reactor = _MCPEventLoopRunner()
            return self._reactor

    def _validate_endpoint_if_http(self, server_cfg: MCPServerConfig) -> None:
        if server_cfg.transport != MCPTransportType.STREAMABLE_HTTP or server_cfg.in_process_handler is not None:
            return
        allowed_hosts = set(getattr(self._config, "mcp_allowed_private_hosts", []) or [])
        allow_insecure = bool(getattr(self._config, "mcp_allow_insecure_http", False))
        _validate_mcp_endpoint_url(
            server_cfg.endpoint_url or "",
            allow_insecure_http=allow_insecure,
            allowed_private_hosts=allowed_hosts,
        )

    def list_tools(
        self,
        server_id: str | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[MCPToolSpec]:
        ttl = max(0, int(getattr(self._config, "mcp_tool_cache_ttl_seconds", 300) or 0))
        target_ids = (
            [server_id]
            if server_id
            else [sid for sid, cfg in self._servers.items() if cfg.enabled]
        )
        specs: list[MCPToolSpec] = []
        now = time.monotonic()

        for sid in target_ids:
            cfg = self._servers.get(sid)
            if cfg is None or not cfg.enabled:
                continue
            cached = self._tool_specs_cache.get(sid)
            if not force_refresh and cached and (now - cached[0]) <= ttl:
                specs.extend(cached[1])
                continue

            self._validate_endpoint_if_http(cfg)
            result = self._get_reactor().submit(
                self._pools[sid].send_jsonrpc(method="tools/list", params={}),
                timeout=cfg.connect_timeout_seconds,
            )
            raw_tools = result.get("tools") if isinstance(result, dict) else []
            parsed_specs: list[MCPToolSpec] = []
            for item in raw_tools or ():
                if not isinstance(item, dict):
                    continue
                tname = str(item.get("name") or "").strip()
                if not tname:
                    continue
                parsed_specs.append(
                    MCPToolSpec(
                        server_id=sid,
                        tool_name=tname,
                        qualified_name=f"{sid}__{tname}",
                        description=str(item.get("description") or "").strip(),
                        input_schema=dict(item.get("inputSchema") or {}),
                        annotations=MCPToolAnnotationsContract.from_raw(item.get("annotations")),
                    )
                )
            self._tool_specs_cache[sid] = (time.monotonic(), parsed_specs)
            specs.extend(parsed_specs)

        return specs

    def get_tool_spec(self, server_id: str, tool_name: str) -> MCPToolSpec | None:
        cached = self._tool_specs_cache.get(server_id)
        if cached:
            for spec in cached[1]:
                if spec.tool_name == tool_name:
                    return spec
        try:
            refreshed = self.list_tools(server_id, force_refresh=True)
        except Exception:
            return None
        for spec in refreshed:
            if spec.tool_name == tool_name:
                return spec
        return None

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        tenant_id: str = "default",
        request_id: str = "local",
        task_id: int | None = None,
        approved_token: dict[str, Any] | None = None,
        approved_registry: dict[str, dict[str, Any]] | None = None,
        observer: RequestTrace | None = None,
    ) -> MCPCallResult:
        started = time.perf_counter()
        metrics_registry.increment("mcp_call_total")
        cfg = self._servers.get(server_id)
        if cfg is None or not cfg.enabled:
            return MCPCallResult(
                ok=False,
                server_id=server_id,
                tool_name=tool_name,
                latency_ms=0.0,
                error_code="server_disabled",
                error_message=f"MCP server '{server_id}' is disabled or unknown",
            )

        # 1. SSRF validation before network contact
        try:
            self._validate_endpoint_if_http(cfg)
        except ValueError as exc:
            return MCPCallResult(
                ok=False,
                server_id=server_id,
                tool_name=tool_name,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                error_code="ssrf_blocked",
                error_message=str(exc),
            )

        # 2. Fail-Closed readOnlyHint & HITL gate
        spec = self.get_tool_spec(server_id, tool_name)
        args_dict = dict(arguments or {})
        enforce_readonly = bool(getattr(self._config, "mcp_enforce_readonly", True))
        if spec is None or not spec.is_readonly:
            if enforce_readonly:
                allow_hitl = bool(getattr(self._config, "mcp_allow_hitl_write_tools", False))
                if not (allow_hitl and tool_name in cfg.allow_write_hitl_tools):
                    return MCPCallResult(
                        ok=False,
                        server_id=server_id,
                        tool_name=tool_name,
                        latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                        error_code="write_forbidden",
                        error_message=f"MCP tool {server_id}:{tool_name} lacks readOnlyHint=True",
                    )
                digest = compute_arguments_digest(args_dict)
                approval_key = f"{server_id}:{tool_name}:{digest}"
                matched_entry = approved_token
                if matched_entry is None and isinstance(approved_registry, dict):
                    matched_entry = approved_registry.get(approval_key)
                if not _matches_single_use_approval(
                    matched_entry,
                    server_id=server_id,
                    tool_name=tool_name,
                    arguments_digest=digest,
                ):
                    metrics_registry.increment("mcp_hitl_intercept_total")
                    raise MCPHitlApprovalRequired(
                        server_id=server_id,
                        tool_name=tool_name,
                        arguments_digest=digest,
                        arguments_preview=args_dict,
                        task_id=task_id,
                        destructive_hint=(spec.annotations.destructive_hint if spec else True),
                    )
                # Consume single-use approval token upon execution
                if isinstance(matched_entry, dict):
                    matched_entry["status"] = "consumed"

        # 3. Circuit breaker check
        breaker = self._breakers[server_id]
        allowed, _ = breaker.allow_request()
        if not allowed:
            metrics_registry.increment("mcp_circuit_open_total")
            if observer is not None:
                observer.record_degraded(f"mcp_circuit_open:{server_id}")
            return MCPCallResult(
                ok=False,
                server_id=server_id,
                tool_name=tool_name,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                error_code="circuit_open",
                error_message=f"Circuit breaker OPEN for MCP server '{server_id}'",
            )

        # 4. Scoped Tenant Token & timeout-bounded execution
        scope = "mcp:read" if (spec and spec.is_readonly) else "mcp:write_approved"
        scoped_token = mint_scoped_tenant_token(
            secret=getattr(self._config, "mcp_tenant_token_secret", None),
            tenant_id=tenant_id,
            server_id=server_id,
            request_id=request_id,
            scope=scope,
        )
        headers: dict[str, str] = {"X-Tenant-ID": tenant_id, "X-Request-ID": request_id}
        if scoped_token:
            headers["Authorization"] = f"Bearer {scoped_token}"

        meta: dict[str, Any] = {
            "tenant_id": tenant_id,
            "request_id": request_id,
        }
        if scoped_token:
            meta["scoped_token"] = scoped_token

        timeout_s = min(
            float(cfg.tool_timeout_seconds or 10.0),
            float(getattr(self._config, "mcp_tool_timeout_seconds", cfg.tool_timeout_seconds) or 10.0),
        )
        can_retry = bool(spec and spec.is_readonly and spec.annotations.idempotent_hint)
        attempts = 1 + (max(0, int(cfg.max_retries or 0)) if can_retry else 0)

        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                raw_result = self._get_reactor().submit(
                    self._pools[server_id].send_jsonrpc(
                        method="tools/call",
                        params={"name": tool_name, "arguments": args_dict, "_meta": meta},
                        headers=headers,
                    ),
                    timeout=timeout_s,
                )
                breaker.record_success()
                metrics_registry.increment("mcp_success_total")
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
                metrics_registry.add_latency("mcp_call_ms", elapsed_ms)
                return self._normalize_call_result(cfg, tool_name, raw_result, latency_ms=elapsed_ms)
            except TimeoutError as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    breaker.record_failure()
                    metrics_registry.increment("mcp_timeout_total")
                    if observer is not None:
                        observer.record_degraded(f"mcp_timeout:{server_id}:{tool_name}")
                    return MCPCallResult(
                        ok=False,
                        server_id=server_id,
                        tool_name=tool_name,
                        latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                        error_code="timeout",
                        error_message=str(exc),
                    )
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    breaker.record_failure()
                    if observer is not None:
                        observer.record_degraded(f"mcp_call_failed:{server_id}:{tool_name}")
                    return MCPCallResult(
                        ok=False,
                        server_id=server_id,
                        tool_name=tool_name,
                        latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                        error_code="transport_error",
                        error_message=str(last_exc),
                    )

        return MCPCallResult(
            ok=False,
            server_id=server_id,
            tool_name=tool_name,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            error_code="transport_error",
            error_message=str(last_exc or "unknown error"),
        )

    @staticmethod
    def _source_type_for_trust_tier(trust_tier: str) -> str:
        return {
            "curated_private": "mcp_curated_filing",
            "internal_metrics": "mcp_internal_metrics",
            "external_mcp": "mcp_external",
        }.get(str(trust_tier or "").strip(), "mcp_curated_filing")

    @classmethod
    def _normalize_call_result(
        cls,
        server_cfg: MCPServerConfig,
        tool_name: str,
        raw_result: dict[str, Any],
        *,
        latency_ms: float,
    ) -> MCPCallResult:
        content_blocks = raw_result.get("content") if isinstance(raw_result, dict) else []
        text_blocks: list[str] = []
        for blk in content_blocks or ():
            if isinstance(blk, dict) and str(blk.get("text") or "").strip():
                text_blocks.append(str(blk["text"]).strip())

        structured = raw_result.get("structuredContent") if isinstance(raw_result, dict) else None
        if structured is None and text_blocks:
            try:
                parsed = json.loads(text_blocks[0])
                if isinstance(parsed, dict):
                    structured = parsed
            except Exception:
                structured = None

        source_type = cls._source_type_for_trust_tier(server_cfg.trust_tier)
        normalized_evidence: list[dict[str, Any]] = []
        if isinstance(structured, dict):
            company = str(structured.get("company_name") or "").strip()
            sha1 = str(structured.get("pdf_sha1") or "").strip()
            passages = structured.get("passages")
            if isinstance(passages, list):
                for p in passages:
                    if not isinstance(p, dict):
                        continue
                    pno = p.get("page_index_1based")
                    text = str(p.get("text") or "").strip()
                    if not text:
                        continue
                    uri = str(p.get("citation_uri") or f"{server_cfg.server_id}://{sha1}#page={pno}")
                    normalized_evidence.append(
                        {
                            "title": f"{company or server_cfg.server_id} p.{pno}" if pno else (company or tool_name),
                            "url": uri,
                            "snippet": text[:600],
                            "full_content": text,
                            "source_type": source_type,
                            "page_no": pno if isinstance(pno, int) else None,
                            "provenance_meta": {
                                "pdf_sha1": sha1,
                                "page_index_1based": pno,
                                "company_name": company,
                                "score": p.get("combined_score") or p.get("relevance_score"),
                            },
                        }
                    )
            pages = structured.get("pages") or structured.get("excerpts")
            if not normalized_evidence and isinstance(pages, list):
                for pg in pages:
                    if not isinstance(pg, dict):
                        continue
                    pno = pg.get("page_index")
                    text = str(pg.get("text") or "").strip()
                    if not text:
                        continue
                    uri = f"{server_cfg.server_id}://{sha1}#page={pno}" if pno else f"{server_cfg.server_id}://{tool_name}"
                    normalized_evidence.append(
                        {
                            "title": f"{company or server_cfg.server_id} p.{pno}" if pno else (company or tool_name),
                            "url": uri,
                            "snippet": text[:600],
                            "full_content": text,
                            "source_type": source_type,
                            "page_no": pno if isinstance(pno, int) else None,
                            "provenance_meta": {"pdf_sha1": sha1, "page_index_1based": pno, "company_name": company},
                        }
                    )
            if not normalized_evidence:
                summary_text = json.dumps(structured, ensure_ascii=False)
                normalized_evidence.append(
                    {
                        "title": f"{server_cfg.server_id}:{tool_name}",
                        "url": f"mcp://{server_cfg.server_id}/{tool_name}",
                        "snippet": summary_text[:600],
                        "full_content": summary_text,
                        "source_type": source_type,
                        "provenance_meta": dict(structured),
                    }
                )
        elif text_blocks:
            joined = "\n".join(text_blocks)
            normalized_evidence.append(
                {
                    "title": f"{server_cfg.server_id}:{tool_name}",
                    "url": f"mcp://{server_cfg.server_id}/{tool_name}",
                    "snippet": joined[:600],
                    "full_content": joined,
                    "source_type": source_type,
                    "provenance_meta": {},
                }
            )

        return MCPCallResult(
            ok=not bool(raw_result.get("isError")) if isinstance(raw_result, dict) else True,
            server_id=server_cfg.server_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            structured_payload=structured if isinstance(structured, dict) else None,
            text_blocks=text_blocks,
            normalized_evidence=normalized_evidence,
        )

    def build_tool_adapters(
        self,
        *,
        evidence_store: Any,
        tenant_id_getter: Callable[[], str],
        request_id_getter: Callable[[], str],
        approved_registry_getter: Callable[[], dict[str, dict[str, Any]] | None] | None = None,
        observer_getter: Callable[[], RequestTrace | None] | None = None,
    ) -> list[MCPToolAdapter]:
        adapters: list[MCPToolAdapter] = []
        for spec in self.list_tools():
            adapters.append(
                MCPToolAdapter(
                    spec=spec,
                    bridge=self,
                    evidence_store=evidence_store,
                    tenant_id_getter=tenant_id_getter,
                    request_id_getter=request_id_getter,
                    approved_registry_getter=approved_registry_getter,
                    observer_getter=observer_getter,
                )
            )
        return adapters

    def close(self) -> None:
        if self._reactor is not None:
            for pool in list(self._pools.values()):
                try:
                    self._reactor.submit(pool.close(), timeout=2.0)
                except Exception:
                    pass
            self._reactor.close()
            self._reactor = None


class MCPToolAdapter(Tool):
    """HelloAgents Tool adapter wrapping an MCPToolSpec cleanly into EvidenceStore."""

    def __init__(
        self,
        *,
        spec: MCPToolSpec,
        bridge: MCPBridgeManager,
        evidence_store: Any = None,
        tenant_id_getter: Callable[[], str] | None = None,
        request_id_getter: Callable[[], str] | None = None,
        approved_registry_getter: Callable[[], dict[str, dict[str, Any]] | None] | None = None,
        observer_getter: Callable[[], RequestTrace | None] | None = None,
    ) -> None:
        super().__init__(
            name=spec.qualified_name,
            description=spec.description or f"MCP tool {spec.server_id}:{spec.tool_name}",
        )
        self.spec = spec
        self._bridge = bridge
        self._evidence_store = evidence_store
        self._tenant_id_getter = tenant_id_getter
        self._request_id_getter = request_id_getter
        self._approved_registry_getter = approved_registry_getter
        self._observer_getter = observer_getter

    def get_parameters(self) -> list[ToolParameter]:
        props = (self.spec.input_schema or {}).get("properties") or {}
        required_set = set((self.spec.input_schema or {}).get("required") or [])
        params = [
            ToolParameter(name="task_id", type="integer", description="所属任务 ID", required=False)
        ]
        for key, schema in props.items():
            if not isinstance(schema, dict):
                continue
            params.append(
                ToolParameter(
                    name=str(key),
                    type=str(schema.get("type") or "string"),
                    description=str(schema.get("description") or key),
                    required=(key in required_set),
                )
            )
        return params

    def run(self, parameters: dict[str, Any]) -> str:
        raw_args = dict(parameters or {})
        task_id_raw = raw_args.pop("task_id", None)
        task_id = int(task_id_raw) if task_id_raw not in (None, "") else 0
        raw_args.pop("approved_token", None)
        tenant_id = self._tenant_id_getter() if self._tenant_id_getter else "default"
        request_id = self._request_id_getter() if self._request_id_getter else "local"
        approved_registry = self._approved_registry_getter() if self._approved_registry_getter else None
        observer = self._observer_getter() if self._observer_getter else None

        result = self._bridge.call_tool(
            server_id=self.spec.server_id,
            tool_name=self.spec.tool_name,
            arguments=raw_args,
            tenant_id=tenant_id,
            request_id=request_id,
            task_id=(task_id or None),
            approved_registry=approved_registry,
            observer=observer,
        )
        if not result.ok:
            return json.dumps(
                {
                    "ok": False,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                },
                ensure_ascii=False,
            )

        recorded: list[dict[str, Any]] = []
        if self._evidence_store is not None and task_id > 0 and result.normalized_evidence:
            recorded = self._evidence_store.record_mcp_results(
                task_id=task_id,
                query=str(raw_args.get("query") or raw_args.get("question") or self.spec.tool_name),
                server_id=self.spec.server_id,
                tool_name=self.spec.tool_name,
                items=result.normalized_evidence,
                tenant_id=tenant_id,
            )
        return json.dumps(
            {
                "ok": True,
                "server_id": result.server_id,
                "tool_name": result.tool_name,
                "recorded_source_ids": [r.get("source_id") for r in recorded if r.get("source_id")],
                "structured_payload": result.structured_payload,
            },
            ensure_ascii=False,
        )


class ScopedToolRegistry(ToolRegistry):
    """Skill- & Stage-scoped ToolRegistry preventing cross-domain Tool Bloat and swallowing of HITL."""

    def __init__(
        self,
        *,
        core_tools: list[Tool] | None = None,
        mcp_adapters: list[MCPToolAdapter] | None = None,
        active_scopes: dict[str, Any] | None = None,
        stage: str = "task_execution",
        max_mcp_tools: int = 6,
        enforce_readonly: bool = True,
    ) -> None:
        super().__init__()
        self._mcp_tool_names: list[str] = []
        for tool in core_tools or ():
            self._tools[tool.name] = tool

        scopes = active_scopes if isinstance(active_scopes, dict) else {}
        if not scopes:
            return

        stage_allowlist = {
            str(s).strip()
            for s in (scopes.get("stage_allowlist") or ["task_execution", "review"])
            if str(s).strip()
        }
        if stage not in stage_allowlist:
            return

        allowed_servers = {str(s).strip() for s in (scopes.get("allowed_servers") or []) if str(s).strip()}
        allowed_tools = {str(t).strip() for t in (scopes.get("allowed_tools") or []) if str(t).strip()}
        readonly_only = bool(scopes.get("readonly_only", True)) or bool(enforce_readonly)
        requested_cap = int(scopes.get("max_tools_injected") or max_mcp_tools)
        cap = min(max(0, requested_cap), max(0, int(max_mcp_tools or 6)))

        injected = 0
        for adapter in mcp_adapters or ():
            if injected >= cap:
                break
            if readonly_only and not adapter.spec.is_readonly:
                continue
            if allowed_servers and adapter.spec.server_id not in allowed_servers:
                continue
            canonical_ref = f"{adapter.spec.server_id}:{adapter.spec.tool_name}"
            if allowed_tools and canonical_ref not in allowed_tools and adapter.spec.tool_name not in allowed_tools:
                continue
            self._tools[adapter.name] = adapter
            self._mcp_tool_names.append(adapter.name)
            injected += 1

    @property
    def injected_mcp_tools(self) -> list[str]:
        return list(self._mcp_tool_names)

    def execute_tool(self, name: str, input_text: str) -> str:
        """Preserve MCPHitlApprovalRequired propagation instead of flattening to string."""
        if name in self._tools:
            try:
                return self._tools[name].run({"input": input_text})
            except MCPHitlApprovalRequired:
                raise
            except Exception as exc:
                return f"错误：执行工具 '{name}' 时发生异常: {exc}"
        return super().execute_tool(name, input_text)
