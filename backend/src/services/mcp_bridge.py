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
import math
import os
import signal
import socket
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlsplit

import httpx
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import TypeAdapter, ValidationError
from referencing import Registry

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
_MAX_MCP_MESSAGE_BYTES = 2 * 1024 * 1024


class _MCPEndpointError(ValueError):
    pass


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
        tenant_id: str = "default",
        request_id: str = "local",
        expires_at: float | None = None,
    ) -> None:
        self.approval_id = approval_id or f"mcp_{uuid.uuid4().hex}"
        self.server_id = server_id
        self.tool_name = tool_name
        self.arguments_digest = arguments_digest
        self.arguments_preview = dict(arguments_preview or {})
        self.task_id = task_id
        self.destructive_hint = bool(destructive_hint)
        self.tenant_id = str(tenant_id or "default")
        self.request_id = str(request_id or "local")
        self.created_at = time.time()
        self.expires_at = float(expires_at if expires_at is not None else self.created_at + 86400)
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
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
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
    cleaned = str(url or "").strip()
    _resolve_mcp_endpoint(
        cleaned, allow_insecure_http=allow_insecure_http, allowed_private_hosts=allowed_private_hosts,
        resolve_allowlisted_host=False,
    )
    return cleaned


def _resolve_mcp_endpoint(
    url: str, *, allow_insecure_http: bool, allowed_private_hosts: set[str] | None,
    resolve_allowlisted_host: bool = True,
) -> list[str]:
    cleaned = str(url or "").strip()
    parsed = urlsplit(cleaned)
    scheme = (parsed.scheme or "").lower()
    allowed_schemes = {"https", "http"} if allow_insecure_http else {"https"}
    if scheme not in allowed_schemes:
        raise ValueError(f"blocked MCP endpoint scheme: {scheme or 'empty'}")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError("MCP endpoint hostname is required")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError("MCP endpoint cannot contain credentials or a fragment")
    port = parsed.port or (443 if scheme == "https" else 80)

    allowset = {item.strip().lower().rstrip(".") for item in (allowed_private_hosts or set()) if item.strip()}
    host_allowed = _host_or_ip_in_allowlist(hostname, allowset)
    if host_allowed and not resolve_allowlisted_host:
        return [hostname]

    try:
        ip_obj = ipaddress.ip_address(hostname)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        if not host_allowed and not (ip_obj.is_global and _is_public_ip(hostname)):
            raise ValueError(f"blocked non-public MCP endpoint IP {hostname}")
        return [hostname]

    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"failed to resolve MCP endpoint hostname {hostname}: {exc}") from exc

    resolved_ips = list(dict.fromkeys(str(info[4][0]) for info in infos if info and info[4]))
    if not resolved_ips:
        raise ValueError(f"no IP addresses resolved for MCP endpoint: {hostname}")
    for ip_str in resolved_ips:
        if host_allowed or _host_or_ip_in_allowlist(ip_str, allowset):
            continue
        if not (_is_public_ip(ip_str) and ipaddress.ip_address(ip_str).is_global):
            raise ValueError(f"blocked non-public MCP endpoint IP {ip_str} for {hostname}")
    return resolved_ips


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
    request_id: str,
    tenant_id: str,
    task_id: int | None,
) -> bool:
    if not isinstance(approved_token, dict):
        return False
    if str(approved_token.get("status") or "") != "approved_once":
        return False
    if str(approved_token.get("server_id") or "") != server_id:
        return False
    if str(approved_token.get("tool_name") or "") != tool_name:
        return False
    if str(approved_token.get("request_id") or "") != request_id:
        return False
    if str(approved_token.get("tenant_id") or "") != tenant_id:
        return False
    if approved_token.get("task_id") != task_id or tool_approval_expired(approved_token):
        return False
    expected = str(approved_token.get("arguments_digest") or "")
    return bool(expected) and hmac.compare_digest(expected, arguments_digest)


def tool_approval_expired(approval: dict[str, Any]) -> bool:
    expiry = approval.get("expires_at")
    return type(expiry) not in {int, float} or not math.isfinite(expiry) or expiry <= time.time()


def _tool_input_validator(schema: Any) -> Draft202012Validator:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("MCP inputSchema must describe an object")
    Draft202012Validator.check_schema(schema)
    # A tool schema may resolve local references only.
    return Draft202012Validator(schema, registry=Registry())


def _scope_allows_tool(
    scopes: Any,
    *,
    server_id: str,
    tool_name: str,
    stage: str,
    readonly: bool,
) -> bool:
    if not isinstance(scopes, dict) or not scopes:
        return False
    servers = scopes.get("allowed_servers")
    tools = scopes.get("allowed_tools")
    stages = scopes.get("stage_allowlist", ["task_execution", "review"])
    if not isinstance(servers, list) or server_id not in servers:
        return False
    if not isinstance(tools, list) or f"{server_id}:{tool_name}" not in tools:
        return False
    if not isinstance(stages, list) or stage not in stages:
        return False
    if scopes.get("max_tools_injected") == 0:
        return False
    if scopes.get("readonly_only", True) is not False and not readonly:
        return False
    return True


class _MCPEventLoopRunner:
    """Dedicated background event-loop thread bridging sync workers with async MCP transports."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = Event()
        self._closed = False
        self._thread = Thread(target=self._run_loop, name="mcp-async-reactor", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self.close()
            raise RuntimeError("MCP reactor failed to start")

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
            shutdown = asyncio.run_coroutine_threadsafe(self._cancel_pending_tasks(), self._loop)
            shutdown.result(timeout=3.0)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=3.0)
            if not self._thread.is_alive():
                self._loop.close()
        except Exception:
            pass

    async def _cancel_pending_tasks(self) -> None:
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks(self._loop) if task is not current and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


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
        self._generation = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def acquire(self) -> tuple[bool, CircuitState, int]:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True, self._state, self._generation
            if self._state == CircuitState.OPEN:
                if (time.monotonic() - self._opened_at) >= self._recovery_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_probe_inflight = True
                    return True, CircuitState.HALF_OPEN, self._generation
                return False, CircuitState.OPEN, self._generation
            if not self._half_open_probe_inflight:
                self._half_open_probe_inflight = True
                return True, CircuitState.HALF_OPEN, self._generation
            return False, CircuitState.HALF_OPEN, self._generation

    def allow_request(self) -> tuple[bool, CircuitState]:
        allowed, state, _ = self.acquire()
        return allowed, state

    def record_success(self, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._half_open_probe_inflight = False

    def record_failure(self, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            self._half_open_probe_inflight = False
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._generation += 1
                return
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._generation += 1

    def release_probe(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation and self._state == CircuitState.HALF_OPEN:
                self._half_open_probe_inflight = False


_INITIALIZE_PARAMS = {
    "protocolVersion": "2025-03-26",
    "clientInfo": {"name": "deepresearch-mcp", "version": "1.0.0"},
    "capabilities": {},
}
_SUPPORTED_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}


class _MCPRemoteError(RuntimeError):
    """The transport worked and the server returned an application error."""


def _rpc_result(reply: Any, request_id: int) -> dict[str, Any]:
    if (
        not isinstance(reply, dict)
        or reply.get("jsonrpc") != "2.0"
        or type(reply.get("id")) is not int
        or reply["id"] != request_id
        or ("result" in reply) == ("error" in reply)
    ):
        raise ValueError("Invalid MCP JSON-RPC response")
    if "error" in reply:
        error = reply["error"]
        if not isinstance(error, dict) or type(error.get("code")) is not int or not isinstance(error.get("message"), str):
            raise ValueError("Invalid MCP JSON-RPC error")
        raise _MCPRemoteError(error["message"])
    result = reply["result"]
    if not isinstance(result, dict):
        raise ValueError("MCP JSON-RPC result must be an object")
    return result


def _negotiated_protocol(result: dict[str, Any]) -> str:
    version = result.get("protocolVersion")
    if version not in _SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError("Unsupported MCP protocol version")
    return version


class _StdioSession:
    """Persistent newline-delimited JSON-RPC session with one in-flight request."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._cfg = config
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._req_id = 0

    async def _start(self) -> None:
        cmd = [*self._cfg.command, *self._cfg.args]
        if not cmd:
            raise ValueError("MCP stdio command required")
        if cmd[0] == "{python}":
            cmd[0] = sys.executable
        allowed = {"PATH", "LANG", *self._cfg.env_allowlist}
        env = {key: value for key, value in os.environ.items() if key in allowed}
        env.setdefault("PATH", os.defpath)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self._cfg.cwd,
            env=env,
            limit=_MAX_MCP_MESSAGE_BYTES,
            start_new_session=(os.name == "posix"),
        )
        _negotiated_protocol(await self._send_rpc("initialize", _INITIALIZE_PARAMS))
        await self._write({"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            try:
                if self._proc is None or self._proc.returncode is not None:
                    try:
                        await asyncio.wait_for(self._start(), self._cfg.connect_timeout_seconds)
                    except _MCPRemoteError as exc:
                        raise RuntimeError("MCP initialization failed") from exc
                return await self._send_rpc(method, params)
            except _MCPRemoteError:
                raise
            except (Exception, asyncio.CancelledError):
                await self._stop()
                raise

    async def _write(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _send_rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        self._req_id += 1
        rid = self._req_id
        await self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP stdio server closed stdout")
            reply = json.loads(line)
            if isinstance(reply, dict) and "method" in reply:
                if "id" in reply:
                    response = {"jsonrpc": "2.0", "id": reply["id"]}
                    if reply["method"] == "ping":
                        response["result"] = {}
                    else:
                        response["error"] = {"code": -32601, "message": "Method not found"}
                    await self._write(response)
                continue
            return _rpc_result(reply, rid)

    async def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin is not None:
            proc.stdin.close()
        # Drain stdout while waiting so a full pipe cannot prevent process reaping.
        assert proc.stdout is not None
        drain = asyncio.create_task(self._drain(proc.stdout))
        try:
            await asyncio.wait_for(proc.wait(), timeout=0.25)
        except TimeoutError:
            try:
                self._terminate(proc, kill=False)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except TimeoutError:
                try:
                    self._terminate(proc, kill=True)
                except ProcessLookupError:
                    pass
                await proc.wait()
        finally:
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    @staticmethod
    def _terminate(proc: asyncio.subprocess.Process, *, kill: bool) -> None:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL if kill else signal.SIGTERM)
        elif kill:
            proc.kill()
        else:
            proc.terminate()

    @staticmethod
    async def _drain(reader: asyncio.StreamReader) -> None:
        while await reader.read(65536):
            pass

    async def close(self) -> None:
        async with self._lock:
            await self._stop()


@dataclass
class _HTTPSession:
    protocol: str
    session_id: str | None
    headers: dict[str, str] = field(repr=False)


class MCPServerPool:
    """Bounded stdio workers and tenant-specific Streamable HTTP sessions."""

    def __init__(self, config: MCPServerConfig, *, allowed_hosts: set[str] | None = None,
                 allow_insecure: bool = False) -> None:
        self._cfg = config
        self._allowed_hosts = allowed_hosts
        self._allow_insecure = allow_insecure
        self._pool_size = max(1, int(config.pool_size))
        self._sem = asyncio.Semaphore(self._pool_size)
        self._stdio_sessions = [
            _StdioSession(config) for _ in range(min(self._pool_size, 2))
        ] if config.transport == MCPTransportType.STDIO and not config.in_process_handler else []
        self._idle_stdio: asyncio.Queue[_StdioSession] = asyncio.Queue()
        for session in self._stdio_sessions:
            self._idle_stdio.put_nowait(session)
        self._http_client: httpx.AsyncClient | None = None
        self._http_sessions: dict[str, _HTTPSession] = {}
        self._http_init_lock = asyncio.Lock()
        self._http_req_id = 0

    async def send_jsonrpc(
        self, *, method: str, params: dict[str, Any], headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        async with self._sem:
            if self._cfg.in_process_handler is not None:
                self._http_req_id += 1
                rid = self._http_req_id
                payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
                # The in-process adapter is synchronous and may perform disk I/O.
                reply = await asyncio.to_thread(self._cfg.in_process_handler, payload, dict(headers or {}))
                return _rpc_result(reply, rid)
            if self._stdio_sessions:
                session = await self._idle_stdio.get()
                try:
                    return await session.request(method, params)
                finally:
                    self._idle_stdio.put_nowait(session)
            return await self._http_request(method, params, headers or {})

    async def discover_tools(self, meta: dict[str, str], headers: dict[str, str]) -> list[Any]:
        params: dict[str, Any] = {"_meta": meta}
        tools: list[Any] = []
        cursors: set[str] = set()
        while True:
            result = await self.send_jsonrpc(method="tools/list", params=params, headers=headers)
            page = result.get("tools")
            if not isinstance(page, list):
                raise ValueError("Invalid MCP tools/list result")
            tools.extend(page)
            if len(tools) > 2048:
                raise ValueError("MCP catalog exceeds discovery limit")
            cursor = result.get("nextCursor")
            if cursor is None:
                return tools
            if not isinstance(cursor, str) or not cursor or cursor in cursors or len(cursors) >= 64:
                raise ValueError("Invalid MCP tools/list cursor")
            cursors.add(cursor)
            params["cursor"] = cursor

    async def _http_request(
        self, method: str, params: dict[str, Any], headers: dict[str, str],
    ) -> dict[str, Any]:
        tenant = headers.get("X-Tenant-ID", "")
        session = self._http_sessions.get(tenant)
        if session is None:
            async with self._http_init_lock:
                session = self._http_sessions.get(tenant)
                if session is None:
                    result, session_id = await self._post("initialize", _INITIALIZE_PARAMS, headers)
                    session = _HTTPSession(_negotiated_protocol(result), session_id, dict(headers))
                    try:
                        await self._post("notifications/initialized", {}, headers, session=session)
                    except (Exception, asyncio.CancelledError):
                        await self._close_http_session(session)
                        raise
                    self._http_sessions[tenant] = session
        session.headers = dict(headers)
        try:
            result, _ = await self._post(method, params, headers, session=session)
        except httpx.HTTPStatusError as exc:
            if session.session_id and exc.response.status_code == 404 and self._http_sessions.get(tenant) is session:
                self._http_sessions.pop(tenant)
            raise
        return result

    async def _post(
        self, method: str, params: dict[str, Any], extra_headers: dict[str, str],
        *, session: _HTTPSession | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        notification = method.startswith("notifications/")
        self._http_req_id += 1
        rid = self._http_req_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            payload["id"] = rid
        async with await self._http_stream("POST", extra_headers, session=session, payload=payload) as response:
            response.raise_for_status()
            if notification:
                if response.status_code != 202:
                    raise ValueError("MCP notification was not accepted")
                return {}, None
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type == "text/event-stream":
                result = await self._read_sse(response, rid)
            elif content_type == "application/json":
                body = bytearray()
                async for chunk in self._response_chunks(response):
                    body.extend(chunk)
                result = _rpc_result(json.loads(body), rid)
            else:
                raise ValueError("Unsupported MCP response content type")
            return result, response.headers.get("Mcp-Session-Id")

    async def _http_stream(
        self, verb: str, extra_headers: dict[str, str], *, session: _HTTPSession | None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        endpoint = self._cfg.endpoint_url or ""
        try:
            ips = await asyncio.to_thread(
                _resolve_mcp_endpoint, endpoint,
                allow_insecure_http=self._allow_insecure, allowed_private_hosts=self._allowed_hosts,
            )
        except ValueError as exc:
            raise _MCPEndpointError(str(exc)) from exc
        url = httpx.URL(endpoint)
        headers = httpx.Headers(self._cfg.headers)
        headers.update(extra_headers)
        headers["Accept"] = "application/json, text/event-stream"
        headers["Accept-Encoding"] = "identity"
        headers["Host"] = url.netloc.decode("ascii")
        headers.pop("Mcp-Session-Id", None)
        headers.pop("MCP-Protocol-Version", None)
        if session is not None:
            headers["MCP-Protocol-Version"] = session.protocol
            if session.session_id is not None:
                headers["Mcp-Session-Id"] = session.session_id
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._cfg.tool_timeout_seconds, connect=self._cfg.connect_timeout_seconds),
                limits=httpx.Limits(max_connections=self._pool_size),
                follow_redirects=False,
                trust_env=False,
            )
        # Connect to a validated IP while retaining the endpoint's Host header and TLS identity.
        target = url.copy_with(host=ips[0])
        return self._http_client.stream(
            verb, target, headers=headers, json=payload, extensions={"sni_hostname": url.host},
        )

    @staticmethod
    async def _response_chunks(response: httpx.Response) -> AsyncIterator[bytes]:
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("Unsupported MCP response encoding")
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > _MAX_MCP_MESSAGE_BYTES:
                raise ValueError("MCP response exceeds the size limit")
            yield chunk

    @classmethod
    async def _read_sse(cls, response: httpx.Response, rid: int) -> dict[str, Any]:
        buffer = bytearray()
        data: list[bytes] = []
        async for chunk in cls._response_chunks(response):
            buffer.extend(chunk)
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for raw_line in lines:
                line = raw_line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    data.append(line[5:].removeprefix(b" "))
                elif not line and data:
                    reply = json.loads(b"\n".join(data))
                    data.clear()
                    if isinstance(reply, dict) and "method" not in reply and reply.get("id") == rid:
                        return _rpc_result(reply, rid)
        raise ValueError("MCP SSE stream ended before the response")

    async def _close_http_session(self, session: _HTTPSession) -> None:
        if session.session_id:
            try:
                async with await self._http_stream("DELETE", session.headers, session=session):
                    pass
            except Exception:
                logger.debug("MCP HTTP session cleanup failed", exc_info=True)

    async def close(self) -> None:
        await asyncio.gather(*(session.close() for session in self._stdio_sessions))
        if self._http_client is not None:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(self._close_http_session(session) for session in self._http_sessions.values())),
                    timeout=1.0,
                )
            except TimeoutError:
                pass
            finally:
                self._http_sessions.clear()
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
        self._tool_specs_cache: dict[tuple[str, str], tuple[float, list[MCPToolSpec]]] = {}
        self._reactor: _MCPEventLoopRunner | None = None
        self._closed = False

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
        parser = TypeAdapter(MCPServerConfig)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            server_id = entry.get("server_id")
            if not isinstance(server_id, str) or not server_id.strip():
                continue
            entry = dict(entry)
            entry.pop("in_process_handler", None)
            transport = entry.get("transport")
            if isinstance(transport, str):
                entry["transport"] = transport.strip().lower().replace("-", "_")
            try:
                server = parser.validate_python(entry)
                if any(not math.isfinite(value) or value <= 0 for value in (
                    server.connect_timeout_seconds, server.tool_timeout_seconds, server.circuit_recovery_seconds,
                )) or server.pool_size < 1 or server.max_retries < 0 or server.circuit_failure_threshold < 1:
                    raise ValueError("Invalid MCP connection limits")
                if server.cwd and not Path(server.cwd).is_absolute():
                    server.cwd = str((path.parent.parent / server.cwd).resolve())
            except (ValidationError, ValueError, OSError):
                logger.warning("Ignoring malformed MCP server configuration for %s", server_id)
                continue
            result.append(server)
        return result

    def register_server(self, cfg: MCPServerConfig) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP bridge is closed")
            previous_pool = self._pools.get(cfg.server_id)
            self._servers[cfg.server_id] = cfg
            self._pools[cfg.server_id] = MCPServerPool(
                cfg,
                allowed_hosts=set(getattr(self._config, "mcp_allowed_private_hosts", []) or []),
                allow_insecure=bool(getattr(self._config, "mcp_allow_insecure_http", False)),
            )
            self._breakers[cfg.server_id] = MCPCircuitBreaker(
                failure_threshold=cfg.circuit_failure_threshold,
                recovery_seconds=cfg.circuit_recovery_seconds,
            )
            self._tool_specs_cache = {
                key: value for key, value in self._tool_specs_cache.items() if key[0] != cfg.server_id
            }
            reactor = self._reactor
        if previous_pool is not None and reactor is not None:
            reactor.submit(previous_pool.close(), timeout=3.0)

    def _get_reactor(self) -> _MCPEventLoopRunner:
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP bridge is closed")
            if self._reactor is None:
                self._reactor = _MCPEventLoopRunner()
            return self._reactor

    def list_tools(
        self,
        server_id: str | None = None,
        *,
        force_refresh: bool = False,
        tenant_id: str = "default",
        request_id: str = "local",
    ) -> list[MCPToolSpec]:
        if self._closed or not bool(getattr(self._config, "mcp_enabled", False)):
            return []
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
            cache_key = (sid, tenant_id if cfg.tenant_scoped else "")
            cached = self._tool_specs_cache.get(cache_key)
            if not force_refresh and ttl and cached and now - cached[0] < ttl:
                specs.extend(cached[1])
                continue

            breaker = self._breakers[sid]
            allowed, _, generation = breaker.acquire()
            if not allowed:
                continue
            pool = self._pools[sid]
            headers, meta = self._request_identity(cfg, tenant_id, request_id, "mcp:read")
            try:
                raw_tools = self._get_reactor().submit(
                    pool.discover_tools(meta, headers),
                    timeout=cfg.connect_timeout_seconds,
                )
            except Exception as exc:
                breaker.record_failure(generation)
                logger.warning("MCP tool discovery failed server_id=%s error=%s", sid, exc)
                continue
            breaker.record_success(generation)
            parsed_specs: list[MCPToolSpec] = []
            seen: set[str] = set()
            duplicates: set[str] = set()
            for item in raw_tools:
                if not isinstance(item, dict):
                    continue
                tname = str(item.get("name") or "").strip()
                if tname in seen:
                    duplicates.add(tname)
                seen.add(tname)
                input_schema = item.get("inputSchema")
                try:
                    _tool_input_validator(input_schema)
                except (ValueError, SchemaError, RecursionError):
                    logger.warning("Ignoring MCP tool with invalid input schema server_id=%s tool=%s", sid, tname)
                    continue
                if not tname:
                    continue
                parsed_specs.append(
                    MCPToolSpec(
                        server_id=sid,
                        tool_name=tname,
                        qualified_name=f"{sid}__{tname}",
                        description=str(item.get("description") or "").strip(),
                        input_schema=dict(input_schema),
                        annotations=MCPToolAnnotationsContract.from_raw(item.get("annotations")),
                    )
                )
            parsed_specs = [spec for spec in parsed_specs if spec.tool_name not in duplicates]
            with self._lock:
                if not self._closed and self._pools.get(sid) is pool and ttl:
                    self._tool_specs_cache[cache_key] = (time.monotonic(), parsed_specs)
            specs.extend(parsed_specs)

        return specs

    def get_tool_spec(
        self, server_id: str, tool_name: str, *, tenant_id: str = "default", request_id: str = "local",
    ) -> MCPToolSpec | None:
        specs = self.list_tools(server_id, tenant_id=tenant_id, request_id=request_id)
        return next((spec for spec in specs if spec.tool_name == tool_name), None)

    def _request_identity(
        self, cfg: MCPServerConfig, tenant_id: str, request_id: str, scope: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if not cfg.tenant_scoped:
            return {}, {}
        token = mint_scoped_tenant_token(
            secret=getattr(self._config, "mcp_tenant_token_secret", None),
            tenant_id=tenant_id, server_id=cfg.server_id, request_id=request_id, scope=scope,
        )
        headers = {"X-Tenant-ID": tenant_id, "X-Request-ID": request_id}
        meta = {"tenant_id": tenant_id, "request_id": request_id}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            meta["scoped_token"] = token
        return headers, meta

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        tenant_id: str = "default",
        request_id: str = "local",
        task_id: int | None = None,
        active_scopes: dict[str, Any] | None = None,
        stage: str = "task_execution",
        approved_registry: dict[str, dict[str, Any]] | None = None,
        consume_approval: Callable[[str, dict[str, Any]], bool] | None = None,
        readonly_required: bool = False,
        observer: RequestTrace | None = None,
    ) -> MCPCallResult:
        started = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        def fail(code: str, message: str, *, degraded: bool = False) -> MCPCallResult:
            if degraded and observer is not None:
                observer.record_degraded(f"mcp_{code}:{server_id}:{tool_name}")
            return MCPCallResult(
                ok=False, server_id=server_id, tool_name=tool_name, latency_ms=elapsed(),
                error_code=code, error_message=message,
            )

        cfg = self._servers.get(server_id)
        if self._closed or not getattr(self._config, "mcp_enabled", False) or cfg is None or not cfg.enabled:
            return fail("server_disabled", "MCP server is disabled or unknown")
        if not getattr(self._config, "mcp_max_tools_per_stage", 6) or not _scope_allows_tool(
            active_scopes, server_id=server_id, tool_name=tool_name, stage=stage, readonly=True,
        ):
            return fail("scope_forbidden", "MCP tool is outside the active Skill scope")
        metrics_registry.increment("mcp_call_total")
        breaker, pool = self._breakers[server_id], self._pools[server_id]
        spec = self.get_tool_spec(server_id, tool_name, tenant_id=tenant_id, request_id=request_id)
        if spec is None:
            if breaker.state != CircuitState.CLOSED:
                metrics_registry.increment("mcp_circuit_open_total")
                return fail("circuit_open", "MCP circuit is open", degraded=True)
            return fail("tool_unavailable", "MCP tool is unavailable", degraded=True)

        args = arguments if arguments is not None else {}
        try:
            args = json.loads(json.dumps(args, allow_nan=False))
            error = next(_tool_input_validator(spec.input_schema).iter_errors(args), None)
        except Exception:
            return fail("invalid_arguments", "Arguments or input schema could not be validated")
        if error is not None:
            return fail("invalid_arguments", f"Invalid argument at {error.json_path}")

        approval_key = ""
        entry = None
        if not spec.is_readonly:
            if (
                readonly_required
                or not _scope_allows_tool(
                    active_scopes, server_id=server_id, tool_name=tool_name, stage=stage, readonly=False,
                )
                or not getattr(self._config, "mcp_allow_hitl_write_tools", False)
                or tool_name not in cfg.allow_write_hitl_tools
                or consume_approval is None
            ):
                return fail("write_forbidden", "MCP write tool requires a configured approval gate")
            digest = compute_arguments_digest(args)
            approval_key = f"{server_id}:{tool_name}:{digest}"
            entry = (approved_registry or {}).get(approval_key)
            if not _matches_single_use_approval(
                entry, server_id=server_id, tool_name=tool_name, arguments_digest=digest,
                tenant_id=tenant_id, request_id=request_id, task_id=task_id,
            ):
                if isinstance(entry, dict) and (
                    entry.get("status") in {"rejected", "expired", "consumed"} or tool_approval_expired(entry)
                ):
                    return fail("approval_closed", "MCP tool approval has already been resolved")
                metrics_registry.increment("mcp_hitl_intercept_total")
                raise MCPHitlApprovalRequired(
                    server_id=server_id, tool_name=tool_name, arguments_digest=digest,
                    arguments_preview=args, task_id=task_id, tenant_id=tenant_id, request_id=request_id,
                    destructive_hint=spec.annotations.destructive_hint,
                    expires_at=time.time() + getattr(self._config, "mcp_approval_timeout_seconds", 86400),
                )

        allowed, circuit_state, generation = breaker.acquire()
        if not allowed:
            metrics_registry.increment("mcp_circuit_open_total")
            return fail("circuit_open", "MCP circuit is open", degraded=True)
        if entry is not None:
            assert consume_approval is not None
            try:
                consumed = consume_approval(approval_key, entry)
            except Exception:
                logger.exception("Could not persist MCP approval consumption")
                consumed = False
            if not consumed:
                breaker.release_probe(generation)
                return fail("approval_closed", "MCP tool approval is no longer available")

        scope = "mcp:read" if spec.is_readonly else "mcp:write_approved"
        headers, meta = self._request_identity(cfg, tenant_id, request_id, scope)
        timeout = min(cfg.tool_timeout_seconds, getattr(self._config, "mcp_tool_timeout_seconds", 10.0))
        deadline = time.monotonic() + timeout
        attempts = 1
        if spec.is_readonly and spec.annotations.idempotent_hint and circuit_state == CircuitState.CLOSED:
            attempts += max(0, cfg.max_retries)

        failure = ("transport_error", "MCP transport failed")
        try:
            for _ in range(attempts):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    result = self._get_reactor().submit(
                        pool.send_jsonrpc(
                            method="tools/call",
                            params={"name": tool_name, "arguments": args, "_meta": meta},
                            headers=headers,
                        ),
                        timeout=remaining,
                    )
                    normalized = self._normalize_call_result(cfg, tool_name, result, latency_ms=elapsed())
                except _MCPRemoteError as exc:
                    breaker.record_success(generation)
                    return fail("tool_error", str(exc))
                except (TimeoutError, httpx.TimeoutException):
                    failure = ("timeout", "MCP request timed out")
                except _MCPEndpointError:
                    failure = ("ssrf_blocked", "MCP endpoint failed network validation")
                    break
                except Exception as exc:
                    logger.debug("MCP call failed server_id=%s error=%s", server_id, type(exc).__name__)
                    failure = ("transport_error", "MCP transport failed")
                else:
                    breaker.record_success(generation)
                    if normalized.ok:
                        metrics_registry.increment("mcp_success_total")
                    return normalized
            breaker.record_failure(generation)
            if failure[0] == "timeout":
                metrics_registry.increment("mcp_timeout_total")
            return fail(*failure, degraded=True)
        finally:
            metrics_registry.add_latency("mcp_call_ms", elapsed())

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
        if raw_result.get("isError") is True:
            return MCPCallResult(
                ok=False, server_id=server_cfg.server_id, tool_name=tool_name, latency_ms=latency_ms,
                error_code="tool_error", error_message=text_blocks[0][:500] if text_blocks else "MCP tool failed",
            )

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
                    if not text or type(pno) is not int or pno <= 0 or not sha1:
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
                                "score": p.get("combined_score", p.get("relevance_score")),
                                "retrieval_mode": structured.get("retrieval_mode"),
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
                    if not text or type(pno) is not int or pno <= 0 or not sha1:
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
            if not normalized_evidence and not ({"passages", "pages", "excerpts", "companies"} & structured.keys()):
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
            ok=True,
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
        active_scopes_getter: Callable[[], dict[str, Any] | None] | None = None,
        stage: str = "task_execution",
        approved_registry_getter: Callable[[], dict[str, dict[str, Any]] | None] | None = None,
        consume_approval: Callable[[str, dict[str, Any]], bool] | None = None,
        observer_getter: Callable[[], RequestTrace | None] | None = None,
    ) -> list[MCPToolAdapter]:
        adapters: list[MCPToolAdapter] = []
        scopes = active_scopes_getter() if active_scopes_getter else None
        servers = scopes.get("allowed_servers", []) if isinstance(scopes, dict) else []
        refs = scopes.get("allowed_tools", []) if isinstance(scopes, dict) else []
        if not isinstance(servers, list) or not isinstance(refs, list):
            return adapters
        for server_id in dict.fromkeys(
            ref.partition(":")[0] for ref in refs if isinstance(ref, str)
            and _scope_allows_tool(
                scopes, server_id=ref.partition(":")[0], tool_name=ref.partition(":")[2], stage=stage, readonly=True,
            )
        ):
            for spec in self.list_tools(server_id, tenant_id=tenant_id_getter(), request_id=request_id_getter()):
                if not _scope_allows_tool(
                    scopes, server_id=spec.server_id, tool_name=spec.tool_name, stage=stage, readonly=spec.is_readonly,
                ):
                    continue
                adapters.append(
                    MCPToolAdapter(
                        spec=spec, bridge=self, evidence_store=evidence_store,
                        tenant_id_getter=tenant_id_getter, request_id_getter=request_id_getter,
                        active_scopes_getter=active_scopes_getter, stage=stage,
                        approved_registry_getter=approved_registry_getter, consume_approval=consume_approval,
                        observer_getter=observer_getter,
                    )
                )
        return adapters

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            reactor, self._reactor = self._reactor, None
            pools = list(self._pools.values())
            self._tool_specs_cache.clear()
        if reactor is not None:
            async def close_pools() -> None:
                await asyncio.gather(*(pool.close() for pool in pools), return_exceptions=True)

            try:
                reactor.submit(close_pools(), timeout=3.0)
            except TimeoutError:
                logger.warning("Timed out closing MCP pools")
            finally:
                reactor.close()


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
        active_scopes_getter: Callable[[], dict[str, Any] | None] | None = None,
        stage: str = "task_execution",
        approved_registry_getter: Callable[[], dict[str, dict[str, Any]] | None] | None = None,
        consume_approval: Callable[[str, dict[str, Any]], bool] | None = None,
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
        self._active_scopes_getter = active_scopes_getter
        self._stage = stage
        self._approved_registry_getter = approved_registry_getter
        self._consume_approval = consume_approval
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
        try:
            task_id = int(task_id_raw) if task_id_raw not in (None, "") else 0
        except (TypeError, ValueError):
            return json.dumps({"ok": False, "error_code": "invalid_arguments"})
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
            active_scopes=self._active_scopes_getter() if self._active_scopes_getter else None,
            stage=self._stage,
            approved_registry=approved_registry,
            consume_approval=self._consume_approval,
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
        if not hasattr(self, "_tools") or not isinstance(self._tools, dict):
            self._tools = {}
        self._mcp_tool_names: list[str] = []
        for tool in core_tools or ():
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if tool_name:
                self._tools[tool_name] = tool

        scopes = active_scopes if isinstance(active_scopes, dict) else {}
        requested_cap = scopes.get("max_tools_injected", max_mcp_tools)
        if type(requested_cap) is not int:
            return
        cap = min(max(0, requested_cap), max(0, max_mcp_tools))

        injected = 0
        for adapter in mcp_adapters or ():
            if injected >= cap:
                break
            if enforce_readonly and not adapter.spec.is_readonly:
                continue
            if adapter.name in self._tools or not _scope_allows_tool(
                scopes, server_id=adapter.spec.server_id, tool_name=adapter.spec.tool_name,
                stage=stage, readonly=adapter.spec.is_readonly,
            ):
                continue
            self._tools[adapter.name] = adapter
            self._mcp_tool_names.append(adapter.name)
            injected += 1

    @property
    def injected_mcp_tools(self) -> list[str]:
        return list(self._mcp_tool_names)

    def get_tool(self, name: str) -> Any | None:
        return getattr(self, "_tools", {}).get(name)

    def execute_tool(self, name: str, input_text: str) -> str:
        """Preserve MCPHitlApprovalRequired propagation instead of flattening to string."""
        if name in self._tools:
            try:
                if name in self._mcp_tool_names:
                    parameters = json.loads(input_text or "{}")
                    if not isinstance(parameters, dict):
                        return "MCP arguments must be a JSON object"
                    return self._tools[name].run(parameters)
                return super().execute_tool(name, input_text)
            except MCPHitlApprovalRequired:
                raise
            except Exception as exc:
                return f"错误：执行工具 '{name}' 时发生异常: {exc}"
        return super().execute_tool(name, input_text)
