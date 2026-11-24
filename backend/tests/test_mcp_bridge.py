import asyncio
import csv
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from main import create_app
from metrics import metrics_registry
from models import SummaryState, TodoItem
from services.evidence import EvidenceStore, detect_source_type
from services.evidence_index import EvidenceMemoryService, EvidenceRuntimeIndex
from services.finarag_mcp_server import FinaRAGAdapter, FinaRAGMCPServer, TenantACL
from services.mcp_bridge import (
    CircuitState,
    MCPBridgeManager,
    MCPCircuitBreaker,
    MCPHitlApprovalRequired,
    MCPServerConfig,
    MCPServerPool,
    MCPToolAnnotationsContract,
    MCPTransportType,
    ScopedToolRegistry,
    _validate_mcp_endpoint_url,
    compute_arguments_digest,
    mint_scoped_tenant_token,
    verify_scoped_tenant_token,
)
from services.reporter import ReportingService
from services.skill_pack import (
    SkillPackService,
    apply_skill_pack_overrides,
    load_pack,
    validate_pack,
)


class MCPBridgeAndFinaRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()

    def test_late_circuit_success_cannot_close_new_generation(self) -> None:
        breaker = MCPCircuitBreaker(failure_threshold=1)
        _, _, generation = breaker.acquire()
        breaker.record_failure(generation)
        breaker.record_success(generation)
        self.assertEqual(breaker.state, CircuitState.OPEN)

    def test_tool_approval_routes_reject_cross_tenant_duplicate_and_expired_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = Configuration.from_env(overrides={
                "auth_enabled": True, "tenant_api_keys": {"owner-key": "owner", "other-key": "other"},
                "mcp_enabled": True, "mcp_allow_hitl_write_tools": True,
                "request_state_enabled": True, "request_state_dir": directory,
            }, load_env_file=False)
            app = create_app(base_config=cfg)
            client = TestClient(app)
            store = app.state.request_state_store
            owner = {"X-API-Key": "owner-key"}

            def pause(request_id, *, expired=False):
                args = {"name": "item"}
                approval = MCPHitlApprovalRequired(
                    server_id="ops", tool_name="write", arguments_preview=args,
                    arguments_digest=compute_arguments_digest(args), request_id=request_id, tenant_id="owner",
                    expires_at=time.time() - 1 if expired else None,
                ).to_dict()
                saved = store.save(request_id, {
                    "phase": "awaiting_tool_approval", "status": "awaiting_approval",
                    "pending_tool_approval": approval,
                }, tenant_id="owner")
                return approval["approval_id"], saved["state_version"]

            approval_id, version = pause("req-reject")
            path = f"/requests/req-reject/tool-approvals/{approval_id}"
            self.assertEqual(client.post(path, json={"decision": "approve"}).status_code, 401)
            self.assertEqual(client.post(path, json={"decision": "approve"}, headers={"X-API-Key": "other-key"}).status_code, 404)
            self.assertEqual(client.post("/requests/req-reject/resume", headers=owner).status_code, 409)
            self.assertEqual(client.put("/requests/req-reject/plan", headers=owner, json={
                "tasks": [{"id": 1, "title": "New", "intent": "Test", "query": "New plan"}],
            }).status_code, 409)
            self.assertEqual(client.post(path, json={"decision": "reject"}, headers=owner).status_code, 200)
            self.assertEqual(client.post(path, json={"decision": "reject"}, headers=owner).status_code, 200)
            self.assertEqual(client.post(path, json={"decision": "approve"}, headers=owner).status_code, 409)
            self.assertEqual(client.post("/requests/req-reject/resume", headers=owner).status_code, 409)
            self.assertEqual(store.load("req-reject")["state_version"], version + 1)

            approval_id, version = pause("req-expired", expired=True)
            expired_body = {"request_id": "req-expired", "approval_id": approval_id, "decision": "approve"}
            self.assertEqual(client.post("/api/research/tool-approval", json=expired_body, headers=owner).status_code, 410)
            self.assertEqual(client.post("/requests/req-expired/resume/stream", headers=owner).status_code, 410)
            self.assertEqual(store.load("req-expired")["state_version"], version + 1)

    def test_streamable_http_negotiates_per_tenant_and_stops_at_matching_sse_frame(self) -> None:
        calls = []
        closed = []

        class Events(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b": keepalive\r\n\r\n"
                yield b'data: {"jsonrpc":"2.0",\r\n'
                yield f'data: "id":{request_id[0]},"result":{{"tools":[]}}}}\r\n\r\n'.encode()
                raise AssertionError("The bridge must close the stream after the matching response")

            async def aclose(self):
                closed.append(True)

        request_id = [0]

        def respond(request):
            tenant = request.headers["X-Tenant-ID"]
            self.assertEqual(request.url.host, "93.184.216.34")
            self.assertEqual(request.headers["Host"], "mcp.example")
            self.assertEqual(request.extensions["sni_hostname"], "mcp.example")
            if request.method == "DELETE":
                self.assertEqual(request.headers["Mcp-Session-Id"], tenant)
                calls.append((tenant, "DELETE"))
                return httpx.Response(204)
            payload = json.loads(request.content)
            method = payload["method"]
            calls.append((tenant, method))
            if method == "initialize":
                return httpx.Response(200, headers={"Mcp-Session-Id": tenant}, json={
                    "jsonrpc": "2.0", "id": payload["id"], "result": {"protocolVersion": "2025-03-26"},
                })
            self.assertEqual(request.headers["Mcp-Session-Id"], tenant)
            self.assertEqual(request.headers["MCP-Protocol-Version"], "2025-03-26")
            if method == "notifications/initialized":
                self.assertNotIn("id", payload)
                return httpx.Response(202)
            request_id[0] = payload["id"]
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Events())

        async def run():
            pool = MCPServerPool(MCPServerConfig(
                server_id="http", transport=MCPTransportType.STREAMABLE_HTTP, endpoint_url="https://mcp.example/mcp",
            ))
            pool._http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            try:
                for tenant in ("a", "b", "a"):
                    result = await pool.send_jsonrpc(method="tools/list", params={}, headers={"X-Tenant-ID": tenant})
                    self.assertEqual(result, {"tools": []})
            finally:
                await pool.close()

        with patch("services.mcp_bridge._resolve_mcp_endpoint", return_value=["93.184.216.34"]):
            asyncio.run(run())
        self.assertEqual(len(closed), 3)
        self.assertEqual(calls.count(("a", "initialize")), 1)
        self.assertEqual(calls.count(("b", "initialize")), 1)
        self.assertEqual(calls[-2:], [("a", "DELETE"), ("b", "DELETE")])

    def test_bundled_stdio_server_starts_from_yaml_and_is_reaped(self) -> None:
        cfg = Configuration.from_env(overrides={"mcp_enabled": True}, load_env_file=False)
        server = replace(MCPBridgeManager._load_servers_from_config(cfg)[0], enabled=True, pool_size=1)
        bridge = MCPBridgeManager(cfg, servers=[server])
        try:
            self.assertEqual(len(bridge.list_tools("finarag")), 4)
            process = bridge._pools["finarag"]._stdio_sessions[0]._proc
            self.assertIsNotNone(process)
        finally:
            bridge.close()
        with self.assertRaises(ProcessLookupError):
            os.kill(process.pid, 0)

    def test_stdio_timeout_reaps_worker_and_next_call_restarts_it(self) -> None:
        script = """
import json
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    if request["method"] == "initialize":
        result = {"protocolVersion": "2025-03-26"}
    elif request["method"] == "tools/list":
        result = {"tools": [{"name": "read", "inputSchema": {"type": "object"},
                             "annotations": {"readOnlyHint": True}}]}
    elif request["params"]["arguments"].get("slow"):
        time.sleep(20)
    else:
        result = {"structuredContent": {"ready": True}}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""
        cfg = Configuration.from_env(overrides={"mcp_enabled": True}, load_env_file=False)
        server = MCPServerConfig(
            server_id="local-test", command=[sys.executable, "-c", script], pool_size=1, tool_timeout_seconds=0.2,
        )
        bridge = MCPBridgeManager(cfg, servers=[server])
        scope = {"allowed_servers": ["local-test"], "allowed_tools": ["local-test:read"]}
        try:
            self.assertEqual(len(bridge.list_tools()), 1)
            session = bridge._pools["local-test"]._stdio_sessions[0]
            first = session._proc
            timed_out = bridge.call_tool(
                server_id="local-test", tool_name="read", arguments={"slow": True}, active_scopes=scope,
            )
            self.assertEqual(timed_out.error_code, "timeout")
            deadline = time.monotonic() + 2
            while first.returncode is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(first.returncode)
            recovered = bridge.call_tool(
                server_id="local-test", tool_name="read", arguments={}, active_scopes=scope,
            )
            self.assertTrue(recovered.ok)
            self.assertNotEqual(first.pid, session._proc.pid)
        finally:
            bridge.close()

    def test_discovery_paginates_and_isolates_failed_and_disabled_servers(self) -> None:
        history = []

        def handler(payload, headers):
            cursor = payload["params"].get("cursor")
            tenant = headers["X-Tenant-ID"]
            history.append((tenant, cursor))
            result = {
                "tools": [{"name": f"{tenant}_{cursor or 'first'}", "inputSchema": {"type": "object"}}],
            }
            if cursor is None:
                result["nextCursor"] = "next"
            return {"jsonrpc": "2.0", "id": payload["id"], "result": result}

        cfg = Configuration.from_env(overrides={"mcp_enabled": True}, load_env_file=False)
        bridge = MCPBridgeManager(cfg, servers=[
            MCPServerConfig(server_id="failed", in_process_handler=lambda *_: None),
            MCPServerConfig(server_id="disabled", enabled=False, in_process_handler=handler),
            MCPServerConfig(server_id="working", in_process_handler=handler),
        ])
        try:
            self.assertEqual([spec.tool_name for spec in bridge.list_tools(tenant_id="a")], ["a_first", "a_next"])
            self.assertEqual([spec.tool_name for spec in bridge.list_tools(tenant_id="b")], ["b_first", "b_next"])
            self.assertEqual([spec.tool_name for spec in bridge.list_tools(tenant_id="a")], ["a_first", "a_next"])
            self.assertEqual(history, [("a", None), ("a", "next"), ("b", None), ("b", "next")])
        finally:
            bridge.close()

    def test_finarag_http_jsonrpc_envelope_and_tenant_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FinaRAGAdapter(data_root=directory, acl=TenantACL({"team": ["Alpha"]}), token_secret="secret")
            app = FastAPI()
            app.include_router(FinaRAGMCPServer(adapter).create_http_router())
            client = TestClient(app)
            self.assertEqual(client.post("/mcp", content="{").json()["error"]["code"], -32700)
            self.assertEqual(client.post("/mcp", json=[]).json()["error"]["code"], -32600)
            message = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            self.assertEqual(client.post("/mcp", json=message).status_code, 202)
            self.assertEqual(client.post("/mcp", json=message, headers={"Origin": "https://example.com"}).status_code, 403)
            message.update(id=1, method="tools/call", params={"name": "finarag_list_companies"})
            self.assertEqual(client.post("/mcp", json=message).json()["error"]["code"], -32003)
            token = mint_scoped_tenant_token(secret="secret", tenant_id="team", server_id="finarag", request_id="req")
            headers = {"Authorization": f"Bearer {token}", "X-Tenant-ID": "team", "X-Request-ID": "req"}
            self.assertIn("result", client.post("/mcp", json=message, headers=headers).json())
            self.assertEqual(
                client.post("/mcp", json=message, headers={**headers, "X-Request-ID": "other"}).json()["error"]["code"],
                -32003,
            )
            message["params"]["arguments"] = []
            self.assertEqual(client.post("/mcp", json=message, headers=headers).json()["error"]["code"], -32602)

    def test_finarag_ignores_external_report_links_and_preserves_page_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "data"
            reports = root / "databases/chunked_reports"
            reports.mkdir(parents=True)
            (root / "subset.csv").write_text("company_name,sha1\nAlpha,sha_alpha\nOutside,sha_outside\n")
            pages = [{"page": n, "text": f"Revenue for period {n}"} for n in (2, 3)]
            report = {"metainfo": {"company_name": "Alpha", "sha1_name": "sha_alpha"}, "content": {"pages": pages}}
            (reports / "alpha.json").write_text(json.dumps(report))
            outside = parent / "outside.json"
            outside.write_text(json.dumps({**report, "metainfo": {"company_name": "Outside", "sha1_name": "sha_outside"}}))
            (reports / "outside.json").symlink_to(outside)
            adapter = FinaRAGAdapter(data_root=root, acl=TenantACL({"default": ["*"]}))
            server = FinaRAGMCPServer(adapter)
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "finarag_filings", "arguments": {"company_name": "Alpha", "query": "revenue"}},
            }
            reply = server.handle_jsonrpc(payload)["result"]
            self.assertEqual(reply["structuredContent"]["retrieval_mode"], "bm25")
            self.assertTrue(reply["structuredContent"]["notices"])
            spec = MCPServerConfig(server_id="finarag")
            normalized = MCPBridgeManager._normalize_call_result(spec, "finarag_filings", reply, latency_ms=0)
            evidence = EvidenceStore().record_mcp_results(
                task_id=1, server_id="finarag", tool_name="finarag_filings", items=normalized.normalized_evidence,
            )
            self.assertEqual([item["source_id"] for item in evidence], ["T1-M1", "T1-M2"])
            self.assertEqual([item["url"] for item in evidence], ["finarag://sha_alpha#page=2", "finarag://sha_alpha#page=3"])
            self.assertTrue(all(item["domain"] == "finarag.mcp.internal" for item in evidence))
            payload["params"]["arguments"]["company_name"] = "Outside"
            self.assertTrue(server.handle_jsonrpc(payload)["result"]["isError"])

    def test_skill_composition_intersects_permissions_and_respects_empty_stage(self) -> None:
        pack = load_pack(BACKEND_SRC.parent / "skills/research/financial-research/SKILL.md")
        cfg = Configuration.from_env(overrides={"active_mcp_tool_scopes": pack.mcp_tool_scopes}, load_env_file=False)
        secondary = replace(pack, mcp_tool_scopes={
            "allowed_servers": ["finarag"], "allowed_tools": ["finarag:finarag_filings"],
            "stage_allowlist": [], "max_tools_injected": 0,
        })
        scoped, _ = apply_skill_pack_overrides(cfg, pack, secondary_pack=secondary)
        self.assertEqual(scoped.active_mcp_tool_scopes["allowed_tools"], ["finarag:finarag_filings"])
        self.assertEqual(scoped.active_mcp_tool_scopes["stage_allowlist"], [])
        self.assertEqual(scoped.active_mcp_tool_scopes["max_tools_injected"], 0)
        bad_pack = replace(pack, mcp_tool_scopes={"allowed_servers": 123, "allowed_tools": None})
        self.assertTrue(validate_pack(bad_pack, char_limit=1200, directory_name="financial-research"))

    def test_ssrf_guard_blocks_private_and_allows_allowlisted_hosts(self) -> None:
        with self.assertRaises(ValueError):
            _validate_mcp_endpoint_url(
                "http://127.0.0.1:8010/mcp",
                allow_insecure_http=False,
                allowed_private_hosts=set(),
            )
        with self.assertRaises(ValueError):
            _validate_mcp_endpoint_url(
                "https://169.254.169.254/latest",
                allow_insecure_http=False,
                allowed_private_hosts=set(),
            )
        with patch(
            "services.mcp_bridge.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ):
            self.assertEqual(
                _validate_mcp_endpoint_url(
                    "https://mcp.example.com/mcp",
                    allow_insecure_http=False,
                    allowed_private_hosts=set(),
                ),
                "https://mcp.example.com/mcp",
            )
        self.assertEqual(
            _validate_mcp_endpoint_url(
                "http://finarag-mcp:8010/mcp",
                allow_insecure_http=True,
                allowed_private_hosts={"finarag-mcp"},
            ),
            "http://finarag-mcp:8010/mcp",
        )

    def test_hmac_scoped_token_lifecycle_and_fail_closed(self) -> None:
        secret = "unit-test-hmac-secret"
        now = 1_700_000_000.0
        token = mint_scoped_tenant_token(
            secret=secret,
            tenant_id="tenant-a",
            server_id="finarag",
            request_id="req-1",
            ttl_seconds=60,
            now=now,
        )
        claims = verify_scoped_tenant_token(
            token=token,
            secret=secret,
            expected_server_id="finarag",
            expected_tenant_id="tenant-a",
            now=now + 10,
        )
        self.assertEqual(claims["tenant_id"], "tenant-a")

        with self.assertRaises(PermissionError):
            verify_scoped_tenant_token(
                token=token,
                secret=secret,
                expected_server_id="finarag",
                expected_tenant_id="tenant-b",
                now=now + 10,
            )
        with self.assertRaises(PermissionError):
            verify_scoped_tenant_token(
                token=token + "tampered",
                secret=secret,
                expected_server_id="finarag",
                now=now + 10,
            )
        with self.assertRaises(PermissionError):
            verify_scoped_tenant_token(
                token=token,
                secret=secret,
                expected_server_id="finarag",
                now=now + 120,
            )

    def test_circuit_breaker_transitions_closed_open_half_open_closed(self) -> None:
        breaker = MCPCircuitBreaker(failure_threshold=2, recovery_seconds=0.06)
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.OPEN)

        allowed_open, state_open = breaker.allow_request()
        self.assertFalse(allowed_open)
        self.assertEqual(state_open, CircuitState.OPEN)

        time.sleep(0.08)
        allowed_probe, state_probe = breaker.allow_request()
        self.assertTrue(allowed_probe)
        self.assertEqual(state_probe, CircuitState.HALF_OPEN)
        # Second concurrent caller during HALF_OPEN probe is held back
        allowed_second, _ = breaker.allow_request()
        self.assertFalse(allowed_second)

        breaker.record_success()
        self.assertEqual(breaker.state, CircuitState.CLOSED)

    def test_tool_calls_require_exact_active_skill_allowlist(self) -> None:
        handler_calls: list[str] = []

        def handler(payload: dict, _headers: dict) -> dict:
            req_id = payload.get("id")
            if payload.get("method") == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "allowed",
                                "inputSchema": {"type": "object", "properties": {}},
                                "annotations": {"readOnlyHint": True},
                            },
                            {
                                "name": "hidden",
                                "inputSchema": {"type": "object", "properties": {}},
                                "annotations": {"readOnlyHint": True},
                            },
                        ]
                    },
                }
            handler_calls.append(payload["params"]["name"])
            return {"jsonrpc": "2.0", "id": req_id, "result": {"structuredContent": {"ok": True}}}

        cfg = Configuration.from_env(overrides={"mcp_enabled": True}, load_env_file=False)
        bridge = MCPBridgeManager(
            cfg,
            servers=[MCPServerConfig(server_id="scoped", in_process_handler=handler)],
        )
        scopes = {
            "allowed_servers": ["scoped"],
            "allowed_tools": ["scoped:allowed"],
            "stage_allowlist": ["task_execution"],
            "readonly_only": True,
        }
        try:
            no_scope = bridge.call_tool(
                server_id="scoped",
                tool_name="allowed",
                arguments={},
                tenant_id="tenant-a",
                request_id="req-scope",
            )
            self.assertEqual(no_scope.error_code, "scope_forbidden")

            hidden = bridge.call_tool(
                server_id="scoped",
                tool_name="hidden",
                arguments={},
                tenant_id="tenant-a",
                request_id="req-scope",
                active_scopes=scopes,
            )
            self.assertEqual(hidden.error_code, "scope_forbidden")

            allowed = bridge.call_tool(
                server_id="scoped",
                tool_name="allowed",
                arguments={},
                tenant_id="tenant-a",
                request_id="req-scope",
                active_scopes=scopes,
            )
            self.assertTrue(allowed.ok)
            self.assertEqual(handler_calls, ["allowed"])
        finally:
            bridge.close()

    def test_tool_schema_validation_rejects_malformed_metadata_and_arguments(self) -> None:
        def handler(payload: dict, _headers: dict) -> dict:
            req_id = payload.get("id")
            if payload.get("method") == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "valid",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                    "additionalProperties": False,
                                },
                                "annotations": {"readOnlyHint": True},
                            },
                            {"name": "malformed", "inputSchema": {"properties": []}},
                        ]
                    },
                }
            return {"jsonrpc": "2.0", "id": req_id, "result": {"structuredContent": {"ok": True}}}

        cfg = Configuration.from_env(overrides={"mcp_enabled": True}, load_env_file=False)
        bridge = MCPBridgeManager(
            cfg,
            servers=[MCPServerConfig(server_id="schema", in_process_handler=handler)],
        )
        scopes = {
            "allowed_servers": ["schema"],
            "allowed_tools": ["schema:valid"],
            "stage_allowlist": ["task_execution"],
            "readonly_only": True,
        }
        try:
            self.assertEqual([spec.tool_name for spec in bridge.list_tools()], ["valid"])
            bad_arguments = bridge.call_tool(
                server_id="schema",
                tool_name="valid",
                arguments={"unexpected": "value"},
                tenant_id="tenant-a",
                request_id="req-schema",
                active_scopes=scopes,
            )
            self.assertEqual(bad_arguments.error_code, "invalid_arguments")
        finally:
            bridge.close()

    def test_tool_metadata_cache_refreshes_after_ttl(self) -> None:
        names = ["first"]

        def handler(payload: dict, _headers: dict) -> dict:
            req_id = payload.get("id")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": names[0],
                            "inputSchema": {"type": "object", "properties": {}},
                            "annotations": {"readOnlyHint": True},
                        }
                    ]
                },
            }

        cfg = Configuration.from_env(
            overrides={"mcp_enabled": True, "mcp_tool_cache_ttl_seconds": 1},
            load_env_file=False,
        )
        bridge = MCPBridgeManager(
            cfg,
            servers=[MCPServerConfig(server_id="cache", in_process_handler=handler)],
        )
        try:
            self.assertEqual(bridge.get_tool_spec("cache", "first").tool_name, "first")
            names[0] = "second"
            with patch("services.mcp_bridge.time.monotonic", return_value=time.monotonic() + 2):
                self.assertIsNone(bridge.get_tool_spec("cache", "first"))
                self.assertEqual(bridge.get_tool_spec("cache", "second").tool_name, "second")
        finally:
            bridge.close()

    def test_finarag_mcp_server_acl_fail_closed_and_evidence_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunks_dir = root / "databases" / "chunked_reports"
            chunks_dir.mkdir(parents=True)
            with (root / "subset.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["company_name", "sha1", "cur", "major_industry"])
                writer.writeheader()
                writer.writerow({"company_name": "AlphaCorp", "sha1": "sha_alpha", "cur": "USD", "major_industry": "Semis"})
                writer.writerow({"company_name": "BetaBank", "sha1": "sha_beta", "cur": "EUR", "major_industry": "Finance"})

            (chunks_dir / "alpha.json").write_text(
                json.dumps(
                    {
                        "metainfo": {"company_name": "AlphaCorp", "sha1_name": "sha_alpha", "currency": "USD"},
                        "content": {
                            "pages": [{"page": 12, "text": "AlphaCorp FY2025 gross margin improved to 74.2%."}],
                            "chunks": [{"page": 12, "text": "AlphaCorp FY2025 gross margin improved to 74.2%.", "type": "content"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (chunks_dir / "beta.json").write_text(
                json.dumps(
                    {
                        "metainfo": {"company_name": "BetaBank", "sha1_name": "sha_beta", "currency": "EUR"},
                        "content": {
                            "pages": [{"page": 5, "text": "BetaBank Tier 1 capital ratio reached 15.1%."}],
                            "chunks": [{"page": 5, "text": "BetaBank Tier 1 capital ratio reached 15.1%.", "type": "content"}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            secret = "shared-mcp-secret"
            acl = TenantACL({"tenant-a": ["AlphaCorp"], "tenant-b": ["BetaBank"]}, enforce=True)
            adapter = FinaRAGAdapter(data_root=root, acl=acl, token_secret=secret)
            server = FinaRAGMCPServer(adapter)

            cfg = Configuration.from_env(
                overrides={
                    "mcp_enabled": True,
                    "mcp_tenant_token_secret": secret,
                    "skill_pack_enabled": True,
                },
                load_env_file=False,
            )
            server_cfg = MCPServerConfig(
                server_id="finarag",
                transport=MCPTransportType.STDIO,
                trust_tier="curated_private",
                in_process_handler=lambda payload, headers: server.handle_jsonrpc(payload, headers=headers),
            )
            bridge = MCPBridgeManager(cfg, servers=[server_cfg])
            fin_pack = load_pack(BACKEND_SRC.parent / "skills/research/financial-research/SKILL.md")
            scope = fin_pack.mcp_tool_scopes
            try:
                # 1. Fail-Closed ACL checks
                res_a = bridge.call_tool(
                    server_id="finarag",
                    tool_name="finarag_list_companies",
                    arguments={},
                    tenant_id="tenant-a",
                    request_id="req-acl",
                    active_scopes=scope,
                )
                self.assertTrue(res_a.ok)
                self.assertEqual([c["company_name"] for c in res_a.structured_payload["companies"]], ["AlphaCorp"])

                denied_cross = bridge.call_tool(
                    server_id="finarag",
                    tool_name="finarag_filings",
                    arguments={"company_name": "BetaBank", "query": "capital ratio"},
                    tenant_id="tenant-a",
                    request_id="req-acl",
                    active_scopes=scope,
                )
                self.assertFalse(denied_cross.ok)
                self.assertIn("not authorized", denied_cross.error_message or "")

                denied_unknown = bridge.call_tool(
                    server_id="finarag",
                    tool_name="finarag_list_companies",
                    arguments={},
                    tenant_id="unknown-tenant",
                    request_id="req-acl",
                    active_scopes=scope,
                )
                self.assertTrue(denied_unknown.ok)
                self.assertEqual(denied_unknown.structured_payload["companies"], [])

                # 2. Anti-Tool-Bloat via SkillPack + ScopedToolRegistry
                svc = SkillPackService(cfg)
                self.assertEqual(validate_pack(fin_pack, char_limit=1200, directory_name="financial-research"), [])
                neutral_sel = svc.select("今天杭州天气怎么样")
                self.assertEqual(neutral_sel.pack.mcp_tool_scopes, {})

                adapters = bridge.build_tool_adapters(
                    evidence_store=EvidenceStore(),
                    tenant_id_getter=lambda: "tenant-a",
                    request_id_getter=lambda: "req-acl",
                    active_scopes_getter=lambda: scope,
                )
                reg_neutral = ScopedToolRegistry(core_tools=[], mcp_adapters=adapters, active_scopes={}, stage="task_execution")
                self.assertEqual(reg_neutral.injected_mcp_tools, [])
                reg_plan = ScopedToolRegistry(
                    core_tools=[],
                    mcp_adapters=adapters,
                    active_scopes=fin_pack.mcp_tool_scopes,
                    stage="planning",
                )
                self.assertEqual(reg_plan.injected_mcp_tools, [])
                reg_exec = ScopedToolRegistry(
                    core_tools=[],
                    mcp_adapters=adapters,
                    active_scopes=fin_pack.mcp_tool_scopes,
                    stage="task_execution",
                )
                self.assertEqual(len(reg_exec.injected_mcp_tools), 4)

                # 3. Record MCP result -> [T1-M1] -> private archival skip -> report whitelist preservation
                mem_dir = Path(tmpdir) / "evidence_mem"
                ev_cfg = Configuration.from_env(
                    overrides={"evidence_runtime_enabled": True, "evidence_memory_enabled": True, "evidence_memory_dir": str(mem_dir)},
                    load_env_file=False,
                )
                mem = EvidenceMemoryService(ev_cfg)
                runtime = EvidenceRuntimeIndex(ev_cfg, archive_loader=mem)
                store = EvidenceStore(listeners=[runtime, mem])

                res_filings = bridge.call_tool(
                    server_id="finarag",
                    tool_name="finarag_filings",
                    arguments={"company_name": "AlphaCorp", "query": "gross margin"},
                    tenant_id="tenant-a",
                    request_id="req-acl",
                    task_id=1,
                    active_scopes=scope,
                )
                self.assertTrue(res_filings.ok)
                recorded = store.record_mcp_results(
                    task_id=1,
                    query="gross margin",
                    server_id="finarag",
                    tool_name="finarag_filings",
                    items=res_filings.normalized_evidence,
                    tenant_id="tenant-a",
                    trust_tier="curated_private",
                )
                self.assertEqual(recorded[0]["source_id"], "T1-M1")
                self.assertEqual(detect_source_type(recorded[0]), "mcp_curated_filing")
                self.assertFalse(mem._manifest_path.exists())

                sanitized = ReportingService._sanitize_chunk_citations(
                    "AlphaCorp 毛利率为 74.2% [T1-M1]，伪造 [T9-M9] 删除。",
                    {"T1-M1"},
                )
                self.assertIn("[T1-M1]", sanitized)
                self.assertNotIn("[T9-M9]", sanitized)
            finally:
                bridge.close()

    def test_readonly_enforcement_hitl_propagation_and_approval_api(self) -> None:
        # Missing readOnlyHint must be Fail-Closed (is_safe_readonly == False)
        self.assertFalse(MCPToolAnnotationsContract.from_raw({}).is_safe_readonly)
        self.assertFalse(MCPToolAnnotationsContract.from_raw({"destructiveHint": False}).is_safe_readonly)

        def write_handler(payload: dict, _headers: dict) -> dict:
            method = payload.get("method")
            rid = payload.get("id")
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "tools": [
                            {
                                "name": "write_ticket",
                                "description": "Create an internal ticket",
                                "inputSchema": {"type": "object"},
                                "annotations": {"readOnlyHint": False, "destructiveHint": True},
                            }
                        ]
                    },
                }
            if method == "tools/call":
                args = (payload.get("params") or {}).get("arguments") or {}
                return {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"structuredContent": {"ticket_id": "TCK-42", "title": args.get("title")}},
                }
            return {"jsonrpc": "2.0", "id": rid, "result": {}}

        cfg = Configuration.from_env(
            overrides={"mcp_enabled": True, "mcp_enforce_readonly": True, "mcp_allow_hitl_write_tools": False},
            load_env_file=False,
        )
        srv_cfg = MCPServerConfig(
            server_id="ops",
            transport=MCPTransportType.STDIO,
            allow_write_hitl_tools=["write_ticket"],
            in_process_handler=write_handler,
        )
        scope = {"allowed_servers": ["ops"], "allowed_tools": ["ops:write_ticket"], "readonly_only": False}
        bridge_strict = MCPBridgeManager(cfg, servers=[srv_cfg])
        try:
            blocked = bridge_strict.call_tool(
                server_id="ops",
                tool_name="write_ticket",
                arguments={"title": "Audit"},
                tenant_id="t1",
                request_id="r1",
                active_scopes=scope,
            )
            self.assertFalse(blocked.ok)
            self.assertEqual(blocked.error_code, "write_forbidden")
        finally:
            bridge_strict.close()

        # Enable HITL for allowlisted write tool & test API approval + StreamCoordinator propagation
        cfg_hitl = Configuration.from_env(
            overrides={
                "mcp_enabled": True,
                "mcp_enforce_readonly": True,
                "mcp_allow_hitl_write_tools": True,
                "request_state_enabled": True,
            },
            load_env_file=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_hitl = cfg_hitl.model_copy(update={"request_state_dir": tmpdir})
            bridge_hitl = MCPBridgeManager(cfg_hitl, servers=[srv_cfg])
            try:
                args = {"title": "Approve me"}
                digest = compute_arguments_digest(args)
                with self.assertRaises(MCPHitlApprovalRequired) as exc_ctx:
                    bridge_hitl.call_tool(
                        server_id="ops",
                        tool_name="write_ticket",
                        arguments=args,
                        tenant_id="default",
                        request_id="req-hitl-1",
                        task_id=1,
                        active_scopes=scope,
                        consume_approval=lambda *_: False,
                    )
                pending = exc_ctx.exception.to_dict()

                app = create_app(base_config=cfg_hitl)
                app.state.request_state_store.save(
                    "req-hitl-1",
                    {
                        "request_id": "req-hitl-1",
                        "tenant_id": "default",
                        "topic": "ops audit",
                        "phase": "awaiting_tool_approval",
                        "status": "awaiting_approval",
                        "todo_items": [],
                        "pending_tool_approval": pending,
                        "approved_tool_calls": {},
                    },
                    tenant_id="default",
                )
                client = TestClient(app)
                resp = client.post(
                    f"/requests/req-hitl-1/tool-approvals/{pending['approval_id']}",
                    json={"decision": "approve", "reason": "verified"},
                )
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["status"], "approved_once")

                updated_snap = app.state.request_state_store.load("req-hitl-1", tenant_id="default")
                from state_manager import StateManagerMixin

                class Consumer(StateManagerMixin):
                    request_id = "req-hitl-1"
                    tenant_id = "default"
                    _request_state_store = app.state.request_state_store
                    _active_state = SummaryState(
                        approved_tool_calls=dict(updated_snap["approved_tool_calls"]),
                        state_version=updated_snap["state_version"],
                    )

                consumer = Consumer()
                stale_consumer = Consumer()
                stale_consumer._active_state = replace(
                    consumer._active_state, approved_tool_calls=dict(consumer._active_state.approved_tool_calls),
                )
                res_after = bridge_hitl.call_tool(
                    server_id="ops",
                    tool_name="write_ticket",
                    arguments=args,
                    tenant_id="default",
                    request_id="req-hitl-1",
                    task_id=1,
                    active_scopes=scope,
                    approved_registry=consumer._active_state.approved_tool_calls,
                    consume_approval=consumer._consume_mcp_approval,
                )
                self.assertTrue(res_after.ok)
                self.assertEqual(res_after.structured_payload["ticket_id"], "TCK-42")
                # Single-use approval must be consumed after one call
                self.assertEqual(
                    consumer._active_state.approved_tool_calls[f"ops:write_ticket:{digest}"]["status"],
                    "consumed",
                )
                replay = bridge_hitl.call_tool(
                    server_id="ops", tool_name="write_ticket", arguments=args, tenant_id="default",
                    request_id="req-hitl-1", task_id=1, active_scopes=scope,
                    approved_registry=stale_consumer._active_state.approved_tool_calls,
                    consume_approval=stale_consumer._consume_mcp_approval,
                )
                self.assertEqual(replay.error_code, "approval_closed")
            finally:
                bridge_hitl.close()

    def test_vllm_auto_query_prefetch_and_stream_coordinator_hitl_propagation(self) -> None:
        from stream_coordinator import StreamCoordinatorMixin
        from task_executor import TaskExecutorMixin

        class _DummyAgent(StreamCoordinatorMixin, TaskExecutorMixin):
            def __init__(self, cfg, bridge) -> None:
                self.config = cfg
                self._mcp_bridge = bridge
                self.tenant_id = "t1"
                self.request_id = "req-stream"
                self._evidence_store = EvidenceStore()
                self._request_trace = SimpleNamespace(update_task_status_counts=lambda **k: None, record_degraded=lambda r: {"type": "degraded"}, metrics_event=lambda: {"type": "metrics"})

            def _ensure_mcp_bridge(self):
                return self._mcp_bridge

            def _set_tool_event_sink(self, sink) -> None:
                return None

            def _execute_task(self, state, task, *, emit_stream=False, step=None):
                raise MCPHitlApprovalRequired(
                    server_id="ops",
                    tool_name="write_ticket",
                    arguments_digest="sha256:abc",
                    task_id=task.id,
                )

        cfg = Configuration.from_env(
            overrides={
                "llm_provider": "vllm",
                "mcp_enabled": True,
                "active_mcp_tool_scopes": {
                    "allowed_servers": ["finarag"],
                    "allowed_tools": ["finarag:finarag_filings", "finarag:finarag_list_companies"],
                    "auto_query_tools": ["finarag:finarag_filings"],
                },
            },
            load_env_file=False,
        )
        dummy_bridge = SimpleNamespace(
            call_tool=lambda **kwargs: SimpleNamespace(
                ok=True,
                structured_payload=(
                    {"companies": [{"company_name": "AlphaCorp"}]}
                    if kwargs["tool_name"] == "finarag_list_companies"
                    else {"passages": []}
                ),
                normalized_evidence=(
                    [{"title": "Alpha 10-K", "url": "finarag://sha1#page=3", "snippet": "Revenue grew 30%."}]
                    if kwargs["tool_name"] == "finarag_filings"
                    else []
                ),
            )
        )
        harness = _DummyAgent(cfg, dummy_bridge)
        state = SummaryState(research_topic="AlphaCorp FY2025 growth", topic_cache_warmup_completed=True)
        task = TodoItem(id=1, title="AlphaCorp 营收", intent="查财报", query="AlphaCorp revenue")
        harness._dispatch_mcp_scope_prefetch(state, task, harness._evidence_store)
        ev = harness._evidence_store.list_task_evidence(1)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["source_id"], "T1-M1")

        # Verify StreamCoordinatorMixin propagates MCPHitlApprovalRequired instead of swallowing it
        with self.assertRaises(MCPHitlApprovalRequired):
            list(
                harness._execute_task_batch_stream(
                    state,
                    [task],
                    {1: {"step": 1, "token": "tok-1"}},
                )
            )
        self.assertEqual(task.status, "awaiting_approval")


if __name__ == "__main__":
    unittest.main()
