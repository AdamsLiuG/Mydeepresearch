import csv
import json
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    MCPToolAnnotationsContract,
    MCPTransportType,
    ScopedToolRegistry,
    _validate_mcp_endpoint_url,
    compute_arguments_digest,
    mint_scoped_tenant_token,
    verify_scoped_tenant_token,
)
from services.reporter import ReportingService
from services.skill_pack import SkillPackService, load_pack, validate_pack


class MCPBridgeAndFinaRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()

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
            try:
                # 1. Fail-Closed ACL checks
                res_a = bridge.call_tool(
                    server_id="finarag",
                    tool_name="finarag_list_companies",
                    arguments={},
                    tenant_id="tenant-a",
                    request_id="req-acl",
                )
                self.assertTrue(res_a.ok)
                self.assertEqual([c["company_name"] for c in res_a.structured_payload["companies"]], ["AlphaCorp"])

                denied_cross = bridge.call_tool(
                    server_id="finarag",
                    tool_name="finarag_filings",
                    arguments={"company_name": "BetaBank", "query": "capital ratio"},
                    tenant_id="tenant-a",
                    request_id="req-acl",
                )
                self.assertFalse(denied_cross.ok)
                self.assertIn("not authorized", denied_cross.error_message or "")

                denied_unknown = bridge.call_tool(
                    server_id="finarag",
                    tool_name="finarag_list_companies",
                    arguments={},
                    tenant_id="unknown-tenant",
                    request_id="req-acl",
                )
                self.assertTrue(denied_unknown.ok)
                self.assertEqual(denied_unknown.structured_payload["companies"], [])

                # 2. Anti-Tool-Bloat via SkillPack + ScopedToolRegistry
                svc = SkillPackService(cfg)
                fin_pack = load_pack(Path("backend/skills/research/financial-research/SKILL.md"))
                self.assertEqual(validate_pack(fin_pack, char_limit=1200, directory_name="financial-research"), [])
                neutral_sel = svc.select("今天杭州天气怎么样")
                self.assertEqual(neutral_sel.pack.mcp_tool_scopes, {})

                adapters = bridge.build_tool_adapters(
                    evidence_store=EvidenceStore(),
                    tenant_id_getter=lambda: "tenant-a",
                    request_id_getter=lambda: "req-acl",
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
        bridge_strict = MCPBridgeManager(cfg, servers=[srv_cfg])
        try:
            blocked = bridge_strict.call_tool(
                server_id="ops",
                tool_name="write_ticket",
                arguments={"title": "Audit"},
                tenant_id="t1",
                request_id="r1",
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
                res_after = bridge_hitl.call_tool(
                    server_id="ops",
                    tool_name="write_ticket",
                    arguments=args,
                    tenant_id="default",
                    request_id="req-hitl-1",
                    task_id=1,
                    approved_registry=updated_snap["approved_tool_calls"],
                )
                self.assertTrue(res_after.ok)
                self.assertEqual(res_after.structured_payload["ticket_id"], "TCK-42")
                # Single-use approval must be consumed after one call
                self.assertEqual(
                    updated_snap["approved_tool_calls"][f"ops:write_ticket:{digest}"]["status"],
                    "consumed",
                )
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
                "active_mcp_tool_scopes": {"auto_query_tools": ["finarag:finarag_filings"]},
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
