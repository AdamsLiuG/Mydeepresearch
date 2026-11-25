"""Tests for v4-05 cascade model router, OTel GenAI tracing, W3C traceparent, and Prometheus SLI."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration  # noqa: E402
from main import create_app  # noqa: E402
from metrics import MetricsRegistry, RequestTrace  # noqa: E402
from services.mcp_bridge import (  # noqa: E402
    MCPBridgeManager,
    MCPServerConfig,
    MCPTransportType,
)
from services.model_router import CascadeModelRouter, RouteFeatures  # noqa: E402
from services.observability import (  # noqa: E402
    OTelTraceAdapter,
    child_traceparent,
    new_traceparent,
    parse_traceparent,
)
from services.sandbox_executor import SandboxExecutionRequest  # noqa: E402
from services.workflow_runner import (  # noqa: E402
    InlineWorkflowRunner,
    LocalDurableWorkflowRunner,
)


def _disabled_config(**overrides) -> Configuration:
    return Configuration.from_env(
        overrides={
            "request_state_enabled": False,
            "enable_notes": False,
            "llm_model_id": "base-std-model",
            "fast_llm_model_id": "fast-mini-model",
            **overrides,
        },
        load_env_file=False,
    )


def test_cascade_model_router_safe_off_and_modes():
    cfg_off = _disabled_config()
    router_off = CascadeModelRouter(cfg_off)
    assert not router_off.enabled
    assert router_off.mode == "off"
    dec_off = router_off.select_tier(RouteFeatures(stage="task_react", topic="hello world"))
    assert dec_off.tier == "standard"
    assert dec_off.shadow_only is True
    assert dec_off.reason == "router_disabled"
    assert router_off.evaluate_escalation(dec_off, parse_ok=False) is None

    cfg_shadow = _disabled_config(
        v4_model_router_enabled=True,
        v4_model_router_mode="shadow",
        v4_model_router_fast_model="qwen-fast",
        v4_model_router_standard_model="qwen-std",
        v4_model_router_frontier_model="gpt-frontier",
    )
    router_shadow = CascadeModelRouter(cfg_shadow)
    assert router_shadow.enabled and router_shadow.mode == "shadow"
    low = router_shadow.select_tier(RouteFeatures(stage="task_react", topic="simple overview"))
    assert low.tier == "fast" and low.model_name == "qwen-fast" and low.shadow_only is True

    cfg_active = _disabled_config(
        v4_model_router_enabled=True,
        v4_model_router_mode="active",
        v4_model_router_fast_model="qwen-fast",
        v4_model_router_standard_model="qwen-std",
        v4_model_router_frontier_model="gpt-frontier",
        v4_model_router_max_escalations_per_stage=1,
        v4_model_router_confidence_threshold=0.75,
    )
    router = CascadeModelRouter(cfg_active)
    assert router.mode == "active"

    simple = router.select_tier(RouteFeatures(stage="task_react", topic="brief look"))
    assert simple.tier == "fast" and simple.shadow_only is False and simple.model_name == "qwen-fast"

    hard = router.select_tier(
        RouteFeatures(
            stage="report",
            topic="Compare Nvidia vs AMD financial revenue growth margin and regulatory risks across 2023-2025 filings",
            task_count=6,
            conflict_count=2,
            domain="finance_filing",
        )
    )
    assert hard.tier == "frontier" and hard.model_name == "gpt-frontier"

    # Budget pressure caps frontier back down to avoid exhaustion.
    pressured = router.select_tier(
        RouteFeatures(
            stage="report",
            topic="Compare complex financial filings",
            conflict_count=2,
            domain="finance_filing",
            budget_pressure=0.85,
        )
    )
    assert pressured.tier == "standard" and pressured.reason == "budget_pressure_cap"

    very_tight = router.select_tier(
        RouteFeatures(stage="report", topic="Complex comparison", budget_pressure=0.95)
    )
    assert very_tight.tier == "fast" and very_tight.reason == "budget_pressure_fast"

    # Bounded escalation (max 1).
    esc1 = router.evaluate_escalation(simple, parse_ok=False, escalation_count=0, budget_allows=True)
    assert esc1 is not None and esc1.tier == "standard" and esc1.escalated_from == "fast"
    assert esc1.reason == "parse_failure_escalation"
    assert router.evaluate_escalation(esc1, parse_ok=False, escalation_count=1, budget_allows=True) is None
    assert router.evaluate_escalation(simple, parse_ok=False, escalation_count=0, budget_allows=False) is None


def test_w3c_traceparent_validation_and_otel_redaction():
    assert parse_traceparent(None) is None
    assert parse_traceparent("invalid-header") is None
    assert parse_traceparent("00-" + ("0" * 32) + "-" + ("1" * 16) + "-01") is None
    assert parse_traceparent("00-" + ("a" * 32) + "-" + ("0" * 16) + "-01") is None

    tp = new_traceparent()
    parsed = parse_traceparent(tp)
    assert parsed is not None and len(parsed[0]) == 32 and len(parsed[1]) == 16
    child = child_traceparent(tp)
    assert child != tp
    assert parse_traceparent(child)[0] == parsed[0]

    # Malformed parent triggers fresh root instead of erroring.
    fresh = child_traceparent("not-a-traceparent")
    assert parse_traceparent(fresh) is not None

    cfg_off = _disabled_config(v4_otel_enabled=False)
    otel_off = OTelTraceAdapter(cfg_off)
    with otel_off.start_span("noop", traceparent=tp) as ctx:
        assert ctx == {}
    assert otel_off.recorded_spans() == []

    local_metrics = MetricsRegistry()
    cfg_on = _disabled_config(v4_otel_enabled=True, v4_otel_service_name="dr-test")
    otel = OTelTraceAdapter(cfg_on, metrics=local_metrics)
    with otel.start_span(
        "gen_ai.chat",
        traceparent=tp,
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-std",
            "gen_ai.usage.input_tokens": 120,
            "gen_ai.usage.output_tokens": 45,
            "raw_prompt_text": "DO_NOT_LEAK_PROMPT",
            "authorization_header": "Bearer sk-secret-1234",
            "user_api_key": "secret-key",
            "deepresearch.tenant_id": "tenant-alpha",
        },
    ) as ctx:
        assert ctx["trace_id"] == parsed[0]
        assert "raw_prompt_text" not in ctx["attributes"]
        assert "authorization_header" not in ctx["attributes"]
        assert "user_api_key" not in ctx["attributes"]

    ok = otel.export_request_trace(
        {
            "request_id": "req-otel-1",
            "tenant_id": "tenant-alpha",
            "provider": "openai",
            "model": "gpt-std",
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "status": "success",
            "elapsed_ms": 88.5,
        },
        traceparent=tp,
    )
    assert ok is True
    spans = otel.recorded_spans()
    assert len(spans) == 2
    assert spans[-1].attributes["gen_ai.usage.input_tokens"] == 120
    assert spans[-1].attributes["deepresearch.tenant_id"] == "tenant-alpha"
    prom_text = local_metrics.render_prometheus()
    assert "deepresearch_v4_otel_spans_total" in prom_text


def test_traceparent_propagation_mcp_sandbox_workflow(tmp_path):
    tp = new_traceparent()
    cfg = _disabled_config(
        request_state_enabled=True,
        request_state_dir=str(tmp_path / "req"),
        mcp_enabled=True,
        mcp_tenant_token_secret="unit-test-secret",
    )
    captured_headers: list[dict[str, str]] = []

    def in_proc_mcp(payload, headers):
        captured_headers.append(headers)
        if payload["method"] == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "tools": [
                        {
                            "name": "ping",
                            "description": "ping",
                            "inputSchema": {"type": "object", "properties": {}},
                            "annotations": {"readOnlyHint": True, "idempotentHint": True},
                        }
                    ]
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {"content": [{"type": "text", "text": "pong"}]},
        }

    bridge = MCPBridgeManager(
        cfg,
        servers=[
            MCPServerConfig(
                server_id="mock",
                enabled=True,
                transport=MCPTransportType.STDIO,
                tenant_scoped=True,
                in_process_handler=in_proc_mcp,
            )
        ],
    )
    try:
        res = bridge.call_tool(
            server_id="mock",
            tool_name="ping",
            arguments={},
            active_scopes={
                "allowed_servers": ["mock"],
                "allowed_tools": ["mock:ping"],
            },
            traceparent=tp,
        )
        assert res.ok is True
        assert captured_headers[-1].get("traceparent") == tp
    finally:
        bridge.close()

    sb_req = SandboxExecutionRequest(
        task_id=1, code="result = 1 + 1", derived_from_source_ids=["T1-S1"], traceparent=tp,
    )
    assert sb_req.traceparent == tp

    local_trace = RequestTrace(
        request_id="req-trace", topic="t", search_api="stub",
        provider="openai", model="m", pricing_catalog={},
    )
    local_trace.traceparent = tp
    local_trace.record_model_route({"stage": "planning", "tier": "standard", "shadow_only": False})
    snap = local_trace.snapshot()
    assert snap["traceparent"] == tp
    assert snap["model_routes"][0]["stage"] == "planning"

    from services.event_bus import InMemoryStreamEventBus
    from services.request_state import RequestStateStore

    store = RequestStateStore(cfg.request_state_dir)
    runner = LocalDurableWorkflowRunner(
        agent_factory=lambda config, request_id, tenant_id, traceparent=None: SimpleNamespace(
            run_stream=lambda topic: iter([]),
        ),
        event_bus=InMemoryStreamEventBus(),
        state_store=store,
    )
    try:
        handle = runner.submit(
            "Topic", request_id="req-wf-tp", tenant_id="default", config=cfg, traceparent=tp,
        )
        assert handle.request_id == "req-wf-tp"
        persisted = store.load("req-wf-tp", tenant_id="default")
        assert persisted is not None and persisted.get("traceparent") == tp
    finally:
        runner.shutdown(timeout=1.0)

    inline = InlineWorkflowRunner()
    assert inline.submit("Topic", request_id="req-in", tenant_id="default", config=cfg, traceparent=tp).status == "ready"


def test_prometheus_endpoint_gating_and_traceparent_headers(tmp_path):
    cfg_disabled = _disabled_config(
        request_state_enabled=False,
        v4_prometheus_metrics_enabled=False,
        v4_otel_enabled=False,
    )
    app_disabled = create_app(base_config=cfg_disabled)
    with TestClient(app_disabled) as client:
        assert client.get("/metrics/prometheus").status_code == 404
        r = client.get("/healthz")
        assert r.status_code == 200
        assert "traceparent" not in r.headers

    cfg_enabled = _disabled_config(
        request_state_enabled=True,
        request_state_dir=str(tmp_path / "req"),
        v4_prometheus_metrics_enabled=True,
        v4_otel_enabled=True,
        v4_model_router_enabled=True,
        v4_model_router_mode="active",
        benchmark_stub_enabled=True,
        llm_provider="openai",
        llm_model_id="stub-model",
    )
    captured_traceparents: list[str | None] = []

    def fake_agent_factory(*, config, request_id, tenant_id="default", traceparent=None):
        captured_traceparents.append(traceparent)
        return SimpleNamespace(
            tenant_id=tenant_id,
            run=lambda topic: SimpleNamespace(
                report_markdown="# Report [T1-S1]",
                running_summary="",
                todo_items=[],
                report_charts=[],
                evidence_conflicts=[],
                claim_verification_summary={},
                v4=None,
            ),
        )

    app_enabled = create_app(base_config=cfg_enabled, agent_factory=fake_agent_factory)
    incoming_tp = new_traceparent()
    with TestClient(app_enabled) as client:
        res_ok = client.post(
            "/research",
            json={"topic": "Compare vLLM vs SGLang"},
            headers={"traceparent": incoming_tp},
        )
        assert res_ok.status_code == 200
        assert res_ok.headers.get("traceparent") == incoming_tp
        assert captured_traceparents[-1] == incoming_tp

        # Malformed traceparent gets sanitized and replaced safely.
        res_bad_tp = client.get("/healthz", headers={"traceparent": "malformed\ninjection"})
        assert res_bad_tp.status_code == 200
        echoed = res_bad_tp.headers.get("traceparent")
        assert echoed is not None and parse_traceparent(echoed) is not None

        prom_res = client.get("/metrics/prometheus")
        assert prom_res.status_code == 200
        assert prom_res.headers["content-type"].startswith("text/plain; version=0.0.4")
        body = prom_res.text
        assert "# HELP deepresearch_requests_total" in body
        assert "# TYPE deepresearch_request_status_total counter" in body
        assert "deepresearch_llm_tokens_total" in body
