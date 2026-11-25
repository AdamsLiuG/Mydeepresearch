import json
import os
import sys
import tempfile
import types
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import time
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

agent_stub = types.ModuleType("agent")


class ImportSafeAgent:
    def __init__(self, config):
        self.config = config

    def run(self, topic: str):
        raise NotImplementedError

    def run_stream(self, topic: str):
        raise NotImplementedError

    def run_resume(self, request_id: str):
        raise NotImplementedError

    def run_stream_resume(self, request_id: str):
        raise NotImplementedError


agent_stub.DeepResearchAgent = ImportSafeAgent
sys.modules.setdefault("agent", agent_stub)

import main
from metrics import metrics_registry
from services.budget import BudgetGovernor, build_request_budget_for_config

_ORIGINAL_FROM_ENV = main.Configuration.from_env


@contextmanager
def isolated_configuration(*, patch_semantic_warmup: bool = True):
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(
            main.Configuration,
            "from_env",
            side_effect=lambda overrides=None, load_env_file=True: _ORIGINAL_FROM_ENV(
                overrides=overrides,
                load_env_file=False,
            ),
        ):
            if patch_semantic_warmup:
                with patch.object(main, "_warmup_semantic_cache_model", return_value=None):
                    yield
            else:
                yield


def _build_stub_result() -> SimpleNamespace:
    task = SimpleNamespace(
        id=1,
        title="任务1",
        intent="梳理主题背景",
        query="AI agent 最新进展",
        status="completed",
        summary="这是任务总结。",
        sources_summary="* 示例来源 : https://example.com",
        note_id="note_1",
        note_path="/tmp/note_1.md",
        origin="planned",
        round=1,
    )
    return SimpleNamespace(
        report_markdown="# 最终报告",
        running_summary="# 最终报告",
        todo_items=[task],
    )


class StubAgent:
    def __init__(self, config, request_id=None):
        self.config = config
        self.request_id = request_id

    def run(self, topic: str):
        return _build_stub_result()

    def run_resume(self, request_id: str):
        return _build_stub_result()

    def run_stream(self, topic: str):
        yield {"type": "status", "message": "初始化研究流程"}
        yield {"type": "stage_started", "stage": "planning", "scope": "request"}
        yield {
            "type": "stage_completed",
            "stage": "planning",
            "scope": "request",
            "status": "success",
            "elapsed_ms": 12,
        }
        yield {
            "type": "metrics_snapshot",
            "request_metrics": {
                "status": "in_progress",
                "elapsed_ms": 12,
                "cache_hits": 0,
                "cache_exact_hits": 0,
                "cache_semantic_hits": 0,
                "cache_approximate_hits": 0,
                "cache_approximate_dense_hits": 0,
                "cache_approximate_sparse_hits": 0,
                "cache_approximate_hybrid_hits": 0,
                "cache_misses": 1,
                "total_tokens": 120,
                "estimated_cost": 0.0012,
                "token_source": "estimated",
                "reflection_triggered": False,
                "reflection_reason": None,
                "reflection_gap_signals": [],
                "reflection_added_tasks": 0,
            },
            "aggregate_metrics": {
                "success_rate": 1.0,
                "counters": {
                    "fallback_trigger_total": 0,
                    "cache_exact_hit_total": 0,
                    "cache_semantic_hit_total": 0,
                    "cache_approximate_hit_total": 0,
                    "reflection_call_total": 1,
                    "reflection_replan_total": 1,
                    "reflection_skipped_total": 0,
                },
                "cache_exact_hit_total": 0,
                "cache_semantic_hit_total": 0,
                "cache_approximate_hit_total": 0,
            },
        }
        yield {
            "type": "todo_list",
            "tasks": [
                {
                    "id": 1,
                    "title": "任务1",
                    "intent": "梳理主题背景",
                    "query": topic,
                    "status": "pending",
                    "note_id": "note_1",
                    "note_path": "/tmp/note_1.md",
                    "origin": "planned",
                    "round": 1,
                }
            ],
            "step": 0,
        }
        yield {"type": "fallback_triggered", "reason": "planner_returned_no_tasks"}
        yield {"type": "degraded_response", "reason": "fallback_task_used"}
        yield {"type": "stage_started", "stage": "reflection", "scope": "request"}
        yield {
            "type": "stage_completed",
            "stage": "reflection",
            "scope": "request",
            "status": "success",
            "elapsed_ms": 8,
        }
        yield {"type": "status", "message": "发现覆盖缺口，补充 1 个任务继续研究。"}
        yield {
            "type": "todo_list",
            "tasks": [
                {
                    "id": 1,
                    "title": "任务1",
                    "intent": "梳理主题背景",
                    "query": topic,
                    "status": "completed",
                    "note_id": "note_1",
                    "note_path": "/tmp/note_1.md",
                    "origin": "planned",
                    "round": 1,
                },
                {
                    "id": 2,
                    "title": "补充任务",
                    "intent": "补充工程实践",
                    "query": f"{topic} 工程实践",
                    "status": "pending",
                    "note_id": "note_2",
                    "note_path": "/tmp/note_2.md",
                    "origin": "replanned",
                    "round": 2,
                },
            ],
            "step": 2,
        }
        yield {"type": "final_report", "report": "# 最终报告"}
        yield {"type": "done"}

    def run_stream_resume(self, request_id: str):
        yield from self.run_stream("resume topic")


class FailingAgent:
    def __init__(self, config, request_id=None):
        self.config = config
        self.request_id = request_id

    def run(self, topic: str):
        raise RuntimeError("sync failure")

    def run_stream(self, topic: str):
        raise RuntimeError("stream failure")

    def run_resume(self, request_id: str):
        raise RuntimeError("resume failure")

    def run_stream_resume(self, request_id: str):
        raise RuntimeError("resume stream failure")


class InitFailingAgent:
    def __init__(self, config, request_id=None):
        raise RuntimeError("init failure")


class CapturingAgent(StubAgent):
    last_config = None

    def __init__(self, config, request_id=None):
        type(self).last_config = config
        super().__init__(config, request_id=request_id)


class ApiTests(unittest.TestCase):
    def test_request_options_preserve_server_owned_web_policy(self):
        config = main.Configuration(
            v4_web_extractor_enabled=True, v4_web_extractor_tenants=["default"],
            v4_web_l2_adapter="reader_http", v4_web_reader_url="https://reader.example.org/read",
            v4_web_reader_token="private", v4_web_l2_origin_rules={"default": [{"origin": "https://example.org"}]},
        )
        request = main.ResearchRequest(topic="Study", search_api="searxng")
        effective = main._build_config(request, base_config=config)
        self.assertEqual(effective.v4_web_extractor_tenants, ["default"])
        self.assertEqual(effective.v4_web_l2_adapter, "reader_http")
        self.assertEqual(effective.v4_web_l2_origin_rules, config.v4_web_l2_origin_rules)
        self.assertEqual(effective.v4_web_reader_token, "private")

    def setUp(self) -> None:
        metrics_registry.reset()

    def test_v4_creation_routes_validate_depth_and_ignore_client_budget(self):
        class DepthAgent(StubAgent):
            def run(self, topic):
                result = super().run(topic)
                result.v4 = projection(self.config)
                return result

            def plan_only(self, topic):
                return {"topic": topic, "status": "awaiting_approval", "v4": projection(self.config)}

            def run_stream(self, topic):
                yield {"type": "status", "message": "started"}
                yield {"type": "final_report", "report": "# Report", "v4": projection(self.config)}
                yield {"type": "done"}

        def projection(config):
            if not config.v4_research_budget_enabled:
                return None
            budget = build_request_budget_for_config(config, "default")
            return {"schema_version": 1, "budget": asdict(BudgetGovernor(budget).snapshot())}

        def submit(_topic, *, request_id, config, **_kwargs):
            return main.WorkflowRunHandle(
                request_id, "run", "inline", metadata={"v4": projection(config)},
            )

        routes = ("/research", "/research/plan", "/research/stream", "/research/async")
        for enabled in (False, True):
            with isolated_configuration(), patch.object(main, "DeepResearchAgent", DepthAgent):
                config = main.Configuration(
                    v4_research_budget_enabled=enabled,
                    v4_default_research_depth="deep",
                )
                app = main.create_app(base_config=config)
                with TestClient(app) as client, patch.object(app.state.workflow_runner, "submit", side_effect=submit):
                    for route in routes:
                        for choice in ({}, {"research_depth": None}, *(
                            {"research_depth": value} for value in ("quick", "standard", "deep", "unknown", {}, 1, False)
                        )):
                            depth = choice.get("research_depth")
                            with self.subTest(enabled=enabled, route=route, depth=depth):
                                response = client.post(route, json={
                                    "topic": "research", "budget": {"limits": {"tasks": 999}}, **choice,
                                })
                                if enabled and depth not in (None, "quick", "standard", "deep"):
                                    self.assertEqual(response.status_code, 422)
                                    self.assertNotIn("text/event-stream", response.headers["content-type"])
                                    continue
                                self.assertEqual(response.status_code, 202 if route.endswith("/async") else 200)
                                if route.endswith("/stream"):
                                    events = [json.loads(line[6:]) for line in response.text.splitlines()
                                              if line.startswith("data: ")]
                                    self.assertEqual([event["type"] for event in events], ["status", "final_report", "done"])
                                    payload = events[1]
                                else:
                                    payload = response.json()
                                    if route.endswith("/async"):
                                        self.assertEqual(payload["events_url"], payload["stream_url"])
                                        payload = payload["metadata"]
                                public = payload.get("v4")
                                if enabled:
                                    self.assertEqual(public["budget"]["limit"]["depth"], depth or "deep")
                                    self.assertLess(public["budget"]["limit"]["limits"]["tasks"], 999)
                                else:
                                    self.assertIsNone(public)

    def test_v4_request_detail_and_plan_edit_use_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir, isolated_configuration():
            config = main.Configuration(
                request_state_enabled=True, request_state_dir=temp_dir,
                v4_research_budget_enabled=True, v4_default_research_depth="deep",
            )
            budget = build_request_budget_for_config(config, "default", depth="quick")
            snapshot = asdict(BudgetGovernor(budget).snapshot())
            snapshot["internal_path"] = "/private/snapshot"
            raw = {
                "schema_version": 1,
                "budget": {"limit": budget.to_dict(), "snapshot": snapshot, "ledger": {"secret": "private"}},
                "replan": {"dag_revision": 1, "decision_id": "d1", "pruned_task_ids": [3], "plan_locked": False},
                "outbox": [{"value": "private"}],
            }
            with TestClient(main.create_app(base_config=config)) as client:
                store = client.app.state.request_state_store
                store.save("draft", {
                    "topic": "draft", "phase": "awaiting_approval", "status": "awaiting_approval",
                    "plan_revision": 1, "v4": raw,
                })
                endpoint = "/requests/draft"
                detail = client.get(endpoint)
                self.assertEqual(detail.status_code, 200)
                projection = detail.json()["v4"]
                self.assertEqual(projection["budget"]["limit"]["depth"], "quick")
                self.assertEqual(projection["replan"], {"dag_revision": 1, "decision_id": "d1", "pruned_task_ids": [3]})
                self.assertNotIn("private", json.dumps(projection))

                foreign = store.save("foreign", {"v4": raw}, tenant_id="foreign-tenant")
                self.assertEqual(foreign["tenant_id"], "foreign-tenant")
                self.assertEqual(client.get("/requests/foreign").status_code, 404)

                too_many = [{"title": f"task {i}"} for i in range(budget.limits["tasks"] + 1)]
                update = client.put(endpoint + "/plan", json={"revision": 1, "tasks": too_many})
                self.assertEqual(update.status_code, 400)
                invalid = [{"id": 1, "title": "first", "depends_on": [2]},
                           {"id": 2, "title": "second", "depends_on": [1]}]
                update = client.put(endpoint + "/plan", json={"revision": 1, "tasks": invalid})
                self.assertEqual(update.status_code, 400)
                edit = {"revision": 1, "tasks": [{"id": 7, "title": "allowed"}]}
                updated = client.put(endpoint + "/plan", json=edit)
                self.assertEqual(updated.status_code, 200)
                self.assertEqual(updated.json()["todo_items"][0]["id"], 7)
                self.assertEqual(updated.json()["v4"], projection)
                self.assertEqual(store.load("draft")["v4"], raw)
                self.assertEqual(client.put(endpoint + "/plan", json=edit).status_code, 409)

                saved = store.load("draft")
                store.save("draft", {**saved, "phase": "task_execution", "status": "in_progress"},
                           expected_version=saved["state_version"])
                self.assertEqual(client.put(endpoint + "/plan", json={**edit, "revision": 2}).status_code, 409)

    def test_v4_follow_waits_for_committed_events_and_running_terminal_delivery(self):
        with tempfile.TemporaryDirectory() as temp_dir, isolated_configuration():
            config = main.Configuration(request_state_enabled=True, request_state_dir=temp_dir)
            with TestClient(main.create_app(base_config=config)) as client:
                store = client.app.state.request_state_store
                bus = client.app.state.event_bus
                request_id = "finished"
                snapshot = store.save(request_id, {
                    "status": "completed", "v4": {"schema_version": 1, "pending_events": [{"type": "budget_updated"}]},
                })
                cursor = bus.publish("default", request_id, {"type": "status"})
                tail = bus.tail_since

                def exercise_tail(*args, is_terminal_fn: Callable[[], bool], **kwargs):
                    self.assertFalse(is_terminal_fn())
                    leased = {
                        **snapshot,
                        "workflow_lease": {"owner": "worker", "expires_at": time() + 60},
                    }
                    leased.pop("v4")
                    saved = store.save(request_id, leased, expected_version=snapshot["state_version"])
                    self.assertFalse(is_terminal_fn())
                    bus.publish("default", request_id, {"type": "budget_updated", "event_key": "budget-finished"})
                    bus.publish("default", request_id, {"type": "final_report", "report": "Ready"})
                    bus.publish("default", request_id, {"type": "done"})
                    saved["workflow_lease"]["expires_at"] = time() - 60
                    store.save(request_id, saved, expected_version=saved["state_version"])
                    self.assertTrue(is_terminal_fn())
                    yield from tail(*args, is_terminal_fn=is_terminal_fn, **kwargs)

                with patch.object(bus, "tail_since", side_effect=exercise_tail):
                    response = client.get(f"/requests/{request_id}/events?follow=1", headers={"Last-Event-ID": cursor})
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertEqual([event["type"] for event in events], ["budget_updated", "final_report", "done"])

    def test_v4_report_replay_matches_detail_and_keeps_committed_order(self):
        with tempfile.TemporaryDirectory() as temp_dir, isolated_configuration():
            config = main.Configuration(request_state_enabled=True, request_state_dir=temp_dir)
            with TestClient(main.create_app(base_config=config)) as client:
                store = client.app.state.request_state_store
                bus = client.app.state.event_bus
                request_id = "versioned-report"
                endpoint = f"/requests/{request_id}"
                section = {"section_id": "s1", "title": "Findings", "order": 1}
                report = {
                    "outline": {"outline_version": 1, "sections": [section]},
                    "sections_by_id": {"s1": {"status": "completed", "claim_ids": ["C-001"], "draft": "private"}},
                    "critique": {"status": "ok", "diagnostics": "private"},
                    "degradation_reason": "private",
                    "final": {"report_version": 1, "markdown": "private"},
                }
                draft = store.save(request_id, {
                    "status": "in_progress", "v4": {"schema_version": 1, "report": report},
                })
                public_report = {
                    "outline_version": 1,
                    "critic_status": "ok",
                    "sections": [{**section, "status": "completed"}],
                }
                self.assertEqual(client.get(endpoint).json()["v4"]["report"], public_report)

                markdown = "# Report\n\n## Findings\n\nCommitted text.\n"
                workbench = {
                    "section_bindings": [{"section_id": "s1", "title": "Findings", "paragraph_index": 0}],
                    "claim_bindings": [],
                }
                budget = {"type": "budget_updated", "event_key": "versioned-report:budget:completed", "schema_version": 1}
                frames = [
                    {"type": kind, "event_key": key, "schema_version": 1, **fields}
                    for kind, key, fields in (
                        ("report_chunk", "report-v1-chunk-1", {"index": 1, "report_version": 1}),
                        ("report_chunk", "report-v1-chunk-2", {"index": 2, "report_version": 1}),
                        ("final_report", "report-v1-final", {"report_version": 1}),
                        ("done", "report-v1-done", {}),
                    )
                ]
                store.save(request_id, {
                    **draft, "status": "success", "report_markdown": markdown, "workbench": workbench,
                    "v4": {
                        "schema_version": 1,
                        "report": {**report, "final": {"report_version": 1, "published": True}},
                        "pending_events": [budget, *frames],
                    },
                }, expected_version=draft["state_version"])
                cursor = bus.publish("default", request_id, budget)
                response = client.get(endpoint + "/events?follow=1", headers={"Last-Event-ID": cursor})
                self.assertEqual(response.status_code, 200)
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertEqual([event["type"] for event in events], ["report_chunk", "report_chunk", "final_report", "done"])
                self.assertEqual("".join(event["content"] for event in events[:2]), events[-2]["report"])
                self.assertEqual(events[-2]["report"], markdown)
                self.assertEqual(events[-2]["workbench"], workbench)
                self.assertEqual(events[-2]["v4"], client.get(endpoint).json()["v4"])
                self.assertEqual(events[-2]["v4"]["report"], {
                    **public_report, "report_version": 1,
                    "sections": [{**public_report["sections"][0], "claim_ids": ["C-001"]}],
                })
                self.assertEqual(store.load(request_id)["v4"]["pending_events"], [])

    def test_dev_reload_only_watches_backend_src(self):
        with isolated_configuration():
            cfg = main.Configuration.from_env(load_env_file=False)

        reload_kwargs = main._build_dev_reload_kwargs(cfg)
        expected_src_dir = str((main.backend_root() / "src").resolve(strict=False))

        self.assertTrue(reload_kwargs["reload"])
        self.assertEqual(reload_kwargs["reload_dirs"], [expected_src_dir])
        self.assertNotIn(cfg.notes_workspace, reload_kwargs["reload_dirs"])
        self.assertNotIn(cfg.request_state_dir, reload_kwargs["reload_dirs"])
        self.assertNotIn(cfg.search_cache_dir, reload_kwargs["reload_dirs"])
        self.assertNotIn(cfg.note_memory_dir, reload_kwargs["reload_dirs"])

    def test_healthz_returns_ok(self):
        with isolated_configuration():
            with TestClient(main.create_app()) as client:
                response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertIn("X-Request-ID", response.headers)

    def test_research_returns_existing_response_shape(self):
        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", StubAgent):
                with TestClient(main.create_app()) as client:
                    response = client.post("/research", json={"topic": "AI agent"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-ID", response.headers)
        self.assertEqual(
            response.json(),
            {
                "report_markdown": "# 最终报告",
                "todo_items": [
                    {
                        "id": 1,
                        "title": "任务1",
                        "intent": "梳理主题背景",
                        "query": "AI agent 最新进展",
                        "status": "completed",
                        "summary": "这是任务总结。",
                        "sources_summary": "* 示例来源 : https://example.com",
                        "notices": [],
                        "evidence_items": [],
                        "claims": [],
                        "review_issues": [],
                        "review_status": "pending",
                        "note_id": "note_1",
                        "note_path": "/tmp/note_1.md",
                        "origin": "planned",
                        "round": 1,
                    }
                ],
            },
        )

    def test_research_accepts_semanticscholar_override(self):
        CapturingAgent.last_config = None

        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", CapturingAgent):
                with TestClient(main.create_app()) as client:
                    response = client.post(
                        "/research",
                        json={"topic": "AI agent", "search_api": "semanticscholar"},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(CapturingAgent.last_config)
        self.assertEqual(
            CapturingAgent.last_config.search_api,
            main.SearchAPI.SEMANTICSCHOLAR,
        )

    def test_research_returns_400_for_invalid_config(self):
        with isolated_configuration():
            with patch.object(main, "_build_config", side_effect=ValueError("bad config")):
                with TestClient(main.create_app()) as client:
                    response = client.post("/research", json={"topic": "AI agent"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "bad config"})

    def test_research_returns_500_with_request_id(self):
        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", FailingAgent):
                with TestClient(main.create_app()) as client:
                    response = client.post("/research", json={"topic": "AI agent"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("request_id=", response.json()["detail"])
        self.assertIn("X-Request-ID", response.headers)

    def test_stream_endpoint_keeps_sse_contract(self):
        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", StubAgent):
                with TestClient(main.create_app()) as client:
                    with client.stream(
                        "POST",
                        "/research/stream",
                        json={"topic": "AI agent"},
                    ) as response:
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(
                            response.headers["content-type"],
                            "text/event-stream; charset=utf-8",
                        )
                        self.assertIn("X-Request-ID", response.headers)

                        events = []
                        for line in response.iter_lines():
                            if isinstance(line, bytes):
                                line = line.decode("utf-8")
                            if line.startswith("data: "):
                                events.append(json.loads(line[6:]))

        self.assertEqual(
            [event["type"] for event in events],
            [
                "status",
                "stage_started",
                "stage_completed",
                "metrics_snapshot",
                "todo_list",
                "fallback_triggered",
                "degraded_response",
                "stage_started",
                "stage_completed",
                "status",
                "todo_list",
                "final_report",
                "done",
            ],
        )
        metrics_event = next(event for event in events if event["type"] == "metrics_snapshot")
        self.assertIn("cache_exact_hits", metrics_event["request_metrics"])
        self.assertIn("cache_semantic_hits", metrics_event["request_metrics"])
        self.assertIn("cache_approximate_hits", metrics_event["request_metrics"])
        self.assertIn("reflection_added_tasks", metrics_event["request_metrics"])
        self.assertIn("cache_exact_hit_total", metrics_event["aggregate_metrics"])
        self.assertIn("cache_semantic_hit_total", metrics_event["aggregate_metrics"])
        self.assertIn("cache_approximate_hit_total", metrics_event["aggregate_metrics"])

    def test_stream_endpoint_allows_loopback_origin_pair(self):
        base_config = main.Configuration.from_env(
            overrides={
                "cors_origins": "http://localhost:5174",
            },
            load_env_file=False,
        )

        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", StubAgent):
                with TestClient(main.create_app(base_config=base_config)) as client:
                    response = client.options(
                        "/research/stream",
                        headers={
                            "Origin": "http://127.0.0.1:5174",
                            "Access-Control-Request-Method": "POST",
                        },
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://127.0.0.1:5174",
        )

    def test_metrics_json_endpoint_returns_snapshot(self):
        with isolated_configuration():
            with TestClient(main.create_app()) as client:
                response = client.get("/metrics/json")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("counters", payload)
        self.assertIn("latencies_ms", payload)
        self.assertIn("success_rate", payload)
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")
        self.assertEqual(response.headers.get("referrer-policy"), "no-referrer")

    def test_stream_endpoint_returns_500_when_agent_init_fails(self):
        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", InitFailingAgent):
                with TestClient(main.create_app()) as client:
                    response = client.post("/research/stream", json={"topic": "AI agent"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("request_id=", response.json()["detail"])
        self.assertIn("X-Request-ID", response.headers)

    def test_benchmark_stub_mode_returns_deterministic_payload(self):
        base_config = main.Configuration.from_env(
            overrides={
                "benchmark_stub_enabled": True,
                "benchmark_profile": "stub",
            },
            load_env_file=False,
        )

        with isolated_configuration():
            with TestClient(main.create_app(base_config=base_config)) as client:
                response = client.post("/research", json={"topic": "工程 benchmark 验证"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Deterministic benchmark summary.", json.dumps(payload, ensure_ascii=False))
        self.assertIn("Benchmark Stub Task", json.dumps(payload, ensure_ascii=False))

    def test_stream_endpoint_emits_error_event_when_runtime_fails(self):
        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", FailingAgent):
                with TestClient(main.create_app()) as client:
                    with client.stream(
                        "POST",
                        "/research/stream",
                        json={"topic": "AI agent"},
                    ) as response:
                        events = []
                        for line in response.iter_lines():
                            if isinstance(line, bytes):
                                line = line.decode("utf-8")
                            if line.startswith("data: "):
                                events.append(json.loads(line[6:]))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("request_id=", events[0]["detail"])

    def test_request_state_endpoints_return_saved_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_config = main.Configuration.from_env(
                overrides={
                    "request_state_enabled": True,
                    "request_state_dir": temp_dir,
                },
                load_env_file=False,
            )
            with isolated_configuration():
                with TestClient(main.create_app(base_config=base_config)) as client:
                    client.app.state.request_state_store.save(
                        "req-persisted",
                        {
                            "topic": "persisted topic",
                            "status": "in_progress",
                            "phase": "review",
                            "todo_items": [{"id": 1, "title": "任务1"}],
                            "report_markdown": "",
                        },
                    )
                    list_response = client.get("/requests")
                    detail_response = client.get("/requests/req-persisted")

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(list_response.json()["items"][0]["request_id"], "req-persisted")
        self.assertEqual(detail_response.json()["topic"], "persisted topic")

    def test_resume_endpoint_uses_resume_agent_method(self):
        with isolated_configuration():
            with patch.object(main, "DeepResearchAgent", StubAgent):
                with TestClient(main.create_app()) as client:
                    response = client.post("/requests/req-123/resume", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report_markdown"], "# 最终报告")

    def test_startup_warms_semantic_cache_model_when_enabled(self):
        base_config = main.Configuration.from_env(
            overrides={
                "semantic_cache_enabled": True,
                "semantic_cache_warmup_enabled": True,
                "semantic_cache_embedding_model": "dummy-minilm",
            },
            load_env_file=False,
        )

        with isolated_configuration(patch_semantic_warmup=False):
            with patch("services.embeddings.embeddings_available", return_value=True):
                with patch("services.embeddings.load_sentence_transformer", return_value=object()) as mock_load:
                    with TestClient(main.create_app(base_config=base_config)):
                        pass

        mock_load.assert_called_once_with("dummy-minilm")

    def test_startup_skips_semantic_cache_warmup_when_disabled(self):
        base_config = main.Configuration.from_env(
            overrides={
                "semantic_cache_enabled": True,
                "semantic_cache_warmup_enabled": False,
                "semantic_cache_embedding_model": "dummy-minilm",
            },
            load_env_file=False,
        )

        with isolated_configuration(patch_semantic_warmup=False):
            with patch("services.embeddings.embeddings_available", return_value=True):
                with patch("services.embeddings.load_sentence_transformer") as mock_load:
                    with TestClient(main.create_app(base_config=base_config)):
                        pass

        mock_load.assert_not_called()

    def test_startup_continues_when_semantic_cache_warmup_fails(self):
        base_config = main.Configuration.from_env(
            overrides={
                "semantic_cache_enabled": True,
                "semantic_cache_warmup_enabled": True,
                "semantic_cache_embedding_model": "dummy-minilm",
            },
            load_env_file=False,
        )

        with isolated_configuration(patch_semantic_warmup=False):
            with patch("services.embeddings.embeddings_available", return_value=True):
                with patch(
                    "services.embeddings.load_sentence_transformer",
                    side_effect=RuntimeError("warmup failure"),
                ):
                    with TestClient(main.create_app(base_config=base_config)) as client:
                        response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_list_requests_returns_slim_payload_by_default_and_supports_include_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_config = main.Configuration.from_env(
                overrides={
                    "request_state_enabled": True,
                    "request_state_dir": temp_dir,
                },
                load_env_file=False,
            )
            with isolated_configuration():
                with TestClient(main.create_app(base_config=base_config)) as client:
                    client.app.state.request_state_store.save(
                        "req-slim",
                        {
                            "topic": "slim test",
                            "status": "success",
                            "phase": "completed",
                            "todo_items": [{"id": 1, "title": "t1", "evidence_items": [{"snippet": "x" * 500}]}],
                            "report_markdown": "# Full Report\n" + ("body " * 200),
                        },
                    )
                    slim_resp = client.get("/requests")
                    full_resp = client.get("/requests?include_content=true")

        self.assertEqual(slim_resp.status_code, 200)
        slim_item = slim_resp.json()["items"][0]
        self.assertNotIn("report_markdown", slim_item)
        self.assertNotIn("todo_items", slim_item)
        self.assertTrue(slim_item["can_view_content"])
        self.assertTrue(slim_item["has_report"])
        self.assertEqual(slim_item["task_count"], 1)

        full_item = full_resp.json()["items"][0]
        self.assertIn("report_markdown", full_item)
        self.assertIn("todo_items", full_item)

    def test_stream_research_emits_keepalive_heartbeat_and_passes_cancel_event(self):
        import time as _time

        captured_cancel = {}

        class SlowStubAgent:
            def __init__(self, config, request_id=None):
                self.config = config
                self.request_id = request_id

            def run_stream(self, topic: str, cancel_event=None):
                captured_cancel["event"] = cancel_event
                _time.sleep(0.18)
                yield {"type": "status", "message": f"working:{topic}"}
                yield {"type": "done"}

        base_config = main.Configuration.from_env(
            overrides={
                "request_state_enabled": False,
                "sse_heartbeat_interval_seconds": 0.06,
            },
            load_env_file=False,
        )
        with isolated_configuration():
            with patch.dict(os.environ, {"SSE_HEARTBEAT_INTERVAL_SECONDS": "0.06"}):
                with patch.object(main, "DeepResearchAgent", SlowStubAgent):
                    with TestClient(main.create_app(base_config=base_config)) as client:
                        response = client.post("/research/stream", json={"topic": "heartbeat"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-accel-buffering"), "no")
        self.assertIn(": keepalive", response.text)
        self.assertIn("working:heartbeat", response.text)
        self.assertIsNotNone(captured_cancel.get("event"))
        self.assertTrue(captured_cancel["event"].is_set())


if __name__ == "__main__":
    unittest.main()
