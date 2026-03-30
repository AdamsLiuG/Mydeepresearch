import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
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

_ORIGINAL_FROM_ENV = main.Configuration.from_env


@contextmanager
def isolated_configuration():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(
            main.Configuration,
            "from_env",
            side_effect=lambda overrides=None, load_env_file=True: _ORIGINAL_FROM_ENV(
                overrides=overrides,
                load_env_file=False,
            ),
        ):
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
                    "reflection_call_total": 1,
                    "reflection_replan_total": 1,
                    "reflection_skipped_total": 0,
                },
                "cache_exact_hit_total": 0,
                "cache_semantic_hit_total": 0,
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
    def setUp(self) -> None:
        metrics_registry.reset()

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
        self.assertIn("reflection_added_tasks", metrics_event["request_metrics"])
        self.assertIn("cache_exact_hit_total", metrics_event["aggregate_metrics"])
        self.assertIn("cache_semantic_hit_total", metrics_event["aggregate_metrics"])

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


if __name__ == "__main__":
    unittest.main()
