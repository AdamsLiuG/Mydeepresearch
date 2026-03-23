import json
import os
import sys
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
                "cache_misses": 1,
                "total_tokens": 120,
                "estimated_cost": 0.0012,
                "token_source": "estimated",
            },
            "aggregate_metrics": {
                "success_rate": 1.0,
                "counters": {"fallback_trigger_total": 0},
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
                }
            ],
            "step": 0,
        }
        yield {"type": "fallback_triggered", "reason": "planner_returned_no_tasks"}
        yield {"type": "degraded_response", "reason": "fallback_task_used"}
        yield {"type": "final_report", "report": "# 最终报告"}
        yield {"type": "done"}


class FailingAgent:
    def __init__(self, config, request_id=None):
        self.config = config
        self.request_id = request_id

    def run(self, topic: str):
        raise RuntimeError("sync failure")

    def run_stream(self, topic: str):
        raise RuntimeError("stream failure")


class InitFailingAgent:
    def __init__(self, config, request_id=None):
        raise RuntimeError("init failure")


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
                        "note_id": "note_1",
                        "note_path": "/tmp/note_1.md",
                    }
                ],
            },
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
                "final_report",
                "done",
            ],
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


if __name__ == "__main__":
    unittest.main()
