import sys
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from services.observability import LangfuseTraceAdapter


class _FakeLangfuseTrace:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.spans = []
        self.generations = []
        self.scores = []

    def span(self, **kwargs):
        self.spans.append(kwargs)

    def generation(self, **kwargs):
        self.generations.append(kwargs)

    def score(self, **kwargs):
        self.scores.append(kwargs)


class _FakeLangfuseClient:
    def __init__(self):
        self.traces = []
        self.flushed = False

    def trace(self, **kwargs):
        item = _FakeLangfuseTrace(kwargs)
        self.traces.append(item)
        return item

    def flush(self):
        self.flushed = True


class ObservabilityAdapterTests(unittest.TestCase):
    def test_langfuse_adapter_is_noop_when_disabled(self):
        cfg = Configuration.from_env(
            overrides={"langfuse_enabled": False},
            load_env_file=False,
        )
        fake_client = _FakeLangfuseClient()
        adapter = LangfuseTraceAdapter(cfg, client=fake_client)

        self.assertFalse(adapter.enabled)
        self.assertFalse(adapter.export_request_trace({"request_id": "req-1"}))
        self.assertEqual(fake_client.traces, [])

    def test_langfuse_adapter_exports_trace_spans_tokens_and_review_score(self):
        cfg = Configuration.from_env(
            overrides={"langfuse_enabled": True},
            load_env_file=False,
        )
        fake_client = _FakeLangfuseClient()
        adapter = LangfuseTraceAdapter(cfg, client=fake_client)

        snapshot = {
            "request_id": "req-obs-1",
            "topic": "MCP 协议",
            "search_api": "advanced",
            "provider": "vllm",
            "model": "Qwen/Qwen3-32B",
            "status": "success",
            "elapsed_ms": 1234.5,
            "degraded": False,
            "degraded_reasons": [],
            "skill_pack_id": "tech-selection",
            "cache_hits": 2,
            "cache_misses": 1,
            "prompt_tokens": 800,
            "completion_tokens": 300,
            "total_tokens": 1100,
            "token_source": "api_usage",
            "llm_calls": 4,
            "estimated_cost": 0.012,
            "review_overall_status": "passed",
            "stages": [
                {"stage": "planning", "scope": "request", "status": "success", "elapsed_ms": 200.0},
                {"stage": "report", "scope": "request", "status": "success", "elapsed_ms": 500.0},
            ],
        }

        exported = adapter.export_request_trace(snapshot)

        self.assertTrue(exported)
        self.assertTrue(fake_client.flushed)
        self.assertEqual(len(fake_client.traces), 1)
        trace = fake_client.traces[0]
        self.assertEqual(trace.kwargs["id"], "req-obs-1")
        self.assertEqual(len(trace.spans), 2)
        self.assertEqual(len(trace.generations), 1)
        self.assertEqual(trace.generations[0]["usage"]["total"], 1100)
        self.assertEqual(len(trace.scores), 1)
        self.assertEqual(trace.scores[0]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
