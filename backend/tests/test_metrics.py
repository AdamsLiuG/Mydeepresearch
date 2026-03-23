import sys
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from metrics import RequestTrace, metrics_registry


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()

    def test_metrics_accumulate_and_fallback_updates_partial_success(self):
        observer = RequestTrace(
            request_id="req-1",
            topic="AI agent",
            search_api="duckduckgo",
            provider="custom",
            model="demo-model",
            pricing_catalog={
                "custom": {
                    "demo-model": {
                        "prompt_per_1k_tokens": 0.001,
                        "completion_per_1k_tokens": 0.002,
                    }
                }
            },
        )

        planning_span = observer.start_stage("planning", scope="request")
        planning_span.complete(status="success", metadata={"task_count": 1})
        observer.record_fallback("planning_returned_no_tasks")
        observer.record_degraded("fallback_task_used")
        observer.record_search_attempt(cache_hit=False, success=True)
        observer.record_search_attempt(cache_hit=True, success=True)
        observer.record_llm_call(
            success=True,
            prompt_text="p" * 400,
            completion_text="c" * 200,
        )
        observer.set_task_totals(total_tasks=1)
        observer.update_task_status_counts(completed=1)
        request_snapshot = observer.complete_request(status="partial_success")
        aggregate_snapshot = metrics_registry.snapshot()

        self.assertTrue(request_snapshot["fallback_triggered"])
        self.assertEqual(aggregate_snapshot["counters"]["request_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["request_partial_success_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["fallback_trigger_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["search_call_total"], 2)
        self.assertEqual(aggregate_snapshot["counters"]["cache_hit_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["cache_miss_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["llm_call_total"], 1)
        self.assertGreater(aggregate_snapshot["counters"]["total_tokens"], 0)
        self.assertGreater(aggregate_snapshot["estimated_cost"], 0)
        self.assertEqual(aggregate_snapshot["latencies_ms"]["planning_latency_ms"]["count"], 1)
        self.assertEqual(aggregate_snapshot["latencies_ms"]["total_latency_ms"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
