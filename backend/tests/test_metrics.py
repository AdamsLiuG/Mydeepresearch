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
        observer.record_search_attempt(cache_hit=False, success=True, cache_strategy="miss")
        observer.record_search_attempt(cache_hit=True, success=True, cache_strategy="exact")
        observer.record_search_attempt(cache_hit=True, success=True, cache_strategy="semantic")
        observer.record_reflection_call(
            reason="补充执行 1 个任务",
            gap_signals=["task_1_summary_missing"],
            added_tasks=1,
        )
        observer.record_review_summary(
            {
                "overall_status": "warning",
                "reason": "evidence thin",
                "issue_count": 2,
            }
        )
        observer.record_task_react_round()
        observer.record_task_react_continue()
        observer.record_task_react_round()
        observer.record_task_react_stop("max_rounds_reached")
        observer.record_report_repair(added_tasks=1)
        observer.record_note_memory_query(hit_count=2, match_types=["conclusion", "task_state"])
        observer.record_note_memory_prompt_injection(match_types=["conclusion"])
        observer.record_strategy_memory_query(
            hit_count=2,
            match_kinds=["planning_pattern", "anti_pattern"],
        )
        observer.record_strategy_memory_prompt_injection(match_kinds=["planning_pattern"])
        observer.record_llm_call(
            success=True,
            prompt_text="p" * 400,
            completion_text="c" * 200,
        )
        observer.set_task_totals(total_tasks=1)
        observer.update_task_status_counts(completed=1)
        observer.attach_result(
            report_markdown="# Final Report",
            todo_items=[
                {
                    "id": 1,
                    "title": "任务1",
                    "intent": "验证指标",
                    "query": "ai agent",
                    "status": "completed",
                    "summary": "完成",
                    "sources_summary": "source",
                }
            ],
        )
        request_snapshot = observer.complete_request(status="partial_success")
        aggregate_snapshot = metrics_registry.snapshot()

        self.assertTrue(request_snapshot["fallback_triggered"])
        self.assertEqual(request_snapshot["report_markdown"], "# Final Report")
        self.assertEqual(len(request_snapshot["todo_items"]), 1)
        self.assertEqual(aggregate_snapshot["recent_requests"][0]["report_markdown"], "# Final Report")
        self.assertEqual(aggregate_snapshot["counters"]["request_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["request_partial_success_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["fallback_trigger_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["search_call_total"], 3)
        self.assertEqual(aggregate_snapshot["counters"]["cache_hit_total"], 2)
        self.assertEqual(aggregate_snapshot["counters"]["cache_exact_hit_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["cache_semantic_hit_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["cache_miss_total"], 1)
        self.assertEqual(request_snapshot["cache_exact_hits"], 1)
        self.assertEqual(request_snapshot["cache_semantic_hits"], 1)
        self.assertTrue(request_snapshot["reflection_triggered"])
        self.assertEqual(request_snapshot["reflection_added_tasks"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["llm_call_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["reflection_call_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["reflection_replan_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["review_call_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["review_issue_total"], 2)
        self.assertEqual(aggregate_snapshot["counters"]["task_react_round_total"], 2)
        self.assertEqual(aggregate_snapshot["counters"]["task_react_continue_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["task_react_stop_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["report_repair_trigger_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["report_repair_added_task_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["note_memory_query_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["note_memory_hit_total"], 2)
        self.assertEqual(aggregate_snapshot["counters"]["note_memory_prompt_injection_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["strategy_memory_query_total"], 1)
        self.assertEqual(aggregate_snapshot["counters"]["strategy_memory_hit_total"], 2)
        self.assertEqual(aggregate_snapshot["counters"]["strategy_memory_prompt_injection_total"], 1)
        self.assertEqual(aggregate_snapshot["task_react_stop_reason_counts"]["max_rounds_reached"], 1)
        self.assertEqual(request_snapshot["task_react_rounds"], 2)
        self.assertEqual(request_snapshot["task_react_continue_count"], 1)
        self.assertEqual(request_snapshot["task_react_stop_count"], 1)
        self.assertTrue(request_snapshot["report_repair_triggered"])
        self.assertEqual(request_snapshot["report_repair_added_tasks"], 1)
        self.assertEqual(request_snapshot["note_memory_queries"], 1)
        self.assertEqual(request_snapshot["note_memory_hits"], 2)
        self.assertEqual(request_snapshot["note_memory_prompt_injections"], 1)
        self.assertEqual(request_snapshot["note_memory_last_match_types"], ["conclusion"])
        self.assertEqual(request_snapshot["strategy_memory_queries"], 1)
        self.assertEqual(request_snapshot["strategy_memory_hits"], 2)
        self.assertEqual(request_snapshot["strategy_memory_prompt_injections"], 1)
        self.assertEqual(request_snapshot["strategy_memory_last_match_kinds"], ["planning_pattern"])
        self.assertEqual(request_snapshot["avg_task_react_rounds"], 2.0)
        self.assertEqual(request_snapshot["review_summary"]["overall_status"], "warning")
        self.assertGreater(aggregate_snapshot["counters"]["total_tokens"], 0)
        self.assertGreater(aggregate_snapshot["estimated_cost"], 0)
        self.assertEqual(aggregate_snapshot["latencies_ms"]["planning_latency_ms"]["count"], 1)
        self.assertEqual(aggregate_snapshot["latencies_ms"]["total_latency_ms"]["count"], 1)

    def test_metrics_record_approximate_cache_hit_modes_with_semantic_compatibility(self):
        observer = RequestTrace(
            request_id="req-approximate",
            topic="AI cache",
            search_api="duckduckgo",
            provider="custom",
            model="demo-model",
            pricing_catalog={},
        )

        observer.record_search_attempt(cache_hit=True, success=True, cache_strategy="approximate_dense")
        observer.record_search_attempt(cache_hit=True, success=True, cache_strategy="approximate_sparse")
        observer.record_search_attempt(cache_hit=True, success=True, cache_strategy="approximate_hybrid")
        request_snapshot = observer.complete_request(status="success")
        aggregate_snapshot = metrics_registry.snapshot()

        self.assertEqual(request_snapshot["cache_hits"], 3)
        self.assertEqual(request_snapshot["cache_semantic_hits"], 3)
        self.assertEqual(request_snapshot["cache_approximate_hits"], 3)
        self.assertEqual(request_snapshot["cache_approximate_dense_hits"], 1)
        self.assertEqual(request_snapshot["cache_approximate_sparse_hits"], 1)
        self.assertEqual(request_snapshot["cache_approximate_hybrid_hits"], 1)
        self.assertEqual(aggregate_snapshot["cache_approximate_hit_total"], 3)
        self.assertEqual(aggregate_snapshot["cache_approximate_dense_hit_total"], 1)
        self.assertEqual(aggregate_snapshot["cache_approximate_sparse_hit_total"], 1)
        self.assertEqual(aggregate_snapshot["cache_approximate_hybrid_hit_total"], 1)
        self.assertEqual(aggregate_snapshot["cache_semantic_hit_total"], 3)


if __name__ == "__main__":
    unittest.main()
