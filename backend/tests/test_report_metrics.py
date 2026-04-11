import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from evals.report_metrics import build_report_payload, render_markdown


class ReportMetricsTests(unittest.TestCase):
    def test_build_report_payload_aggregates_request_state_and_eval_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_dir = root / "requests"
            request_dir.mkdir(parents=True, exist_ok=True)
            eval_file = root / "http_suite.json"

            request_payload = {
                "request_id": "req-1",
                "phase": "completed",
                "status": "success",
                "todo_items": [
                    {
                        "id": 1,
                        "status": "completed",
                        "claims": [
                            {"support_status": "supported"},
                            {"support_status": "invalid_citation"},
                        ],
                    },
                    {
                        "id": 2,
                        "status": "failed",
                        "claims": [],
                    },
                ],
                "request_metrics": {
                    "request_id": "req-1",
                    "status": "success",
                    "elapsed_ms": 120.0,
                    "cache_hits": 3,
                    "cache_misses": 1,
                    "cache_exact_hits": 1,
                    "cache_semantic_hits": 2,
                },
            }
            (request_dir / "req-1.json").write_text(
                json.dumps(request_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            nonterminal_payload = {
                "request_id": "req-in-progress",
                "phase": "task_execution",
                "status": "in_progress",
                "todo_items": [{"id": 1, "status": "pending", "claims": []}],
                "request_metrics": {"request_id": "req-in-progress", "status": "in_progress"},
            }
            (request_dir / "req-in-progress.json").write_text(
                json.dumps(nonterminal_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            eval_payload = {
                "generated_at": "2026-04-08T00:00:00Z",
                "sync_suite": {
                    "summary": {"total_cases": 2},
                    "results": [
                        {
                            "request_id": "req-1",
                            "trace": {
                                "request_id": "req-1",
                                "status": "success",
                                "elapsed_ms": 120.0,
                                "cache_hits": 3,
                                "cache_misses": 1,
                                "cache_exact_hits": 1,
                                "cache_semantic_hits": 2,
                            },
                            "todo_items": [
                                {
                                    "id": 1,
                                    "status": "completed",
                                    "claims": [{"support_status": "supported"}],
                                }
                            ],
                        },
                        {
                            "request_id": "req-eval-only",
                            "trace": {
                                "request_id": "req-eval-only",
                                "status": "partial_success",
                                "elapsed_ms": 210.0,
                                "cache_hits": 0,
                                "cache_misses": 3,
                                "cache_exact_hits": 0,
                                "cache_semantic_hits": 0,
                            },
                            "todo_items": [
                                {
                                    "id": 2,
                                    "status": "completed",
                                    "claims": [{"support_status": "missing_citation"}],
                                },
                                {
                                    "id": 3,
                                    "status": "skipped",
                                    "claims": [],
                                },
                            ],
                        },
                    ],
                },
                "stream_suite": {"summary": {"total_cases": 0}, "results": []},
            }
            eval_file.write_text(
                json.dumps(eval_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            payload = build_report_payload(
                request_state_dirs=[request_dir],
                eval_result_paths=[eval_file],
                include_nonterminal=False,
            )

            request_summary = payload["request_state"]["summary_all"]
            self.assertEqual(payload["request_state"]["included_records"], 1)
            self.assertEqual(payload["request_state"]["skipped_nonterminal_records"], 1)
            self.assertEqual(request_summary["request_count"], 1)
            self.assertEqual(request_summary["task_counts"]["completed"], 1)
            self.assertEqual(request_summary["task_counts"]["failed"], 1)
            self.assertEqual(request_summary["task_completion_rate"], 0.5)
            self.assertEqual(request_summary["citation_validity_rate"], 0.5)
            self.assertEqual(request_summary["cache"]["hit_rate"], 0.75)
            self.assertEqual(request_summary["latency_ms"]["average"], 120.0)

            matched_summary = payload["request_state"]["summary_matched_eval_request_ids"]
            self.assertIsNotNone(matched_summary)
            self.assertEqual(matched_summary["request_count"], 1)

            eval_summary = payload["eval_results"]["aggregate_summary"]
            self.assertEqual(eval_summary["request_count"], 2)
            self.assertEqual(eval_summary["task_counts"]["completed"], 2)
            self.assertEqual(eval_summary["task_counts"]["skipped"], 1)
            self.assertEqual(eval_summary["task_completion_rate"], 0.6667)
            self.assertEqual(eval_summary["citation_validity_rate"], 0.5)
            self.assertEqual(eval_summary["cache"]["hit_rate"], 0.4286)
            self.assertEqual(eval_summary["latency_ms"]["average"], 165.0)

            markdown = render_markdown(payload)
            self.assertIn("任务完成率", markdown)
            self.assertIn("引用有效率", markdown)
            self.assertIn("http_suite.json", markdown)


if __name__ == "__main__":
    unittest.main()
