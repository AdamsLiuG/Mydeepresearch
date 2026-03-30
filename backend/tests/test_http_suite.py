import json
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from evals.loader import load_benchmark_cases
from evals.run_http_suite import (
    DEFAULT_BENCHMARK_PATH,
    _discover_perf_results,
    _health_check,
    _request_session,
    _send_sync_request,
    _stream_request,
    build_payload,
    render_interview_summary,
)
from evals.schema import BenchmarkCase


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _StubSuiteHandler(BaseHTTPRequestHandler):
    recent_requests: list[dict] = []

    def log_message(self, format, *args):  # noqa: A003
        return

    @classmethod
    def _store_trace(cls, trace: dict) -> None:
        cls.recent_requests.insert(0, trace)
        cls.recent_requests = cls.recent_requests[:25]

    def _write_json(self, status: int, payload: dict, *, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _build_trace(self, request_id: str, topic: str) -> tuple[dict, str, list[dict]]:
        report = (
            f"# 背景\n{topic}\n"
            "## 关键发现\n"
            "- AI agent 可以结合 search 工具完成研究 [T1-S1]\n"
            "## 结论\n"
            "https://example.com/report\n"
            "## 参考来源\n"
            "- [T1-S1] Example Source - https://example.com/report\n"
        )
        todo_items = [
            {
                "id": 1,
                "title": "任务1",
                "intent": "梳理主题背景",
                "query": topic,
                "status": "completed",
                "summary": "AI agent 可以结合 search 工具完成研究 [T1-S1]",
                "sources_summary": "* Example Source : https://example.com/report",
                "notices": [],
                "evidence_items": [],
                "claims": [],
                "review_issues": [],
                "review_status": "passed",
                "note_id": None,
                "note_path": None,
                "origin": "planned",
                "round": 1,
            }
        ]
        trace = {
            "request_id": request_id,
            "topic": topic,
            "search_api": "advanced",
            "status": "success",
            "elapsed_ms": 123.4,
            "fallback_triggered": False,
            "fallback_reasons": [],
            "degraded": False,
            "degraded_reasons": [],
            "total_tasks": 1,
            "completed_tasks": 1,
            "skipped_tasks": 0,
            "failed_tasks": 0,
            "cache_hits": 1,
            "cache_exact_hits": 1,
            "cache_semantic_hits": 0,
            "cache_misses": 0,
            "prompt_tokens": 100,
            "completion_tokens": 80,
            "total_tokens": 180,
            "token_source": "estimated",
            "estimated_cost": 0.0012,
            "report_markdown": report,
            "todo_items": todo_items,
            "stages": [],
        }
        return trace, report, todo_items

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write_json(200, {"status": "ok"}, headers={"X-Request-ID": "healthz-test"})
            return

        if self.path == "/metrics/json":
            payload = {
                "generated_at": "2026-01-01T00:00:00Z",
                "success_rate": 1.0,
                "failure_rate": 0.0,
                "cache_hit_total": len(self.__class__.recent_requests),
                "cache_exact_hit_total": len(self.__class__.recent_requests),
                "cache_semantic_hit_total": 0,
                "cache_miss_total": 0,
                "estimated_cost": round(
                    sum(float(item.get("estimated_cost") or 0.0) for item in self.__class__.recent_requests),
                    6,
                ),
                "latencies_ms": {},
                "counters": {"request_total": len(self.__class__.recent_requests)},
                "recent_requests": list(self.__class__.recent_requests),
            }
            self._write_json(200, payload)
            return

        self._write_json(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(raw_body or "{}")
        topic = str(payload.get("topic") or "").strip()
        request_id = self.headers.get("X-Request-ID") or "missing-request-id"
        trace, report, todo_items = self._build_trace(request_id, topic)
        self.__class__._store_trace(trace)

        if self.path == "/research":
            self._write_json(
                200,
                {"report_markdown": report, "todo_items": todo_items},
                headers={"X-Request-ID": request_id},
            )
            return

        if self.path == "/research/stream":
            events = [
                {"type": "status", "message": "初始化研究流程"},
                {"type": "todo_list", "tasks": todo_items, "step": 0},
                {
                    "type": "tool_call",
                    "tool_name": "search",
                    "task_id": 1,
                    "input": topic,
                },
                {
                    "type": "metrics_snapshot",
                    "request_id": request_id,
                    "request_metrics": trace,
                    "aggregate_metrics": {"success_rate": 1.0},
                },
                {"type": "final_report", "report": report},
                {"type": "done"},
            ]
            body = "".join(
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                for event in events
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Request-ID", request_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self._write_json(404, {"detail": "not found"}, headers={"X-Request-ID": request_id})


class HttpSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        port = _free_port()
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.server = ThreadingHTTPServer(("127.0.0.1", port), _StubSuiteHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        _StubSuiteHandler.recent_requests = []

    def test_full_system_benchmark_file_contains_expected_cases(self):
        cases = load_benchmark_cases(DEFAULT_BENCHMARK_PATH)

        self.assertEqual(len(cases), 12)
        self.assertEqual(cases[0].id, "concept_mcp")
        self.assertEqual(cases[-1].id, "enterprise_tradeoffs_2025")

    def test_sync_and_stream_runner_collect_results_and_render_summary(self):
        case = BenchmarkCase(
            id="case-http",
            topic="AI agent 系统如何结合 search 工具完成研究？",
            expected_keywords=["agent", "search"],
            expected_sections=["背景", "关键发现", "结论"],
            freshness_sensitive=False,
            metadata={"level": "测试层", "category": "http"},
        )

        _health_check(self.base_url, timeout_seconds=5.0)
        with _request_session() as session:
            sync_result = _send_sync_request(
                session,
                base_url=self.base_url,
                case=case,
                search_api="advanced",
                request_id="suite-sync-case-http",
                timeout_seconds=5.0,
                trace_timeout_seconds=5.0,
                poll_interval_seconds=0.1,
            )
            stream_result = _stream_request(
                session,
                base_url=self.base_url,
                case=case,
                search_api="advanced",
                request_id="suite-stream-case-http",
                timeout_seconds=5.0,
                trace_timeout_seconds=5.0,
                poll_interval_seconds=0.1,
            )

        payload = build_payload(
            cases=[case],
            benchmark_path="dummy.jsonl",
            base_url=self.base_url,
            mode="both",
            search_api="advanced",
            sync_results=[sync_result],
            stream_results=[stream_result],
            perf_payloads=[],
        )
        summary_markdown = render_interview_summary(payload)

        self.assertTrue(sync_result["http_ok"])
        self.assertTrue(sync_result["case_passed"])
        self.assertEqual(sync_result["judge_metrics"]["section_completeness"], 1.0)
        self.assertEqual(sync_result["judge_metrics"]["keyword_coverage"], 1.0)
        self.assertTrue(stream_result["http_ok"])
        self.assertTrue(stream_result["case_passed"])
        self.assertEqual(stream_result["missing_required_events"], [])
        self.assertTrue(stream_result["final_report_before_done"])
        self.assertIn("## 用例总表", summary_markdown)
        self.assertIn("case-http", summary_markdown)
        self.assertIn("前端手工验收", summary_markdown)

    def test_discover_perf_results_supports_explicit_paths(self):
        with self.subTest("explicit perf payload"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp_dir:
                result_path = Path(tmp_dir) / "smoke-real_local.json"
                result_path.write_text(
                    json.dumps(
                        {
                            "mode": "smoke",
                            "profile": "real_local",
                            "summary": {"rps": 1.5, "p95_latency_ms": 200, "p99_latency_ms": 300},
                            "baseline_comparison": {"passed": True},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                perf_results = _discover_perf_results("real_local", [str(result_path)])

        self.assertEqual(len(perf_results), 1)
        self.assertEqual(perf_results[0]["mode"], "smoke")
        self.assertTrue(perf_results[0]["baseline_comparison"]["passed"])


if __name__ == "__main__":
    unittest.main()
