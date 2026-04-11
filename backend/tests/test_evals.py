import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from evals.judges.heuristic import HeuristicJudge
from evals.judges.llm import LLMJudge
from evals.loader import load_benchmark_cases
from evals.run_benchmark import build_judge, parse_args
from evals.runner import run_benchmark_suite
from evals.schema import BenchmarkCase


class EvalLoaderTests(unittest.TestCase):
    def test_load_benchmark_cases_supports_json_and_jsonl(self):
        sample_case = {
            "id": "case-1",
            "topic": "AI agent",
            "expected_keywords": ["agent"],
            "expected_sections": ["背景"],
            "freshness_sensitive": False,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_path = tmp_path / "cases.json"
            jsonl_path = tmp_path / "cases.jsonl"

            json_path.write_text(json.dumps([sample_case], ensure_ascii=False), encoding="utf-8")
            jsonl_path.write_text(
                json.dumps(sample_case, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            json_cases = load_benchmark_cases(json_path)
            jsonl_cases = load_benchmark_cases(jsonl_path)

        self.assertEqual(json_cases[0].id, "case-1")
        self.assertEqual(jsonl_cases[0].topic, "AI agent")


class HeuristicJudgeTests(unittest.TestCase):
    def test_heuristic_judge_computes_expected_metrics(self):
        case = BenchmarkCase(
            id="judge-1",
            topic="AI agent",
            expected_keywords=["agent", "search"],
            expected_sections=["背景", "结论"],
            freshness_sensitive=False,
        )
        report = (
            "# 背景\n"
            "AI agent 可以结合 search 工具完成研究。\n"
            "## 结论\n"
            "参考 https://example.com 和 https://docs.example.org。"
        )

        metrics = HeuristicJudge().evaluate(
            case=case,
            report_markdown=report,
            todo_items=[SimpleNamespace(id=1)],
            trace_snapshot={"elapsed_ms": 123.4, "estimated_cost": 0.002, "status": "success"},
        )

        self.assertTrue(metrics["report_generated"])
        self.assertEqual(metrics["section_completeness"], 1.0)
        self.assertEqual(metrics["keyword_coverage"], 1.0)
        self.assertEqual(metrics["citation_count"], 2)
        self.assertEqual(metrics["total_latency_ms"], 123.4)

    def test_heuristic_judge_tracks_grounded_citations_and_reference_match(self):
        case = BenchmarkCase(
            id="judge-grounded",
            topic="AI agent",
            expected_keywords=["agent"],
            expected_sections=["背景", "核心洞见", "参考来源"],
            freshness_sensitive=False,
        )
        report = (
            "# 背景\n"
            "AI agent 系统需要 grounded report。\n"
            "## 核心洞见\n"
            "- 关键结论 A [T1-S1]\n"
            "- 关键结论 B [T1-S2]\n"
            "## 证据与数据\n"
            "- 事实数据 [T1-S1]\n"
            "## 风险与挑战\n"
            "- 风险提示 [T1-S2]\n"
            "## 参考来源\n"
            "- [T1-S1] Source One - https://example.com/1\n"
            "- [T1-S2] Source Two - https://example.com/2\n"
        )

        metrics = HeuristicJudge().evaluate(
            case=case,
            report_markdown=report,
            todo_items=[SimpleNamespace(id=1, status="completed")],
            trace_snapshot={"elapsed_ms": 88.0, "estimated_cost": 0.001, "status": "success"},
        )

        self.assertTrue(metrics["reference_section_present"])
        self.assertEqual(metrics["citation_marker_count"], 2)
        self.assertEqual(metrics["reference_match_rate"], 1.0)
        self.assertEqual(metrics["grounded_bullet_ratio"], 1.0)
        self.assertEqual(metrics["completed_task_count"], 1)


class EvalRunnerTests(unittest.TestCase):
    def test_run_benchmark_suite_writes_output_payload(self):
        class StubTrace:
            def snapshot(self):
                return {
                    "status": "partial_success",
                    "degraded": True,
                    "elapsed_ms": 88.0,
                    "estimated_cost": 0.0015,
                }

        class StubAgent:
            def __init__(self, config, request_id=None):
                self.config = config
                self.request_id = request_id
                self._request_trace = StubTrace()

            def run(self, topic: str):
                todo_item = SimpleNamespace(
                    id=1,
                    title="任务1",
                    intent="梳理背景",
                    query=topic,
                    status="completed",
                    note_id=None,
                    note_path=None,
                )
                return SimpleNamespace(
                    report_markdown="# 背景\nAI agent benchmark\n## 结论\nhttps://example.com",
                    running_summary="# 背景\nAI agent benchmark",
                    todo_items=[todo_item],
                )

        cases = [
            BenchmarkCase(
                id="case-1",
                topic="AI agent",
                expected_keywords=["agent"],
                expected_sections=["背景", "结论"],
                freshness_sensitive=False,
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            payload = run_benchmark_suite(
                cases,
                config=Configuration.from_env(load_env_file=False),
                agent_factory=StubAgent,
                output_path=output_path,
                benchmark_path="sample.jsonl",
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["total_cases"], 1)
        self.assertEqual(payload["summary"]["reports_generated"], 1)
        self.assertEqual(written["results"][0]["metrics"]["citation_count"], 1)
        self.assertTrue(written["results"][0]["metrics"]["degraded_flag"])


class BenchmarkCLITests(unittest.TestCase):
    def test_parse_args_supports_llm_judge_options(self):
        args = parse_args(
            [
                "--judge",
                "llm",
                "--judge-model",
                "gpt-5.4",
                "--judge-base-url",
                "https://example.com/v1",
                "--judge-timeout-seconds",
                "45",
                "--judge-version",
                "judge_v_test",
            ]
        )

        self.assertEqual(args.judge, "llm")
        self.assertEqual(args.judge_model, "gpt-5.4")
        self.assertEqual(args.judge_base_url, "https://example.com/v1")
        self.assertEqual(args.judge_timeout_seconds, 45.0)
        self.assertEqual(args.judge_version, "judge_v_test")

    def test_build_judge_returns_llm_judge_when_requested(self):
        args = parse_args(["--judge", "llm", "--judge-model", "judge-model"])
        judge = build_judge(args, config=Configuration.from_env(load_env_file=False))

        self.assertIsInstance(judge, LLMJudge)


if __name__ == "__main__":
    unittest.main()
