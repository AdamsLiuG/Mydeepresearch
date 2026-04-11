import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from evals.judges.llm import LLMJudge
from evals.schema import BenchmarkCase


class LLMJudgeTests(unittest.TestCase):
    def test_llm_judge_parses_successful_json_response(self):
        calls = []

        def invoke(prompt: str, response_format):
            calls.append({"prompt": prompt, "response_format": response_format})
            return """
{
  "factuality_score": 0.82,
  "coverage_score": 0.75,
  "citation_grounding_score": 0.78,
  "freshness_score": 0.61,
  "conservativeness_score": 0.88,
  "overall_verdict": "warning",
  "reason": "整体较完整，但部分结论证据仍偏弱。",
  "findings": [
    {"severity": "high", "category": "unsupported_claim", "message": "部分断言支撑不足。"}
  ],
  "scoring_notes": {
    "strengths": ["结构清晰"],
    "weaknesses": ["个别结论证据薄弱"]
  }
}
"""

        judge = LLMJudge(
            model="test-judge",
            invocation=invoke,
            load_env_file=False,
        )
        case = BenchmarkCase(
            id="case-1",
            topic="AI agent",
            expected_keywords=["agent"],
            expected_sections=["背景"],
            freshness_sensitive=False,
        )

        metrics = judge.evaluate(
            case=case,
            report_markdown="# 背景\nAI agent report",
            todo_items=[],
            trace_snapshot={"elapsed_ms": 12.5, "estimated_cost": 0.001, "status": "success"},
        )

        self.assertEqual(metrics["judge_status"], "success")
        self.assertEqual(metrics["overall_verdict"], "warning")
        self.assertEqual(metrics["factuality_score"], 0.82)
        self.assertEqual(metrics["findings"][0]["category"], "unsupported_claim")
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})

    def test_llm_judge_falls_back_when_json_mode_is_unsupported(self):
        calls = []

        def invoke(prompt: str, response_format):
            calls.append(response_format)
            if response_format is not None:
                raise TypeError("run() got an unexpected keyword argument 'response_format'")
            return """
{
  "factuality_score": 0.9,
  "coverage_score": 0.8,
  "citation_grounding_score": 0.85,
  "freshness_score": 0.7,
  "conservativeness_score": 0.9,
  "overall_verdict": "pass",
  "reason": "整体较稳。",
  "findings": [],
  "scoring_notes": {"strengths": ["稳健"], "weaknesses": []}
}
"""

        judge = LLMJudge(model="test-judge", invocation=invoke, load_env_file=False)
        case = BenchmarkCase(id="case-2", topic="topic")

        metrics = judge.evaluate(
            case=case,
            report_markdown="# Report\ncontent",
            todo_items=[],
            trace_snapshot={},
        )

        self.assertEqual(metrics["judge_status"], "success")
        self.assertEqual(metrics["overall_verdict"], "pass")
        self.assertEqual(calls, [{"type": "json_object"}, None])

    def test_llm_judge_returns_error_payload_for_invalid_json(self):
        judge = LLMJudge(
            model="test-judge",
            invocation=lambda prompt, response_format: "not a json payload",
            load_env_file=False,
        )
        case = BenchmarkCase(id="case-3", topic="topic")

        metrics = judge.evaluate(
            case=case,
            report_markdown="# Report\ncontent",
            todo_items=[],
            trace_snapshot={"status": "partial_success", "degraded": True},
        )

        self.assertEqual(metrics["judge_status"], "error")
        self.assertEqual(metrics["overall_verdict"], "warning")
        self.assertTrue(metrics["degraded_flag"])

    def test_llm_judge_skips_empty_report(self):
        judge = LLMJudge(model="test-judge", invocation=lambda prompt, response_format: "{}", load_env_file=False)
        case = BenchmarkCase(id="case-4", topic="topic")

        metrics = judge.evaluate(
            case=case,
            report_markdown="",
            todo_items=[],
            trace_snapshot={},
        )

        self.assertEqual(metrics["judge_status"], "skipped")
        self.assertEqual(metrics["overall_verdict"], "fail")
        self.assertFalse(metrics["report_generated"])


if __name__ == "__main__":
    unittest.main()
