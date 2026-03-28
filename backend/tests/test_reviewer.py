import sys
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from models import SummaryState, TodoItem
from services.reviewer import ReviewService


class ReviewerTests(unittest.TestCase):
    def test_review_service_flags_missing_and_invalid_citations(self):
        config = Configuration.from_env(
            overrides={
                "review_agent_enabled": False,
                "review_min_sources_per_task": 2,
                "review_min_domains_per_task": 2,
            },
            load_env_file=False,
        )
        reviewer = ReviewService(None, config)
        task = TodoItem(
            id=1,
            title="任务1",
            intent="梳理最新进展",
            query="AI agent 最新进展",
            status="completed",
            summary=(
                "任务总结\n"
                "- 这是一个缺少引用的结论\n"
                "- 这是一个错误引用的结论 [T1-S9]\n"
            ),
            evidence_items=[
                {
                    "source_id": "T1-S1",
                    "title": "Example",
                    "url": "https://example.com/post",
                    "domain": "example.com",
                    "source_type": "web",
                    "quality_label": "low",
                    "freshness_label": "unknown",
                }
            ],
        )
        state = SummaryState(research_topic="AI agent 最新进展", todo_items=[task])

        summary = reviewer.review_request(state)

        self.assertEqual(summary["overall_status"], "blocked")
        self.assertGreaterEqual(summary["issue_count"], 3)
        self.assertEqual(task.review_status, "blocked")
        self.assertTrue(any(issue["check"] == "missing_citation" for issue in task.review_issues))
        self.assertTrue(any(issue["check"] == "invalid_citation" for issue in task.review_issues))
        self.assertEqual(task.claims[0]["support_status"], "missing_citation")
        self.assertEqual(task.claims[1]["support_status"], "invalid_citation")

    def test_review_service_marks_supported_claims_as_passed(self):
        config = Configuration.from_env(
            overrides={"review_agent_enabled": False},
            load_env_file=False,
        )
        reviewer = ReviewService(None, config)
        task = TodoItem(
            id=2,
            title="任务2",
            intent="梳理稳定背景",
            query="MCP 概念",
            status="completed",
            summary="- MCP 用于标准化模型与工具的上下文交互 [T2-S1]",
            evidence_items=[
                {
                    "source_id": "T2-S1",
                    "title": "Docs",
                    "url": "https://example.edu/docs",
                    "domain": "example.edu",
                    "source_type": "education",
                    "quality_label": "high",
                    "published_at": "2026-03-01",
                    "freshness_label": "fresh",
                },
                {
                    "source_id": "T2-S2",
                    "title": "Spec",
                    "url": "https://example.gov/spec",
                    "domain": "example.gov",
                    "source_type": "government",
                    "quality_label": "high",
                    "published_at": "2026-03-05",
                    "freshness_label": "fresh",
                },
            ],
        )
        state = SummaryState(research_topic="什么是 MCP", todo_items=[task])

        summary = reviewer.review_request(state)

        self.assertEqual(summary["overall_status"], "passed")
        self.assertEqual(summary["issue_count"], 0)
        self.assertEqual(task.review_status, "passed")
        self.assertEqual(task.claims[0]["support_status"], "supported")


if __name__ == "__main__":
    unittest.main()
