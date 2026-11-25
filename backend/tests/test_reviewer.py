import sys
import unittest
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from models import SummaryState, TodoItem
from services.evidence import EvidenceStore
from services.reviewer import ReviewService
from services.web_extractor import ExtractionContext, WebExtractor
from services.web_fetch_gateway import FetchedDocument, FetchScope


class ReviewerTests(unittest.TestCase):
    def test_enabled_numeric_fallback_uses_visible_table_rows_and_units(self):
        store = EvidenceStore(web_extraction_enabled=True)
        url = "https://example.org/results"
        store.record_search_results(task_id=1, query="sales", backend="fixture", search_payload={
            "results": [{"url": url, "content": "Sales reached $91 million in 2025."}],
        })
        document = FetchedDocument(
            url, "text/html",
            b'<main><table><caption>Units: USD million</caption><tr><th>Year</th><th>Sales</th></tr>'
            b'<tr><td>2025</td><td>38</td></tr><tr><td>2024</td><td>19</td></tr></table></main>',
        )
        extractor = WebExtractor(gateway=SimpleNamespace(fetch=lambda *args, **kwargs: document))
        artifact = extractor.extract(
            url, context=ExtractionContext("default", "request", 1, FetchScope(monotonic() + 5)),
        )
        saved = store.update_full_content(
            task_id=1, source_id="T1-S1", url=url, title="", full_content="", web_result=artifact,
        )
        self.assertTrue(store.is_eligible_for_facts("T1-S1"))
        numeric = ReviewService._normalized_numeric_fallback
        self.assertIsNone(numeric("Sales reached $38 million in 2025.", [saved], web_extraction_enabled=True))
        self.assertIsNotNone(numeric("Sales reached $19 million in 2025.", [saved], web_extraction_enabled=True))
        self.assertIsNotNone(numeric("Sales reached $91 million in 2025.", [saved], web_extraction_enabled=True))

    def test_enabled_l0_is_excluded_from_coverage_and_numeric_grounding(self):
        config = Configuration.from_env(overrides={"review_agent_enabled": False}, load_env_file=False)
        store = EvidenceStore(web_extraction_enabled=True)
        items = store.record_search_results(task_id=1, query="measurements", backend="fixture", search_payload={
            "results": [{"url": "https://example.org", "content": "The device reduced latency by 91%."}],
        })
        task = TodoItem(
            id=1, title="Benchmark", intent="Check latency", query="latency", status="completed",
            summary="- The device reduced latency by 91% [T1-S1]",
            evidence_items=[{**items[0], "full_content": "untrusted-state-body"}],
        )
        state = SummaryState(research_topic="Measurements", todo_items=[task])
        numeric = Mock(return_value=(True, ""))
        reviewer = ReviewService(None, config, evidence_store=store, claim_verifier=numeric)
        reviewer.review_request(state)
        numeric.assert_not_called()
        self.assertEqual(task.claims[0]["support_status"], "invalid_citation")
        self.assertIn("weak_evidence", {issue["check"] for issue in task.review_issues})
        prompt = reviewer._build_prompt(state, [])
        self.assertIn("<WEB_EVIDENCE_DATA", prompt)
        self.assertNotIn("untrusted-state-body", prompt)

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

    def test_review_prompt_omits_tool_directives_when_vllm_provider(self):
        config = Configuration.from_env(
            overrides={"llm_provider": "vllm", "review_agent_enabled": True},
            load_env_file=False,
        )
        reviewer = ReviewService(None, config)
        state = SummaryState(
            research_topic="MCP 协议",
            todo_items=[
                TodoItem(
                    id=1,
                    title="概念",
                    intent="解释概念",
                    query="MCP",
                    status="completed",
                    summary="- 结论 [T1-S1]",
                )
            ],
        )
        prompt = reviewer._build_prompt(state, [])
        self.assertNotIn("evidence_lookup", prompt)
        self.assertNotIn("fetch_page", prompt)
        self.assertIn("不启用工具调用", prompt)

    def test_review_prompt_includes_evidence_excerpts_and_flags_numeric_mismatch(self):
        config = Configuration.from_env(
            overrides={"review_agent_enabled": False},
            load_env_file=False,
        )
        reviewer = ReviewService(None, config)
        task = TodoItem(
            id=3,
            title="性能基准",
            intent="核对吞吐指标",
            query="vLLM throughput",
            status="completed",
            summary="- 某引擎在测试中实现了 98% 的延迟下降并达到 120GB 显存节省 [T3-S1]",
            evidence_items=[
                {
                    "source_id": "T3-S1",
                    "title": "vLLM Benchmark",
                    "url": "https://docs.vllm.ai/bench",
                    "domain": "docs.vllm.ai",
                    "snippet": "In our evaluation vLLM achieved 24% latency reduction with PagedAttention.",
                    "quality_label": "high",
                    "freshness_label": "fresh",
                    "published_at": "2026-03-01",
                },
                {
                    "source_id": "T3-S2",
                    "title": "Arxiv Paper",
                    "url": "https://arxiv.org/abs/2309.06180",
                    "domain": "arxiv.org",
                    "snippet": "PagedAttention improves memory utilization significantly.",
                    "quality_label": "high",
                    "freshness_label": "fresh",
                    "published_at": "2026-03-01",
                },
            ],
        )
        state = SummaryState(research_topic="vLLM 性能", todo_items=[task])

        prompt = reviewer._build_prompt(state, [])
        self.assertIn("关键证据摘录", prompt)
        self.assertIn("24% latency reduction", prompt)

        summary = reviewer.review_request(state)
        self.assertEqual(summary["overall_status"], "warning")
        self.assertTrue(any(issue["check"] == "claim_evidence_mismatch" for issue in task.review_issues))
        self.assertEqual(task.claims[0]["support_status"], "unverified_claim")


if __name__ == "__main__":
    unittest.main()


def test_report_verification_requires_frozen_grounding():
    assert ReviewService.verify_report_claim(
        "The sample reached 25%.", ("The sample reached 20%.",),
    ) == "unsupported"
    assert ReviewService.verify_report_claim(
        "The sample reached 20%.", ("The sample reached 20%.",),
    ) == "rule_passed"
    assert ReviewService.verify_report_claim(
        "The method caused growth.", ("The method correlated with growth.",),
    ) == "disabled"
