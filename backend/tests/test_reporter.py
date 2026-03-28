import sys
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from models import SummaryState, TodoItem
from services.evidence import EvidenceStore
from services.reporter import ReportingService


class DummyAgent:
    def __init__(self, response: str):
        self._response = response

    def run(self, prompt: str):
        return self._response

    def clear_history(self):
        return None


class ReportingServiceTests(unittest.TestCase):
    def test_generate_report_renders_grounded_markdown_from_structured_json(self):
        store = EvidenceStore()
        store.record_search_results(
            task_id=1,
            query="AI agent",
            search_payload={
                "results": [
                    {
                        "title": "Grounded Source",
                        "url": "https://example.com/source",
                        "content": "example snippet",
                    }
                ]
            },
            backend="duckduckgo",
        )
        task = TodoItem(
            id=1,
            title="任务1",
            intent="梳理背景",
            query="AI agent",
            status="completed",
            summary="任务总结\n- 研究流程依赖 search 工具 [T1-S1]",
            sources_summary="* [T1-S1] Grounded Source : https://example.com/source",
        )
        task.evidence_items = store.list_task_evidence(1)

        state = SummaryState(research_topic="AI agent", todo_items=[task])
        response = """
{
  "background_overview": "AI agent 在开放研究场景中需要可追溯证据。",
  "key_findings": [{"claim": "多阶段研究流程可以提升任务可控性", "source_ids": ["T1-S1"]}],
  "evidence_and_data": [{"point": "搜索结果被登记为 source_id 以支持引用绑定", "source_ids": ["T1-S1"]}],
  "risks_and_challenges": [{"risk": "如果来源不足，报告应明确证据不足", "source_ids": ["T1-S1"]}],
  "references": [{"source_id": "T1-S1", "title": "Grounded Source", "url": "https://example.com/source"}]
}
"""
        service = ReportingService(
            DummyAgent(response),
            Configuration.from_env(load_env_file=False),
            evidence_store=store,
        )

        report = service.generate_report(state)

        self.assertIn("## 核心洞见", report)
        self.assertIn("[T1-S1]", report)
        self.assertIn("## 参考来源", report)
        self.assertIn("Grounded Source", report)
        self.assertIn("https://example.com/source", report)

    def test_generate_report_filters_invalid_source_ids_and_ignores_model_reference_overrides(self):
        store = EvidenceStore()
        store.record_search_results(
            task_id=1,
            query="resume",
            search_payload={
                "results": [
                    {
                        "title": "Recovered Source",
                        "url": "https://example.com/recovered",
                        "content": "resume snapshot content",
                    }
                ]
            },
            backend="duckduckgo",
        )
        task = TodoItem(
            id=1,
            title="任务1",
            intent="恢复引用",
            query="resume snapshot",
            status="completed",
            summary="任务总结\n- 恢复后的引用应绑定到真实 evidence [T1-S1]",
            summary_payload={
                "key_findings": [
                    {"text": "恢复后的引用应绑定到真实 evidence", "source_ids": ["T1-S1"]}
                ],
                "evidence_gaps": [],
            },
        )
        task.evidence_items = store.list_task_evidence(1)
        state = SummaryState(research_topic="resume", todo_items=[task])

        response = """
{
  "background_overview": "resume path should preserve evidence grounding",
  "key_findings": [
    {"claim": "合法结论", "source_ids": ["T1-S1"]},
    {"claim": "非法结论", "source_ids": ["T1-S9"]}
  ],
  "evidence_and_data": [],
  "risks_and_challenges": [],
  "references": [
    {"source_id": "T1-S1", "title": "Fake Model Title", "url": "https://fake.example.com"},
    {"source_id": "T1-S9", "title": "Fake Missing Source", "url": "https://fake.example.com/missing"}
  ]
}
"""
        service = ReportingService(
            DummyAgent(response),
            Configuration.from_env(load_env_file=False),
            evidence_store=store,
        )

        report = service.generate_report(state)

        self.assertIn("合法结论", report)
        self.assertNotIn("非法结论", report)
        self.assertNotIn("T1-S9", report)
        self.assertIn("Recovered Source", report)
        self.assertNotIn("Fake Model Title", report)

    def test_generate_report_adds_blocked_warning_and_avoids_placeholder_references(self):
        store = EvidenceStore()
        store.hydrate_from_tasks(
            [
                TodoItem(
                    id=2,
                    title="任务2",
                    intent="恢复来源",
                    query="hydrate",
                    evidence_items=[
                        {
                            "source_id": "T2-S2",
                            "title": "",
                            "url": "https://example.com/hydrated",
                            "snippet": "hydrated from snapshot",
                            "domain": "example.com",
                            "source_type": "web",
                            "quality_label": "medium",
                            "freshness_label": "recent",
                        }
                    ],
                )
            ]
        )
        task = TodoItem(
            id=2,
            title="任务2",
            intent="恢复来源",
            query="hydrate",
            status="completed",
            summary_payload={
                "key_findings": [
                    {"text": "resume 后的引用映射必须可恢复", "source_ids": ["T2-S2"]}
                ],
                "evidence_gaps": [],
            },
            evidence_items=store.list_task_evidence(2),
        )
        state = SummaryState(
            research_topic="resume",
            todo_items=[task],
            review_summary={"overall_status": "blocked"},
        )

        response = """
{
  "background_overview": "resume report",
  "key_findings": [{"claim": "引用映射已恢复", "source_ids": ["T2-S2"]}],
  "evidence_and_data": [],
  "risks_and_challenges": [],
  "references": [{"source_id": "T2-S2", "title": "ignored", "url": "https://ignored.example.com"}]
}
"""
        service = ReportingService(
            DummyAgent(response),
            Configuration.from_env(load_env_file=False),
            evidence_store=store,
        )

        report = service.generate_report(state)

        self.assertIn("审查提示：当前结果存在证据风险", report)
        self.assertIn("https://example.com/hydrated", report)
        self.assertNotIn("未知来源", report)
        self.assertNotIn("暂无链接", report)


if __name__ == "__main__":
    unittest.main()
