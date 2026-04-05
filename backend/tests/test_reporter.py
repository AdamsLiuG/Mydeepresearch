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

    def test_generate_report_supports_custom_sections_and_dynamic_order_in_flexible_mode(self):
        store = EvidenceStore()
        store.record_search_results(
            task_id=1,
            query="AI agent market",
            search_payload={
                "results": [
                    {
                        "title": "Source One",
                        "url": "https://example.com/source-one",
                        "content": "agent platform adoption is rising",
                    },
                    {
                        "title": "Source Two",
                        "url": "https://example.com/source-two",
                        "content": "vendors are bundling orchestration capabilities",
                    },
                ]
            },
            backend="duckduckgo",
        )
        task = TodoItem(
            id=1,
            title="市场观察",
            intent="梳理市场变化",
            query="AI agent market",
            status="completed",
            summary_payload={
                "key_findings": [
                    {"text": "AI agent 平台开始走向一体化交付", "source_ids": ["T1-S1"]},
                    {"text": "编排能力正在被主流产品捆绑提供", "source_ids": ["T1-S2"]},
                ],
                "evidence_gaps": [],
            },
            evidence_items=store.list_task_evidence(1),
        )

        state = SummaryState(research_topic="AI agent 市场格局", todo_items=[task])
        response = """
{
  "background_overview": "AI agent 市场正在从单点工具走向平台化整合。",
  "key_findings": [{"claim": "平台化交付成为主要方向", "source_ids": ["T1-S1"]}],
  "evidence_and_data": [{"point": "主流厂商开始捆绑编排能力与检索能力", "source_ids": ["T1-S2"]}],
  "risks_and_challenges": [{"risk": "平台锁定风险会随着能力整合而上升", "source_ids": ["T1-S1"]}],
  "custom_sections": [
    {
      "section_id": "market_landscape",
      "title": "市场格局",
      "content_type": "bullets",
      "items": [{"text": "产品形态正从插件堆叠转向端到端工作流", "source_ids": ["T1-S2"]}]
    }
  ],
  "section_order": ["background_overview", "market_landscape", "key_findings", "evidence_and_data", "risks_and_challenges"],
  "references": [
    {"source_id": "T1-S1", "title": "Source One", "url": "https://example.com/source-one"},
    {"source_id": "T1-S2", "title": "Source Two", "url": "https://example.com/source-two"}
  ]
}
"""
        service = ReportingService(
            DummyAgent(response),
            Configuration.from_env(load_env_file=False),
            evidence_store=store,
        )

        report = service.generate_report(state)

        self.assertIn("## 市场格局", report)
        self.assertLess(report.index("## 市场格局"), report.index("## 核心洞见"))
        self.assertIn("## 参考来源", report)
        self.assertIn("https://example.com/source-two", report)

    def test_generate_report_can_use_fixed_layout_backup_mode(self):
        store = EvidenceStore()
        store.record_search_results(
            task_id=1,
            query="AI agent market",
            search_payload={
                "results": [
                    {
                        "title": "Source One",
                        "url": "https://example.com/source-one",
                        "content": "agent platform adoption is rising",
                    }
                ]
            },
            backend="duckduckgo",
        )
        task = TodoItem(
            id=1,
            title="市场观察",
            intent="梳理市场变化",
            query="AI agent market",
            status="completed",
            summary_payload={
                "key_findings": [
                    {"text": "AI agent 平台开始走向一体化交付", "source_ids": ["T1-S1"]}
                ],
                "evidence_gaps": [],
            },
            evidence_items=store.list_task_evidence(1),
        )
        state = SummaryState(research_topic="AI agent 市场格局", todo_items=[task])
        response = """
{
  "background_overview": "AI agent 市场正在从单点工具走向平台化整合。",
  "key_findings": [{"claim": "平台化交付成为主要方向", "source_ids": ["T1-S1"]}],
  "evidence_and_data": [{"point": "主流厂商开始捆绑编排能力与检索能力", "source_ids": ["T1-S1"]}],
  "risks_and_challenges": [{"risk": "平台锁定风险会随着能力整合而上升", "source_ids": ["T1-S1"]}],
  "custom_sections": [
    {
      "section_id": "market_landscape",
      "title": "市场格局",
      "content_type": "bullets",
      "items": [{"text": "产品形态正从插件堆叠转向端到端工作流", "source_ids": ["T1-S1"]}]
    }
  ],
  "section_order": ["background_overview", "market_landscape", "key_findings", "evidence_and_data", "risks_and_challenges"],
  "references": [{"source_id": "T1-S1", "title": "Source One", "url": "https://example.com/source-one"}]
}
"""
        service = ReportingService(
            DummyAgent(response),
            Configuration.from_env(
                overrides={"report_layout_mode": "fixed"},
                load_env_file=False,
            ),
            evidence_store=store,
        )

        report = service.generate_report(state)

        self.assertNotIn("## 市场格局", report)
        self.assertLess(report.index("## 背景概览"), report.index("## 核心洞见"))
        self.assertLess(report.index("## 核心洞见"), report.index("## 证据与数据"))
        self.assertLess(report.index("## 证据与数据"), report.index("## 风险与挑战"))

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

    def test_generate_report_separates_process_notes_from_formal_sections(self):
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

        self.assertIn("## 研究过程说明（系统生成）", report)
        self.assertIn("本次研究仍有部分维度未达到理想证据覆盖", report)
        self.assertNotIn("审查提示：当前结果存在证据风险", report)
        self.assertIn("https://example.com/hydrated", report)
        self.assertNotIn("未知来源", report)
        self.assertNotIn("暂无链接", report)
        self.assertLess(report.index("## 风险与挑战"), report.index("## 研究过程说明（系统生成）"))
        self.assertLess(report.index("## 研究过程说明（系统生成）"), report.index("## 参考来源"))

    def test_generate_report_falls_back_to_task_grounding_when_model_returns_empty_sections(self):
        store = EvidenceStore()
        store.record_search_results(
            task_id=1,
            query="multimodal 2025",
            search_payload={
                "results": [
                    {
                        "title": "Source One",
                        "url": "https://example.com/one",
                        "content": "cross-modal alignment is improving",
                    },
                    {
                        "title": "Source Two",
                        "url": "https://example.com/two",
                        "content": "unified multimodal systems reduce switching costs",
                    },
                ]
            },
            backend="duckduckgo",
        )
        completed_task = TodoItem(
            id=1,
            title="核心突破",
            intent="梳理多模态突破",
            query="multimodal 2025 breakthrough",
            status="completed",
            summary_payload={
                "key_findings": [
                    {"text": "多模态模型实现了更稳定的跨模态语义对齐", "source_ids": ["T1-S1"]},
                    {"text": "统一模型架构降低了模态切换和集成成本", "source_ids": ["T1-S2"]},
                ],
                "evidence_gaps": [],
            },
            review_issues=[
                {
                    "task_id": 1,
                    "severity": "low",
                    "check": "low_quality_mix",
                    "message": "当前任务没有明显高质量来源，建议补充官方文档、论文或权威站点。",
                    "source_ids": ["T1-S1", "T1-S2"],
                    "origin": "rule",
                }
            ],
            review_status="warning",
        )
        completed_task.evidence_items = store.list_task_evidence(1)

        failed_task = TodoItem(
            id=2,
            title="产业生态影响",
            intent="补充产业视角",
            query="multimodal 2025 ecosystem",
            status="failed",
            summary="总结阶段失败，请参考已收集来源。",
            review_issues=[
                {
                    "task_id": 2,
                    "severity": "high",
                    "check": "missing_angle",
                    "message": "任务 2 未成功完成，可能导致该维度覆盖不足。",
                    "source_ids": [],
                    "origin": "rule",
                }
            ],
            review_status="blocked",
        )

        state = SummaryState(
            research_topic="探索多模态大模型在 2025 年的关键突破",
            todo_items=[completed_task, failed_task],
            review_summary={"overall_status": "blocked"},
        )

        response = """
{
  "background_overview": "暂无相关信息",
  "key_findings": [],
  "evidence_and_data": [],
  "risks_and_challenges": [],
  "references": []
}
"""
        service = ReportingService(
            DummyAgent(response),
            Configuration.from_env(load_env_file=False),
            evidence_store=store,
        )

        report = service.generate_report(state)

        self.assertIn("多模态模型实现了更稳定的跨模态语义对齐", report)
        self.assertIn("统一模型架构降低了模态切换和集成成本", report)
        self.assertIn("## 研究过程说明（系统生成）", report)
        self.assertIn("可信度仍有提升空间", report)
        self.assertNotIn("当前任务没有明显高质量来源", report)
        self.assertIn("Source One", report)
        self.assertNotIn("## 核心洞见\n- 暂无相关信息", report)
        self.assertNotIn("## 参考来源\n- 暂无可用来源", report)
        self.assertIn("## 风险与挑战\n- 暂无相关信息", report)
        self.assertLess(report.index("## 风险与挑战"), report.index("## 研究过程说明（系统生成）"))
        self.assertLess(report.index("## 研究过程说明（系统生成）"), report.index("## 参考来源"))

    def test_generate_report_strips_internal_review_language_from_formal_content(self):
        store = EvidenceStore()
        store.record_search_results(
            task_id=1,
            query="diffusion 2025",
            search_payload={
                "results": [
                    {
                        "title": "Recent Paper",
                        "url": "https://example.com/paper",
                        "content": "new diffusion architecture improves multimodal control",
                    }
                ]
            },
            backend="duckduckgo",
        )
        task = TodoItem(
            id=1,
            title="架构创新",
            intent="梳理扩散模型架构突破",
            query="diffusion architecture 2025",
            status="completed",
            summary_payload={
                "executive_summary": "2025 年扩散模型在可控生成和统一建模上继续推进。",
                "key_findings": [
                    {"text": "统一主干与更强条件控制让多模态扩散系统更易复用。", "source_ids": ["T1-S1"]},
                    {"text": "训练和推理链路被设计得更贴近生产部署。", "source_ids": ["T1-S1"]},
                ],
                "evidence_gaps": [],
            },
            evidence_items=store.list_task_evidence(1),
        )
        state = SummaryState(research_topic="探索扩散生成模型在 2025 年的关键突破", todo_items=[task])

        response = """
{
  "background_overview": "扩散生成模型在 2025 年继续推动图像、视频和多模态生成能力提升。需注意部分任务审查状态为blocked或warning，报告结论采取保守表述。",
  "key_findings": [{"claim": "更强的条件控制提升了复杂场景生成稳定性", "source_ids": ["T1-S1"]}],
  "evidence_and_data": [{"point": "统一架构减少了多阶段拼接带来的工程复杂度", "source_ids": ["T1-S1"]}],
  "risks_and_challenges": [
    {"risk": "高质量训练数据和算力门槛仍然限制扩散模型在长视频场景中的大规模落地", "source_ids": ["T1-S1"]},
    {"risk": "审查提示：当前结果存在证据风险，以下内容仅保留通过 source_id 校验的结论", "source_ids": ["T1-S1"]}
  ],
  "references": [{"source_id": "T1-S1", "title": "Recent Paper", "url": "https://example.com/paper"}]
}
"""
        service = ReportingService(
            DummyAgent(response),
            Configuration.from_env(load_env_file=False),
            evidence_store=store,
        )

        report = service.generate_report(state)

        self.assertIn("扩散生成模型在 2025 年继续推动图像、视频和多模态生成能力提升", report)
        self.assertNotIn("需注意部分任务审查状态为blocked或warning", report)
        self.assertIn("高质量训练数据和算力门槛仍然限制扩散模型在长视频场景中的大规模落地", report)
        self.assertNotIn("审查提示：当前结果存在证据风险", report)


if __name__ == "__main__":
    unittest.main()
