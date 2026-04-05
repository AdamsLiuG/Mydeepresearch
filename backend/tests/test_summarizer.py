import sys
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from models import SummaryState, TodoItem
from services.summarizer import SummarizationService


class DummyAgent:
    def __init__(self, *, response: str = "", stream_chunks: list[str] | None = None):
        self._response = response
        self._stream_chunks = list(stream_chunks or [])

    def run(self, prompt: str) -> str:
        return self._response

    def stream_run(self, prompt: str):
        for chunk in self._stream_chunks:
            yield chunk

    def clear_history(self):
        return None


class SummarizationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Configuration.from_env(load_env_file=False)
        self.state = SummaryState(research_topic="Deep research agent")
        self.task = TodoItem(
            id=1,
            title="任务1",
            intent="梳理系统设计",
            query="Deep research agent workflow",
            evidence_items=[
                {
                    "source_id": "T1-S1",
                    "title": "Official docs",
                    "url": "https://example.com/docs",
                    "snippet": "source aware grounding",
                },
                {
                    "source_id": "T1-S2",
                    "title": "Engineering blog",
                    "url": "https://example.com/blog",
                    "snippet": "resume and checkpoint",
                },
            ],
        )

    def test_summarize_task_drops_meta_reasoning_and_keeps_grounded_findings(self):
        response = (
            "好的，我现在需要先调用 evidence_lookup 查看来源。\n"
            "- 系统通过证据绑定机制保持结论可追溯 [T1-S1]\n"
            "- 仍需补充更高质量官方来源\n"
        )
        service = SummarizationService(
            lambda: DummyAgent(response=response),
            self.config,
        )

        result = service.summarize_task(self.state, self.task, "context")

        self.assertNotIn("我现在需要", result.markdown)
        self.assertEqual(
            result.payload["key_findings"],
            [{"text": "系统通过证据绑定机制保持结论可追溯", "source_ids": ["T1-S1"]}],
        )
        self.assertEqual(result.payload["evidence_gaps"], ["仍需补充更高质量官方来源"])
        self.assertEqual(result.claims[0]["support_status"], "unreviewed")

    def test_summarize_task_rejects_reasoning_only_output_without_grounded_findings(self):
        response = (
            "好的，我现在需要先调用 evidence_lookup。\n"
            "接下来我会根据用户要求继续搜索。\n"
            "- 最后，确保所有source_id正确引用，没有遗漏，并且格式正确，如[T1-S1]或[T1-S2]。同时，避免使用工具调用指令，只输出最终总结。\n"
            "- 仍需补充高质量来源\n"
        )
        service = SummarizationService(
            lambda: DummyAgent(response=response),
            self.config,
        )

        with self.assertRaisesRegex(ValueError, "no grounded findings"):
            service.summarize_task(self.state, self.task, "context")

    def test_stream_task_summary_emits_sanitized_chunks_after_buffering(self):
        chunks = [
            "好的，我现在需要先调用 evidence_lookup。",
            '\n{"key_findings":[{"text":"官方文档确认结论必须绑定到可追溯证据","source_ids":["T1-S1"]}],"evidence_gaps":[]}',
        ]
        service = SummarizationService(
            lambda: DummyAgent(stream_chunks=chunks),
            self.config,
        )

        summary_stream, finalize = service.stream_task_summary(self.state, self.task, "context")
        emitted = "".join(summary_stream)
        result = finalize()

        self.assertNotIn("我现在需要", emitted)
        self.assertIn("官方文档确认结论必须绑定到可追溯证据", emitted)
        self.assertEqual(result.payload["key_findings"][0]["source_ids"], ["T1-S1"])

    def test_stream_task_summary_does_not_emit_reasoning_only_content(self):
        chunks = [
            "好的，我现在需要先调用 evidence_lookup。",
            "\n接下来我会继续检索并整理答案。",
        ]
        service = SummarizationService(
            lambda: DummyAgent(stream_chunks=chunks),
            self.config,
        )

        summary_stream, finalize = service.stream_task_summary(self.state, self.task, "context")
        with self.assertRaisesRegex(ValueError, "no grounded findings"):
            list(summary_stream)
        with self.assertRaisesRegex(ValueError, "no grounded findings"):
            finalize()

    def test_summarize_task_rewrites_contaminated_executive_summary(self):
        response = """
{
  "executive_summary": "扩散模型在可控生成上继续推进。需注意部分任务审查状态为blocked或warning，报告结论采取保守表述。",
  "key_findings": [
    {"text": "更强的条件控制让复杂场景生成更稳定", "source_ids": ["T1-S1"]},
    {"text": "统一建模降低了多模态扩展时的系统割裂", "source_ids": ["T1-S2"]}
  ],
  "evidence_gaps": ["仍需补充更近期的官方评测或论文"]
}
"""
        service = SummarizationService(
            lambda: DummyAgent(response=response),
            self.config,
        )

        result = service.summarize_task(self.state, self.task, "context")

        self.assertIn("## 任务概述", result.markdown)
        self.assertNotIn("审查状态为blocked或warning", result.markdown)
        self.assertTrue(result.payload["executive_summary"].startswith("扩散模型在可控生成上继续推进"))
        self.assertIn("更强的条件控制让复杂场景生成更稳定", result.markdown)

    def test_build_prompt_includes_historical_memory_guardrails(self):
        service = SummarizationService(
            lambda: DummyAgent(response='{"key_findings":[{"text":"结论","source_ids":["T1-S1"]}],"evidence_gaps":[]}'),
            self.config,
        )

        prompt = service._build_prompt(
            self.state,
            self.task,
            "context",
            historical_memory_context="历史研究记忆：过去任务常见遗漏是缺少官方文档。",
        )

        self.assertIn("历史研究记忆", prompt)
        self.assertIn("不能作为当前任务的 `source_id` 证据", prompt)


if __name__ == "__main__":
    unittest.main()
