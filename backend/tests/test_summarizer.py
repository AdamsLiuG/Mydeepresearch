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


if __name__ == "__main__":
    unittest.main()
