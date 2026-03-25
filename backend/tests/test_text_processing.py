import sys
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from services.text_processing import normalize_agent_markdown


class TextProcessingTests(unittest.TestCase):
    def test_normalize_agent_markdown_unwraps_embedded_json_content(self):
        raw = (
            '{"content":"## 任务概览\\\\n- 任务主题：探索多模态大模型在2025年的关键突破'
            '\\\\n- 任务名称：并行检索\\\\n\\\\n## 任务总结\\\\n- 关键发现：统一多模态框架"}'
        )

        cleaned = normalize_agent_markdown(raw)

        self.assertIn("## 任务概览", cleaned)
        self.assertIn("- 任务主题：探索多模态大模型在2025年的关键突破", cleaned)
        self.assertIn("## 任务总结", cleaned)
        self.assertNotIn("\\n", cleaned)

    def test_normalize_agent_markdown_removes_tool_calls_and_outer_wrappers(self):
        raw = (
            '[TOOL_CALL:note:{"action":"update","note_id":"note_1"}]\n'
            '":"## 任务概览\\\\n- 任务名称：技术发展脉络"'
        )

        cleaned = normalize_agent_markdown(raw)

        self.assertTrue(cleaned.startswith("## 任务概览"))
        self.assertIn("任务名称：技术发展脉络", cleaned)
        self.assertNotIn("TOOL_CALL", cleaned)


if __name__ == "__main__":
    unittest.main()
