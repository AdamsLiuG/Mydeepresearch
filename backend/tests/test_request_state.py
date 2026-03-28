import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from services.request_state import RequestStateStore


class RequestStateStoreTests(unittest.TestCase):
    def test_save_load_and_list_recent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RequestStateStore(temp_dir, recent_limit=10)
            store.save(
                "req-1",
                {
                    "topic": "AI agent",
                    "status": "in_progress",
                    "phase": "review",
                    "todo_items": [{"id": 1, "title": "任务1"}],
                    "report_markdown": "",
                },
            )
            store.save(
                "req-2",
                {
                    "topic": "MCP",
                    "status": "success",
                    "phase": "completed",
                    "todo_items": [{"id": 2, "title": "任务2"}],
                    "report_markdown": "# report",
                },
            )

            loaded = store.load("req-1")
            recent = store.list_recent(limit=2)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["topic"], "AI agent")
        self.assertEqual(len(recent), 2)
        recent_by_id = {item["request_id"]: item for item in recent}
        self.assertIn("req-1", recent_by_id)
        self.assertIn("req-2", recent_by_id)
        self.assertTrue(recent_by_id["req-2"]["can_view_content"])
        self.assertFalse(recent_by_id["req-2"]["can_resume"])
        self.assertTrue(recent_by_id["req-1"]["can_resume"])


if __name__ == "__main__":
    unittest.main()
