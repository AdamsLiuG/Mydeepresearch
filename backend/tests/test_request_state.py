import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from services.request_state import RequestStateStore


class RequestStateStoreTests(unittest.TestCase):
    def test_budget_write_cannot_revert_newer_graph_and_preserves_other_v4_fields(self):
        from types import SimpleNamespace

        from models import SummaryState
        from services.request_state import SnapshotConflictError
        from state_manager import StateManagerMixin

        with tempfile.TemporaryDirectory() as temp_dir:
            store = RequestStateStore(temp_dir)
            stale = store.save("v4-budget", {"v4": {"schema_version": 1, "replan": {"dag_revision": 1}}})
            latest = store.save("v4-budget", {"v4": {
                "schema_version": 1, "replan": {"dag_revision": 2}, "future_extension": {"value": True},
            }}, expected_version=stale["state_version"])
            manager = StateManagerMixin()
            manager._request_state_store = store
            manager.request_id = "v4-budget"
            manager.tenant_id = "default"
            state = SummaryState(v4=stale["v4"], state_version=stale["state_version"])
            budget = {"snapshot": {"spent": {"search_calls": 1}}}
            with self.assertRaises(SnapshotConflictError):
                manager._persist_v4_budget(state, budget)
            state.state_version = latest["state_version"]
            manager._persist_v4_budget(state, budget)
            persisted = SimpleNamespace(**store.load("v4-budget")["v4"])
            self.assertEqual(persisted.replan["dag_revision"], 2)
            self.assertEqual(persisted.future_extension, {"value": True})
            self.assertEqual(persisted.budget, budget)

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
                    "cache_diagnostics": {
                        "cache_hits": 3,
                        "cache_exact_hits": 1,
                        "cache_semantic_hits": 2,
                        "cache_misses": 4,
                        "last_search_cache_details": {
                            "cache_hit_mode": "semantic_ann",
                            "ttl_bucket": "evergreen",
                        },
                    },
                    "todo_items": [{"id": 2, "title": "任务2"}],
                    "report_markdown": "# report",
                },
            )
            store.save(
                "req-3",
                {
                    "topic": "MCP connectivity",
                    "status": "failed",
                    "phase": "failed",
                    "request_metrics": {
                        "error": "Connection error.",
                    },
                },
            )

            loaded = store.load("req-1")
            loaded_failed = store.load("req-3")
            recent = store.list_recent(limit=3)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["topic"], "AI agent")
        self.assertIsNotNone(loaded_failed)
        self.assertEqual(loaded_failed["error"], "Connection error.")
        self.assertEqual(len(recent), 3)
        recent_by_id = {item["request_id"]: item for item in recent}
        self.assertIn("req-1", recent_by_id)
        self.assertIn("req-2", recent_by_id)
        self.assertIn("req-3", recent_by_id)
        self.assertTrue(recent_by_id["req-2"]["can_view_content"])
        self.assertFalse(recent_by_id["req-2"]["can_resume"])
        self.assertTrue(recent_by_id["req-1"]["can_resume"])
        self.assertTrue(recent_by_id["req-3"]["can_view_content"])
        self.assertEqual(recent_by_id["req-3"]["error"], "Connection error.")
        self.assertEqual(recent_by_id["req-2"]["cache_diagnostics"]["cache_hits"], 3)
        self.assertEqual(
            recent_by_id["req-2"]["cache_diagnostics"]["last_search_cache_details"]["cache_hit_mode"],
            "semantic_ann",
        )

    def test_save_allows_concurrent_updates_for_same_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RequestStateStore(temp_dir, recent_limit=10)

            def save_snapshot(index: int):
                return store.save(
                    "req-1",
                    {
                        "topic": "AI agent",
                        "status": "in_progress",
                        "phase": "task_execution",
                        "todo_items": [{"id": index, "title": f"任务{index}"}],
                    },
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(save_snapshot, range(20)))

            loaded = store.load("req-1")

        self.assertEqual(len(results), 20)
        self.assertTrue(all(item["request_id"] == "req-1" for item in results))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["request_id"], "req-1")
        self.assertEqual(loaded["topic"], "AI agent")


if __name__ == "__main__":
    unittest.main()
