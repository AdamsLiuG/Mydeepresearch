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
    def test_web_evidence_write_updates_only_unfinished_task_under_snapshot_cas(self):
        from models import SummaryState
        from services.request_state import SnapshotConflictError
        from state_manager import StateManagerMixin

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManagerMixin()
            manager._request_state_store = RequestStateStore(temp_dir)
            manager.request_id = "web-evidence"
            manager._web_budget = object()
            manager._is_cancelled = lambda: False
            existing = {"source_id": "T1-S1", "full_content": "Published content"}
            current = manager._request_state_store.save(manager.request_id, {"todo_items": [
                {"id": 1, "status": "completed", "evidence_items": [existing]},
                {"id": 2, "status": "in_progress", "title": "Second"},
            ]})
            state = SummaryState(state_version=current["state_version"])
            manager._persist_web_evidence(state, 1, [])
            record = {"source_id": "T2-S1", "full_content": "Sanitized content"}
            manager._persist_web_evidence(state, 2, [record])
            saved = manager._request_state_store.load(manager.request_id)
            self.assertEqual(saved["todo_items"][0]["evidence_items"], [existing])
            self.assertEqual(saved["todo_items"][1]["title"], "Second")
            self.assertEqual(saved["todo_items"][1]["evidence_items"], [record])
            stale = SummaryState(state_version=current["state_version"])
            with self.assertRaises(SnapshotConflictError):
                manager._persist_web_evidence(stale, 2, [])

    def test_dynamic_marker_and_dispatch_are_absent_when_dispatch_cas_fails(self):
        from config import Configuration
        from models import SummaryState
        from services.budget import BudgetGovernor, build_request_budget
        from services.request_state import SnapshotConflictError
        from state_manager import (
            RequestBudgetController,
            StateManagerMixin,
            freeze_web_extraction,
        )

        class FailDispatchStore(RequestStateStore):
            def save(self, request_id, payload, **kwargs):
                ledger = payload.get("v4", {}).get("budget", {}).get("ledger", {})
                if any(record["state"] == "dispatched" for record in ledger.get("active", {}).values()):
                    raise SnapshotConflictError("dispatch commit rejected")
                return super().save(request_id, payload, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManagerMixin()
            manager._request_state_store = FailDispatchStore(temp_dir)
            manager.request_id = "web-cas"
            manager._is_cancelled = lambda: False
            config = Configuration(v4_web_extractor_enabled=True, v4_web_extractor_tenants=["default"])
            v4 = {"schema_version": 1, "web_extraction": freeze_web_extraction(config, "default")}
            initial = manager._request_state_store.save(manager.request_id, {"v4": v4})
            state = SummaryState(v4=initial["v4"], state_version=initial["state_version"])
            budget = build_request_budget("quick")
            governor = BudgetGovernor(budget, request_id=manager.request_id)
            controller = RequestBudgetController(manager, state, governor)

            with self.assertRaisesRegex(SnapshotConflictError, "dispatch commit rejected"):
                with controller.attempt(
                    {"dynamic_fetches": 1}, required=False, dynamic_urls=("https://example.org/doc",),
                ):
                    self.fail("A rejected dispatch cannot open a transport")
            self.assertEqual(governor.external_attempt_count, 0)
            persisted = manager._request_state_store.load(manager.request_id)["v4"]
            self.assertEqual(persisted["web_extraction"]["dynamic_targets"], {})
            recovered = BudgetGovernor(budget, request_id=manager.request_id)
            recovered.restore_ledger(persisted["budget"]["ledger"])
            self.assertEqual(recovered.snapshot().remaining["dynamic_fetches"], 1)

    def test_dynamic_dispatch_and_alias_survive_recovery(self):
        from config import Configuration
        from models import SummaryState
        from services.budget import BudgetGovernor, build_request_budget
        from state_manager import (
            BudgetDispatchDenied,
            RequestBudgetController,
            StateManagerMixin,
            freeze_web_extraction,
        )

        config = Configuration(v4_web_extractor_enabled=True, v4_web_extractor_tenants=["default"])
        budget = build_request_budget("quick")
        url = "https://example.org/web"
        redirected = "https://example.org/redirect"

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManagerMixin()
            manager._request_state_store = RequestStateStore(temp_dir)
            manager.request_id = "web-budget"
            manager._is_cancelled = lambda: False
            v4 = {"schema_version": 1, "web_extraction": freeze_web_extraction(config, "default")}
            saved = manager._request_state_store.save(manager.request_id, {"v4": v4})
            state = SummaryState(v4=saved["v4"], state_version=saved["state_version"])
            governor = BudgetGovernor(budget, request_id=manager.request_id)
            controller = RequestBudgetController(manager, state, governor)
            operation = controller.attempt(
                {"web_fetches": 1, "dynamic_fetches": 1, "cost_microusd": 10},
                required=False, dynamic_urls=(url,),
            )
            attempt = operation.__enter__()
            try:
                self.assertTrue(attempt.associate_dynamic_url(redirected))
                committed = manager._request_state_store.load(manager.request_id)
                self.assertNotIn(url, str(committed))
                self.assertNotIn(redirected, str(committed))
                recovered_governor = BudgetGovernor(budget, request_id=manager.request_id)
                recovered_governor.restore_ledger(committed["v4"]["budget"]["ledger"])
                recovered_state = SummaryState(v4=committed["v4"], state_version=committed["state_version"])
                recovered = RequestBudgetController(manager, recovered_state, recovered_governor)
                spent = recovered.snapshot().spent
                self.assertEqual((spent["web_fetches"], spent["dynamic_fetches"], spent["cost_microusd"]), (1, 1, 10))
                self.assertEqual(recovered.snapshot().external_attempts, 2)
                for target in (url, redirected):
                    with self.assertRaisesRegex(BudgetDispatchDenied, "not_eligible"):
                        with recovered.attempt({"dynamic_fetches": 1}, required=False, dynamic_urls=(target,)):
                            self.fail("Recovered target was dispatched twice")
            finally:
                operation.__exit__(None, None, None)

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
