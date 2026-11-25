import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from metrics import RequestTrace
from services.strategy_memory import StrategyMemoryService
from services.strategy_synthesizer import (
    MATURITY_ARCHIVED,
    MATURITY_DRAFT,
    MATURITY_ORDER,
    MATURITY_PROMOTED,
    MATURITY_VERIFIED,
    maturity_meets_gate,
    normalize_maturity,
)


class ArchivedGradeTests(unittest.TestCase):
    """Archived must fall below every configurable gate so no extra filter is required."""

    def test_archived_ranks_below_draft(self):
        self.assertLess(MATURITY_ORDER[MATURITY_ARCHIVED], MATURITY_ORDER[MATURITY_DRAFT])

    def test_archived_fails_the_lowest_gate(self):
        self.assertFalse(maturity_meets_gate(MATURITY_ARCHIVED, MATURITY_DRAFT))

    def test_archived_is_a_recognised_value(self):
        self.assertEqual(normalize_maturity("Archived"), MATURITY_ARCHIVED)


class _RecordingCollection:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, *, ids, metadatas):
        self.updates.append({"ids": list(ids), "metadatas": list(metadatas)})


class StrategyUsageTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _service(self, *, idle_days: int = 0, enabled: bool = True) -> StrategyMemoryService:
        config = Configuration.from_env(
            overrides={
                "strategy_memory_enabled": enabled,
                "request_state_enabled": enabled,
                "request_state_dir": str(self.memory_dir / "state"),
                "strategy_memory_dir": str(self.memory_dir),
                "memory_decay_idle_days": idle_days,
            },
            load_env_file=False,
        )
        service = StrategyMemoryService(config)
        service._collection = _RecordingCollection()
        self._svc = service
        return service

    def _write_card(
        self,
        strategy_id: str,
        *,
        maturity: str = MATURITY_DRAFT,
        created_at: str | None = None,
        last_hit_at: str = "",
        hit_count: int = 0,
    ) -> Path:
        # Written through the service's own path helper so the test never re-encodes the
        # card file naming convention.
        path = self._svc._card_path(strategy_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy_id": strategy_id,
            "strategy_kind": "planning_pattern",
            "stage_scope": "planning",
            "title": f"卡片 {strategy_id}",
            "applicable_when": "适用场景",
            "recommended_actions": ["动作"],
            "origin_request_id": strategy_id.split("::")[0],
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "maturity": maturity,
            "topic_domain": "general",
            "hit_count": hit_count,
            "last_hit_at": last_hit_at,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _load(self, service: StrategyMemoryService, strategy_id: str) -> dict:
        return json.loads(service._card_path(strategy_id).read_text(encoding="utf-8"))


class HitRecordingTests(StrategyUsageTestCase):
    def test_hit_bumps_counter_and_stamps_time(self):
        service = self._service()
        self._write_card("req-a::planning_pattern")

        self.assertEqual(service.record_card_hits(["req-a::planning_pattern"]), 1)

        payload = self._load(service, "req-a::planning_pattern")
        self.assertEqual(payload["hit_count"], 1)
        self.assertTrue(payload["last_hit_at"])

    def test_repeated_hits_accumulate_across_requests(self):
        service = self._service()
        self._write_card("req-a::planning_pattern", hit_count=4)

        service.record_card_hits(["req-a::planning_pattern"])

        self.assertEqual(self._load(service, "req-a::planning_pattern")["hit_count"], 5)

    def test_duplicate_ids_in_one_request_count_once(self):
        service = self._service()
        self._write_card("req-a::planning_pattern")

        service.record_card_hits(["req-a::planning_pattern", "req-a::planning_pattern"])

        self.assertEqual(self._load(service, "req-a::planning_pattern")["hit_count"], 1)

    def test_unknown_ids_are_ignored(self):
        service = self._service()

        self.assertEqual(service.record_card_hits(["missing::planning_pattern"]), 0)

    def test_corrupt_counter_is_repaired(self):
        service = self._service()
        path = self._write_card("req-a::planning_pattern")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["hit_count"] = "not-a-number"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        service.record_card_hits(["req-a::planning_pattern"])

        self.assertEqual(self._load(service, "req-a::planning_pattern")["hit_count"], 1)


class TraceHitCollectionTests(unittest.TestCase):
    """Retrieval only accumulates ids in the trace; nothing is written in the query path."""

    def _trace(self) -> RequestTrace:
        return RequestTrace(
            request_id="req-1",
            topic="话题",
            search_api="duckduckgo",
            provider="openai",
            model="gpt-test",
            pricing_catalog=None,
        )

    def test_hit_ids_are_collected(self):
        trace = self._trace()

        trace.record_strategy_memory_query(
            hit_count=2,
            match_kinds=["planning_pattern"],
            strategy_ids=["req-a::planning_pattern", "req-b::anti_pattern"],
        )

        self.assertEqual(
            trace.strategy_memory_hit_ids,
            ["req-a::planning_pattern", "req-b::anti_pattern"],
        )

    def test_ids_are_deduplicated_across_stages(self):
        trace = self._trace()

        trace.record_strategy_memory_query(hit_count=1, strategy_ids=["req-a::planning_pattern"])
        trace.record_strategy_memory_query(hit_count=1, strategy_ids=["req-a::planning_pattern"])

        self.assertEqual(trace.strategy_memory_hit_ids, ["req-a::planning_pattern"])

    def test_strategy_ids_are_optional(self):
        trace = self._trace()

        trace.record_strategy_memory_query(hit_count=0)

        self.assertEqual(trace.strategy_memory_hit_ids, [])

    def test_hit_ids_appear_in_snapshot(self):
        trace = self._trace()
        trace.record_strategy_memory_query(hit_count=1, strategy_ids=["req-a::planning_pattern"])

        snapshot = trace.snapshot()

        self.assertEqual(
            snapshot["strategy_memory_hit_ids"],
            ["req-a::planning_pattern"],
        )


class DecayTests(StrategyUsageTestCase):
    def _stale(self, days: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def test_decay_is_disabled_by_default(self):
        service = self._service()
        self._write_card("req-a::planning_pattern", created_at=self._stale(400))

        self.assertEqual(service.archive_stale_cards(), [])
        self.assertEqual(
            self._load(service, "req-a::planning_pattern")["maturity"],
            MATURITY_DRAFT,
        )

    def test_idle_draft_is_archived(self):
        service = self._service(idle_days=30)
        self._write_card("req-a::planning_pattern", created_at=self._stale(90))

        self.assertEqual(service.archive_stale_cards(), ["req-a::planning_pattern"])
        self.assertEqual(
            self._load(service, "req-a::planning_pattern")["maturity"],
            MATURITY_ARCHIVED,
        )

    def test_recent_draft_survives(self):
        service = self._service(idle_days=30)
        self._write_card("req-a::planning_pattern", created_at=self._stale(3))

        self.assertEqual(service.archive_stale_cards(), [])

    def test_recent_hit_protects_an_old_card(self):
        service = self._service(idle_days=30)
        self._write_card(
            "req-a::planning_pattern",
            created_at=self._stale(400),
            last_hit_at=self._stale(2),
            hit_count=7,
        )

        self.assertEqual(service.archive_stale_cards(), [])

    def test_verified_and_promoted_never_decay(self):
        service = self._service(idle_days=30)
        self._write_card(
            "req-a::planning_pattern", maturity=MATURITY_VERIFIED, created_at=self._stale(400)
        )
        self._write_card(
            "req-b::planning_pattern", maturity=MATURITY_PROMOTED, created_at=self._stale(400)
        )

        self.assertEqual(service.archive_stale_cards(), [])

    def test_archived_card_is_excluded_from_retrieval_gate(self):
        service = self._service(idle_days=30)
        self._write_card("req-a::planning_pattern", created_at=self._stale(90))
        service.archive_stale_cards()

        maturity = self._load(service, "req-a::planning_pattern")["maturity"]

        self.assertFalse(maturity_meets_gate(maturity, service._config.memory_maturity_gate))

    def test_archiving_also_updates_the_vector_store(self):
        service = self._service(idle_days=30)
        self._write_card("req-a::planning_pattern", created_at=self._stale(90))

        service.archive_stale_cards()

        self.assertEqual(len(service._collection.updates), 1)
        self.assertEqual(
            service._collection.updates[0]["metadatas"][0]["maturity"],
            MATURITY_ARCHIVED,
        )


class MaintenanceTests(StrategyUsageTestCase):
    def test_partition_without_scope_never_updates_legacy_card_usage_or_decay(self):
        service = self._service(idle_days=3)
        strategy_id = "req-1::planning_pattern"
        path = self._write_card(
            strategy_id,
            created_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        )
        original = path.read_bytes()
        service._config = service._config.model_copy(update={"v4_tenant_memory_partition_enabled": True})

        self.assertEqual(service.record_card_hits([strategy_id]), 0)
        self.assertEqual(service.archive_stale_cards(), [])
        self.assertEqual(service.run_maintenance([strategy_id]), {"recorded_hits": 0, "archived": []})
        self.assertEqual(path.read_bytes(), original)

    def test_maintenance_records_hits_and_archives_in_one_pass(self):
        service = self._service(idle_days=30)
        self._write_card("req-a::planning_pattern")
        self._write_card(
            "req-b::planning_pattern",
            created_at=(datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),
        )

        result = service.run_maintenance(["req-a::planning_pattern"])

        self.assertEqual(result["recorded_hits"], 1)
        self.assertEqual(result["archived"], ["req-b::planning_pattern"])

    def test_a_card_used_this_request_is_not_archived_in_the_same_pass(self):
        service = self._service(idle_days=30)
        self._write_card(
            "req-a::planning_pattern",
            created_at=(datetime.now(timezone.utc) - timedelta(days=400)).isoformat(),
        )

        result = service.run_maintenance(["req-a::planning_pattern"])

        self.assertEqual(result["archived"], [])
        self.assertEqual(
            self._load(service, "req-a::planning_pattern")["maturity"],
            MATURITY_DRAFT,
        )

    def test_maintenance_is_inert_when_memory_is_disabled(self):
        service = self._service(idle_days=30, enabled=False)
        self._write_card(
            "req-a::planning_pattern",
            created_at=(datetime.now(timezone.utc) - timedelta(days=400)).isoformat(),
        )

        result = service.run_maintenance(["req-a::planning_pattern"])

        self.assertEqual(result, {"recorded_hits": 0, "archived": []})


if __name__ == "__main__":
    unittest.main()
