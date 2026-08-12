import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

hello_agents_stub = sys.modules.get("hello_agents") or types.ModuleType("hello_agents")


class DummyToolAwareSimpleAgent:
    pass


hello_agents_stub.ToolAwareSimpleAgent = DummyToolAwareSimpleAgent
sys.modules.setdefault("hello_agents", hello_agents_stub)

from config import Configuration
from services.strategy_memory import StrategyHit, StrategyMemoryService
from services.strategy_synthesizer import (
    MATURITY_DRAFT,
    MATURITY_PROMOTED,
    MATURITY_VERIFIED,
    StrategyCard,
    maturity_meets_gate,
    normalize_maturity,
)


def _card(strategy_id: str = "req-1::planning_pattern", **overrides) -> StrategyCard:
    payload = {
        "strategy_id": strategy_id,
        "strategy_kind": "planning_pattern",
        "stage_scope": "planning",
        "title": "拆解模式",
        "applicable_when": "当主题涉及协议时",
        "match_signals": ["协议"],
        "recommended_actions": ["先查官方文档"],
        "query_templates": ["<主题> 官方规范"],
        "preferred_sources": ["官方站点"],
        "pitfalls_to_avoid": ["只看博客"],
        "origin_request_id": "req-1",
        "origin_status": "success",
        "origin_review_status": "pass",
        "origin_task_ids": [1],
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return StrategyCard(**payload)


def _service(tmp_dir: str, *, gate: str = "draft") -> StrategyMemoryService:
    config = Configuration.from_env(
        overrides={"strategy_memory_dir": tmp_dir, "memory_maturity_gate": gate},
        load_env_file=False,
    )
    return StrategyMemoryService(config)


class MaturityNormalizationTests(unittest.TestCase):
    def test_unknown_values_fall_back_to_draft(self):
        for value in (None, "", "bogus", 42):
            self.assertEqual(normalize_maturity(value), MATURITY_DRAFT)

    def test_known_values_are_case_insensitive(self):
        self.assertEqual(normalize_maturity(" Verified "), MATURITY_VERIFIED)
        self.assertEqual(normalize_maturity("PROMOTED"), MATURITY_PROMOTED)

    def test_gate_comparison_is_ordered(self):
        self.assertTrue(maturity_meets_gate(MATURITY_DRAFT, MATURITY_DRAFT))
        self.assertFalse(maturity_meets_gate(MATURITY_DRAFT, MATURITY_VERIFIED))
        self.assertTrue(maturity_meets_gate(MATURITY_PROMOTED, MATURITY_VERIFIED))
        self.assertTrue(maturity_meets_gate(MATURITY_VERIFIED, MATURITY_VERIFIED))


class CardSerializationTests(unittest.TestCase):
    def test_new_cards_default_to_draft(self):
        self.assertEqual(_card().maturity, MATURITY_DRAFT)

    def test_maturity_is_persisted_in_metadata_and_json(self):
        card = _card(maturity=MATURITY_PROMOTED)

        self.assertEqual(card.to_metadata()["maturity"], MATURITY_PROMOTED)
        self.assertEqual(card.to_json_dict()["maturity"], MATURITY_PROMOTED)

    def test_invalid_maturity_is_normalized_on_serialization(self):
        card = _card(maturity="nonsense")

        self.assertEqual(card.to_metadata()["maturity"], MATURITY_DRAFT)


class RetrievalWeightingTests(unittest.TestCase):
    """Draft cards must stay retrievable but lose to better-graded peers."""

    def _score(self, service: StrategyMemoryService, maturity: str) -> float:
        return service._score_hit(
            similarity=0.5,
            stage="planning",
            strategy_kind="planning_pattern",
            origin_status="success",
            maturity=maturity,
        )

    def test_draft_is_penalized_verified_neutral_promoted_boosted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            draft = self._score(service, MATURITY_DRAFT)
            verified = self._score(service, MATURITY_VERIFIED)
            promoted = self._score(service, MATURITY_PROMOTED)

        self.assertLess(draft, verified)
        self.assertLess(verified, promoted)
        self.assertAlmostEqual(verified - draft, 0.15, places=6)
        self.assertAlmostEqual(promoted - verified, 0.05, places=6)

    def test_missing_maturity_defaults_to_draft_penalty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            explicit = self._score(service, MATURITY_DRAFT)
            defaulted = service._score_hit(
                similarity=0.5,
                stage="planning",
                strategy_kind="planning_pattern",
                origin_status="success",
            )

        self.assertAlmostEqual(explicit, defaulted, places=6)


class MatchReasonTests(unittest.TestCase):
    def _hit(self, maturity: str) -> StrategyHit:
        return StrategyHit(
            strategy_id="req-1::planning_pattern",
            strategy_kind="planning_pattern",
            stage_scope="planning",
            title="拆解模式",
            applicable_when="当主题涉及协议时",
            match_signals=[],
            recommended_actions=[],
            query_templates=[],
            preferred_sources=[],
            pitfalls_to_avoid=[],
            origin_request_id="req-1",
            origin_status="success",
            origin_review_status="pass",
            similarity=0.5,
            score=0.5,
            match_reason="",
            maturity=maturity,
        )

    def test_draft_is_flagged_as_unverified_in_the_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            reason = service._build_match_reason(self._hit(MATURITY_DRAFT), stage="planning")

        self.assertIn("未经验证草稿", reason)

    def test_promoted_is_flagged_as_adopted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            reason = service._build_match_reason(self._hit(MATURITY_PROMOTED), stage="planning")

        self.assertIn("已验证并采纳", reason)


class MaturityPersistenceTests(unittest.TestCase):
    """A promotion must survive re-synthesis, which rewrites cards from scratch."""

    def test_read_card_maturity_defaults_to_draft_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)

            self.assertEqual(service._read_card_maturity("missing::kind"), MATURITY_DRAFT)

    def test_read_card_maturity_round_trips_written_card(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            service._write_card_file(_card(maturity=MATURITY_VERIFIED))

            self.assertEqual(
                service._read_card_maturity("req-1::planning_pattern"),
                MATURITY_VERIFIED,
            )

    def test_corrupt_card_file_degrades_to_draft(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            service._cards_dir.mkdir(parents=True, exist_ok=True)
            service._card_path("req-1::planning_pattern").write_text("{not json", encoding="utf-8")

            self.assertEqual(
                service._read_card_maturity("req-1::planning_pattern"),
                MATURITY_DRAFT,
            )

    def test_set_card_maturity_updates_the_card_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            service._write_card_file(_card())

            changed = service.set_card_maturity("req-1::planning_pattern", MATURITY_PROMOTED)

            self.assertTrue(changed)
            payload = json.loads(
                service._card_path("req-1::planning_pattern").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["maturity"], MATURITY_PROMOTED)

    def test_set_card_maturity_rejects_unknown_card(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)

            self.assertFalse(service.set_card_maturity("nope::kind", MATURITY_PROMOTED))
            self.assertFalse(service.set_card_maturity("", MATURITY_PROMOTED))

    def test_set_card_maturity_normalizes_bogus_grade_to_draft(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)
            service._write_card_file(_card(maturity=MATURITY_PROMOTED))

            self.assertTrue(service.set_card_maturity("req-1::planning_pattern", "bogus"))
            payload = json.loads(
                service._card_path("req-1::planning_pattern").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["maturity"], MATURITY_DRAFT)


class MaturityGateConfigTests(unittest.TestCase):
    def test_gate_defaults_to_draft(self):
        config = Configuration.from_env(load_env_file=False)

        self.assertEqual(config.memory_maturity_gate, "draft")

    def test_gate_accepts_known_levels_case_insensitively(self):
        config = Configuration.from_env(
            overrides={"memory_maturity_gate": " VERIFIED "},
            load_env_file=False,
        )

        self.assertEqual(config.memory_maturity_gate, "verified")

    def test_gate_rejects_unknown_level(self):
        with self.assertRaises(Exception):
            Configuration.from_env(
                overrides={"memory_maturity_gate": "gold"},
                load_env_file=False,
            )


if __name__ == "__main__":
    unittest.main()
