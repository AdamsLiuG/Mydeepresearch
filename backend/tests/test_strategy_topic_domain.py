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
from services.strategy_memory import StrategyMemoryService
from services.strategy_synthesizer import (
    DOMAIN_GENERAL,
    TOPIC_DOMAINS,
    StrategyCard,
    StrategySourceRequest,
    StrategySynthesizer,
    normalize_topic_domain,
)


def _source_request(**overrides) -> StrategySourceRequest:
    payload = {
        "request_id": "req-1",
        "topic": "某项新规的适用范围",
        "status": "success",
        "review_status": "pass",
        "report_available": True,
        "completed_task_count": 3,
        "failed_task_count": 0,
        "repair_cycles": 0,
        "requested_kinds": ["planning_pattern"],
    }
    payload.update(overrides)
    return StrategySourceRequest(**payload)


def _synthesizer() -> StrategySynthesizer:
    config = Configuration.from_env(load_env_file=False)
    return StrategySynthesizer(lambda: DummyToolAwareSimpleAgent(), config)


def _card_payload(topic_domain) -> str:
    return json.dumps(
        [
            {
                "strategy_kind": "planning_pattern",
                "title": "先定位原文",
                "applicable_when": "主题涉及新规适用范围时",
                "match_signals": ["新规", "适用范围"],
                "recommended_actions": ["先检索发布机构原文"],
                "query_templates": ["<机构名> <文号> 原文"],
                "preferred_sources": ["发布机构官网"],
                "pitfalls_to_avoid": ["只看二手解读"],
                "topic_domain": topic_domain,
            }
        ],
        ensure_ascii=False,
    )


class TopicDomainNormalizationTests(unittest.TestCase):
    def test_known_domains_pass_through(self):
        for domain in TOPIC_DOMAINS:
            self.assertEqual(normalize_topic_domain(domain), domain)

    def test_separator_and_case_variants_are_accepted(self):
        self.assertEqual(normalize_topic_domain("Policy-Analysis"), "policy_analysis")
        self.assertEqual(normalize_topic_domain(" TECH SELECTION "), "tech_selection")

    def test_unknown_or_invented_values_degrade_to_general(self):
        for value in (None, "", "政策", "policy", "brand_new_domain", 7):
            self.assertEqual(normalize_topic_domain(value), DOMAIN_GENERAL)

    def test_vocabulary_stays_closed(self):
        self.assertIn(DOMAIN_GENERAL, TOPIC_DOMAINS)
        self.assertEqual(len(TOPIC_DOMAINS), len(set(TOPIC_DOMAINS)))


class SynthesisPromptTests(unittest.TestCase):
    """The tag is produced by the existing synthesis call, not a second LLM round trip."""

    def test_prompt_declares_the_closed_vocabulary(self):
        prompt = _synthesizer()._build_prompt(_source_request())

        self.assertIn("<TOPIC_DOMAIN>", prompt)
        for domain in TOPIC_DOMAINS:
            self.assertIn(domain, prompt)

    def test_prompt_requires_topic_domain_in_output(self):
        prompt = _synthesizer()._build_prompt(_source_request())

        self.assertIn('"topic_domain"', prompt)

    def test_prompt_forbids_inventing_new_values(self):
        prompt = _synthesizer()._build_prompt(_source_request())

        self.assertIn("不要自己发明新值", prompt)


class SynthesisParsingTests(unittest.TestCase):
    def _parse(self, raw: str) -> list[StrategyCard]:
        return _synthesizer()._parse_response(raw, _source_request())

    def test_valid_domain_is_captured(self):
        cards = self._parse(_card_payload("policy_analysis"))

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].topic_domain, "policy_analysis")

    def test_invented_domain_degrades_without_dropping_the_card(self):
        cards = self._parse(_card_payload("政策法规解读"))

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].topic_domain, DOMAIN_GENERAL)

    def test_missing_domain_defaults_to_general(self):
        payload = json.loads(_card_payload("policy_analysis"))
        payload[0].pop("topic_domain")
        cards = self._parse(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].topic_domain, DOMAIN_GENERAL)

    def test_domain_is_persisted_in_metadata_and_json(self):
        card = self._parse(_card_payload("tech_selection"))[0]

        self.assertEqual(card.to_metadata()["topic_domain"], "tech_selection")
        self.assertEqual(card.to_json_dict()["topic_domain"], "tech_selection")


class _FakeCollection:
    """Minimal stand-in returning every record at a fixed distance."""

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def query(self, *, query_embeddings, n_results, include):
        selected = self._records[:n_results]
        return {
            "documents": [[record["title"] for record in selected]],
            "distances": [[0.1 for _ in selected]],
            "metadatas": [list(selected)],
        }


class DomainFilterTests(unittest.TestCase):
    """Filtering is opt-in: retrieval must not silently narrow by domain."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._tmp_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _service(self, records: list[dict]) -> StrategyMemoryService:
        config = Configuration.from_env(
            overrides={"strategy_memory_dir": self._tmp_dir},
            load_env_file=False,
        )
        service = StrategyMemoryService(config)
        service._collection = _FakeCollection(records)
        return service

    def _record(self, strategy_id: str, domain: str) -> dict:
        return {
            "strategy_id": strategy_id,
            "strategy_kind": "planning_pattern",
            "stage_scope": "planning",
            "title": f"卡片 {strategy_id}",
            "applicable_when": "适用场景",
            "match_signals": "[]",
            "recommended_actions": '["动作"]',
            "query_templates": "[]",
            "preferred_sources": "[]",
            "pitfalls_to_avoid": "[]",
            "origin_request_id": strategy_id.split("::")[0],
            "origin_status": "success",
            "origin_review_status": "pass",
            "maturity": "verified",
            "topic_domain": domain,
        }

    def test_no_filter_returns_all_domains(self):
        service = self._service(
            [
                self._record("req-a::planning_pattern", "policy_analysis"),
                self._record("req-b::planning_pattern", "tech_selection"),
            ]
        )

        hits = service._query_hits(
            query_embedding=[1.0],
            stage="planning",
            top_k=5,
            current_request_id=None,
        )

        self.assertEqual(
            {hit.topic_domain for hit in hits},
            {"policy_analysis", "tech_selection"},
        )

    def test_filter_narrows_to_requested_domains(self):
        service = self._service(
            [
                self._record("req-a::planning_pattern", "policy_analysis"),
                self._record("req-b::planning_pattern", "tech_selection"),
            ]
        )

        hits = service._query_hits(
            query_embedding=[1.0],
            stage="planning",
            top_k=5,
            current_request_id=None,
            topic_domains={"policy_analysis"},
        )

        self.assertEqual([hit.topic_domain for hit in hits], ["policy_analysis"])

    def test_records_without_domain_are_treated_as_general(self):
        record = self._record("req-a::planning_pattern", "policy_analysis")
        record.pop("topic_domain")
        service = self._service([record])

        hits = service._query_hits(
            query_embedding=[1.0],
            stage="planning",
            top_k=5,
            current_request_id=None,
        )

        self.assertEqual([hit.topic_domain for hit in hits], [DOMAIN_GENERAL])

    def test_filter_can_exclude_everything(self):
        service = self._service([self._record("req-a::planning_pattern", "policy_analysis")])

        hits = service._query_hits(
            query_embedding=[1.0],
            stage="planning",
            top_k=5,
            current_request_id=None,
            topic_domains={"academic_review"},
        )

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
