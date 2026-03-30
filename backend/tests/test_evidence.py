import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from models import TodoItem
from services.evidence import (
    EvidenceLookupTool,
    EvidenceStore,
    FetchPageTool,
    SearchWebTool,
    build_task_context,
    extract_citation_ids,
    format_evidence_sources,
)


class EvidenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = EvidenceStore()

    def test_record_search_results_assigns_stable_source_ids(self):
        first_payload = {
            "results": [
                {
                    "title": "Alpha",
                    "url": "https://example.com/a?utm_source=feed",
                    "content": "short snippet",
                }
            ]
        }
        second_payload = {
            "results": [
                {
                    "title": "Alpha Improved",
                    "url": "https://example.com/a",
                    "content": "a much longer snippet than before",
                }
            ]
        }

        first = self.store.record_search_results(
            task_id=1,
            query="alpha",
            search_payload=first_payload,
            backend="duckduckgo",
        )
        second = self.store.record_search_results(
            task_id=1,
            query="alpha",
            search_payload=second_payload,
            backend="duckduckgo",
        )

        self.assertEqual(first[0]["source_id"], "T1-S1")
        self.assertEqual(second[0]["source_id"], "T1-S1")
        self.assertIn("much longer snippet", second[0]["snippet"])

    def test_fetch_page_tool_updates_full_content(self):
        self.store.record_search_results(
            task_id=2,
            query="beta",
            search_payload={
                "results": [
                    {
                        "title": "Beta",
                        "url": "https://example.com/b",
                        "content": "snippet",
                    }
                ]
            },
            backend="duckduckgo",
        )

        tool = FetchPageTool(evidence_store=self.store, timeout_seconds=1.0)
        with patch(
            "services.evidence._fetch_page_text",
            return_value=("Beta Full", "full article body"),
        ):
            payload = json.loads(
                tool.run({"task_id": 2, "source_id": "T2-S1", "url": "https://example.com/b"})
            )

        self.assertTrue(payload["has_full_content"])
        self.assertEqual(payload["source_id"], "T2-S1")

    def test_record_search_results_handles_semanticscholar_publication_fields(self):
        evidence = self.store.record_search_results(
            task_id=5,
            query="semantic scholar",
            search_payload={
                "results": [
                    {
                        "title": "Semantic Scholar Paper",
                        "url": "https://www.semanticscholar.org/paper/abc123",
                        "content": "paper abstract",
                        "publicationDate": "2025-02-14",
                        "year": 2025,
                        "citation_count": 42,
                    }
                ]
            },
            backend="semanticscholar",
        )

        self.assertEqual(evidence[0]["source_type"], "paper")
        self.assertEqual(evidence[0]["published_at"], "2025-02-14")
        self.assertEqual(evidence[0]["quality_label"], "high")

    def test_record_search_results_falls_back_to_year_when_publication_date_is_missing(self):
        evidence = self.store.record_search_results(
            task_id=6,
            query="semantic scholar year fallback",
            search_payload={
                "results": [
                    {
                        "title": "Semantic Scholar Year Only",
                        "url": "https://www.semanticscholar.org/paper/xyz456",
                        "content": "paper abstract",
                        "year": 2024,
                    }
                ]
            },
            backend="semanticscholar",
        )

        self.assertEqual(evidence[0]["published_at"], "2024-01-01")

    def test_search_web_tool_records_structured_evidence(self):
        tool = SearchWebTool(
            config=Configuration.from_env(load_env_file=False),
            evidence_store=self.store,
            observer_getter=lambda: None,
        )

        with patch(
            "services.evidence.dispatch_search",
            return_value=(
                {
                    "results": [
                        {
                            "title": "Gamma",
                            "url": "https://example.com/g",
                            "content": "gamma snippet",
                        }
                    ]
                },
                [],
                None,
                "duckduckgo",
                False,
                "miss",
            ),
        ):
            payload = json.loads(
                tool.run(
                    {
                        "task_id": 3,
                        "query": "gamma",
                        "research_topic": "AI agent",
                        "task_title": "Gamma search",
                        "task_intent": "collect gamma",
                    }
                )
            )

        self.assertEqual(payload["evidence"][0]["source_id"], "T3-S1")
        self.assertEqual(self.store.get_evidence("T3-S1")["title"], "Gamma")

    def test_evidence_lookup_and_context_helpers_keep_source_ids_visible(self):
        self.store.record_search_results(
            task_id=4,
            query="delta",
            search_payload={
                "results": [
                    {
                        "title": "Delta",
                        "url": "https://example.com/d",
                        "content": "delta snippet",
                    }
                ]
            },
            backend="duckduckgo",
        )
        items = self.store.list_task_evidence(4)
        lookup = EvidenceLookupTool(evidence_store=self.store)
        payload = json.loads(lookup.run({"task_id": 4}))
        context = build_task_context(
            items,
            answer_text="direct answer",
            config=Configuration.from_env(load_env_file=False),
        )

        self.assertEqual(payload["evidence"][0]["source_id"], "T4-S1")
        self.assertIn("[T4-S1]", format_evidence_sources(items))
        self.assertIn("[T4-S1]", context)
        self.assertEqual(extract_citation_ids("结论 [T4-S1][T4-S1] [T4-S2]"), ["T4-S1", "T4-S2"])

    def test_hydrate_from_tasks_restores_reference_map_with_original_source_ids(self):
        task = TodoItem(
            id=3,
            title="任务3",
            intent="恢复引用",
            query="resume",
            evidence_items=[
                {
                    "source_id": "T3-S5",
                    "title": "Recovered Ref",
                    "url": "https://example.com/recovered-ref",
                    "snippet": "restored from snapshot",
                    "domain": "example.com",
                    "source_type": "web",
                    "quality_label": "medium",
                    "freshness_label": "recent",
                }
            ],
        )

        self.store.hydrate_from_tasks([task])
        refs = self.store.build_reference_map(["T3-S5", "T3-S9"])

        self.assertEqual(refs, [
            {
                "source_id": "T3-S5",
                "title": "Recovered Ref",
                "url": "https://example.com/recovered-ref",
                "domain": "example.com",
                "published_at": "",
            }
        ])


if __name__ == "__main__":
    unittest.main()
