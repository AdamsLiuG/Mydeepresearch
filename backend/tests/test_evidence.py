import json
import sys
import unittest
from datetime import datetime, timezone
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
    compute_content_quality_score,
    compute_freshness_label,
    compute_quality_score_and_label,
    compute_source_reliability_score,
    extract_domain,
    extract_citation_ids,
    format_evidence_sources,
    score_evidence_quality,
)


class EvidenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = EvidenceStore()
        self.fixed_now = datetime(2026, 3, 31, tzinfo=timezone.utc)

    @staticmethod
    def _rich_text() -> str:
        body = " ".join(f"analysis{i}" for i in range(620))
        return (
            "# Overview\n"
            f"{body}\n\n"
            "## References\n"
            "https://doi.org/10.1000/xyz123\n"
            "https://www.rfc-editor.org/rfc/rfc9110\n"
            "https://arxiv.org/abs/2401.00001\n"
            "https://github.com/openai/openai-python\n"
            "https://docs.python.org/3/\n"
        )

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
                        "content": " ".join(f"abstract{i}" for i in range(30)),
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
        self.assertEqual(evidence[0]["quality_label"], "medium")

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

    def test_extract_domain_normalizes_host_and_rejects_invalid_url(self):
        self.assertEqual(
            extract_domain("https://www.Example.com:443/path?utm_source=feed"),
            "example.com",
        )
        self.assertEqual(extract_domain("not a url"), "")

    def test_compute_freshness_label_prefers_publish_date_then_source_update(self):
        self.assertEqual(
            compute_freshness_label({"published_at": "2026-03-10"}, now=self.fixed_now),
            (21, "fresh"),
        )
        self.assertEqual(
            compute_freshness_label({"source_updated_at": "2025-12-15"}, now=self.fixed_now),
            (106, "recent"),
        )
        self.assertEqual(
            compute_freshness_label({"published_at": "2025-01-01"}, now=self.fixed_now),
            (454, "stale"),
        )
        self.assertEqual(
            compute_freshness_label({}, now=self.fixed_now),
            (None, "unknown"),
        )

    def test_compute_source_reliability_score_rewards_official_documentation(self):
        ev = {
            "title": "OpenAI API Reference",
            "url": "https://docs.openai.com/docs/api",
            "source_type": "documentation",
            "provider_count": 2,
        }
        self.assertEqual(compute_source_reliability_score(ev), 8.3)
        scored = score_evidence_quality(ev, now=self.fixed_now)
        self.assertIn("official_owner_match", scored["quality_flags"])

    def test_compute_source_reliability_score_rewards_forum_expert_and_social_official(self):
        forum_ev = {
            "title": "Maintainer answer",
            "url": "https://stackoverflow.com/questions/123/example",
            "byline": "Project maintainer",
        }
        social_ev = {
            "title": "OpenAI Official",
            "url": "https://twitter.com/OpenAI",
            "organization": "OpenAI",
        }
        self.assertEqual(compute_source_reliability_score(forum_ev), 4.3)
        self.assertEqual(compute_source_reliability_score(social_ev), 4.3)

    def test_compute_content_quality_score_rewards_rich_well_attributed_content(self):
        ev = {
            "title": "Deep guide",
            "url": "https://example.com/guide",
            "full_content": self._rich_text(),
            "published_at": "2026-03-01",
            "source_updated_at": "2026-03-05",
            "author": "Staff Engineer",
            "references": [
                "https://doi.org/10.1000/xyz123",
                "https://www.rfc-editor.org/rfc/rfc9110",
                "https://arxiv.org/abs/2401.00001",
                "https://github.com/openai/openai-python",
                "https://docs.python.org/3/",
            ],
        }
        self.assertEqual(compute_content_quality_score(ev), 8.9)

    def test_compute_content_quality_score_penalizes_low_value_aggregator_pages(self):
        ev = {
            "title": "You won't believe this ultimate guide?",
            "url": "https://www.msn.com/news/example",
            "full_content": "Read more. Subscribe now. Sponsored content. Shop now. Promo offer.",
            "duplicate": True,
        }
        self.assertEqual(compute_content_quality_score(ev), 0.0)

    def test_compute_quality_score_and_label_downgrades_high_for_poor_extraction(self):
        score, label = compute_quality_score_and_label(
            10.0,
            8.0,
            {
                "source_type_v2": "official_documentation",
                "word_count": 500,
                "extraction_quality": "poor",
                "navigation_page": False,
            },
        )
        self.assertEqual(score, 91)
        self.assertEqual(label, "medium")

    def test_compute_quality_score_and_label_downgrades_high_for_navigation_page(self):
        score, label = compute_quality_score_and_label(
            9.0,
            8.0,
            {
                "source_type_v2": "official_documentation",
                "word_count": 500,
                "extraction_quality": "good",
                "navigation_page": True,
            },
        )
        self.assertEqual(score, 86)
        self.assertEqual(label, "medium")

    def test_compute_quality_score_and_label_downgrades_high_for_short_content(self):
        score, label = compute_quality_score_and_label(
            9.0,
            8.0,
            {
                "source_type_v2": "official_documentation",
                "word_count": 100,
                "extraction_quality": "good",
                "navigation_page": False,
            },
        )
        self.assertEqual(score, 86)
        self.assertEqual(label, "medium")

    def test_score_evidence_quality_returns_stable_fields_and_unique_flags(self):
        scored = score_evidence_quality(
            {
                "title": "OpenAI Docs",
                "url": "https://docs.openai.com/docs/api",
                "full_content": self._rich_text(),
                "published_at": "2026-03-01",
                "source_updated_at": "2026-03-05",
                "author": "Research Scientist",
                "references": [
                    "https://doi.org/10.1000/xyz123",
                    "https://www.rfc-editor.org/rfc/rfc9110",
                    "https://arxiv.org/abs/2401.00001",
                    "https://github.com/openai/openai-python",
                    "https://docs.python.org/3/",
                ],
            },
            now=self.fixed_now,
        )
        self.assertEqual(scored["quality_label"], "high")
        self.assertTrue(scored["quality_reasons"])
        self.assertTrue(scored["quality_flags"])
        self.assertEqual(len(scored["quality_flags"]), len(set(scored["quality_flags"])))
        self.assertEqual(scored["freshness_label"], "fresh")

    def test_record_search_results_and_update_full_content_recompute_v2_scores(self):
        before = self.store.record_search_results(
            task_id=7,
            query="openai docs",
            search_payload={
                "results": [
                    {
                        "title": "OpenAI API Reference",
                        "url": "https://docs.openai.com/docs/api",
                        "content": " ".join(f"snippet{i}" for i in range(30)),
                        "publicationDate": "2026-03-01",
                        "updatedAt": "2026-03-05",
                        "provider_count": 2,
                    }
                ]
            },
            backend="duckduckgo",
        )[0]

        after = self.store.update_full_content(
            task_id=7,
            source_id="T7-S1",
            url="https://docs.openai.com/docs/api",
            title="OpenAI API Reference",
            full_content=self._rich_text(),
        )

        self.assertIn("source_reliability_score", before)
        self.assertIn("content_quality_score", before)
        self.assertGreater(after["quality_score"], before["quality_score"])
        self.assertGreater(after["content_quality_score"], before["content_quality_score"])
        self.assertEqual(after["quality_label"], "high")

    def test_hydrate_from_tasks_recomputes_v2_fields_and_keeps_source_id(self):
        task = TodoItem(
            id=9,
            title="任务9",
            intent="恢复质量元数据",
            query="hydrate quality",
            evidence_items=[
                {
                    "source_id": "T9-S4",
                    "title": "Recovered Doc",
                    "url": "https://docs.openai.com/docs/api",
                    "snippet": " ".join(f"snippet{i}" for i in range(30)),
                    "source_type": "documentation",
                    "published_at": "2026-03-01",
                    "quality_label": "medium",
                }
            ],
        )
        self.store.hydrate_from_tasks([task])
        restored = self.store.get_evidence("T9-S4")

        self.assertEqual(restored["source_id"], "T9-S4")
        self.assertIn("source_reliability_score", restored)
        self.assertIn("content_quality_score", restored)
        self.assertTrue(restored["quality_reasons"])
        self.assertIn(restored["quality_label"], {"medium", "high"})


if __name__ == "__main__":
    unittest.main()
