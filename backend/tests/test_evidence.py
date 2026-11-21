import json
import sys
import tempfile
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
    build_task_context_from_hits,
    compute_content_quality_score,
    compute_freshness_label,
    compute_quality_score_and_label,
    compute_source_reliability_score,
    extract_citation_ids,
    extract_domain,
    format_evidence_sources,
    score_evidence_quality,
)
from services.evidence_index import (
    EvidenceMemoryService,
    EvidenceRetrievalService,
    EvidenceRuntimeIndex,
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

    def test_runtime_index_and_lookup_query_return_grounding_hits(self):
        config = Configuration.from_env(load_env_file=False)
        runtime_index = EvidenceRuntimeIndex(config)
        retrieval = EvidenceRetrievalService(
            runtime_index=runtime_index,
            memory_service=None,
            config=config,
        )
        store = EvidenceStore(listeners=[runtime_index])
        store.record_search_results(
            task_id=7,
            query="mcp architecture",
            search_payload={
                "results": [
                    {
                        "title": "MCP Overview",
                        "url": "https://example.com/mcp",
                        "content": "Model Context Protocol architecture overview and grounding model",
                    }
                ]
            },
            backend="duckduckgo",
        )
        tool = EvidenceLookupTool(
            evidence_store=store,
            retrieval_service=retrieval,
            request_id_getter=lambda: "req-runtime",
        )

        payload = json.loads(
            tool.run(
                {
                    "task_id": 7,
                    "query": "Model Context Protocol architecture",
                    "scope": "current",
                    "mode": "grounding",
                    "top_k": 3,
                }
            )
        )

        self.assertEqual(payload["hits"][0]["source_id"], "T7-S1")
        self.assertTrue(payload["hits"][0]["citation_eligible"])
        context = build_task_context_from_hits(payload["hits"], answer_text=None, config=config)
        self.assertIn("[T7-S1]", context)

    def test_evidence_memory_persists_versions_and_runtime_hydrate_restores_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Configuration.from_env(
                overrides={
                    "evidence_memory_enabled": True,
                    "evidence_memory_dir": temp_dir,
                },
                load_env_file=False,
            )
            memory = EvidenceMemoryService(config)
            runtime = EvidenceRuntimeIndex(config, archive_loader=memory)
            store = EvidenceStore(
                listeners=[memory, runtime],
                request_id_getter=lambda: "req-archive-1",
            )

            store.record_search_results(
                task_id=8,
                query="evidence archive",
                search_payload={
                    "results": [
                        {
                            "title": "Archive Doc",
                            "url": "https://example.com/archive-doc",
                            "content": "short snippet for archive doc",
                        }
                    ]
                },
                backend="duckduckgo",
            )
            before_fetch = store.get_evidence("T8-S1")
            self.assertTrue(before_fetch["archive_version_id"])

            store.update_full_content(
                task_id=8,
                source_id="T8-S1",
                url="https://example.com/archive-doc",
                title="Archive Doc",
                full_content="Detailed recovered full content for runtime hydrate and archive restore.",
            )
            after_fetch = store.get_evidence("T8-S1", include_full_content=True)
            self.assertTrue(after_fetch["archive_has_full_content"])
            self.assertNotEqual(before_fetch["archive_version_id"], after_fetch["archive_version_id"])

            restored_runtime = EvidenceRuntimeIndex(config, archive_loader=memory)
            restored_store = EvidenceStore(listeners=[restored_runtime])
            restored_task = TodoItem(
                id=8,
                title="任务8",
                intent="恢复 archive evidence",
                query="archive hydrate",
                evidence_items=[store.get_evidence("T8-S1")],
            )
            restored_store.hydrate_from_tasks([restored_task])
            restored_hits = restored_runtime.query(
                text="runtime hydrate archive restore",
                task_id=8,
                top_k=3,
            )

        self.assertTrue(restored_hits)
        self.assertIn("Detailed recovered full content", restored_hits[0].text)
        self.assertEqual(restored_hits[0].archive_version_id, after_fetch["archive_version_id"])

    def test_history_grounding_scope_is_forced_back_to_current(self):
        config = Configuration.from_env(load_env_file=False)
        runtime_index = EvidenceRuntimeIndex(config)
        retrieval = EvidenceRetrievalService(
            runtime_index=runtime_index,
            memory_service=None,
            config=config,
        )
        store = EvidenceStore(listeners=[runtime_index])
        store.record_search_results(
            task_id=9,
            query="grounding scope",
            search_payload={
                "results": [
                    {
                        "title": "Grounding Doc",
                        "url": "https://example.com/grounding",
                        "content": "Grounding scope stays on current request evidence only.",
                    }
                ]
            },
            backend="duckduckgo",
        )

        result = retrieval.query(
            "current request grounding",
            task_id=9,
            scope="history",
            mode="grounding",
            request_id="req-scope",
        )

        self.assertEqual(result.scope, "current")
        self.assertTrue(result.notices)
        self.assertTrue(result.hits[0].citation_eligible)

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

    def test_fetch_page_blocks_ssrf_schemes_and_private_ips(self):
        from services.evidence import _fetch_page_text, _validate_fetch_url

        for blocked_url in (
            "file:///etc/passwd",
            "ftp://example.com/a",
            "http://127.0.0.1:8000/healthz",
            "http://localhost:8000/metrics/json",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/internal",
            "http://192.168.1.1/admin",
        ):
            with self.assertRaises(ValueError, msg=blocked_url):
                _validate_fetch_url(blocked_url)

        with patch(
            "services.evidence.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("169.254.169.254", 80))],
        ):
            with self.assertRaisesRegex(ValueError, "non-public"):
                _fetch_page_text("https://evil.example.com/metadata", timeout_seconds=1.0)

    def test_fetch_page_validates_redirects_content_type_and_size_limit(self):
        from services.evidence import _fetch_page_text

        class FakeResponse:
            def __init__(self, *, status_code=200, headers=None, chunks=None):
                self.status_code = status_code
                self.headers = headers or {}
                self._chunks = chunks or [b"<html><title>Doc</title><script>evil()</script><p>Safe text</p></html>"]
                self.encoding = "utf-8"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def iter_content(self, chunk_size=65536):
                yield from self._chunks

        public_addr = [(2, 1, 6, "", ("93.184.216.34", 443))]

        # Redirect to loopback must be blocked
        with patch("services.evidence.socket.getaddrinfo", return_value=public_addr):
            with patch(
                "services.evidence.requests.get",
                return_value=FakeResponse(
                    status_code=302,
                    headers={"Location": "http://127.0.0.1/secret"},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "non-public"):
                    _fetch_page_text("https://example.com/start", timeout_seconds=1.0)

        # Oversized body must be rejected
        with patch("services.evidence.socket.getaddrinfo", return_value=public_addr):
            with patch(
                "services.evidence.requests.get",
                return_value=FakeResponse(
                    status_code=200,
                    headers={"Content-Type": "text/html"},
                    chunks=[b"a" * 600, b"b" * 600],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "exceeds max byte limit"):
                    _fetch_page_text("https://example.com/huge", timeout_seconds=1.0, max_bytes=1024)

        # Disallowed binary content type must be rejected
        with patch("services.evidence.socket.getaddrinfo", return_value=public_addr):
            with patch(
                "services.evidence.requests.get",
                return_value=FakeResponse(
                    status_code=200,
                    headers={"Content-Type": "application/octet-stream"},
                    chunks=[b"\x00\x01"],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unsupported content type"):
                    _fetch_page_text("https://example.com/bin", timeout_seconds=1.0)

        # Safe HTML strips <script> before extraction
        with patch("services.evidence.socket.getaddrinfo", return_value=public_addr):
            with patch(
                "services.evidence.requests.get",
                return_value=FakeResponse(
                    status_code=200,
                    headers={"Content-Type": "text/html; charset=utf-8"},
                ),
            ):
                title, text = _fetch_page_text("https://example.com/ok", timeout_seconds=1.0)
                self.assertEqual(title, "Doc")
                self.assertIn("Safe text", text)
                self.assertNotIn("evil()", text)

    def test_structured_html_extraction_preserves_headings_and_strips_boilerplate(self):
        from services.evidence import _extract_structured_html_text

        html = (
            "<html><body>"
            "<nav>Site Navigation Menu</nav>"
            "<header>Top Banner</header>"
            "<main>"
            "<h1>Architecture Overview</h1>"
            "<p>First paragraph explains the core pipeline.</p>"
            "<h2>Benchmark Results</h2>"
            "<p>Second paragraph reports latency metrics.</p>"
            "</main>"
            "<footer>Copyright 2026</footer>"
            "<script>window.track()</script>"
            "</body></html>"
        )
        extracted = _extract_structured_html_text(html)
        self.assertIn("Architecture Overview", extracted)
        self.assertIn("Benchmark Results", extracted)
        self.assertIn("\n", extracted)
        self.assertNotIn("Site Navigation Menu", extracted)
        self.assertNotIn("Copyright 2026", extracted)
        self.assertNotIn("window.track()", extracted)

    def test_bilingual_structure_aware_chunking_and_bm25_rrf_runtime_scoring(self):
        from services.evidence_index import _chunk_text

        zh_section = (
            "# 核心架构\n"
            + ("这是第一段关于向量检索与重排架构的详细说明，包含混合打分机制。" * 12)
            + "\n\n## 延迟优化\n"
            + ("这是第二段关于并发缓存隔离和异步流水线的详细基准测试结果。" * 15)
        )
        chunks = _chunk_text(zh_section, max_chars=240, overlap=30, max_chunks=32)
        self.assertGreater(len(chunks), 3)
        # Continuation chunks inside a section carry heading breadcrumb context
        self.assertTrue(any("## 延迟优化" in chunk_text for _, _, chunk_text in chunks))
        # Chunks respect Chinese sentence boundaries instead of mid-clause slicing
        mid_chunks = [text for _, _, text in chunks[:-1]]
        self.assertTrue(any(text.endswith("。") for text in mid_chunks))

        # Long document supports more than legacy 12 chunks
        very_long_doc = "\n\n".join(
            f"## 章节 {idx}\n" + ("长文段落内容用于验证切片上限扩展到三十二个以上。" * 10)
            for idx in range(1, 20)
        )
        many_chunks = _chunk_text(very_long_doc, max_chars=220, overlap=20, max_chunks=32)
        self.assertGreater(len(many_chunks), 12)

        # Runtime index scores relevant chunk highest using BM25 + RRF even without embeddings
        cfg = Configuration.from_env(
            overrides={"evidence_runtime_enabled": True, "evidence_chunk_chars": 240, "evidence_chunk_overlap": 20},
            load_env_file=False,
        )
        runtime_index = EvidenceRuntimeIndex(cfg)
        store = EvidenceStore(listeners=[runtime_index])
        store.record_search_results(
            task_id=1,
            query="hybrid retrieval",
            search_payload={
                "results": [
                    {
                        "title": "Doc A",
                        "url": "https://docs.example.com/a",
                        "content": "# 混合检索\nBM25 与稠密向量的 RRF 融合排序能显著降低噪声。",
                    },
                    {
                        "title": "Doc B",
                        "url": "https://docs.example.com/b",
                        "content": "# 饮食指南\n本文介绍冬季蔬菜种植与烹饪步骤。",
                    },
                ]
            },
            backend="tavily",
        )
        with patch("services.evidence_index.encode_text", return_value=None):
            hits = runtime_index.query(text="RRF 融合排序 BM25", task_id=1, top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].title, "Doc A")
        self.assertIn("bm25", hits[0].score_breakdown)
        self.assertIn("rrf", hits[0].score_breakdown)
        self.assertGreater(hits[0].score, hits[1].score)


if __name__ == "__main__":
    unittest.main()
