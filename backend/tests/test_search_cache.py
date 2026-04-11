import importlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from metrics import RequestTrace, metrics_registry


class DummyDiskCache:
    def __init__(self, directory):
        self.directory = directory
        self._store = {}

    def get(self, key, default=None):
        entry = self._store.get(key)
        if entry is None:
            return default
        expires_at = entry.get("expires_at")
        if expires_at is not None and expires_at <= time.time():
            self._store.pop(key, None)
            return default
        return deepcopy(entry.get("value"))

    def set(self, key, value, expire=None):
        expires_at = None
        if expire is not None:
            expires_at = time.time() + float(expire)
        self._store[key] = {"value": deepcopy(value), "expires_at": expires_at}

    def delete(self, key):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()

    def close(self):
        return None


class DummyChromaCollection:
    def __init__(self):
        self._records = {}

    def upsert(self, ids, documents=None, metadatas=None, embeddings=None):
        for index, record_id in enumerate(ids):
            self._records[record_id] = {
                "document": documents[index] if documents else None,
                "metadata": deepcopy(metadatas[index]) if metadatas else {},
                "embedding": list(embeddings[index]) if embeddings else None,
            }

    def delete(self, ids=None, where=None):
        if ids:
            for record_id in ids:
                self._records.pop(record_id, None)
            return
        if where and "scope_key" in where:
            scope_key = where["scope_key"]
            removable = [
                record_id
                for record_id, record in self._records.items()
                if str((record.get("metadata") or {}).get("scope_key") or "") == str(scope_key)
            ]
            for record_id in removable:
                self._records.pop(record_id, None)

    def query(self, query_embeddings=None, n_results=10, where=None, include=None):
        query_embedding = list((query_embeddings or [[0.0]])[0])
        scope_key = str((where or {}).get("scope_key") or "")
        ranked = []
        for record_id, record in self._records.items():
            metadata = record.get("metadata") or {}
            if scope_key and str(metadata.get("scope_key") or "") != scope_key:
                continue
            embedding = list(record.get("embedding") or [])
            score = sum(left * right for left, right in zip(query_embedding, embedding))
            ranked.append((score, record_id, deepcopy(metadata)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        limited = ranked[: max(int(n_results or 1), 1)]
        return {
            "ids": [[record_id for _, record_id, _ in limited]],
            "metadatas": [[metadata for _, _, metadata in limited]],
            "distances": [[1.0 - score for score, _, _ in limited]],
        }


class DummyChromaClient:
    _collections_by_path = {}

    def __init__(self, path):
        self.path = path
        self._collections = self._collections_by_path.setdefault(path, {})

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = DummyChromaCollection()
        return self._collections[name]

    def delete_collection(self, name):
        self._collections.pop(name, None)


class DummySentenceTransformer:
    embeddings = {}

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, text, normalize_embeddings=True):
        normalized = " ".join((text or "").strip().lower().split())
        vector = self.embeddings.get(normalized)
        if vector is None:
            vector = [float(len(normalized) or 1), 0.0, 0.0]

        if not normalize_embeddings:
            return vector

        length = sum(value * value for value in vector) ** 0.5
        if length == 0:
            return vector
        return [value / length for value in vector]


class DummySearchTool:
    call_count = 0
    calls = []
    responses = {}

    def __init__(self, backend="hybrid"):
        self.backend = backend

    def run(self, payload):
        DummySearchTool.call_count += 1
        DummySearchTool.calls.append(deepcopy(payload))
        response = DummySearchTool.responses.get(payload["backend"])
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(deepcopy(payload))
        if response is not None:
            return deepcopy(response)
        return {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "cached content",
                }
            ],
            "backend": payload["backend"],
            "answer": None,
            "notices": [],
        }


class MockHTTPResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = deepcopy(payload)
        self.text = text if text is not None else json.dumps(payload or {}, ensure_ascii=False)

    def json(self):
        if self._payload is None:
            raise ValueError("missing json payload")
        return deepcopy(self._payload)


def rerank_completion_payload(ranked_ids):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"ranked_ids": ranked_ids}, ensure_ascii=False),
                }
            }
        ]
    }


def rerank_api_payload(ranked_indices):
    return {
        "results": [
            {
                "index": index,
                "document": {"text": f"document-{index}"},
                "relevance_score": 1.0 / (position + 1),
            }
            for position, index in enumerate(ranked_indices)
        ]
    }


diskcache_stub = types.ModuleType("diskcache")
diskcache_stub.Cache = DummyDiskCache
sys.modules["diskcache"] = diskcache_stub

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = DummyChromaClient
sys.modules["chromadb"] = chromadb_stub

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.SentenceTransformer = DummySentenceTransformer
sys.modules["sentence_transformers"] = sentence_transformers_stub

tools_stub = types.ModuleType("hello_agents.tools")
tools_stub.SearchTool = DummySearchTool
sys.modules["hello_agents.tools"] = tools_stub

importlib.reload(importlib.import_module("services.embeddings"))
services_search = importlib.reload(importlib.import_module("services.search"))


class SearchCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patcher = patch.dict(
            os.environ,
            {
                "ADVANCED_RERANK_ENABLED": "false",
                "ADVANCED_RERANK_BASE_URL": "",
                "ADVANCED_RERANK_API_KEY": "",
                "ADVANCED_RERANK_MODEL": "",
                "ADVANCED_RERANK_CANDIDATE_POOL": "3",
                "ADVANCED_RERANK_TIMEOUT_SECONDS": "0.1",
                "ADVANCED_RERANK_MAX_CONTENT_CHARS": "100",
            },
        )
        self.env_patcher.start()
        DummySearchTool.call_count = 0
        DummySearchTool.calls = []
        DummySearchTool.responses = {}
        DummySentenceTransformer.embeddings = {}
        DummyChromaClient._collections_by_path = {}
        metrics_registry.reset()
        services_search.clear_search_cache()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        services_search.clear_search_cache()
        self.temp_dir.cleanup()
        self.env_patcher.stop()

    def test_dispatch_search_uses_cache_and_updates_metrics(self):
        observer = RequestTrace(
            request_id="req-cache",
            topic="cache test",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        first = services_search.dispatch_search("same query", config, 0, observer=observer)
        second = services_search.dispatch_search("same query", config, 1, observer=observer)

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertFalse(first[4])
        self.assertTrue(second[4])
        self.assertEqual(first[5], "miss")
        self.assertEqual(second[5], "exact")
        self.assertEqual(observer.snapshot()["cache_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_exact_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_semantic_hits"], 0)
        self.assertEqual(observer.snapshot()["cache_misses"], 1)
        self.assertEqual(observer.snapshot()["last_search_cache_details"]["cache_hit_mode"], "exact")
        metrics_snapshot = metrics_registry.snapshot()
        self.assertEqual(metrics_snapshot["cache_hit_total"], 1)
        self.assertEqual(metrics_snapshot["cache_exact_hit_total"], 1)
        self.assertEqual(metrics_snapshot["cache_semantic_hit_total"], 0)
        self.assertEqual(metrics_snapshot["cache_miss_total"], 1)

    def test_dispatch_search_assigns_dynamic_ttl_buckets(self):
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_dynamic_ttl_enabled": True,
                "search_cache_ttl_seconds": 43200,
                "search_cache_fresh_ttl_seconds": 14400,
                "search_cache_evergreen_ttl_seconds": 172800,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        services_search.dispatch_search("latest multimodal model benchmark 2026", config, 0)
        fresh_key = services_search._build_cache_key("latest multimodal model benchmark 2026", "duckduckgo", config)
        fresh_entry = services_search._read_exact_cache(fresh_key, config)
        self.assertIsNotNone(fresh_entry)
        self.assertEqual(fresh_entry.ttl_bucket, "fresh")
        self.assertEqual(fresh_entry.ttl_seconds, 14400)

        services_search.dispatch_search("what is mcp protocol architecture", config, 0)
        evergreen_key = services_search._build_cache_key("what is mcp protocol architecture", "duckduckgo", config)
        evergreen_entry = services_search._read_exact_cache(evergreen_key, config)
        self.assertIsNotNone(evergreen_entry)
        self.assertEqual(evergreen_entry.ttl_bucket, "evergreen")
        self.assertEqual(evergreen_entry.ttl_seconds, 172800)

    def test_dispatch_search_drops_expired_exact_cache_entries(self):
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_dynamic_ttl_enabled": False,
                "search_cache_ttl_seconds": 1,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        first = services_search.dispatch_search("stable cache query", config, 0)
        time.sleep(1.05)
        second = services_search.dispatch_search("stable cache query", config, 1)

        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(DummySearchTool.call_count, 2)

    def test_dispatch_search_fuses_advanced_backends_and_deduplicates_urls(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha from SearXNG",
                        "url": "https://example.com/a?utm_source=feed",
                        "content": "short alpha",
                    },
                    {
                        "title": "Beta from SearXNG",
                        "url": "https://example.com/b",
                        "content": "beta content",
                    },
                ],
                "backend": "searxng",
                "answer": None,
                "notices": ["rate limited but returned partial results"],
            },
            "tavily": {
                "results": [
                    {
                        "title": "Alpha from Tavily",
                        "url": "https://example.com/a",
                        "content": "much longer alpha content from tavily",
                    },
                    {
                        "title": "Gamma from Tavily",
                        "url": "https://example.com/c",
                        "content": "gamma content",
                    },
                ],
                "backend": "tavily",
                "answer": "direct answer",
                "notices": [],
            },
            "serpapi": RuntimeError("quota exceeded"),
        }
        observer = RequestTrace(
            request_id="req-advanced",
            topic="fusion test",
            search_api="advanced",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng", "tavily", "serpapi"],
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        payload, notices, answer_text, backend_label, cache_hit, cache_strategy = services_search.dispatch_search(
            "same fusion query",
            config,
            0,
            observer=observer,
        )

        self.assertFalse(cache_hit)
        self.assertEqual(cache_strategy, "miss")
        self.assertEqual(answer_text, "direct answer")
        self.assertEqual(backend_label, "advanced[searxng, tavily]")
        self.assertEqual([item["url"] for item in payload["results"]], [
            "https://example.com/a?utm_source=feed",
            "https://example.com/b",
            "https://example.com/c",
        ])
        self.assertEqual(payload["results"][0]["provider_count"], 2)
        self.assertEqual(payload["results"][0]["backend_sources"], ["searxng", "tavily"])
        self.assertEqual(
            payload["results"][0]["content"],
            "much longer alpha content from tavily",
        )
        self.assertEqual(
            payload["ranking"],
            {
                "strategy": "rules",
                "rerank_applied": False,
                "candidate_count": 3,
                "model": None,
                "fallback_reason": None,
            },
        )
        self.assertTrue(any("searxng:" in notice for notice in notices))
        self.assertTrue(any("serpapi 搜索失败" in notice for notice in notices))
        self.assertEqual(
            sorted(call["backend"] for call in DummySearchTool.calls),
            ["searxng", "serpapi", "tavily"],
        )
        metrics_snapshot = metrics_registry.snapshot()
        self.assertEqual(metrics_snapshot["cache_hit_total"], 0)
        self.assertEqual(metrics_snapshot["cache_miss_total"], 1)

    def test_dispatch_search_fuses_advanced_backends_in_parallel(self):
        def slow_response(backend: str):
            def _response(payload):
                time.sleep(0.2)
                return {
                    "results": [
                        {
                            "title": f"{backend} result",
                            "url": f"https://example.com/{backend}",
                            "content": f"{backend} content",
                        }
                    ],
                    "backend": backend,
                    "answer": None,
                    "notices": [],
                }

            return _response

        DummySearchTool.responses = {
            "searxng": slow_response("searxng"),
            "tavily": slow_response("tavily"),
            "serpapi": slow_response("serpapi"),
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng", "tavily", "serpapi"],
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        started_at = time.perf_counter()
        payload, notices, answer_text, backend_label, cache_hit, cache_strategy = services_search.dispatch_search(
            "parallel fusion query",
            config,
            0,
        )
        duration = time.perf_counter() - started_at

        self.assertLess(duration, 0.45)
        self.assertFalse(cache_hit)
        self.assertEqual(cache_strategy, "miss")
        self.assertIsNone(answer_text)
        self.assertFalse(notices)
        self.assertEqual(backend_label, "advanced[searxng, tavily, serpapi]")
        self.assertEqual(len(payload["results"]), 3)
        self.assertEqual(payload["ranking"]["strategy"], "rules")
        self.assertFalse(payload["ranking"]["rerank_applied"])

    def test_advanced_search_defaults_to_snippet_mode_without_override(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    }
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng"],
                "fetch_full_page": True,
                "advanced_search_fetch_full_page_override": None,
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        services_search.dispatch_search("snippet mode query", config, 0)

        self.assertEqual(len(DummySearchTool.calls), 1)
        self.assertFalse(DummySearchTool.calls[0]["fetch_full_page"])

    def test_advanced_search_skips_slow_backend_after_deadline(self):
        def _slow_response(payload):
            time.sleep(0.35)
            return {
                "results": [
                    {
                        "title": "Slow result",
                        "url": "https://example.com/slow",
                        "content": "slow",
                    }
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            }

        def _fast_response(payload):
            time.sleep(0.02)
            return {
                "results": [
                    {
                        "title": "Fast result",
                        "url": "https://example.com/fast",
                        "content": "fast",
                    }
                ],
                "backend": "tavily",
                "answer": None,
                "notices": [],
            }

        DummySearchTool.responses = {
            "searxng": _slow_response,
            "tavily": _fast_response,
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng", "tavily"],
                "advanced_backend_timeout_seconds": 0.1,
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        started_at = time.perf_counter()
        payload, notices, _, backend_label, _, _ = services_search.dispatch_search(
            "deadline query",
            config,
            0,
        )
        duration = time.perf_counter() - started_at

        self.assertLess(duration, 0.32)
        self.assertEqual(backend_label, "advanced[tavily]")
        self.assertEqual([item["url"] for item in payload["results"]], ["https://example.com/fast"])
        self.assertTrue(any("searxng 搜索超时" in notice for notice in notices))

    def test_private_rerank_endpoints_bypass_proxy_env(self):
        self.assertTrue(services_search._should_bypass_proxy_for_url("http://127.0.0.1:8082/v1/rerank"))
        self.assertTrue(services_search._should_bypass_proxy_for_url("http://192.168.1.136:8082/v1/rerank"))
        self.assertFalse(services_search._should_bypass_proxy_for_url("https://api.example.com/v1/rerank"))

    def test_dispatch_search_reranks_after_deduplication_and_reorders_candidate_pool(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha from SearXNG",
                        "url": "https://example.com/a?utm_source=feed",
                        "content": "alpha short",
                    },
                    {
                        "title": "Beta from SearXNG",
                        "url": "https://example.com/b",
                        "content": "beta content",
                    },
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
            "tavily": {
                "results": [
                    {
                        "title": "Alpha from Tavily",
                        "url": "https://example.com/a",
                        "content": "alpha much longer content from tavily",
                    },
                    {
                        "title": "Gamma from Tavily",
                        "url": "https://example.com/c",
                        "content": "gamma content",
                    },
                    {
                        "title": "Delta from Tavily",
                        "url": "https://example.com/d",
                        "content": "delta content",
                    },
                ],
                "backend": "tavily",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng", "tavily"],
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1",
                "advanced_rerank_model": "qwen-rerank",
                "advanced_rerank_candidate_pool": 3,
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        with patch.object(
            services_search.requests,
            "post",
            return_value=MockHTTPResponse(status_code=200, payload=rerank_api_payload([1, 0, 2])),
        ) as mock_post:
            payload, notices, answer_text, backend_label, cache_hit, cache_strategy = services_search.dispatch_search(
                "rerank query",
                config,
                0,
                max_results=2,
            )

        self.assertFalse(cache_hit)
        self.assertEqual(cache_strategy, "miss")
        self.assertEqual(answer_text, None)
        self.assertEqual(backend_label, "advanced[searxng, tavily]")
        self.assertFalse(notices)
        self.assertEqual(
            [item["url"] for item in payload["results"]],
            ["https://example.com/b", "https://example.com/a?utm_source=feed"],
        )
        self.assertEqual(
            payload["ranking"],
            {
                "strategy": "rules+llm_rerank",
                "rerank_applied": True,
                "candidate_count": 4,
                "model": "qwen-rerank",
                "fallback_reason": None,
            },
        )
        self.assertEqual(mock_post.call_args.args[0], "http://rerank.local/v1/rerank")
        rerank_documents = mock_post.call_args.kwargs["json"]["documents"]
        self.assertEqual(mock_post.call_args.kwargs["json"]["query"], "rerank query")
        self.assertEqual(mock_post.call_args.kwargs["json"]["top_n"], 3)
        self.assertEqual(len(rerank_documents), 3)
        self.assertTrue(any("https://example.com/a?utm_source=feed" in document for document in rerank_documents))
        self.assertTrue(any("https://example.com/b" in document for document in rerank_documents))
        self.assertTrue(any("https://example.com/c" in document for document in rerank_documents))

    def test_dispatch_search_falls_back_to_chat_completions_when_rerank_endpoint_is_unsupported(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    },
                    {
                        "title": "Beta",
                        "url": "https://example.com/b",
                        "content": "beta",
                    },
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng"],
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1",
                "advanced_rerank_model": "qwen-rerank",
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        def _mock_post(url, **kwargs):
            if url.endswith("/v1/rerank"):
                return MockHTTPResponse(status_code=404, payload={"detail": "Not Found"}, text="Not Found")
            if url.endswith("/v1/chat/completions"):
                return MockHTTPResponse(status_code=200, payload=rerank_completion_payload(["doc-2", "doc-1"]))
            raise AssertionError(f"unexpected rerank URL: {url}")

        with patch.object(services_search.requests, "post", side_effect=_mock_post) as mock_post:
            payload, notices, _, _, _, _ = services_search.dispatch_search(
                "fallback rerank query",
                config,
                0,
            )

        self.assertFalse(notices)
        self.assertEqual(
            [item["url"] for item in payload["results"]],
            ["https://example.com/b", "https://example.com/a"],
        )
        self.assertTrue(payload["ranking"]["rerank_applied"])
        self.assertEqual(
            [call.args[0] for call in mock_post.call_args_list],
            ["http://rerank.local/v1/rerank", "http://rerank.local/v1/chat/completions"],
        )

    def test_dispatch_search_rerank_invalid_results_falls_back_to_rule_order(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    },
                    {
                        "title": "Beta",
                        "url": "https://example.com/b",
                        "content": "beta",
                    },
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng"],
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1",
                "advanced_rerank_model": "qwen-rerank",
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        with patch.object(
            services_search.requests,
            "post",
            return_value=MockHTTPResponse(status_code=200, payload=rerank_api_payload([0])),
        ):
            payload, notices, _, _, _, _ = services_search.dispatch_search(
                "invalid rerank indices",
                config,
                0,
            )

        self.assertEqual([item["url"] for item in payload["results"]], [
            "https://example.com/a",
            "https://example.com/b",
        ])
        self.assertEqual(payload["ranking"]["strategy"], "rules")
        self.assertFalse(payload["ranking"]["rerank_applied"])
        self.assertEqual(payload["ranking"]["fallback_reason"], "rerank_invalid_results")
        self.assertTrue(any("advanced rerank 回退" in notice for notice in notices))

    def test_dispatch_search_rerank_timeout_falls_back_without_failing_advanced_search(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    }
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
            "tavily": {
                "results": [
                    {
                        "title": "Beta",
                        "url": "https://example.com/b",
                        "content": "beta",
                    }
                ],
                "backend": "tavily",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng", "tavily"],
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1/chat/completions",
                "advanced_rerank_model": "qwen-rerank",
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        with patch.object(
            services_search.requests,
            "post",
            side_effect=services_search.requests.Timeout("timed out"),
        ):
            payload, notices, _, _, _, _ = services_search.dispatch_search(
                "rerank timeout",
                config,
                0,
            )

        self.assertEqual([item["url"] for item in payload["results"]], [
            "https://example.com/a",
            "https://example.com/b",
        ])
        self.assertEqual(payload["ranking"]["fallback_reason"], "rerank_timeout")
        self.assertTrue(any("advanced rerank 回退" in notice for notice in notices))

    def test_dispatch_search_rerank_http_error_falls_back_to_rule_order(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    },
                    {
                        "title": "Beta",
                        "url": "https://example.com/b",
                        "content": "beta",
                    },
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng"],
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1/chat/completions",
                "advanced_rerank_model": "qwen-rerank",
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        with patch.object(
            services_search.requests,
            "post",
            return_value=MockHTTPResponse(status_code=500, payload={"error": "backend failed"}, text="backend failed"),
        ):
            payload, notices, _, _, _, _ = services_search.dispatch_search(
                "rerank http error",
                config,
                0,
            )

        self.assertEqual(payload["ranking"]["fallback_reason"], "rerank_http_error")
        self.assertTrue(any("advanced rerank 回退" in notice for notice in notices))
        self.assertEqual([item["url"] for item in payload["results"]], [
            "https://example.com/a",
            "https://example.com/b",
        ])

    def test_dispatch_search_rerank_invalid_json_falls_back_to_rule_order(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    },
                    {
                        "title": "Beta",
                        "url": "https://example.com/b",
                        "content": "beta",
                    },
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng"],
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1/chat/completions",
                "advanced_rerank_model": "qwen-rerank",
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        with patch.object(
            services_search.requests,
            "post",
            return_value=MockHTTPResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "not valid json"}}]},
            ),
        ):
            payload, notices, _, _, _, _ = services_search.dispatch_search(
                "rerank invalid json",
                config,
                0,
            )

        self.assertEqual(payload["ranking"]["fallback_reason"], "rerank_response_invalid_json")
        self.assertTrue(any("advanced rerank 回退" in notice for notice in notices))
        self.assertEqual([item["url"] for item in payload["results"]], [
            "https://example.com/a",
            "https://example.com/b",
        ])

    def test_dispatch_search_exact_cache_isolated_by_rerank_signature(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    }
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
        }
        base_overrides = {
            "search_api": "advanced",
            "advanced_search_backends": ["searxng"],
            "search_cache_enabled": True,
            "search_cache_ttl_seconds": 900,
            "search_cache_dir": self.temp_dir.name,
            "search_cache_vector_dir": self.temp_dir.name,
            "semantic_cache_enabled": False,
        }
        config_without_rerank = Configuration.from_env(
            overrides=base_overrides,
            load_env_file=False,
        )
        config_with_rerank = Configuration.from_env(
            overrides={
                **base_overrides,
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1",
                "advanced_rerank_model": "qwen-rerank",
            },
            load_env_file=False,
        )

        with patch.object(
            services_search.requests,
            "post",
            return_value=MockHTTPResponse(status_code=200, payload=rerank_api_payload([0])),
        ):
            first = services_search.dispatch_search("same advanced query", config_without_rerank, 0)
            second = services_search.dispatch_search("same advanced query", config_with_rerank, 1)

        self.assertEqual(DummySearchTool.call_count, 2)
        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(first[5], "miss")
        self.assertEqual(second[5], "miss")

    def test_dispatch_search_semantic_cache_isolated_by_rerank_signature(self):
        DummySentenceTransformer.embeddings = {
            "advanced cache query one": [1.0, 0.0, 0.0],
            "advanced cache query two": [1.0, 0.0, 0.0],
        }
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    }
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
        }
        base_overrides = {
            "search_api": "advanced",
            "advanced_search_backends": ["searxng"],
            "search_cache_enabled": True,
            "search_cache_ttl_seconds": 900,
            "search_cache_dir": self.temp_dir.name,
            "search_cache_vector_dir": self.temp_dir.name,
            "semantic_cache_enabled": True,
            "semantic_cache_embedding_model": "dummy-minilm",
            "semantic_cache_similarity_threshold": 0.90,
        }
        config_a = Configuration.from_env(
            overrides={
                **base_overrides,
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1",
                "advanced_rerank_model": "qwen-rerank",
                "advanced_rerank_candidate_pool": 5,
            },
            load_env_file=False,
        )
        config_b = Configuration.from_env(
            overrides={
                **base_overrides,
                "advanced_rerank_enabled": True,
                "advanced_rerank_base_url": "http://rerank.local/v1",
                "advanced_rerank_model": "qwen-rerank",
                "advanced_rerank_candidate_pool": 6,
            },
            load_env_file=False,
        )

        with patch.object(
            services_search.requests,
            "post",
            return_value=MockHTTPResponse(status_code=200, payload=rerank_api_payload([0])),
        ):
            first = services_search.dispatch_search("advanced cache query one", config_a, 0)
            second = services_search.dispatch_search("advanced cache query two", config_b, 1)

        self.assertEqual(DummySearchTool.call_count, 2)
        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(second[5], "miss")

    def test_dispatch_search_advanced_fetch_full_page_override_is_applied_to_backend_calls(self):
        DummySearchTool.responses = {
            "searxng": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "content": "alpha",
                    }
                ],
                "backend": "searxng",
                "answer": None,
                "notices": [],
            },
            "tavily": {
                "results": [
                    {
                        "title": "Beta",
                        "url": "https://example.com/b",
                        "content": "beta",
                    }
                ],
                "backend": "tavily",
                "answer": None,
                "notices": [],
            },
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "advanced",
                "advanced_search_backends": ["searxng", "tavily"],
                "fetch_full_page": True,
                "advanced_search_fetch_full_page_override": False,
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        services_search.dispatch_search("advanced fetch override", config, 0)

        self.assertEqual(
            [call["fetch_full_page"] for call in DummySearchTool.calls],
            [False, False],
        )

    def test_dispatch_search_uses_semantic_cache_for_similar_queries(self):
        DummySentenceTransformer.embeddings = {
            "multimodal llm capability progress": [1.0, 0.0, 0.0],
            "multimodal model capability advances": [1.0, 0.0, 0.0],
        }
        observer = RequestTrace(
            request_id="req-semantic-cache",
            topic="semantic cache test",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": True,
                "semantic_cache_embedding_model": "dummy-minilm",
                "semantic_cache_similarity_threshold": 0.90,
            },
            load_env_file=False,
        )

        first = services_search.dispatch_search(
            "multimodal llm capability progress", config, 0, observer=observer
        )
        second = services_search.dispatch_search(
            "multimodal model capability advances", config, 1, observer=observer
        )

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertFalse(first[4])
        self.assertTrue(second[4])
        self.assertEqual(second[5], "semantic_ann")
        self.assertEqual(observer.snapshot()["cache_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_semantic_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_misses"], 1)
        self.assertEqual(observer.snapshot()["last_search_cache_details"]["cache_hit_mode"], "semantic_ann")

    def test_dispatch_search_uses_semantic_cache_for_fresh_queries_with_same_topic_context(self):
        DummySentenceTransformer.embeddings = {
            "探索多模态大模型在 2025 年的关键突破 探索多模态大模型在 2025 年的关键突破 技术架构创新": [1.0, 0.0, 0.0],
            "探索多模态大模型在 2025 年的关键突破 探索多模态大模型在 2025 年的关键突破 核心能力突破": [1.0, 0.0, 0.0],
        }
        observer = RequestTrace(
            request_id="req-semantic-cache-fresh-topic",
            topic="fresh semantic cache test",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": True,
                "semantic_cache_embedding_model": "dummy-minilm",
                "semantic_cache_similarity_threshold": 0.90,
            },
            load_env_file=False,
        )
        cache_context = {"research_topic": "探索多模态大模型在 2025 年的关键突破"}

        first = services_search.dispatch_search(
            "探索多模态大模型在 2025 年的关键突破 技术架构创新",
            config,
            0,
            observer=observer,
            cache_context=cache_context,
        )
        second = services_search.dispatch_search(
            "探索多模态大模型在 2025 年的关键突破 核心能力突破",
            config,
            1,
            observer=observer,
            cache_context=cache_context,
        )

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertFalse(first[4])
        self.assertTrue(second[4])
        self.assertEqual(second[5], "semantic_ann")
        self.assertEqual(observer.snapshot()["cache_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_semantic_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_misses"], 1)
        self.assertEqual(observer.snapshot()["last_search_cache_details"]["cache_hit_mode"], "semantic_ann")

    def test_dispatch_search_skips_semantic_cache_for_fresh_queries(self):
        DummySentenceTransformer.embeddings = {
            "multimodal llm progress in 2025": [1.0, 0.0, 0.0],
            "2025 multimodal model advances": [1.0, 0.0, 0.0],
        }
        observer = RequestTrace(
            request_id="req-semantic-cache-fresh",
            topic="semantic cache fresh test",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": True,
                "semantic_cache_embedding_model": "dummy-minilm",
                "semantic_cache_similarity_threshold": 0.90,
            },
            load_env_file=False,
        )

        first = services_search.dispatch_search(
            "multimodal llm progress in 2025", config, 0, observer=observer
        )
        second = services_search.dispatch_search(
            "2025 multimodal model advances", config, 1, observer=observer
        )

        self.assertEqual(DummySearchTool.call_count, 2)
        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(second[5], "miss")
        self.assertEqual(observer.snapshot()["cache_hits"], 0)
        self.assertEqual(observer.snapshot()["cache_semantic_hits"], 0)
        self.assertEqual(observer.snapshot()["cache_misses"], 2)

    def test_dispatch_search_does_not_index_fresh_results_for_semantic_reuse(self):
        DummySentenceTransformer.embeddings = {
            "multimodal llm progress in 2025": [1.0, 0.0, 0.0],
            "multimodal llm capability progress": [1.0, 0.0, 0.0],
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": True,
                "semantic_cache_embedding_model": "dummy-minilm",
                "semantic_cache_similarity_threshold": 0.90,
            },
            load_env_file=False,
        )

        first = services_search.dispatch_search("multimodal llm progress in 2025", config, 0)
        second = services_search.dispatch_search("multimodal llm capability progress", config, 1)

        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(second[5], "miss")
        self.assertEqual(DummySearchTool.call_count, 2)

    def test_dispatch_search_uses_lexical_semantic_cache_with_same_topic_context(self):
        observer = RequestTrace(
            request_id="req-lexical-cache",
            topic="多模态大模型前沿技术",
            search_api="duckduckgo",
            provider="custom",
            model="Qwen/Qwen3.5-27B",
            pricing_catalog={},
        )
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": True,
                "semantic_cache_similarity_threshold": 0.95,
                "semantic_cache_lexical_threshold": 0.35,
            },
            load_env_file=False,
        )

        first_context = {
            "research_topic": "多模态大模型前沿技术",
            "task_title": "架构创新与模型设计",
            "task_intent": "梳理多模态大模型的核心架构演进与融合机制",
        }
        second_context = {
            "research_topic": "多模态大模型前沿技术",
            "task_title": "架构创新调研",
            "task_intent": "梳理核心架构设计、跨模态融合与注意力机制创新",
        }

        with patch.object(services_search, "_embed_query", return_value=None):
            first = services_search.dispatch_search(
                "多模态大模型 架构设计 跨模态融合 注意力机制",
                config,
                0,
                observer=observer,
                cache_context=first_context,
            )
            second = services_search.dispatch_search(
                "多模态大模型 架构创新 跨模态融合 注意力机制",
                config,
                1,
                observer=observer,
                cache_context=second_context,
            )

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertFalse(first[4])
        self.assertEqual(first[5], "miss")
        self.assertTrue(second[4])
        self.assertEqual(second[5], "semantic_lexical")
        self.assertEqual(observer.snapshot()["cache_semantic_hits"], 1)
        metrics_snapshot = metrics_registry.snapshot()
        self.assertEqual(metrics_snapshot["cache_semantic_hit_total"], 1)

    def test_dispatch_search_ignores_ann_candidates_when_cached_payload_is_missing(self):
        DummySentenceTransformer.embeddings = {
            "model serving latency optimization": [1.0, 0.0, 0.0],
            "latency optimization for model serving": [1.0, 0.0, 0.0],
        }
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": True,
                "semantic_cache_embedding_model": "dummy-minilm",
                "semantic_cache_similarity_threshold": 0.90,
            },
            load_env_file=False,
        )

        first = services_search.dispatch_search("model serving latency optimization", config, 0)
        cache_key = services_search._build_cache_key("model serving latency optimization", "duckduckgo", config)
        services_search._get_disk_cache(config).delete(cache_key)
        second = services_search.dispatch_search("latency optimization for model serving", config, 1)

        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(DummySearchTool.call_count, 2)

    def test_dispatch_search_does_not_reuse_semantic_cache_across_topics(self):
        observer = RequestTrace(
            request_id="req-topic-isolation",
            topic="多模态大模型前沿技术",
            search_api="duckduckgo",
            provider="custom",
            model="Qwen/Qwen3.5-27B",
            pricing_catalog={},
        )
        config = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": True,
                "semantic_cache_similarity_threshold": 0.95,
                "semantic_cache_lexical_threshold": 0.35,
            },
            load_env_file=False,
        )

        first_context = {
            "research_topic": "多模态大模型前沿技术",
            "task_title": "架构创新调研",
            "task_intent": "梳理核心架构设计、跨模态融合与注意力机制创新",
        }
        second_context = {
            "research_topic": "医疗多模态模型落地实践",
            "task_title": "架构创新调研",
            "task_intent": "梳理核心架构设计、跨模态融合与注意力机制创新",
        }

        with patch.object(services_search, "_embed_query", return_value=None):
            first = services_search.dispatch_search(
                "多模态大模型 架构设计 跨模态融合 注意力机制",
                config,
                0,
                observer=observer,
                cache_context=first_context,
            )
            second = services_search.dispatch_search(
                "多模态模型 架构创新 跨模态融合 注意力创新",
                config,
                1,
                observer=observer,
                cache_context=second_context,
            )

        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(second[5], "miss")
        self.assertEqual(DummySearchTool.call_count, 2)

    def test_dispatch_search_normalizes_semanticscholar_results_without_fetching_full_page(self):
        config = Configuration.from_env(
            overrides={
                "search_api": "semanticscholar",
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
                "fetch_full_page": True,
                "semantic_scholar_api_key": "test-semantic-key",
            },
            load_env_file=False,
        )
        response_payload = {
            "total": 1,
            "data": [
                {
                    "paperId": "paper-1",
                    "url": "https://www.semanticscholar.org/paper/paper-1",
                    "title": "Semantic Scholar Powered Research",
                    "abstract": "This paper evaluates grounded research workflows.",
                    "year": 2025,
                    "publicationDate": "2025-02-14",
                    "citationCount": 321,
                    "authors": [{"name": "Ada Lovelace"}, {"name": "Alan Turing"}],
                    "venue": "Journal of Agent Systems",
                    "publicationTypes": ["JournalArticle"],
                    "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                    "tldr": {"text": "A concise overview of the paper."},
                }
            ],
        }

        with patch.object(
            services_search.requests,
            "get",
            return_value=MockHTTPResponse(status_code=200, payload=response_payload),
        ) as mock_get:
            payload, notices, answer_text, backend_label, cache_hit, cache_strategy = services_search.dispatch_search(
                "grounded research papers",
                config,
                0,
            )

        self.assertEqual(mock_get.call_count, 1)
        self.assertFalse(cache_hit)
        self.assertEqual(cache_strategy, "miss")
        self.assertEqual(notices, [])
        self.assertIsNone(answer_text)
        self.assertEqual(backend_label, "semanticscholar")
        self.assertEqual(len(payload["results"]), 1)
        first = payload["results"][0]
        self.assertEqual(first["url"], "https://www.semanticscholar.org/paper/paper-1")
        self.assertEqual(first["paper_id"], "paper-1")
        self.assertEqual(first["published_at"], "2025-02-14")
        self.assertEqual(first["citation_count"], 321)
        self.assertEqual(first["open_access_pdf_url"], "https://example.com/paper.pdf")
        self.assertIn("Semantic Scholar Powered Research", first["content"])
        self.assertIn("Ada Lovelace", first["content"])
        self.assertIn("This paper evaluates grounded research workflows.", first["content"])
        self.assertIn("https://example.com/paper.pdf", first["raw_content"])
        self.assertIn("x-api-key", mock_get.call_args.kwargs["headers"] or {})

    def test_dispatch_search_semanticscholar_reuses_cache_only_with_same_backend(self):
        config_duckduckgo = Configuration.from_env(
            overrides={
                "search_api": "duckduckgo",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )
        config_semanticscholar = Configuration.from_env(
            overrides={
                "search_api": "semanticscholar",
                "search_cache_enabled": True,
                "search_cache_ttl_seconds": 900,
                "search_cache_dir": self.temp_dir.name,
                "search_cache_vector_dir": self.temp_dir.name,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        response_payload = {
            "total": 1,
            "data": [
                {
                    "paperId": "paper-2",
                    "title": "Cache Isolation for Semantic Scholar",
                    "abstract": "Testing backend-specific cache namespaces.",
                    "year": 2024,
                }
            ],
        }

        services_search.dispatch_search("same query", config_duckduckgo, 0)
        with patch.object(
            services_search.requests,
            "get",
            return_value=MockHTTPResponse(status_code=200, payload=response_payload),
        ) as mock_get:
            first = services_search.dispatch_search("same query", config_semanticscholar, 0)
            second = services_search.dispatch_search("same query", config_semanticscholar, 1)

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertEqual(mock_get.call_count, 1)
        self.assertFalse(first[4])
        self.assertTrue(second[4])
        self.assertEqual(second[5], "exact")

    def test_dispatch_search_semanticscholar_raises_clear_auth_and_rate_limit_errors(self):
        config = Configuration.from_env(
            overrides={
                "search_api": "semanticscholar",
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )

        status_expectations = {
            401: "SEMANTIC_SCHOLAR_API_KEY 是否有效",
            403: "SEMANTIC_SCHOLAR_API_KEY 权限",
            429: "SEMANTIC_SCHOLAR_API_KEY",
            500: "服务暂时不可用",
        }

        for status_code, message in status_expectations.items():
            with self.subTest(status_code=status_code):
                with patch.object(
                    services_search.requests,
                    "get",
                    return_value=MockHTTPResponse(status_code=status_code, payload={"code": status_code}),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        services_search.dispatch_search("same query", config, 0)


if __name__ == "__main__":
    unittest.main()
