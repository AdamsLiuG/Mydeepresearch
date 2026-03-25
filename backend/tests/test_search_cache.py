import importlib
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
        return self._store.get(key, default)

    def set(self, key, value, expire=None):
        self._store[key] = value

    def clear(self):
        self._store.clear()

    def close(self):
        return None


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


diskcache_stub = types.ModuleType("diskcache")
diskcache_stub.Cache = DummyDiskCache
sys.modules["diskcache"] = diskcache_stub

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.SentenceTransformer = DummySentenceTransformer
sys.modules["sentence_transformers"] = sentence_transformers_stub

tools_stub = types.ModuleType("hello_agents.tools")
tools_stub.SearchTool = DummySearchTool
sys.modules["hello_agents.tools"] = tools_stub

services_search = importlib.reload(importlib.import_module("services.search"))


class SearchCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        DummySearchTool.call_count = 0
        DummySearchTool.calls = []
        DummySearchTool.responses = {}
        DummySentenceTransformer.embeddings = {}
        metrics_registry.reset()
        services_search.clear_search_cache()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        services_search.clear_search_cache()
        self.temp_dir.cleanup()

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
        metrics_snapshot = metrics_registry.snapshot()
        self.assertEqual(metrics_snapshot["cache_hit_total"], 1)
        self.assertEqual(metrics_snapshot["cache_exact_hit_total"], 1)
        self.assertEqual(metrics_snapshot["cache_semantic_hit_total"], 0)
        self.assertEqual(metrics_snapshot["cache_miss_total"], 1)

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

    def test_dispatch_search_uses_semantic_cache_for_similar_queries(self):
        DummySentenceTransformer.embeddings = {
            "multimodal llm progress in 2025": [1.0, 0.0, 0.0],
            "2025 multimodal model advances": [1.0, 0.0, 0.0],
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

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertFalse(first[4])
        self.assertTrue(second[4])
        self.assertEqual(second[5], "semantic")
        self.assertEqual(observer.snapshot()["cache_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_semantic_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_misses"], 1)

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
                "多模态大模型 架构 视觉语言融合 最新研究 2024 2025",
                config,
                0,
                observer=observer,
                cache_context=first_context,
            )
            second = services_search.dispatch_search(
                "多模态大模型 架构设计 2024 2025 跨模态融合 注意力机制",
                config,
                1,
                observer=observer,
                cache_context=second_context,
            )

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertFalse(first[4])
        self.assertEqual(first[5], "miss")
        self.assertTrue(second[4])
        self.assertEqual(second[5], "semantic")
        self.assertEqual(observer.snapshot()["cache_semantic_hits"], 1)
        metrics_snapshot = metrics_registry.snapshot()
        self.assertEqual(metrics_snapshot["cache_semantic_hit_total"], 1)

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
                "多模态大模型 架构设计 2024 2025 跨模态融合 注意力机制",
                config,
                0,
                observer=observer,
                cache_context=first_context,
            )
            second = services_search.dispatch_search(
                "多模态模型 架构设计 2025 跨模态融合 注意力创新",
                config,
                1,
                observer=observer,
                cache_context=second_context,
            )

        self.assertFalse(first[4])
        self.assertFalse(second[4])
        self.assertEqual(second[5], "miss")
        self.assertEqual(DummySearchTool.call_count, 2)


if __name__ == "__main__":
    unittest.main()
