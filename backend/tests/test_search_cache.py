import importlib
import sys
import types
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from metrics import RequestTrace, metrics_registry


class DummySearchTool:
    call_count = 0

    def __init__(self, backend="hybrid"):
        self.backend = backend

    def run(self, payload):
        DummySearchTool.call_count += 1
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


tools_stub = types.ModuleType("hello_agents.tools")
tools_stub.SearchTool = DummySearchTool
sys.modules["hello_agents.tools"] = tools_stub

services_search = importlib.reload(importlib.import_module("services.search"))


class SearchCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        DummySearchTool.call_count = 0
        metrics_registry.reset()
        services_search.clear_search_cache()

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
            },
            load_env_file=False,
        )

        first = services_search.dispatch_search("same query", config, 0, observer=observer)
        second = services_search.dispatch_search("same query", config, 1, observer=observer)

        self.assertEqual(DummySearchTool.call_count, 1)
        self.assertFalse(first[4])
        self.assertTrue(second[4])
        self.assertEqual(observer.snapshot()["cache_hits"], 1)
        self.assertEqual(observer.snapshot()["cache_misses"], 1)
        self.assertEqual(metrics_registry.snapshot()["counters"]["cache_hit_total"], 1)
        self.assertEqual(metrics_registry.snapshot()["counters"]["cache_miss_total"], 1)


if __name__ == "__main__":
    unittest.main()
