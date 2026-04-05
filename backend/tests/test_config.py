import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import config


class ConfigurationTests(unittest.TestCase):
    def test_relative_notes_workspace_resolves_under_backend_root(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={"notes_workspace": "custom-notes"},
                load_env_file=False,
            )

        expected = str((config.backend_root() / "custom-notes").resolve(strict=False))
        self.assertEqual(cfg.notes_workspace, expected)

    def test_relative_strategy_memory_dir_resolves_under_backend_root(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={
                    "strategy_memory_dir": ".memory/strategies",
                    "semantic_cache_embedding_model": "demo-embedding",
                },
                load_env_file=False,
            )

        expected = str((config.backend_root() / ".memory/strategies").resolve(strict=False))
        self.assertEqual(cfg.strategy_memory_dir, expected)
        self.assertEqual(cfg.resolved_strategy_memory_embedding_model(), "demo-embedding")

    def test_string_overrides_are_normalized(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={
                    "search_api": "advanced",
                    "advanced_search_backends": " searxng, tavily, serpapi, searxng ",
                    "advanced_search_max_concurrency": "2",
                    "advanced_search_fetch_full_page_override": "false",
                    "advanced_rerank_enabled": "true",
                    "advanced_rerank_model": "qwen-rerank",
                    "advanced_rerank_candidate_pool": "12",
                    "advanced_rerank_timeout_seconds": "1.75",
                    "advanced_rerank_max_content_chars": "800",
                    "log_level": "debug",
                    "port": "9001",
                    "cors_origins": "http://localhost:5174, http://localhost:3000",
                    "semantic_cache_warmup_enabled": "false",
                    "semantic_cache_lexical_threshold": "0.81",
                    "max_agent_tasks": "3",
                    "request_reflection_enabled": "true",
                    "reflection_max_additional_tasks": "2",
                    "task_query_rewrite_enabled": "false",
                    "search_tool_timeout_seconds": "4.5",
                    "search_tool_retry_attempts": "2",
                    "search_tool_retry_backoff_seconds": "0.2",
                },
                load_env_file=False,
            )

        self.assertEqual(cfg.search_api, config.SearchAPI.ADVANCED)
        self.assertEqual(
            cfg.advanced_search_backends,
            ["searxng", "tavily", "serpapi"],
        )
        self.assertEqual(cfg.advanced_search_max_concurrency, 2)
        self.assertFalse(cfg.advanced_search_fetch_full_page_override)
        self.assertTrue(cfg.advanced_rerank_enabled)
        self.assertEqual(cfg.advanced_rerank_model, "qwen-rerank")
        self.assertEqual(cfg.advanced_rerank_candidate_pool, 12)
        self.assertEqual(cfg.advanced_rerank_timeout_seconds, 1.75)
        self.assertEqual(cfg.advanced_rerank_max_content_chars, 800)
        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertEqual(cfg.port, 9001)
        self.assertFalse(cfg.semantic_cache_warmup_enabled)
        self.assertEqual(cfg.semantic_cache_lexical_threshold, 0.81)
        self.assertEqual(cfg.max_agent_tasks, 3)
        self.assertTrue(cfg.request_reflection_enabled)
        self.assertEqual(cfg.reflection_max_additional_tasks, 2)
        self.assertFalse(cfg.task_query_rewrite_enabled)
        self.assertEqual(cfg.search_tool_timeout_seconds, 4.5)
        self.assertEqual(cfg.search_tool_retry_attempts, 2)
        self.assertEqual(cfg.search_tool_retry_backoff_seconds, 0.2)
        self.assertEqual(
            cfg.cors_origins,
            [
                "http://localhost:5174",
                "http://127.0.0.1:5174",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ],
        )

    def test_loopback_cors_origins_expand_to_localhost_and_127(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={
                    "cors_origins": ["http://127.0.0.1:5174"],
                },
                load_env_file=False,
            )

        self.assertEqual(
            cfg.cors_origins,
            ["http://127.0.0.1:5174", "http://localhost:5174"],
        )

    def test_perf_settings_are_normalized(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={
                    "benchmark_profile": "STUB",
                    "perf_thresholds_path": "perf/baselines/stub_baseline.json",
                    "perf_sample_interval_seconds": "0.25",
                    "request_state_dir": ".state/requests",
                },
                load_env_file=False,
            )

        expected_thresholds = str(
            (config.backend_root() / "perf/baselines/stub_baseline.json").resolve(strict=False)
        )
        expected_state_dir = str(
            (config.backend_root() / ".state/requests").resolve(strict=False)
        )
        self.assertEqual(cfg.benchmark_profile, "stub")
        self.assertEqual(cfg.perf_thresholds_path, expected_thresholds)
        self.assertEqual(cfg.perf_sample_interval_seconds, 0.25)
        self.assertEqual(cfg.request_state_dir, expected_state_dir)

    def test_semanticscholar_search_api_and_key_are_supported(self):
        with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "secret-key"}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={"search_api": "semanticscholar"},
                load_env_file=False,
            )

        self.assertEqual(cfg.search_api, config.SearchAPI.SEMANTICSCHOLAR)
        self.assertEqual(cfg.semantic_scholar_api_key, "secret-key")

    def test_advanced_backends_reject_semanticscholar(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported advanced_search_backends"):
                config.Configuration.from_env(
                    overrides={
                        "search_api": "advanced",
                        "advanced_search_backends": ["searxng", "semanticscholar"],
                    },
                    load_env_file=False,
                )

    def test_advanced_rerank_settings_fall_back_to_global_llm_config(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={
                    "llm_provider": "custom",
                    "llm_base_url": "http://localhost:8001/v1",
                    "llm_api_key": "secret-token",
                    "llm_model_id": "Qwen/Qwen3-32B",
                    "search_api": "advanced",
                    "advanced_rerank_enabled": True,
                },
                load_env_file=False,
            )

        self.assertEqual(cfg.resolved_advanced_rerank_base_url(), "http://localhost:8001/v1")
        self.assertEqual(cfg.resolved_advanced_rerank_api_key(), "secret-token")
        self.assertEqual(cfg.resolved_advanced_rerank_model(), "Qwen/Qwen3-32B")
        self.assertEqual(
            cfg.resolved_search_cache_signature("advanced"),
            {
                "advanced_search_backends": ["searxng", "tavily", "serpapi", "duckduckgo"],
                "advanced_search_fetch_full_page_override": None,
                "advanced_rerank_enabled": True,
                "advanced_rerank_model": "Qwen/Qwen3-32B",
                "advanced_rerank_candidate_pool": 20,
            },
        )

    def test_report_layout_mode_is_normalized_and_resolved(self):
        with patch.dict(os.environ, {"REPORT_LAYOUT_MODE": "FIXED"}, clear=True):
            cfg = config.Configuration.from_env(load_env_file=False)

        self.assertEqual(cfg.report_layout_mode, "fixed")
        self.assertEqual(cfg.resolved_report_layout_mode(), "fixed")


if __name__ == "__main__":
    unittest.main()
