import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import config
from metrics import MetricsRegistry


class ConfigurationTests(unittest.TestCase):
    def test_v4_web_extractor_defaults_disable_dynamic_extraction(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(load_env_file=False)

        self.assertFalse(cfg.v4_web_extractor_enabled)
        self.assertFalse(cfg.v4_browser_use_enabled)
        self.assertEqual(cfg.v4_web_l2_adapter, "off")
        self.assertEqual(cfg.v4_web_extractor_tenants, [])
        self.assertEqual(cfg.v4_web_l2_origin_rules, {})
        self.assertEqual(cfg.v4_web_l2_max_concurrency, 2)
        self.assertEqual(cfg.v4_web_l2_per_tenant, 1)
        self.assertEqual(cfg.v4_web_l2_timeout_seconds, 12)
        self.assertIsNone(cfg.v4_web_config_reason)

    def test_v4_web_extractor_alias_applies_only_when_canonical_key_is_absent(self):
        with patch.dict(os.environ, {"V4_WEB_EXTRACTION_ENABLED": "true"}, clear=True):
            legacy = config.Configuration.from_env(load_env_file=False)
            matching = config.Configuration.from_env(
                overrides={"v4_web_extractor_enabled": True}, load_env_file=False,
            )
            with self.assertLogs("config", level="WARNING") as logs:
                conflict = config.Configuration.from_env(
                    overrides={"v4_web_extractor_enabled": False}, load_env_file=False,
                )

        self.assertTrue(legacy.v4_web_extractor_enabled)
        self.assertTrue(matching.v4_web_extractor_enabled)
        self.assertFalse(conflict.v4_web_extractor_enabled)
        self.assertEqual(conflict.v4_web_config_reason, "web_extractor_alias_conflict")
        self.assertIn("web_extractor_alias_conflict", logs.output[0])

        with patch.dict(
            os.environ,
            {"V4_WEB_EXTRACTOR_ENABLED": "true", "V4_WEB_EXTRACTION_ENABLED": "false"},
            clear=True,
        ), self.assertLogs("config", level="WARNING"):
            conflict = config.Configuration.from_env(load_env_file=False)
        self.assertFalse(conflict.v4_web_extractor_enabled)
        self.assertEqual(conflict.v4_web_config_reason, "web_extractor_alias_conflict")

    def test_v4_web_extractor_accepts_trusted_reader_settings_and_keeps_them_private(self):
        env = {
            "V4_WEB_EXTRACTOR_ENABLED": "true",
            "V4_WEB_L2_ADAPTER": "reader_http",
            "V4_WEB_EXTRACTOR_TENANTS": '["private-tenant"]',
            "V4_WEB_L2_ORIGIN_RULES": '{"private-tenant":[{"origin":"https://document.example.org"}]}',
            "V4_WEB_READER_URL": "https://reader.example.org/read",
            "V4_WEB_READER_TOKEN": "secret-reader-token",
            "V4_WEB_L2_MAX_CONCURRENCY": "1",
            "V4_WEB_L2_TIMEOUT_SECONDS": "0.5",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = config.Configuration.from_env(load_env_file=False)

        self.assertTrue(cfg.v4_web_extractor_enabled)
        self.assertEqual(cfg.v4_web_l2_adapter, "reader_http")
        self.assertFalse(cfg.v4_browser_use_enabled)
        self.assertEqual(cfg.v4_web_extractor_tenants, ["private-tenant"])
        self.assertEqual(cfg.v4_web_l2_origin_rules["private-tenant"][0]["origin"], "https://document.example.org")
        self.assertEqual(cfg.v4_web_reader_url, "https://reader.example.org/read")
        self.assertEqual(cfg.v4_web_reader_token, "secret-reader-token")
        self.assertEqual(cfg.v4_web_l2_max_concurrency, 1)
        self.assertEqual(cfg.v4_web_l2_timeout_seconds, 0.5)
        self.assertIsNone(cfg.v4_web_config_reason)
        for output in (str(cfg), repr(cfg), str(cfg.model_dump()), cfg.model_dump_json()):
            for value in ("private-tenant", "document.example.org", "reader.example.org", "secret-reader-token"):
                self.assertNotIn(value, output)

    def test_invalid_v4_web_extractor_settings_fail_closed(self):
        for field, value, reason in (
            ("v4_web_extractor_enabled", "", "invalid_web_extractor_enabled"),
            ("v4_web_extractor_enabled", None, "invalid_web_extractor_enabled"),
            ("v4_web_extractor_tenants", ["default", 7], "invalid_web_extractor_tenants"),
            ("v4_web_extractor_tenants", ["*"], "invalid_web_extractor_tenants"),
        ):
            with self.subTest(field=field, value=value), self.assertLogs("config", level="WARNING"):
                cfg = config.Configuration.model_validate({"v4_web_extractor_enabled": True, field: value})
                self.assertFalse(cfg.v4_web_extractor_enabled)
                self.assertEqual(cfg.v4_web_l2_adapter, "off")
                self.assertEqual(cfg.v4_web_config_reason, reason)

    def test_invalid_v4_dynamic_settings_preserve_static_extraction(self):
        settings = {
            "v4_web_extractor_enabled": True,
            "v4_web_extractor_tenants": ["default"],
            "v4_web_l2_origin_rules": {"default": [{"origin": "https://example.org"}]},
            "v4_web_l2_adapter": "reader_http",
            "v4_web_reader_url": "https://reader.example.org/read",
            "v4_web_reader_token": "valid-secret-token",
        }
        for field, value, reason in (
            ("v4_web_l2_adapter", "", "invalid_web_l2_adapter"),
            ("v4_web_l2_adapter", 123, "invalid_web_l2_adapter"),
            ("v4_browser_use_enabled", "sometimes", "invalid_browser_use_enabled"),
            ("v4_web_l2_origin_rules", "", "invalid_web_l2_origin_rules"),
            ("v4_web_l2_origin_rules", {"default": [{"origin": "https://*.example.org"}]}, "invalid_web_l2_origin_rules"),
            ("v4_web_l2_origin_rules", {"default": [{"origin": "https://example.org", "path_prefix": "/a/%2Fb"}]}, "invalid_web_l2_origin_rules"),
            ("v4_web_reader_url", "http://reader.example.org/read", "invalid_web_reader_url"),
            ("v4_web_reader_url", "", "invalid_web_reader_url"),
            ("v4_web_reader_token", "", "invalid_web_reader_token"),
            ("v4_web_reader_token", "secret\r\nHeader", "invalid_web_reader_token"),
            ("v4_web_l2_max_concurrency", 0, "invalid_web_l2_limits"),
            ("v4_web_l2_max_concurrency", "3", "invalid_web_l2_limits"),
            ("v4_web_l2_per_tenant", True, "invalid_web_l2_limits"),
            ("v4_web_l2_per_tenant", 2, "invalid_web_l2_limits"),
            ("v4_web_l2_timeout_seconds", "nan", "invalid_web_l2_limits"),
            ("v4_web_l2_timeout_seconds", 12.1, "invalid_web_l2_limits"),
            ("v4_web_l2_timeout_seconds", "", "invalid_web_l2_limits"),
        ):
            with self.subTest(field=field, value=value), self.assertLogs("config", level="WARNING") as logs:
                with patch.dict(os.environ, {}, clear=True):
                    cfg = config.Configuration.from_env(
                        overrides={**settings, field: value}, load_env_file=False,
                    )
                self.assertTrue(cfg.v4_web_extractor_enabled)
                self.assertEqual(cfg.v4_web_l2_adapter, "off")
                self.assertEqual(cfg.v4_web_config_reason, reason)
                self.assertNotIn("secret", logs.output[0])

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

    def test_relative_search_cache_vector_dir_resolves_under_backend_root(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={"search_cache_vector_dir": ".cache/search/vector"},
                load_env_file=False,
            )

        expected = str((config.backend_root() / ".cache/search/vector").resolve(strict=False))
        self.assertEqual(cfg.search_cache_vector_dir, expected)
        self.assertEqual(cfg.resolved_search_cache_vector_dir(), expected)

    def test_relative_evidence_memory_dir_resolves_under_backend_root(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={"evidence_memory_dir": ".memory/evidence"},
                load_env_file=False,
            )

        expected = str((config.backend_root() / ".memory/evidence").resolve(strict=False))
        self.assertEqual(cfg.evidence_memory_dir, expected)

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
                    "evidence_runtime_enabled": "true",
                    "evidence_runtime_top_k": "9",
                    "evidence_chunk_chars": "950",
                    "evidence_chunk_overlap": "100",
                    "evidence_memory_enabled": "true",
                    "search_cache_dynamic_ttl_enabled": "true",
                    "search_cache_ttl_seconds": "43200",
                    "search_cache_fresh_ttl_seconds": "14400",
                    "search_cache_evergreen_ttl_seconds": "172800",
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
        self.assertTrue(cfg.evidence_runtime_enabled)
        self.assertEqual(cfg.evidence_runtime_top_k, 9)
        self.assertEqual(cfg.evidence_chunk_chars, 950)
        self.assertEqual(cfg.evidence_chunk_overlap, 100)
        self.assertTrue(cfg.evidence_memory_enabled)
        self.assertTrue(cfg.search_cache_dynamic_ttl_enabled)
        self.assertEqual(cfg.search_cache_ttl_seconds, 43200)
        self.assertEqual(cfg.search_cache_fresh_ttl_seconds, 14400)
        self.assertEqual(cfg.search_cache_evergreen_ttl_seconds, 172800)
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

    def test_approximate_cache_config_accepts_new_names_and_legacy_aliases(self):
        with patch.dict(
            os.environ,
            {
                "APPROXIMATE_CACHE_ENABLED": "true",
                "APPROXIMATE_CACHE_DENSE_ENABLED": "false",
                "APPROXIMATE_CACHE_SPARSE_ENABLED": "true",
                "APPROXIMATE_CACHE_DENSE_THRESHOLD": "0.88",
                "APPROXIMATE_CACHE_SPARSE_THRESHOLD": "0.42",
                "APPROXIMATE_CACHE_DENSE_TOP_K": "7",
                "APPROXIMATE_CACHE_SPARSE_TOP_K": "5",
            },
            clear=True,
        ):
            cfg = config.Configuration.from_env(load_env_file=False)

        self.assertTrue(cfg.resolved_approximate_cache_enabled())
        self.assertFalse(cfg.resolved_approximate_cache_dense_enabled())
        self.assertTrue(cfg.resolved_approximate_cache_sparse_enabled())
        self.assertEqual(cfg.resolved_approximate_cache_dense_threshold(), 0.88)
        self.assertEqual(cfg.resolved_approximate_cache_sparse_threshold(), 0.42)
        self.assertEqual(cfg.resolved_approximate_cache_dense_top_k(), 7)
        self.assertEqual(cfg.resolved_approximate_cache_sparse_top_k(), 5)

        with patch.dict(
            os.environ,
            {
                "SEMANTIC_CACHE_ENABLED": "false",
                "SEMANTIC_CACHE_SIMILARITY_THRESHOLD": "0.77",
                "SEMANTIC_CACHE_LEXICAL_THRESHOLD": "0.33",
            },
            clear=True,
        ):
            legacy_cfg = config.Configuration.from_env(load_env_file=False)

        self.assertFalse(legacy_cfg.resolved_approximate_cache_enabled())
        self.assertEqual(legacy_cfg.resolved_approximate_cache_dense_threshold(), 0.77)
        self.assertEqual(legacy_cfg.resolved_approximate_cache_sparse_threshold(), 0.33)

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

    def test_skill_pack_dir_is_normalized_and_budget_tiers_resolve(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={"skill_pack_dir": "./skills/research"},
                load_env_file=False,
            )

        expected_skills_dir = str(
            (config.backend_root() / "skills/research").resolve(strict=False)
        )
        self.assertEqual(cfg.skill_pack_dir, expected_skills_dir)
        self.assertEqual(cfg.resolved_initial_plan_max_tasks(), 4)
        self.assertEqual(cfg.resolved_max_total_tasks(), 7)

    def test_semanticscholar_search_api_and_key_are_supported(self):
        with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "secret-key"}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={"search_api": "semanticscholar"},
                load_env_file=False,
            )

        self.assertEqual(cfg.search_api, config.SearchAPI.SEMANTICSCHOLAR)
        self.assertEqual(cfg.semantic_scholar_api_key, "secret-key")

    def test_advanced_backends_allow_semanticscholar_and_reject_unknown(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={
                    "search_api": "advanced",
                    "advanced_search_backends": ["searxng", "semanticscholar"],
                },
                load_env_file=False,
            )
            self.assertEqual(cfg.advanced_search_backends, ["searxng", "semanticscholar"])

            with self.assertRaisesRegex(ValueError, "Unsupported advanced_search_backends"):
                config.Configuration.from_env(
                    overrides={
                        "search_api": "advanced",
                        "advanced_search_backends": ["searxng", "unknown_backend"],
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

    def test_fast_llm_config_tiering_defaults_and_env_resolution(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg_default = config.Configuration.from_env(
                overrides={"llm_model_id": "Qwen/Qwen3-32B"},
                load_env_file=False,
            )
            self.assertFalse(cfg_default.resolved_fast_llm_enabled())
            self.assertEqual(cfg_default.resolved_fast_llm_model(), "Qwen/Qwen3-32B")

        with patch.dict(
            os.environ,
            {
                "LLM_MODEL_ID": "Qwen/Qwen3-32B",
                "FAST_LLM_MODEL_ID": "Qwen/Qwen3-7B-Instruct",
                "FAST_LLM_PROVIDER": "vllm",
                "FAST_LLM_BASE_URL": "http://localhost:8002/v1",
                "FAST_LLM_TEMPERATURE": "0.1",
            },
            clear=True,
        ):
            cfg_tiered = config.Configuration.from_env(load_env_file=False)
            self.assertTrue(cfg_tiered.resolved_fast_llm_enabled())
            self.assertEqual(cfg_tiered.resolved_fast_llm_model(), "Qwen/Qwen3-7B-Instruct")
            self.assertEqual(cfg_tiered.fast_llm_provider, "vllm")
            self.assertEqual(cfg_tiered.fast_llm_base_url, "http://localhost:8002/v1")
            self.assertAlmostEqual(cfg_tiered.fast_llm_temperature, 0.1)


class WebExtractionMetricsTests(unittest.TestCase):
    def test_web_extraction_metrics_only_publish_observed_finite_labels(self):
        registry = MetricsRegistry()
        original = registry.snapshot()
        registry.record_web_extraction("result", tier="L1", adapter="static", outcome="degraded")
        registry.record_web_extraction("http_attempt", adapter="reader_http", kind="reader_service")
        registry.record_web_extraction("degraded", reason="timeout")
        registry.record_web_extraction("sanitizer_dropped_text", amount=4)
        registry.record_web_extraction("elapsed", adapter="static", elapsed_ms=10.5)
        registry.record_web_extraction("elapsed", adapter="static", elapsed_ms=2.5)
        registry.record_web_extraction("l2_inflight", amount=1)
        registry.record_web_extraction("result", tier="L1", adapter="https://private.example.org", outcome="success")
        registry.record_web_extraction("egress_denied", reason="private-tenant")
        registry.record_web_extraction("queue", adapter="static", elapsed_ms=float("nan"))

        updated = registry.snapshot()
        self.assertNotIn("web_extraction", original)
        self.assertEqual(updated["counters"]["web_extraction_result_total"], 1)
        self.assertEqual(updated["counters"]["web_extraction_http_attempt_total"], 1)
        self.assertEqual(updated["counters"]["web_extraction_degraded_total"], 1)
        self.assertEqual(updated["counters"]["web_extraction_sanitizer_dropped_text_total"], 4)
        self.assertNotIn("web_extraction_egress_denied_total", updated["counters"])
        self.assertNotIn("web_extraction_queue_ms", updated["latencies_ms"])
        self.assertEqual(updated["latencies_ms"]["web_extraction_elapsed_ms"]["avg_ms"], 6.5)
        self.assertEqual(updated["web_extraction"]["web_extraction_result_total"], [
            {"tier": "L1", "adapter": "static", "outcome": "degraded", "count": 1},
        ])
        self.assertEqual(updated["web_extraction"]["web_extraction_l2_inflight"], 1)
        self.assertNotIn("private", str(updated))

        registry.record_web_extraction("l2_inflight", amount=-1)
        self.assertEqual(registry.snapshot()["web_extraction"]["web_extraction_l2_inflight"], 0)
        registry.reset()
        self.assertNotIn("web_extraction", registry.snapshot())
        self.assertEqual(registry.snapshot()["counters"], original["counters"])


if __name__ == "__main__":
    unittest.main()
