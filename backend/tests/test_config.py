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

    def test_string_overrides_are_normalized(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.Configuration.from_env(
                overrides={
                    "search_api": "advanced",
                    "advanced_search_backends": " searxng, tavily, serpapi, searxng ",
                    "log_level": "debug",
                    "port": "9001",
                    "cors_origins": "http://localhost:5174, http://localhost:3000",
                    "semantic_cache_lexical_threshold": "0.81",
                },
                load_env_file=False,
            )

        self.assertEqual(cfg.search_api, config.SearchAPI.ADVANCED)
        self.assertEqual(
            cfg.advanced_search_backends,
            ["searxng", "tavily", "serpapi"],
        )
        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertEqual(cfg.port, 9001)
        self.assertEqual(cfg.semantic_cache_lexical_threshold, 0.81)
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
                },
                load_env_file=False,
            )

        expected_thresholds = str(
            (config.backend_root() / "perf/baselines/stub_baseline.json").resolve(strict=False)
        )
        self.assertEqual(cfg.benchmark_profile, "stub")
        self.assertEqual(cfg.perf_thresholds_path, expected_thresholds)
        self.assertEqual(cfg.perf_sample_interval_seconds, 0.25)


if __name__ == "__main__":
    unittest.main()
