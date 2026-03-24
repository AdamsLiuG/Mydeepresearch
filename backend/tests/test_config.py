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
        self.assertEqual(
            cfg.cors_origins,
            ["http://localhost:5174", "http://localhost:3000"],
        )


if __name__ == "__main__":
    unittest.main()
