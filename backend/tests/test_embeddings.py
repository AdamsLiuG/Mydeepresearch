import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from services import embeddings


class RecordingSentenceTransformer:
    calls = []
    fail_on_local_only = False

    def __init__(self, model_name, **kwargs):
        RecordingSentenceTransformer.calls.append((model_name, dict(kwargs)))
        if self.fail_on_local_only and kwargs.get("local_files_only"):
            raise TypeError("__init__() got an unexpected keyword argument 'local_files_only'")
        self.model_name = model_name


class EmbeddingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        embeddings._MODEL_CACHE.clear()
        RecordingSentenceTransformer.calls = []
        RecordingSentenceTransformer.fail_on_local_only = False

    def tearDown(self) -> None:
        embeddings._MODEL_CACHE.clear()

    def test_load_sentence_transformer_prefers_local_cache(self):
        with patch.object(embeddings, "SentenceTransformer", RecordingSentenceTransformer):
            with patch.object(embeddings, "_has_local_sentence_transformer_cache", return_value=True):
                model = embeddings.load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

        self.assertIsInstance(model, RecordingSentenceTransformer)
        self.assertEqual(
            RecordingSentenceTransformer.calls,
            [("sentence-transformers/all-MiniLM-L6-v2", {"local_files_only": True})],
        )

    def test_load_sentence_transformer_uses_default_lookup_without_cache(self):
        with patch.object(embeddings, "SentenceTransformer", RecordingSentenceTransformer):
            with patch.object(embeddings, "_has_local_sentence_transformer_cache", return_value=False):
                model = embeddings.load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

        self.assertIsInstance(model, RecordingSentenceTransformer)
        self.assertEqual(
            RecordingSentenceTransformer.calls,
            [("sentence-transformers/all-MiniLM-L6-v2", {})],
        )

    def test_load_sentence_transformer_falls_back_when_local_only_is_unsupported(self):
        RecordingSentenceTransformer.fail_on_local_only = True

        with patch.object(embeddings, "SentenceTransformer", RecordingSentenceTransformer):
            with patch.object(embeddings, "_has_local_sentence_transformer_cache", return_value=True):
                model = embeddings.load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

        self.assertIsInstance(model, RecordingSentenceTransformer)
        self.assertEqual(
            RecordingSentenceTransformer.calls,
            [
                ("sentence-transformers/all-MiniLM-L6-v2", {"local_files_only": True}),
                ("sentence-transformers/all-MiniLM-L6-v2", {}),
            ],
        )

    def test_local_cache_detection_accepts_existing_local_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "local-model"
            model_dir.mkdir()

            self.assertTrue(embeddings._has_local_sentence_transformer_cache(str(model_dir)))

    def test_local_cache_detection_accepts_hf_hub_snapshot_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            snapshot_dir = cache_root / "models--sentence-transformers--all-MiniLM-L6-v2"
            snapshot_dir.mkdir()

            with patch.object(embeddings, "_hf_hub_cache_dir", return_value=cache_root):
                self.assertTrue(
                    embeddings._has_local_sentence_transformer_cache(
                        "sentence-transformers/all-MiniLM-L6-v2"
                    )
                )


if __name__ == "__main__":
    unittest.main()
