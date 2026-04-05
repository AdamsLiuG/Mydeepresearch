"""Shared sentence-transformer embedding helpers."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

try:  # pragma: no cover - exercised through runtime fallback
    from huggingface_hub import constants as huggingface_constants
except Exception:  # pragma: no cover - exercised through runtime fallback
    huggingface_constants = None

try:  # pragma: no cover - exercised through runtime fallback
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - exercised through runtime fallback
    SentenceTransformer = None

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, Any] = {}


def _coerce_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]

    if not isinstance(value, list):
        return None

    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def embeddings_available() -> bool:
    """Return whether sentence-transformers is importable."""

    return SentenceTransformer is not None


def _hf_hub_cache_dir() -> Path | None:
    cache_dir = str(getattr(huggingface_constants, "HF_HUB_CACHE", "") or "").strip()
    if not cache_dir:
        return None
    return Path(cache_dir).expanduser()


def _candidate_local_model_paths(model_name: str) -> list[Path]:
    normalized_name = str(model_name or "").strip()
    if not normalized_name:
        return []

    candidates: list[Path] = [Path(normalized_name).expanduser()]
    hf_snapshot_dir = f"models--{normalized_name.replace('/', '--')}"
    legacy_dir = normalized_name.replace("/", "_")

    cache_roots: list[Path] = []

    hf_cache_dir = _hf_hub_cache_dir()
    if hf_cache_dir is not None:
        cache_roots.append(hf_cache_dir)

    sentence_transformers_home = str(os.getenv("SENTENCE_TRANSFORMERS_HOME", "") or "").strip()
    if sentence_transformers_home:
        cache_roots.append(Path(sentence_transformers_home).expanduser())

    cache_roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")

    seen: set[Path] = set()
    for cache_root in cache_roots:
        if cache_root in seen:
            continue
        seen.add(cache_root)
        candidates.append(cache_root / hf_snapshot_dir)
        candidates.append(cache_root / legacy_dir)

    return candidates


def _has_local_sentence_transformer_cache(model_name: str) -> bool:
    return any(candidate.exists() for candidate in _candidate_local_model_paths(model_name))


def load_sentence_transformer(model_name: str) -> Any | None:
    """Load a sentence-transformer model once per process."""

    normalized_name = str(model_name or "").strip()
    if not normalized_name or SentenceTransformer is None:
        return None

    with _MODEL_LOCK:
        if normalized_name in _MODEL_CACHE:
            return _MODEL_CACHE[normalized_name]

        load_kwargs: dict[str, Any] = {}
        # When a local cache is already present, avoid slow Hub metadata checks.
        if _has_local_sentence_transformer_cache(normalized_name):
            load_kwargs["local_files_only"] = True

        try:
            model = SentenceTransformer(normalized_name, **load_kwargs)
        except TypeError as exc:
            if not load_kwargs or "local_files_only" not in str(exc):
                raise
            model = SentenceTransformer(normalized_name)
        _MODEL_CACHE[normalized_name] = model
        return model


def encode_text(
    text: str,
    *,
    model_name: str,
    normalize_embeddings: bool = True,
) -> list[float] | None:
    """Encode a single text string into a dense vector."""

    model = load_sentence_transformer(model_name)
    if model is None:
        return None

    try:
        embedding = model.encode(text.strip(), normalize_embeddings=normalize_embeddings)
    except TypeError:
        embedding = model.encode(text.strip())

    return _coerce_embedding(embedding)


def encode_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    normalize_embeddings: bool = True,
) -> list[list[float] | None]:
    """Encode a batch of texts into dense vectors."""

    normalized_texts = [str(text or "").strip() for text in texts]
    if not normalized_texts:
        return []

    model = load_sentence_transformer(model_name)
    if model is None:
        return [None for _ in normalized_texts]

    try:
        payload = model.encode(normalized_texts, normalize_embeddings=normalize_embeddings)
    except TypeError:
        payload = model.encode(normalized_texts)

    if hasattr(payload, "tolist"):
        payload = payload.tolist()

    if not isinstance(payload, list):
        return [None for _ in normalized_texts]

    return [_coerce_embedding(item) for item in payload]
