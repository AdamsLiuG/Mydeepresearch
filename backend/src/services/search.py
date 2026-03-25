"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from hello_agents.tools import SearchTool

from config import Configuration
from metrics import RequestTrace
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
    truncate_text,
)

try:  # pragma: no cover - exercised through runtime fallback
    from diskcache import Cache as DiskCache
except ImportError:  # pragma: no cover - exercised through runtime fallback
    DiskCache = None

try:  # pragma: no cover - exercised through runtime fallback
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - exercised through runtime fallback
    SentenceTransformer = None

logger = logging.getLogger(__name__)

_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")
_CACHE_LOCK = Lock()
_MODEL_LOCK = Lock()
_TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}
_CJK_CHAR_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

_MEMORY_CACHE: dict[str, SearchCacheEntry] = {}
_MEMORY_SCOPE_INDEX: dict[str, list[str]] = {}
_DISK_CACHE: Any | None = None
_DISK_CACHE_DIR: str | None = None
_DISK_CACHE_WARNING_EMITTED = False
_EMBEDDING_MODEL: Any | None = None
_EMBEDDING_MODEL_NAME: str | None = None
_EMBEDDING_WARNING_EMITTED = False


@dataclass
class SearchCacheEntry:
    query: str
    normalized_query: str
    semantic_text: str
    topic_scope: str
    search_api: str
    fetch_full_page: bool
    payload: dict[str, Any]
    notices: list[str]
    answer_text: str | None
    backend_label: str
    created_at: float
    embedding: list[float] | None = None

    def clone(self) -> SearchCacheEntry:
        """Return a defensive copy of the cache entry."""

        return SearchCacheEntry(
            query=self.query,
            normalized_query=self.normalized_query,
            semantic_text=self.semantic_text,
            topic_scope=self.topic_scope,
            search_api=self.search_api,
            fetch_full_page=self.fetch_full_page,
            payload=deepcopy(self.payload),
            notices=list(self.notices),
            answer_text=self.answer_text,
            backend_label=self.backend_label,
            created_at=self.created_at,
            embedding=list(self.embedding) if self.embedding else None,
        )

    def to_record(self) -> dict[str, Any]:
        """Serialize the cache entry for persistence."""

        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "semantic_text": self.semantic_text,
            "topic_scope": self.topic_scope,
            "search_api": self.search_api,
            "fetch_full_page": self.fetch_full_page,
            "payload": deepcopy(self.payload),
            "notices": list(self.notices),
            "answer_text": self.answer_text,
            "backend_label": self.backend_label,
            "created_at": self.created_at,
            "embedding": list(self.embedding) if self.embedding else None,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SearchCacheEntry:
        """Deserialize a cache entry from disk or memory."""

        return cls(
            query=str(record.get("query") or ""),
            normalized_query=str(record.get("normalized_query") or ""),
            semantic_text=str(record.get("semantic_text") or record.get("normalized_query") or ""),
            topic_scope=str(record.get("topic_scope") or ""),
            search_api=str(record.get("search_api") or ""),
            fetch_full_page=bool(record.get("fetch_full_page", False)),
            payload=deepcopy(record.get("payload") or {}),
            notices=list(record.get("notices") or []),
            answer_text=record.get("answer_text"),
            backend_label=str(record.get("backend_label") or record.get("backend") or ""),
            created_at=float(record.get("created_at") or 0.0),
            embedding=_coerce_embedding(record.get("embedding")),
        )


def clear_search_cache() -> None:
    """Clear the configured search cache store."""

    global _DISK_CACHE
    global _DISK_CACHE_DIR

    with _CACHE_LOCK:
        _MEMORY_CACHE.clear()
        _MEMORY_SCOPE_INDEX.clear()
        if _DISK_CACHE is not None:
            _DISK_CACHE.clear()
            _DISK_CACHE.close()
            _DISK_CACHE = None
            _DISK_CACHE_DIR = None


def _normalize_query(query: str) -> str:
    """Normalize a query for more stable exact cache keys."""

    return " ".join((query or "").strip().lower().split())


def _normalize_cache_context(cache_context: dict[str, Any] | None) -> dict[str, str]:
    """Return a sanitized cache-context mapping."""

    if not isinstance(cache_context, dict):
        return {}

    normalized: dict[str, str] = {}
    for field in ("research_topic", "task_title", "task_intent"):
        value = cache_context.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized[field] = text
    return normalized


def _build_topic_scope(cache_context: dict[str, str] | None) -> str:
    """Return the normalized topic scope used to isolate semantic cache reuse."""

    if not cache_context:
        return ""
    return _normalize_query(cache_context.get("research_topic") or "")


def _build_semantic_text(query: str, cache_context: dict[str, str] | None) -> str:
    """Return the normalized text used for semantic and lexical cache matching."""

    parts = [
        cache_context.get("research_topic", "") if cache_context else "",
        cache_context.get("task_title", "") if cache_context else "",
        cache_context.get("task_intent", "") if cache_context else "",
        query,
    ]
    combined = " ".join(part.strip() for part in parts if part and str(part).strip())
    return _normalize_query(combined)


def _contains_cjk(text: str) -> bool:
    """Return whether the text contains any CJK characters."""

    return bool(_CJK_CHAR_PATTERN.search(text or ""))


def _lexical_tokens(text: str) -> set[str]:
    """Tokenize text into robust word and character n-grams for lexical similarity."""

    normalized = _normalize_query(text)
    if not normalized:
        return set()

    compact = normalized.replace(" ", "")
    tokens = {part for part in normalized.split(" ") if part}
    if not compact:
        return tokens

    if _contains_cjk(compact):
        tokens.update(char for char in compact if _contains_cjk(char))
        sizes = (2, 3)
    else:
        sizes = (3,)

    for size in sizes:
        if len(compact) < size:
            continue
        for index in range(len(compact) - size + 1):
            tokens.add(compact[index : index + size])

    if len(compact) <= 3:
        tokens.add(compact)

    return tokens


def _lexical_similarity(left: str, right: str) -> float:
    """Compute Jaccard similarity over normalized lexical features."""

    left_tokens = _lexical_tokens(left)
    right_tokens = _lexical_tokens(right)
    if not left_tokens or not right_tokens:
        return -1.0

    union = left_tokens | right_tokens
    if not union:
        return -1.0

    return len(left_tokens & right_tokens) / len(union)


def _coerce_embedding(value: Any) -> list[float] | None:
    """Normalize an embedding payload into a list of floats."""

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


def _build_scope_key(search_api: str, fetch_full_page: bool, topic_scope: str = "") -> str:
    """Return the namespace key used to group semantically comparable entries."""

    base = f"scope::{search_api}::{int(fetch_full_page)}"
    normalized_topic_scope = _normalize_query(topic_scope)
    if not normalized_topic_scope:
        return base

    digest = hashlib.sha256(normalized_topic_scope.encode("utf-8")).hexdigest()
    return f"{base}::{digest}"


def _build_cache_key(query: str, search_api: str, config: Configuration) -> str:
    """Return the persistent key for an exact query match."""

    payload = {
        "query": _normalize_query(query),
        "search_api": search_api,
        "fetch_full_page": config.fetch_full_page,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"search::{search_api}::{int(config.fetch_full_page)}::{digest}"


def _emit_diskcache_warning_once() -> None:
    global _DISK_CACHE_WARNING_EMITTED

    if _DISK_CACHE_WARNING_EMITTED:
        return

    _DISK_CACHE_WARNING_EMITTED = True
    logger.warning(
        "diskcache is not installed; falling back to in-memory search cache without persistence"
    )


def _emit_embedding_warning_once(message: str, *args: Any) -> None:
    global _EMBEDDING_WARNING_EMITTED

    if _EMBEDDING_WARNING_EMITTED:
        return

    _EMBEDDING_WARNING_EMITTED = True
    logger.warning(message, *args)


def _get_disk_cache(config: Configuration) -> Any | None:
    """Return the persistent disk cache, or None if unavailable."""

    global _DISK_CACHE
    global _DISK_CACHE_DIR

    if DiskCache is None:
        _emit_diskcache_warning_once()
        return None

    cache_dir = config.resolved_search_cache_dir()
    with _CACHE_LOCK:
        if _DISK_CACHE is None or _DISK_CACHE_DIR != cache_dir:
            if _DISK_CACHE is not None:
                _DISK_CACHE.close()
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            _DISK_CACHE = DiskCache(cache_dir)
            _DISK_CACHE_DIR = cache_dir
        return _DISK_CACHE


def _get_scope_keys(config: Configuration, scope_key: str) -> list[str]:
    """Load cached entry keys for a given semantic namespace."""

    disk_cache = _get_disk_cache(config)
    if disk_cache is not None:
        return list(disk_cache.get(scope_key, default=[]))

    with _CACHE_LOCK:
        return list(_MEMORY_SCOPE_INDEX.get(scope_key, []))


def _set_scope_keys(config: Configuration, scope_key: str, keys: list[str]) -> None:
    """Persist the semantic namespace index."""

    disk_cache = _get_disk_cache(config)
    if disk_cache is not None:
        disk_cache.set(scope_key, list(keys))
        return

    with _CACHE_LOCK:
        _MEMORY_SCOPE_INDEX[scope_key] = list(keys)


def _read_exact_cache(key: str, ttl_seconds: int, config: Configuration) -> SearchCacheEntry | None:
    """Read an exact cache entry from disk or memory."""

    disk_cache = _get_disk_cache(config)
    if disk_cache is not None:
        record = disk_cache.get(key)
        if record is None:
            return None
        return SearchCacheEntry.from_record(record)

    with _CACHE_LOCK:
        entry = _MEMORY_CACHE.get(key)
        if not entry:
            return None

        if (time.time() - entry.created_at) > ttl_seconds:
            _MEMORY_CACHE.pop(key, None)
            for scope_key, keys in list(_MEMORY_SCOPE_INDEX.items()):
                _MEMORY_SCOPE_INDEX[scope_key] = [candidate for candidate in keys if candidate != key]
            return None

        return entry.clone()


def _write_cache(key: str, entry: SearchCacheEntry, config: Configuration) -> None:
    """Persist a search cache entry to disk or memory."""

    disk_cache = _get_disk_cache(config)
    scope_key = _build_scope_key(entry.search_api, entry.fetch_full_page, entry.topic_scope)
    if disk_cache is not None:
        disk_cache.set(key, entry.to_record(), expire=config.search_cache_ttl_seconds)
    else:
        with _CACHE_LOCK:
            _MEMORY_CACHE[key] = entry.clone()

    scope_keys = _get_scope_keys(config, scope_key)
    if key not in scope_keys:
        scope_keys.append(key)
        _set_scope_keys(config, scope_key, scope_keys)


def _load_embedding_model(config: Configuration) -> Any | None:
    """Lazily load the configured embedding model."""

    global _EMBEDDING_MODEL
    global _EMBEDDING_MODEL_NAME
    global _EMBEDDING_WARNING_EMITTED

    if not config.semantic_cache_enabled:
        return None

    if SentenceTransformer is None:
        _emit_embedding_warning_once(
            "sentence-transformers is not installed; semantic cache will fall back to exact-match persistence"
        )
        return None

    model_name = config.semantic_cache_embedding_model
    with _MODEL_LOCK:
        if _EMBEDDING_MODEL is not None and _EMBEDDING_MODEL_NAME == model_name:
            return _EMBEDDING_MODEL

        try:
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
            _EMBEDDING_MODEL_NAME = model_name
            _EMBEDDING_WARNING_EMITTED = False
        except Exception as exc:  # pragma: no cover - depends on local runtime state
            _EMBEDDING_MODEL = None
            _EMBEDDING_MODEL_NAME = model_name
            _emit_embedding_warning_once(
                "Failed to load semantic cache embedding model=%s; falling back to exact-match persistence error=%s",
                model_name,
                exc,
            )
            return None

        logger.info("Loaded semantic cache embedding model=%s", model_name)
        return _EMBEDDING_MODEL


def _embed_query(query: str, config: Configuration) -> list[float] | None:
    """Return the query embedding, or None if semantic cache is unavailable."""

    model = _load_embedding_model(config)
    if model is None:
        return None

    try:
        embedding = model.encode(query.strip(), normalize_embeddings=True)
    except TypeError:
        embedding = model.encode(query.strip())
    except Exception as exc:  # pragma: no cover - depends on local runtime state
        _emit_embedding_warning_once(
            "Failed to encode semantic cache query; falling back to exact-match persistence error=%s",
            exc,
        )
        return None

    return _coerce_embedding(embedding)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity without introducing extra numerical dependencies."""

    if not left or not right or len(left) != len(right):
        return -1.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0

    return dot / (left_norm * right_norm)


def _read_semantic_cache(
    query_embedding: list[float] | None,
    semantic_text: str,
    search_api: str,
    config: Configuration,
    *,
    topic_scope: str = "",
) -> tuple[SearchCacheEntry | None, float, float]:
    """Read the best semantic cache match for a query."""

    if query_embedding is None and not semantic_text:
        return None, 0.0, 0.0

    scope_key = _build_scope_key(search_api, config.fetch_full_page, topic_scope)
    candidate_keys = _get_scope_keys(config, scope_key)
    if not candidate_keys:
        return None, 0.0, 0.0

    best_entry: SearchCacheEntry | None = None
    best_similarity = -1.0
    best_lexical_similarity = -1.0
    best_score = 0.0
    live_keys: list[str] = []
    semantic_threshold = max(config.semantic_cache_similarity_threshold, 1e-9)
    lexical_threshold = max(config.semantic_cache_lexical_threshold, 1e-9)

    for candidate_key in candidate_keys:
        entry = _read_exact_cache(candidate_key, config.search_cache_ttl_seconds, config)
        if entry is None:
            continue

        live_keys.append(candidate_key)
        similarity = (
            _cosine_similarity(query_embedding, entry.embedding)
            if query_embedding is not None and entry.embedding is not None
            else -1.0
        )
        lexical_similarity = _lexical_similarity(
            semantic_text,
            entry.semantic_text or entry.normalized_query,
        )

        semantic_score = (similarity / semantic_threshold) if similarity >= 0.0 else 0.0
        lexical_score = (lexical_similarity / lexical_threshold) if lexical_similarity >= 0.0 else 0.0
        combined_score = max(semantic_score, lexical_score)

        if combined_score > best_score or (
            math.isclose(combined_score, best_score)
            and (similarity, lexical_similarity) > (best_similarity, best_lexical_similarity)
        ):
            best_score = combined_score
            best_similarity = similarity
            best_lexical_similarity = lexical_similarity
            best_entry = entry

    if live_keys != candidate_keys:
        _set_scope_keys(config, scope_key, live_keys)

    if best_entry and best_score >= 1.0:
        return best_entry, max(best_similarity, 0.0), max(best_lexical_similarity, 0.0)

    return None, max(best_similarity, 0.0), max(best_lexical_similarity, 0.0)


def _normalize_search_payload(
    raw_response: Any,
    *,
    requested_backend: str,
) -> tuple[dict[str, Any], list[str], str | None, str]:
    """Normalize a backend response into a consistent payload shape."""

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", requested_backend, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": requested_backend,
            "answer": None,
            "notices": notices,
        }
    elif isinstance(raw_response, dict):
        payload = deepcopy(raw_response)
        notices = [str(item) for item in (payload.get("notices") or []) if item]
        payload["results"] = list(payload.get("results") or [])
        payload["notices"] = notices
        payload["backend"] = str(payload.get("backend") or requested_backend)
    else:
        notices = [f"Unexpected search response type: {type(raw_response).__name__}"]
        payload = {
            "results": [],
            "backend": requested_backend,
            "answer": None,
            "notices": notices,
        }

    backend_label = str(payload.get("backend") or requested_backend)
    answer_text = payload.get("answer")
    return payload, notices, answer_text, backend_label


def _execute_search_backend(
    query: str,
    backend: str,
    config: Configuration,
    loop_count: int,
    *,
    max_results: int,
) -> tuple[dict[str, Any], list[str], str | None, str]:
    """Execute a single backend through HelloAgents SearchTool."""

    raw_response = _GLOBAL_SEARCH_TOOL.run(
        {
            "input": query,
            "backend": backend,
            "mode": "structured",
            "fetch_full_page": config.fetch_full_page,
            "max_results": max_results,
            "max_tokens_per_source": config.resolved_max_tokens_per_source(),
            "loop_count": loop_count,
        }
    )
    return _normalize_search_payload(raw_response, requested_backend=backend)


def _normalize_result_url(url: str) -> str:
    """Normalize a result URL so near-identical sources can be fused."""

    raw_url = (url or "").strip()
    if not raw_url:
        return ""

    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return raw_url

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_PARAMS and not key.lower().startswith("utm_")
    ]
    query = urlencode(filtered_query, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _result_identity(result: dict[str, Any]) -> str | None:
    """Return the stable identity key used to merge equivalent results."""

    normalized_url = _normalize_result_url(str(result.get("url") or ""))
    if normalized_url:
        return normalized_url

    title = " ".join(str(result.get("title") or "").strip().lower().split())
    if title:
        return f"title::{title}"

    content = " ".join(str(result.get("content") or "").strip().lower().split())
    if content:
        return f"content::{content[:200]}"

    return None


def _merge_fused_result(
    merged_results: dict[str, dict[str, Any]],
    result: dict[str, Any],
    *,
    backend_label: str,
    backend_order: int,
    result_rank: int,
) -> None:
    """Merge a single backend result into the fused result set."""

    key = _result_identity(result) or f"anonymous::{backend_label}::{backend_order}::{result_rank}"
    candidate = deepcopy(result)
    existing = merged_results.get(key)

    if existing is None:
        candidate["backend_sources"] = [backend_label]
        candidate["provider_count"] = 1
        candidate["_backend_order"] = backend_order
        candidate["_best_rank"] = result_rank
        merged_results[key] = candidate
        return

    backend_sources = list(existing.get("backend_sources") or [])
    if backend_label not in backend_sources:
        backend_sources.append(backend_label)
        existing["backend_sources"] = backend_sources
        existing["provider_count"] = len(backend_sources)

    existing["_backend_order"] = min(int(existing.get("_backend_order", backend_order)), backend_order)
    existing["_best_rank"] = min(int(existing.get("_best_rank", result_rank)), result_rank)

    for field in ("title", "content", "raw_content"):
        current = str(existing.get(field) or "")
        replacement = str(candidate.get(field) or "")
        if len(replacement) > len(current):
            existing[field] = candidate.get(field)

    if not existing.get("url") and candidate.get("url"):
        existing["url"] = candidate["url"]


def _fuse_advanced_search_results(
    query: str,
    config: Configuration,
    loop_count: int,
    *,
    max_results: int,
) -> tuple[dict[str, Any], list[str], str | None, str]:
    """Execute multiple backends and fuse results into one ranked payload."""

    merged_results: dict[str, dict[str, Any]] = {}
    notices: list[str] = []
    successful_backends: list[str] = []
    answer_text: str | None = None
    backends = config.resolved_advanced_search_backends()

    with ThreadPoolExecutor(
        max_workers=max(1, len(backends)),
        thread_name_prefix="advanced-search",
    ) as executor:
        future_plan = [
            (
                backend_order,
                backend,
                executor.submit(
                    _execute_search_backend,
                    query,
                    backend,
                    config,
                    loop_count,
                    max_results=max_results,
                ),
            )
            for backend_order, backend in enumerate(backends)
        ]

        for backend_order, backend, future in future_plan:
            try:
                payload, backend_notices, backend_answer, backend_label = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Advanced search backend %s failed: %s", backend, exc)
                notices.append(f"{backend} 搜索失败: {exc}")
                continue

            successful_backends.append(backend_label)
            if backend_answer and not answer_text:
                answer_text = backend_answer

            notices.extend(
                f"{backend_label}: {notice}"
                for notice in backend_notices
                if str(notice or "").strip()
            )

            results = payload.get("results") or []
            for result_rank, item in enumerate(results):
                if isinstance(item, dict):
                    _merge_fused_result(
                        merged_results,
                        item,
                        backend_label=backend_label,
                        backend_order=backend_order,
                        result_rank=result_rank,
                    )

    ranked_results = sorted(
        merged_results.values(),
        key=lambda item: (
            -int(item.get("provider_count", 1)),
            int(item.get("_backend_order", 10_000)),
            int(item.get("_best_rank", 10_000)),
            -len(str(item.get("content") or "")),
        ),
    )

    fused_results: list[dict[str, Any]] = []
    for item in ranked_results[:max_results]:
        normalized_item = deepcopy(item)
        normalized_item.pop("_backend_order", None)
        normalized_item.pop("_best_rank", None)
        fused_results.append(normalized_item)

    backend_label = (
        f"advanced[{', '.join(successful_backends)}]"
        if successful_backends
        else "advanced"
    )
    payload = {
        "results": fused_results,
        "backend": backend_label,
        "answer": answer_text,
        "notices": notices,
    }
    return payload, notices, answer_text, backend_label


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
    observer: RequestTrace | None = None,
    cache_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str], str | None, str, bool, str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)
    cache_key = _build_cache_key(query, search_api, config)
    normalized_cache_context = _normalize_cache_context(cache_context)
    topic_scope = _build_topic_scope(normalized_cache_context)
    semantic_text = _build_semantic_text(query, normalized_cache_context)
    semantic_embedding: list[float] | None = None
    max_results = 5

    if config.search_cache_enabled:
        exact_cached = _read_exact_cache(cache_key, config.search_cache_ttl_seconds, config)
        if exact_cached:
            if observer:
                observer.record_search_attempt(cache_hit=True, success=True, cache_strategy="exact")
            logger.info("Search exact cache hit: backend=%s query=%s", search_api, query)
            return (
                exact_cached.payload,
                exact_cached.notices,
                exact_cached.answer_text,
                exact_cached.backend_label,
                True,
                "exact",
            )

        if config.semantic_cache_enabled:
            semantic_embedding = _embed_query(semantic_text or query, config)
            semantic_cached, similarity, lexical_similarity = _read_semantic_cache(
                semantic_embedding,
                semantic_text,
                search_api,
                config,
                topic_scope=topic_scope,
            )
            if semantic_cached:
                if observer:
                    observer.record_search_attempt(cache_hit=True, success=True, cache_strategy="semantic")
                logger.info(
                    "Search semantic cache hit: backend=%s query=%s matched_query=%s similarity=%.4f lexical_similarity=%.4f topic_scope=%s",
                    search_api,
                    query,
                    semantic_cached.query,
                    similarity,
                    lexical_similarity,
                    topic_scope or "<global>",
                )
                return (
                    semantic_cached.payload,
                    semantic_cached.notices,
                    semantic_cached.answer_text,
                    semantic_cached.backend_label,
                    True,
                    "semantic",
                )

    try:
        if search_api == "advanced":
            payload, notices, answer_text, backend_label = _fuse_advanced_search_results(
                query,
                config,
                loop_count,
                max_results=max_results,
            )
        else:
            payload, notices, answer_text, backend_label = _execute_search_backend(
                query,
                search_api,
                config,
                loop_count,
                max_results=max_results,
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        if observer:
            observer.record_search_attempt(cache_hit=False, success=False, error=exc)
        logger.exception("Search backend %s failed: %s", search_api, exc)
        raise
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    if observer:
        observer.record_search_attempt(cache_hit=False, success=True, cache_strategy="miss")

    if config.search_cache_enabled:
        cache_entry = SearchCacheEntry(
            query=query.strip(),
            normalized_query=_normalize_query(query),
            semantic_text=semantic_text,
            topic_scope=topic_scope,
            search_api=search_api,
            fetch_full_page=config.fetch_full_page,
            payload=payload,
            notices=notices,
            answer_text=answer_text,
            backend_label=backend_label,
            created_at=time.time(),
            embedding=semantic_embedding,
        )
        _write_cache(cache_key, cache_entry, config)

    return payload, notices, answer_text, backend_label, False, "miss"


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: str | None,
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=config.resolved_max_tokens_per_source(),
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        answer_excerpt = truncate_text(
            answer_text,
            config.resolved_direct_answer_char_limit(),
        )
        context = f"AI直接答案：\n{answer_excerpt}\n\n{context}"

    context = truncate_text(context, config.resolved_task_context_char_limit())

    return sources_summary, context
