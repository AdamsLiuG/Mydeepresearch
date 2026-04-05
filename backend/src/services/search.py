"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import math
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from hello_agents.tools import SearchTool

from config import Configuration
from metrics import RequestTrace
from services.embeddings import (
    embeddings_available,
    encode_text,
    load_sentence_transformer,
)
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

logger = logging.getLogger(__name__)

_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")
_CACHE_LOCK = Lock()
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
_SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SEMANTIC_SCHOLAR_USER_AGENT = "helloagents-deepresearch/1.0"
_SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "title",
        "url",
        "abstract",
        "year",
        "publicationDate",
        "citationCount",
        "authors",
        "venue",
        "publicationTypes",
        "openAccessPdf",
    ]
)

_MEMORY_CACHE: dict[str, SearchCacheEntry] = {}
_MEMORY_SCOPE_INDEX: dict[str, list[str]] = {}
_DISK_CACHE: Any | None = None
_DISK_CACHE_DIR: str | None = None
_DISK_CACHE_WARNING_EMITTED = False
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
    cache_signature: dict[str, Any] | None
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
            cache_signature=deepcopy(self.cache_signature) if self.cache_signature else None,
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
            "cache_signature": deepcopy(self.cache_signature) if self.cache_signature else None,
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
            cache_signature=deepcopy(record.get("cache_signature") or None),
            payload=deepcopy(record.get("payload") or {}),
            notices=list(record.get("notices") or []),
            answer_text=record.get("answer_text"),
            backend_label=str(record.get("backend_label") or record.get("backend") or ""),
            created_at=float(record.get("created_at") or 0.0),
            embedding=_coerce_embedding(record.get("embedding")),
        )


@dataclass
class AdvancedBackendOutcome:
    backend_order: int
    requested_backend: str
    duration_ms: float
    payload: dict[str, Any] | None = None
    notices: list[str] = field(default_factory=list)
    answer_text: str | None = None
    backend_label: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.payload is not None

    @property
    def result_count(self) -> int:
        if not isinstance(self.payload, dict):
            return 0
        return len(self.payload.get("results") or [])


class AdvancedRerankError(RuntimeError):
    """Raised when optional advanced reranking cannot produce a valid ordering."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason


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
    for field_name in ("research_topic", "task_title", "task_intent"):
        value = cache_context.get(field_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized[field_name] = text
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


def _canonical_cache_signature(cache_signature: dict[str, Any] | None) -> str:
    """Return a stable JSON string for cache-isolating search settings."""

    if not cache_signature:
        return ""
    return json.dumps(cache_signature, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _resolved_search_fetch_full_page(search_api: str, config: Configuration) -> bool:
    """Return the effective fetch_full_page flag for the requested search mode."""

    if str(search_api or "").strip().lower() == "advanced":
        return config.resolved_advanced_fetch_full_page()
    return bool(config.fetch_full_page)


def _build_scope_key(
    search_api: str,
    fetch_full_page: bool,
    cache_signature: dict[str, Any] | None = None,
    topic_scope: str = "",
) -> str:
    """Return the namespace key used to group semantically comparable entries."""

    base = f"scope::{search_api}::{int(fetch_full_page)}"
    signature_text = _canonical_cache_signature(cache_signature)
    if signature_text:
        signature_digest = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()
        base = f"{base}::{signature_digest}"

    normalized_topic_scope = _normalize_query(topic_scope)
    if not normalized_topic_scope:
        return base

    digest = hashlib.sha256(normalized_topic_scope.encode("utf-8")).hexdigest()
    return f"{base}::{digest}"


def _build_cache_key(query: str, search_api: str, config: Configuration) -> str:
    """Return the persistent key for an exact query match."""

    fetch_full_page = _resolved_search_fetch_full_page(search_api, config)
    payload = {
        "query": _normalize_query(query),
        "search_api": search_api,
        "fetch_full_page": fetch_full_page,
        "cache_signature": config.resolved_search_cache_signature(search_api),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"search::{search_api}::{int(fetch_full_page)}::{digest}"


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
    scope_key = _build_scope_key(
        entry.search_api,
        entry.fetch_full_page,
        entry.cache_signature,
        entry.topic_scope,
    )
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

    global _EMBEDDING_MODEL_NAME
    global _EMBEDDING_WARNING_EMITTED

    if not config.semantic_cache_enabled:
        return None

    if not embeddings_available():
        _emit_embedding_warning_once(
            "sentence-transformers is not installed; semantic cache will fall back to exact-match persistence"
        )
        return None

    model_name = config.semantic_cache_embedding_model
    try:
        model = load_sentence_transformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on local runtime state
        _EMBEDDING_MODEL_NAME = model_name
        _emit_embedding_warning_once(
            "Failed to load semantic cache embedding model=%s; falling back to exact-match persistence error=%s",
            model_name,
            exc,
        )
        return None

    if model is None:
        return None
    if _EMBEDDING_MODEL_NAME != model_name:
        logger.info("Loaded semantic cache embedding model=%s", model_name)
    _EMBEDDING_MODEL_NAME = model_name
    _EMBEDDING_WARNING_EMITTED = False
    return model


def _embed_query(query: str, config: Configuration) -> list[float] | None:
    """Return the query embedding, or None if semantic cache is unavailable."""

    if _load_embedding_model(config) is None:
        return None

    try:
        return encode_text(
            query.strip(),
            model_name=config.semantic_cache_embedding_model,
            normalize_embeddings=True,
        )
    except Exception as exc:  # pragma: no cover - depends on local runtime state
        _emit_embedding_warning_once(
            "Failed to encode semantic cache query; falling back to exact-match persistence error=%s",
            exc,
        )
        return None


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

    scope_key = _build_scope_key(
        search_api,
        _resolved_search_fetch_full_page(search_api, config),
        config.resolved_search_cache_signature(search_api),
        topic_scope,
    )
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


def _semantic_scholar_timeout_seconds(config: Configuration) -> float:
    """Return the HTTP timeout used for Semantic Scholar requests."""

    configured = float(config.search_tool_timeout_seconds or 10.0)
    return max(1.0, configured)


def _semantic_scholar_headers(config: Configuration) -> dict[str, str]:
    """Build the Semantic Scholar request headers."""

    headers = {
        "Accept": "application/json",
        "User-Agent": _SEMANTIC_SCHOLAR_USER_AGENT,
    }
    api_key = str(config.semantic_scholar_api_key or "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _semantic_scholar_text(value: Any) -> str:
    """Normalize a Semantic Scholar field into a trimmed string."""

    return " ".join(str(value or "").strip().split())


def _semantic_scholar_author_names(value: Any) -> list[str]:
    """Return the normalized author names from a Semantic Scholar paper payload."""

    if not isinstance(value, list):
        return []

    names: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _semantic_scholar_text(item.get("name"))
        if name:
            names.append(name)
    return names


def _semantic_scholar_publication_types(value: Any) -> list[str]:
    """Return publication types as a normalized string list."""

    if not isinstance(value, list):
        return []

    publication_types: list[str] = []
    for item in value:
        normalized = _semantic_scholar_text(item)
        if normalized:
            publication_types.append(normalized)
    return publication_types


def _semantic_scholar_paper_url(result: dict[str, Any]) -> str:
    """Return a stable Semantic Scholar paper URL."""

    url = _semantic_scholar_text(result.get("url"))
    if url:
        return url

    paper_id = _semantic_scholar_text(result.get("paperId") or result.get("paper_id"))
    if paper_id:
        return f"https://www.semanticscholar.org/paper/{paper_id}"
    return ""


def _semantic_scholar_published_at(result: dict[str, Any]) -> str | None:
    """Return a normalized publication date when available."""

    publication_date = _semantic_scholar_text(result.get("publicationDate"))
    if publication_date:
        return publication_date

    year = result.get("year")
    try:
        if year is not None and str(year).strip():
            normalized_year = int(year)
            if 1900 <= normalized_year <= 2100:
                return f"{normalized_year:04d}-01-01"
    except (TypeError, ValueError):
        return None
    return None


def _semantic_scholar_tldr_text(result: dict[str, Any]) -> str:
    """Return a normalized TL;DR string when available."""

    value = result.get("tldr")
    if isinstance(value, dict):
        return _semantic_scholar_text(value.get("text"))
    return _semantic_scholar_text(value)


def _normalize_semantic_scholar_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Semantic Scholar paper into the repo's search result shape."""

    paper_id = _semantic_scholar_text(result.get("paperId"))
    url = _semantic_scholar_paper_url(result)
    title = _semantic_scholar_text(result.get("title")) or url or paper_id or "Semantic Scholar Paper"
    authors = _semantic_scholar_author_names(result.get("authors"))
    year = result.get("year")
    try:
        normalized_year = int(year) if year is not None and str(year).strip() else None
    except (TypeError, ValueError):
        normalized_year = None

    venue = _semantic_scholar_text(result.get("venue"))
    citation_count = result.get("citationCount")
    try:
        normalized_citation_count = (
            int(citation_count) if citation_count is not None and str(citation_count).strip() else None
        )
    except (TypeError, ValueError):
        normalized_citation_count = None

    publication_types = _semantic_scholar_publication_types(result.get("publicationTypes"))
    open_access_pdf = result.get("openAccessPdf")
    open_access_pdf_url = ""
    if isinstance(open_access_pdf, dict):
        open_access_pdf_url = _semantic_scholar_text(open_access_pdf.get("url"))

    abstract = _semantic_scholar_text(result.get("abstract"))
    tldr_text = _semantic_scholar_tldr_text(result)
    published_at = _semantic_scholar_published_at(result)

    lines = [f"论文标题: {title}"]
    if authors:
        lines.append(f"作者: {', '.join(authors)}")
    if normalized_year is not None:
        lines.append(f"年份: {normalized_year}")
    if published_at:
        lines.append(f"发布时间: {published_at}")
    if venue:
        lines.append(f"期刊/会议: {venue}")
    if normalized_citation_count is not None:
        lines.append(f"引用数: {normalized_citation_count}")
    if publication_types:
        lines.append(f"发表类型: {', '.join(publication_types)}")
    if tldr_text:
        lines.append(f"TL;DR: {tldr_text}")
    if abstract:
        lines.append(f"摘要: {abstract}")
    if open_access_pdf_url:
        lines.append(f"开放 PDF: {open_access_pdf_url}")

    content = "\n".join(lines).strip()
    return {
        "title": title,
        "url": url,
        "content": content,
        "raw_content": content,
        "paper_id": paper_id or None,
        "authors": authors,
        "year": normalized_year,
        "citation_count": normalized_citation_count,
        "venue": venue or None,
        "publication_types": publication_types,
        "open_access_pdf_url": open_access_pdf_url or None,
        "published_at": published_at,
        "provider_count": 1,
    }


def _execute_semantic_scholar_backend(
    query: str,
    config: Configuration,
    *,
    max_results: int,
) -> tuple[dict[str, Any], list[str], str | None, str]:
    """Execute the local Semantic Scholar provider without relying on hello-agents."""

    try:
        response = requests.get(
            _SEMANTIC_SCHOLAR_SEARCH_URL,
            params={
                "query": query,
                "fields": _SEMANTIC_SCHOLAR_FIELDS,
                "limit": max(1, int(max_results or 1)),
            },
            headers=_semantic_scholar_headers(config),
            timeout=_semantic_scholar_timeout_seconds(config),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Semantic Scholar 请求失败: {exc}") from exc

    if response.status_code == 401:
        raise RuntimeError("Semantic Scholar 认证失败（401），请检查 SEMANTIC_SCHOLAR_API_KEY 是否有效。")
    if response.status_code == 403:
        raise RuntimeError("Semantic Scholar 拒绝访问（403），请检查 SEMANTIC_SCHOLAR_API_KEY 权限。")
    if response.status_code == 429:
        raise RuntimeError(
            "Semantic Scholar 请求触发限流（429），请稍后重试或配置 SEMANTIC_SCHOLAR_API_KEY。"
        )
    if response.status_code >= 500:
        raise RuntimeError(
            f"Semantic Scholar 服务暂时不可用（{response.status_code}），请稍后重试。"
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Semantic Scholar 请求失败（{response.status_code}）：{_semantic_scholar_text(response.text)}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Semantic Scholar 返回了无法解析的 JSON 响应。") from exc

    results = [
        _normalize_semantic_scholar_result(item)
        for item in list(payload.get("data") or [])
        if isinstance(item, dict)
    ]
    normalized_payload = {
        "results": results,
        "backend": "semanticscholar",
        "answer": None,
        "notices": [],
    }
    return normalized_payload, [], None, "semanticscholar"


def _execute_search_backend(
    query: str,
    backend: str,
    config: Configuration,
    loop_count: int,
    *,
    max_results: int,
    fetch_full_page: bool | None = None,
) -> tuple[dict[str, Any], list[str], str | None, str]:
    """Execute a single backend through HelloAgents SearchTool."""

    if backend == "semanticscholar":
        return _execute_semantic_scholar_backend(
            query,
            config,
            max_results=max_results,
        )

    resolved_fetch_full_page = config.fetch_full_page if fetch_full_page is None else bool(fetch_full_page)
    raw_response = _GLOBAL_SEARCH_TOOL.run(
        {
            "input": query,
            "backend": backend,
            "mode": "structured",
            "fetch_full_page": resolved_fetch_full_page,
            "max_results": max_results,
            "max_tokens_per_source": config.resolved_max_tokens_per_source(),
            "loop_count": loop_count,
        }
    )
    return _normalize_search_payload(raw_response, requested_backend=backend)


def _execute_advanced_backend(
    query: str,
    backend: str,
    config: Configuration,
    loop_count: int,
    *,
    backend_order: int,
    max_results: int,
    fetch_full_page: bool,
) -> AdvancedBackendOutcome:
    """Execute one advanced backend and capture timing and failure details."""

    started_at = time.perf_counter()
    try:
        payload, notices, answer_text, backend_label = _execute_search_backend(
            query,
            backend,
            config,
            loop_count,
            max_results=max_results,
            fetch_full_page=fetch_full_page,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        error_text = str(exc).strip() or exc.__class__.__name__
        return AdvancedBackendOutcome(
            backend_order=backend_order,
            requested_backend=backend,
            backend_label=backend,
            duration_ms=(time.perf_counter() - started_at) * 1000.0,
            error=error_text,
        )

    return AdvancedBackendOutcome(
        backend_order=backend_order,
        requested_backend=backend,
        backend_label=backend_label,
        duration_ms=(time.perf_counter() - started_at) * 1000.0,
        payload=payload,
        notices=notices,
        answer_text=answer_text,
    )


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

    for field_name in ("title", "content", "raw_content"):
        current = str(existing.get(field_name) or "")
        replacement = str(candidate.get(field_name) or "")
        if len(replacement) > len(current):
            existing[field_name] = candidate.get(field_name)

    if not existing.get("url") and candidate.get("url"):
        existing["url"] = candidate["url"]


def _sort_fused_results_by_rules(merged_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the baseline rule-based ordering for fused advanced results."""

    return sorted(
        merged_results.values(),
        key=lambda item: (
            -int(item.get("provider_count", 1)),
            int(item.get("_backend_order", 10_000)),
            int(item.get("_best_rank", 10_000)),
            -len(str(item.get("content") or "")),
        ),
    )


def _build_advanced_ranking_metadata(
    *,
    strategy: str,
    rerank_applied: bool,
    candidate_count: int,
    model: str | None,
    fallback_reason: str | None,
) -> dict[str, Any]:
    """Build ranking metadata returned with advanced fused payloads."""

    return {
        "strategy": strategy,
        "rerank_applied": rerank_applied,
        "candidate_count": candidate_count,
        "model": model,
        "fallback_reason": fallback_reason,
    }


def _coerce_openai_text(value: Any) -> str:
    """Best-effort extraction of text from OpenAI-compatible response content."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_openai_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            text = value.get(key)
            if isinstance(text, str):
                return text
        return "".join(_coerce_openai_text(item) for item in value.values())
    return ""


def _extract_openai_completion_text(payload: dict[str, Any]) -> str:
    """Return the visible content from an OpenAI-compatible chat completion payload."""

    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ""

    choice = choices[0]
    if not isinstance(choice, dict):
        return ""

    for field_name in ("message", "delta"):
        field_value = choice.get(field_name)
        if field_value is not None:
            text = _coerce_openai_text(field_value)
            if text:
                return text

    return _coerce_openai_text(choice.get("text"))


def _normalize_advanced_rerank_base_url(base_url: str) -> str:
    """Return the normalized advanced rerank base URL."""

    normalized = str(base_url or "").rstrip("/")
    if not normalized:
        raise AdvancedRerankError("rerank_not_configured", "advanced rerank base_url is not configured")
    return normalized


def _advanced_chat_completions_endpoint(base_url: str) -> str:
    """Return the OpenAI-compatible chat completions endpoint."""

    normalized = _normalize_advanced_rerank_base_url(base_url)
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/rerank"):
        return f"{normalized[: -len('/rerank')]}/chat/completions"
    return f"{normalized}/chat/completions"


def _advanced_vllm_rerank_endpoint(base_url: str) -> str:
    """Return the vLLM-compatible rerank endpoint."""

    normalized = _normalize_advanced_rerank_base_url(base_url)
    if normalized.endswith("/rerank"):
        return normalized
    if normalized.endswith("/chat/completions"):
        return f"{normalized[: -len('/chat/completions')]}/rerank"
    return f"{normalized}/rerank"


def _should_bypass_proxy_for_url(url: str) -> bool:
    """Return whether rerank requests should bypass environment proxy settings."""

    hostname = (urlsplit(str(url or "")).hostname or "").strip().lower()
    if not hostname:
        return False
    if hostname == "localhost":
        return True

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return bool(address.is_private or address.is_loopback or address.is_link_local)


def _post_advanced_rerank_request(
    endpoint: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> requests.Response:
    """Post a rerank request, bypassing proxy env for local/private endpoints."""

    if _should_bypass_proxy_for_url(endpoint):
        with requests.Session() as session:
            session.trust_env = False
            return session.post(endpoint, headers=headers, json=payload, timeout=timeout)
    return requests.post(endpoint, headers=headers, json=payload, timeout=timeout)


def _advanced_rerank_candidates(
    ranked_results: list[dict[str, Any]],
    config: Configuration,
) -> list[dict[str, Any]]:
    """Prepare the top fused candidates sent to the optional reranker."""

    content_limit = config.resolved_advanced_rerank_max_content_chars()
    candidate_pool = min(len(ranked_results), config.resolved_advanced_rerank_candidate_pool())
    candidates: list[dict[str, Any]] = []

    for index, item in enumerate(ranked_results[:candidate_pool], start=1):
        content_source = str(item.get("raw_content") or item.get("content") or "")
        candidates.append(
            {
                "id": f"doc-{index}",
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "content": truncate_text(content_source, content_limit),
                "backend_sources": list(item.get("backend_sources") or []),
                "provider_count": int(item.get("provider_count", 1)),
            }
        )

    return candidates


def _parse_rerank_ids(text: str, candidate_ids: list[str]) -> list[str]:
    """Validate reranker output as a strict permutation of candidate ids."""

    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AdvancedRerankError("rerank_response_invalid_json", str(exc)) from exc

    ranked_ids = payload.get("ranked_ids")
    if not isinstance(ranked_ids, list) or not all(isinstance(item, str) for item in ranked_ids):
        raise AdvancedRerankError("rerank_invalid_ranked_ids", "reranker must return ranked_ids as a string list")

    if len(ranked_ids) != len(candidate_ids):
        raise AdvancedRerankError("rerank_invalid_ranked_ids", "reranker must return every candidate exactly once")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise AdvancedRerankError("rerank_invalid_ranked_ids", "reranker returned duplicate candidate ids")
    if set(ranked_ids) != set(candidate_ids):
        raise AdvancedRerankError("rerank_invalid_ranked_ids", "reranker returned unknown or missing candidate ids")

    return ranked_ids


def _build_vllm_rerank_documents(candidates: list[dict[str, Any]]) -> list[str]:
    """Serialize candidates into text documents accepted by the vLLM rerank API."""

    documents: list[str] = []
    for candidate in candidates:
        parts: list[str] = []

        title = str(candidate.get("title") or "").strip()
        if title:
            parts.append(f"Title: {title}")

        url = str(candidate.get("url") or "").strip()
        if url:
            parts.append(f"URL: {url}")

        backend_sources = [str(item).strip() for item in candidate.get("backend_sources") or [] if str(item).strip()]
        if backend_sources:
            parts.append(f"Sources: {', '.join(backend_sources)}")

        content = str(candidate.get("content") or "").strip()
        if content:
            parts.append(f"Content: {content}")

        documents.append("\n".join(parts) or title or url or candidate["id"])

    return documents


def _parse_vllm_rerank_ids(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    """Validate vLLM rerank results and convert returned indices into candidate ids."""

    results = payload.get("results")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise AdvancedRerankError("rerank_invalid_results", "reranker must return results as an object list")

    candidate_count = len(candidates)
    ranked_indices: list[int] = []
    for item in results:
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise AdvancedRerankError("rerank_invalid_results", "reranker must return integer result indices")
        if index < 0 or index >= candidate_count:
            raise AdvancedRerankError("rerank_invalid_results", "reranker returned an out-of-range result index")
        ranked_indices.append(index)

    if len(ranked_indices) != candidate_count:
        raise AdvancedRerankError("rerank_invalid_results", "reranker must return every candidate exactly once")
    if len(set(ranked_indices)) != len(ranked_indices):
        raise AdvancedRerankError("rerank_invalid_results", "reranker returned duplicate result indices")

    return [candidates[index]["id"] for index in ranked_indices]


def _invoke_vllm_reranker(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    timeout: float,
) -> list[str]:
    """Call the vLLM rerank API and return the ordered candidate ids."""

    payload = {
        "model": model,
        "query": query,
        "documents": _build_vllm_rerank_documents(candidates),
        "top_n": len(candidates),
    }

    try:
        response = _post_advanced_rerank_request(
            endpoint,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise AdvancedRerankError("rerank_timeout", str(exc) or "advanced rerank request timed out") from exc
    except requests.RequestException as exc:
        raise AdvancedRerankError("rerank_request_failed", str(exc) or "advanced rerank request failed") from exc

    if response.status_code in {404, 405}:
        raise AdvancedRerankError(
            "rerank_endpoint_unsupported",
            f"advanced rerank endpoint unsupported HTTP {response.status_code}: {str(response.text or '').strip()}",
        )
    if response.status_code >= 400:
        raise AdvancedRerankError(
            "rerank_http_error",
            f"advanced rerank HTTP {response.status_code}: {str(response.text or '').strip()}",
        )

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise AdvancedRerankError("rerank_response_not_json", "advanced rerank returned invalid JSON") from exc

    return _parse_vllm_rerank_ids(response_payload, candidates)


def _invoke_chat_completion_reranker(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    timeout: float,
) -> list[str]:
    """Call a chat-completions reranker and return the ordered candidate ids."""

    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rerank deduplicated web search candidates for relevance to a user query. "
                    "Return JSON only with the exact schema {\"ranked_ids\": [\"doc-1\", \"doc-2\"]}. "
                    "Use every candidate id exactly once. Do not add commentary."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    try:
        response = _post_advanced_rerank_request(
            endpoint,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise AdvancedRerankError("rerank_timeout", str(exc) or "advanced rerank request timed out") from exc
    except requests.RequestException as exc:
        raise AdvancedRerankError("rerank_request_failed", str(exc) or "advanced rerank request failed") from exc

    if response.status_code >= 400:
        raise AdvancedRerankError(
            "rerank_http_error",
            f"advanced rerank HTTP {response.status_code}: {str(response.text or '').strip()}",
        )

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise AdvancedRerankError("rerank_response_not_json", "advanced rerank returned invalid JSON") from exc

    text = _extract_openai_completion_text(response_payload).strip()
    if not text:
        raise AdvancedRerankError("rerank_response_missing_content", "advanced rerank returned no message content")

    return _parse_rerank_ids(text, [candidate["id"] for candidate in candidates])


def _invoke_advanced_reranker(
    query: str,
    candidates: list[dict[str, Any]],
    config: Configuration,
) -> list[str]:
    """Call the optional reranker and return the ordered candidate ids."""

    model = config.resolved_advanced_rerank_model()
    base_url = config.resolved_advanced_rerank_base_url()
    if not model or not base_url:
        raise AdvancedRerankError("rerank_not_configured", "advanced rerank model or base_url is not configured")

    headers = {"Content-Type": "application/json"}
    api_key = config.resolved_advanced_rerank_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    normalized_base_url = _normalize_advanced_rerank_base_url(base_url)
    timeout = config.resolved_advanced_rerank_timeout_seconds()

    if normalized_base_url.endswith("/chat/completions"):
        return _invoke_chat_completion_reranker(
            query,
            candidates,
            endpoint=normalized_base_url,
            model=model,
            headers=headers,
            timeout=timeout,
        )

    try:
        return _invoke_vllm_reranker(
            query,
            candidates,
            endpoint=_advanced_vllm_rerank_endpoint(normalized_base_url),
            model=model,
            headers=headers,
            timeout=timeout,
        )
    except AdvancedRerankError as exc:
        if exc.reason != "rerank_endpoint_unsupported":
            raise

    return _invoke_chat_completion_reranker(
        query,
        candidates,
        endpoint=_advanced_chat_completions_endpoint(normalized_base_url),
        model=model,
        headers=headers,
        timeout=timeout,
    )


def _rerank_advanced_results(
    query: str,
    ranked_results: list[dict[str, Any]],
    config: Configuration,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Optionally rerank the top fused advanced candidates and fall back safely."""

    ranking = _build_advanced_ranking_metadata(
        strategy="rules",
        rerank_applied=False,
        candidate_count=len(ranked_results),
        model=config.resolved_advanced_rerank_model() if config.advanced_rerank_enabled else None,
        fallback_reason=None,
    )

    if not config.advanced_rerank_enabled or len(ranked_results) <= 1:
        return ranked_results, ranking, []

    candidates = _advanced_rerank_candidates(ranked_results, config)
    if len(candidates) <= 1:
        return ranked_results, ranking, []

    started_at = time.perf_counter()
    try:
        ranked_ids = _invoke_advanced_reranker(query, candidates, config)
    except AdvancedRerankError as exc:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        ranking["fallback_reason"] = exc.reason
        logger.warning(
            "Advanced rerank fallback reason=%s model=%s candidates=%s duration_ms=%.2f",
            exc.reason,
            ranking["model"] or "<unset>",
            len(candidates),
            duration_ms,
        )
        return ranked_results, ranking, [f"advanced rerank 回退: {exc.reason}"]

    reranked_pool = [ranked_results[int(doc_id.split("-", 1)[1]) - 1] for doc_id in ranked_ids]
    remaining_results = ranked_results[len(candidates) :]
    ranking["strategy"] = "rules+llm_rerank"
    ranking["rerank_applied"] = True
    logger.info(
        "Advanced rerank applied model=%s candidates=%s duration_ms=%.2f",
        ranking["model"] or "<unset>",
        len(candidates),
        (time.perf_counter() - started_at) * 1000.0,
    )
    return reranked_pool + remaining_results, ranking, []


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
    effective_fetch_full_page = config.resolved_advanced_fetch_full_page()
    deadline_seconds = config.resolved_advanced_backend_timeout_seconds()
    outcomes: dict[int, AdvancedBackendOutcome] = {}
    timed_out_backends: set[str] = set()

    executor = ThreadPoolExecutor(
        max_workers=config.resolved_advanced_search_max_concurrency(),
        thread_name_prefix="advanced-search",
    )
    future_map = {
        executor.submit(
            _execute_advanced_backend,
            query,
            backend,
            config,
            loop_count,
            backend_order=backend_order,
            max_results=max_results,
            fetch_full_page=effective_fetch_full_page,
        ): backend
        for backend_order, backend in enumerate(backends)
    }
    pending = set(future_map.keys())
    deadline_at = time.perf_counter() + deadline_seconds

    try:
        while pending:
            remaining = deadline_at - time.perf_counter()
            if remaining <= 0:
                break

            done, pending = wait(
                pending,
                timeout=remaining,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                break

            for future in done:
                try:
                    outcome = future.result()
                except Exception as exc:  # pragma: no cover - defensive guardrail
                    backend = future_map[future]
                    outcome = AdvancedBackendOutcome(
                        backend_order=backends.index(backend),
                        requested_backend=backend,
                        backend_label=backend,
                        duration_ms=(time.perf_counter() - (deadline_at - deadline_seconds)) * 1000.0,
                        error=str(exc).strip() or exc.__class__.__name__,
                    )

                outcomes[outcome.backend_order] = outcome
                log_payload = {
                    "requested_backend": outcome.requested_backend,
                    "resolved_backend": outcome.backend_label or outcome.requested_backend,
                    "success": outcome.success,
                    "duration_ms": round(outcome.duration_ms, 2),
                    "result_count": outcome.result_count,
                    "notice_count": len(outcome.notices),
                }
                if outcome.success:
                    logger.info("Advanced search backend completed %s", log_payload)
                else:
                    logger.warning("Advanced search backend failed %s error=%s", log_payload, outcome.error)
    finally:
        for future in pending:
            backend = future_map[future]
            timed_out_backends.add(backend)
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    for backend_order, backend in enumerate(backends):
        outcome = outcomes.get(backend_order)
        if outcome is None:
            if backend in timed_out_backends:
                notices.append(
                    f"{backend} 搜索超时: advanced backend deadline exceeded ({deadline_seconds:.2f}s)"
                )
                continue
            notices.append(f"{backend} 搜索失败: backend worker did not return")
            continue

        if not outcome.success or not isinstance(outcome.payload, dict):
            notices.append(f"{backend} 搜索失败: {outcome.error}")
            continue

        backend_label = outcome.backend_label or backend
        successful_backends.append(backend_label)
        if outcome.answer_text and not answer_text:
            answer_text = outcome.answer_text

        notices.extend(
            f"{backend_label}: {notice}"
            for notice in outcome.notices
            if str(notice or "").strip()
        )

        results = outcome.payload.get("results") or []
        for result_rank, item in enumerate(results):
            if isinstance(item, dict):
                _merge_fused_result(
                    merged_results,
                    item,
                    backend_label=backend_label,
                    backend_order=backend_order,
                    result_rank=result_rank,
                )

    ranked_results = _sort_fused_results_by_rules(merged_results)
    ranked_results, ranking, rerank_notices = _rerank_advanced_results(query, ranked_results, config)
    notices.extend(rerank_notices)

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
        "ranking": ranking,
    }
    return payload, notices, answer_text, backend_label


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
    observer: RequestTrace | None = None,
    cache_context: dict[str, Any] | None = None,
    max_results: int = 5,
) -> tuple[dict[str, Any] | None, list[str], str | None, str, bool, str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)
    effective_fetch_full_page = _resolved_search_fetch_full_page(search_api, config)
    cache_signature = config.resolved_search_cache_signature(search_api)
    cache_key = _build_cache_key(query, search_api, config)
    normalized_cache_context = _normalize_cache_context(cache_context)
    topic_scope = _build_topic_scope(normalized_cache_context)
    semantic_text = _build_semantic_text(query, normalized_cache_context)
    semantic_embedding: list[float] | None = None
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
            fetch_full_page=effective_fetch_full_page,
            cache_signature=cache_signature or None,
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
