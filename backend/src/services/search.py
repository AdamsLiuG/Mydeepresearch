"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Optional, Tuple

from hello_agents.tools import SearchTool

from config import Configuration
from metrics import RequestTrace
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
    truncate_text,
)

logger = logging.getLogger(__name__)

_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")
_CACHE_LOCK = Lock()


@dataclass
class SearchCacheEntry:
    payload: dict[str, Any]
    notices: list[str]
    answer_text: Optional[str]
    backend_label: str
    created_at: float


_SEARCH_CACHE: dict[str, SearchCacheEntry] = {}


def clear_search_cache() -> None:
    """Clear the in-process search cache."""

    with _CACHE_LOCK:
        _SEARCH_CACHE.clear()


def _build_cache_key(query: str, search_api: str, config: Configuration) -> str:
    payload = {
        "query": query.strip(),
        "search_api": search_api,
        "fetch_full_page": config.fetch_full_page,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_cache(key: str, ttl_seconds: int) -> SearchCacheEntry | None:
    with _CACHE_LOCK:
        entry = _SEARCH_CACHE.get(key)
        if not entry:
            return None

        age_seconds = perf_counter() - entry.created_at
        if age_seconds > ttl_seconds:
            _SEARCH_CACHE.pop(key, None)
            return None

        return SearchCacheEntry(
            payload=deepcopy(entry.payload),
            notices=list(entry.notices),
            answer_text=entry.answer_text,
            backend_label=entry.backend_label,
            created_at=entry.created_at,
        )


def _write_cache(key: str, entry: SearchCacheEntry) -> None:
    with _CACHE_LOCK:
        _SEARCH_CACHE[key] = SearchCacheEntry(
            payload=deepcopy(entry.payload),
            notices=list(entry.notices),
            answer_text=entry.answer_text,
            backend_label=entry.backend_label,
            created_at=entry.created_at,
        )


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
    observer: RequestTrace | None = None,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str, bool]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)
    cache_key = _build_cache_key(query, search_api, config)
    cache_hit = False

    if config.search_cache_enabled:
        cached = _read_cache(cache_key, config.search_cache_ttl_seconds)
        if cached:
            cache_hit = True
            if observer:
                observer.record_search_attempt(cache_hit=True, success=True)
            logger.info("Search cache hit: backend=%s query=%s", search_api, query)
            return (
                cached.payload,
                cached.notices,
                cached.answer_text,
                cached.backend_label,
                True,
            )

    try:
        raw_response = _GLOBAL_SEARCH_TOOL.run(
            {
                "input": query,
                "backend": search_api,
                "mode": "structured",
                "fetch_full_page": config.fetch_full_page,
                "max_results": 5,
                "max_tokens_per_source": config.resolved_max_tokens_per_source(),
                "loop_count": loop_count,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        if observer:
            observer.record_search_attempt(cache_hit=cache_hit, success=False, error=exc)
        logger.exception("Search backend %s failed: %s", search_api, exc)
        raise

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
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
        observer.record_search_attempt(cache_hit=cache_hit, success=True)

    if config.search_cache_enabled:
        _write_cache(
            cache_key,
            SearchCacheEntry(
                payload=payload,
                notices=notices,
                answer_text=answer_text,
                backend_label=backend_label,
                created_at=perf_counter(),
            ),
        )

    return payload, notices, answer_text, backend_label, False


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
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
