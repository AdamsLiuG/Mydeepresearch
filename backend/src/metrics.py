"""Lightweight in-process tracing and metrics for deep research requests."""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
from typing import Any

CHARS_PER_TOKEN = 4
_CJK_TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")

DEFAULT_PRICING_CATALOG: dict[str, dict[str, dict[str, float]]] = {
    "default": {
        "default": {
            "prompt_per_1k_tokens": 0.0,
            "completion_per_1k_tokens": 0.0,
        }
    }
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    cjk_count = len(_CJK_TOKEN_PATTERN.findall(cleaned))
    non_cjk_count = max(0, len(cleaned) - cjk_count)
    if cjk_count == 0:
        return max(1, (non_cjk_count + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)
    return max(1, int(math.ceil(cjk_count * 1.2 + non_cjk_count / CHARS_PER_TOKEN)))


def extract_llm_usage(agent_or_llm: Any) -> dict[str, int] | None:
    """Read and clear thread-local reported token usage from an agent or LLM."""
    if agent_or_llm is None:
        return None
    consumer = getattr(agent_or_llm, "consume_last_usage", None)
    if callable(consumer):
        usage = consumer()
        if isinstance(usage, dict):
            return usage
    llm = getattr(agent_or_llm, "llm", None)
    if llm is not None:
        llm_consumer = getattr(llm, "consume_last_usage", None)
        if callable(llm_consumer):
            usage = llm_consumer()
            if isinstance(usage, dict):
                return usage
    return None


_REPORTED_USAGE_KEYS = (
    "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens",
    "cached_prompt_tokens", "cached_tokens", "cache_read_input_tokens",
    "cache_creation_tokens", "cache_creation_input_tokens", "provider_calls",
    "estimated_provider_calls",
)


def _token_count(value: Any) -> int | None:
    try:
        return max(0, int(value)) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def reported_usage_fields(raw: Any) -> dict[str, int]:
    """Read optional counters without conflating an omitted count with zero."""

    def get_field(name: str) -> Any:
        return raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)

    fields = {
        key: count
        for key in _REPORTED_USAGE_KEYS
        if (count := _token_count(get_field(key))) is not None
    }
    details = get_field("prompt_tokens_details")
    detailed_count = _token_count(
        details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    )
    if detailed_count is not None:
        fields["cached_tokens"] = detailed_count
    return fields


def normalize_reported_usage(raw: Any) -> dict[str, int] | None:
    fields = reported_usage_fields(raw)
    cached = fields.get("cached_tokens", fields.get("cached_prompt_tokens", fields.get("cache_read_input_tokens")))
    creation = fields.get("cache_creation_tokens", fields.get("cache_creation_input_tokens"))
    prompt = fields.get("prompt_tokens")
    if prompt is None:
        prompt = fields.get("input_tokens", 0)
        if "cache_read_input_tokens" in fields or "cache_creation_input_tokens" in fields:
            prompt += (cached or 0) + (creation or 0)
    completion = fields.get("completion_tokens", fields.get("output_tokens", 0))
    total = max(prompt + completion, fields.get("total_tokens", 0))
    if not fields or not (total or fields.get("provider_calls")):
        return None
    normalized = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    if cached is not None:
        normalized["cached_tokens"] = min(prompt, cached)
    if creation is not None:
        normalized["cache_creation_tokens"] = min(prompt - (normalized.get("cached_tokens") or 0), creation)
    for key in ("provider_calls", "estimated_provider_calls"):
        if key in fields:
            normalized[key] = fields[key]
    return normalized


def _cache_hit_ratio(cached_tokens: int, prompt_tokens: int) -> float:
    return round(cached_tokens / prompt_tokens, 4) if prompt_tokens > 0 else 0.0


def _add_kv_cache_stage(
    stages: dict[str, dict[str, Any]],
    stage: str,
    *,
    calls: int,
    prompt_tokens: int,
    cached_prompt_tokens: int,
    cache_creation_tokens: int,
    uncached_prompt_tokens: int,
) -> None:
    entry = stages.setdefault(
        stage,
        {"calls": 0, "prompt_tokens": 0, "cached_prompt_tokens": 0,
         "cache_creation_tokens": 0, "uncached_prompt_tokens": 0},
    )
    entry["calls"] += calls
    entry["prompt_tokens"] += prompt_tokens
    entry["cached_prompt_tokens"] += cached_prompt_tokens
    entry["cache_creation_tokens"] += cache_creation_tokens
    entry["uncached_prompt_tokens"] += uncached_prompt_tokens
    ratio = _cache_hit_ratio(entry["cached_prompt_tokens"], entry["prompt_tokens"])
    entry.update(cache_hit_ratio=ratio, prompt_cache_hit_ratio=ratio)


def _safe_error(error: Any) -> str | None:
    if error is None:
        return None
    return str(error).strip() or None


def _clone_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(json.dumps(value, ensure_ascii=False))


@dataclass
class StageTrace:
    stage: str
    scope: str
    start_time: str
    status: str = "in_progress"
    end_time: str | None = None
    elapsed_ms: float | None = None
    error: str | None = None
    task_id: int | None = None
    task_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "scope": self.scope,
            "task_id": self.task_id,
            "task_title": self.task_title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_ms": self.elapsed_ms,
            "status": self.status,
            "error": self.error,
            "metadata": _clone_dict(self.metadata),
        }


@dataclass
class LatencyStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float | None = None
    max_ms: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total_ms += value
        self.min_ms = value if self.min_ms is None else min(self.min_ms, value)
        self.max_ms = value if self.max_ms is None else max(self.max_ms, value)

    def to_dict(self) -> dict[str, Any]:
        avg_ms = self.total_ms / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms is not None else None,
            "max_ms": round(self.max_ms, 2) if self.max_ms is not None else None,
        }


class MetricsRegistry:
    """Thread-safe process-local metrics aggregation."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._recent_request_limit = 25
        self._recent_requests: deque[dict[str, Any]] = deque(maxlen=self._recent_request_limit)
        self._counters: dict[str, int] = {
            "request_total": 0,
            "request_success_total": 0,
            "request_partial_success_total": 0,
            "request_failed_total": 0,
            "fallback_trigger_total": 0,
            "llm_call_total": 0,
            "llm_success_total": 0,
            "llm_failed_total": 0,
            "search_call_total": 0,
            "search_success_total": 0,
            "search_failed_total": 0,
            "cache_hit_total": 0,
            "cache_exact_hit_total": 0,
            "cache_semantic_hit_total": 0,
            "cache_approximate_hit_total": 0,
            "cache_approximate_dense_hit_total": 0,
            "cache_approximate_sparse_hit_total": 0,
            "cache_approximate_hybrid_hit_total": 0,
            "cache_miss_total": 0,
            "reflection_call_total": 0,
            "reflection_replan_total": 0,
            "reflection_skipped_total": 0,
            "review_call_total": 0,
            "review_issue_total": 0,
            "task_react_round_total": 0,
            "task_react_continue_total": 0,
            "task_react_stop_total": 0,
            "report_repair_trigger_total": 0,
            "report_repair_added_task_total": 0,
            "note_memory_query_total": 0,
            "note_memory_hit_total": 0,
            "note_memory_miss_total": 0,
            "note_memory_prompt_injection_total": 0,
            "note_memory_refresh_total": 0,
            "note_memory_refresh_failed_total": 0,
            "strategy_memory_query_total": 0,
            "strategy_memory_hit_total": 0,
            "strategy_memory_miss_total": 0,
            "strategy_memory_prompt_injection_total": 0,
            "strategy_memory_refresh_total": 0,
            "strategy_memory_refresh_failed_total": 0,
            "strategy_memory_synthesized_card_total": 0,
            "strategy_memory_archived_card_total": 0,
            "skill_pack_selection_total": 0,
            "skill_pack_fallback_total": 0,
            "memory_shadow_suppression_total": 0,
            "evidence_archive_ingest_total": 0,
            "evidence_archive_query_total": 0,
            "evidence_archive_dedup_total": 0,
            "evidence_archive_restore_total": 0,
            "evidence_archive_restore_miss_total": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "cache_creation_tokens": 0,
            "uncached_prompt_tokens": 0,
            "mcp_call_total": 0,
            "mcp_success_total": 0,
            "mcp_timeout_total": 0,
            "mcp_circuit_open_total": 0,
            "mcp_hitl_intercept_total": 0,
            "multimodal_pages_triaged_total": 0,
            "multimodal_vlm_calls_total": 0,
            "multimodal_vlm_fallback_total": 0,
            "sandbox_executions_total": 0,
            "sandbox_timeouts_total": 0,
            "sandbox_ast_rejections_total": 0,
            "nli_claims_verified_total": 0,
            "nli_contradictions_total": 0,
            "nli_degraded_total": 0,
            "evidence_conflicts_total": 0,
            "policy_shadow_decisions_total": 0,
            "policy_shadow_agreement_total": 0,
            "policy_canary_decisions_total": 0,
            "policy_circuit_breaker_open_total": 0,
            "policy_model_call_total": 0,
            "policy_model_fallback_total": 0,
            "policy_model_shadow_agree_total": 0,
            "policy_model_canary_total": 0,
            "policy_circuit_breaker_trip_total": 0,
        }
        self._task_react_stop_reason_counts: dict[str, int] = {}
        self._kv_cache_by_stage: dict[str, dict[str, Any]] = {}
        self._estimated_cost = 0.0
        self._latencies: dict[str, LatencyStats] = {
            "planning_latency_ms": LatencyStats(),
            "search_latency_ms": LatencyStats(),
            "summarization_latency_ms": LatencyStats(),
            "reflection_latency_ms": LatencyStats(),
            "review_latency_ms": LatencyStats(),
            "nli_latency_ms": LatencyStats(),
            "repair_latency_ms": LatencyStats(),
            "report_latency_ms": LatencyStats(),
            "total_latency_ms": LatencyStats(),
        }

    def configure(self, *, recent_request_limit: int) -> None:
        with self._lock:
            limit = max(1, recent_request_limit)
            if limit == self._recent_request_limit:
                return
            self._recent_request_limit = limit
            self._recent_requests = deque(self._recent_requests, maxlen=limit)

    def reset(self) -> None:
        with self._lock:
            self._recent_requests = deque(maxlen=self._recent_request_limit)
            for key in list(self._counters.keys()):
                self._counters[key] = 0
            self._task_react_stop_reason_counts = {}
            self._kv_cache_by_stage.clear()
            self._estimated_cost = 0.0
            self._latencies = {
                "planning_latency_ms": LatencyStats(),
                "search_latency_ms": LatencyStats(),
                "summarization_latency_ms": LatencyStats(),
                "reflection_latency_ms": LatencyStats(),
                "review_latency_ms": LatencyStats(),
                "nli_latency_ms": LatencyStats(),
                "repair_latency_ms": LatencyStats(),
                "report_latency_ms": LatencyStats(),
                "total_latency_ms": LatencyStats(),
            }

    def increment(self, metric: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[metric] = self._counters.get(metric, 0) + amount

    def record_policy_observation(self, telemetry: dict[str, Any]) -> None:
        called = telemetry.get("model_called") is True
        shadow = telemetry.get("mode") == "shadow"
        canary = telemetry.get("mode") == "canary" and telemetry.get("canary_selected") is True
        with self._lock:
            if called:
                self._counters["policy_model_call_total"] += 1
                self._latencies.setdefault("policy_model_latency_ms", LatencyStats()).add(telemetry["latency_ms"])
            if telemetry.get("fallback_reason"):
                self._counters["policy_model_fallback_total"] += 1
            if shadow and called:
                self._counters["policy_shadow_decisions_total"] += 1
            if shadow and telemetry.get("shadow_agreement") is True:
                self._counters["policy_shadow_agreement_total"] += 1
                self._counters["policy_model_shadow_agree_total"] += 1
            if canary and called:
                self._counters["policy_canary_decisions_total"] += 1
                self._counters["policy_model_canary_total"] += 1
            if telemetry.get("circuit_tripped") is True:
                self._counters["policy_circuit_breaker_open_total"] += 1
                self._counters["policy_circuit_breaker_trip_total"] += 1

    def add_tokens(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        cached_prompt_tokens: int = 0,
        cache_creation_tokens: int = 0,
        uncached_prompt_tokens: int | None = None,
        stage: str | None = None,
        calls: int = 1,
    ) -> None:
        resolved_cached = max(0, int(cached_prompt_tokens or 0))
        resolved_creation = max(0, int(cache_creation_tokens or 0))
        resolved_uncached = (
            max(0, int(uncached_prompt_tokens))
            if uncached_prompt_tokens is not None
            else max(0, int(prompt_tokens or 0) - resolved_cached)
        )
        with self._lock:
            self._counters["prompt_tokens"] += prompt_tokens
            self._counters["completion_tokens"] += completion_tokens
            self._counters["total_tokens"] += total_tokens
            self._counters["cached_prompt_tokens"] += resolved_cached
            self._counters["cache_creation_tokens"] += resolved_creation
            self._counters["uncached_prompt_tokens"] += resolved_uncached
            self._estimated_cost += estimated_cost
            if stage:
                _add_kv_cache_stage(
                    self._kv_cache_by_stage,
                    stage,
                    calls=calls,
                    prompt_tokens=prompt_tokens,
                    cached_prompt_tokens=resolved_cached,
                    cache_creation_tokens=resolved_creation,
                    uncached_prompt_tokens=resolved_uncached,
                )

    def add_latency(self, metric: str, elapsed_ms: float) -> None:
        with self._lock:
            if metric not in self._latencies:
                self._latencies[metric] = LatencyStats()
            self._latencies[metric].add(elapsed_ms)

    def increment_task_react_stop_reason(self, reason: str) -> None:
        normalized = str(reason or "").strip() or "unknown"
        with self._lock:
            self._task_react_stop_reason_counts[normalized] = (
                self._task_react_stop_reason_counts.get(normalized, 0) + 1
            )

    def store_request(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._recent_requests.appendleft(_clone_dict(payload))

    def snapshot(
        self, *, include_recent_requests: bool = True, tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            latencies = {name: stats.to_dict() for name, stats in self._latencies.items()}
            estimated_cost = round(self._estimated_cost, 6)
            recent_requests = [
                _clone_dict(request)
                for request in self._recent_requests
                if tenant_id is None or request.get("tenant_id") == tenant_id
            ] if include_recent_requests else []
            stop_reason_counts = dict(self._task_react_stop_reason_counts)
            kv_cache_by_stage = _clone_dict(self._kv_cache_by_stage)

        request_total = counters.get("request_total", 0)
        success_total = counters.get("request_success_total", 0)
        partial_total = counters.get("request_partial_success_total", 0)
        failed_total = counters.get("request_failed_total", 0)
        cache_hits = counters.get("cache_hit_total", 0)
        cache_exact_hits = counters.get("cache_exact_hit_total", 0)
        cache_semantic_hits = counters.get("cache_semantic_hit_total", 0)
        cache_approximate_hits = counters.get("cache_approximate_hit_total", 0)
        cache_approximate_dense_hits = counters.get("cache_approximate_dense_hit_total", 0)
        cache_approximate_sparse_hits = counters.get("cache_approximate_sparse_hit_total", 0)
        cache_approximate_hybrid_hits = counters.get("cache_approximate_hybrid_hit_total", 0)
        cache_misses = counters.get("cache_miss_total", 0)
        cache_total = cache_hits + cache_misses
        task_react_round_total = counters.get("task_react_round_total", 0)
        task_react_stop_total = counters.get("task_react_stop_total", 0)
        cached_prompt_tokens = counters.get("cached_prompt_tokens", 0)
        cache_creation_tokens = counters.get("cache_creation_tokens", 0)
        uncached_prompt_tokens = counters.get("uncached_prompt_tokens", 0)
        prompt_tokens_total = counters.get("prompt_tokens", 0)
        kv_cache_ratio = _cache_hit_ratio(cached_prompt_tokens, prompt_tokens_total)

        return {
            "generated_at": _utc_now(),
            "counters": counters,
            "success_rate": round(((success_total + partial_total) / request_total), 4)
            if request_total
            else 0.0,
            "failure_rate": round((failed_total / request_total), 4) if request_total else 0.0,
            "cache_hit_total": cache_hits,
            "cache_exact_hit_total": cache_exact_hits,
            "cache_semantic_hit_total": cache_semantic_hits,
            "cache_approximate_hit_total": cache_approximate_hits,
            "cache_approximate_dense_hit_total": cache_approximate_dense_hits,
            "cache_approximate_sparse_hit_total": cache_approximate_sparse_hits,
            "cache_approximate_hybrid_hit_total": cache_approximate_hybrid_hits,
            "cache_miss_total": cache_misses,
            "cache_hit_rate": round((cache_hits / cache_total), 4) if cache_total else 0.0,
            "cached_prompt_tokens": cached_prompt_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "uncached_prompt_tokens": uncached_prompt_tokens,
            "cache_hit_ratio": kv_cache_ratio,
            "kv_cache_hit_ratio": kv_cache_ratio,
            "prompt_cache_hit_ratio": kv_cache_ratio,
            "kv_cache_by_stage": kv_cache_by_stage,
            "latencies_ms": latencies,
            "estimated_cost": estimated_cost,
            "task_react_stop_reason_counts": stop_reason_counts,
            "avg_task_react_rounds": round(task_react_round_total / task_react_stop_total, 4)
            if task_react_stop_total
            else 0.0,
            "recent_requests": recent_requests,
        }


metrics_registry = MetricsRegistry()


def _pricing_entry_from_catalog(model_catalog: dict[str, Any]) -> dict[str, float]:
    prompt_rate = float(model_catalog.get("prompt_per_1k_tokens", 0.0))
    completion_rate = float(model_catalog.get("completion_per_1k_tokens", 0.0))
    cached_prompt_rate = float(model_catalog.get("cached_prompt_per_1k_tokens", prompt_rate * 0.5))
    cache_creation_rate = float(model_catalog.get("cache_creation_per_1k_tokens", prompt_rate * 1.25))
    return {
        "prompt_per_1k_tokens": prompt_rate,
        "completion_per_1k_tokens": completion_rate,
        "cached_prompt_per_1k_tokens": cached_prompt_rate,
        "cache_creation_per_1k_tokens": cache_creation_rate,
    }


def resolve_pricing(
    provider: str | None,
    model: str | None,
    pricing_catalog: dict[str, Any] | None,
) -> tuple[dict[str, float], str]:
    catalog = pricing_catalog or DEFAULT_PRICING_CATALOG

    provider_key = (provider or "default").strip() or "default"
    model_key = (model or "default").strip() or "default"

    provider_catalog = catalog.get(provider_key) if isinstance(catalog, dict) else None
    if isinstance(provider_catalog, dict):
        model_catalog = provider_catalog.get(model_key) or provider_catalog.get("default")
        if isinstance(model_catalog, dict):
            return _pricing_entry_from_catalog(model_catalog), "configured"

    default_catalog = catalog.get("default", {}) if isinstance(catalog, dict) else {}
    model_catalog = default_catalog.get("default", {}) if isinstance(default_catalog, dict) else {}
    return _pricing_entry_from_catalog(model_catalog if isinstance(model_catalog, dict) else {}), "default"


def build_token_usage(
    *,
    prompt_text: str,
    completion_text: str,
    provider: str | None,
    model: str | None,
    pricing_catalog: dict[str, Any] | None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    cached_prompt_tokens = 0
    cache_creation_tokens = 0
    provider_calls = 1
    normalized = normalize_reported_usage(usage)
    if normalized is not None:
        prompt_tokens = normalized["prompt_tokens"]
        completion_tokens = normalized["completion_tokens"]
        total_tokens = normalized["total_tokens"]
        cached_prompt_tokens = normalized.get("cached_tokens", 0)
        cache_creation_tokens = normalized.get("cache_creation_tokens", 0)
        provider_calls = max(1, normalized.get("provider_calls", 1))
        estimated_calls = normalized.get("estimated_provider_calls", 0)
        if estimated_calls == 0:
            token_source = "reported"
        else:
            token_source = "estimated" if estimated_calls >= provider_calls else "mixed"
    else:
        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(completion_text)
        total_tokens = prompt_tokens + completion_tokens
        token_source = "estimated"

    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
    pricing, pricing_source = resolve_pricing(provider, model, pricing_catalog)
    estimated_cost = (
        ((uncached_prompt_tokens - cache_creation_tokens) / 1000) * pricing["prompt_per_1k_tokens"]
        + (cached_prompt_tokens / 1000) * pricing["cached_prompt_per_1k_tokens"]
        + (cache_creation_tokens / 1000) * pricing["cache_creation_per_1k_tokens"]
        + (completion_tokens / 1000) * pricing["completion_per_1k_tokens"]
    )
    cache_hit_ratio = _cache_hit_ratio(cached_prompt_tokens, prompt_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "cache_hit_ratio": cache_hit_ratio,
        "prompt_cache_hit_ratio": cache_hit_ratio,
        "provider_calls": provider_calls,
        "token_source": token_source,
        "estimated_cost": round(estimated_cost, 6),
        "pricing_source": pricing_source,
        "provider": provider,
        "model": model,
    }


class StageSpan:
    """Mutable stage span bound to a request trace."""

    def __init__(
        self,
        observer: RequestTrace,
        *,
        stage: str,
        scope: str,
        task_id: int | None = None,
        task_title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._observer = observer
        self._perf_start = perf_counter()
        self._trace = StageTrace(
            stage=stage,
            scope=scope,
            task_id=task_id,
            task_title=task_title,
            start_time=_utc_now(),
            metadata=_clone_dict(metadata),
        )
        self._observer._register_stage(self._trace)

    @property
    def trace(self) -> StageTrace:
        return self._trace

    def started_event(self) -> dict[str, Any]:
        return {
            "type": "stage_started",
            "request_id": self._observer.request_id,
            "stage": self._trace.stage,
            "scope": self._trace.scope,
            "task_id": self._trace.task_id,
            "task_title": self._trace.task_title,
            "start_time": self._trace.start_time,
            "metadata": _clone_dict(self._trace.metadata),
        }

    def complete(
        self,
        *,
        status: str,
        error: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        elapsed_ms = round((perf_counter() - self._perf_start) * 1000, 2)
        payload = self._observer._complete_stage(
            self._trace,
            elapsed_ms=elapsed_ms,
            status=status,
            error=error,
            metadata=metadata,
        )
        return {
            "type": "stage_completed",
            "request_id": self._observer.request_id,
            **payload,
        }


class RequestTrace:
    """Per-request trace collector feeding the global metrics registry."""

    def __init__(
        self,
        *,
        request_id: str,
        topic: str,
        search_api: str,
        provider: str | None,
        model: str | None,
        pricing_catalog: dict[str, Any] | None,
        tenant_id: str = "default",
    ) -> None:
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.topic = topic
        self.search_api = search_api
        self.provider = provider
        self.model = model
        self.pricing_catalog = pricing_catalog
        self._lock = Lock()
        self._perf_start = perf_counter()
        self.start_time = _utc_now()
        self.end_time: str | None = None
        self.elapsed_ms: float | None = None
        self.status = "in_progress"
        self.error: str | None = None
        self.fallback_count = 0
        self.fallback_reasons: list[str] = []
        self.degraded_reasons: list[str] = []
        self.stages: list[StageTrace] = []
        self.total_tasks = 0
        self.completed_tasks = 0
        self.skipped_tasks = 0
        self.failed_tasks = 0
        self.cache_hits = 0
        self.cache_exact_hits = 0
        self.cache_semantic_hits = 0
        self.cache_approximate_hits = 0
        self.cache_approximate_dense_hits = 0
        self.cache_approximate_sparse_hits = 0
        self.cache_approximate_hybrid_hits = 0
        self.cache_misses = 0
        self.last_search_cache_details: dict[str, Any] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated_cost = 0.0
        self.report_markdown: str | None = None
        self.todo_items: list[dict[str, Any]] = []
        self.report_note_id: str | None = None
        self.report_note_path: str | None = None
        self.token_sources: set[str] = set()
        self.reflection_triggered = False
        self.reflection_reason: str | None = None
        self.reflection_gap_signals: list[str] = []
        self.reflection_added_tasks = 0
        self.review_summary: dict[str, Any] = {}
        self.task_react_rounds = 0
        self.task_react_continue_count = 0
        self.task_react_stop_count = 0
        self.task_react_stop_reasons: dict[str, int] = {}
        self.report_repair_triggered = False
        self.report_repair_added_tasks = 0
        self.report_repair_cycles = 0
        self.note_memory_queries = 0
        self.note_memory_hits = 0
        self.note_memory_prompt_injections = 0
        self.note_memory_last_match_types: list[str] = []
        self.strategy_memory_queries = 0
        self.strategy_memory_hits = 0
        self.strategy_memory_prompt_injections = 0
        self.strategy_memory_last_match_kinds: list[str] = []
        self.strategy_memory_hit_ids: list[str] = []
        self.memory_shadow_mode = False
        self.memory_shadow_suppressions = 0
        self.memory_shadow_suppressed_stages: dict[str, int] = {}
        self.skill_pack_id = ""
        self.skill_pack_version = ""
        self.skill_pack_match_score = 0.0
        self.skill_pack_fallback_reason = ""
        self.skill_pack_shadow = False
        self.policy_decisions: list[dict[str, Any]] = []
        self.policy_shadow_decisions: list[dict[str, Any]] = []
        self.policy_model_latency = LatencyStats()
        self.cached_prompt_tokens = 0
        self.cache_creation_tokens = 0
        self.uncached_prompt_tokens = 0
        self.kv_cache_by_stage: dict[str, dict[str, Any]] = {}
        self.claim_verification_summary: dict[str, Any] = {}
        metrics_registry.increment("request_total")

    def record_policy_decision(self, decision: dict[str, Any]) -> None:
        with self._lock:
            self.policy_decisions.append(_clone_dict(decision))
            telemetry = decision.get("telemetry") or {}
            if telemetry.get("model_called") is True:
                self.policy_model_latency.add(telemetry["latency_ms"])

    def record_policy_shadow_decision(self, task_id: int, round_index: int, telemetry: dict[str, Any]) -> None:
        with self._lock:
            self.policy_shadow_decisions.append(_clone_dict({
                "task_id": task_id, "round": round_index, **telemetry,
            }))
            if telemetry.get("model_called") is True:
                self.policy_model_latency.add(telemetry["latency_ms"])

    def record_policy_execution(self, task_id: int, round_index: int, decision: dict[str, Any]) -> None:
        with self._lock:
            for entry in reversed(self.policy_decisions):
                if entry.get("task_id") == task_id and entry.get("round") == round_index:
                    entry["decision"] = _clone_dict(decision)
                    entry["applied"] = True
                    break

    def start_stage(
        self,
        stage: str,
        *,
        scope: str,
        task_id: int | None = None,
        task_title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StageSpan:
        return StageSpan(
            self,
            stage=stage,
            scope=scope,
            task_id=task_id,
            task_title=task_title,
            metadata=metadata,
        )

    def record_search_attempt(
        self,
        *,
        cache_hit: bool,
        success: bool,
        error: Any = None,
        cache_strategy: str = "miss",
        cache_metadata: dict[str, Any] | None = None,
    ) -> None:
        metrics_registry.increment("search_call_total")
        if success:
            metrics_registry.increment("search_success_total")
        else:
            metrics_registry.increment("search_failed_total")

        hit_mode = str((cache_metadata or {}).get("cache_hit_mode") or cache_strategy or "miss")
        normalized_strategy = "miss"
        approximate_hit_mode: str | None = None
        if hit_mode == "exact" or cache_strategy == "exact":
            normalized_strategy = "exact"
        elif hit_mode in {"approximate_dense", "approximate_sparse", "approximate_hybrid"}:
            normalized_strategy = "approximate"
            approximate_hit_mode = hit_mode
        elif cache_strategy in {"semantic", "semantic_ann", "semantic_lexical"}:
            normalized_strategy = "approximate"
            if cache_strategy == "semantic_ann":
                approximate_hit_mode = "approximate_dense"
            elif cache_strategy == "semantic_lexical":
                approximate_hit_mode = "approximate_sparse"

        if cache_metadata:
            with self._lock:
                self.last_search_cache_details = _clone_dict(cache_metadata)

        if cache_hit:
            metrics_registry.increment("cache_hit_total")
            with self._lock:
                self.cache_hits += 1
            if normalized_strategy == "exact":
                metrics_registry.increment("cache_exact_hit_total")
                with self._lock:
                    self.cache_exact_hits += 1
            elif normalized_strategy == "approximate":
                metrics_registry.increment("cache_semantic_hit_total")
                metrics_registry.increment("cache_approximate_hit_total")
                with self._lock:
                    self.cache_semantic_hits += 1
                    self.cache_approximate_hits += 1
                if approximate_hit_mode == "approximate_dense":
                    metrics_registry.increment("cache_approximate_dense_hit_total")
                    with self._lock:
                        self.cache_approximate_dense_hits += 1
                elif approximate_hit_mode == "approximate_sparse":
                    metrics_registry.increment("cache_approximate_sparse_hit_total")
                    with self._lock:
                        self.cache_approximate_sparse_hits += 1
                elif approximate_hit_mode == "approximate_hybrid":
                    metrics_registry.increment("cache_approximate_hybrid_hit_total")
                    with self._lock:
                        self.cache_approximate_hybrid_hits += 1
        else:
            metrics_registry.increment("cache_miss_total")
            with self._lock:
                self.cache_misses += 1

        if error:
            self.record_degraded(f"search_error:{_safe_error(error)}")

    def record_llm_call(
        self,
        *,
        success: bool,
        prompt_text: str,
        completion_text: str,
        usage: dict[str, Any] | None = None,
        error: Any = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        token_usage = build_token_usage(
            prompt_text=prompt_text,
            completion_text=completion_text,
            provider=self.provider,
            model=self.model,
            pricing_catalog=self.pricing_catalog,
            usage=usage,
        )
        calls = token_usage["provider_calls"]
        metrics_registry.increment("llm_call_total", calls)
        metrics_registry.increment("llm_success_total" if success else "llm_failed_total", calls)
        metrics_registry.add_tokens(
            prompt_tokens=token_usage["prompt_tokens"],
            completion_tokens=token_usage["completion_tokens"],
            total_tokens=token_usage["total_tokens"],
            estimated_cost=token_usage["estimated_cost"],
            cached_prompt_tokens=token_usage["cached_prompt_tokens"],
            cache_creation_tokens=token_usage["cache_creation_tokens"],
            uncached_prompt_tokens=token_usage["uncached_prompt_tokens"],
            stage=stage,
            calls=calls,
        )

        with self._lock:
            self.prompt_tokens += token_usage["prompt_tokens"]
            self.completion_tokens += token_usage["completion_tokens"]
            self.total_tokens += token_usage["total_tokens"]
            self.cached_prompt_tokens += token_usage["cached_prompt_tokens"]
            self.cache_creation_tokens += token_usage["cache_creation_tokens"]
            self.uncached_prompt_tokens += token_usage["uncached_prompt_tokens"]
            self.estimated_cost += token_usage["estimated_cost"]
            if token_usage["token_source"] == "mixed":
                self.token_sources.update(("reported", "estimated"))
            else:
                self.token_sources.add(token_usage["token_source"])
            if stage:
                _add_kv_cache_stage(
                    self.kv_cache_by_stage,
                    stage,
                    calls=calls,
                    prompt_tokens=token_usage["prompt_tokens"],
                    cached_prompt_tokens=token_usage["cached_prompt_tokens"],
                    cache_creation_tokens=token_usage["cache_creation_tokens"],
                    uncached_prompt_tokens=token_usage["uncached_prompt_tokens"],
                )

        if error:
            self.record_degraded(f"llm_error:{_safe_error(error)}")

        return token_usage

    def record_fallback(self, reason: str) -> dict[str, Any]:
        metrics_registry.increment("fallback_trigger_total")
        with self._lock:
            self.fallback_count += 1
            self.fallback_reasons.append(reason)
        return {
            "type": "fallback_triggered",
            "request_id": self.request_id,
            "reason": reason,
            "fallback_count": self.fallback_count,
        }

    def record_degraded(self, reason: str) -> dict[str, Any]:
        with self._lock:
            if reason not in self.degraded_reasons:
                self.degraded_reasons.append(reason)
        return {
            "type": "degraded_response",
            "request_id": self.request_id,
            "reason": reason,
            "degraded": True,
        }

    def record_reflection_call(
        self,
        *,
        reason: str,
        gap_signals: list[str] | None = None,
        added_tasks: int = 0,
    ) -> None:
        metrics_registry.increment("reflection_call_total")
        if added_tasks > 0:
            metrics_registry.increment("reflection_replan_total")

        with self._lock:
            self.reflection_triggered = True
            self.reflection_reason = reason.strip() or reason
            self.reflection_gap_signals = list(gap_signals or [])
            self.reflection_added_tasks = max(0, int(added_tasks or 0))

    def record_reflection_skip(
        self,
        *,
        reason: str,
        gap_signals: list[str] | None = None,
    ) -> None:
        metrics_registry.increment("reflection_skipped_total")
        with self._lock:
            self.reflection_triggered = True
            self.reflection_reason = reason.strip() or reason
            self.reflection_gap_signals = list(gap_signals or [])
            self.reflection_added_tasks = 0

    def set_task_totals(self, *, total_tasks: int) -> None:
        with self._lock:
            self.total_tasks = total_tasks

    def update_task_status_counts(self, *, completed: int = 0, skipped: int = 0, failed: int = 0) -> None:
        with self._lock:
            self.completed_tasks += completed
            self.skipped_tasks += skipped
            self.failed_tasks += failed

    def record_review_summary(self, summary: dict[str, Any]) -> None:
        metrics_registry.increment("review_call_total")
        metrics_registry.increment("review_issue_total", int(summary.get("issue_count") or 0))
        with self._lock:
            self.review_summary = _clone_dict(summary)

    def record_task_react_round(self) -> None:
        metrics_registry.increment("task_react_round_total")
        with self._lock:
            self.task_react_rounds += 1

    def record_task_react_continue(self) -> None:
        metrics_registry.increment("task_react_continue_total")
        with self._lock:
            self.task_react_continue_count += 1

    def record_task_react_stop(self, reason: str) -> None:
        normalized = str(reason or "").strip() or "unknown"
        metrics_registry.increment("task_react_stop_total")
        metrics_registry.increment_task_react_stop_reason(normalized)
        with self._lock:
            self.task_react_stop_count += 1
            self.task_react_stop_reasons[normalized] = (
                self.task_react_stop_reasons.get(normalized, 0) + 1
            )

    def record_report_repair(self, *, added_tasks: int, triggered: bool = True) -> None:
        if triggered:
            metrics_registry.increment("report_repair_trigger_total")
        if added_tasks > 0:
            metrics_registry.increment("report_repair_added_task_total", added_tasks)
        with self._lock:
            self.report_repair_triggered = self.report_repair_triggered or triggered
            self.report_repair_added_tasks += max(0, int(added_tasks or 0))
            if triggered:
                self.report_repair_cycles += 1

    def record_note_memory_query(
        self,
        *,
        hit_count: int,
        match_types: list[str] | None = None,
    ) -> None:
        metrics_registry.increment("note_memory_query_total")
        if hit_count > 0:
            metrics_registry.increment("note_memory_hit_total", hit_count)
        else:
            metrics_registry.increment("note_memory_miss_total")

        normalized_types = [
            str(match_type).strip()
            for match_type in match_types or []
            if str(match_type).strip()
        ]
        with self._lock:
            self.note_memory_queries += 1
            self.note_memory_hits += max(0, int(hit_count or 0))
            if normalized_types:
                self.note_memory_last_match_types = normalized_types

    def record_note_memory_prompt_injection(
        self,
        *,
        match_types: list[str] | None = None,
    ) -> None:
        metrics_registry.increment("note_memory_prompt_injection_total")
        normalized_types = [
            str(match_type).strip()
            for match_type in match_types or []
            if str(match_type).strip()
        ]
        with self._lock:
            self.note_memory_prompt_injections += 1
            if normalized_types:
                self.note_memory_last_match_types = normalized_types

    def record_strategy_memory_query(
        self,
        *,
        hit_count: int,
        match_kinds: list[str] | None = None,
        strategy_ids: list[str] | None = None,
    ) -> None:
        metrics_registry.increment("strategy_memory_query_total")
        if hit_count > 0:
            metrics_registry.increment("strategy_memory_hit_total", hit_count)
        else:
            metrics_registry.increment("strategy_memory_miss_total")

        normalized_kinds = [
            str(match_kind).strip()
            for match_kind in match_kinds or []
            if str(match_kind).strip()
        ]
        normalized_ids = [
            str(strategy_id).strip()
            for strategy_id in strategy_ids or []
            if str(strategy_id).strip()
        ]
        with self._lock:
            self.strategy_memory_queries += 1
            self.strategy_memory_hits += max(0, int(hit_count or 0))
            if normalized_kinds:
                self.strategy_memory_last_match_kinds = normalized_kinds
            # Accumulated across the whole request so usage can be flushed to the cards once,
            # instead of writing on every retrieval in the hot path.
            for strategy_id in normalized_ids:
                if strategy_id not in self.strategy_memory_hit_ids:
                    self.strategy_memory_hit_ids.append(strategy_id)

    def record_strategy_memory_prompt_injection(
        self,
        *,
        match_kinds: list[str] | None = None,
    ) -> None:
        metrics_registry.increment("strategy_memory_prompt_injection_total")
        normalized_kinds = [
            str(match_kind).strip()
            for match_kind in match_kinds or []
            if str(match_kind).strip()
        ]
        with self._lock:
            self.strategy_memory_prompt_injections += 1
            if normalized_kinds:
                self.strategy_memory_last_match_kinds = normalized_kinds

    def record_memory_shadow_suppression(self, stage: str) -> None:
        """Record that a rendered memory block was withheld from the prompt."""

        metrics_registry.increment("memory_shadow_suppression_total")
        normalized_stage = str(stage or "unknown").strip() or "unknown"
        with self._lock:
            self.memory_shadow_suppressions += 1
            self.memory_shadow_suppressed_stages[normalized_stage] = (
                self.memory_shadow_suppressed_stages.get(normalized_stage, 0) + 1
            )

    def record_skill_pack_selection(
        self,
        *,
        pack_id: str,
        version: str = "",
        match_score: float = 0.0,
        fallback_reason: str = "",
        shadow: bool = False,
    ) -> None:
        """Record which research skill pack was selected for this request.

        Captured once per request so eval reports can stratify outcomes by pack, and so a
        fallback to the neutral pack is always attributable to a reason.
        """

        metrics_registry.increment("skill_pack_selection_total")
        if fallback_reason:
            metrics_registry.increment("skill_pack_fallback_total")
        with self._lock:
            self.skill_pack_id = str(pack_id or "").strip()
            self.skill_pack_version = str(version or "").strip()
            self.skill_pack_match_score = round(float(match_score or 0.0), 4)
            self.skill_pack_fallback_reason = str(fallback_reason or "").strip()
            self.skill_pack_shadow = bool(shadow)

    def record_nli_verification(
        self,
        summary: dict[str, Any],
        *,
        elapsed_ms: float | None = None,
    ) -> None:
        if elapsed_ms is not None and elapsed_ms >= 0:
            metrics_registry.add_latency("nli_latency_ms", float(elapsed_ms))
        with self._lock:
            self.claim_verification_summary = _clone_dict(summary)

    def restore_review_result(
        self,
        review_summary: dict[str, Any],
        claim_verification_summary: dict[str, Any],
    ) -> None:
        """Populate a resumed trace without recording a second review or NLI execution."""
        with self._lock:
            self.review_summary = _clone_dict(review_summary)
            self.claim_verification_summary = _clone_dict(claim_verification_summary)

    def attach_result(
        self,
        *,
        report_markdown: str | None,
        todo_items: list[dict[str, Any]] | None = None,
        report_note_id: str | None = None,
        report_note_path: str | None = None,
    ) -> None:
        with self._lock:
            self.report_markdown = report_markdown.strip() if isinstance(report_markdown, str) else None
            self.todo_items = _clone_dict({"items": todo_items or []}).get("items", [])
            self.report_note_id = report_note_id
            self.report_note_path = report_note_path

    def complete_request(self, *, status: str, error: Any = None) -> dict[str, Any]:
        elapsed_ms = round((perf_counter() - self._perf_start) * 1000, 2)
        with self._lock:
            self.status = status
            self.error = _safe_error(error)
            self.end_time = _utc_now()
            self.elapsed_ms = elapsed_ms

        metrics_registry.add_latency("total_latency_ms", elapsed_ms)
        if status == "success":
            metrics_registry.increment("request_success_total")
        elif status == "partial_success":
            metrics_registry.increment("request_partial_success_total")
        else:
            metrics_registry.increment("request_failed_total")

        payload = self.snapshot()
        metrics_registry.store_request(payload)
        return payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            token_source = (
                "mixed"
                if len(self.token_sources) > 1
                else next(iter(self.token_sources), "unavailable")
            )
            return {
                "request_id": self.request_id,
                "tenant_id": self.tenant_id,
                "topic": self.topic,
                "search_api": self.search_api,
                "provider": self.provider,
                "model": self.model,
                "status": self.status,
                "error": self.error,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "elapsed_ms": self.elapsed_ms,
                "fallback_triggered": self.fallback_count > 0,
                "fallback_reasons": list(self.fallback_reasons),
                "degraded": bool(self.degraded_reasons),
                "degraded_reasons": list(self.degraded_reasons),
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "skipped_tasks": self.skipped_tasks,
                "failed_tasks": self.failed_tasks,
                "cache_hits": self.cache_hits,
                "cache_exact_hits": self.cache_exact_hits,
                "cache_semantic_hits": self.cache_semantic_hits,
                "cache_approximate_hits": self.cache_approximate_hits,
                "cache_approximate_dense_hits": self.cache_approximate_dense_hits,
                "cache_approximate_sparse_hits": self.cache_approximate_sparse_hits,
                "cache_approximate_hybrid_hits": self.cache_approximate_hybrid_hits,
                "cache_misses": self.cache_misses,
                "last_search_cache_details": _clone_dict(self.last_search_cache_details),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cached_prompt_tokens": self.cached_prompt_tokens,
                "cache_creation_tokens": self.cache_creation_tokens,
                "uncached_prompt_tokens": self.uncached_prompt_tokens,
                "cache_hit_ratio": _cache_hit_ratio(self.cached_prompt_tokens, self.prompt_tokens),
                "prompt_cache_hit_ratio": _cache_hit_ratio(self.cached_prompt_tokens, self.prompt_tokens),
                "kv_cache_by_stage": _clone_dict(self.kv_cache_by_stage),
                "token_source": token_source,
                "estimated_cost": round(self.estimated_cost, 6),
                "reflection_triggered": self.reflection_triggered,
                "reflection_reason": self.reflection_reason,
                "reflection_gap_signals": list(self.reflection_gap_signals),
                "reflection_added_tasks": self.reflection_added_tasks,
                "review_summary": _clone_dict(self.review_summary),
                "claim_verification_summary": _clone_dict(self.claim_verification_summary),
                "task_react_rounds": self.task_react_rounds,
                "task_react_continue_count": self.task_react_continue_count,
                "task_react_stop_count": self.task_react_stop_count,
                "task_react_stop_reasons": dict(self.task_react_stop_reasons),
                "avg_task_react_rounds": round(
                    self.task_react_rounds / self.task_react_stop_count,
                    4,
                )
                if self.task_react_stop_count
                else 0.0,
                "report_repair_triggered": self.report_repair_triggered,
                "report_repair_added_tasks": self.report_repair_added_tasks,
                "report_repair_cycles": self.report_repair_cycles,
                "note_memory_queries": self.note_memory_queries,
                "note_memory_hits": self.note_memory_hits,
                "note_memory_prompt_injections": self.note_memory_prompt_injections,
                "note_memory_last_match_types": list(self.note_memory_last_match_types),
                "strategy_memory_queries": self.strategy_memory_queries,
                "strategy_memory_hits": self.strategy_memory_hits,
                "strategy_memory_prompt_injections": self.strategy_memory_prompt_injections,
                "strategy_memory_last_match_kinds": list(self.strategy_memory_last_match_kinds),
                "strategy_memory_hit_ids": list(self.strategy_memory_hit_ids),
                "memory_shadow_mode": self.memory_shadow_mode,
                "memory_shadow_suppressions": self.memory_shadow_suppressions,
                "memory_shadow_suppressed_stages": dict(self.memory_shadow_suppressed_stages),
                "skill_pack_id": self.skill_pack_id,
                "skill_pack_version": self.skill_pack_version,
                "skill_pack_match_score": self.skill_pack_match_score,
                "skill_pack_fallback_reason": self.skill_pack_fallback_reason,
                "skill_pack_shadow": self.skill_pack_shadow,
                "policy_decisions": _clone_dict({"items": self.policy_decisions}).get("items", []),
                "policy_shadow_decisions": _clone_dict({"items": self.policy_shadow_decisions}).get("items", []),
                "policy_model_latency_ms": self.policy_model_latency.to_dict(),
                "report_markdown": self.report_markdown,
                "todo_items": _clone_dict({"items": self.todo_items}).get("items", []),
                "report_note_id": self.report_note_id,
                "report_note_path": self.report_note_path,
                "stages": [stage.to_dict() for stage in self.stages],
            }

    def metrics_event(self) -> dict[str, Any]:
        return {
            "type": "metrics_snapshot",
            "request_id": self.request_id,
            "request_metrics": self.snapshot(),
            "aggregate_metrics": metrics_registry.snapshot(include_recent_requests=False),
        }

    def _register_stage(self, stage: StageTrace) -> None:
        with self._lock:
            self.stages.append(stage)

    def _complete_stage(
        self,
        stage: StageTrace,
        *,
        elapsed_ms: float,
        status: str,
        error: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage.end_time = _utc_now()
        stage.elapsed_ms = elapsed_ms
        stage.status = status
        stage.error = _safe_error(error)
        if metadata:
            stage.metadata.update(_clone_dict(metadata))

        metric_name = {
            "planning": "planning_latency_ms",
            "search": "search_latency_ms",
            "summarization": "summarization_latency_ms",
            "reflection": "reflection_latency_ms",
            "review": "review_latency_ms",
            "report": "report_latency_ms",
        }.get(stage.stage)
        if metric_name:
            metrics_registry.add_latency(metric_name, elapsed_ms)

        if status not in {"success", "completed"} and stage.error:
            self.record_degraded(f"{stage.stage}_error:{stage.error}")

        return stage.to_dict()
