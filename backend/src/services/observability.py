"""Non-intrusive observability adapters for external tracing backends (e.g., Langfuse)."""

from __future__ import annotations

import logging
import re
import secrets
import time
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from config import Configuration

_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_FORBIDDEN_ATTR_SUBSTRINGS = (
    "prompt",
    "completion",
    "content",
    "secret",
    "token",
    "password",
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "credential",
)

try:  # pragma: no cover - optional dependency
    from langfuse import Langfuse as _LangfuseClient
except Exception:  # pragma: no cover - optional dependency
    _LangfuseClient = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class LangfuseTraceAdapter:
    """Opt-in exporter translating RequestTrace snapshots into Langfuse traces & spans.

    When ``langfuse_enabled`` is False or the optional ``langfuse`` SDK / keys are absent,
    every method degrades to a fast zero-overhead no-op.
    """

    def __init__(
        self,
        config: Configuration,
        *,
        client: Any | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._initialized = client is not None

    @property
    def enabled(self) -> bool:
        if not getattr(self._config, "langfuse_enabled", False):
            return False
        if self._client is not None:
            return True
        return bool(
            _LangfuseClient is not None
            and (getattr(self._config, "langfuse_public_key", None) or "").strip()
            and (getattr(self._config, "langfuse_secret_key", None) or "").strip()
        )

    def _get_client(self) -> Any | None:
        if not self.enabled:
            return None
        if self._initialized:
            return self._client
        self._initialized = True
        if _LangfuseClient is None:
            return None
        kwargs: dict[str, Any] = {
            "public_key": str(self._config.langfuse_public_key or "").strip(),
            "secret_key": str(self._config.langfuse_secret_key or "").strip(),
        }
        host = str(self._config.langfuse_host or "").strip()
        if host:
            kwargs["host"] = host
        try:
            self._client = _LangfuseClient(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime SDK initialization guard
            logger.warning("Langfuse client initialization failed: %s", exc)
            self._client = None
        return self._client

    def export_request_trace(self, snapshot: Mapping[str, Any]) -> bool:
        """Export one completed RequestTrace snapshot to Langfuse; never raises."""

        client = self._get_client()
        if client is None or not isinstance(snapshot, Mapping):
            return False

        try:
            request_id = str(snapshot.get("request_id") or "").strip() or "unknown"
            trace = client.trace(
                id=request_id,
                name="deep_research",
                input={"topic": str(snapshot.get("topic") or "")},
                output={
                    "status": str(snapshot.get("status") or ""),
                    "elapsed_ms": float(snapshot.get("elapsed_ms") or 0.0),
                    "degraded": bool(snapshot.get("degraded")),
                    "degraded_reasons": list(snapshot.get("degraded_reasons") or []),
                },
                metadata={
                    "search_api": snapshot.get("search_api"),
                    "provider": snapshot.get("provider"),
                    "model": snapshot.get("model"),
                    "skill_pack_id": snapshot.get("skill_pack_id"),
                    "cache_hits": snapshot.get("cache_hits"),
                    "cache_misses": snapshot.get("cache_misses"),
                    "token_source": snapshot.get("token_source"),
                    "estimated_cost": snapshot.get("estimated_cost"),
                },
            )

            for stage in snapshot.get("stages") or []:
                if not isinstance(stage, Mapping):
                    continue
                trace.span(
                    name=str(stage.get("stage") or "stage"),
                    metadata={
                        "scope": stage.get("scope"),
                        "task_id": stage.get("task_id"),
                        "status": stage.get("status"),
                        "elapsed_ms": stage.get("elapsed_ms"),
                        **dict(stage.get("metadata") or {}),
                    },
                )

            prompt_tokens = int(snapshot.get("prompt_tokens") or 0)
            completion_tokens = int(snapshot.get("completion_tokens") or 0)
            total_tokens = int(snapshot.get("total_tokens") or (prompt_tokens + completion_tokens))
            if total_tokens > 0:
                trace.generation(
                    name="llm_usage_summary",
                    model=str(snapshot.get("model") or "unknown"),
                    usage={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    },
                    metadata={
                        "token_source": snapshot.get("token_source"),
                        "llm_calls": snapshot.get("llm_calls"),
                        "estimated_cost": snapshot.get("estimated_cost"),
                        "cached_prompt_tokens": int(snapshot.get("cached_prompt_tokens") or 0),
                        "cache_creation_tokens": int(snapshot.get("cache_creation_tokens") or 0),
                        "uncached_prompt_tokens": int(snapshot.get("uncached_prompt_tokens") or 0),
                        "cache_hit_ratio": float(snapshot.get("cache_hit_ratio") or 0.0),
                        "kv_cache_by_stage": dict(snapshot.get("kv_cache_by_stage") or {}),
                    },
                )

            review_status = str(snapshot.get("review_overall_status") or "").strip().lower()
            if review_status in {"passed", "warning", "blocked"}:
                score_value = {"passed": 1.0, "warning": 0.5, "blocked": 0.0}[review_status]
                trace.score(
                    name="review_status",
                    value=score_value,
                    comment=f"review_overall_status={review_status}",
                )

            flush = getattr(client, "flush", None)
            if callable(flush):
                flush()
            return True
        except Exception as exc:
            logger.warning("Langfuse trace export failed: %s", exc)
            return False


def parse_traceparent(header: Any) -> tuple[str, str, str] | None:
    """Validate and unpack a W3C traceparent header (`00-<32hex>-<16hex>-<2hex>`)."""
    if not isinstance(header, str):
        return None
    cleaned = header.strip().lower()
    if len(cleaned) != 55:
        return None
    match = _TRACEPARENT_RE.fullmatch(cleaned)
    if match is None:
        return None
    trace_id, span_id, flags = match.groups()
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id, span_id, flags


def format_traceparent(trace_id: str, span_id: str, flags: str = "01") -> str:
    """Format and re-validate a W3C traceparent string."""
    candidate = f"00-{trace_id.lower()}-{span_id.lower()}-{flags.lower()}"
    if parse_traceparent(candidate) is None:
        raise ValueError("invalid W3C traceparent components")
    return candidate


def new_traceparent(*, sampled: bool = True) -> str:
    """Mint a cryptographically random W3C traceparent value."""
    trace_id = secrets.token_hex(16)
    while trace_id == "0" * 32:  # pragma: no cover
        trace_id = secrets.token_hex(16)
    span_id = secrets.token_hex(8)
    while span_id == "0" * 16:  # pragma: no cover
        span_id = secrets.token_hex(8)
    return format_traceparent(trace_id, span_id, "01" if sampled else "00")


def child_traceparent(parent: Any, *, sampled: bool | None = None) -> str:
    """Derive a child span ID preserving trace_id when valid, or mint a fresh root."""
    parsed = parse_traceparent(parent)
    if parsed is None:
        return new_traceparent(sampled=True if sampled is None else sampled)
    trace_id, _, parent_flags = parsed
    flags = ("01" if sampled else "00") if sampled is not None else parent_flags
    span_id = secrets.token_hex(8)
    while span_id == "0" * 16:  # pragma: no cover
        span_id = secrets.token_hex(8)
    return format_traceparent(trace_id, span_id, flags)


def _sanitize_otel_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Strip prompt text, secrets, credentials, and non-scalar bloat from OTel attributes."""
    if not isinstance(attributes, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for raw_key, value in attributes.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key or len(key) > 96:
            continue
        lowered = key.lower()
        if (
            any(part in lowered for part in _FORBIDDEN_ATTR_SUBSTRINGS)
            and lowered not in {"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.usage.total_tokens"}
        ):
            continue
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, int):
            cleaned[key] = value
        elif isinstance(value, float):
            if value == value and abs(value) != float("inf"):
                cleaned[key] = round(value, 6)
        elif isinstance(value, str):
            safe_val = value.replace("\r", " ").replace("\n", " ").strip()
            if safe_val:
                cleaned[key] = safe_val[:160]
    return cleaned


@dataclass(frozen=True)
class RecordedSpan:
    """In-memory span record emitted by OTelTraceAdapter for telemetry & testing."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    duration_ms: float = 0.0


class OTelTraceAdapter:
    """Lazy-imported OpenTelemetry GenAI tracer with strict redaction and no-op fallback."""

    def __init__(self, config: Any, *, metrics: Any | None = None) -> None:
        self._config = config
        self._metrics = metrics
        self._service_name = str(getattr(config, "v4_otel_service_name", "deepresearch-backend") or "deepresearch-backend").strip()
        self._endpoint = str(getattr(config, "v4_otel_otlp_endpoint", "") or "").strip()
        self._sample_ratio = min(1.0, max(0.0, float(getattr(config, "v4_otel_sample_ratio", 1.0) or 1.0)))
        self._lock = Lock()
        self._spans: deque[RecordedSpan] = deque(maxlen=256)
        self._sdk_tracer: Any | None = None
        self._sdk_checked = False

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._config, "v4_otel_enabled", False)) and self._sample_ratio > 0.0

    def _get_sdk_tracer(self) -> Any | None:
        if not self.enabled:
            return None
        if self._sdk_checked:
            return self._sdk_tracer
        self._sdk_checked = True
        try:  # pragma: no cover - optional dependency
            from opentelemetry import trace

            self._sdk_tracer = trace.get_tracer(self._service_name)
        except Exception:
            self._sdk_tracer = None
        return self._sdk_tracer

    def recorded_spans(self) -> list[RecordedSpan]:
        with self._lock:
            return list(self._spans)

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        traceparent: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if not self.enabled:
            yield {}
            return
        parsed = parse_traceparent(traceparent)
        trace_id = parsed[0] if parsed else parse_traceparent(new_traceparent())[0]  # type: ignore[index]
        parent_span_id = parsed[1] if parsed else None
        span_id = secrets.token_hex(8)
        while span_id == "0" * 16:  # pragma: no cover
            span_id = secrets.token_hex(8)
        safe_attrs = _sanitize_otel_attributes({
            "service.name": self._service_name,
            **(attributes or {}),
        })
        span_ctx: dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": span_id,
            "traceparent": format_traceparent(trace_id, span_id, parsed[2] if parsed else "01"),
            "status": "ok",
            "attributes": safe_attrs,
        }
        started = time.perf_counter()
        sdk_tracer = self._get_sdk_tracer()
        sdk_ctx = (
            sdk_tracer.start_as_current_span(name, attributes=safe_attrs)
            if sdk_tracer is not None
            else None
        )
        try:
            if sdk_ctx is not None:  # pragma: no cover
                with sdk_ctx:
                    yield span_ctx
            else:
                yield span_ctx
        except BaseException as exc:
            span_ctx["status"] = "error"
            span_ctx["attributes"]["error.type"] = type(exc).__name__
            raise
        finally:
            elapsed = round((time.perf_counter() - started) * 1000.0, 3)
            recorded = RecordedSpan(
                name=str(name)[:80],
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                attributes=_sanitize_otel_attributes(span_ctx["attributes"]),
                status=str(span_ctx.get("status") or "ok"),
                duration_ms=elapsed,
            )
            with self._lock:
                self._spans.append(recorded)
            if self._metrics is not None and hasattr(self._metrics, "record_v4_otel_span"):
                self._metrics.record_v4_otel_span(str(name)[:32], recorded.status)

    def export_request_trace(
        self,
        snapshot: dict[str, Any] | None,
        *,
        traceparent: str | None = None,
    ) -> bool:
        """Emit summarized GenAI request and stage spans without exposing prompt content."""
        if not self.enabled or not isinstance(snapshot, dict):
            return False
        attrs = {
            "gen_ai.system": str(snapshot.get("provider") or "unknown"),
            "gen_ai.operation.name": "research_workflow",
            "gen_ai.request.model": str(snapshot.get("model") or "unknown"),
            "gen_ai.usage.input_tokens": int(snapshot.get("prompt_tokens") or 0),
            "gen_ai.usage.output_tokens": int(snapshot.get("completion_tokens") or 0),
            "deepresearch.tenant_id": str(snapshot.get("tenant_id") or "default"),
            "deepresearch.request_id": str(snapshot.get("request_id") or "unknown"),
            "deepresearch.status": str(snapshot.get("status") or "unknown"),
            "deepresearch.elapsed_ms": float(snapshot.get("elapsed_ms") or 0.0),
        }
        with self.start_span("deepresearch.request", traceparent=traceparent, attributes=attrs) as ctx:
            if snapshot.get("status") not in {"success", "completed", "partial_success", "awaiting_approval"}:
                ctx["status"] = "error"
        return True

