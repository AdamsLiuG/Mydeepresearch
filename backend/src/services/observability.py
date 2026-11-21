"""Non-intrusive observability adapters for external tracing backends (e.g., Langfuse)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from config import Configuration

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
