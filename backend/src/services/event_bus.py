"""Stream event bus with in-memory and optional Redis Streams replay backends."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

try:  # pragma: no cover - optional dependency
    import redis as _redis_mod
except Exception:  # pragma: no cover - optional dependency
    _redis_mod = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventEnvelope:
    """Single persisted SSE event with a replayable monotonic/stream ID."""

    event_id: str
    payload: dict[str, Any]


class StreamEventBus(Protocol):
    """Protocol implemented by SSE event publish/replay backends."""

    def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
        """Persist an event and return its SSE event_id."""

    def replay_since(
        self,
        tenant_id: str,
        request_id: str,
        last_event_id: str | None = None,
    ) -> list[EventEnvelope]:
        """Return events emitted strictly after ``last_event_id`` (or all buffered events when None)."""


class InMemoryStreamEventBus:
    """Process-local ring-buffered event bus keyed by (tenant_id, request_id)."""

    def __init__(self, *, max_events_per_request: int = 500) -> None:
        self._max_events = max(10, int(max_events_per_request or 500))
        self._lock = Lock()
        self._buffers: dict[tuple[str, str], deque[tuple[int, dict[str, Any]]]] = defaultdict(
            lambda: deque(maxlen=self._max_events)
        )
        self._counters: dict[tuple[str, str], int] = defaultdict(int)

    def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
        key = (str(tenant_id or "default"), str(request_id or ""))
        with self._lock:
            seq = self._counters[key] + 1
            self._counters[key] = seq
            self._buffers[key].append((seq, dict(payload)))
            return str(seq)

    def replay_since(
        self,
        tenant_id: str,
        request_id: str,
        last_event_id: str | None = None,
    ) -> list[EventEnvelope]:
        key = (str(tenant_id or "default"), str(request_id or ""))
        after_seq = 0
        if last_event_id is not None and str(last_event_id).strip():
            raw = str(last_event_id).strip().split("-", 1)[0]
            try:
                after_seq = int(raw)
            except ValueError:
                after_seq = 0

        with self._lock:
            items = list(self._buffers.get(key, ()))
        return [
            EventEnvelope(event_id=str(seq), payload=dict(payload))
            for seq, payload in items
            if seq > after_seq
        ]


class RedisStreamEventBus:
    """Redis Streams-backed SSE event bus supporting Last-Event-ID replay across gateway replicas."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        client: Any | None = None,
        max_events_per_request: int = 500,
    ) -> None:
        self._max_events = max(10, int(max_events_per_request or 500))
        if client is not None:
            self._client = client
        else:
            if _redis_mod is None:
                raise RuntimeError("redis package is not installed")
            self._client = _redis_mod.Redis.from_url(str(redis_url or ""), decode_responses=True)

    @staticmethod
    def _stream_key(tenant_id: str, request_id: str) -> str:
        return f"deepresearch:stream:{tenant_id or 'default'}:{request_id}"

    def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
        stream_key = self._stream_key(tenant_id, request_id)
        encoded = json.dumps(payload, ensure_ascii=False)
        entry_id = self._client.xadd(
            stream_key,
            {"payload": encoded},
            maxlen=self._max_events,
            approximate=True,
        )
        return entry_id.decode("utf-8") if isinstance(entry_id, bytes) else str(entry_id)

    def replay_since(
        self,
        tenant_id: str,
        request_id: str,
        last_event_id: str | None = None,
    ) -> list[EventEnvelope]:
        stream_key = self._stream_key(tenant_id, request_id)
        min_id = f"({str(last_event_id).strip()}" if last_event_id and str(last_event_id).strip() else "-"
        rows = self._client.xrange(stream_key, min=min_id, max="+")
        envelopes: list[EventEnvelope] = []
        for entry_id, fields in rows or ():
            eid = entry_id.decode("utf-8") if isinstance(entry_id, bytes) else str(entry_id)
            raw_payload = fields.get("payload") if isinstance(fields, dict) else None
            if isinstance(raw_payload, bytes):
                raw_payload = raw_payload.decode("utf-8")
            if not raw_payload:
                continue
            try:
                parsed = json.loads(raw_payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                envelopes.append(EventEnvelope(event_id=eid, payload=parsed))
        return envelopes


def build_stream_event_bus(config: Any) -> StreamEventBus:
    """Build RedisStreamEventBus when REDIS_URL & driver are ready, otherwise InMemoryStreamEventBus."""

    max_events = int(getattr(config, "event_bus_max_events_per_request", 500) or 500)
    redis_url = str(getattr(config, "redis_url", None) or "").strip()
    if redis_url and _redis_mod is not None:
        try:
            return RedisStreamEventBus(
                redis_url=redis_url,
                max_events_per_request=max_events,
            )
        except Exception as exc:
            logger.warning("Failed to initialize RedisStreamEventBus; falling back to in-memory bus: %s", exc)
    return InMemoryStreamEventBus(max_events_per_request=max_events)
