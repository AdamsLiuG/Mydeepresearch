"""Stream event bus with in-memory and optional Redis Streams replay backends."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from threading import Condition
from typing import Any, Protocol

try:  # pragma: no cover - optional dependency
    import redis as _redis_mod
except Exception:  # pragma: no cover - optional dependency
    _redis_mod = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
TERMINAL_EVENT_TYPES = frozenset({"done", "error", "cancelled"})


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

    def tail_since(
        self,
        tenant_id: str,
        request_id: str,
        last_event_id: str | None = None,
        *,
        is_terminal_fn: Callable[[], bool] | None = None,
        block_ms: int = 5000,
        poll_interval: float = 0.05,
        max_idle_cycles: int | None = None,
    ) -> Iterator[EventEnvelope]:
        """Replay buffered events and block-tail new live events until terminal."""


class InMemoryStreamEventBus:
    """Process-local ring-buffered event bus keyed by (tenant_id, request_id)."""

    def __init__(self, *, max_events_per_request: int = 500) -> None:
        self._max_events = max(10, int(max_events_per_request or 500))
        self._condition = Condition()
        self._buffers: dict[tuple[str, str], deque[tuple[int, dict[str, Any]]]] = defaultdict(
            lambda: deque(maxlen=self._max_events)
        )
        self._counters: dict[tuple[str, str], int] = defaultdict(int)
        self._terminal_keys: set[tuple[str, str]] = set()

    def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
        key = (str(tenant_id or "default"), str(request_id or ""))
        with self._condition:
            seq = self._counters[key] + 1
            self._counters[key] = seq
            stored = dict(payload)
            self._buffers[key].append((seq, stored))
            if str(stored.get("type") or "").strip().lower() in TERMINAL_EVENT_TYPES:
                self._terminal_keys.add(key)
            self._condition.notify_all()
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

        with self._condition:
            items = list(self._buffers.get(key, ()))
        return [
            EventEnvelope(event_id=str(seq), payload=dict(payload))
            for seq, payload in items
            if seq > after_seq
        ]

    def tail_since(
        self,
        tenant_id: str,
        request_id: str,
        last_event_id: str | None = None,
        *,
        is_terminal_fn: Callable[[], bool] | None = None,
        block_ms: int = 5000,
        poll_interval: float = 0.05,
        max_idle_cycles: int | None = None,
    ) -> Iterator[EventEnvelope]:
        del poll_interval
        key = (str(tenant_id or "default"), str(request_id or ""))
        cursor = last_event_id
        wait_sec = max(0.05, float(block_ms or 5000) / 1000.0)
        idle_cycles = 0

        while True:
            batch = self.replay_since(tenant_id, request_id, last_event_id=cursor)
            saw_terminal = False
            if batch:
                idle_cycles = 0
                for env in batch:
                    cursor = env.event_id
                    yield env
                    if str(env.payload.get("type") or "").strip().lower() in TERMINAL_EVENT_TYPES:
                        saw_terminal = True
                if saw_terminal:
                    return

            if is_terminal_fn is not None and is_terminal_fn():
                return

            with self._condition:
                # Re-check under lock so no event published between replay and wait is missed
                latest_seq = self._counters.get(key, 0)
                cur_seq = 0
                if cursor and str(cursor).strip():
                    try:
                        cur_seq = int(str(cursor).strip().split("-", 1)[0])
                    except ValueError:
                        cur_seq = 0
                if latest_seq > cur_seq:
                    continue
                if key in self._terminal_keys:
                    return
                notified = self._condition.wait(timeout=wait_sec)

            if not notified:
                if is_terminal_fn is not None and is_terminal_fn():
                    return
                idle_cycles += 1
                if max_idle_cycles is not None and idle_cycles > max(0, int(max_idle_cycles)):
                    return
                yield EventEnvelope(event_id="", payload={"type": "__keepalive__"})


class RedisStreamEventBus:
    """Redis Streams-backed SSE event bus supporting Last-Event-ID replay across gateway replicas."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        client: Any | None = None,
        max_events_per_request: int = 500,
        stream_ttl_seconds: int = 86400,
    ) -> None:
        self._max_events = max(10, int(max_events_per_request or 500))
        self._stream_ttl_seconds = max(60, int(stream_ttl_seconds or 86400))
        if client is not None:
            self._client = client
        else:
            if _redis_mod is None:
                raise RuntimeError("redis package is not installed")
            self._client = _redis_mod.Redis.from_url(str(redis_url or ""), decode_responses=True)

    @staticmethod
    def _stream_key(tenant_id: str, request_id: str) -> str:
        tenant = tenant_id or "default"
        if ":" in tenant or ":" in request_id:
            # Use a separate namespace so ambiguous legacy keys are never read.
            pair = json.dumps((tenant, request_id), ensure_ascii=False, separators=(",", ":"))
            return f"deepresearch:stream-v2:{pair}"
        return f"deepresearch:stream:{tenant}:{request_id}"

    @staticmethod
    def _decode_entry(entry_id: Any, fields: Any) -> EventEnvelope | None:
        eid = entry_id.decode("utf-8") if isinstance(entry_id, bytes) else str(entry_id)
        raw_payload = fields.get("payload") if isinstance(fields, dict) else None
        if isinstance(raw_payload, bytes):
            raw_payload = raw_payload.decode("utf-8")
        if not raw_payload:
            return None
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return EventEnvelope(event_id=eid, payload=parsed)

    def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
        stream_key = self._stream_key(tenant_id, request_id)
        encoded = json.dumps(payload, ensure_ascii=False)
        entry_id = self._client.xadd(
            stream_key,
            {"payload": encoded},
            maxlen=self._max_events,
            approximate=True,
        )
        if str(payload.get("type") or "").strip().lower() in TERMINAL_EVENT_TYPES and hasattr(
            self._client, "expire"
        ):
            self._client.expire(stream_key, self._stream_ttl_seconds)
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
            env = self._decode_entry(entry_id, fields)
            if env is not None:
                envelopes.append(env)
        return envelopes

    def tail_since(
        self,
        tenant_id: str,
        request_id: str,
        last_event_id: str | None = None,
        *,
        is_terminal_fn: Callable[[], bool] | None = None,
        block_ms: int = 5000,
        poll_interval: float = 0.05,
        max_idle_cycles: int | None = None,
    ) -> Iterator[EventEnvelope]:
        stream_key = self._stream_key(tenant_id, request_id)
        cursor = str(last_event_id).strip() if last_event_id and str(last_event_id).strip() else None

        initial = self.replay_since(tenant_id, request_id, last_event_id=cursor)
        for env in initial:
            cursor = env.event_id
            yield env
            if str(env.payload.get("type") or "").strip().lower() in TERMINAL_EVENT_TYPES:
                return

        if is_terminal_fn is not None and is_terminal_fn():
            return

        xread_cursor = cursor or "0-0"
        idle_cycles = 0
        while True:
            if hasattr(self._client, "xread"):
                rows_by_stream = self._client.xread({stream_key: xread_cursor}, block=max(50, int(block_ms or 5000)))
            else:  # pragma: no cover - fallback when fake client lacks xread
                time.sleep(max(0.01, poll_interval))
                rows_by_stream = []

            if rows_by_stream:
                idle_cycles = 0
                for _, entries in rows_by_stream:
                    for entry_id, fields in entries:
                        env = self._decode_entry(entry_id, fields)
                        if env is None:
                            continue
                        xread_cursor = env.event_id
                        yield env
                        if str(env.payload.get("type") or "").strip().lower() in TERMINAL_EVENT_TYPES:
                            return
                continue

            if is_terminal_fn is not None and is_terminal_fn():
                return
            idle_cycles += 1
            if max_idle_cycles is not None and idle_cycles > max(0, int(max_idle_cycles)):
                return
            yield EventEnvelope(event_id="", payload={"type": "__keepalive__"})


def build_stream_event_bus(config: Any) -> StreamEventBus:
    """Build RedisStreamEventBus when REDIS_URL & driver are ready, otherwise InMemoryStreamEventBus."""

    max_events = int(getattr(config, "event_bus_max_events_per_request", 500) or 500)
    stream_ttl = int(getattr(config, "event_bus_stream_ttl_seconds", 86400) or 86400)
    redis_url = str(getattr(config, "redis_url", None) or "").strip()
    if redis_url and _redis_mod is not None:
        try:
            return RedisStreamEventBus(
                redis_url=redis_url,
                max_events_per_request=max_events,
                stream_ttl_seconds=stream_ttl,
            )
        except Exception as exc:
            logger.warning("Failed to initialize RedisStreamEventBus; falling back to in-memory bus: %s", exc)
    return InMemoryStreamEventBus(max_events_per_request=max_events)
