"""Stream event bus with in-memory and optional Redis Streams replay backends."""

from __future__ import annotations

import json
import logging
import re
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
_STREAM_ID = re.compile(r"[0-9]{1,20}(?:-[0-9]{1,20})?")


def event_id_parts(event_id: str) -> tuple[int, int]:
    if not _STREAM_ID.fullmatch(event_id):
        raise ValueError("Invalid Last-Event-ID")
    major, _, minor = event_id.partition("-")
    parts = (int(major), int(minor or 0))
    if any(part >= 2**64 for part in parts):
        raise ValueError("Invalid Last-Event-ID")
    return parts


def _is_terminal(payload: dict[str, Any]) -> bool:
    return str(payload.get("type") or "").strip().lower() in TERMINAL_EVENT_TYPES


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


def _report_event_key(payload: dict[str, Any]) -> str:
    key = payload.get("event_key")
    return key if payload.get("schema_version") == 1 and isinstance(key, str) and key.startswith("report-") else ""


class InMemoryStreamEventBus:
    """Process-local ring-buffered event bus keyed by (tenant_id, request_id)."""

    def __init__(self, *, max_events_per_request: int = 2000) -> None:
        self._max_events = max(10, int(max_events_per_request or 2000))
        self._condition = Condition()
        self._buffers: dict[tuple[str, str], deque[tuple[int, dict[str, Any]]]] = defaultdict(
            lambda: deque(maxlen=self._max_events)
        )
        self._event_keys: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
        self._counters: dict[tuple[str, str], int] = defaultdict(int)
        self._terminal_keys: set[tuple[str, str]] = set()

    def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
        key = (str(tenant_id or "default"), str(request_id or ""))
        with self._condition:
            buffer = self._buffers[key]
            event_key = _report_event_key(payload)
            if event_key and (previous := self._event_keys[key].get(event_key)) is not None:
                return str(previous)
            if len(buffer) == self._max_events:
                expired = _report_event_key(buffer[0][1])
                if expired:
                    self._event_keys[key].pop(expired, None)
            seq = self._counters[key] + 1
            if event_key:
                self._event_keys[key][event_key] = seq
            self._counters[key] = seq
            stored = dict(payload)
            buffer.append((seq, stored))
            if _is_terminal(stored):
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

    def has_events(self, tenant_id: str, request_id: str) -> bool:
        with self._condition:
            return bool(self._buffers.get((tenant_id, request_id)))

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
        with self._condition:
            buffer = self._buffers.get(key)
            has_gap = bool(cursor and (
                not buffer or not buffer[0][0] <= event_id_parts(cursor)[0] <= buffer[-1][0]
            ))
        if has_gap:
            cursor = None
            yield EventEnvelope(event_id="", payload={"type": "replay_gap"})

        while True:
            batch = self.replay_since(tenant_id, request_id, last_event_id=cursor)
            if batch:
                idle_cycles = 0
                for env in batch:
                    cursor = env.event_id
                    yield env
                    if _is_terminal(env.payload):
                        return

            terminal = is_terminal_fn is not None and is_terminal_fn()

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
                if terminal or key in self._terminal_keys:
                    return
                notified = self._condition.wait(timeout=wait_sec)

            if not notified:
                idle_cycles += 1
                if max_idle_cycles is not None and idle_cycles > max(0, int(max_idle_cycles)):
                    return
                yield EventEnvelope(event_id="", payload={"type": "__keepalive__"})


# Keep only keys with a replayable entry. The sorted set is scored by stream time
# and the hash is trimmed with the stream in the same transaction.
_REPORT_APPEND = """
local previous = redis.call('HGET', KEYS[2], ARGV[1])
if previous and #redis.call('XRANGE', KEYS[1], previous, previous) > 0 then
    return previous
end
local id = redis.call('XADD', KEYS[1], 'MAXLEN', ARGV[3], '*', 'payload', ARGV[2])
redis.call('HSET', KEYS[2], ARGV[1], id)
redis.call('ZADD', KEYS[3], 0, id .. '|' .. ARGV[1])
-- Report history is bounded. Remove stale keys even when the stream was trimmed by a legacy frame.
for _, entry in ipairs(redis.call('ZRANGE', KEYS[3], 0, -1)) do
    local delimiter = string.find(entry, '|', 1, true)
    local cursor = string.sub(entry, 1, delimiter - 1)
    if #redis.call('XRANGE', KEYS[1], cursor, cursor) == 0 then
        local key = string.sub(entry, delimiter + 1)
        if redis.call('HGET', KEYS[2], key) == cursor then
            redis.call('HDEL', KEYS[2], key)
        end
        redis.call('ZREM', KEYS[3], entry)
    end
end
if ARGV[4] == '1' then
    for _, key in ipairs(KEYS) do redis.call('EXPIRE', key, ARGV[5]) end
end
return id
"""


class RedisStreamEventBus:
    """Redis Streams-backed SSE event bus supporting Last-Event-ID replay across gateway replicas."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        client: Any | None = None,
        max_events_per_request: int = 2000,
        stream_ttl_seconds: int = 86400,
    ) -> None:
        self._max_events = max(10, int(max_events_per_request or 2000))
        self._stream_ttl_seconds = max(60, int(stream_ttl_seconds or 86400))
        if client is not None:
            self._client = client
        else:
            if _redis_mod is None:
                raise RuntimeError("redis package is not installed")
            self._client = _redis_mod.Redis.from_url(
                str(redis_url or ""), decode_responses=True, socket_connect_timeout=2, socket_timeout=20,
            )

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
        if raw_payload is None and isinstance(fields, dict):
            raw_payload = fields.get(b"payload")
        if not raw_payload:
            return None
        try:
            parsed = json.loads(raw_payload)
        except (ValueError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        return EventEnvelope(event_id=eid, payload=parsed)

    def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
        stream_key = self._stream_key(tenant_id, request_id)
        encoded = json.dumps(payload, ensure_ascii=False)
        terminal = _is_terminal(payload)
        event_key = _report_event_key(payload)
        if event_key:
            entry_id = self._client.eval(
                _REPORT_APPEND, 3, stream_key, f"{stream_key}:report-keys", f"{stream_key}:report-cursors",
                event_key, encoded, self._max_events, "1" if terminal else "0", self._stream_ttl_seconds,
            )
            return entry_id.decode("utf-8") if isinstance(entry_id, bytes) else str(entry_id)
        pipeline = self._client.pipeline() if terminal and hasattr(self._client, "pipeline") else None
        entry_id = (pipeline if pipeline is not None else self._client).xadd(
            stream_key,
            {"payload": encoded},
            maxlen=self._max_events,
            approximate=True,
        )
        if pipeline is not None:
            pipeline.expire(stream_key, self._stream_ttl_seconds)
            entry_id = pipeline.execute()[0]
        elif terminal and hasattr(self._client, "expire"):
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

    def has_events(self, tenant_id: str, request_id: str) -> bool:
        return bool(self._client.exists(self._stream_key(tenant_id, request_id)))

    def close(self) -> None:
        self._client.close()

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

        if cursor and hasattr(self._client, "xrevrange"):
            oldest = self._client.xrange(stream_key, count=1)
            newest = self._client.xrevrange(stream_key, count=1)
            def parts(row: Any) -> tuple[int, int]:
                entry_id = row[0][0]
                return event_id_parts(entry_id.decode() if isinstance(entry_id, bytes) else entry_id)

            if not oldest or not newest or not parts(oldest) <= event_id_parts(cursor) <= parts(newest):
                cursor = None
                yield EventEnvelope(event_id="", payload={"type": "replay_gap"})
            elif event_id_parts(cursor) == parts(newest):
                last = self._decode_entry(*newest[0])
                if last is not None and _is_terminal(last.payload):
                    return
        initial = self.replay_since(tenant_id, request_id, last_event_id=cursor)
        for env in initial:
            cursor = env.event_id
            yield env
            if _is_terminal(env.payload):
                return

        xread_cursor = cursor or "0-0"
        idle_cycles = 0
        while True:
            terminal = is_terminal_fn is not None and is_terminal_fn()
            if hasattr(self._client, "xread"):
                kwargs = {"block": max(50, int(block_ms or 5000))} if not terminal else {}
                if hasattr(self._client, "xrevrange"):
                    kwargs["count"] = 64
                rows_by_stream = self._client.xread({stream_key: xread_cursor}, **kwargs)
            else:  # pragma: no cover - fallback when fake client lacks xread
                time.sleep(max(0.01, poll_interval))
                rows_by_stream = []

            if rows_by_stream:
                idle_cycles = 0
                for _, entries in rows_by_stream:
                    for entry_id, fields in entries:
                        xread_cursor = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                        env = self._decode_entry(entry_id, fields)
                        if env is None:
                            continue
                        yield env
                        if _is_terminal(env.payload):
                            return
                continue

            if terminal:
                return
            idle_cycles += 1
            if max_idle_cycles is not None and idle_cycles > max(0, int(max_idle_cycles)):
                return
            yield EventEnvelope(event_id="", payload={"type": "__keepalive__"})


def build_stream_event_bus(config: Any) -> StreamEventBus:
    """Build RedisStreamEventBus when REDIS_URL & driver are ready, otherwise InMemoryStreamEventBus."""

    max_events = int(getattr(config, "event_bus_max_events_per_request", 2000) or 2000)
    stream_ttl = int(getattr(config, "event_bus_stream_ttl_seconds", 86400) or 86400)
    redis_url = str(getattr(config, "redis_url", None) or "").strip()
    if redis_url and _redis_mod is not None:
        try:
            bus = RedisStreamEventBus(
                redis_url=redis_url,
                max_events_per_request=max_events,
                stream_ttl_seconds=stream_ttl,
            )
            try:
                bus._client.ping()
            except Exception:
                bus.close()
                raise
            return bus
        except Exception:
            logger.warning("Failed to initialize RedisStreamEventBus; falling back to in-memory bus")
    return InMemoryStreamEventBus(max_events_per_request=max_events)
