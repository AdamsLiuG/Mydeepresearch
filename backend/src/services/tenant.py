"""Multi-tenant API key authentication and per-tenant rate/concurrency limiting."""

from __future__ import annotations

import hmac
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import ceil
from threading import Lock
from typing import Any
from uuid import uuid4

from services.memory_scope import MemoryPolicy, MemoryScope, scope_is_current

logger = logging.getLogger(__name__)

_ACQUIRE_REDIS = """
local rpm_limit = tonumber(ARGV[1])
local concurrent_limit = tonumber(ARGV[2])
local lease = ARGV[3]
local lease_ms = tonumber(ARGV[4])
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
if concurrent_limit > 0 then
    redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
    if redis.call('ZSCORE', KEYS[2], lease) then
        return {1, 0}
    end
    if redis.call('ZCARD', KEYS[2]) >= concurrent_limit then
        return {0, 5}
    end
end
if rpm_limit > 0 then
    redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - 60000)
    if redis.call('ZCARD', KEYS[1]) >= rpm_limit then
        local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
        return {2, math.max(1, math.ceil((60000 - now + tonumber(oldest[2])) / 1000))}
    end
    redis.call('ZADD', KEYS[1], now, lease)
    redis.call('PEXPIRE', KEYS[1], 60000)
end
if concurrent_limit > 0 then
    redis.call('ZADD', KEYS[2], now + lease_ms, lease)
    redis.call('PEXPIRE', KEYS[2], lease_ms)
end
return {1, 0}
"""
_LEASE_MILLISECONDS = 3_600_000


@dataclass(frozen=True)
class TenantContext:
    """Resolved tenant identity attached to an incoming request."""

    tenant_id: str
    authenticated: bool


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Extract an API key from X-API-Key or Authorization: Bearer headers."""

    lowered = {str(k).strip().lower(): str(v).strip() for k, v in (headers or {}).items()}
    direct = lowered.get("x-api-key")
    if direct:
        return direct
    auth_header = lowered.get("authorization")
    if auth_header:
        parts = auth_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    return None


def authenticate_tenant(api_key: str | None, config: Any) -> TenantContext:
    """Authenticate the caller against configured tenant API keys using constant-time comparison."""

    default_tenant = str(getattr(config, "default_tenant_id", "default") or "default").strip() or "default"
    key_map = dict(getattr(config, "tenant_api_keys", {}) or {})
    auth_enabled = bool(getattr(config, "auth_enabled", False))

    provided = str(api_key or "").strip()
    if provided and key_map:
        matched_tenant: str | None = None
        for configured_key, tenant_id in key_map.items():
            candidate = str(configured_key or "").strip()
            if not candidate:
                continue
            if hmac.compare_digest(provided.encode("utf-8"), candidate.encode("utf-8")):
                matched_tenant = str(tenant_id or default_tenant).strip() or default_tenant
        if matched_tenant is not None:
            return TenantContext(tenant_id=matched_tenant, authenticated=True)
        if auth_enabled:
            raise PermissionError("invalid API key")

    if auth_enabled:
        raise PermissionError("API key is required")

    return TenantContext(tenant_id=default_tenant, authenticated=False)


class TenantRateLimiter:
    """Thread-safe per-tenant sliding-window RPM and active-concurrency limiter with optional Redis."""

    def __init__(
        self,
        *,
        rpm_limit: int = 0,
        max_concurrent: int = 0,
        redis_client: Any | None = None,
        key_prefix: str = "deepresearch:ratelimit",
    ) -> None:
        self._rpm_limit = max(0, int(rpm_limit or 0))
        self._max_concurrent = max(0, int(max_concurrent or 0))
        self._redis = redis_client
        self._key_prefix = str(key_prefix or "deepresearch:ratelimit").strip()
        self._lock = Lock()
        self._timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._inflight: dict[str, dict[str, bool]] = defaultdict(dict)
        self._anonymous_leases: dict[str, deque[str]] = defaultdict(deque)

    def _redis_keys(self, tenant_id: str) -> tuple[str, str]:
        tenant_key = tenant_id.encode("utf-8").hex()
        tenant_prefix = f"{self._key_prefix}:{{{tenant_key}}}"
        return f"{tenant_prefix}:rpm", f"{tenant_prefix}:leases"

    def _try_acquire_redis(
        self,
        tenant_id: str,
        *,
        effective_rpm: int,
        effective_conc: int,
        lease_id: str,
    ) -> tuple[bool, int, str]:
        code, retry_after = self._redis.eval(
            _ACQUIRE_REDIS, 2, *self._redis_keys(tenant_id),
            effective_rpm, effective_conc, lease_id, _LEASE_MILLISECONDS,
        )
        if code == 1:
            return True, 0, ""
        return False, int(retry_after), "max_concurrent_requests" if code == 0 else "rate_limit_rpm"

    def try_acquire(
        self,
        tenant_id: str,
        *,
        rpm_limit: int | None = None,
        max_concurrent: int | None = None,
        lease_id: str | None = None,
    ) -> tuple[bool, int, str]:
        """Attempt to reserve a rate/concurrency slot; returns (allowed, retry_after_sec, reason)."""

        effective_rpm = self._rpm_limit if rpm_limit is None else max(0, int(rpm_limit or 0))
        effective_conc = self._max_concurrent if max_concurrent is None else max(0, int(max_concurrent or 0))
        if effective_rpm == 0 and effective_conc == 0:
            return True, 0, ""

        lease = lease_id if lease_id is not None else uuid4().hex
        with self._lock:
            active = self._inflight.get(tenant_id, {})
            if lease in active:
                return True, 0, ""
            if effective_conc > 0 and len(active) >= effective_conc:
                return False, 5, "max_concurrent_requests"

            distributed = False
            if self._redis is not None:
                local_count = sum(1 for shared in active.values() if not shared)
                try:
                    result = self._try_acquire_redis(
                        tenant_id,
                        effective_rpm=effective_rpm,
                        effective_conc=max(0, effective_conc - local_count),
                        lease_id=lease,
                    )
                    if not result[0]:
                        return result
                    distributed = True
                except Exception as exc:
                    logger.warning("Redis rate limiter degraded to in-process lock (%s)", type(exc).__name__)

            if effective_rpm > 0:
                now = time.monotonic()
                window = self._timestamps[tenant_id]
                while window and window[0] <= now - 60.0:
                    window.popleft()
                if not distributed and len(window) >= effective_rpm:
                    retry_after = max(1, ceil(60.0 - (now - window[0])))
                    return False, retry_after, "rate_limit_rpm"
                window.append(now)

            if effective_conc > 0:
                self._inflight[tenant_id][lease] = distributed
                if lease_id is None:
                    self._anonymous_leases[tenant_id].append(lease)
            return True, 0, ""

    def release(self, tenant_id: str, *, lease_id: str | None = None) -> None:
        with self._lock:
            if lease_id is None:
                anonymous = self._anonymous_leases.get(tenant_id)
                if not anonymous:
                    return
                lease_id = anonymous.popleft()
                if not anonymous:
                    self._anonymous_leases.pop(tenant_id)
            active = self._inflight.get(tenant_id)
            if active is None:
                return
            distributed = active.pop(lease_id, False)
            if not active:
                self._inflight.pop(tenant_id)
        if distributed:
            try:
                self._redis.zrem(self._redis_keys(tenant_id)[1], lease_id)
            except Exception as exc:
                logger.warning("Redis rate limiter release failed (%s)", type(exc).__name__)


def authenticated_memory_scope(
    tenant: TenantContext, request_id: str, *, policy_version: int = 1,
) -> MemoryScope | None:
    """Bind scope to the tenant authentication result at the gateway."""
    if not tenant.authenticated:
        return None
    try:
        return MemoryScope.trusted(tenant.tenant_id, request_id, policy_version=policy_version)
    except ValueError:
        return None


def verified_worker_memory_scope(
    tenant_id: str, request_id: str, owner_epoch: int, *,
    verify_owner: Callable[[MemoryScope], bool], policy_version: int = 1,
) -> MemoryScope | None:
    """Recheck current durable ownership through the trusted runner adapter."""
    try:
        scope = MemoryScope.trusted(tenant_id, request_id, owner_epoch=owner_epoch, policy_version=policy_version)
        return scope if scope.owner_epoch is not None and verify_owner(scope) is True else None
    except Exception:
        return None


def restored_memory_scope(
    scope: MemoryScope | None, frozen: Any, *, policy: MemoryPolicy,
) -> MemoryScope | None:
    """Accept a frozen partition only when its trusted request owner and live policy agree."""
    if not scope_is_current(policy, scope) or not isinstance(frozen, dict):
        return None
    expected = scope.frozen_partition()
    return scope if all(frozen.get(key) == value for key, value in expected.items()) else None
