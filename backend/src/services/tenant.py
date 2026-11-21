"""Multi-tenant API key authentication and per-tenant rate/concurrency limiting."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any


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
            if hmac.compare_digest(provided, candidate):
                matched_tenant = str(tenant_id or default_tenant).strip() or default_tenant
        if matched_tenant is not None:
            return TenantContext(tenant_id=matched_tenant, authenticated=True)
        if auth_enabled:
            raise PermissionError("invalid API key")

    if auth_enabled:
        raise PermissionError("API key is required")

    return TenantContext(tenant_id=default_tenant, authenticated=False)


class TenantRateLimiter:
    """Thread-safe per-tenant sliding-window RPM and active-concurrency limiter."""

    def __init__(self, *, rpm_limit: int = 0, max_concurrent: int = 0) -> None:
        self._rpm_limit = max(0, int(rpm_limit or 0))
        self._max_concurrent = max(0, int(max_concurrent or 0))
        self._lock = Lock()
        self._timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._inflight: dict[str, int] = defaultdict(int)

    def try_acquire(
        self,
        tenant_id: str,
        *,
        rpm_limit: int | None = None,
        max_concurrent: int | None = None,
    ) -> tuple[bool, int, str]:
        """Attempt to reserve a rate/concurrency slot; returns (allowed, retry_after_sec, reason)."""

        effective_rpm = self._rpm_limit if rpm_limit is None else max(0, int(rpm_limit or 0))
        effective_conc = self._max_concurrent if max_concurrent is None else max(0, int(max_concurrent or 0))
        now = time.monotonic()

        with self._lock:
            if effective_conc > 0 and self._inflight[tenant_id] >= effective_conc:
                return False, 5, "max_concurrent_requests"

            if effective_rpm > 0:
                window = self._timestamps[tenant_id]
                cutoff = now - 60.0
                while window and window[0] < cutoff:
                    window.popleft()
                if len(window) >= effective_rpm:
                    retry_after = max(1, int(60.0 - (now - window[0])) + 1)
                    return False, retry_after, "rate_limit_rpm"
                window.append(now)

            self._inflight[tenant_id] += 1
            return True, 0, ""

    def release(self, tenant_id: str) -> None:
        with self._lock:
            current = self._inflight.get(tenant_id, 0)
            if current <= 1:
                self._inflight.pop(tenant_id, None)
            else:
                self._inflight[tenant_id] = current - 1
