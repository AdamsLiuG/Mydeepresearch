"""Request-scoped budget profiles and atomic operation leases."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from threading import RLock
from time import monotonic
from typing import Any, Callable
from uuid import UUID, uuid4

from metrics import metrics_registry
from models import (
    RESOURCE_NAMES,
    BudgetDenial,
    BudgetLease,
    BudgetSettlement,
    BudgetSnapshot,
    RequestBudget,
)

RESOURCE_SET = frozenset(RESOURCE_NAMES)
EXTERNAL_RESOURCES = frozenset(
    {"search_calls", "web_fetches", "dynamic_fetches", "sandbox_calls", "llm_calls"}
)
FINALIZATION_STAGES = frozenset({"review", "report", "required_repair"})


@dataclass(frozen=True)
class BudgetProfile:
    depth: str
    duration_seconds: int
    finalization_reserve_ms: int
    limits: dict[str, int]
    finalization_reserve: dict[str, int]
    wave_one_limit: int
    deepen_limit: int
    version: int = 1


_PROFILE_VALUES: dict[str, BudgetProfile] = {
    "quick": BudgetProfile(
        depth="quick",
        duration_seconds=90,
        finalization_reserve_ms=20_000,
        limits={
            "tasks": 4,
            "search_calls": 8,
            "web_fetches": 6,
            "dynamic_fetches": 1,
            "sandbox_calls": 1,
            "llm_calls": 12,
            "input_tokens": 32_000,
            "output_tokens": 6_000,
            "cost_microusd": 150_000,
        },
        finalization_reserve={
            "llm_calls": 3,
            "input_tokens": 8_000,
            "output_tokens": 2_000,
            "cost_microusd": 40_000,
        },
        wave_one_limit=1,
        deepen_limit=0,
    ),
    "standard": BudgetProfile(
        depth="standard",
        duration_seconds=240,
        finalization_reserve_ms=45_000,
        limits={
            "tasks": 8,
            "search_calls": 22,
            "web_fetches": 16,
            "dynamic_fetches": 3,
            "sandbox_calls": 3,
            "llm_calls": 28,
            "input_tokens": 90_000,
            "output_tokens": 16_000,
            "cost_microusd": 600_000,
        },
        finalization_reserve={
            "llm_calls": 5,
            "input_tokens": 18_000,
            "output_tokens": 4_000,
            "cost_microusd": 150_000,
        },
        wave_one_limit=2,
        deepen_limit=2,
    ),
    "deep": BudgetProfile(
        depth="deep",
        duration_seconds=600,
        finalization_reserve_ms=90_000,
        limits={
            "tasks": 14,
            "search_calls": 48,
            "web_fetches": 36,
            "dynamic_fetches": 6,
            "sandbox_calls": 6,
            "llm_calls": 56,
            "input_tokens": 200_000,
            "output_tokens": 36_000,
            "cost_microusd": 2_000_000,
        },
        finalization_reserve={
            "llm_calls": 8,
            "input_tokens": 32_000,
            "output_tokens": 8_000,
            "cost_microusd": 400_000,
        },
        wave_one_limit=3,
        deepen_limit=4,
    ),
}


def profile_for_depth(depth: str) -> BudgetProfile:
    """Return a defensive profile copy for one of the supported research depths."""
    normalized = str(depth).strip().lower()
    try:
        profile = _PROFILE_VALUES[normalized]
    except KeyError as exc:
        raise ValueError("depth must be quick, standard, or deep") from exc
    return BudgetProfile(
        depth=profile.depth,
        duration_seconds=profile.duration_seconds,
        finalization_reserve_ms=profile.finalization_reserve_ms,
        limits=dict(profile.limits),
        finalization_reserve=dict(profile.finalization_reserve),
        wave_one_limit=profile.wave_one_limit,
        deepen_limit=profile.deepen_limit,
        version=profile.version,
    )


def _utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        normalized = str(value).strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("deadline must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _resource_limits(values: dict[str, int] | None, *, default: dict[str, int]) -> dict[str, int]:
    if values is None:
        return dict(default)
    if not isinstance(values, dict):
        raise ValueError("resource limits must be a dictionary")
    unknown = set(values).difference(RESOURCE_SET)
    if unknown:
        raise ValueError(f"unknown resources: {', '.join(sorted(unknown))}")
    resolved = dict(default)
    for resource, amount in values.items():
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f"limit for {resource} must be a non-negative integer")
        resolved[resource] = amount
    return resolved


def build_request_budget(
    depth: str = "standard",
    *,
    now: datetime | None = None,
    deadline_at: str | None = None,
    duration_seconds: int | None = None,
    finalization_reserve_ms: int | None = None,
    service_limits: dict[str, int] | None = None,
    tenant_limits: dict[str, int] | None = None,
    limits: dict[str, int] | None = None,
    finalization_reserve: dict[str, int] | None = None,
) -> RequestBudget:
    """Build a frozen request budget after profile, service, and tenant clamping."""
    profile = profile_for_depth(depth)
    current = _utc_datetime(now or datetime.now(timezone.utc))
    duration = profile.duration_seconds if duration_seconds is None else duration_seconds
    reserve_ms = profile.finalization_reserve_ms if finalization_reserve_ms is None else finalization_reserve_ms
    if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
        raise ValueError("duration_seconds must be a positive integer")
    if isinstance(reserve_ms, bool) or not isinstance(reserve_ms, int) or reserve_ms < 0:
        raise ValueError("finalization_reserve_ms must be a non-negative integer")
    reserve_ms = min(reserve_ms, duration * 1000)

    profile_limits = _resource_limits(limits, default=profile.limits)
    service_caps = _resource_limits(service_limits, default=profile_limits)
    tenant_caps = _resource_limits(tenant_limits, default=profile_limits)
    effective_limits = {
        name: min(profile_limits[name], service_caps[name], tenant_caps[name])
        for name in RESOURCE_NAMES
    }
    reserve_defaults = dict(profile.finalization_reserve)
    if finalization_reserve is not None:
        reserve_defaults = _resource_limits(finalization_reserve, default=reserve_defaults)
    effective_reserve = {
        name: min(reserve_defaults.get(name, 0), effective_limits[name])
        for name in RESOURCE_NAMES
    }
    deadline = _utc_datetime(deadline_at) if deadline_at is not None else current + timedelta(seconds=duration)
    if deadline <= current:
        raise ValueError("deadline_at must be in the future")
    reserve_ms = min(reserve_ms, max(0, int((deadline - current).total_seconds() * 1000)))
    return RequestBudget(
        depth=profile.depth,
        limits=effective_limits,
        finalization_reserve=effective_reserve,
        deadline_at=_format_utc(deadline),
        finalization_reserve_ms=reserve_ms,
        profile_version=profile.version,
    )


def build_request_budget_for_config(
    config: Any,
    tenant_id: str,
    *,
    depth: str | None = None,
    now: datetime | None = None,
    deadline_at: str | None = None,
) -> RequestBudget:
    """Apply configuration service caps and the selected tenant's caps to a profile."""
    profile = profile_for_depth(depth or config.v4_default_research_depth)
    service_limits = dict(config.v4_research_budget_limits)
    service_limits["tasks"] = min(
        service_limits.get("tasks", profile.limits["tasks"]),
        int(config.resolved_max_total_tasks()),
    )
    tenant_limits = config.v4_tenant_budget_limits.get(str(tenant_id), {})
    return build_request_budget(
        depth or config.v4_default_research_depth,
        now=now,
        deadline_at=deadline_at,
        service_limits=service_limits,
        tenant_limits=tenant_limits,
    )


def _valid_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _demand_vector(demand: dict[str, int], *, sparse: bool = False) -> dict[str, int]:
    if not isinstance(demand, dict):
        raise ValueError("demand must be a resource dictionary")
    unknown = set(demand).difference(RESOURCE_SET)
    if unknown:
        raise ValueError(f"unknown resources: {', '.join(sorted(unknown))}")
    resolved = {} if sparse else {name: 0 for name in RESOURCE_NAMES}
    for name, amount in demand.items():
        resolved[name] = _valid_nonnegative_int(amount, name=name)
    return resolved


def estimate_llm_cost_microusd(
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, Any] | None,
) -> int | None:
    """Return an upward-rounded cost bound using the highest applicable input rate."""
    _valid_nonnegative_int(input_tokens, name="input_tokens")
    _valid_nonnegative_int(output_tokens, name="output_tokens")
    if not isinstance(pricing, dict):
        return None
    try:
        prompt_rate = Decimal(str(pricing["prompt_per_1k_tokens"]))
        completion_rate = Decimal(str(pricing["completion_per_1k_tokens"]))
        input_rate = max(
            prompt_rate,
            Decimal(str(pricing.get("cached_prompt_per_1k_tokens", prompt_rate))),
            Decimal(str(pricing.get("cache_creation_per_1k_tokens", prompt_rate))),
        )
        if any(not rate.is_finite() or rate < 0 for rate in (input_rate, completion_rate)):
            return None
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return None
    cost = (
        Decimal(input_tokens) * input_rate / Decimal(1000)
        + Decimal(output_tokens) * completion_rate / Decimal(1000)
    ) * Decimal(1_000_000)
    return int(cost.to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class _IssuedBudgetLease(BudgetLease):
    lease_id: UUID = field(default_factory=uuid4, repr=False)


@dataclass
class _LeaseRecord:
    lease: _IssuedBudgetLease
    signature: tuple[int, ...]
    general_reserved: dict[str, int]
    finalization_reserved: dict[str, int]
    state: str = "reserved"
    settlement: BudgetSettlement | None = None


class BudgetGovernor:
    """Thread-safe in-memory governor for one tenant and request scope."""

    def __init__(
        self,
        budget: RequestBudget,
        *,
        tenant_id: str = "default",
        request_id: str = "",
        monotonic_clock: Callable[[], float] = monotonic,
        utc_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.budget = RequestBudget(**budget.to_dict())
        self.tenant_id = str(tenant_id)
        self.request_id = str(request_id)
        self._monotonic = monotonic_clock
        self._utc = utc_clock
        self._lock = RLock()
        self._records: dict[str, _LeaseRecord] = {}
        self._spent_general = {name: 0 for name in RESOURCE_NAMES}
        self._spent_finalization = {name: 0 for name in RESOURCE_NAMES}
        self._overrun = {name: 0 for name in RESOURCE_NAMES}
        self._last_denial_reason: str | None = None
        self._external_attempt_count = 0
        now_utc = _utc_datetime(self._utc())
        deadline = _utc_datetime(budget.deadline_at)
        self._monotonic_deadline = self._monotonic() + max(0.0, (deadline - now_utc).total_seconds())
        self._utc_deadline = deadline

    @property
    def external_attempt_count(self) -> int:
        with self._lock:
            return self._external_attempt_count

    @property
    def identity(self) -> tuple[str, str]:
        return self.tenant_id, self.request_id

    def _remaining_seconds(self) -> float:
        utc_remaining = (self._utc_deadline - _utc_datetime(self._utc())).total_seconds()
        monotonic_remaining = self._monotonic_deadline - self._monotonic()
        return max(0.0, min(utc_remaining, monotonic_remaining))

    def remaining_timeout(
        self,
        maximum_seconds: float | None = None,
        *,
        stage: str = "research",
        work_class: str = "optional",
    ) -> float:
        with self._lock:
            available = self._remaining_seconds()
            if stage not in FINALIZATION_STAGES:
                available = min(
                    available,
                    max(0.0, available - self.budget.finalization_reserve_ms / 1000),
                )
            if maximum_seconds is not None:
                if maximum_seconds < 0:
                    raise ValueError("maximum_seconds must be non-negative")
                available = min(available, maximum_seconds)
            return max(0.0, available)

    def _active_totals(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        reserved = {name: 0 for name in RESOURCE_NAMES}
        general = {name: 0 for name in RESOURCE_NAMES}
        finalization = {name: 0 for name in RESOURCE_NAMES}
        for record in self._records.values():
            if record.state not in {"reserved", "dispatched"}:
                continue
            for name in RESOURCE_NAMES:
                reserved[name] += record.lease.reserved.get(name, 0)
                general[name] += record.general_reserved.get(name, 0)
                finalization[name] += record.finalization_reserved.get(name, 0)
        return reserved, general, finalization

    def _pool_remaining(self, resource: str, *, finalization: bool) -> int:
        _, general_reserved, held_reserved = self._active_totals()
        limit = self.budget.limits[resource]
        general_limit = max(0, limit - self.budget.finalization_reserve.get(resource, 0))
        if finalization:
            spent = self._spent_general[resource] + self._spent_finalization[resource]
            reserved = general_reserved[resource] + held_reserved[resource]
            return max(0, limit - spent - reserved)
        return max(0, general_limit - self._spent_general[resource] - general_reserved[resource])

    def _deny(self, reason: str, resource: str | None = None) -> BudgetDenial:
        self._last_denial_reason = reason
        metrics_registry.record_budget_event("denied", reason=reason)
        return BudgetDenial(reason=reason, resource=resource)

    def _paid_work_blocked(self, cost_microusd: int) -> bool:
        return cost_microusd > 0 and any(
            amount for resource, amount in self._overrun.items() if resource != "tasks"
        )

    def reserve(
        self,
        operation_id: str,
        demand: dict[str, int],
        work_class: str = "optional",
        *,
        stage: str = "research",
    ) -> BudgetLease | BudgetDenial:
        resolved = _demand_vector(demand)
        if work_class not in {"required", "optional"}:
            raise ValueError("work_class must be required or optional")
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        signature = tuple(resolved[name] for name in RESOURCE_NAMES)
        key = operation_id.strip()
        with self._lock:
            previous = self._records.get(key)
            if previous is not None:
                if previous.signature != signature or previous.lease.work_class != work_class or previous.lease.stage != stage:
                    return self._deny("operation_conflict")
                metrics_registry.record_budget_event("reserve")
                return replace(previous.lease)
            if self._paid_work_blocked(resolved["cost_microusd"]):
                return self._deny("dispatch_unbounded", "cost_microusd")
            if self._remaining_seconds() <= 0:
                return self._deny("deadline_exceeded")
            finalization = work_class == "required" and stage in FINALIZATION_STAGES
            if not finalization and self.remaining_timeout(stage=stage, work_class=work_class) <= 0:
                return self._deny("deadline_exceeded")

            general_allocations = {name: 0 for name in RESOURCE_NAMES}
            finalization_allocations = {name: 0 for name in RESOURCE_NAMES}
            for resource, amount in resolved.items():
                if amount == 0:
                    continue
                general_available = self._pool_remaining(resource, finalization=False)
                if finalization:
                    total_available = self._pool_remaining(resource, finalization=True)
                    if amount > total_available:
                        return self._deny("limit_exceeded", resource)
                    general_amount = min(amount, general_available)
                    held_amount = amount - general_amount
                    general_allocations[resource] = general_amount
                    finalization_allocations[resource] = held_amount
                    continue
                if amount > general_available:
                    total_available = self._pool_remaining(resource, finalization=True)
                    reason = "finalization_reserved" if amount <= total_available else "limit_exceeded"
                    return self._deny(reason, resource)
                general_allocations[resource] = amount

            lease = _IssuedBudgetLease(
                operation_id=key,
                reserved={name: amount for name, amount in resolved.items() if amount},
                deadline_at=self.budget.deadline_at,
                work_class=work_class,
                stage=stage,
            )
            self._records[key] = _LeaseRecord(
                lease=lease,
                signature=signature,
                general_reserved=general_allocations,
                finalization_reserved=finalization_allocations,
            )
            self._last_denial_reason = None
            metrics_registry.record_budget_event("reserve")
            return replace(lease)

    def reserve_llm_attempt(
        self,
        operation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        pricing: dict[str, Any] | None,
        work_class: str = "optional",
        stage: str = "research",
    ) -> BudgetLease | BudgetDenial:
        cost = estimate_llm_cost_microusd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        )
        if cost is None:
            with self._lock:
                return self._deny("pricing_unavailable")
        return self.reserve(
            operation_id,
            {
                "llm_calls": 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_microusd": cost,
            },
            work_class,
            stage=stage,
        )

    def dispatch(self, lease: BudgetLease, *, commit: Callable[[], None] | None = None) -> bool:
        with self._lock:
            record = self._records.get(lease.operation_id)
            if record is None or record.lease != lease or record.state != "reserved":
                self._deny("lease_lost")
                return False
            if self._paid_work_blocked(record.lease.reserved.get("cost_microusd", 0)):
                record.state = "released"
                self._deny("dispatch_unbounded", "cost_microusd")
                return False
            if self._remaining_seconds() <= 0:
                record.state = "released"
                self._deny("deadline_exceeded")
                return False
            record.state = "dispatched"
            attempts = sum(record.lease.reserved.get(name, 0) for name in EXTERNAL_RESOURCES)
            self._external_attempt_count += attempts
            if commit is not None:
                try:
                    commit()
                except BaseException:
                    self._external_attempt_count -= attempts
                    record.state = "reserved"
                    raise
            if attempts:
                metrics_registry.increment("budget_external_attempt_total", attempts)
            metrics_registry.record_budget_event("dispatch")
            return True

    def settle(
        self,
        lease: BudgetLease,
        usage: dict[str, int] | None,
        *,
        usage_source: str = "reported",
        outcome_unknown: bool = False,
    ) -> BudgetSettlement | BudgetDenial:
        if usage_source not in {"reported", "estimated", "unknown"}:
            raise ValueError("usage_source must be reported, estimated, or unknown")
        resolved_usage = _demand_vector({} if usage is None else usage, sparse=True)
        with self._lock:
            record = self._records.get(lease.operation_id)
            if record is None or record.lease != lease:
                return self._deny("lease_lost")
            if record.settlement is not None:
                return record.settlement
            if record.state != "dispatched":
                return self._deny("lease_lost")
            reserved = _demand_vector(record.lease.reserved)
            charged = reserved.copy()
            source = "unknown" if outcome_unknown else usage_source
            if source == "reported":
                charged.update(resolved_usage)
                # Dispatch already counted these attempts, even if the provider omitted them.
                for name in EXTERNAL_RESOURCES:
                    charged[name] = max(charged[name], reserved[name])
                if record.lease.reserved.keys() - resolved_usage.keys() - EXTERNAL_RESOURCES:
                    source = "estimated"
            accounted = {name: min(charged[name], reserved[name]) for name in RESOURCE_NAMES}
            for name in RESOURCE_NAMES:
                self._overrun[name] += max(0, charged[name] - reserved[name])
                general_charge = min(accounted[name], record.general_reserved.get(name, 0))
                held_charge = min(
                    max(0, accounted[name] - general_charge),
                    record.finalization_reserved.get(name, 0),
                )
                self._spent_general[name] += general_charge
                self._spent_finalization[name] += held_charge
            record.state = "settled"
            settlement = BudgetSettlement(
                operation_id=lease.operation_id,
                usage={name: accounted[name] for name in RESOURCE_NAMES if accounted[name]},
                usage_source=source,
                charged_cost_microusd=accounted["cost_microusd"],
            )
            record.settlement = settlement
            metrics_registry.record_budget_event("settle")
            return settlement

    def release(self, lease: BudgetLease) -> bool:
        """Release a lease only while it is explicitly known to be undispatched."""
        with self._lock:
            record = self._records.get(lease.operation_id)
            if record is None or record.lease != lease or record.state != "reserved":
                self._deny("lease_lost")
                return False
            record.state = "released"
            metrics_registry.record_budget_event("release")
            return True

    def settle_timeout(self, lease: BudgetLease) -> BudgetSettlement | BudgetDenial:
        """Conservatively close a dispatched operation when its outcome is unknown."""
        return self.settle(lease, usage=None, usage_source="unknown", outcome_unknown=True)

    def export_ledger(self) -> dict[str, Any]:
        """Serialize balances and attempts that still need a recovery decision."""
        with self._lock:
            return {
                "version": 1,
                "tenant_id": self.tenant_id,
                "request_id": self.request_id,
                "deadline_at": self.budget.deadline_at,
                "spent_general": dict(self._spent_general),
                "spent_finalization": dict(self._spent_finalization),
                "overrun": dict(self._overrun),
                "external_attempts": self._external_attempt_count,
                "last_denial_reason": self._last_denial_reason,
                "active": {
                    operation_id: {
                        "state": record.state,
                        "general": dict(record.general_reserved),
                        "finalization": dict(record.finalization_reserved),
                    }
                    for operation_id, record in self._records.items()
                    if record.state in {"reserved", "dispatched"}
                },
            }

    def restore_ledger(self, ledger: dict[str, Any]) -> None:
        """Recover once under a new owner. An old attempt can never dispatch again."""
        if (
            ledger.get("version") != 1
            or (ledger.get("tenant_id"), ledger.get("request_id")) != self.identity
            or ledger.get("deadline_at") != self.budget.deadline_at
        ):
            raise ValueError("Invalid budget ledger identity or version")
        general = _demand_vector(ledger["spent_general"])
        finalization = _demand_vector(ledger["spent_finalization"])
        overrun = _demand_vector(ledger["overrun"])
        attempts = _valid_nonnegative_int(ledger["external_attempts"], name="external_attempts")
        for operation_id, record in ledger["active"].items():
            if not operation_id or record["state"] not in {"reserved", "dispatched"}:
                raise ValueError("Invalid budget attempt record")
            reserved_general = _demand_vector(record["general"])
            reserved_finalization = _demand_vector(record["finalization"])
            if record["state"] == "dispatched":
                for resource in RESOURCE_NAMES:
                    general[resource] += reserved_general[resource]
                    finalization[resource] += reserved_finalization[resource]
        for resource, limit in self.budget.limits.items():
            if (
                general[resource] > limit - self.budget.finalization_reserve.get(resource, 0)
                or general[resource] + finalization[resource] > limit
            ):
                raise ValueError("Budget ledger exceeds frozen limits")
        with self._lock:
            self._spent_general = general
            self._spent_finalization = finalization
            self._overrun = overrun
            self._external_attempt_count = attempts
            self._last_denial_reason = ledger.get("last_denial_reason")

    def snapshot(
        self,
        *,
        minimum_demand: dict[str, int] | None = None,
        work_class: str = "optional",
        stage: str = "research",
    ) -> BudgetSnapshot:
        if work_class not in {"required", "optional"}:
            raise ValueError("work_class must be required or optional")
        minimum = _demand_vector(minimum_demand or {})
        with self._lock:
            reserved, _, _ = self._active_totals()
            spent = {
                name: self._spent_general[name] + self._spent_finalization[name]
                for name in RESOURCE_NAMES
            }
            remaining = {
                name: max(0, self.budget.limits[name] - spent[name] - reserved[name])
                for name in RESOURCE_NAMES
            }
            finalization = work_class == "required" and stage in FINALIZATION_STAGES
            pressure = "normal"
            if self._paid_work_blocked(minimum["cost_microusd"]) or self._remaining_seconds() <= 0:
                pressure = "exhausted"
            elif not finalization and self._remaining_seconds() <= self.budget.finalization_reserve_ms / 1000:
                pressure = "high"
            elif minimum_demand is not None:
                for resource, amount in minimum.items():
                    if amount == 0:
                        continue
                    available = self._pool_remaining(resource, finalization=finalization)
                    if amount > available:
                        pressure = "exhausted"
                        break
                    pool_size = self.budget.limits[resource]
                    if not finalization:
                        pool_size = max(0, pool_size - self.budget.finalization_reserve.get(resource, 0))
                        used = self._spent_general[resource] + self._active_totals()[1][resource]
                    else:
                        used = spent[resource] + reserved[resource]
                    if pool_size == 0 or used / pool_size >= 0.8:
                        pressure = "high"
            return BudgetSnapshot(
                limit=RequestBudget(**self.budget.to_dict()),
                spent=spent,
                reserved=reserved,
                remaining=remaining,
                pressure=pressure,
                last_denial_reason=self._last_denial_reason,
                external_attempts=self._external_attempt_count,
                overrun=dict(self._overrun),
            )
