"""Request snapshot and resume helpers for the deep research workflow."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from inspect import Parameter, signature
from json import dumps
from re import split
from threading import RLock
from typing import Any, Callable, Iterator
from uuid import uuid4

from metrics import WEB_EXTRACTION_REASONS, RequestTrace, metrics_registry
from models import BudgetDenial, RequestBudget, SummaryState, TodoItem
from services.budget import (
    BudgetGovernor,
    build_request_budget,
    build_request_budget_for_config,
)
from services.dag_scheduler import normalize_depends_on
from services.evidence import EvidenceStore, format_evidence_sources
from services.mcp_bridge import tool_approval_expired
from services.request_state import SnapshotConflictError, queue_snapshot_events
from services.web_extractor import (
    POLICY_VERSION,
    QUALITY_VERSION,
    WEB_SANITIZER_VERSION,
)
from services.web_fetch_gateway import GatewayError, normalize_url

_WEB_VERSIONS = {
    "schema_version": 1,
    "policy_version": POLICY_VERSION,
    "quality_version": QUALITY_VERSION,
    "sanitizer_version": WEB_SANITIZER_VERSION,
}
_WEB_TIERS = frozenset({"L0", "L1", "L2"})
_DYNAMIC_ADAPTERS = frozenset({"reader_http", "playwright"})
_MAX_DYNAMIC_TARGETS_PER_ATTEMPT = 5  # Requested URL, L1 final URL and three L2 redirects.
_WEB_TERMINAL_STATUSES = frozenset({
    "success", "partial_success", "completed", "cancelled", "failed", "workflow_failed", "rejected", "expired",
})


def freeze_web_extraction(config: Any, tenant_id: str) -> dict[str, Any] | None:
    if not getattr(config, "web_extraction_enabled_for", lambda _: False)(tenant_id):
        return None
    adapter = config.v4_web_l2_adapter
    rules = config.v4_web_l2_origin_rules.get(tenant_id, [])
    if not rules or (adapter == "playwright" and not config.v4_browser_use_enabled):
        adapter = "off"
    policy = {"tenant_id": tenant_id, "adapter": adapter, "rules": rules}
    if adapter == "reader_http":
        policy["reader_url"] = config.v4_web_reader_url
    return {
        **_WEB_VERSIONS, "enabled": True, "adapter": adapter,
        "policy_id": sha256(dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "dynamic_targets": {}, "tier_counts": {}, "degraded_counts": {},
    }


def supported_web_extraction(frozen: Any) -> bool:
    return isinstance(frozen, dict) and frozen.get("enabled") is True and all(
        type(frozen.get(key)) is int and frozen[key] == version for key, version in _WEB_VERSIONS.items()
    )


def web_extraction_adapter(frozen: Any, config: Any, tenant_id: str) -> str:
    if not supported_web_extraction(frozen) or frozen.get("adapter") not in ("reader_http", "playwright"):
        return "off"
    current = freeze_web_extraction(config, tenant_id)
    if current is None or current["policy_id"] != frozen.get("policy_id"):
        return "off"
    return current["adapter"]


def record_web_extraction_result(frozen: dict[str, Any], *, tier: str | None, reason: str | None) -> None:
    for field, value, allowed in (
        ("tier_counts", tier, _WEB_TIERS), ("degraded_counts", reason, WEB_EXTRACTION_REASONS),
    ):
        if value in allowed:
            counts = _public_web_counts(frozen.get(field), allowed)
            counts[value] = counts.get(value, 0) + 1
            frozen[field] = counts


def _public_web_counts(value: Any, allowed: frozenset[str]) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {key: count for key, count in value.items() if key in allowed and type(count) is int and count >= 0}


def replace_unfinished_task_evidence(snapshot: dict[str, Any], task_id: int, records: list[dict]) -> bool:
    if snapshot.get("status") in _WEB_TERMINAL_STATUSES or snapshot.get("workflow_cancel_requested"):
        return False
    for task in snapshot.get("todo_items") or ():
        if (
            isinstance(task, dict) and task.get("id") == task_id
            and task.get("status") not in {"completed", "skipped", "failed"}
            and task.get("evidence_items") != records
        ):
            task["evidence_items"] = deepcopy(records)
            return True
    return False


def freeze_report(config: Any, tenant_id: str, request_id: str) -> dict[str, Any] | None:
    if not (getattr(config, "parallel_report_enabled", False) or getattr(config, "red_team_enabled", False)):
        return None
    quick = build_request_budget_for_config(config, tenant_id, depth="quick")
    return {
        "schema_version": 1, "tenant_id": tenant_id, "request_id": request_id,
        "frozen_options": {**config.report_config_projection(), "report_layout_mode": config.resolved_report_layout_mode()},
        "budget_policy": {"limits": quick.limits, "reserve": quick.finalization_reserve,
                          "reserve_ms": quick.finalization_reserve_ms},
        "operations_by_key": {}, "sections_by_id": {},
    }


_REPORT_REASONS = frozenset({
    "budget_denied", "deadline", "timeout", "invalid_output", "missing_scope", "outcome_unknown",
    "input_limit", "no_model", "no_candidates", "provider_error", "input_changed", "disabled",
})


def public_report_projection(report: Any) -> dict[str, Any] | None:
    outline = report.get("outline") if isinstance(report, dict) else None
    if not isinstance(outline, dict) or type(outline.get("outline_version")) is not int:
        return None
    final = report.get("final") or {}
    published = final.get("published") is True
    finished_sections = report.get("sections_by_id") or {}
    sections = []
    for item in (outline.get("sections") or [])[:10]:
        if not isinstance(item, dict):
            continue
        sid, title, order = item.get("section_id"), item.get("title"), item.get("order")
        if not isinstance(sid, str) or not isinstance(title, str) or type(order) is not int:
            continue
        row = finished_sections.get(sid) or {}
        status = row.get("status")
        section = {"section_id": sid[:80], "title": title[:120], "order": order,
                   "status": status if status in ("pending", "started", "completed", "degraded") else "pending"}
        if published:
            section["claim_ids"] = [claim[:80] for claim in (row.get("claim_ids") or [])[:64] if isinstance(claim, str)]
        sections.append(section)
    critic = (report.get("critique") or {}).get("status")
    result = {
        "outline_version": outline["outline_version"], "sections": sections,
        "critic_status": critic if critic in ("ok", "timeout", "invalid", "skipped") else "skipped",
    }
    if published and type(final.get("report_version")) is int:
        result["report_version"] = final["report_version"]
    reason = report.get("degradation_reason")
    if isinstance(reason, str) and reason in _REPORT_REASONS:
        result["degradation_reason"] = reason
    return result


def public_v4_projection(v4: Any) -> dict[str, Any] | None:
    if not isinstance(v4, dict) or v4.get("schema_version") != 1:
        return None
    result: dict[str, Any] = {"schema_version": 1}
    budget = v4.get("budget")
    snapshot = budget.get("snapshot") if isinstance(budget, dict) else None
    if isinstance(snapshot, dict):
        result["budget"] = {
            key: deepcopy(snapshot[key]) for key in (
                "spent", "reserved", "remaining", "pressure", "last_denial_reason",
                "external_attempts", "overrun",
            ) if key in snapshot and snapshot[key] is not None
        }
        limit = snapshot.get("limit")
        if isinstance(limit, dict):
            result["budget"]["limit"] = {
                key: deepcopy(limit[key]) for key in (
                    "depth", "limits", "finalization_reserve", "deadline_at",
                    "finalization_reserve_ms", "profile_version",
                ) if key in limit
            }
    replan = v4.get("replan")
    if isinstance(replan, dict) and not (v4.get("web_downstream_only") or v4.get("report_only")):
        result["replan"] = {
            key: deepcopy(replan[key]) for key in ("dag_revision", "decision_id", "pruned_task_ids")
            if key in replan and replan[key] is not None
        }
    web = v4.get("web_extraction")
    if supported_web_extraction(web):
        result["web_extraction"] = {
            "schema_version": web["schema_version"],
            "tier_counts": _public_web_counts(web.get("tier_counts"), _WEB_TIERS),
            "degraded_counts": _public_web_counts(web.get("degraded_counts"), WEB_EXTRACTION_REASONS),
        }
    report = public_report_projection(v4.get("report"))
    if report is not None:
        result["report"] = report
    return result


def public_report_workbench(workbench):
    """Project guarded report bindings onto the public workbench contract."""
    public = dict(workbench or {})
    sections = public.pop("report_sections", [])
    blocks = public.pop("report_blocks", [])
    red_team = public.pop("red_team", {})
    public["section_bindings"] = [
        {key: item[key] for key in ("section_id", "title", "paragraph_index") if key in item}
        for item in sections
    ]
    public["claim_bindings"] = [
        {**{key: block[key] for key in ("section_id", "block_id", "claim_id", "paragraph_index", "text_fingerprint")},
         "citation_ids": block["source_ids"],
         "evidence_anchors": [
             {key: ref[key] for key in ("source_id", "artifact_id", "anchor_id") if key in ref}
             for ref in block["evidence_refs"]
         ]}
        for block in blocks if block["kind"] == "fact" and block["claim_id"]
    ]
    if red_team:
        public["critic_issues"] = [
            {**{key: item[key] for key in ("section_id", "claim_id", "kind", "severity", "source_ids", "issue_id")},
             "status": "handled" if item["resolution"] == "withheld" else "unresolved"}
            for item in red_team["issues"][:8]
        ]
    return public


def report_chunks(markdown: str) -> list[str]:
    return split(r"(?<=\n\n)(?=## )", markdown)


def materialize_report_event(snapshot, descriptor):
    """Build wire frames from the single committed Markdown/workbench pair."""
    event = dict(descriptor)
    kind = event["type"]
    markdown = snapshot.get("report_markdown") or ""
    workbench = snapshot.get("workbench") or {}
    charts = workbench.get("charts_spec") or []
    if kind == "report_chunk" and "content" not in event:
        chunks = report_chunks(markdown)
        event.update(content=chunks[event["index"] - 1], total_chunks=len(chunks))
    elif kind == "report_chart" and "chart_index" in event:
        event["chart"] = charts[event.pop("chart_index")]
    elif kind == "report_note":
        event.update(note_id=snapshot.get("report_note_id"), content=markdown.strip())
        if snapshot.get("report_note_path"):
            event["note_path"] = snapshot["report_note_path"]
    elif kind == "metrics_snapshot" and "request_metrics" not in event:
        event.update(request_metrics=snapshot.get("request_metrics") or {},
                     aggregate_metrics=metrics_registry.snapshot(include_recent_requests=False))
    elif kind == "final_report":
        event.update(
            report=markdown, note_id=snapshot.get("report_note_id"), note_path=snapshot.get("report_note_path"),
            evidence_conflicts=snapshot.get("evidence_conflicts") or [],
            claim_verification_summary=snapshot.get("claim_verification_summary") or {},
            v4=public_v4_projection(snapshot.get("v4")),
        )
        if charts:
            event["charts_spec"] = charts
        if workbench:
            event.update(workbench=workbench, citation_drawer_map=workbench.get("citation_index", {}),
                         verifier_badges=workbench.get("verifier_badges", []))
    return event


class ReportCancelled(Exception):
    """Cancellation observed before publishing a final report."""


class BudgetDispatchDenied(RuntimeError):
    """The request cannot dispatch this external attempt."""

    budget_denied = True


@dataclass
class BudgetAttempt:
    timeout: float
    usage: dict[str, int] | None = None
    associate_dynamic_url: Callable[[str], bool] | None = None
    outcome_unknown: bool = False
    reserved: dict[str, int] | None = None


class RequestBudgetController:
    def __init__(self, agent: Any, state: SummaryState, governor: BudgetGovernor, *, blocked: bool = False):
        self.agent = agent
        self.state = state
        self.governor = governor
        self.blocked = blocked
        self.stage = "planning"
        self.lock = getattr(agent, "_request_snapshot_lock", None) or RLock()
        self.observation = 0
        self.pressure = None
        self.last_emitted_pressure = None
        self._scope = uuid4().hex
        self._attempt_number = 0

    def _operation_id(self) -> str:
        self._attempt_number += 1
        return f"{self._scope}:{self._attempt_number}"

    @property
    def replan(self) -> dict[str, Any]:
        return self.state.v4["replan"]

    def snapshot(self):
        demand = {"llm_calls": 1} if self.stage in {"planning", "reflection", "review", "report"} else {"search_calls": 1}
        return self.governor.snapshot(minimum_demand=demand, work_class="required", stage=self.stage)

    def payload(self) -> dict[str, Any]:
        return {
            "limit": self.governor.budget.to_dict(),
            "snapshot": asdict(self.snapshot()),
            "ledger": self.governor.export_ledger(),
        }

    def degrade(self, reason: str) -> None:
        with self.lock:
            if not self.blocked:
                self.state.v4.setdefault("degradation_reason", reason)

    def refuse(self, reason: str, *, required: bool = True) -> None:
        metrics_registry.record_budget_event("denied", reason=reason)
        if required:
            self.degrade(reason)
        raise BudgetDispatchDenied(reason)

    def can_dispatch(
        self, demand: dict[str, int], *, required: bool = True, stage: str | None = None,
    ) -> bool:
        work_class = "required" if required else "optional"
        stage = stage or self.stage
        return (
            not self.blocked
            and self.governor.remaining_timeout(stage=stage, work_class=work_class) > 0
            and self.governor.snapshot(minimum_demand=demand, stage=stage, work_class=work_class).pressure
            in ({"normal", "high"} if required else {"normal"})
        )

    def web_owner_is_current(self) -> bool:
        with self.lock:
            checkpoint = getattr(self.agent, "_workflow_checkpoint", None)
            if checkpoint is not None:
                return checkpoint.web_owner_is_current()
            store = self.agent._request_state_store
            if store is None:
                return False
            current = store.load(self.agent.request_id, tenant_id=self.governor.tenant_id) or {}
            return (
                current.get("state_version") == self.state.state_version
                and current.get("status") not in _WEB_TERMINAL_STATUSES
            )

    @staticmethod
    def _target_key(url: str) -> str:
        return sha256(normalize_url(url).encode()).hexdigest()

    def _dynamic_targets(self) -> dict[str, str]:
        web = self.state.v4.get("web_extraction")
        targets = web.get("dynamic_targets") if supported_web_extraction(web) else None
        if not isinstance(targets, dict):
            self.refuse("dispatch_unbounded", required=False)
        return targets

    def _associate_dynamic_url(self, url: str, *, operation_id: str) -> bool:
        with self.lock:
            if self.agent._is_cancelled() or not self.web_owner_is_current():
                return False
            key = self._target_key(url)
            targets = self._dynamic_targets()
            if key in targets:
                return targets[key] == operation_id
            if sum(owner == operation_id for owner in targets.values()) >= _MAX_DYNAMIC_TARGETS_PER_ATTEMPT:
                return False
            if operation_id not in targets.values():
                return False
            self._persist_web_dispatch({**targets, key: operation_id})
            return True

    def _persist_web_dispatch(self, targets: dict[str, str]) -> None:
        web = {**self.state.v4["web_extraction"], "dynamic_targets": targets}
        self.agent._persist_v4_budget(self.state, self.payload(), web_extraction=web)

    def charge_tasks(self, count: int) -> None:
        # Graph and this balance must be committed in the same CAS.
        if not count:
            return
        lease = self.governor.reserve(self._operation_id(), {"tasks": count}, "required", stage=self.stage)
        if isinstance(lease, BudgetDenial):
            self.refuse(lease.reason)
        if not self.governor.dispatch(lease):
            self.refuse("deadline_exceeded")
        self.governor.settle(lease, {"tasks": count})

    @contextmanager
    def attempt(
        self, demand: dict[str, int], *, llm=None, required: bool = True,
        stage: str | None = None, dynamic_urls: tuple[str, ...] | None = None,
        operation_id: str | None = None, commit_attempt: Callable[..., None] | None = None,
        allow_optional_finalization: bool = False, keep_available: dict[str, int] | None = None,
        deadline_margin: float = 0, maximum_seconds: float | None = None,
    ) -> Iterator[BudgetAttempt]:
        work_class = "required" if required else "optional"
        stage = stage or self.stage
        def persist(phase: str, lease=None) -> None:
            if commit_attempt is not None:
                commit_attempt(phase, lease)
            else:
                self.agent._persist_v4_budget(self.state, self.payload())

        def remaining_timeout() -> float:
            remaining = self.governor.remaining_timeout(stage=stage, work_class=work_class) - deadline_margin
            return min(remaining, maximum_seconds) if maximum_seconds is not None else remaining

        with self.lock:
            if self.blocked or self.agent._is_cancelled():
                self.refuse("dispatch_unbounded" if self.blocked else "lease_lost", required=required)
            operation_id = operation_id or self._operation_id()
            dynamic_targets = None
            if dynamic_urls is not None:
                targets = self._dynamic_targets()
                try:
                    keys = {self._target_key(url) for url in dynamic_urls}
                except GatewayError:
                    self.refuse("url_denied", required=False)
                if not keys or len(keys) > 2 or keys.intersection(targets):
                    self.refuse("not_eligible", required=False)
                if demand.get("dynamic_fetches") != 1 or llm is not None or required:
                    self.refuse("dispatch_unbounded", required=False)
                if self.governor.snapshot(minimum_demand=demand, stage=stage, work_class=work_class).pressure == "high":
                    self.refuse("optional_pressure", required=False)
                dynamic_targets = {**targets, **dict.fromkeys(keys, operation_id)}
            if llm is None:
                lease = self.governor.reserve(operation_id, demand, work_class, stage=stage)
            else:
                input_tokens, output_tokens, pricing = llm
                lease = self.governor.reserve_llm_attempt(
                    operation_id, input_tokens=input_tokens, output_tokens=output_tokens,
                    pricing=pricing, work_class=work_class, stage=stage,
                    allow_optional_finalization=allow_optional_finalization, keep_available=keep_available,
                )
            if isinstance(lease, BudgetDenial):
                if required:
                    self.degrade(lease.reason)
                if commit_attempt is None:
                    persist("denied")
                raise BudgetDispatchDenied(lease.reason)
            persist("reserved", lease)
            timeout = remaining_timeout()
            if timeout <= 0 or self.agent._is_cancelled():
                self.governor.release(lease)
                persist("released", lease)
                self.refuse("deadline_exceeded" if timeout <= 0 else "lease_lost", required=required)
            def commit() -> None:
                if dynamic_targets is None:
                    persist("dispatched", lease)
                else:
                    self._persist_web_dispatch(dynamic_targets)

            try:
                dispatched = self.governor.dispatch(lease, commit=commit)
            except BaseException:
                self.blocked = True
                raise
            if not dispatched:
                persist("released", lease)
                self.refuse("deadline_exceeded", required=required)
            if dynamic_targets is not None:
                adapter = self.state.v4["web_extraction"].get("adapter")
                if adapter in _DYNAMIC_ADAPTERS:
                    metrics_registry.record_web_extraction("l2_selected", adapter=adapter)
            attempt = BudgetAttempt(remaining_timeout(), reserved=lease.reserved)
            if dynamic_targets is not None:
                attempt.associate_dynamic_url = lambda url: self._associate_dynamic_url(url, operation_id=operation_id)
        try:
            if attempt.timeout <= 0 or self.agent._is_cancelled():
                self.refuse("deadline_exceeded" if attempt.timeout <= 0 else "lease_lost", required=required)
            yield attempt
        except BaseException:
            with self.lock:
                self.governor.settle_timeout(lease)
                persist("settled", lease)
            raise
        else:
            with self.lock:
                self.governor.settle(
                    lease, attempt.usage, usage_source="reported" if attempt.usage is not None else "estimated",
                    outcome_unknown=attempt.outcome_unknown,
                )
                persist("settled", lease)


class ReportCheckpoint:
    """Single-writer report journal. Workers receive only detached model requests."""

    def __init__(self, agent, state, controller, *, step, emit_stream, supported=True):
        self.agent = agent
        self.state = state
        self.controller = controller
        self.step = step
        self.emit_stream = emit_stream
        self.supported = supported
        self.events = []

    @property
    def data(self):
        return self.state.v4["report"]

    def key(self, kind, section_id=""):
        return sha256(dumps((self.agent.tenant_id, self.agent.request_id, 1, kind, section_id)).encode()).hexdigest()

    def event(self, event_type, key, **fields):
        return {"type": event_type, "event_key": f"report-{key}", "step": self.step, **fields}

    def section_event(self, section_id, status, reason=None):
        return self.event(
            "report_section_status", f"section-1-{section_id}-{status}", section_id=section_id,
            outline_version=1, status=status, **({"reason_code": reason} if reason in _REPORT_REASONS else {}),
        )

    def commit(self, update, *events, operation=None):
        if self.agent._is_cancelled():
            raise ReportCancelled()
        current = self.data
        expected = current.get("revision_number", 0)
        proposed = {**current, **update, "revision_number": expected + 1}
        if operation:
            key, record = operation
            proposed["operations_by_key"] = {**current["operations_by_key"], key: record}
        v4_update = {"report": proposed}
        if self.state.v4.get("report_only"):
            v4_update["report_only"] = True
        if self.controller and not self.controller.blocked:
            v4_update["budget"] = self.controller.payload()
        pending = events if self.emit_stream else ()
        if self.supported:
            saved = self.agent._persist_report_checkpoint(self.state, v4_update, expected, pending)
        else:
            saved = None
        self.state.v4.update(v4_update)
        committed_version = saved["state_version"] if saved else None
        self.events.extend({
            **event, "schema_version": 1, "request_id": self.agent.request_id,
            **({"state_version": committed_version} if committed_version is not None else {}),
        } for event in pending)

    def drain(self):
        events, self.events = self.events, []
        return events

    def start(self, kind, section_id, request, pricing, *, required, keep_available=None):
        """Return an entered budget attempt whose settlement commits the guarded result."""
        if not self.controller or self.controller.blocked or not self.supported or pricing is None:
            raise BudgetDispatchDenied("dispatch_unbounded")
        key = self.key(kind, section_id)
        digest = sha256((request.messages_json + request.options_json).encode()).hexdigest()
        if (self.data["operations_by_key"].get(key) or {}).get("status") not in (None, "reserved"):
            raise SnapshotConflictError("Report operation already dispatched")
        update = {}
        events = ()
        terminal = {"status": "degraded", "reason": "outcome_unknown"}

        def commit_attempt(phase, lease):
            if phase == "released":
                phase = "reserved"
            record = {"kind": kind, "section_id": section_id, "input_digest": digest,
                      "upper_bound": lease.reserved, "status": phase}
            if phase == "dispatched" and section_id:
                started = {**self.data["sections_by_id"], section_id: {"status": "started"}}
                self.commit({"sections_by_id": started}, self.section_event(section_id, "started"),
                            operation=(key, record))
            elif phase == "settled":
                self.commit(update, *events, operation=(key, {**record, **terminal}))
            else:
                self.commit({}, operation=(key, record))

        context = self.controller.attempt(
            {}, llm=(request.input_upper_bound, request.max_tokens, pricing), required=required,
            stage="report", operation_id=key, commit_attempt=commit_attempt, allow_optional_finalization=True,
            keep_available=keep_available, deadline_margin=3, maximum_seconds=request.timeout,
        )
        attempt = context.__enter__()

        def settle(result, changes, progress, usage, reason):
            nonlocal events
            update.update(changes)
            events = progress
            terminal.update(result)
            attempt.usage = usage
            attempt.outcome_unknown = reason in {"timeout", "provider_error", "outcome_unknown"}
            context.__exit__(None, None, None)

        return attempt.timeout, settle

    def previous(self, kind, section_id=""):
        return self.data["operations_by_key"].get(self.key(kind, section_id), {})

    def complete(self, kind, section_id, reason, result, *, settle=None, usage=None, update=None):
        record = {"kind": kind, "section_id": section_id, "reason": reason, "result": result,
                  "status": "completed" if reason == "ok" else "degraded"}
        changes = dict(update or {})
        events = ()
        if kind == "map":
            status = record["status"]
            changes["sections_by_id"] = {**self.data["sections_by_id"], section_id: {"status": status}}
            events = (self.section_event(section_id, status, reason),)
        if settle is None:
            previous = self.previous(kind, section_id)
            self.commit(changes, *events, operation=(self.key(kind, section_id), {**previous, **record}))
        else:
            settle(record, changes, events, usage, reason)


class StateManagerMixin:
    """Persist, restore, and serialize request workflow state."""

    def _init_request_budget(self, state: SummaryState, *, resumed: bool) -> RequestBudgetController | None:
        self._research_budget = None
        self._web_budget = None
        if not hasattr(self, "_request_snapshot_lock"):
            self._request_snapshot_lock = RLock()
        checkpoint = getattr(self, "_workflow_checkpoint", None)
        frozen = checkpoint.load() if checkpoint is not None and not resumed else None
        if frozen:
            checkpoint.check_owner(frozen)
            state.state_version = frozen["state_version"]
            state.v4 = deepcopy(frozen.get("v4"))
        tenant_id = getattr(self, "tenant_id", "default")
        research_budget_enabled = getattr(self.config, "v4_research_budget_enabled", False)
        if state.v4 is None and not (resumed or frozen):
            web = freeze_web_extraction(self.config, tenant_id)
            if web is not None:
                state.v4 = {"schema_version": 1, "web_extraction": web, "replan": {}}
                if not research_budget_enabled:
                    state.v4["web_downstream_only"] = True
        if not (resumed or frozen):
            report = freeze_report(self.config, tenant_id, self.request_id)
            if report is not None:
                if state.v4 is None:
                    state.v4 = {"schema_version": 1}
                state.v4["report"] = report
                if research_budget_enabled:
                    state.v4.setdefault("replan", {})
        v4 = state.v4
        if v4 is None and (resumed or frozen or not research_budget_enabled):
            return None
        store = self._request_state_store
        if store is not None and not (
            self._store_supports_tenant_id("load") and self._store_supports_tenant_id("save")
            and self._callable_accepts_kwarg(store.save, "expected_version")
        ):
            return None
        downstream_only = isinstance(v4, dict) and v4.get("web_downstream_only") is True
        web = v4.get("web_extraction") if isinstance(v4, dict) else None
        new_web_request = web is not None and not (resumed or frozen)
        if isinstance(v4, dict) and (v4.get("report_only") or (
            "budget" not in v4 and not new_web_request and (resumed or frozen or not research_budget_enabled)
        )):
            return None
        blocked = False
        try:
            if v4 is None or "budget" not in v4:
                budget = build_request_budget_for_config(
                    self.config, tenant_id=tenant_id, depth="quick" if downstream_only else None,
                )
            else:
                if not isinstance(v4, dict) or v4.get("schema_version") != 1:
                    raise ValueError("Unsupported budget snapshot version")
                budget = RequestBudget(**v4["budget"]["limit"])
                if budget.profile_version != 1:
                    raise ValueError("Unsupported budget profile version")
            governor = BudgetGovernor(budget, tenant_id=tenant_id, request_id=self.request_id)
            if v4 is not None and "budget" in v4:
                if "ledger" in v4["budget"]:
                    governor.restore_ledger(v4["budget"]["ledger"])
                elif not frozen or frozen.get("phase") not in {"queued", "planning"}:
                    raise ValueError("Missing budget ledger")
                if not isinstance(v4.get("replan"), dict):
                    raise ValueError("Missing budget execution checkpoint")
        except (TypeError, ValueError, KeyError, AttributeError):
            governor = BudgetGovernor(
                build_request_budget_for_config(self.config, tenant_id=tenant_id, depth="quick" if downstream_only else None),
                tenant_id=tenant_id, request_id=self.request_id,
            )
            blocked = True
            if not isinstance(v4, dict):
                state.v4 = {}
        if state.v4 is None:
            state.v4 = {"schema_version": 1, "replan": {}}
        if not blocked:
            for key, value in {
                "dag_revision": 0, "pruned_task_ids": [], "wave_1_committed": False,
                "plan_locked": False, "slots_allocated": False,
            }.items():
                state.v4["replan"].setdefault(key, value)
        controller = RequestBudgetController(self, state, governor, blocked=blocked)
        if supported_web_extraction(web) and store is not None and not blocked:
            self._web_budget = controller
        if not downstream_only:
            self._research_budget = controller
        return self._research_budget

    def _init_report_checkpoint(self, state, budget, *, step, emit_stream):
        store = self._request_state_store
        supported = store is None or (
            self._store_supports_tenant_id("load") and self._store_supports_tenant_id("save")
            and self._callable_accepts_kwarg(store.save, "expected_version")
        )
        controller = budget or getattr(self, "_web_budget", None)
        if controller is None and supported:
            v4 = state.v4
            existing = v4.get("budget")
            policy = v4["report"].get("budget_policy") or {}
            try:
                if existing:
                    limit = RequestBudget(**existing["limit"])
                else:
                    limit = build_request_budget(
                        "quick", limits=policy["limits"], finalization_reserve=policy["reserve"],
                        finalization_reserve_ms=policy["reserve_ms"],
                    )
                governor = BudgetGovernor(limit, tenant_id=self.tenant_id, request_id=self.request_id)
                if existing:
                    governor.restore_ledger(existing["ledger"])
                else:
                    v4["report_only"] = True
                controller = RequestBudgetController(self, state, governor)
            except (KeyError, TypeError, ValueError):
                controller = None
        if controller:
            controller.stage = "report"
        return ReportCheckpoint(self, state, controller, step=step, emit_stream=emit_stream, supported=supported)

    def _persist_report_checkpoint(self, state, updates, expected, events):
        checkpoint = getattr(self, "_workflow_checkpoint", None)
        if checkpoint is not None:
            saved = checkpoint.persist(
                {"v4": updates}, expected_report_revision=expected, pending_events=events,
            )
        elif self._request_state_store is not None:
            current = self._load_resume_snapshot(self.request_id)
            v4 = current.get("v4") or {}
            if current["state_version"] != state.state_version or (v4.get("report") or {}).get("revision_number", 0) != expected:
                raise SnapshotConflictError("Report checkpoint changed")
            current["v4"] = {**v4, **updates}
            if events:
                current["v4"]["pending_events"] = queue_snapshot_events(current, v4, events, self.request_id)
            saved = self._request_state_store.save(
                self.request_id, current, tenant_id=self.tenant_id, expected_version=state.state_version,
            )
        else:
            return None
        state.state_version = saved["state_version"]
        return saved

    def _persist_v4_budget(
        self, state: SummaryState, budget: dict[str, Any], *, web_extraction: dict[str, Any] | None = None,
    ) -> None:
        """Write the ledger without changing committed plan, approvals or outbox."""
        checkpoint = getattr(self, "_workflow_checkpoint", None)
        if checkpoint is not None:
            saved = (
                checkpoint.persist_budget(budget, web_extraction=web_extraction)
                if web_extraction is not None else checkpoint.persist_budget(budget)
            )
        elif self._request_state_store is not None:
            current = self._load_resume_snapshot(self.request_id)
            if current.get("state_version") != state.state_version:
                raise SnapshotConflictError("Request budget ownership changed")
            current["v4"] = {**current.get("v4", {}), "budget": budget}
            if web_extraction is not None:
                current["v4"]["web_extraction"] = web_extraction
            saved = self._request_state_store.save(
                self.request_id, current, tenant_id=getattr(self, "tenant_id", "default"),
                expected_version=state.state_version,
            )
        else:
            saved = None
        state.v4["budget"] = budget
        if web_extraction is not None:
            state.v4["web_extraction"] = web_extraction
        if saved:
            state.state_version = saved["state_version"]

    def _persist_web_evidence(self, state: SummaryState, task_id: int, records: list[dict]) -> None:
        """Commit a tool result to its current task before returning it to a caller."""
        if getattr(self, "_web_budget", None) is None or self._is_cancelled():
            return
        with getattr(self, "_request_snapshot_lock", nullcontext()):
            checkpoint = getattr(self, "_workflow_checkpoint", None)
            if checkpoint is not None:
                saved = checkpoint.persist_web_evidence(task_id, records)
            else:
                current = self._load_resume_snapshot(self.request_id)
                if current.get("state_version") != state.state_version:
                    raise SnapshotConflictError("Request evidence ownership changed")
                if not replace_unfinished_task_evidence(current, task_id, records):
                    return
                saved = self._request_state_store.save(
                    self.request_id, current, tenant_id=getattr(self, "tenant_id", "default"),
                    expected_version=state.state_version,
                )
            state.state_version = saved["state_version"]

    def _restore_task_counters(self, observer: RequestTrace, state: SummaryState) -> None:
        """Replay persisted task counts into a fresh request trace."""

        observer.set_task_totals(total_tasks=len(state.todo_items))
        observer.update_task_status_counts(
            completed=sum(1 for task in state.todo_items if task.status == "completed"),
            skipped=sum(1 for task in state.todo_items if task.status == "skipped"),
            failed=sum(1 for task in state.todo_items if task.status == "failed"),
        )
        observer.restore_review_result(state.review_summary, state.claim_verification_summary)

    def _persist_request_state(
        self,
        state: SummaryState,
        *,
        phase: str,
        status: str = "in_progress",
        report_markdown: str | None = None,
        v4_expected: dict[str, Any] | None = None,
        v4_events: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Persist the latest request snapshot for history and resume."""

        research_controller = getattr(self, "_research_budget", None)
        controller = research_controller or getattr(self, "_web_budget", None)
        web_enabled = isinstance(state.v4, dict) and "web_extraction" in state.v4
        report_state = (state.v4 or {}).get("report")
        report_cas = report_state is not None and self._request_state_store is not None and (
            self._store_supports_tenant_id("load") and self._store_supports_tenant_id("save")
            and self._callable_accepts_kwarg(self._request_state_store.save, "expected_version")
        )
        if self._request_state_store is None:
            if controller is not None and not controller.blocked:
                state.v4["budget"] = controller.payload()
            return list(v4_events or [])

        observer_snapshot = self._request_trace.snapshot() if self._request_trace else {}
        cache_diagnostics = {
            "cache_hits": observer_snapshot.get("cache_hits", 0),
            "cache_exact_hits": observer_snapshot.get("cache_exact_hits", 0),
            "cache_semantic_hits": observer_snapshot.get("cache_semantic_hits", 0),
            "cache_misses": observer_snapshot.get("cache_misses", 0),
            "last_search_cache_details": dict(observer_snapshot.get("last_search_cache_details") or {}),
        }
        payload = {
            "snapshot_version": 2,
            "request_id": self.request_id,
            "topic": state.research_topic,
            "phase": phase,
            "status": status,
            "error": observer_snapshot.get("error"),
            "search_api": observer_snapshot.get("search_api"),
            "elapsed_ms": observer_snapshot.get("elapsed_ms"),
            "report_markdown": (report_markdown or state.structured_report or "") if report_state is not None
            else (report_markdown or state.structured_report or state.running_summary or ""),
            "report_charts": list(getattr(state, "report_charts", None) or []),
            **({"charts_spec": list(state.report_charts)} if state.report_charts else {}),
            **({
                "workbench": state.report_workbench,
                "citation_drawer_map": state.report_workbench.get("citation_index", {}),
                "verifier_badges": state.report_workbench.get("verifier_badges", []),
            } if getattr(state, "report_workbench", None) else {}),
            "report_note_id": state.report_note_id,
            "report_note_path": state.report_note_path,
            "review_summary": dict(state.review_summary or {}),
            "reflection_completed": bool(state.reflection_completed),
            "review_completed": bool(state.review_completed),
            "report_repair_completed": bool(state.report_repair_completed),
            "report_repair_cycles": int(state.report_repair_cycles or 0),
            "topic_cache_warmup_completed": bool(state.topic_cache_warmup_completed),
            "skill_pack": dict(state.skill_pack_metadata or {}),
            "pending_tool_approval": (
                dict(state.pending_tool_approval)
                if isinstance(getattr(state, "pending_tool_approval", None), dict)
                else None
            ),
            "approved_tool_calls": dict(getattr(state, "approved_tool_calls", None) or {}),
            "evidence_conflicts": list(getattr(state, "evidence_conflicts", None) or []),
            "claim_verification_summary": dict(
                getattr(state, "claim_verification_summary", None) or {}
            ),
            "state_version": int(getattr(state, "state_version", 0) or 0),
            "runner_backend": str(getattr(state, "runner_backend", "inline") or "inline"),
            "workflow_run_id": getattr(state, "workflow_run_id", None),
            "tenant_id": str(getattr(self, "tenant_id", "default") or "default").strip()
            or "default",
            "cache_diagnostics": cache_diagnostics,
            "request_metrics": observer_snapshot,
        }
        tenant_id = str(getattr(self, "tenant_id", "default") or "default").strip() or "default"
        with getattr(self, "_request_snapshot_lock", nullcontext()):
            serialized_tasks = [self._serialize_task(task) for task in state.todo_items]
            if isinstance(self._evidence_store, EvidenceStore) and (
                getattr(self.config, "evidence_pointer_disclosure_enabled", True)
                or getattr(self.config, "resolved_sandbox_effective_enabled", lambda: False)()
                or web_enabled
            ):
                for task in serialized_tasks:
                    evidence = self._evidence_store.list_task_evidence(task["id"], include_full_content=True)
                    if evidence:
                        task["evidence_items"] = evidence
            payload.update(
                todo_items=serialized_tasks,
                fact_ledger=deepcopy(state.fact_ledger),
                evidence_index_cards=deepcopy(state.evidence_index_cards),
                frozen_prefix_version=state.frozen_prefix_version,
                context_checkpoint_blocks=deepcopy(state.context_checkpoint_blocks),
            )
            payload["approved_tool_calls"] = dict(getattr(state, "approved_tool_calls", None) or {})
            payload["state_version"] = int(getattr(state, "state_version", 0) or 0)
            checkpoint = getattr(self, "_workflow_checkpoint", None)
            if controller is not None and not controller.blocked:
                state.v4["budget"] = controller.payload()
                payload["v4"] = {key: deepcopy(value) for key, value in state.v4.items() if key != "pending_events"}
                if checkpoint is not None and v4_expected is None:
                    # The coordinator sends the graph explicitly when it advances.
                    payload["v4"].pop("replan", None)
            elif state.v4 is not None and (checkpoint is None or web_enabled or report_state):
                payload["v4"] = {key: deepcopy(value) for key, value in state.v4.items() if key != "pending_events"}
            if checkpoint is not None:
                payload["runner_backend"] = checkpoint.backend
                payload["workflow_run_id"] = checkpoint.run_id
                state.runner_backend = checkpoint.backend
                state.workflow_run_id = checkpoint.run_id
                if v4_expected is not None or v4_events:
                    saved = checkpoint.persist(payload, **(v4_expected or {}), pending_events=v4_events or ())
                else:
                    saved = checkpoint.persist(payload)
            elif self._store_supports_tenant_id("save"):
                kwargs: dict[str, Any] = {"tenant_id": tenant_id}
                if controller is not None or report_cas:
                    current = self._request_state_store.load(self.request_id, tenant_id=tenant_id) or {}
                    previous_v4 = current.get("v4")
                    if isinstance(previous_v4, dict) and "v4" in payload:
                        payload["v4"] = {**previous_v4, **payload["v4"]}
                    payload = {**current, **payload}
                    if report_cas and v4_events:
                        payload["v4"]["pending_events"] = queue_snapshot_events(
                            current, previous_v4 or {}, v4_events, self.request_id,
                        )
                    if v4_expected and "expected_report_revision" in v4_expected:
                        if ((previous_v4 or {}).get("report") or {}).get("revision_number") != v4_expected["expected_report_revision"]:
                            raise SnapshotConflictError("Report checkpoint changed")
                if controller is not None or report_cas or payload["pending_tool_approval"] or payload["approved_tool_calls"]:
                    kwargs["expected_version"] = payload["state_version"]
                saved = self._request_state_store.save(self.request_id, payload, **kwargs)
            else:
                saved = self._request_state_store.save(self.request_id, payload)
            if isinstance(saved, dict) and "state_version" in saved:
                state.state_version = int(saved.get("state_version") or state.state_version or 0)
        if research_controller is None:
            self._refresh_note_memory_for_state(state, observer=self._request_trace)
        return [{**event, "state_version": state.state_version} for event in v4_events or []]

    def _consume_mcp_approval(self, key: str, entry: dict[str, Any]) -> bool:
        state = self._active_state
        if state is None or self._request_state_store is None:
            return False
        with getattr(self, "_request_snapshot_lock", nullcontext()):
            checkpoint = getattr(self, "_workflow_checkpoint", None)
            if checkpoint is not None:
                saved = checkpoint.consume_tool_approval(key, entry)
                if saved is None:
                    return False
                state.state_version = saved["state_version"]
                state.approved_tool_calls[key] = saved["approved_tool_calls"][key]
                return True
            snapshot = self._load_resume_snapshot(self.request_id)
            registry = dict(snapshot.get("approved_tool_calls") or {})
            current = registry.get(key)
            if (
                snapshot.get("state_version") != state.state_version
                or current != entry or not isinstance(current, dict) or current.get("status") != "approved_once"
                or tool_approval_expired(current)
            ):
                return False
            consumed = {**current, "status": "consumed"}
            registry[key] = consumed
            snapshot["approved_tool_calls"] = registry
            try:
                saved = self._request_state_store.save(
                    self.request_id, snapshot, tenant_id=self.tenant_id, expected_version=state.state_version,
                )
            except SnapshotConflictError:
                return False
            state.state_version = saved["state_version"]
            state.approved_tool_calls[key] = consumed
        return True

    def _store_supports_tenant_id(self, method_name: str) -> bool:
        method = getattr(self._request_state_store, method_name, None)
        if method is None:
            return False
        try:
            params = signature(method).parameters
        except (TypeError, ValueError):
            return False
        return "tenant_id" in params or any(
            param.kind == Parameter.VAR_KEYWORD for param in params.values()
        )

    def _load_resume_snapshot(self, request_id: str) -> dict[str, Any]:
        if self._request_state_store is None:
            raise ValueError("request_state store is disabled")
        tenant_id = str(getattr(self, "tenant_id", "default") or "default").strip() or "default"
        if self._store_supports_tenant_id("load"):
            payload = self._request_state_store.load(request_id, tenant_id=tenant_id)
        else:
            payload = self._request_state_store.load(request_id)
        if (
            not payload
            or (payload.get("tenant_id") or "default") != tenant_id
            or (payload.get("request_id") or request_id) != request_id
        ):
            raise ValueError(f"resume snapshot not found: {request_id}")
        return payload

    @staticmethod
    def _task_from_payload(payload: dict[str, Any]) -> TodoItem:
        """Deserialize a persisted task payload into TodoItem."""

        return TodoItem(
            id=int(payload.get("id") or 0),
            title=str(payload.get("title") or "").strip() or "任务",
            intent=str(payload.get("intent") or "").strip() or "恢复执行任务",
            query=str(payload.get("query") or "").strip(),
            status=str(payload.get("status") or "pending").strip() or "pending",
            summary=str(payload.get("summary") or "").strip() or None,
            summary_payload=(
                dict(payload.get("summary_payload") or {})
                if isinstance(payload.get("summary_payload"), dict)
                else None
            ),
            sources_summary=str(payload.get("sources_summary") or "").strip() or None,
            notices=[
                str(item).strip()
                for item in payload.get("notices") or []
                if str(item).strip()
            ],
            evidence_items=list(payload.get("evidence_items") or []),
            claims=list(payload.get("claims") or []),
            review_issues=list(payload.get("review_issues") or []),
            review_status=str(payload.get("review_status") or "pending").strip() or "pending",
            note_id=str(payload.get("note_id") or "").strip() or None,
            note_path=str(payload.get("note_path") or "").strip() or None,
            stream_token=str(payload.get("stream_token") or "").strip() or None,
            origin=str(payload.get("origin") or "planned").strip() or "planned",
            round=max(1, int(payload.get("round") or 1)),
            react_rounds=max(0, int(payload.get("react_rounds") or 0)),
            react_fetch_count=max(0, int(payload.get("react_fetch_count") or 0)),
            react_additional_search_count=max(
                0,
                int(payload.get("react_additional_search_count") or 0),
            ),
            react_gap_signals=[
                str(item).strip()
                for item in payload.get("react_gap_signals") or []
                if str(item).strip()
            ],
            react_last_action=str(payload.get("react_last_action") or "").strip() or None,
            react_stop_reason=str(payload.get("react_stop_reason") or "").strip() or None,
            react_observation=(
                dict(payload.get("react_observation") or {})
                if isinstance(payload.get("react_observation"), dict)
                else {}
            ),
            depends_on=normalize_depends_on(payload.get("depends_on")),
            upstream_context=str(payload.get("upstream_context") or "").strip(),
            fact_claims=list(payload.get("fact_claims") or []),
            atomic_claims=[
                deepcopy(claim)
                for claim in payload["atomic_claims"]
                if isinstance(claim, dict)
            ] if isinstance(payload.get("atomic_claims"), list) else [],
        )

    def _state_from_snapshot(self, payload: dict[str, Any]) -> tuple[SummaryState, str]:
        """Restore SummaryState and phase from a persisted snapshot."""

        if payload.get("phase") == "awaiting_tool_approval":
            pending = payload.get("pending_tool_approval")
            if (
                not isinstance(pending, dict)
                or pending.get("status") != "approved_once"
                or tool_approval_expired(pending)
            ):
                raise ValueError("A valid tool approval is required before resuming")

        todo_items = []
        for item in payload.get("todo_items") or []:
            if isinstance(item, dict):
                todo_items.append(self._task_from_payload(item))

        raw_report = payload.get("report_markdown")
        tracked_report = (payload.get("v4") or {}).get("report")
        report_markdown = (
            raw_report if (tracked_report.get("final") or {}).get("published") is True else None
        ) if isinstance(tracked_report, dict) else (str(raw_report or "").strip() or None)
        state = SummaryState(
            research_topic=str(payload.get("topic") or "").strip() or None,
            running_summary=report_markdown,
            todo_items=todo_items,
            structured_report=report_markdown,
            report_note_id=str(payload.get("report_note_id") or "").strip() or None,
            report_note_path=str(payload.get("report_note_path") or "").strip() or None,
            review_summary=dict(payload.get("review_summary") or {}),
            reflection_completed=bool(payload.get("reflection_completed")),
            review_completed=bool(payload.get("review_completed")),
            report_repair_completed=bool(payload.get("report_repair_completed")),
            report_repair_cycles=max(0, int(payload.get("report_repair_cycles") or 0)),
            topic_cache_warmup_completed=bool(payload.get("topic_cache_warmup_completed")),
            skill_pack_metadata=(
                dict(payload.get("skill_pack") or {})
                if isinstance(payload.get("skill_pack"), dict)
                else {}
            ),
            fact_ledger=(
                deepcopy(payload["fact_ledger"])
                if isinstance(payload.get("fact_ledger"), dict)
                else {}
            ),
            evidence_index_cards=deepcopy(payload.get("evidence_index_cards") or []),
            frozen_prefix_version=int(payload.get("frozen_prefix_version") or 0),
            context_checkpoint_blocks=deepcopy(payload.get("context_checkpoint_blocks") or []),
            pending_tool_approval=(
                dict(payload.get("pending_tool_approval"))
                if isinstance(payload.get("pending_tool_approval"), dict)
                else None
            ),
            approved_tool_calls=(
                dict(payload.get("approved_tool_calls") or {})
                if isinstance(payload.get("approved_tool_calls"), dict)
                else {}
            ),
            report_charts=(
                list(payload.get("report_charts") or [])
                if isinstance(payload.get("report_charts"), list)
                else list(payload["charts_spec"]) if isinstance(payload.get("charts_spec"), list) else []
            ),
            report_workbench=(
                deepcopy(payload["workbench"]) if isinstance(payload.get("workbench"), dict) else {}
            ),
            evidence_conflicts=(
                [deepcopy(conflict) for conflict in payload["evidence_conflicts"] if isinstance(conflict, dict)]
                if isinstance(payload.get("evidence_conflicts"), list)
                else []
            ),
            claim_verification_summary=(
                deepcopy(payload["claim_verification_summary"])
                if isinstance(payload.get("claim_verification_summary"), dict)
                else {}
            ),
            state_version=max(0, int(payload.get("state_version") or 0)),
            runner_backend=str(payload.get("runner_backend") or "inline").strip() or "inline",
            workflow_run_id=str(payload.get("workflow_run_id") or "").strip() or None,
            v4=deepcopy(payload.get("v4")),
        )
        web = state.v4.get("web_extraction") if isinstance(state.v4, dict) else None
        web_enabled = web is not None
        self._evidence_store.web_extraction_enabled = web_enabled
        if web_enabled:
            self._evidence_store.hydrate_from_tasks(state.todo_items, trusted_web_snapshot=supported_web_extraction(web))
        else:
            self._evidence_store.hydrate_from_tasks(state.todo_items)
        sandbox = getattr(self, "_sandbox_service", None)
        if sandbox is not None:
            sandbox.restore_task_counts(state.todo_items)
        for task in state.todo_items:
            if (web_enabled or not task.sources_summary) and task.id > 0:
                evidence_items = self._evidence_store.list_task_evidence(task.id)
                if evidence_items or web_enabled:
                    task.evidence_items = evidence_items
                    task.sources_summary = format_evidence_sources(evidence_items)
        phase = str(payload.get("phase") or "planning").strip() or "planning"
        return state, "task_execution" if phase == "awaiting_tool_approval" else phase
