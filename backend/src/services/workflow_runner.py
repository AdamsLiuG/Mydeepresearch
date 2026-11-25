"""Durable workflow execution abstraction supporting inline, local thread, ARQ, and Temporal."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from threading import Event, Lock, Thread
from time import monotonic, time
from typing import Any, Protocol
from uuid import uuid4

from services.budget import (
    BudgetGovernor,
    build_request_budget_for_config,
    profile_for_depth,
)
from services.event_bus import TERMINAL_EVENT_TYPES, RedisStreamEventBus, StreamEventBus
from services.mcp_bridge import tool_approval_expired
from services.request_state import SnapshotConflictError, queue_snapshot_events

logger = logging.getLogger(__name__)

_FINISHED_STATUSES = {"completed", "success", "partial_success", "cancelled", "rejected", "expired", "workflow_failed"}
_SIGNAL_NAMES = {"approve_plan", "approve_tool", "approve_tool_call", "cancel"}


class WorkflowUnavailableError(RuntimeError):
    """The selected workflow backend could not accept the submission."""


class WorkflowLeaseLost(RuntimeError):
    """The worker can no longer update or publish events for this request."""


class WorkflowConnectionError(WorkflowUnavailableError):
    """Temporal could not be contacted before submission."""


class WorkflowEventDeliveryError(RuntimeError):
    """A committed event is still pending publication or acknowledgment."""


def require_shared_workflow_backends(config: Any, store: Any, bus: Any) -> None:
    from services.storage_backends import PostgresRequestStateStore

    if not isinstance(bus, RedisStreamEventBus):
        raise WorkflowUnavailableError("External workflows require a shared Redis event bus")
    if store is None:
        raise WorkflowUnavailableError("External workflows require a shared request snapshot store")
    if getattr(config, "database_url", None) and not isinstance(store, PostgresRequestStateStore):
        raise WorkflowUnavailableError("External workflows could not initialize the configured PostgreSQL snapshot store")


class WorkflowCheckpoint:
    """CAS updates for one request's execution lease and persisted controls."""

    def __init__(
        self, store: Any, tenant_id: str, request_id: str, backend: str, run_id: str,
        *, lease_seconds: float = 30,
    ) -> None:
        self.store = store
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.backend = backend
        self.run_id = run_id
        self.lease_seconds = lease_seconds
        self.owner = uuid4().hex
        self.lost = Event()
        self._lock = Lock()

    def load(self) -> dict[str, Any] | None:
        snapshot = self.store.load(self.request_id, tenant_id=self.tenant_id)
        return snapshot if snapshot and snapshot.get("workflow_run_id") == self.run_id else None

    def _update(
        self, change: Callable[[dict[str, Any]], bool],
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            for _ in range(5):
                snapshot = self.store.load(self.request_id, tenant_id=self.tenant_id) or {}
                if snapshot and snapshot.get("workflow_run_id") != self.run_id:
                    raise ValueError("Request ID is already in use")
                version = int(snapshot.get("state_version") or 0)
                if not change(snapshot):
                    return snapshot, False
                try:
                    return self.store.save(
                        self.request_id, snapshot, tenant_id=self.tenant_id,
                        expected_version=version,
                    ), True
                except SnapshotConflictError:
                    continue
        raise SnapshotConflictError(f"Concurrent workflow updates for {self.request_id}")

    def reserve(
        self,
        topic: str,
        *,
        plan_first: bool = False,
        config: Any = None,
        research_depth: str | None = None,
        output_format: str | None = None,
        traceparent: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        from state_manager import freeze_report, freeze_web_extraction

        budget_enabled = getattr(config, "v4_research_budget_enabled", False)
        web = freeze_web_extraction(config, self.tenant_id)
        report = freeze_report(config, self.tenant_id, self.request_id)
        downstream_only = web is not None and not budget_enabled
        requested_depth = research_depth
        if budget_enabled and requested_depth is None:
            requested_depth = config.v4_default_research_depth
        options = {
            "search_api": _search_api_value(config),
            "default_output_format": _output_format_value(config, output_format),
        }

        def change(snapshot: dict[str, Any]) -> bool:
            if snapshot:
                different_parameters = (
                    snapshot.get("topic") != topic or bool(snapshot.get("workflow_plan_first")) != plan_first
                )
                saved_options = snapshot.get("workflow_request_options")
                if saved_options is not None:
                    saved_depth = saved_options.get("research_depth")
                    depth = profile_for_depth(requested_depth).depth if requested_depth else saved_depth
                    if different_parameters or saved_options != {**options, "research_depth": depth}:
                        raise SnapshotConflictError("Request ID already has different research parameters")
                elif different_parameters:
                    raise ValueError("Request ID already has different research parameters")
                return False
            snapshot.update(
                topic=topic, phase="queued", status="queued", runner_backend=self.backend,
                workflow_run_id=self.run_id, workflow_plan_first=plan_first,
            )
            if traceparent:
                from services.observability import parse_traceparent

                if parse_traceparent(traceparent):
                    snapshot["traceparent"] = traceparent.strip().lower()
            if budget_enabled or downstream_only:
                budget = build_request_budget_for_config(
                    config, self.tenant_id, depth="quick" if downstream_only else requested_depth,
                )
                governor = BudgetGovernor(budget, tenant_id=self.tenant_id, request_id=self.request_id)
                snapshot["workflow_request_options"] = {
                    **options, "research_depth": budget.depth if budget_enabled else None,
                }
                snapshot["v4"] = {
                    "schema_version": 1,
                    "budget": {
                        "limit": budget.to_dict(), "snapshot": asdict(governor.snapshot()),
                        "ledger": governor.export_ledger(),
                    },
                    "replan": {
                        "dag_revision": 0, "wave_1_committed": False, "pruned_task_ids": [],
                        "plan_locked": False, "slots_allocated": False,
                    },
                }
                if web is not None:
                    snapshot["v4"]["web_extraction"] = web
                if downstream_only:
                    snapshot["v4"]["web_downstream_only"] = True
            if report is not None:
                snapshot.setdefault("v4", {"schema_version": 1})["report"] = report
            if getattr(config, "v4_tenant_memory_partition_enabled", False):
                snapshot.setdefault("v4", {"schema_version": 1})["tenant_memory"] = {
                    "schema_version": 1,
                    "partition_enabled": True,
                    "tenant_id": self.tenant_id,
                }
                from state_manager import _ensure_strict_snapshot_owner

                _ensure_strict_snapshot_owner(
                    config, self.store, tenant_id=self.tenant_id, request_id=self.request_id,
                )
            return True

        return self._update(change)

    def _owns(self, snapshot: dict[str, Any]) -> bool:
        lease = snapshot.get("workflow_lease") or {}
        return lease.get("owner") == self.owner and float(lease.get("expires_at") or 0) > time()

    def claim(self, *, repair_v4_terminal: bool = False) -> tuple[dict[str, Any], bool]:
        def change(snapshot: dict[str, Any]) -> bool:
            if not snapshot:
                raise ValueError("Workflow snapshot is missing")
            lease = snapshot.get("workflow_lease") or {}
            if (
                snapshot.get("status") in _FINISHED_STATUSES
                and not (repair_v4_terminal and isinstance(snapshot.get("v4"), dict))
            ):
                return False
            if float(lease.get("expires_at") or 0) > time():
                return False
            snapshot["workflow_lease"] = {"owner": self.owner, "expires_at": time() + self.lease_seconds}
            return True

        return self._update(change)

    def refresh(self, cancel_event: Event) -> None:
        def change(snapshot: dict[str, Any]) -> bool:
            self.check_owner(snapshot)
            if snapshot.get("workflow_cancel_requested"):
                cancel_event.set()
            lease = snapshot["workflow_lease"]
            now = time()
            if float(lease["expires_at"]) - now > self.lease_seconds / 2:
                return False
            lease["expires_at"] = now + self.lease_seconds
            return True

        self._update(change)

    def check_owner(self, snapshot: dict[str, Any]) -> None:
        if self.lost.is_set() or not self._owns(snapshot):
            raise WorkflowLeaseLost(f"Execution lease expired for {self.request_id}")

    def web_owner_is_current(self) -> bool:
        snapshot = self.load() or {}
        return (
            not self.lost.is_set() and self._owns(snapshot)
            and not snapshot.get("workflow_cancel_requested")
            and snapshot.get("status") not in _FINISHED_STATUSES
        )

    def persist(
        self,
        payload: dict[str, Any],
        *,
        expected_dag_revision: int | None = None,
        expected_wave_1_committed: bool | None = None,
        expected_plan_locked: bool | None = None,
        pending_events: tuple[dict[str, Any], ...] = (),
        expected_report_revision: int | None = None,
    ) -> dict[str, Any]:
        updates = {name: value for name, value in payload.items() if name != "v4"}
        v4_updates = {name: value for name, value in (payload.get("v4") or {}).items() if name != "pending_events"}
        expectations = {
            "dag_revision": (expected_dag_revision, 0),
            "wave_1_committed": (expected_wave_1_committed, False),
            "plan_locked": (expected_plan_locked, False),
        }

        def change(snapshot: dict[str, Any]) -> bool:
            self.check_owner(snapshot)
            v4 = snapshot.get("v4") or {}
            if expected_report_revision is not None:
                if (v4.get("report") or {}).get("revision_number", 0) != expected_report_revision:
                    raise SnapshotConflictError(f"Report checkpoint changed for {self.request_id}")
                if snapshot.get("workflow_cancel_requested") and updates.get("status") != "cancelled":
                    from state_manager import ReportCancelled

                    raise ReportCancelled()
            replan = v4.get("replan") or {}
            proposed = v4_updates.get("replan") or {}
            if any(
                expected is not None and replan.get(name, default) != expected
                for name, (expected, default) in expectations.items()
            ):
                if (
                    expected_wave_1_committed is False and replan.get("wave_1_committed") is True
                    and proposed.get("decision_id") and all(replan.get(key) == value for key, value in proposed.items())
                ):
                    return False
                raise SnapshotConflictError(f"Research plan changed for {self.request_id}")
            limit = (v4.get("budget") or {}).get("limit")
            budget_update = v4_updates.get("budget") or {}
            if limit is not None and budget_update.get("limit", limit) != limit:
                raise SnapshotConflictError(f"Frozen budget changed for {self.request_id}")
            phase = updates.get("phase")
            if phase is not None and phase != snapshot.get("phase"):
                snapshot.pop("workflow_waiting_since", None)
                if phase == "awaiting_approval":
                    snapshot["workflow_plan_pending_revision"] = int(snapshot["state_version"]) + 1
                    snapshot["workflow_waiting_since"] = time()
                elif phase == "awaiting_tool_approval":
                    snapshot["workflow_waiting_since"] = time()
            snapshot.update(updates)
            if v4_updates or pending_events:
                merged = {**v4, **v4_updates}
                for name in ("budget", "replan"):
                    if name in v4_updates:
                        merged[name] = {**(v4.get(name) or {}), **v4_updates[name]}
                if pending_events:
                    merged["pending_events"] = queue_snapshot_events(snapshot, v4, pending_events, self.request_id)
                if expected_report_revision is not None and (final := (merged.get("report") or {}).get("final")):
                    if final.get("published") and final.get("frames"):
                        merged["report"] = {**merged["report"], "final": {**final, "frames": [
                            {**frame, "state_version": int(snapshot["state_version"]) + 1}
                            for frame in final["frames"]
                        ]}}
                snapshot["v4"] = merged
            return True

        snapshot, _ = self._update(change)
        return snapshot

    def persist_budget(
        self, budget: dict[str, Any], *, web_extraction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        v4 = {"budget": budget}
        if web_extraction is not None:
            v4["web_extraction"] = web_extraction
        return self.persist({"v4": v4})

    def persist_web_evidence(self, task_id: int, records: list[dict]) -> dict[str, Any]:
        from state_manager import replace_unfinished_task_evidence

        def change(snapshot: dict[str, Any]) -> bool:
            self.check_owner(snapshot)
            return replace_unfinished_task_evidence(snapshot, task_id, records)

        snapshot, _ = self._update(change)
        return snapshot

    def acknowledge_event(self, event_key: str) -> None:
        def change(snapshot: dict[str, Any]) -> bool:
            self.check_owner(snapshot)
            v4 = snapshot.get("v4") or {}
            pending = v4.get("pending_events") or []
            remaining = [event for event in pending if event["event_key"] != event_key]
            if len(pending) == len(remaining):
                return False
            snapshot["v4"] = {**v4, "pending_events": remaining}
            return True

        self._update(change)

    def consume_tool_approval(self, key: str, entry: dict[str, Any]) -> dict[str, Any] | None:
        def change(snapshot: dict[str, Any]) -> bool:
            self.check_owner(snapshot)
            registry = snapshot.get("approved_tool_calls") or {}
            if (
                snapshot.get("workflow_cancel_requested") or registry.get(key) != entry
                or entry.get("status") != "approved_once" or tool_approval_expired(entry)
            ):
                return False
            snapshot["approved_tool_calls"] = {**registry, key: {**entry, "status": "consumed"}}
            return True

        snapshot, changed = self._update(change)
        return snapshot if changed else None

    def release(self) -> None:
        def change(snapshot: dict[str, Any]) -> bool:
            if (snapshot.get("workflow_lease") or {}).get("owner") != self.owner:
                return False
            snapshot.pop("workflow_lease")
            return True

        self._update(change)

    def signal(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.load() is None:
            return {
                "request_id": self.request_id, "tenant_id": self.tenant_id, "runner_backend": self.backend,
                "signal_name": name, "accepted": False, "run_found": False, "status": None,
            }
        accepted = False

        def change(snapshot: dict[str, Any]) -> bool:
            nonlocal accepted
            accepted = False
            if not snapshot or name not in _SIGNAL_NAMES:
                return False
            if name == "approve_plan" and snapshot.get("workflow_plan_revision") is not None:
                supplied = payload.get("revision", payload.get("state_version", payload.get("plan_revision")))
                accepted = supplied is None or str(supplied) == str(snapshot["workflow_plan_revision"])
                return False
            if snapshot.get("status") in _FINISHED_STATUSES:
                return False
            if name == "cancel":
                accepted = True
                if snapshot.get("workflow_cancel_requested"):
                    return False
                snapshot["workflow_cancel_requested"] = True
                return True
            if name == "approve_plan" and snapshot.get("phase") == "awaiting_approval":
                revision = snapshot.get("state_version")
                supplied = payload.get("revision", payload.get("state_version", payload.get("plan_revision")))
                valid_revisions = {str(revision)}
                edited_revision = snapshot.get("plan_revision")
                if edited_revision is not None:
                    valid_revisions.add(str(edited_revision))
                elif snapshot.get("workflow_plan_pending_revision") is not None:
                    valid_revisions.add(str(snapshot["workflow_plan_pending_revision"]))
                v4 = snapshot.get("v4")
                budgeted_plan = isinstance(v4, dict) and v4.get("budget") and not v4.get("web_downstream_only")
                if budgeted_plan:
                    revision = edited_revision or snapshot.get("workflow_plan_pending_revision") or revision
                    valid_revisions = {str(revision)}
                if supplied is not None and str(supplied) not in valid_revisions:
                    return False
                accepted = True
                snapshot["workflow_plan_revision"] = supplied if supplied is not None else edited_revision or revision
                if budgeted_plan:
                    snapshot["v4"] = {**v4, "replan": {**(v4.get("replan") or {}), "plan_locked": True}}
                return True
            if name in {"approve_tool", "approve_tool_call"}:
                pending = snapshot.get("pending_tool_approval") or {}
                desired = "rejected" if payload.get("decision") == "reject" else "approved_once"
                accepted = (
                    pending.get("approval_id") == payload.get("approval_id")
                    and pending.get("status") == desired
                    and not tool_approval_expired(pending)
                )
            return False

        snapshot, _ = self._update(change)
        return {
            "request_id": self.request_id, "tenant_id": self.tenant_id, "runner_backend": self.backend,
            "signal_name": name, "accepted": accepted, "run_found": bool(snapshot),
            "status": "cancelling" if accepted and name == "cancel" else snapshot.get("status"),
        }


def _workflow_run_id(backend: str, tenant_id: str, request_id: str) -> str:
    identity = sha256(f"{tenant_id}\0{request_id}".encode()).hexdigest()
    return f"deepresearch-{backend}-{identity}"


def _stored_workflow_status(
    store: Any, backend: str, tenant_id: str, request_id: str,
) -> dict[str, Any] | None:
    snapshot = store.load(request_id, tenant_id=tenant_id) if store is not None else None
    if not snapshot or snapshot.get("runner_backend") != backend:
        return None
    return {
        "request_id": request_id, "tenant_id": tenant_id, "runner_backend": backend,
        "workflow_run_id": snapshot["workflow_run_id"], "status": snapshot["status"],
        "phase": snapshot.get("phase"),
    }


def run_durable_attempt(
    checkpoint: WorkflowCheckpoint,
    agent_factory: Callable[[], Any],
    bus: StreamEventBus,
    cancel_event: Event,
    *,
    hitl_timeout: float = 86400,
    retry_error: bool = False,
    stop_event: Event | None = None,
) -> dict[str, Any]:
    """Resume from the last saved step, stopping at an approval or terminal state."""

    request_id = checkpoint.request_id
    tenant_id = checkpoint.tenant_id
    snapshot = checkpoint.load() or {}
    v4 = snapshot.get("v4")
    phase = snapshot.get("phase")
    pending = snapshot.get("pending_tool_approval") or {}
    waiting_since = snapshot.get("workflow_waiting_since")
    if (
        waiting_since is not None and time() - float(waiting_since) < hitl_timeout
        and not (isinstance(v4, dict) and v4.get("pending_events"))
        and not snapshot.get("workflow_cancel_requested") and not cancel_event.is_set()
        and (
            (phase == "awaiting_approval" and snapshot.get("workflow_plan_revision") is None)
            or (
                phase == "awaiting_tool_approval" and pending.get("status") == "awaiting_approval"
                and not tool_approval_expired(pending)
            )
        )
    ):
        return {"request_id": request_id, "tenant_id": tenant_id, "status": "awaiting_approval"}
    snapshot, claimed = checkpoint.claim(repair_v4_terminal=True)
    has_v4 = isinstance(snapshot.get("v4"), dict)
    delivered_keys: set[str] = set()

    def publish(event: dict[str, Any]) -> None:
        if has_v4:
            checkpoint.check_owner(checkpoint.load() or {})
        elif checkpoint.lost.is_set():
            raise WorkflowLeaseLost(f"Execution lease expired for {request_id}")
        try:
            bus.publish(tenant_id, request_id, event)
        except Exception as exc:
            if has_v4 and (event.get("type") in TERMINAL_EVENT_TYPES or event.get("type") == "final_report"):
                raise WorkflowEventDeliveryError(f"Terminal frames are pending for {request_id}") from exc
            raise

    def replay_pending_events() -> None:
        if not has_v4:
            return
        saved = checkpoint.load() or {}
        checkpoint.check_owner(saved)
        for event in (saved.get("v4") or {}).get("pending_events", []):
            try:
                from state_manager import materialize_report_event

                publish(materialize_report_event(saved, event) if event["event_key"].startswith("report-") else event)
                key = event["event_key"]
                checkpoint.acknowledge_event(key)
                delivered_keys.add(key)
            except WorkflowLeaseLost:
                raise
            except Exception as exc:
                raise WorkflowEventDeliveryError(f"State event is pending for {request_id}") from exc

    def report_projection(saved: dict[str, Any]) -> dict[str, Any]:
        if not has_v4:
            return {}
        from state_manager import public_v4_projection

        projected = public_v4_projection(saved.get("v4"))
        return {"v4": projected} if projected else {}

    def finish(status: str, *, detail: str = "") -> dict[str, Any]:
        saved = checkpoint.load() or {}
        if status == "completed" and saved.get("status") in {"success", "partial_success"}:
            status = saved["status"]
        checkpoint.persist({"status": status, "phase": "completed" if status in {
            "success", "partial_success", "completed",
        } else status, **({"error": detail} if detail else {})})
        replay_pending_events()
        return result(status)

    def result(status: str) -> dict[str, Any]:
        return {"request_id": request_id, "tenant_id": tenant_id, "status": status}

    def repair_terminal(status: str) -> dict[str, Any]:
        replay_pending_events()
        last_events = bus.replay_since(tenant_id, request_id)
        if not last_events or last_events[-1].payload.get("type") not in TERMINAL_EVENT_TYPES:
            event_type = "cancelled" if status == "cancelled" else "done"
            if status in {"rejected", "expired", "workflow_failed"}:
                event_type = "error"
            if event_type == "done" and snapshot.get("report_markdown"):
                final_report = {
                    "type": "final_report", "report": snapshot["report_markdown"],
                    **{name: snapshot[name] for name in (
                        "charts_spec", "workbench", "evidence_conflicts", "claim_verification_summary",
                    ) if name in snapshot},
                }
                if has_v4:
                    final_report.update(
                        note_id=snapshot.get("report_note_id"),
                        note_path=snapshot.get("report_note_path"),
                        **report_projection(snapshot),
                    )
                if not has_v4 or not any(
                    event.payload.get("type") == "final_report"
                    and event.payload.get("report") == final_report["report"]
                    and event.payload.get("v4") == final_report.get("v4")
                    for event in last_events
                ):
                    publish(final_report)
            publish({"type": event_type, "detail": snapshot.get("error")})
        return result(status)

    status = str(snapshot.get("status") or "")
    if not claimed:
        if status not in _FINISHED_STATUSES or float((snapshot.get("workflow_lease") or {}).get("expires_at") or 0) > time():
            return result("lease_busy")
        return repair_terminal(status)

    stop_monitor = Event()

    def monitor() -> None:
        while not stop_monitor.wait(min(0.5, checkpoint.lease_seconds / 4)):
            try:
                if stop_event is not None and stop_event.is_set():
                    raise WorkflowLeaseLost(f"Worker stopped for {request_id}")
                checkpoint.refresh(cancel_event)
            except Exception:
                checkpoint.lost.set()
                cancel_event.set()
                return

    monitor_thread = Thread(target=monitor, name=f"workflow-lease-{request_id[:8]}", daemon=True)
    monitor_thread.start()
    terminal_event = None
    try:
        if status in _FINISHED_STATUSES:
            try:
                return repair_terminal(status)
            except (WorkflowLeaseLost, WorkflowEventDeliveryError):
                raise
            except Exception as exc:
                raise WorkflowEventDeliveryError(f"Terminal frames are pending for {request_id}") from exc
        replay_pending_events()
        if snapshot.get("workflow_cancel_requested"):
            cancel_event.set()
        if cancel_event.is_set():
            terminal = finish("cancelled")
            publish({"type": "cancelled", "request_id": request_id})
            return terminal

        phase = snapshot.get("phase")
        waiting_since = snapshot.get("workflow_waiting_since")
        if phase == "awaiting_tool_approval":
            pending = snapshot.get("pending_tool_approval") or {}
            if pending.get("status") in {"rejected", "expired"} or tool_approval_expired(pending):
                status = "rejected" if pending.get("status") == "rejected" else "expired"
                terminal = finish(status, detail=f"Tool approval {status}")
                publish({"type": "error", "detail": f"Tool approval {status}"})
                return terminal
            waiting = pending.get("status") != "approved_once"
        else:
            waiting = phase == "awaiting_approval" and snapshot.get("workflow_plan_revision") is None
        if waiting:
            if waiting_since is None:
                checkpoint.persist({"workflow_waiting_since": time()})
            elif time() - float(waiting_since) >= hitl_timeout:
                terminal = finish("expired", detail="Approval timed out")
                publish({"type": "error", "detail": "Approval timed out"})
                return terminal
            return result("awaiting_approval")

        options = snapshot.get("workflow_request_options")
        agent_kwargs = {}
        if options and _accepts_kwarg(agent_factory, "workflow_request_options"):
            agent_kwargs["workflow_request_options"] = options
        agent = agent_factory(**agent_kwargs)
        bind_checkpoint = getattr(agent, "bind_workflow_checkpoint", None)
        if bind_checkpoint is not None:
            bind_checkpoint(checkpoint)
        try:
            if snapshot.get("workflow_plan_first") and phase == "queued":
                plan = agent.plan_only(str(snapshot.get("topic") or ""))
                saved = checkpoint.load() or {}
                if saved.get("phase") != "awaiting_approval":
                    saved = checkpoint.persist({
                        "phase": "awaiting_approval", "status": "awaiting_approval",
                        "todo_items": plan["todo_items"],
                    })
                revision = saved.get("plan_revision") or saved.get("workflow_plan_pending_revision")
                replay_pending_events()
                publish({
                    **plan, "type": "plan_approval_required",
                    "todo_items": saved["todo_items"], "revision": revision, "plan_revision": revision,
                })
                return result("awaiting_approval")
            if phase != "queued" and hasattr(agent, "run_stream_resume"):
                stream = agent.run_stream_resume
                argument = request_id
            else:
                stream = agent.run_stream
                argument = str(snapshot.get("topic") or "")
            kwargs = {"cancel_event": cancel_event} if _accepts_kwarg(stream, "cancel_event") else {}
            iterator = stream(argument, **kwargs)
            try:
                for event in iterator:
                    if not isinstance(event, dict):
                        continue
                    event_type = event.get("type")
                    if event.get("event_key") or event_type in TERMINAL_EVENT_TYPES | {
                        "todo_list", "final_report", "tool_approval_required",
                    }:
                        replay_pending_events()
                    if event_type in TERMINAL_EVENT_TYPES:
                        terminal_event = event
                        continue
                    if not cancel_event.is_set() and event.get("event_key") not in delivered_keys:
                        if event_type == "final_report" and has_v4:
                            event = {**event, **report_projection(checkpoint.load() or {})}
                        publish(event)
                    if event_type == "tool_approval_required":
                        waiting = True
            finally:
                close_iterator = getattr(iterator, "close", None)
                if close_iterator is not None:
                    close_iterator()
        finally:
            close_agent = getattr(agent, "close_workflow", None)
            if close_agent is not None:
                close_agent()

        if cancel_event.is_set() or (terminal_event or {}).get("type") == "cancelled":
            terminal = finish("cancelled")
            publish({"type": "cancelled", "request_id": request_id})
            return terminal
        if waiting:
            return result("awaiting_approval")
        if terminal_event and terminal_event.get("type") == "error":
            terminal = finish("workflow_failed", detail=str(terminal_event.get("detail") or "Research failed"))
            publish(terminal_event)
            return terminal
        terminal = finish("completed")
        if not terminal_event or terminal_event.get("event_key") not in delivered_keys:
            publish(terminal_event or {"type": "done", "request_id": request_id})
        return terminal
    except (WorkflowLeaseLost, WorkflowEventDeliveryError):
        raise
    except Exception as exc:
        if retry_error:
            raise
        logger.exception("Research workflow failed request_id=%s", request_id)
        terminal = finish("workflow_failed", detail=str(exc))
        publish({"type": "error", "detail": str(exc), "request_id": request_id})
        return terminal
    finally:
        stop_monitor.set()
        monitor_thread.join()
        checkpoint.release()


def _repair_terminal_submission(
    checkpoint: WorkflowCheckpoint, bus: StreamEventBus, snapshot: dict[str, Any],
) -> None:
    if not isinstance(snapshot.get("v4"), dict):
        return
    try:
        run_durable_attempt(checkpoint, lambda: None, bus, Event())
    except WorkflowEventDeliveryError as exc:
        raise WorkflowUnavailableError("Could not republish completed research events") from exc


@dataclass(frozen=True)
class WorkflowRunHandle:
    """Identifier and status summary returned when submitting a durable workflow run."""

    request_id: str
    workflow_run_id: str
    runner_backend: str
    status: str = "queued"
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "workflow_run_id": self.workflow_run_id,
            "runner_backend": self.runner_backend,
            "status": self.status,
            "tenant_id": self.tenant_id,
            "metadata": dict(self.metadata),
        }


class WorkflowRunner(Protocol):
    """Protocol implemented by durable and inline research execution backends."""

    backend_name: str

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
        plan_first: bool = False,
        lease_id: str | None = None,
        traceparent: str | None = None,
    ) -> WorkflowRunHandle:
        """Schedule a research workflow run asynchronously (or inline handle)."""

    def signal(
        self,
        request_id: str,
        signal_name: str,
        payload: dict[str, Any] | None = None,
        *,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Deliver a signal (cancel, approve_plan, approve_tool) to an active run."""

    def status(
        self,
        request_id: str,
        *,
        tenant_id: str = "default",
    ) -> dict[str, Any] | None:
        """Return workflow execution status if tracked."""


def _accepts_kwarg(fn: Any, arg_name: str) -> bool:
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or (
            param.name == arg_name and param.kind != inspect.Parameter.POSITIONAL_ONLY
        ):
            return True
    return False


def _instantiate_with_supported_kwargs(
    agent_factory: Callable[..., Any],
    *,
    config: Any,
    request_id: str,
    tenant_id: str,
    traceparent: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if _accepts_kwarg(agent_factory, "request_id"):
        kwargs["request_id"] = request_id
    if _accepts_kwarg(agent_factory, "tenant_id"):
        kwargs["tenant_id"] = tenant_id
    if traceparent and _accepts_kwarg(agent_factory, "traceparent"):
        kwargs["traceparent"] = traceparent
    if _accepts_kwarg(agent_factory, "config"):
        return agent_factory(config=config, **kwargs)
    return agent_factory(config, **kwargs)


class InlineWorkflowRunner:
    """In-process synchronous/streaming baseline runner preserving existing direct SSE behavior."""

    backend_name = "inline"

    def __init__(self, *, background: LocalDurableWorkflowRunner | None = None) -> None:
        self._background = background

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
        plan_first: bool = False,
        lease_id: str | None = None,
        traceparent: str | None = None,
    ) -> WorkflowRunHandle:
        if self._background is not None:
            return self._background.submit(
                topic, request_id=request_id, tenant_id=tenant_id, config=config,
                output_format=output_format, plan_first=plan_first, lease_id=lease_id,
                traceparent=traceparent,
            )
        return WorkflowRunHandle(
            request_id=request_id,
            workflow_run_id=f"inline-{request_id}",
            runner_backend=self.backend_name,
            status="ready",
            tenant_id=tenant_id,
        )

    def signal(
        self,
        request_id: str,
        signal_name: str,
        payload: dict[str, Any] | None = None,
        *,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        if self._background is not None:
            return self._background.signal(request_id, signal_name, payload, tenant_id=tenant_id)
        return {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "runner_backend": self.backend_name,
            "signal_name": signal_name,
            "accepted": False,
            "detail": "Inline runner does not keep detached background runs; use local_durable/temporal/arq.",
            "payload": dict(payload or {}),
        }

    def status(self, request_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
        return self._background.status(request_id, tenant_id=tenant_id) if self._background is not None else None

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._background is not None:
            self._background.shutdown(timeout)


@dataclass
class _TrackedLocalRun:
    request_id: str
    workflow_run_id: str
    tenant_id: str
    cancel_event: Event
    wake_event: Event = field(default_factory=Event)
    status: str = "queued"
    error: str | None = None
    thread: Thread | None = None
    signals: list[dict[str, Any]] = field(default_factory=list)


class LocalDurableWorkflowRunner:
    """Daemon-thread background runner decoupled from SSE disconnects with event_bus + signal support."""

    backend_name = "local_durable"

    def __init__(
        self,
        *,
        agent_factory: Callable[..., Any],
        event_bus: StreamEventBus,
        state_store: Any | None = None,
        on_release: Callable[..., None] | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._event_bus = event_bus
        self._state_store = state_store
        self._on_release = on_release
        self._lock = Lock()
        self._runs: dict[tuple[str, str], _TrackedLocalRun] = {}
        self._closed = False

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
        plan_first: bool = False,
        lease_id: str | None = None,
        traceparent: str | None = None,
    ) -> WorkflowRunHandle:
        tenant_id = str(tenant_id or "default").strip() or "default"
        workflow_run_id = _workflow_run_id(self.backend_name, tenant_id, request_id)
        cancel_event = Event()
        tracked = _TrackedLocalRun(
            request_id=request_id,
            workflow_run_id=workflow_run_id,
            tenant_id=tenant_id,
            cancel_event=cancel_event,
            status="queued",
        )
        key = (tracked.tenant_id, request_id)
        store = self._state_store
        if store is None:
            from services.storage_backends import build_request_state_store

            store = build_request_state_store(config)
        if store is None:
            raise WorkflowUnavailableError("Local durable workflows require request snapshots")
        checkpoint = WorkflowCheckpoint(store, tenant_id, request_id, self.backend_name, workflow_run_id)

        def agent_factory(*, workflow_request_options: dict[str, Any] | None = None) -> Any:
            return _instantiate_with_supported_kwargs(
                self._agent_factory,
                config=_apply_workflow_request_options(config, workflow_request_options),
                request_id=request_id, tenant_id=tenant_id, traceparent=traceparent,
            )

        def _worker() -> None:
            with self._lock:
                tracked.status = "running"
            try:
                while True:
                    try:
                        result = run_durable_attempt(
                            checkpoint, agent_factory, self._event_bus, cancel_event,
                            hitl_timeout=getattr(config, "hitl_signal_timeout_seconds", 86400),
                        )
                    except WorkflowEventDeliveryError:
                        if self._closed:
                            raise
                        tracked.wake_event.wait(0.5)
                        tracked.wake_event.clear()
                        continue
                    with self._lock:
                        tracked.status = result["status"]
                    if result["status"] not in {"awaiting_approval", "lease_busy"}:
                        break
                    tracked.wake_event.wait(0.5)
                    tracked.wake_event.clear()
            except Exception as exc:
                logger.exception("Local workflow failed request_id=%s", request_id)
                with self._lock:
                    tracked.status = "failed"
                    tracked.error = str(exc)
            finally:
                if self._on_release is not None:
                    try:
                        if _accepts_kwarg(self._on_release, "lease_id"):
                            self._on_release(tenant_id, lease_id=lease_id)
                        else:
                            self._on_release(tenant_id)
                    except Exception:
                        logger.exception("Failed to release workflow rate limit for %s", request_id)

        with self._lock:
            if self._closed:
                raise WorkflowUnavailableError("Local workflow runner is shutting down")
            self._state_store = store
            snapshot, created = checkpoint.reserve(
                topic, plan_first=plan_first, config=config, output_format=output_format,
                traceparent=traceparent,
            )
            existing = self._runs.get(key)
            if existing and existing.thread and existing.thread.is_alive():
                return WorkflowRunHandle(
                    request_id, workflow_run_id, self.backend_name,
                    status=str(snapshot["status"]), tenant_id=tenant_id, metadata={"duplicate": True},
                )
            finished = snapshot["status"] in _FINISHED_STATUSES
            if not finished:
                self._runs[key] = tracked
                if created:
                    self._event_bus.publish(
                        tenant_id, request_id, {
                            "type": "status", "message": "已提交后台持久执行任务。",
                            "request_id": request_id, "workflow_run_id": workflow_run_id,
                            "runner_backend": self.backend_name,
                        },
                    )
                tracked.thread = Thread(target=_worker, name=f"durable-run-{request_id[:8]}", daemon=True)
                tracked.thread.start()
        if finished:
            _repair_terminal_submission(checkpoint, self._event_bus, snapshot)
            return WorkflowRunHandle(
                request_id, workflow_run_id, self.backend_name,
                status=str(snapshot["status"]), tenant_id=tenant_id, metadata={"duplicate": True},
            )
        return WorkflowRunHandle(
            request_id=request_id,
            workflow_run_id=workflow_run_id,
            runner_backend=self.backend_name,
            status="queued",
            tenant_id=tracked.tenant_id,
        )

    def signal(
        self,
        request_id: str,
        signal_name: str,
        payload: dict[str, Any] | None = None,
        *,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        resolved_tenant = str(tenant_id or "default").strip() or "default"
        sig_norm = str(signal_name or "").strip().lower()
        key = (resolved_tenant, request_id)
        with self._lock:
            tracked = self._runs.get(key)
        run_id = _workflow_run_id(self.backend_name, resolved_tenant, request_id)
        if self._state_store is None:
            return {"accepted": False, "run_found": False}
        checkpoint = WorkflowCheckpoint(self._state_store, resolved_tenant, request_id, self.backend_name, run_id)
        response = checkpoint.signal(sig_norm, dict(payload or {}))
        if response["accepted"] and tracked is not None:
            with self._lock:
                tracked.signals.append({"signal": sig_norm, "payload": dict(payload or {})})
                if sig_norm == "cancel":
                    tracked.cancel_event.set()
                    tracked.status = "cancelling"
                tracked.wake_event.set()
        return response

    def status(self, request_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
        resolved_tenant = str(tenant_id or "default").strip() or "default"
        with self._lock:
            tracked = self._runs.get((resolved_tenant, request_id))
            if tracked is None:
                return _stored_workflow_status(self._state_store, self.backend_name, resolved_tenant, request_id)
            return {
                "request_id": tracked.request_id,
                "workflow_run_id": tracked.workflow_run_id,
                "runner_backend": self.backend_name,
                "tenant_id": tracked.tenant_id,
                "status": tracked.status,
                "error": tracked.error,
                "signals": list(tracked.signals),
            }

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            runs = list(self._runs.values())
        deadline = monotonic() + timeout
        for tracked in runs:
            if tracked.thread and tracked.thread.is_alive():
                tracked.cancel_event.set()
                tracked.wake_event.set()
        for tracked in runs:
            if tracked.thread:
                tracked.thread.join(timeout=max(0.0, deadline - monotonic()))


class _ExternalWorkflowRunner:
    backend_name: str

    def __init__(self, state_store: Any | None, event_bus: StreamEventBus | None) -> None:
        self._state_store = state_store
        self._event_bus = event_bus

    def _checkpoint(self, tenant_id: str, request_id: str) -> WorkflowCheckpoint:
        if self._state_store is None:
            raise WorkflowUnavailableError("External workers require a shared request snapshot store")
        return WorkflowCheckpoint(
            self._state_store, tenant_id, request_id, self.backend_name,
            _workflow_run_id(self.backend_name, tenant_id, request_id),
        )

    def signal(
        self, request_id: str, signal_name: str, payload: dict[str, Any] | None = None,
        *, tenant_id: str = "default",
    ) -> dict[str, Any]:
        return self._checkpoint(tenant_id, request_id).signal(signal_name, payload or {})

    def status(self, request_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
        return _stored_workflow_status(self._state_store, self.backend_name, tenant_id, request_id)


def _search_api_value(config: Any) -> str | None:
    value = getattr(config, "search_api", None)
    return getattr(value, "value", value)


def _output_format_value(config: Any, output_format: str | None) -> str | None:
    return str(output_format or getattr(config, "default_output_format", "") or "").strip().lower() or None


def _apply_workflow_request_options(config: Any, options: dict[str, Any] | None) -> Any:
    if not options:
        return config
    values = config.request_config_values()
    if "policy_model_mode" not in config.model_fields_set:
        values.pop("policy_model_mode", None)
    overrides = {
        config_key: options[option_key]
        for config_key, option_key in (
            ("search_api", "search_api"),
            ("default_output_format", "default_output_format"),
            ("v4_default_research_depth", "research_depth"),
        )
        if options.get(option_key) is not None
    }
    return type(config).model_validate({**values, **overrides})


class ArqWorkflowRunner(_ExternalWorkflowRunner):
    """Optional ARQ/Redis job queue runner (lazy-imports arq so base installs remain lean)."""

    backend_name = "arq"

    def __init__(
        self, *, redis_dsn: str, state_store: Any | None = None, event_bus: StreamEventBus | None = None,
    ) -> None:
        super().__init__(state_store, event_bus)
        self._redis_dsn = str(redis_dsn or "").strip()

    @staticmethod
    def _ensure_arq() -> tuple[Any, Any]:
        try:  # pragma: no cover - optional runtime dependency
            from arq import create_pool
            from arq.connections import RedisSettings
        except ImportError as exc:
            raise WorkflowUnavailableError(
                "arq package is not installed; use workflow_runner='local_durable' or install arq."
            ) from exc
        return create_pool, RedisSettings

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
        plan_first: bool = False,
        lease_id: str | None = None,
        traceparent: str | None = None,
    ) -> WorkflowRunHandle:
        del lease_id
        create_pool, RedisSettings = self._ensure_arq()
        require_shared_workflow_backends(config, self._state_store, self._event_bus)
        checkpoint = self._checkpoint(tenant_id, request_id)

        async def _enqueue() -> tuple[str, bool]:
            try:
                pool = await create_pool(RedisSettings.from_dsn(self._redis_dsn))
            except Exception as exc:
                raise WorkflowUnavailableError("Could not connect to the research job queue") from exc
            try:
                snapshot, created = checkpoint.reserve(
                    topic, plan_first=plan_first, config=config, output_format=output_format,
                )
                if snapshot["status"] in _FINISHED_STATUSES:
                    await asyncio.to_thread(_repair_terminal_submission, checkpoint, self._event_bus, snapshot)
                    return snapshot["status"], True
                job = await pool.enqueue_job(
                    "run_research_arq_job",
                    topic=topic,
                    request_id=request_id,
                    tenant_id=tenant_id,
                    search_api=_search_api_value(config),
                    output_format=output_format or getattr(config, "default_output_format", None),
                    workflow_run_id=checkpoint.run_id,
                    **({"traceparent": traceparent} if traceparent else {}),
                    _job_id=checkpoint.run_id,
                )
                return snapshot["status"], not created or job is None
            finally:
                await pool.close()

        status, duplicate = asyncio.run(_enqueue())
        return WorkflowRunHandle(
            request_id=request_id,
            workflow_run_id=checkpoint.run_id,
            runner_backend=self.backend_name,
            status=status,
            tenant_id=tenant_id,
            metadata={"duplicate": duplicate},
        )


class TemporalWorkflowRunner(_ExternalWorkflowRunner):
    """Optional Temporal workflow client runner (lazy-imports temporalio)."""

    backend_name = "temporal"

    def __init__(
        self,
        *,
        address: str = "127.0.0.1:7233",
        namespace: str = "default",
        task_queue: str = "deepresearch-tasks",
        state_store: Any | None = None,
        event_bus: StreamEventBus | None = None,
        fallback: LocalDurableWorkflowRunner | None = None,
    ) -> None:
        super().__init__(state_store, event_bus)
        self._address = str(address or "127.0.0.1:7233").strip()
        self._namespace = str(namespace or "default").strip()
        self._task_queue = str(task_queue or "deepresearch-tasks").strip()
        self._fallback = fallback

    @staticmethod
    def _ensure_temporalio() -> Any:
        try:  # pragma: no cover - optional runtime dependency
            from temporalio.client import Client
        except ImportError as exc:
            raise WorkflowUnavailableError(
                "temporalio package is not installed; use workflow_runner='local_durable' or install temporalio."
            ) from exc
        return Client

    async def _connect(self) -> Any:
        try:
            client = self._ensure_temporalio()
            return await asyncio.wait_for(
                client.connect(self._address, namespace=self._namespace), timeout=5,
            )
        except Exception as exc:
            raise WorkflowConnectionError("Could not connect to the Temporal research backend") from exc

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
        plan_first: bool = False,
        lease_id: str | None = None,
        traceparent: str | None = None,
    ) -> WorkflowRunHandle:
        if self._fallback is not None and self._fallback.status(request_id, tenant_id=tenant_id):
            return self._submit_fallback(topic, request_id, tenant_id, config, output_format, plan_first, lease_id)

        async def _start() -> tuple[str, bool]:
            client = await self._connect()
            from temporalio.common import WorkflowIDReusePolicy
            from temporalio.exceptions import WorkflowAlreadyStartedError

            require_shared_workflow_backends(config, self._state_store, self._event_bus)
            checkpoint = self._checkpoint(tenant_id, request_id)
            snapshot, created = checkpoint.reserve(
                topic, plan_first=plan_first, config=config, output_format=output_format,
                traceparent=traceparent,
            )
            if snapshot["status"] in _FINISHED_STATUSES:
                await asyncio.to_thread(_repair_terminal_submission, checkpoint, self._event_bus, snapshot)
                return snapshot["status"], True
            try:
                await client.start_workflow(
                    "DeepResearchWorkflow",
                    {
                        "topic": topic, "request_id": request_id, "tenant_id": tenant_id,
                        "workflow_run_id": checkpoint.run_id,
                        "search_api": _search_api_value(config),
                        "output_format": output_format or getattr(config, "default_output_format", None),
                        "heartbeat_seconds": getattr(config, "temporal_activity_heartbeat_seconds", 15),
                        **({"traceparent": traceparent} if traceparent else {}),
                    },
                    id=checkpoint.run_id,
                    task_queue=self._task_queue,
                    id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
                )
            except WorkflowAlreadyStartedError:
                return snapshot["status"], True
            return snapshot["status"], not created

        run_id = _workflow_run_id(self.backend_name, tenant_id, request_id)
        try:
            status, duplicate = asyncio.run(_start())
        except WorkflowConnectionError:
            if self._fallback is None:
                raise
            return self._submit_fallback(topic, request_id, tenant_id, config, output_format, plan_first, lease_id)
        return WorkflowRunHandle(
            request_id=request_id,
            workflow_run_id=run_id,
            runner_backend=self.backend_name,
            status=status,
            tenant_id=tenant_id,
            metadata={"duplicate": duplicate},
        )

    def _submit_fallback(
        self, topic: str, request_id: str, tenant_id: str, config: Any, output_format: str | None,
        plan_first: bool, lease_id: str | None,
    ) -> WorkflowRunHandle:
        handle = self._fallback.submit(
            topic, request_id=request_id, tenant_id=tenant_id, config=config,
            output_format=output_format, plan_first=plan_first, lease_id=lease_id,
        )
        return replace(handle, metadata={**handle.metadata, "degraded_reason": "temporal_unavailable_fallback_inline"})

    def signal(
        self,
        request_id: str,
        signal_name: str,
        payload: dict[str, Any] | None = None,
        *,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        if self._fallback is not None and self._fallback.status(request_id, tenant_id=tenant_id):
            return self._fallback.signal(request_id, signal_name, payload, tenant_id=tenant_id)
        response = super().signal(request_id, signal_name, payload, tenant_id=tenant_id)
        if not response["accepted"]:
            return response

        async def _send() -> None:
            client = await self._connect()
            handle = client.get_workflow_handle(_workflow_run_id(self.backend_name, tenant_id, request_id))
            await asyncio.wait_for(handle.signal(signal_name, payload or {}), timeout=5)

        try:
            asyncio.run(_send())
        except Exception:
            logger.warning("Temporal wake-up failed for %s, retaining persisted signal", request_id)
        return response

    def status(self, request_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
        status = super().status(request_id, tenant_id=tenant_id)
        if status is None and self._fallback is not None:
            return self._fallback.status(request_id, tenant_id=tenant_id)
        return status

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._fallback is not None:
            self._fallback.shutdown(timeout)


def build_workflow_runner(
    config: Any,
    *,
    agent_factory: Callable[..., Any],
    event_bus: StreamEventBus,
    state_store: Any | None = None,
    on_release: Callable[..., None] | None = None,
) -> WorkflowRunner:
    """Instantiate the configured WorkflowRunner backend."""

    backend = str(getattr(config, "workflow_runner", "inline") or "inline").strip().lower()
    local_runner = LocalDurableWorkflowRunner(
        agent_factory=agent_factory, event_bus=event_bus, state_store=state_store, on_release=on_release,
    )
    if backend == "local_durable":
        return local_runner
    if backend == "arq":
        redis_dsn = str(
            getattr(config, "arq_redis_dsn", None)
            or getattr(config, "redis_url", None)
            or "redis://127.0.0.1:6379/0"
        ).strip()
        return ArqWorkflowRunner(redis_dsn=redis_dsn, state_store=state_store, event_bus=event_bus)
    if backend == "temporal":
        return TemporalWorkflowRunner(
            address=str(getattr(config, "temporal_address", "127.0.0.1:7233") or "127.0.0.1:7233"),
            namespace=str(getattr(config, "temporal_namespace", "default") or "default"),
            task_queue=str(getattr(config, "temporal_task_queue", "deepresearch-tasks") or "deepresearch-tasks"),
            state_store=state_store,
            event_bus=event_bus,
            fallback=local_runner if getattr(config, "workflow_runner_fallback_inline", False) else None,
        )
    return InlineWorkflowRunner(background=local_runner)
