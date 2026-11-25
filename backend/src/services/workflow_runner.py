"""Durable workflow execution abstraction supporting inline, local thread, ARQ, and Temporal."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import Event, Lock, Thread
from time import monotonic, time
from typing import Any, Protocol
from uuid import uuid4

from services.event_bus import TERMINAL_EVENT_TYPES, RedisStreamEventBus, StreamEventBus
from services.mcp_bridge import tool_approval_expired
from services.request_state import SnapshotConflictError

logger = logging.getLogger(__name__)

_FINISHED_STATUSES = {"completed", "success", "partial_success", "cancelled", "rejected", "expired", "workflow_failed"}
_SIGNAL_NAMES = {"approve_plan", "approve_tool", "approve_tool_call", "cancel"}


class WorkflowUnavailableError(RuntimeError):
    """The selected workflow backend could not accept the submission."""


class WorkflowLeaseLost(RuntimeError):
    """The worker can no longer update or publish events for this request."""


class WorkflowConnectionError(WorkflowUnavailableError):
    """Temporal could not be contacted before submission."""


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

    def reserve(self, topic: str, *, plan_first: bool = False) -> tuple[dict[str, Any], bool]:
        def change(snapshot: dict[str, Any]) -> bool:
            if snapshot:
                if snapshot.get("topic") != topic or bool(snapshot.get("workflow_plan_first")) != plan_first:
                    raise ValueError("Request ID already has different research parameters")
                return False
            snapshot.update(
                topic=topic, phase="queued", status="queued", runner_backend=self.backend,
                workflow_run_id=self.run_id, workflow_plan_first=plan_first,
            )
            return True

        return self._update(change)

    def _owns(self, snapshot: dict[str, Any]) -> bool:
        lease = snapshot.get("workflow_lease") or {}
        return lease.get("owner") == self.owner and float(lease.get("expires_at") or 0) > time()

    def claim(self) -> tuple[dict[str, Any], bool]:
        def change(snapshot: dict[str, Any]) -> bool:
            if not snapshot:
                raise ValueError("Workflow snapshot is missing")
            lease = snapshot.get("workflow_lease") or {}
            if snapshot.get("status") in _FINISHED_STATUSES:
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

    def persist(self, payload: dict[str, Any]) -> dict[str, Any]:
        def change(snapshot: dict[str, Any]) -> bool:
            self.check_owner(snapshot)
            phase = payload.get("phase")
            if phase is not None and phase != snapshot.get("phase"):
                snapshot.pop("workflow_waiting_since", None)
                if phase == "awaiting_approval":
                    snapshot["workflow_plan_pending_revision"] = int(snapshot["state_version"]) + 1
                    snapshot["workflow_waiting_since"] = time()
                elif phase == "awaiting_tool_approval":
                    snapshot["workflow_waiting_since"] = time()
            snapshot.update(payload)
            return True

        snapshot, _ = self._update(change)
        return snapshot

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
                if supplied is not None and str(supplied) not in valid_revisions:
                    return False
                accepted = True
                snapshot["workflow_plan_revision"] = supplied if supplied is not None else edited_revision or revision
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
    phase = snapshot.get("phase")
    pending = snapshot.get("pending_tool_approval") or {}
    waiting_since = snapshot.get("workflow_waiting_since")
    if (
        waiting_since is not None and time() - float(waiting_since) < hitl_timeout
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
    snapshot, claimed = checkpoint.claim()

    def publish(event: dict[str, Any]) -> None:
        if checkpoint.lost.is_set():
            raise WorkflowLeaseLost(f"Execution lease expired for {request_id}")
        bus.publish(tenant_id, request_id, event)

    def finish(status: str, *, detail: str = "") -> dict[str, Any]:
        saved = checkpoint.load() or {}
        if status == "completed" and saved.get("status") in {"success", "partial_success"}:
            status = saved["status"]
        checkpoint.persist({"status": status, "phase": "completed" if status in {
            "success", "partial_success", "completed",
        } else status, **({"error": detail} if detail else {})})
        return result(status)

    def result(status: str) -> dict[str, Any]:
        return {"request_id": request_id, "tenant_id": tenant_id, "status": status}

    if not claimed:
        status = str(snapshot.get("status") or "")
        if status in _FINISHED_STATUSES:
            if float((snapshot.get("workflow_lease") or {}).get("expires_at") or 0) > time():
                return result("lease_busy")
            last_events = bus.replay_since(tenant_id, request_id)
            if not last_events or last_events[-1].payload.get("type") not in TERMINAL_EVENT_TYPES:
                event_type = "cancelled" if status == "cancelled" else "done"
                if status in {"rejected", "expired", "workflow_failed"}:
                    event_type = "error"
                if event_type == "done" and snapshot.get("report_markdown"):
                    bus.publish(tenant_id, request_id, {
                        "type": "final_report", "report": snapshot["report_markdown"],
                        **{name: snapshot[name] for name in (
                            "charts_spec", "workbench", "evidence_conflicts", "claim_verification_summary",
                        ) if name in snapshot},
                    })
                bus.publish(tenant_id, request_id, {"type": event_type, "detail": snapshot.get("error")})
            return result(status)
        return result("lease_busy")

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

        agent = agent_factory()
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
                    if event.get("type") in TERMINAL_EVENT_TYPES:
                        terminal_event = event
                        continue
                    if not cancel_event.is_set():
                        publish(event)
                    if event.get("type") == "tool_approval_required":
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
        publish(terminal_event or {"type": "done", "request_id": request_id})
        return terminal
    except WorkflowLeaseLost:
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
) -> Any:
    kwargs: dict[str, Any] = {}
    if _accepts_kwarg(agent_factory, "request_id"):
        kwargs["request_id"] = request_id
    if _accepts_kwarg(agent_factory, "tenant_id"):
        kwargs["tenant_id"] = tenant_id
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
    ) -> WorkflowRunHandle:
        if self._background is not None:
            return self._background.submit(
                topic, request_id=request_id, tenant_id=tenant_id, config=config,
                output_format=output_format, plan_first=plan_first, lease_id=lease_id,
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
    ) -> WorkflowRunHandle:
        del output_format
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

        def _worker() -> None:
            with self._lock:
                tracked.status = "running"
            try:
                while True:
                    result = run_durable_attempt(
                        checkpoint,
                        lambda: _instantiate_with_supported_kwargs(
                            self._agent_factory, config=config, request_id=request_id, tenant_id=tenant_id,
                        ),
                        self._event_bus, cancel_event,
                        hitl_timeout=getattr(config, "hitl_signal_timeout_seconds", 86400),
                    )
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
            snapshot, created = checkpoint.reserve(topic, plan_first=plan_first)
            existing = self._runs.get(key)
            if (
                existing and existing.thread and existing.thread.is_alive()
            ) or snapshot.get("status") in _FINISHED_STATUSES:
                return WorkflowRunHandle(
                    request_id, workflow_run_id, self.backend_name,
                    status=str(snapshot["status"]), tenant_id=tenant_id, metadata={"duplicate": True},
                )
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
                snapshot, created = checkpoint.reserve(topic, plan_first=plan_first)
                if snapshot["status"] in _FINISHED_STATUSES:
                    return snapshot["status"], True
                job = await pool.enqueue_job(
                    "run_research_arq_job",
                    topic=topic,
                    request_id=request_id,
                    tenant_id=tenant_id,
                    search_api=_search_api_value(config),
                    output_format=output_format or getattr(config, "default_output_format", None),
                    workflow_run_id=checkpoint.run_id,
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
    ) -> WorkflowRunHandle:
        if self._fallback is not None and self._fallback.status(request_id, tenant_id=tenant_id):
            return self._submit_fallback(topic, request_id, tenant_id, config, output_format, plan_first, lease_id)

        async def _start() -> tuple[str, bool]:
            client = await self._connect()
            from temporalio.common import WorkflowIDReusePolicy
            from temporalio.exceptions import WorkflowAlreadyStartedError

            require_shared_workflow_backends(config, self._state_store, self._event_bus)
            checkpoint = self._checkpoint(tenant_id, request_id)
            snapshot, created = checkpoint.reserve(topic, plan_first=plan_first)
            if snapshot["status"] in _FINISHED_STATUSES:
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
