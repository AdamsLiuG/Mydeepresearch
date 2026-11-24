"""Durable workflow execution abstraction supporting inline, local thread, ARQ, and Temporal."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any, Protocol
from uuid import uuid4

from services.event_bus import TERMINAL_EVENT_TYPES, StreamEventBus

logger = logging.getLogger(__name__)


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
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == arg_name:
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
    try:
        return agent_factory(config=config, **kwargs)
    except TypeError:
        return agent_factory(config, **kwargs)


class InlineWorkflowRunner:
    """In-process synchronous/streaming baseline runner preserving existing direct SSE behavior."""

    backend_name = "inline"

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
    ) -> WorkflowRunHandle:
        del topic, config, output_format
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
        del request_id, tenant_id
        return None


@dataclass
class _TrackedLocalRun:
    request_id: str
    workflow_run_id: str
    tenant_id: str
    topic: str
    cancel_event: Event
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
        on_release: Callable[[str], None] | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._event_bus = event_bus
        self._state_store = state_store
        self._on_release = on_release
        self._lock = Lock()
        self._runs: dict[tuple[str, str], _TrackedLocalRun] = {}

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
    ) -> WorkflowRunHandle:
        del output_format
        workflow_run_id = f"local-{uuid4().hex[:12]}"
        cancel_event = Event()
        tracked = _TrackedLocalRun(
            request_id=request_id,
            workflow_run_id=workflow_run_id,
            tenant_id=tenant_id or "default",
            topic=topic,
            cancel_event=cancel_event,
            status="queued",
        )
        key = (tracked.tenant_id, request_id)
        with self._lock:
            self._runs[key] = tracked

        self._event_bus.publish(
            tracked.tenant_id,
            request_id,
            {
                "type": "status",
                "message": "已提交后台持久执行任务。",
                "request_id": request_id,
                "workflow_run_id": workflow_run_id,
                "runner_backend": self.backend_name,
            },
        )

        def _worker() -> None:
            with self._lock:
                tracked.status = "running"
            saw_terminal = False
            try:
                agent = _instantiate_with_supported_kwargs(
                    self._agent_factory,
                    config=config,
                    request_id=request_id,
                    tenant_id=tracked.tenant_id,
                )
                stream_fn = agent.run_stream
                iterator = (
                    stream_fn(topic, cancel_event=cancel_event)
                    if _accepts_kwarg(stream_fn, "cancel_event")
                    else stream_fn(topic)
                )
                for event in iterator:
                    if cancel_event.is_set():
                        break
                    if isinstance(event, dict):
                        self._event_bus.publish(tracked.tenant_id, request_id, event)
                        if str(event.get("type") or "").strip().lower() in TERMINAL_EVENT_TYPES:
                            saw_terminal = True

                if cancel_event.is_set():
                    with self._lock:
                        tracked.status = "cancelled"
                    if not saw_terminal:
                        self._event_bus.publish(
                            tracked.tenant_id,
                            request_id,
                            {
                                "type": "cancelled",
                                "request_id": request_id,
                                "workflow_run_id": workflow_run_id,
                            },
                        )
                else:
                    with self._lock:
                        tracked.status = "completed"
                    if not saw_terminal:
                        self._event_bus.publish(
                            tracked.tenant_id,
                            request_id,
                            {"type": "done", "request_id": request_id},
                        )
            except Exception as exc:  # pragma: no cover - defensive error boundary
                logger.exception(
                    "LocalDurableWorkflowRunner run failed request_id=%s",
                    request_id,
                )
                with self._lock:
                    tracked.status = "failed"
                    tracked.error = str(exc)
                if not saw_terminal:
                    self._event_bus.publish(
                        tracked.tenant_id,
                        request_id,
                        {
                            "type": "error",
                            "detail": f"{exc} (request_id={request_id})",
                            "request_id": request_id,
                        },
                    )
            finally:
                if self._on_release is not None:
                    try:
                        self._on_release(tracked.tenant_id)
                    except Exception:
                        pass

        thread = Thread(
            target=_worker,
            name=f"durable-run-{request_id[:8]}",
            daemon=True,
        )
        tracked.thread = thread
        thread.start()
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
            if tracked is not None:
                tracked.signals.append({"signal": sig_norm, "payload": dict(payload or {})})
                if sig_norm == "cancel":
                    tracked.cancel_event.set()
                    tracked.status = "cancelling"

        if sig_norm == "cancel":
            self._event_bus.publish(
                resolved_tenant,
                request_id,
                {
                    "type": "status",
                    "message": "收到中止任务信号。",
                    "signal": "cancel",
                    "request_id": request_id,
                },
            )
            if tracked is None:
                self._event_bus.publish(
                    resolved_tenant,
                    request_id,
                    {"type": "cancelled", "request_id": request_id},
                )
        elif sig_norm in {"approve_plan", "approve_tool"}:
            self._event_bus.publish(
                resolved_tenant,
                request_id,
                {
                    "type": "status",
                    "message": f"收到人工确认信号：{sig_norm}",
                    "signal": sig_norm,
                    "request_id": request_id,
                },
            )

        return {
            "request_id": request_id,
            "tenant_id": resolved_tenant,
            "runner_backend": self.backend_name,
            "signal_name": sig_norm,
            "accepted": True,
            "run_found": tracked is not None,
            "status": tracked.status if tracked is not None else "dispatched",
        }

    def status(self, request_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
        resolved_tenant = str(tenant_id or "default").strip() or "default"
        with self._lock:
            tracked = self._runs.get((resolved_tenant, request_id))
            if tracked is None:
                return None
            return {
                "request_id": tracked.request_id,
                "workflow_run_id": tracked.workflow_run_id,
                "runner_backend": self.backend_name,
                "tenant_id": tracked.tenant_id,
                "status": tracked.status,
                "error": tracked.error,
                "signals": list(tracked.signals),
            }


class ArqWorkflowRunner:
    """Optional ARQ/Redis job queue runner (lazy-imports arq so base installs remain lean)."""

    backend_name = "arq"

    def __init__(self, *, redis_dsn: str) -> None:
        self._redis_dsn = str(redis_dsn or "").strip()

    @staticmethod
    def _ensure_arq() -> tuple[Any, Any]:
        try:  # pragma: no cover - optional runtime dependency
            from arq import create_pool
            from arq.connections import RedisSettings
        except ImportError as exc:
            raise RuntimeError(
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
    ) -> WorkflowRunHandle:
        del config, output_format
        create_pool, RedisSettings = self._ensure_arq()

        async def _enqueue() -> str:
            pool = await create_pool(RedisSettings.from_dsn(self._redis_dsn))
            try:
                job = await pool.enqueue_job(
                    "run_research_arq_job",
                    topic=topic,
                    request_id=request_id,
                    tenant_id=tenant_id,
                    _job_id=f"deepresearch:{request_id}",
                )
                return str(getattr(job, "job_id", None) or request_id)
            finally:
                await pool.close()

        run_id = asyncio.run(_enqueue())
        return WorkflowRunHandle(
            request_id=request_id,
            workflow_run_id=run_id,
            runner_backend=self.backend_name,
            status="queued",
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
        return {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "runner_backend": self.backend_name,
            "signal_name": signal_name,
            "accepted": True,
            "payload": dict(payload or {}),
        }

    def status(self, request_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
        del request_id, tenant_id
        return None


class TemporalWorkflowRunner:
    """Optional Temporal workflow client runner (lazy-imports temporalio)."""

    backend_name = "temporal"

    def __init__(
        self,
        *,
        address: str = "127.0.0.1:7233",
        namespace: str = "default",
        task_queue: str = "deepresearch-tasks",
    ) -> None:
        self._address = str(address or "127.0.0.1:7233").strip()
        self._namespace = str(namespace or "default").strip()
        self._task_queue = str(task_queue or "deepresearch-tasks").strip()

    @staticmethod
    def _ensure_temporalio() -> Any:
        try:  # pragma: no cover - optional runtime dependency
            from temporalio.client import Client
        except ImportError as exc:
            raise RuntimeError(
                "temporalio package is not installed; use workflow_runner='local_durable' or install temporalio."
            ) from exc
        return Client

    def submit(
        self,
        topic: str,
        *,
        request_id: str,
        tenant_id: str,
        config: Any,
        output_format: str | None = None,
    ) -> WorkflowRunHandle:
        del config, output_format
        Client = self._ensure_temporalio()

        async def _start() -> str:
            client = await Client.connect(self._address, namespace=self._namespace)
            handle = await client.start_workflow(
                "DeepResearchWorkflow",
                args=[{"topic": topic, "request_id": request_id, "tenant_id": tenant_id}],
                id=f"deepresearch-{request_id}",
                task_queue=self._task_queue,
            )
            return str(getattr(handle, "first_execution_run_id", None) or handle.id)

        run_id = asyncio.run(_start())
        return WorkflowRunHandle(
            request_id=request_id,
            workflow_run_id=run_id,
            runner_backend=self.backend_name,
            status="queued",
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
        Client = self._ensure_temporalio()

        async def _send() -> None:
            client = await Client.connect(self._address, namespace=self._namespace)
            handle = client.get_workflow_handle(f"deepresearch-{request_id}")
            await handle.signal(signal_name, payload or {})

        asyncio.run(_send())
        return {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "runner_backend": self.backend_name,
            "signal_name": signal_name,
            "accepted": True,
        }

    def status(self, request_id: str, *, tenant_id: str = "default") -> dict[str, Any] | None:
        del request_id, tenant_id
        return None


def build_workflow_runner(
    config: Any,
    *,
    agent_factory: Callable[..., Any],
    event_bus: StreamEventBus,
    state_store: Any | None = None,
    on_release: Callable[[str], None] | None = None,
) -> WorkflowRunner:
    """Instantiate the configured WorkflowRunner backend."""

    backend = str(getattr(config, "workflow_runner", "inline") or "inline").strip().lower()
    if backend == "local_durable":
        return LocalDurableWorkflowRunner(
            agent_factory=agent_factory,
            event_bus=event_bus,
            state_store=state_store,
            on_release=on_release,
        )
    if backend == "arq":
        redis_dsn = str(
            getattr(config, "arq_redis_dsn", None)
            or getattr(config, "redis_url", None)
            or "redis://127.0.0.1:6379/0"
        ).strip()
        return ArqWorkflowRunner(redis_dsn=redis_dsn)
    if backend == "temporal":
        return TemporalWorkflowRunner(
            address=str(getattr(config, "temporal_address", "127.0.0.1:7233") or "127.0.0.1:7233"),
            namespace=str(getattr(config, "temporal_namespace", "default") or "default"),
            task_queue=str(getattr(config, "temporal_task_queue", "deepresearch-tasks") or "deepresearch-tasks"),
        )
    return InlineWorkflowRunner()
