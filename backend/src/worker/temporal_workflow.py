"""Temporal & ARQ workflow activity definitions with lazy optional SDK imports."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from threading import Event
from typing import Any

logger = logging.getLogger(__name__)

WORKFLOW_STAGES = ("planning", "execution", "reflection", "review", "reporting")


@dataclass(frozen=True)
class DeepResearchWorkflowSpec:
    """Serializable workflow input descriptor shared across Temporal and ARQ workers."""

    topic: str
    request_id: str
    tenant_id: str = "default"
    search_api: str | None = None
    output_format: str | None = None
    workflow_run_id: str | None = None


def execute_durable_research_activity(
    spec: DeepResearchWorkflowSpec,
    *,
    backend: str = "temporal",
    failure: str | None = None,
    stop_event: Event | None = None,
) -> dict[str, Any]:
    """Execute or resume one persisted run, returning only control metadata."""

    from agent import DeepResearchAgent
    from config import Configuration
    from services.event_bus import build_stream_event_bus
    from services.storage_backends import build_request_state_store
    from services.workflow_runner import (
        WorkflowCheckpoint,
        _apply_workflow_request_options,
        _workflow_run_id,
        require_shared_workflow_backends,
        run_durable_attempt,
    )

    config = Configuration.from_env()
    overrides = {
        name: value for name, value in {
            "search_api": spec.search_api, "default_output_format": spec.output_format,
        }.items() if value is not None
    }
    if overrides:
        values = config.model_dump()
        if "policy_model_mode" not in config.model_fields_set:
            values.pop("policy_model_mode", None)
        config = Configuration.model_validate({**values, **overrides})
    store = build_request_state_store(config)
    if store is None:
        raise RuntimeError("Durable research workers require request snapshots")
    checkpoint = WorkflowCheckpoint(
        store, spec.tenant_id, spec.request_id, backend,
        spec.workflow_run_id or _workflow_run_id(backend, spec.tenant_id, spec.request_id),
        lease_seconds=max(5, getattr(config, "temporal_activity_heartbeat_seconds", 15) * 2),
    )
    event_bus = None
    try:
        event_bus = build_stream_event_bus(config)
        require_shared_workflow_backends(config, store, event_bus)
        if checkpoint.load() is None:
            checkpoint.reserve(spec.topic)

        def agent_factory(*, workflow_request_options: dict[str, Any] | None = None) -> DeepResearchAgent:
            if failure is not None:
                raise RuntimeError(failure)
            return DeepResearchAgent(
                config=_apply_workflow_request_options(config, workflow_request_options),
                request_id=spec.request_id, tenant_id=spec.tenant_id,
            )

        return run_durable_attempt(
            checkpoint, agent_factory, event_bus, Event(),
            hitl_timeout=getattr(config, "hitl_signal_timeout_seconds", 86400),
            retry_error=failure is None, stop_event=stop_event,
        )
    finally:
        for resource in (event_bus, store):
            close = getattr(resource, "close", None)
            if close is not None:
                close()


async def run_research_arq_job(
    ctx: Any, *, topic: str, request_id: str, tenant_id: str = "default",
    search_api: str | None = None, output_format: str | None = None, workflow_run_id: str | None = None,
) -> dict[str, Any]:
    """Resume a queued request and yield to ARQ while waiting for human approval."""

    from arq import Retry

    spec = DeepResearchWorkflowSpec(topic, request_id, tenant_id, search_api, output_format, workflow_run_id)
    stop_event = Event()
    while True:
        task = asyncio.create_task(asyncio.to_thread(
            execute_durable_research_activity, spec, backend="arq", stop_event=stop_event,
        ))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            stop_event.set()
            await asyncio.gather(task, return_exceptions=True)
            raise
        except Exception:
            if ctx.get("job_try", 1) < 3:
                raise Retry(defer=2 ** ctx.get("job_try", 1)) from None
            return await asyncio.to_thread(
                execute_durable_research_activity, spec, backend="arq", failure="Research worker retries exhausted",
            )
        if result["status"] not in {"awaiting_approval", "lease_busy"}:
            return result
        await asyncio.sleep(2)


def build_temporal_worker_definitions() -> tuple[Any, list[Any]]:
    """Import the SDK-backed definitions only when a Temporal worker starts."""
    try:  # pragma: no cover - optional runtime dependency
        from worker.temporal_definitions import (
            DeepResearchWorkflow,
            run_deepresearch_request,
        )
    except ImportError as exc:
        raise RuntimeError("temporalio package is not installed") from exc
    return DeepResearchWorkflow, [run_deepresearch_request]
