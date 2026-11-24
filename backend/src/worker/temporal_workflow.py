"""Temporal & ARQ workflow activity definitions with lazy optional SDK imports."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from config import Configuration
from services.event_bus import build_stream_event_bus

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


def execute_durable_research_activity(spec: DeepResearchWorkflowSpec) -> dict[str, Any]:
    """Run one complete research job inside a background worker and publish events to StreamEventBus."""

    from agent import DeepResearchAgent

    config = Configuration.from_env()
    event_bus = build_stream_event_bus(config)
    agent = DeepResearchAgent(
        config=config,
        request_id=spec.request_id,
        tenant_id=spec.tenant_id,
    )
    final_payload: dict[str, Any] = {}
    for event in agent.run_stream(spec.topic):
        if isinstance(event, dict):
            event_bus.publish(spec.tenant_id, spec.request_id, event)
            if event.get("type") == "final_report":
                final_payload = event
    return {
        "request_id": spec.request_id,
        "tenant_id": spec.tenant_id,
        "status": "completed",
        "has_report": bool(final_payload.get("report")),
    }


async def run_research_arq_job(ctx: Any, *, topic: str, request_id: str, tenant_id: str = "default") -> dict[str, Any]:
    """ARQ worker entrypoint wrapping the synchronous research activity in a worker thread."""

    del ctx
    spec = DeepResearchWorkflowSpec(topic=topic, request_id=request_id, tenant_id=tenant_id)
    return await asyncio.to_thread(execute_durable_research_activity, spec)


def build_temporal_worker_definitions() -> tuple[Any, list[Any]]:
    """Return Temporal Workflow/Activity classes if temporalio is installed."""

    try:  # pragma: no cover - optional runtime dependency
        from datetime import timedelta

        from temporalio import activity, workflow
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("temporalio package is not installed") from exc

    @activity.defn(name="run_deepresearch_request")
    async def run_deepresearch_request(payload: dict[str, Any]) -> dict[str, Any]:
        spec = DeepResearchWorkflowSpec(
            topic=str(payload.get("topic") or ""),
            request_id=str(payload.get("request_id") or ""),
            tenant_id=str(payload.get("tenant_id") or "default"),
        )
        return await asyncio.to_thread(execute_durable_research_activity, spec)

    @workflow.defn(name="DeepResearchWorkflow")
    class DeepResearchWorkflow:
        def __init__(self) -> None:
            self._signals: list[tuple[str, dict[str, Any]]] = []
            self._cancelled = False

        @workflow.signal(name="cancel")
        def cancel(self, payload: dict[str, Any] | None = None) -> None:
            self._cancelled = True
            self._signals.append(("cancel", dict(payload or {})))

        @workflow.signal(name="approve_plan")
        def approve_plan(self, payload: dict[str, Any] | None = None) -> None:
            self._signals.append(("approve_plan", dict(payload or {})))

        @workflow.signal(name="approve_tool")
        def approve_tool(self, payload: dict[str, Any] | None = None) -> None:
            self._signals.append(("approve_tool", dict(payload or {})))

        @workflow.run
        async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
            if self._cancelled:
                return {"status": "cancelled", "request_id": payload.get("request_id")}
            return await workflow.execute_activity(
                run_deepresearch_request,
                payload,
                start_to_close_timeout=timedelta(minutes=30),
            )

    return DeepResearchWorkflow, [run_deepresearch_request]
