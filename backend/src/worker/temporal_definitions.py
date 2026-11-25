"""Temporal definitions loaded only by the optional worker profile."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from threading import Event

    from worker.temporal_workflow import (
        DeepResearchWorkflowSpec,
        execute_durable_research_activity,
    )


@activity.defn(name="run_deepresearch_request")
async def run_deepresearch_request(payload: dict[str, Any]) -> dict[str, Any]:
    spec = DeepResearchWorkflowSpec(
        topic=payload["topic"], request_id=payload["request_id"], tenant_id=payload["tenant_id"],
        search_api=payload.get("search_api"), output_format=payload.get("output_format"),
        workflow_run_id=payload.get("workflow_run_id"),
    )
    stop_event = Event()
    task = asyncio.create_task(asyncio.to_thread(
        execute_durable_research_activity, spec, failure=payload.get("failure"), stop_event=stop_event,
    ))
    try:
        while not task.done():
            activity.heartbeat()
            await asyncio.wait({task}, timeout=min(3, float(payload.get("heartbeat_seconds", 15)) / 3))
        return await task
    except asyncio.CancelledError:
        stop_event.set()
        await asyncio.gather(task, return_exceptions=True)
        raise


@workflow.defn(name="DeepResearchWorkflow")
class DeepResearchWorkflow:
    def __init__(self) -> None:
        self._signal_version = 0

    @workflow.signal(name="cancel")
    def cancel(self, payload: dict[str, Any]) -> None:
        self._signal_version += 1

    @workflow.signal(name="approve_plan")
    def approve_plan(self, payload: dict[str, Any]) -> None:
        self._signal_version += 1

    @workflow.signal(name="approve_tool")
    def approve_tool(self, payload: dict[str, Any]) -> None:
        self._signal_version += 1

    @workflow.signal(name="approve_tool_call")
    def approve_tool_call(self, payload: dict[str, Any]) -> None:
        self._signal_version += 1

    @workflow.run
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        failure_payload = None
        while True:
            signal_version = self._signal_version
            try:
                result = await workflow.execute_activity(
                    "run_deepresearch_request",
                    failure_payload or payload,
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(seconds=float(payload.get("heartbeat_seconds", 15))),
                    retry_policy=RetryPolicy(initial_interval=timedelta(seconds=2), maximum_attempts=3),
                )
            except ActivityError:
                if failure_payload is not None:
                    raise
                failure_payload = {**payload, "failure": "Research worker retries exhausted"}
                continue
            if result["status"] not in {"awaiting_approval", "lease_busy"}:
                return result
            if workflow.info().is_continue_as_new_suggested():
                workflow.continue_as_new(failure_payload or payload)
            try:
                await workflow.wait_condition(
                    lambda: self._signal_version != signal_version,
                    timeout=timedelta(seconds=60 if result["status"] == "awaiting_approval" else 2),
                )
            except asyncio.TimeoutError:
                pass
