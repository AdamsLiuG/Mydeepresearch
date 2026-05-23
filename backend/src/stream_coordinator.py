"""Streaming task coordination helpers for the deep research workflow."""

from __future__ import annotations

import logging
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

from models import SummaryState, TodoItem
from services.evidence import extract_citation_ids
from services.summarizer import TaskSummaryResult
from services.text_processing import strip_citation_markers

logger = logging.getLogger(__name__)


class StreamCoordinatorMixin:
    """Assign stream channels and multiplex task execution events."""

    def _assign_stream_channels(
        self,
        tasks: list[TodoItem],
        channel_map: dict[int, dict[str, Any]],
        *,
        start_step: int,
    ) -> int:
        """Assign step and stream-token metadata for streamable tasks."""

        step = start_step
        for task in tasks:
            token = task.stream_token or f"task_{task.id}"
            task.stream_token = token
            channel_map[task.id] = {"step": step, "token": token}
            step += 1
        return step

    @staticmethod
    def _coerce_summary_result(value: Any) -> TaskSummaryResult:
        """Normalize legacy summary strings into the structured summary result shape."""

        if isinstance(value, TaskSummaryResult):
            return value

        markdown = str(value or "").strip() or "暂无可用信息"
        key_findings: list[dict[str, Any]] = []
        evidence_gaps: list[str] = []
        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            citation_ids = extract_citation_ids(line)
            text = strip_citation_markers(line).strip("-* ").strip()
            if citation_ids and text:
                key_findings.append({"text": text, "source_ids": citation_ids})
                continue
            if any(token in text for token in ("证据不足", "暂无可用信息", "待补充")) and text:
                evidence_gaps.append(text)

        payload = {"key_findings": key_findings, "evidence_gaps": evidence_gaps}
        claims = [
            {
                "text": item["text"],
                "source_ids": list(item["source_ids"]),
                "support_status": "unreviewed",
            }
            for item in key_findings
        ]
        return TaskSummaryResult(markdown=markdown, payload=payload, claims=claims)

    def _execute_task_batch_stream(
        self,
        state: SummaryState,
        tasks: list[TodoItem],
        channel_map: dict[int, dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Execute a batch of tasks for the streaming request path."""

        observer = self._request_trace
        if not tasks or observer is None:
            return

        self._warm_topic_search_cache(state, observer=observer)
        event_queue: Queue[dict[str, Any]] = Queue()

        def enqueue(
            event: dict[str, Any],
            *,
            task: TodoItem | None = None,
            step_override: int | None = None,
        ) -> None:
            payload = dict(event)
            target_task_id = payload.get("task_id")
            if task is not None:
                target_task_id = task.id
                payload["task_id"] = task.id

            channel = channel_map.get(target_task_id) if target_task_id is not None else None
            if channel:
                payload.setdefault("step", channel["step"])
                payload["stream_token"] = channel["token"]
            if step_override is not None:
                payload["step"] = step_override
            event_queue.put(payload)

        def tool_event_sink(event: dict[str, Any]) -> None:
            enqueue(event)

        self._set_tool_event_sink(tool_event_sink)
        threads: list[Thread] = []

        def worker(task: TodoItem) -> None:
            step = channel_map.get(task.id, {}).get("step", 0)
            try:
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "in_progress",
                        "title": task.title,
                        "intent": task.intent,
                        "query": task.query,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )
                for event in self._execute_task(state, task, emit_stream=True, step=step):
                    enqueue(event, task=task)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Task execution failed", exc_info=exc)
                task.status = "failed"
                observer.update_task_status_counts(failed=1)
                enqueue(observer.record_degraded(f"task_failed:{task.id}"), task=task)
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "failed",
                        "detail": str(exc),
                        "title": task.title,
                        "intent": task.intent,
                        "query": task.query,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                        "notices": list(task.notices),
                    },
                    task=task,
                )
                enqueue(observer.metrics_event(), task=task)
            finally:
                enqueue({"type": "__task_done__", "task_id": task.id})

        for task in tasks:
            thread = Thread(target=worker, args=(task,), daemon=True)
            threads.append(thread)
            thread.start()

        active_workers = len(tasks)
        finished_workers = 0

        try:
            while finished_workers < active_workers:
                event = event_queue.get()
                if event.get("type") == "__task_done__":
                    finished_workers += 1
                    continue
                yield event

            while True:
                try:
                    event = event_queue.get_nowait()
                except Empty:
                    break
                if event.get("type") != "__task_done__":
                    yield event
        finally:
            self._set_tool_event_sink(None)
            for thread in threads:
                thread.join()
