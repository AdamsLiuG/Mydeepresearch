"""Deterministic benchmark stub agent used by smoke and CI perf runs."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Iterator

from metrics import RequestTrace


class BenchmarkStubAgent:
    """Predictable agent implementation used for benchmark and CI smoke runs."""

    def __init__(self, config: Any, request_id: str | None = None) -> None:
        self.config = config
        self.request_id = request_id or "perf-stub"
        self._request_trace: RequestTrace | None = None

    def run(self, topic: str) -> SimpleNamespace:
        trace = self._new_trace(topic)
        todo_item = self._build_todo_item(topic)
        report_markdown = self._build_report(topic)

        self._simulate_stage(trace, "planning", scope="request", metadata={"task_count": 1})
        self._simulate_stage(
            trace,
            "search",
            scope="task",
            task_id=todo_item.id,
            task_title=todo_item.title,
            metadata={"backend": "benchmark_stub"},
        )
        trace.record_search_attempt(cache_hit=False, success=True)
        self._simulate_stage(
            trace,
            "summarization",
            scope="task",
            task_id=todo_item.id,
            task_title=todo_item.title,
            metadata={"summary_kind": "stub"},
        )
        trace.record_llm_call(
            success=True,
            prompt_text=f"benchmark prompt for {topic}",
            completion_text=report_markdown,
        )
        self._simulate_stage(trace, "report", scope="request", metadata={"section_count": 3})

        trace.set_task_totals(total_tasks=1)
        trace.update_task_status_counts(completed=1)
        trace.attach_result(report_markdown=report_markdown, todo_items=[self._todo_payload(todo_item)])
        trace.complete_request(status="success")

        return SimpleNamespace(
            report_markdown=report_markdown,
            running_summary=report_markdown,
            todo_items=[todo_item],
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        trace = self._new_trace(topic)
        todo_item = self._build_todo_item(topic)
        report_markdown = self._build_report(topic)

        yield {"type": "status", "message": "benchmark stub started"}

        planning_events = self._simulate_stage_with_events(
            trace,
            "planning",
            scope="request",
            metadata={"task_count": 1},
        )
        yield planning_events[0]
        yield planning_events[1]
        yield trace.metrics_event()

        yield {
            "type": "todo_list",
            "tasks": [self._todo_payload(todo_item)],
            "step": 0,
        }

        search_events = self._simulate_stage_with_events(
            trace,
            "search",
            scope="task",
            task_id=todo_item.id,
            task_title=todo_item.title,
            metadata={"backend": "benchmark_stub"},
        )
        yield search_events[0]
        trace.record_search_attempt(cache_hit=False, success=True)
        yield search_events[1]
        yield trace.metrics_event()

        summarization_events = self._simulate_stage_with_events(
            trace,
            "summarization",
            scope="task",
            task_id=todo_item.id,
            task_title=todo_item.title,
            metadata={"summary_kind": "stub"},
        )
        yield summarization_events[0]
        trace.record_llm_call(
            success=True,
            prompt_text=f"benchmark prompt for {topic}",
            completion_text=report_markdown,
        )
        yield summarization_events[1]
        yield trace.metrics_event()

        report_events = self._simulate_stage_with_events(
            trace,
            "report",
            scope="request",
            metadata={"section_count": 3},
        )
        yield report_events[0]
        yield report_events[1]

        trace.set_task_totals(total_tasks=1)
        trace.update_task_status_counts(completed=1)
        trace.attach_result(report_markdown=report_markdown, todo_items=[self._todo_payload(todo_item)])
        trace.complete_request(status="success")

        yield trace.metrics_event()
        yield {"type": "final_report", "report": report_markdown}
        yield {"type": "done"}

    def _new_trace(self, topic: str) -> RequestTrace:
        search_api = getattr(getattr(self.config, "search_api", None), "value", None) or str(
            getattr(self.config, "search_api", "benchmark_stub")
        )
        trace = RequestTrace(
            request_id=self.request_id,
            topic=topic,
            search_api=search_api,
            provider=getattr(self.config, "llm_provider", "benchmark_stub"),
            model=getattr(self.config, "resolved_model", lambda: "benchmark-stub")(),
            pricing_catalog=getattr(self.config, "llm_pricing_json", {}),
        )
        self._request_trace = trace
        return trace

    def _build_todo_item(self, topic: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=1,
            title="Benchmark Stub Task",
            intent="Validate engineering benchmark path",
            query=topic,
            status="completed",
            summary="Deterministic benchmark summary.",
            sources_summary="* Benchmark Stub Source : https://example.com/benchmark-stub",
            note_id=None,
            note_path=None,
            origin="planned",
            round=1,
        )

    def _todo_payload(self, todo_item: SimpleNamespace) -> dict[str, Any]:
        return {
            "id": todo_item.id,
            "title": todo_item.title,
            "intent": todo_item.intent,
            "query": todo_item.query,
            "status": todo_item.status,
            "summary": todo_item.summary,
            "sources_summary": todo_item.sources_summary,
            "note_id": todo_item.note_id,
            "note_path": todo_item.note_path,
            "origin": getattr(todo_item, "origin", "planned"),
            "round": getattr(todo_item, "round", 1),
        }

    def _build_report(self, topic: str) -> str:
        return (
            "# 背景\n"
            f"{topic}\n\n"
            "## 关键发现\n"
            "- 该请求由 deterministic benchmark stub 返回。\n"
            "- 用于 smoke benchmark、regression baseline 和 locust 压测验证。\n\n"
            "## 结论\n"
            "https://example.com/benchmark-stub"
        )

    def _simulate_stage(
        self,
        trace: RequestTrace,
        stage: str,
        *,
        scope: str,
        task_id: int | None = None,
        task_title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        span = trace.start_stage(
            stage,
            scope=scope,
            task_id=task_id,
            task_title=task_title,
            metadata=metadata,
        )
        time.sleep(0.01)
        span.complete(status="success", metadata=metadata)

    def _simulate_stage_with_events(
        self,
        trace: RequestTrace,
        stage: str,
        *,
        scope: str,
        task_id: int | None = None,
        task_title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        span = trace.start_stage(
            stage,
            scope=scope,
            task_id=task_id,
            task_title=task_title,
            metadata=metadata,
        )
        started = span.started_event()
        time.sleep(0.01)
        completed = span.complete(status="success", metadata=metadata)
        return started, completed
