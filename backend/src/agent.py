"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Semaphore, Thread
from typing import Any, Callable, Iterator

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool

from config import Configuration
from metrics import RequestTrace
from models import SummaryState, SummaryStateOutput, TodoItem
from prompts import (
    report_writer_instructions,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from services.planner import PlanningService
from services.reporter import ReportingService
from services.search import dispatch_search, prepare_research_context
from services.summarizer import SummarizationService
from services.tool_events import ToolCallTracker

logger = logging.getLogger(__name__)


class SafeHelloAgentsLLM(HelloAgentsLLM):
    """Provide safer local-vLLM response handling for sync and streaming calls."""

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Best-effort extraction of text from OpenAI-compatible response fields."""
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return "".join(SafeHelloAgentsLLM._coerce_text(item) for item in value)

        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"]
            if isinstance(value.get("content"), str):
                return value["content"]
            if isinstance(value.get("reasoning"), str):
                return value["reasoning"]
            if isinstance(value.get("reasoning_content"), str):
                return value["reasoning_content"]
            return "".join(SafeHelloAgentsLLM._coerce_text(item) for item in value.values())

        if hasattr(value, "model_dump"):
            return SafeHelloAgentsLLM._coerce_text(value.model_dump(exclude_none=True))

        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text

        return ""

    @classmethod
    def _extract_message_text(cls, payload: Any) -> tuple[str, str]:
        """Return `(content, reasoning)` from a message or delta object."""
        if payload is None:
            return "", ""

        content = cls._coerce_text(getattr(payload, "content", None))
        reasoning = cls._coerce_text(getattr(payload, "reasoning", None))

        if not reasoning:
            reasoning = cls._coerce_text(getattr(payload, "reasoning_content", None))

        if not content and hasattr(payload, "model_dump"):
            payload_dict = payload.model_dump(exclude_none=True)
            content = cls._coerce_text(payload_dict.get("content"))
            if not reasoning:
                reasoning = cls._coerce_text(
                    payload_dict.get("reasoning") or payload_dict.get("reasoning_content")
                )

        return content, reasoning

    @classmethod
    def _extract_chunk_text(cls, chunk: Any) -> tuple[str, str]:
        """Return `(content, reasoning)` from a streamed completion chunk."""
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return "", ""

        choice = choices[0]
        content, reasoning = cls._extract_message_text(getattr(choice, "delta", None))
        if content or reasoning:
            return content, reasoning

        return cls._extract_message_text(getattr(choice, "message", None))

    def _build_request_kwargs(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request kwargs shared by sync and streaming invocations."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        request_kwargs.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"temperature", "max_tokens", "stream"}
            }
        )
        return request_kwargs

    def _normalize_response_text(self, response: Any) -> str:
        """Normalize empty and non-string responses to strings."""
        if response is None:
            logger.warning(
                "LLM returned empty content; normalizing to empty string provider=%s model=%s",
                getattr(self, "provider", "unknown"),
                getattr(self, "model", "unknown"),
            )
            return ""

        if not isinstance(response, str):
            logger.warning(
                "LLM returned non-string content type=%s; coercing to string provider=%s model=%s",
                type(response).__name__,
                getattr(self, "provider", "unknown"),
                getattr(self, "model", "unknown"),
            )
            return str(response)

        return response

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Aggregate a streaming response to avoid long blocking reads with local vLLM."""
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []

        response = self._client.chat.completions.create(
            **self._build_request_kwargs(messages, **kwargs),
            stream=True,
        )

        for chunk in response:
            content, reasoning = self._extract_chunk_text(chunk)
            if content:
                visible_parts.append(content)
            elif reasoning:
                reasoning_parts.append(reasoning)

        final_text = "".join(visible_parts) or "".join(reasoning_parts)
        if not visible_parts and reasoning_parts:
            logger.info(
                "LLM response fell back to reasoning text provider=%s model=%s",
                getattr(self, "provider", "unknown"),
                getattr(self, "model", "unknown"),
            )
        return self._normalize_response_text(final_text)

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """Stream visible content, falling back to reasoning text if no answer content exists."""
        saw_visible_content = False
        buffered_reasoning: list[str] = []

        response = self._client.chat.completions.create(
            **self._build_request_kwargs(messages, **kwargs),
            stream=True,
        )

        for chunk in response:
            content, reasoning = self._extract_chunk_text(chunk)
            if content:
                saw_visible_content = True
                yield content
            elif not saw_visible_content and reasoning:
                buffered_reasoning.append(reasoning)

        if not saw_visible_content and buffered_reasoning:
            logger.info(
                "Streaming LLM response fell back to reasoning text provider=%s model=%s",
                getattr(self, "provider", "unknown"),
                getattr(self, "model", "unknown"),
            )
            for piece in buffered_reasoning:
                if piece:
                    yield piece

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Keep the upstream think API but route through the safer streaming path."""
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        yield from self.stream_invoke(messages, **kwargs)


class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow using HelloAgents."""

    def __init__(
        self,
        config: Configuration | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        """Initialise the coordinator with configuration and shared tools."""
        self.config = config or Configuration.from_env()
        self.request_id = request_id or "local-request"
        self.llm = self._init_llm()
        self._summary_slots = Semaphore(self.config.task_summary_max_concurrency)

        self.note_tool = (
            NoteTool(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )
        self.tools_registry: ToolRegistry | None = None
        if self.note_tool:
            registry = ToolRegistry()
            registry.register_tool(self.note_tool)
            self.tools_registry = registry

        self._tool_tracker = ToolCallTracker(
            self.config.notes_workspace if self.config.enable_notes else None
        )
        self._tool_event_sink_enabled = False
        self._state_lock = Lock()

        self.todo_agent = self._create_tool_aware_agent(
            name="研究规划专家",
            system_prompt=todo_planner_system_prompt.strip(),
        )
        self.report_agent = self._create_tool_aware_agent(
            name="报告撰写专家",
            system_prompt=report_writer_instructions.strip(),
        )

        self._summarizer_factory: Callable[[], ToolAwareSimpleAgent] = lambda: self._create_tool_aware_agent(  # noqa: E501
            name="任务总结专家",
            system_prompt=task_summarizer_instructions.strip(),
        )

        self.planner = PlanningService(self.todo_agent, self.config)
        self.summarizer = SummarizationService(self._summarizer_factory, self.config)
        self.reporting = ReportingService(self.report_agent, self.config)
        self._last_search_notices: list[str] = []
        self._request_trace: RequestTrace | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _init_llm(self) -> HelloAgentsLLM:
        """Instantiate HelloAgentsLLM following configuration preferences."""
        llm_kwargs: dict[str, Any] = {"temperature": 0.0}

        model_id = self.config.llm_model_id or self.config.local_llm
        if model_id:
            llm_kwargs["model"] = model_id

        provider = (self.config.llm_provider or "").strip()
        if provider:
            llm_kwargs["provider"] = provider

        if provider == "ollama":
            llm_kwargs["base_url"] = self.config.sanitized_ollama_url()
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif provider == "lmstudio":
            llm_kwargs["base_url"] = self.config.lmstudio_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
        else:
            if self.config.llm_base_url:
                llm_kwargs["base_url"] = self.config.llm_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key

        return SafeHelloAgentsLLM(**llm_kwargs)

    def _create_tool_aware_agent(self, *, name: str, system_prompt: str) -> ToolAwareSimpleAgent:
        """Instantiate a ToolAwareSimpleAgent sharing tool registry and tracker."""
        return ToolAwareSimpleAgent(
            name=name,
            llm=self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=self.tools_registry is not None,
            tool_registry=self.tools_registry,
            tool_call_listener=self._tool_tracker.record,
        )

    def _set_tool_event_sink(self, sink: Callable[[dict[str, Any]], None] | None) -> None:
        """Enable or disable immediate tool event callbacks."""
        self._tool_event_sink_enabled = sink is not None
        self._tool_tracker.set_event_sink(sink)

    def _start_request_trace(self, topic: str) -> RequestTrace:
        """Create the per-request trace collector."""
        search_api = (
            self.config.search_api.value
            if hasattr(self.config.search_api, "value")
            else str(self.config.search_api)
        )
        self._request_trace = RequestTrace(
            request_id=self.request_id,
            topic=topic,
            search_api=search_api,
            provider=self.config.llm_provider,
            model=self.config.resolved_model(),
            pricing_catalog=self.config.llm_pricing_json,
        )
        return self._request_trace

    def _request_status(self, state: SummaryState, report: str | None = None) -> str:
        """Resolve success state for the overall request."""
        if not report or not report.strip():
            return "failed"

        if not self._request_trace:
            return "success"

        has_incomplete_tasks = any(task.status in {"skipped", "failed"} for task in state.todo_items)
        if self._request_trace.fallback_count or self._request_trace.degraded_reasons or has_incomplete_tasks:
            return "partial_success"

        return "success"

    def _metrics_event(self) -> dict[str, Any] | None:
        if not self._request_trace:
            return None
        return self._request_trace.metrics_event()

    def run(self, topic: str) -> SummaryStateOutput:
        """Execute the research workflow and return the final report."""
        observer = self._start_request_trace(topic)
        state = SummaryState(research_topic=topic)
        report = ""
        try:
            planning_span = observer.start_stage("planning", scope="request")
            try:
                state.todo_items = self.planner.plan_todo_list(state, observer=observer)
                planning_span.complete(
                    status="success",
                    metadata={"task_count": len(state.todo_items)},
                )
            except Exception as exc:
                planning_span.complete(status="failed", error=exc)
                raise

            self._drain_tool_events(state)

            if not state.todo_items:
                logger.info("No TODO items generated; falling back to single task")
                observer.record_fallback("planning_returned_no_tasks")
                observer.record_degraded("fallback_task_used")
                state.todo_items = [self.planner.create_fallback_task(state)]

            observer.set_task_totals(total_tasks=len(state.todo_items))

            for task in state.todo_items:
                for _ in self._execute_task(state, task, emit_stream=False):
                    pass

            report_span = observer.start_stage("report", scope="request")
            try:
                report = self.reporting.generate_report(state, observer=observer)
                report_span.complete(
                    status="success",
                    metadata={"report_length": len(report or "")},
                )
            except Exception as exc:
                report_span.complete(status="failed", error=exc)
                raise

            self._drain_tool_events(state)
            state.structured_report = report
            state.running_summary = report
            self._persist_final_report(state, report)
            observer.attach_result(
                report_markdown=report,
                todo_items=[self._serialize_task(task) for task in state.todo_items],
                report_note_id=state.report_note_id,
                report_note_path=state.report_note_path,
            )

            status = self._request_status(state, report)
            observer.complete_request(status=status)

            return SummaryStateOutput(
                running_summary=report,
                report_markdown=report,
                todo_items=state.todo_items,
            )
        except Exception as exc:
            observer.complete_request(status="failed", error=exc)
            raise

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """Execute the workflow yielding incremental progress events."""
        observer = self._start_request_trace(topic)
        state = SummaryState(research_topic=topic)
        logger.debug("Starting streaming research: topic=%s", topic)
        yield {"type": "status", "message": "初始化研究流程"}
        try:
            planning_span = observer.start_stage(
                "planning",
                scope="request",
                metadata={"topic": topic},
            )
            yield planning_span.started_event()

            try:
                state.todo_items = self.planner.plan_todo_list(state, observer=observer)
                yield planning_span.complete(
                    status="success",
                    metadata={"task_count": len(state.todo_items)},
                )
            except Exception as exc:
                yield planning_span.complete(status="failed", error=exc)
                yield observer.metrics_event()
                observer.complete_request(status="failed", error=exc)
                raise

            metrics_event = self._metrics_event()
            if metrics_event:
                yield metrics_event

            for event in self._drain_tool_events(state, step=0):
                yield event
            if not state.todo_items:
                state.todo_items = [self.planner.create_fallback_task(state)]
                yield observer.record_fallback("planning_returned_no_tasks")
                yield observer.record_degraded("fallback_task_used")

            observer.set_task_totals(total_tasks=len(state.todo_items))

            channel_map: dict[int, dict[str, Any]] = {}
            for index, task in enumerate(state.todo_items, start=1):
                token = f"task_{task.id}"
                task.stream_token = token
                channel_map[task.id] = {"step": index, "token": token}

            yield {
                "type": "todo_list",
                "tasks": [self._serialize_task(t) for t in state.todo_items],
                "step": 0,
            }

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

            def worker(task: TodoItem, step: int) -> None:
                try:
                    enqueue(
                        {
                            "type": "task_status",
                            "task_id": task.id,
                            "status": "in_progress",
                            "title": task.title,
                            "intent": task.intent,
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
                            "note_id": task.note_id,
                            "note_path": task.note_path,
                        },
                        task=task,
                    )
                    enqueue(observer.metrics_event(), task=task)
                finally:
                    enqueue({"type": "__task_done__", "task_id": task.id})

            for task in state.todo_items:
                step = channel_map.get(task.id, {}).get("step", 0)
                thread = Thread(target=worker, args=(task, step), daemon=True)
                threads.append(thread)
                thread.start()

            active_workers = len(state.todo_items)
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

            final_step = len(state.todo_items) + 1
            report_span = observer.start_stage(
                "report",
                scope="request",
                metadata={"task_count": len(state.todo_items)},
            )
            report_started = report_span.started_event()
            report_started["step"] = final_step
            yield report_started
            try:
                report = self.reporting.generate_report(state, observer=observer)
                completed_event = report_span.complete(
                    status="success",
                    metadata={"report_length": len(report or "")},
                )
                completed_event["step"] = final_step
                yield completed_event
            except Exception as exc:
                completed_event = report_span.complete(status="failed", error=exc)
                completed_event["step"] = final_step
                yield completed_event
                yield observer.metrics_event()
                observer.complete_request(status="failed", error=exc)
                raise

            for event in self._drain_tool_events(state, step=final_step):
                yield event
            state.structured_report = report
            state.running_summary = report

            note_event = self._persist_final_report(state, report)
            if note_event:
                yield note_event

            request_status = self._request_status(state, report)
            if request_status == "partial_success":
                yield observer.record_degraded("partial_result_completed")

            observer.attach_result(
                report_markdown=report,
                todo_items=[self._serialize_task(task) for task in state.todo_items],
                report_note_id=state.report_note_id,
                report_note_path=state.report_note_path,
            )
            observer.complete_request(status=request_status)
            yield observer.metrics_event()

            yield {
                "type": "final_report",
                "report": report,
                "note_id": state.report_note_id,
                "note_path": state.report_note_path,
            }
            yield {"type": "done"}
        except Exception:
            raise

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _task_search_queries(self, state: SummaryState, task: TodoItem) -> list[str]:
        """Return progressively broader queries for a task-level search retry."""

        topic = (state.research_topic or "").strip()
        title = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip())
        intent = (task.intent or "").strip()

        candidates: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = " ".join((value or "").strip().split())
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        add(task.query or "")
        if topic and title:
            add(f"{topic} {title}")
            add(f"{topic} {title} 最新进展")
        if topic and intent:
            add(f"{topic} {intent}")
        add(topic)

        return candidates or [" ".join(part for part in [title, intent] if part).strip()]

    def _search_with_fallback_queries(
        self,
        state: SummaryState,
        task: TodoItem,
        observer: RequestTrace | None,
    ) -> tuple[dict[str, Any] | None, list[str], str | None, str, bool]:
        """Retry a task search with broader queries before marking it skipped."""

        search_result: dict[str, Any] | None = None
        answer_text: str | None = None
        backend = ""
        cache_hit = False
        notices: list[str] = []
        original_query = (task.query or "").strip()

        for attempt_index, candidate in enumerate(self._task_search_queries(state, task)):
            (
                current_result,
                current_notices,
                current_answer,
                current_backend,
                current_cache_hit,
            ) = dispatch_search(
                candidate,
                self.config,
                state.research_loop_count,
                observer=observer,
            )

            if attempt_index > 0:
                retry_notice = f"原检索词未命中，改用更宽泛检索词：{candidate}"
                if retry_notice not in notices:
                    notices.append(retry_notice)

            for notice in current_notices:
                if notice and notice not in notices:
                    notices.append(notice)

            search_result = current_result
            answer_text = current_answer
            backend = current_backend
            cache_hit = current_cache_hit
            task.query = candidate

            if current_result and current_result.get("results"):
                if attempt_index > 0 and original_query and original_query != candidate:
                    logger.info(
                        "Recovered task search with broader query task_id=%s title=%s original_query=%s fallback_query=%s",
                        task.id,
                        task.title,
                        original_query,
                        candidate,
                    )
                return search_result, notices, answer_text, backend, cache_hit

        return search_result, notices, answer_text, backend, cache_hit

    def _execute_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run search + summarization for a single task."""
        task.status = "in_progress"
        observer = self._request_trace

        search_span = (
            observer.start_stage(
                "search",
                scope="task",
                task_id=task.id,
                task_title=task.title,
                metadata={"query": task.query},
            )
            if observer
            else None
        )
        if emit_stream and search_span:
            started = search_span.started_event()
            started["step"] = step
            yield started

        try:
            search_result, notices, answer_text, backend, cache_hit = self._search_with_fallback_queries(
                state,
                task,
                observer,
            )
        except Exception as exc:
            if search_span:
                completed = search_span.complete(
                    status="failed",
                    error=exc,
                    metadata={"query": task.query},
                )
                if emit_stream:
                    completed["step"] = step
                    yield completed
                    if observer:
                        yield observer.metrics_event()
            failure_message = f"搜索失败：{exc}"
            logger.warning(
                "Task search failed task_id=%s title=%s error=%s",
                task.id,
                task.title,
                exc,
            )
            for event in self._record_task_failure(
                task,
                observer=observer,
                reason=f"task_search_failed:{task.id}",
                detail=failure_message,
                summary="搜索阶段失败，未获得可用来源。",
                emit_stream=emit_stream,
                step=step,
            ):
                yield event
            return

        self._last_search_notices = notices
        task.notices = notices

        if search_span:
            completed = search_span.complete(
                status="success",
                metadata={
                    "query": task.query,
                    "backend": backend,
                    "result_count": len((search_result or {}).get("results", [])),
                    "cache_hit": cache_hit,
                    "notice_count": len(notices),
                },
            )
            if emit_stream:
                completed["step"] = step
                yield completed
                if observer:
                    yield observer.metrics_event()

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
        else:
            self._drain_tool_events(state)

        if notices and emit_stream:
            for notice in notices:
                if notice:
                    yield {
                        "type": "status",
                        "message": notice,
                        "task_id": task.id,
                        "step": step,
                    }

        if not search_result or not search_result.get("results"):
            task.status = "skipped"
            if observer:
                observer.update_task_status_counts(skipped=1)
                degraded_event = observer.record_degraded(f"task_skipped_no_results:{task.id}")
                if emit_stream:
                    degraded_event["step"] = step
                    yield degraded_event
            if emit_stream:
                for event in self._drain_tool_events(state, step=step):
                    yield event
                yield {
                    "type": "task_status",
                    "task_id": task.id,
                    "status": "skipped",
                    "title": task.title,
                    "intent": task.intent,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                    "step": step,
                }
                if observer:
                    yield observer.metrics_event()
            else:
                self._drain_tool_events(state)
            return
        else:
            if not emit_stream:
                self._drain_tool_events(state)

        sources_summary, context = prepare_research_context(
            search_result,
            answer_text,
            self.config,
        )

        task.sources_summary = sources_summary

        with self._state_lock:
            state.web_research_results.append(context)
            state.sources_gathered.append(sources_summary)
            state.research_loop_count += 1

        summary_text: str | None = None
        summary_slots = getattr(self, "_summary_slots", None)
        if summary_slots is None:
            summary_slots = Semaphore(self.config.task_summary_max_concurrency)
            self._summary_slots = summary_slots

        with_summary_slot = summary_slots.acquire
        release_summary_slot = summary_slots.release

        with_summary_slot()
        try:
            summary_span = (
                observer.start_stage(
                    "summarization",
                    scope="task",
                    task_id=task.id,
                    task_title=task.title,
                    metadata={
                        "query": task.query,
                        "max_concurrency": self.config.task_summary_max_concurrency,
                    },
                )
                if observer
                else None
            )

            if emit_stream:
                if summary_span:
                    started = summary_span.started_event()
                    started["step"] = step
                    yield started
                for event in self._drain_tool_events(state, step=step):
                    yield event
                yield {
                    "type": "sources",
                    "task_id": task.id,
                    "latest_sources": sources_summary,
                    "raw_context": context,
                    "step": step,
                    "backend": backend,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                }

                summary_stream, summary_getter = self.summarizer.stream_task_summary(
                    state,
                    task,
                    context,
                    observer=observer,
                )
                try:
                    for event in self._drain_tool_events(state, step=step):
                        yield event
                    for chunk in summary_stream:
                        if chunk:
                            yield {
                                "type": "task_summary_chunk",
                                "task_id": task.id,
                                "content": chunk,
                                "note_id": task.note_id,
                                "step": step,
                            }
                        for event in self._drain_tool_events(state, step=step):
                            yield event
                except Exception as exc:
                    if summary_span:
                        completed = summary_span.complete(status="failed", error=exc)
                        completed["step"] = step
                        yield completed
                        if observer:
                            yield observer.metrics_event()
                    failure_message = f"总结失败：{exc}"
                    logger.warning(
                        "Task summarization failed task_id=%s title=%s error=%s",
                        task.id,
                        task.title,
                        exc,
                    )
                    for event in self._record_task_failure(
                        task,
                        observer=observer,
                        reason=f"task_summary_failed:{task.id}",
                        detail=failure_message,
                        summary="总结阶段失败，请参考已收集来源。",
                        emit_stream=emit_stream,
                        step=step,
                    ):
                        yield event
                    return
                finally:
                    summary_text = summary_getter()
                if summary_span:
                    completed = summary_span.complete(
                        status="success",
                        metadata={"summary_length": len(summary_text or "")},
                    )
                    completed["step"] = step
                    yield completed
                    if observer:
                        yield observer.metrics_event()
            else:
                try:
                    summary_text = self.summarizer.summarize_task(
                        state,
                        task,
                        context,
                        observer=observer,
                    )
                except Exception as exc:
                    if summary_span:
                        summary_span.complete(status="failed", error=exc)
                    logger.warning(
                        "Task summarization failed task_id=%s title=%s error=%s",
                        task.id,
                        task.title,
                        exc,
                    )
                    self._record_task_failure(
                        task,
                        observer=observer,
                        reason=f"task_summary_failed:{task.id}",
                        detail=f"总结失败：{exc}",
                        summary="总结阶段失败，请参考已收集来源。",
                        emit_stream=False,
                    )
                    return
                else:
                    if summary_span:
                        summary_span.complete(
                            status="success",
                            metadata={"summary_length": len(summary_text or "")},
                        )
                self._drain_tool_events(state)
        finally:
            release_summary_slot()

        task.summary = summary_text.strip() if summary_text else "暂无可用信息"
        task.status = "completed"
        if observer:
            observer.update_task_status_counts(completed=1)

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "completed",
                "summary": task.summary,
                "sources_summary": task.sources_summary,
                "note_id": task.note_id,
                "note_path": task.note_path,
                "step": step,
            }
        else:
            self._drain_tool_events(state)

    def _record_task_failure(
        self,
        task: TodoItem,
        *,
        observer: RequestTrace | None,
        reason: str,
        detail: str,
        summary: str,
        emit_stream: bool,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        """Update task state for a recoverable failure and emit stream events when needed."""
        task.status = "failed"
        task.summary = summary
        if detail and detail not in task.notices:
            task.notices.append(detail)

        events: list[dict[str, Any]] = []
        if observer:
            observer.update_task_status_counts(failed=1)
            if emit_stream:
                degraded_event = observer.record_degraded(reason)
                if step is not None:
                    degraded_event["step"] = step
                events.append(degraded_event)

        if emit_stream:
            events.append(
                {
                    "type": "task_status",
                    "task_id": task.id,
                    "status": "failed",
                    "detail": detail,
                    "summary": task.summary,
                    "sources_summary": task.sources_summary,
                    "title": task.title,
                    "intent": task.intent,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                    "step": step,
                }
            )
            if observer:
                events.append(observer.metrics_event())

        return events

    def _drain_tool_events(
        self,
        state: SummaryState,
        *,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        """Proxy to the shared tool call tracker."""
        events = self._tool_tracker.drain(state, step=step)
        if self._tool_event_sink_enabled:
            return []
        return events

    @property
    def _tool_call_events(self) -> list[dict[str, Any]]:
        """Expose recorded tool events for legacy integrations."""
        return self._tool_tracker.as_dicts()

    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        """Convert task dataclass to serializable dict for frontend."""
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "stream_token": task.stream_token,
        }

    def _persist_final_report(self, state: SummaryState, report: str) -> dict[str, Any] | None:
        if not self.note_tool or not report or not report.strip():
            return None

        note_title = f"研究报告：{state.research_topic}".strip() or "研究报告"
        tags = ["deep_research", "report"]
        content = report.strip()

        note_id = self._find_existing_report_note_id(state)
        response = ""

        if note_id:
            response = self.note_tool.run(
                {
                    "action": "update",
                    "note_id": note_id,
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            if response.startswith("❌"):
                note_id = None

        if not note_id:
            response = self.note_tool.run(
                {
                    "action": "create",
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            note_id = self._extract_note_id_from_text(response)

        if not note_id:
            return None

        state.report_note_id = note_id
        if self.config.notes_workspace:
            note_path = Path(self.config.notes_workspace) / f"{note_id}.md"
            state.report_note_path = str(note_path)
        else:
            note_path = None

        payload = {
            "type": "report_note",
            "note_id": note_id,
            "title": note_title,
            "content": content,
        }
        if note_path:
            payload["note_path"] = str(note_path)

        return payload

    def _find_existing_report_note_id(self, state: SummaryState) -> str | None:
        if state.report_note_id:
            return state.report_note_id

        for event in reversed(self._tool_tracker.as_dicts()):
            if event.get("tool") != "note":
                continue

            parameters = event.get("parsed_parameters") or {}
            if not isinstance(parameters, dict):
                continue

            action = parameters.get("action")
            if action not in {"create", "update"}:
                continue

            note_type = parameters.get("note_type")
            if note_type != "conclusion":
                title = parameters.get("title")
                if not (isinstance(title, str) and title.startswith("研究报告")):
                    continue

            note_id = parameters.get("note_id")
            if not note_id:
                note_id = self._tool_tracker._extract_note_id(event.get("result", ""))  # type: ignore[attr-defined]

            if note_id:
                return note_id

        return None

    @staticmethod
    def _extract_note_id_from_text(response: str) -> str | None:
        if not response:
            return None

        match = re.search(r"ID:\s*([^\n]+)", response)
        if not match:
            return None

        return match.group(1).strip()


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    """Convenience function mirroring the class-based API."""
    agent = DeepResearchAgent(config=config)
    return agent.run(topic)
