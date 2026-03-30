"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Semaphore, Thread
from time import sleep
from typing import Any, Callable, Iterator

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool

from config import Configuration
from metrics import RequestTrace
from models import SummaryState, SummaryStateOutput, TodoItem
from prompts import (
    report_writer_instructions,
    request_reflection_system_prompt,
    request_reviewer_system_prompt,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from services.evidence import (
    EvidenceLookupTool,
    EvidenceStore,
    FetchPageTool,
    SearchWebTool,
    build_task_context,
    extract_citation_ids,
    format_evidence_sources,
)
from services.planner import PlanningService
from services.reflection import ReflectionAssessment, ReflectionService
from services.reporter import ReportingService
from services.request_state import RequestStateStore
from services.reviewer import ReviewService
from services.search import dispatch_search
from services.summarizer import SummarizationService, TaskSummaryResult
from services.text_processing import strip_citation_markers
from services.tool_events import ToolCallTracker

logger = logging.getLogger(__name__)


_LEADING_SEARCH_QUERY_PATTERNS = (
    re.compile(
        r"^(?:请问|请你|请先|请帮我|请帮忙|请|麻烦你|麻烦|帮我|帮忙)\s*"
        r"(?:简要|简单|简明|详细|系统地|系统性地|深入地)?\s*"
        r"(?:研究|分析|介绍|说明|梳理|总结|评估|对比|调研|看看|看下|看一看)"
        r"(?:一下|一遍|一轮)?[:：,，、-]*\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:简要|简单|简明|详细|系统地|系统性地|深入地)\s*"
        r"(?:研究|分析|介绍|说明|梳理|总结|评估|对比|调研)"
        r"(?:一下|一遍|一轮)?[:：,，、-]*\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please|can you|could you|would you|help me)\s+"
        r"(?:(?:briefly|quickly)\s+)?"
        r"(?:research|analyze|analyse|explain|summarize|summarise|compare)\s*[:,-]*\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:briefly|quickly)\s+"
        r"(?:research|analyze|analyse|explain|summarize|summarise|compare)\s*[:,-]*\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:research|analyze|analyse|explain|summarize|summarise|compare)\s*[:,-]\s*",
        re.IGNORECASE,
    ),
)


class ToolExecutionTimeoutError(TimeoutError):
    """Raised when a guarded external tool invocation exceeds its timeout budget."""


class SafeHelloAgentsLLM(HelloAgentsLLM):
    """Provide safer local-vLLM response handling for sync and streaming calls."""

    def __init__(self, *args: Any, allow_reasoning_fallback: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_reasoning_fallback = allow_reasoning_fallback

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
            elif reasoning and self.allow_reasoning_fallback:
                reasoning_parts.append(reasoning)

        final_text = "".join(visible_parts) or "".join(reasoning_parts)
        if not visible_parts and reasoning_parts and self.allow_reasoning_fallback:
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
            elif self.allow_reasoning_fallback and not saw_visible_content and reasoning:
                buffered_reasoning.append(reasoning)

        if not saw_visible_content and buffered_reasoning and self.allow_reasoning_fallback:
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
        self._content_only_llm = self._init_llm(allow_reasoning_fallback=False)
        self._summary_slots = Semaphore(self.config.task_summary_max_concurrency)
        self._evidence_store = EvidenceStore(
            freshness_reference_days=self.config.freshness_reference_days,
        )
        self._request_state_store = (
            RequestStateStore(
                self.config.request_state_dir,
                recent_limit=self.config.request_state_recent_limit,
            )
            if self.config.request_state_enabled
            else None
        )

        self.note_tool = (
            NoteTool(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )
        self.tools_registry: ToolRegistry | None = ToolRegistry()
        self.tools_registry.register_tool(
            SearchWebTool(
                config=self.config,
                evidence_store=self._evidence_store,
                observer_getter=lambda: self._request_trace,
            )
        )
        self.tools_registry.register_tool(
            FetchPageTool(
                evidence_store=self._evidence_store,
                timeout_seconds=float(self.config.search_tool_timeout_seconds or 10.0),
            )
        )
        self.tools_registry.register_tool(
            EvidenceLookupTool(evidence_store=self._evidence_store)
        )
        if self.note_tool:
            self.tools_registry.register_tool(self.note_tool)

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
            llm=self._content_only_llm,
        )
        self.reflection_agent = self._create_tool_aware_agent(
            name="研究覆盖评估专家",
            system_prompt=request_reflection_system_prompt.strip(),
        )
        self.review_agent = self._create_tool_aware_agent(
            name="研究质量审查专家",
            system_prompt=request_reviewer_system_prompt.strip(),
            llm=self._content_only_llm,
        )

        self._summarizer_factory: Callable[[], ToolAwareSimpleAgent] = lambda: self._create_tool_aware_agent(  # noqa: E501
            name="任务总结专家",
            system_prompt=task_summarizer_instructions.strip(),
            llm=self._content_only_llm,
        )

        self.planner = PlanningService(self.todo_agent, self.config)
        self.reflection = ReflectionService(self.reflection_agent, self.config)
        self.reviewer = ReviewService(
            self.review_agent if self.config.review_agent_enabled else None,
            self.config,
        )
        self.summarizer = SummarizationService(
            self._summarizer_factory,
            self.config,
            evidence_store=self._evidence_store,
        )
        self.reporting = ReportingService(
            self.report_agent,
            self.config,
            evidence_store=self._evidence_store,
        )
        self._last_search_notices: list[str] = []
        self._request_trace: RequestTrace | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _init_llm(self, *, allow_reasoning_fallback: bool = True) -> HelloAgentsLLM:
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

        llm_kwargs["allow_reasoning_fallback"] = allow_reasoning_fallback
        return SafeHelloAgentsLLM(**llm_kwargs)

    def _create_tool_aware_agent(
        self,
        *,
        name: str,
        system_prompt: str,
        llm: HelloAgentsLLM | None = None,
    ) -> ToolAwareSimpleAgent:
        """Instantiate a ToolAwareSimpleAgent sharing tool registry and tracker."""
        return ToolAwareSimpleAgent(
            name=name,
            llm=llm or self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=self.config.use_tool_calling and self.tools_registry is not None,
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
        review_blocked = str(state.review_summary.get("overall_status") or "").strip() == "blocked"
        if (
            self._request_trace.fallback_count
            or self._request_trace.degraded_reasons
            or has_incomplete_tasks
            or review_blocked
        ):
            return "partial_success"

        return "success"

    def _metrics_event(self) -> dict[str, Any] | None:
        if not self._request_trace:
            return None
        return self._request_trace.metrics_event()

    def _restore_task_counters(self, observer: RequestTrace, state: SummaryState) -> None:
        """Replay persisted task counts into a fresh request trace."""

        observer.set_task_totals(total_tasks=len(state.todo_items))
        observer.update_task_status_counts(
            completed=sum(1 for task in state.todo_items if task.status == "completed"),
            skipped=sum(1 for task in state.todo_items if task.status == "skipped"),
            failed=sum(1 for task in state.todo_items if task.status == "failed"),
        )

    def _persist_request_state(
        self,
        state: SummaryState,
        *,
        phase: str,
        status: str = "in_progress",
        report_markdown: str | None = None,
    ) -> None:
        """Persist the latest request snapshot for history and resume."""

        if self._request_state_store is None:
            return

        observer_snapshot = self._request_trace.snapshot() if self._request_trace else {}
        payload = {
            "snapshot_version": 1,
            "request_id": self.request_id,
            "topic": state.research_topic,
            "phase": phase,
            "status": status,
            "search_api": observer_snapshot.get("search_api"),
            "elapsed_ms": observer_snapshot.get("elapsed_ms"),
            "report_markdown": report_markdown or state.structured_report or state.running_summary or "",
            "report_note_id": state.report_note_id,
            "report_note_path": state.report_note_path,
            "todo_items": [self._serialize_task(task) for task in state.todo_items],
            "review_summary": dict(state.review_summary or {}),
            "reflection_completed": bool(state.reflection_completed),
            "review_completed": bool(state.review_completed),
            "request_metrics": observer_snapshot,
        }
        self._request_state_store.save(self.request_id, payload)

    def _load_resume_snapshot(self, request_id: str) -> dict[str, Any]:
        if self._request_state_store is None:
            raise ValueError("request_state store is disabled")
        payload = self._request_state_store.load(request_id)
        if not payload:
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
        )

    def _state_from_snapshot(self, payload: dict[str, Any]) -> tuple[SummaryState, str]:
        """Restore SummaryState and phase from a persisted snapshot."""

        todo_items = []
        for item in payload.get("todo_items") or []:
            if isinstance(item, dict):
                todo_items.append(self._task_from_payload(item))

        report_markdown = str(payload.get("report_markdown") or "").strip() or None
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
        )
        self._evidence_store.hydrate_from_tasks(state.todo_items)
        for task in state.todo_items:
            if not task.sources_summary and task.id > 0:
                evidence_items = self._evidence_store.list_task_evidence(task.id)
                if evidence_items:
                    task.evidence_items = evidence_items
                    task.sources_summary = format_evidence_sources(evidence_items)
        return state, str(payload.get("phase") or "planning").strip() or "planning"

    def _execute_task_batch_sync(
        self,
        state: SummaryState,
        tasks: list[TodoItem],
    ) -> None:
        """Execute a task batch for the synchronous request path."""

        for task in tasks:
            for _ in self._execute_task(state, task, emit_stream=False):
                pass

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

    def _reflection_gap_signals(self, state: SummaryState) -> list[str]:
        """Return request-level signals that justify running reflection."""

        signals: list[str] = []
        observer = self._request_trace

        if observer and (observer.fallback_count or observer.degraded_reasons):
            signals.append("request_has_fallback_or_degraded")

        for task in state.todo_items:
            if task.status == "failed":
                signals.append(f"task_{task.id}_failed")
            elif task.status == "skipped":
                signals.append(f"task_{task.id}_skipped")

            if task.status == "completed":
                summary = (task.summary or "").strip()
                sources = (task.sources_summary or "").strip()
                if not summary or summary == "暂无可用信息":
                    signals.append(f"task_{task.id}_summary_missing")
                if not sources:
                    signals.append(f"task_{task.id}_sources_missing")

        deduped: list[str] = []
        seen: set[str] = set()
        for signal in signals:
            if signal in seen:
                continue
            seen.add(signal)
            deduped.append(signal)
        return deduped

    def _remaining_task_budget(self, state: SummaryState) -> int:
        """Return how many additional tasks can still be appended."""

        max_tasks = max(int(self.config.max_agent_tasks or 1), 1)
        return max(0, max_tasks - len(state.todo_items))

    def _run_reflection_cycle(
        self,
        state: SummaryState,
        observer: RequestTrace,
    ) -> tuple[list[TodoItem], str | None, ReflectionAssessment | None, list[str]]:
        """Run the shared reflection/replan logic without emitting stream events."""

        if not self.config.request_reflection_enabled:
            return [], None, None, []

        gap_signals = self._reflection_gap_signals(state)
        if not gap_signals:
            return [], None, None, []

        remaining_budget = self._remaining_task_budget(state)
        max_additional_tasks = min(
            remaining_budget,
            max(int(self.config.reflection_max_additional_tasks or 1), 1),
        )
        if max_additional_tasks <= 0:
            reason = "发现覆盖缺口，但任务预算已满，直接生成报告。"
            observer.record_reflection_skip(reason=reason, gap_signals=gap_signals)
            return [], reason, None, gap_signals

        assessment = self.reflection.assess_request(
            state,
            gap_signals=gap_signals,
            observer=observer,
        )

        additional_tasks: list[TodoItem] = []
        if assessment.needs_more_research:
            additional_tasks = self.planner.plan_additional_tasks(
                state,
                missing_angles=assessment.missing_angles,
                existing_tasks=list(state.todo_items),
                max_additional_tasks=max_additional_tasks,
                observer=observer,
            )
            if additional_tasks:
                state.todo_items.extend(additional_tasks)

        reason = assessment.reason
        if assessment.needs_more_research:
            if additional_tasks:
                reason = f"{assessment.reason} 已补充 {len(additional_tasks)} 个任务继续研究。"
            else:
                reason = f"{assessment.reason} 但未生成有效补充任务，直接生成报告。"

        observer.record_reflection_call(
            reason=reason,
            gap_signals=assessment.gap_signals or gap_signals,
            added_tasks=len(additional_tasks),
        )
        return additional_tasks, reason, assessment, gap_signals

    def _run_review_stage(
        self,
        state: SummaryState,
        observer: RequestTrace,
    ) -> dict[str, Any]:
        """Run request-level review checks and persist the resulting summary."""

        if not self.config.review_stage_enabled or not hasattr(self, "reviewer"):
            state.review_summary = {
                "overall_status": "skipped",
                "reason": "review stage disabled",
                "issue_count": 0,
                "severity_counts": {"high": 0, "medium": 0, "low": 0},
                "issues": [],
            }
            state.review_completed = True
            return state.review_summary

        review_summary = self.reviewer.review_request(
            state,
            observer=observer,
        )

        if review_summary.get("overall_status") == "blocked":
            observer.record_degraded("review_blocked")
        elif review_summary.get("overall_status") == "warning":
            observer.record_degraded("review_warning")

        return review_summary

    def run(
        self,
        topic: str,
        *,
        initial_state: SummaryState | None = None,
        resume_phase: str | None = None,
    ) -> SummaryStateOutput:
        """Execute the research workflow and return the final report."""
        observer = self._start_request_trace(topic)
        state = initial_state or SummaryState(research_topic=topic)
        report = state.structured_report or ""
        try:
            resumed = initial_state is not None

            if resumed and resume_phase:
                self._restore_task_counters(observer, state)
                self._persist_request_state(state, phase=resume_phase, status="in_progress")

            if not resumed or not state.todo_items:
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
                self._persist_request_state(state, phase="planned", status="in_progress")

            if not state.todo_items:
                logger.info("No TODO items generated; falling back to single task")
                observer.record_fallback("planning_returned_no_tasks")
                observer.record_degraded("fallback_task_used")
                state.todo_items = [self.planner.create_fallback_task(state)]

            self._apply_task_budget(state, observer)
            observer.set_task_totals(total_tasks=len(state.todo_items))

            pending_tasks = [
                task for task in state.todo_items if task.status not in {"completed"}
            ]
            if pending_tasks:
                self._execute_task_batch_sync(state, pending_tasks)
                self._persist_request_state(state, phase="task_execution", status="in_progress")

            if not state.reflection_completed:
                reflection_gap_signals = self._reflection_gap_signals(state)
                if self.config.request_reflection_enabled and reflection_gap_signals:
                    remaining_budget = self._remaining_task_budget(state)
                    if remaining_budget <= 0:
                        observer.record_reflection_skip(
                            reason="发现覆盖缺口，但任务预算已满，直接生成报告。",
                            gap_signals=reflection_gap_signals,
                        )
                    else:
                        reflection_span = observer.start_stage(
                            "reflection",
                            scope="request",
                            metadata={
                                "gap_signal_count": len(reflection_gap_signals),
                                "remaining_budget": remaining_budget,
                            },
                        )
                        try:
                            additional_tasks, _, assessment, _ = self._run_reflection_cycle(state, observer)
                            reflection_span.complete(
                                status="success",
                                metadata={
                                    "coverage_status": assessment.coverage_status if assessment else "sufficient",
                                    "added_tasks": len(additional_tasks),
                                },
                            )
                            self._drain_tool_events(state)
                            if additional_tasks:
                                observer.set_task_totals(total_tasks=len(state.todo_items))
                                self._execute_task_batch_sync(state, additional_tasks)
                        except Exception as exc:
                            reflection_span.complete(status="failed", error=exc)
                            observer.record_reflection_call(
                                reason=f"反思阶段失败：{exc}",
                                gap_signals=reflection_gap_signals,
                                added_tasks=0,
                            )
                            logger.warning("Reflection stage failed topic=%s error=%s", topic, exc)
                state.reflection_completed = True
                self._persist_request_state(state, phase="reflection", status="in_progress")

            if not state.review_completed:
                review_span = observer.start_stage("review", scope="request")
                try:
                    review_summary = self._run_review_stage(state, observer)
                    if hasattr(observer, "record_review_summary"):
                        observer.record_review_summary(review_summary)
                    review_span.complete(
                        status="success",
                        metadata={
                            "issue_count": int(review_summary.get("issue_count") or 0),
                            "overall_status": review_summary.get("overall_status"),
                        },
                    )
                except Exception as exc:
                    review_span.complete(status="failed", error=exc)
                    raise
                self._persist_request_state(state, phase="review", status="in_progress")

            if not report:
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
            self._persist_request_state(
                state,
                phase="completed",
                status=status,
                report_markdown=report,
            )

            return SummaryStateOutput(
                running_summary=report,
                report_markdown=report,
                todo_items=state.todo_items,
            )
        except Exception as exc:
            observer.complete_request(status="failed", error=exc)
            self._persist_request_state(
                state,
                phase="failed",
                status="failed",
                report_markdown=report,
            )
            raise

    def run_resume(self, request_id: str) -> SummaryStateOutput:
        """Resume a persisted request snapshot."""

        snapshot = self._load_resume_snapshot(request_id)
        state, phase = self._state_from_snapshot(snapshot)
        topic = state.research_topic or str(snapshot.get("topic") or "").strip()
        return self.run(topic, initial_state=state, resume_phase=phase)

    def run_stream(
        self,
        topic: str,
        *,
        initial_state: SummaryState | None = None,
        resume_phase: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Execute the workflow yielding incremental progress events."""
        observer = self._start_request_trace(topic)
        state = initial_state or SummaryState(research_topic=topic)
        report = state.structured_report or ""
        logger.debug("Starting streaming research: topic=%s", topic)
        yield {"type": "status", "message": "初始化研究流程"}
        try:
            resumed = initial_state is not None
            if resumed and resume_phase:
                self._restore_task_counters(observer, state)
                self._persist_request_state(state, phase=resume_phase, status="in_progress")
                yield {
                    "type": "status",
                    "message": f"已从持久化快照恢复请求：{self.request_id}",
                }

            if not resumed or not state.todo_items:
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
                    self._persist_request_state(state, phase="failed", status="failed")
                    raise
                self._persist_request_state(state, phase="planned", status="in_progress")

            metrics_event = self._metrics_event()
            if metrics_event:
                yield metrics_event

            for event in self._drain_tool_events(state, step=0):
                yield event
            if not state.todo_items:
                state.todo_items = [self.planner.create_fallback_task(state)]
                yield observer.record_fallback("planning_returned_no_tasks")
                yield observer.record_degraded("fallback_task_used")

            budget_notice = self._apply_task_budget(state, observer)
            if budget_notice:
                yield {
                    "type": "status",
                    "message": budget_notice,
                    "step": 0,
                }
                yield observer.metrics_event()

            observer.set_task_totals(total_tasks=len(state.todo_items))

            channel_map: dict[int, dict[str, Any]] = {}
            next_step = self._assign_stream_channels(
                state.todo_items,
                channel_map,
                start_step=1,
            )

            yield {
                "type": "todo_list",
                "tasks": [self._serialize_task(t) for t in state.todo_items],
                "step": 0,
            }

            pending_tasks = [
                task for task in state.todo_items if task.status not in {"completed"}
            ]
            if pending_tasks:
                for event in self._execute_task_batch_stream(state, pending_tasks, channel_map):
                    yield event
                self._persist_request_state(state, phase="task_execution", status="in_progress")

            reflection_gap_signals = self._reflection_gap_signals(state)
            if (
                not state.reflection_completed
                and self.config.request_reflection_enabled
                and reflection_gap_signals
            ):
                reflection_step = next_step
                remaining_budget = self._remaining_task_budget(state)
                if remaining_budget <= 0:
                    observer.record_reflection_skip(
                        reason="发现覆盖缺口，但任务预算已满，直接生成报告。",
                        gap_signals=reflection_gap_signals,
                    )
                    yield {
                        "type": "status",
                        "message": "发现覆盖缺口，但任务预算已满，直接生成报告。",
                        "step": reflection_step,
                    }
                    yield observer.metrics_event()
                    next_step = reflection_step + 1
                else:
                    reflection_span = observer.start_stage(
                        "reflection",
                        scope="request",
                        metadata={
                            "gap_signal_count": len(reflection_gap_signals),
                            "remaining_budget": remaining_budget,
                        },
                    )
                    reflection_started = reflection_span.started_event()
                    reflection_started["step"] = reflection_step
                    yield reflection_started
                    try:
                        additional_tasks, reflection_notice, assessment, _ = self._run_reflection_cycle(
                            state,
                            observer,
                        )
                        completed_event = reflection_span.complete(
                            status="success",
                            metadata={
                                "coverage_status": assessment.coverage_status if assessment else "sufficient",
                                "added_tasks": len(additional_tasks),
                            },
                        )
                        completed_event["step"] = reflection_step
                        yield completed_event
                        for event in self._drain_tool_events(state, step=reflection_step):
                            yield event
                        if reflection_notice:
                            yield {
                                "type": "status",
                                "message": reflection_notice,
                                "step": reflection_step,
                            }
                        yield observer.metrics_event()

                        next_step = reflection_step + 1
                        if additional_tasks:
                            observer.set_task_totals(total_tasks=len(state.todo_items))
                            next_step = self._assign_stream_channels(
                                additional_tasks,
                                channel_map,
                                start_step=next_step,
                            )
                            yield {
                                "type": "todo_list",
                                "tasks": [self._serialize_task(t) for t in state.todo_items],
                                "step": reflection_step,
                            }
                            yield observer.metrics_event()
                            for event in self._execute_task_batch_stream(
                                state,
                                additional_tasks,
                                channel_map,
                            ):
                                yield event
                    except Exception as exc:
                        completed_event = reflection_span.complete(status="failed", error=exc)
                        completed_event["step"] = reflection_step
                        yield completed_event
                        observer.record_reflection_call(
                            reason=f"反思阶段失败：{exc}",
                            gap_signals=reflection_gap_signals,
                            added_tasks=0,
                        )
                        yield {
                            "type": "status",
                            "message": "反思阶段失败，直接进入报告生成。",
                            "step": reflection_step,
                        }
                        yield observer.metrics_event()
                        next_step = reflection_step + 1
                state.reflection_completed = True
                self._persist_request_state(state, phase="reflection", status="in_progress")
            else:
                state.reflection_completed = True

            if not state.review_completed:
                review_step = next_step
                review_span = observer.start_stage(
                    "review",
                    scope="request",
                    metadata={"task_count": len(state.todo_items)},
                )
                review_started = review_span.started_event()
                review_started["step"] = review_step
                yield review_started
                try:
                    review_summary = self._run_review_stage(state, observer)
                    if hasattr(observer, "record_review_summary"):
                        observer.record_review_summary(review_summary)
                    completed_event = review_span.complete(
                        status="success",
                        metadata={
                            "issue_count": int(review_summary.get("issue_count") or 0),
                            "overall_status": review_summary.get("overall_status"),
                        },
                    )
                    completed_event["step"] = review_step
                    yield completed_event
                    yield {
                        "type": "review_summary",
                        "step": review_step,
                        "summary": review_summary,
                        "tasks": [self._serialize_task(task) for task in state.todo_items],
                    }
                    yield observer.metrics_event()
                except Exception as exc:
                    completed_event = review_span.complete(status="failed", error=exc)
                    completed_event["step"] = review_step
                    yield completed_event
                    observer.complete_request(status="failed", error=exc)
                    self._persist_request_state(state, phase="failed", status="failed")
                    raise
                next_step = review_step + 1
                self._persist_request_state(state, phase="review", status="in_progress")

            final_step = next_step
            if not report:
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
                    self._persist_request_state(state, phase="failed", status="failed")
                    raise
            else:
                yield {
                    "type": "status",
                    "message": "检测到已持久化的报告内容，直接恢复最终结果。",
                    "step": final_step,
                }

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
            self._persist_request_state(
                state,
                phase="completed",
                status=request_status,
                report_markdown=report,
            )
            yield observer.metrics_event()

            yield {
                "type": "final_report",
                "report": report,
                "note_id": state.report_note_id,
                "note_path": state.report_note_path,
            }
            yield {"type": "done"}
        except Exception:
            self._persist_request_state(state, phase="failed", status="failed", report_markdown=report)
            raise

    def run_stream_resume(self, request_id: str) -> Iterator[dict[str, Any]]:
        """Resume a persisted request snapshot in streaming mode."""

        snapshot = self._load_resume_snapshot(request_id)
        state, phase = self._state_from_snapshot(snapshot)
        topic = state.research_topic or str(snapshot.get("topic") or "").strip()
        yield from self.run_stream(topic, initial_state=state, resume_phase=phase)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _run_with_timeout(
        self,
        operation: Callable[[], Any],
        *,
        timeout_seconds: float | None,
        operation_name: str,
    ) -> Any:
        """Run an operation with a best-effort timeout guard."""

        if timeout_seconds is None or timeout_seconds <= 0:
            return operation()

        result_queue: Queue[tuple[str, Any]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put(("result", operation()))
            except Exception as exc:  # pragma: no cover - defensive guardrail
                result_queue.put(("error", exc))

        thread = Thread(target=worker, daemon=True)
        thread.start()

        try:
            outcome, payload = result_queue.get(timeout=timeout_seconds)
        except Empty as exc:
            raise ToolExecutionTimeoutError(
                f"{operation_name} 超时（>{timeout_seconds:.2f}s）"
            ) from exc

        if outcome == "error":
            raise payload

        return payload

    def _dispatch_search_with_guardrails(
        self,
        candidate: str,
        *,
        state: SummaryState,
        task: TodoItem,
        observer: RequestTrace | None,
        notices: list[str],
    ) -> tuple[dict[str, Any] | None, list[str], str | None, str, bool, str]:
        """Execute a search invocation with timeout and retry guardrails."""

        retry_attempts = max(int(self.config.search_tool_retry_attempts or 0), 0)
        retry_backoff_seconds = max(float(self.config.search_tool_retry_backoff_seconds or 0.0), 0.0)
        timeout_seconds = self.config.search_tool_timeout_seconds
        operation_name = f"搜索工具调用[{candidate}]"

        for retry_index in range(retry_attempts + 1):
            try:
                return self._run_with_timeout(
                    lambda: dispatch_search(
                        candidate,
                        self.config,
                        state.research_loop_count,
                        observer=observer,
                        cache_context={
                            "research_topic": state.research_topic,
                            "task_title": task.title,
                            "task_intent": task.intent,
                        },
                    ),
                    timeout_seconds=timeout_seconds,
                    operation_name=operation_name,
                )
            except Exception as exc:
                timed_out = isinstance(exc, ToolExecutionTimeoutError)
                logger.warning(
                    "Search tool call failed task_id=%s title=%s query=%s retry=%s/%s error=%s",
                    task.id,
                    task.title,
                    candidate,
                    retry_index,
                    retry_attempts,
                    exc,
                )

                if observer:
                    reason = (
                        f"search_tool_timeout:{task.id}"
                        if timed_out
                        else f"search_tool_error:{task.id}"
                    )
                    observer.record_degraded(reason)

                if retry_index >= retry_attempts:
                    raise

                retry_notice = (
                    f"搜索工具调用超时，准备第 {retry_index + 1} 次重试：{candidate}"
                    if timed_out
                    else f"搜索工具调用失败，准备第 {retry_index + 1} 次重试：{candidate}"
                )
                if retry_notice not in notices:
                    notices.append(retry_notice)

                if observer:
                    observer.record_degraded(f"search_tool_retry:{task.id}:{retry_index + 1}")

                if retry_backoff_seconds > 0:
                    sleep(retry_backoff_seconds)

    def _normalize_query_candidate(self, value: str) -> str:
        """Normalize a candidate search query and strip request-style boilerplate."""

        cleaned = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (value or "").strip())

        while cleaned:
            previous = cleaned
            for pattern in _LEADING_SEARCH_QUERY_PATTERNS:
                cleaned = pattern.sub("", cleaned, count=1)
            cleaned = cleaned.lstrip("：:,，。！？、；;[]【】()（）- ")
            if cleaned == previous:
                break

        return " ".join(cleaned.strip().split())

    def _should_rewrite_task_query(
        self,
        *,
        original_query: str,
        title: str,
        intent: str,
    ) -> bool:
        """Return whether the task query is generic enough to benefit from rewriting."""

        if not self.config.task_query_rewrite_enabled:
            return False

        normalized_query = self._normalize_query_candidate(original_query).casefold()
        normalized_title = self._normalize_query_candidate(title).casefold()
        normalized_intent = self._normalize_query_candidate(intent).casefold()

        if not normalized_query:
            return True
        if normalized_query in {normalized_title, normalized_intent}:
            return True
        if len(normalized_query) <= 12:
            return True
        if len(normalized_query.split()) <= 2:
            return True
        return False

    def _rewritten_task_query(self, state: SummaryState, task: TodoItem) -> str:
        """Return a richer query candidate combining topic, title and intent."""

        topic = (state.research_topic or "").strip()
        title = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip())
        intent = (task.intent or "").strip()
        return self._normalize_query_candidate(" ".join(part for part in [topic, title, intent] if part))

    def _task_search_queries(
        self,
        state: SummaryState,
        task: TodoItem,
    ) -> list[tuple[str, str]]:
        """Return progressively broader queries for a task-level search retry."""

        topic = (state.research_topic or "").strip()
        title = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip())
        intent = (task.intent or "").strip()
        original_query = (task.query or "").strip()

        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(value: str, strategy: str) -> None:
            normalized = self._normalize_query_candidate(value)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append((normalized, strategy))

        add(original_query, "original")
        if self._should_rewrite_task_query(
            original_query=original_query,
            title=title,
            intent=intent,
        ):
            add(self._rewritten_task_query(state, task), "rewrite")
        if topic and title:
            add(f"{topic} {title}", "expand")
            add(f"{topic} {title} 最新进展", "expand")
        if topic and intent:
            add(f"{topic} {intent}", "expand")
        add(topic, "expand")

        fallback = self._normalize_query_candidate(" ".join(part for part in [title, intent] if part))
        return candidates or [(fallback, "rewrite")]

    def _apply_task_budget(
        self,
        state: SummaryState,
        observer: RequestTrace | None,
    ) -> str | None:
        """Trim planned tasks to the configured execution budget."""

        max_tasks = max(int(self.config.max_agent_tasks or 1), 1)
        if len(state.todo_items) <= max_tasks:
            return None

        dropped = len(state.todo_items) - max_tasks
        state.todo_items = state.todo_items[:max_tasks]
        notice = f"任务数超过预算，已仅保留前 {max_tasks} 个任务继续执行（截断 {dropped} 个）"

        logger.info(
            "Applied task budget limit max_tasks=%s dropped=%s topic=%s",
            max_tasks,
            dropped,
            state.research_topic,
        )

        if observer:
            observer.record_degraded(f"task_budget_applied:{max_tasks}")

        if state.todo_items and notice not in state.todo_items[0].notices:
            state.todo_items[0].notices.append(notice)

        return notice

    def _search_with_fallback_queries(
        self,
        state: SummaryState,
        task: TodoItem,
        observer: RequestTrace | None,
    ) -> tuple[dict[str, Any] | None, list[str], str | None, str, bool, str]:
        """Retry a task search with broader queries before marking it skipped."""

        search_result: dict[str, Any] | None = None
        answer_text: str | None = None
        backend = ""
        cache_hit = False
        cache_strategy = "miss"
        notices: list[str] = []
        original_query = (task.query or "").strip()
        candidates = self._task_search_queries(state, task)

        for attempt_index, (candidate, strategy) in enumerate(candidates):
            task.query = candidate
            (
                current_result,
                current_notices,
                current_answer,
                current_backend,
                current_cache_hit,
                current_cache_strategy,
            ) = self._dispatch_search_with_guardrails(
                candidate,
                state=state,
                task=task,
                observer=observer,
                notices=notices,
            )

            if attempt_index > 0:
                retry_notice = (
                    f"原检索词未命中，改用重写检索词：{candidate}"
                    if strategy == "rewrite"
                    else f"原检索词未命中，改用更宽泛检索词：{candidate}"
                )
                if retry_notice not in notices:
                    notices.append(retry_notice)

            for notice in current_notices:
                if notice and notice not in notices:
                    notices.append(notice)

            search_result = current_result
            answer_text = current_answer
            backend = current_backend
            cache_hit = current_cache_hit
            cache_strategy = current_cache_strategy

            if current_result and current_result.get("results"):
                if attempt_index > 0 and original_query and original_query != candidate:
                    logger.info(
                        "Recovered task search with broader query task_id=%s title=%s original_query=%s fallback_query=%s",
                        task.id,
                        task.title,
                        original_query,
                        candidate,
                    )
                return search_result, notices, answer_text, backend, cache_hit, cache_strategy

        return search_result, notices, answer_text, backend, cache_hit, cache_strategy

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
            (
                search_result,
                notices,
                answer_text,
                backend,
                cache_hit,
                cache_strategy,
            ) = self._search_with_fallback_queries(
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
            self._persist_request_state(state, phase="task_execution", status="in_progress")
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
                    "cache_strategy": cache_strategy,
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
                    "query": task.query,
                    "sources_summary": task.sources_summary,
                    "evidence_items": list(task.evidence_items),
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                    "step": step,
                }
                if observer:
                    yield observer.metrics_event()
            else:
                self._drain_tool_events(state)
            self._persist_request_state(state, phase="task_execution", status="in_progress")
            return
        else:
            if not emit_stream:
                self._drain_tool_events(state)

        evidence_store = getattr(self, "_evidence_store", None)
        if evidence_store is None:
            evidence_store = EvidenceStore()
            self._evidence_store = evidence_store

        evidence_items = evidence_store.record_search_results(
            task_id=task.id,
            query=task.query,
            search_payload=search_result,
            backend=backend,
        )
        sources_summary = format_evidence_sources(evidence_items)
        context = build_task_context(
            evidence_items,
            answer_text=answer_text,
            config=self.config,
        )

        task.sources_summary = sources_summary
        task.evidence_items = evidence_items

        with self._state_lock:
            state.web_research_results.append(context)
            state.sources_gathered.append(sources_summary)
            state.research_loop_count += 1

        summary_result: TaskSummaryResult | None = None
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
                    "evidence_items": list(task.evidence_items),
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
                    summary_result = self._coerce_summary_result(summary_getter())
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
                    self._persist_request_state(state, phase="task_execution", status="in_progress")
                    return
                if summary_span:
                    completed = summary_span.complete(
                        status="success",
                        metadata={"summary_length": len(summary_result.markdown or "")},
                    )
                    completed["step"] = step
                    yield completed
                    if observer:
                        yield observer.metrics_event()
            else:
                try:
                    summary_result = self._coerce_summary_result(
                        self.summarizer.summarize_task(
                            state,
                            task,
                            context,
                            observer=observer,
                        )
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
                    self._persist_request_state(state, phase="task_execution", status="in_progress")
                    return
                else:
                    if summary_span:
                        summary_span.complete(
                            status="success",
                            metadata={"summary_length": len(summary_result.markdown or "")},
                        )
                self._drain_tool_events(state)
        finally:
            release_summary_slot()

        task.evidence_items = evidence_store.list_task_evidence(task.id)
        task.summary = (summary_result.markdown or "").strip() if summary_result else "暂无可用信息"
        task.summary_payload = dict(summary_result.payload or {}) if summary_result else None
        task.claims = list(summary_result.claims or []) if summary_result else []
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
                "evidence_items": list(task.evidence_items),
                "claims": list(task.claims),
                "review_issues": list(task.review_issues),
                "review_status": task.review_status,
                "notices": list(task.notices),
                "note_id": task.note_id,
                "note_path": task.note_path,
                "step": step,
            }
        else:
            self._drain_tool_events(state)
        self._persist_request_state(state, phase="task_execution", status="in_progress")

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
        task.summary_payload = None
        task.claims = []
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
                    "query": task.query,
                    "evidence_items": list(task.evidence_items),
                    "review_issues": list(task.review_issues),
                    "review_status": task.review_status,
                    "notices": list(task.notices),
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
            "summary_payload": dict(task.summary_payload or {}) if task.summary_payload else None,
            "sources_summary": task.sources_summary,
            "notices": list(task.notices),
            "evidence_items": list(task.evidence_items),
            "claims": list(task.claims),
            "review_issues": list(task.review_issues),
            "review_status": task.review_status,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "stream_token": task.stream_token,
            "origin": task.origin,
            "round": task.round,
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
