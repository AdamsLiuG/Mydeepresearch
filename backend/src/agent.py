"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import ipaddress
import logging
import re
from pathlib import Path
from threading import Lock, Semaphore, local
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

import httpx
from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool
from openai import OpenAI

from config import Configuration
from metrics import RequestTrace
from models import SummaryState, SummaryStateOutput, TodoItem
from prompts import (
    report_writer_instructions,
    request_reflection_system_prompt,
    request_reviewer_system_prompt,
    task_react_plan_prompt,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from repair_orchestrator import RepairOrchestratorMixin
from services.evidence import (
    EvidenceLookupTool,
    EvidenceStore,
    FetchPageTool,
    SearchWebTool,
)
from services.evidence_index import (
    EvidenceMemoryService,
    EvidenceRetrievalService,
    EvidenceRuntimeIndex,
)
from services.note_memory import NoteMemoryService
from services.planner import PlanningService
from services.reflection import ReflectionAssessment, ReflectionService
from services.reporter import ReportingService
from services.request_state import RequestStateStore
from services.reviewer import ReviewService
from services.search import classify_search_query_bucket
from services.skill_pack import (
    SkillPackService,
    SkillSelection,
    apply_skill_pack_overrides,
)
from services.strategy_memory import StrategyMemoryService
from services.strategy_synthesizer import StrategySynthesizer
from services.summarizer import SummarizationService
from services.tool_events import ToolCallTracker
from state_manager import StateManagerMixin
from stream_coordinator import StreamCoordinatorMixin
from task_executor import TaskExecutorMixin

logger = logging.getLogger(__name__)


class SafeHelloAgentsLLM(HelloAgentsLLM):
    """Provide safer local-vLLM response handling for sync and streaming calls."""

    def __init__(self, *args: Any, allow_reasoning_fallback: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_reasoning_fallback = allow_reasoning_fallback
        self._usage_local = local()

    def consume_last_usage(self) -> dict[str, int] | None:
        """Return and clear the most recent API-reported token usage on this thread."""
        usage = getattr(self._usage_local, "last_usage", None)
        self._usage_local.last_usage = None
        return dict(usage) if isinstance(usage, dict) else None

    @staticmethod
    def _extract_chunk_usage(chunk: Any) -> dict[str, int] | None:
        raw_usage = getattr(chunk, "usage", None)
        if raw_usage is None:
            return None
        if isinstance(raw_usage, dict):
            usage_dict = raw_usage
        elif hasattr(raw_usage, "model_dump"):
            usage_dict = raw_usage.model_dump(exclude_none=True)
        else:
            usage_dict = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
                "completion_tokens": getattr(raw_usage, "completion_tokens", None),
                "total_tokens": getattr(raw_usage, "total_tokens", None),
            }
        prompt_tokens = int(usage_dict.get("prompt_tokens") or 0)
        completion_tokens = int(usage_dict.get("completion_tokens") or 0)
        total_tokens = int(usage_dict.get("total_tokens") or (prompt_tokens + completion_tokens))
        if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
            return None
        parsed: dict[str, int] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        prompt_details = usage_dict.get("prompt_tokens_details")
        if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens") is not None:
            parsed["cached_tokens"] = max(0, int(prompt_details.get("cached_tokens") or 0))
        return parsed

    @staticmethod
    def _stream_options_unsupported(exc: Exception) -> bool:
        if isinstance(exc, TypeError):
            return True
        message = str(exc or "").lower()
        return any(
            token in message
            for token in (
                "stream_options",
                "include_usage",
                "extra_forbidden",
                "extra inputs are not permitted",
                "unknown parameter",
                "unrecognized request argument",
            )
        )

    def _create_stream_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        request_kwargs = self._build_request_kwargs(messages, **kwargs)
        try:
            return self._client.chat.completions.create(
                **request_kwargs,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            if not self._stream_options_unsupported(exc):
                raise
            return self._client.chat.completions.create(
                **request_kwargs,
                stream=True,
            )

    @staticmethod
    def _should_bypass_env_proxy(base_url: str | None) -> bool:
        """Bypass environment proxies for loopback/private OpenAI-compatible endpoints."""

        parsed = urlsplit(str(base_url or "").strip())
        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            return False
        if hostname == "localhost":
            return True
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return False
        return bool(address.is_loopback or address.is_private or address.is_link_local)

    def _create_client(self) -> OpenAI:
        """Create an OpenAI client while avoiding proxies for local/private LLM endpoints."""

        http_client: httpx.Client | None = None
        if self._should_bypass_env_proxy(getattr(self, "base_url", None)):
            http_client = httpx.Client(trust_env=False)

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=http_client,
        )

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
        self._usage_local.last_usage = None

        response = self._create_stream_completion(messages, **kwargs)

        for chunk in response:
            chunk_usage = self._extract_chunk_usage(chunk)
            if chunk_usage is not None:
                self._usage_local.last_usage = chunk_usage
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
        self._usage_local.last_usage = None

        response = self._create_stream_completion(messages, **kwargs)

        for chunk in response:
            chunk_usage = self._extract_chunk_usage(chunk)
            if chunk_usage is not None:
                self._usage_local.last_usage = chunk_usage
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


class DeepResearchAgent(
    StateManagerMixin,
    StreamCoordinatorMixin,
    RepairOrchestratorMixin,
    TaskExecutorMixin,
):
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
        self._evidence_memory = EvidenceMemoryService(self.config)
        self._evidence_runtime = EvidenceRuntimeIndex(
            self.config,
            archive_loader=self._evidence_memory if self._evidence_memory.enabled else None,
        )
        evidence_listeners = []
        if self._evidence_memory.enabled:
            evidence_listeners.append(self._evidence_memory)
        if self._evidence_runtime.enabled:
            evidence_listeners.append(self._evidence_runtime)
        self._evidence_store = EvidenceStore(
            freshness_reference_days=self.config.freshness_reference_days,
            listeners=evidence_listeners,
            request_id_getter=lambda: self.request_id,
        )
        self._evidence_retrieval = EvidenceRetrievalService(
            runtime_index=self._evidence_runtime if self._evidence_runtime.enabled else None,
            memory_service=self._evidence_memory if self._evidence_memory.enabled else None,
            config=self.config,
        )
        self._request_state_store = (
            RequestStateStore(
                self.config.request_state_dir,
                recent_limit=self.config.request_state_recent_limit,
            )
            if self.config.request_state_enabled
            else None
        )
        self._note_memory = NoteMemoryService(self.config)
        self._strategy_synthesizer = StrategySynthesizer(
            lambda: self._create_tool_aware_agent(
                name="策略记忆提炼器",
                system_prompt="你是一名策略记忆提炼器，只输出严格 JSON。",
                llm=self._content_only_llm,
                enable_tool_calling=False,
            ),
            self.config,
        )
        self._strategy_memory = StrategyMemoryService(
            self.config,
            synthesizer=self._strategy_synthesizer,
        )
        self._skill_pack = SkillPackService(self.config)
        self._skill_selection: SkillSelection | None = None

        self.note_tool = (
            NoteTool(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )
        self.tools_registry: ToolRegistry | None = ToolRegistry()
        self.search_web_tool = SearchWebTool(
            config=self.config,
            evidence_store=self._evidence_store,
            observer_getter=lambda: self._request_trace,
        )
        self.fetch_page_tool = FetchPageTool(
            evidence_store=self._evidence_store,
            timeout_seconds=float(self.config.search_tool_timeout_seconds or 10.0),
        )
        self.evidence_lookup_tool = EvidenceLookupTool(
            evidence_store=self._evidence_store,
            retrieval_service=self._evidence_retrieval,
            request_id_getter=lambda: self.request_id,
        )
        self.tools_registry.register_tool(self.search_web_tool)
        self.tools_registry.register_tool(self.fetch_page_tool)
        self.tools_registry.register_tool(self.evidence_lookup_tool)
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
            llm=self._content_only_llm,
            enable_tool_calling=False,
        )
        self.review_agent = self._create_tool_aware_agent(
            name="研究质量审查专家",
            system_prompt=request_reviewer_system_prompt.strip(),
            llm=self._content_only_llm,
        )
        self.task_react_agent = self._create_tool_aware_agent(
            name="任务证据修补规划器",
            system_prompt=task_react_plan_prompt.strip(),
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
            retrieval_service=self._evidence_retrieval,
            request_id_getter=lambda: self.request_id,
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

    def _ensure_evidence_retrieval(self) -> EvidenceRetrievalService:
        if not hasattr(self, "_evidence_memory"):
            self._evidence_memory = EvidenceMemoryService(self.config)
        if not hasattr(self, "_evidence_runtime"):
            self._evidence_runtime = EvidenceRuntimeIndex(
                self.config,
                archive_loader=self._evidence_memory if self._evidence_memory.enabled else None,
            )
        if not hasattr(self, "_evidence_retrieval"):
            self._evidence_retrieval = EvidenceRetrievalService(
                runtime_index=self._evidence_runtime if self._evidence_runtime.enabled else None,
                memory_service=self._evidence_memory if self._evidence_memory.enabled else None,
                config=self.config,
            )
        return self._evidence_retrieval

    def _create_tool_aware_agent(
        self,
        *,
        name: str,
        system_prompt: str,
        llm: HelloAgentsLLM | None = None,
        enable_tool_calling: bool | None = None,
    ) -> ToolAwareSimpleAgent:
        """Instantiate a ToolAwareSimpleAgent sharing tool registry and tracker."""
        tool_calling_enabled = self.config.use_tool_calling and self.tools_registry is not None
        if enable_tool_calling is not None:
            tool_calling_enabled = bool(enable_tool_calling) and self.tools_registry is not None
        if tool_calling_enabled and self._provider_requires_prompt_only_agents():
            tool_calling_enabled = False

        return ToolAwareSimpleAgent(
            name=name,
            llm=llm or self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=tool_calling_enabled,
            tool_registry=self.tools_registry,
            tool_call_listener=self._tool_tracker.record,
        )

    def _provider_requires_prompt_only_agents(self) -> bool:
        """Return whether the active provider should avoid auto tool-calling.

        The current vLLM deployment is reachable for normal chat completions, but it
        rejects OpenAI-style auto tool choice unless the server is started with
        dedicated tool-calling flags. Falling back to prompt-only agents keeps the
        research pipeline usable on that deployment instead of failing during planning.
        """

        provider = str(self.config.llm_provider or "").strip().lower()
        return provider == "vllm"

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
        self._request_trace.memory_shadow_mode = bool(self.config.memory_shadow_mode)
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

    def _current_note_ids(self, state: SummaryState, *, include_report: bool = True) -> set[str]:
        note_ids = {
            str(task.note_id).strip()
            for task in state.todo_items
            if str(task.note_id or "").strip()
        }
        if include_report and str(state.report_note_id or "").strip():
            note_ids.add(str(state.report_note_id or "").strip())
        return note_ids

    def _select_skill_pack(
        self,
        topic: str,
        observer: RequestTrace | None,
        *,
        state: SummaryState | None = None,
    ) -> None:
        """Pick exactly one research skill pack for this request, once, before planning.

        Kept separate from the stage helpers so selection is a single, observable decision:
        which pack (or neutral fallback) governs the whole request is recorded on the trace,
        which is what lets eval reports stratify outcomes by pack.
        """

        self._skill_selection = None
        if not self._skill_pack.enabled:
            return
        try:
            selection = self._skill_pack.select(topic or "")
        except Exception as exc:  # pragma: no cover - selection must never fail a request
            logger.warning("skill pack selection failed error=%s", exc)
            if observer is not None:
                observer.record_degraded("skill_pack_invalid")
            return
        if selection is None:
            # Even the neutral fallback is missing/invalid; proceed with built-in prompts.
            if observer is not None:
                observer.record_degraded("skill_pack_unavailable")
            return
        self._skill_selection = selection
        applied_overrides: dict[str, Any] = {}
        if not selection.shadow and not selection.pack.is_neutral:
            self._apply_request_config_overrides(selection.pack, applied_overrides)
        if state is not None:
            state.skill_pack_metadata = {
                "pack_id": selection.pack.pack_id,
                "version": str(selection.pack.version),
                "topic_domain": selection.pack.topic_domain,
                "match_score": round(float(selection.match_score), 4),
                "fallback_reason": selection.fallback_reason,
                "shadow": bool(selection.shadow),
                "applied_overrides": applied_overrides,
            }
        if observer is not None:
            observer.record_skill_pack_selection(
                pack_id=selection.pack.pack_id,
                version=str(selection.pack.version),
                match_score=selection.match_score,
                fallback_reason=selection.fallback_reason,
                shadow=selection.shadow,
            )

    def _apply_request_config_overrides(
        self,
        pack: Any,
        applied_out: dict[str, Any],
    ) -> None:
        """Apply request-scoped skill pack config overrides and propagate to sub-services."""

        base_config = getattr(self, "_base_config", None) or self.config
        self._base_config = base_config
        effective_config, applied = apply_skill_pack_overrides(base_config, pack)
        if not applied:
            return
        applied_out.update(applied)
        self._swap_runtime_config(effective_config)

    def _restore_request_config(self) -> None:
        base_config = getattr(self, "_base_config", None)
        if base_config is not None:
            self._swap_runtime_config(base_config)
            self._base_config = None

    def _swap_runtime_config(self, target_config: Configuration) -> None:
        self.config = target_config
        for attr in ("planner", "summarizer", "reporting", "reflection", "reviewer"):
            service = getattr(self, attr, None)
            if service is not None and hasattr(service, "_config"):
                service._config = target_config

    def _active_strategy_topic_domains(self) -> set[str] | None:
        """Return strategy-memory domain filter derived from the active skill pack."""

        selection = getattr(self, "_skill_selection", None)
        if selection is None or selection.shadow or selection.pack.is_neutral:
            return None
        domain = str(selection.pack.topic_domain or "").strip().lower()
        if not domain or domain == "general":
            return None
        return {domain, "general"}

    def _skill_stage_context(self, stage: str) -> str:
        """Render the skill fragment for one stage, or empty when nothing should be injected.

        A pack with ``status: shadow`` is recorded at selection time but never injected, mirroring
        memory shadow mode so a candidate pack can be observed before it influences behaviour.
        """

        selection = self._skill_selection
        if selection is None or selection.shadow:
            return ""
        return self._skill_pack.render_block(selection.pack, stage)

    def _shadow_gate(
        self,
        rendered: str,
        *,
        stage: str,
        observer: RequestTrace | None,
    ) -> str:
        """Withhold retrieved memory from the prompt while keeping retrieval observable.

        Shadow mode lets us collect "what would have been injected" without changing
        any downstream behaviour, so memory can be measured before it is enabled.
        """

        if not rendered or not self.config.memory_shadow_mode:
            return rendered
        if observer is not None:
            observer.record_memory_shadow_suppression(stage)
        return ""

    def _planning_memory_context(
        self,
        state: SummaryState,
        observer: RequestTrace | None,
    ) -> str:
        return self._shadow_gate(
            self._note_memory.search_for_planning(
                state.research_topic or "",
                current_request_id=self.request_id,
                exclude_note_ids=self._current_note_ids(state, include_report=False),
                observer=observer,
            ),
            stage="planning_note",
            observer=observer,
        )

    def _planning_strategy_context(
        self,
        state: SummaryState,
        observer: RequestTrace | None,
    ) -> str:
        return self._shadow_gate(
            self._strategy_memory.search_for_planning(
                state.research_topic or "",
                current_request_id=self.request_id,
                observer=observer,
                topic_domains=self._active_strategy_topic_domains(),
            ),
            stage="planning_strategy",
            observer=observer,
        )

    def _task_memory_context(
        self,
        state: SummaryState,
        task: TodoItem,
        observer: RequestTrace | None,
    ) -> str:
        excluded_note_ids = self._current_note_ids(state)
        if str(task.note_id or "").strip():
            excluded_note_ids.add(str(task.note_id or "").strip())
        return self._shadow_gate(
            self._note_memory.search_for_task(
                state.research_topic or "",
                task.title,
                task.intent,
                current_request_id=self.request_id,
                exclude_note_ids=excluded_note_ids,
                observer=observer,
            ),
            stage="task_execution_note",
            observer=observer,
        )

    def _reflection_strategy_context(
        self,
        state: SummaryState,
        observer: RequestTrace | None,
    ) -> str:
        task_titles = [
            task.title
            for task in state.todo_items
            if task.status in {"failed", "skipped"} or not str(task.summary or "").strip()
        ]
        return self._shadow_gate(
            self._strategy_memory.search_for_reflection(
                state.research_topic or "",
                gap_signals=self._reflection_gap_signals(state),
                task_titles=task_titles,
                current_request_id=self.request_id,
                observer=observer,
                topic_domains=self._active_strategy_topic_domains(),
            ),
            stage="reflection_strategy",
            observer=observer,
        )

    def _refresh_note_memory_for_state(
        self,
        state: SummaryState,
        *,
        observer: RequestTrace | None,
    ) -> None:
        note_ids = self._current_note_ids(state)
        if not note_ids:
            return
        self._note_memory.refresh_notes(note_ids, observer=observer)

    def _refresh_strategy_memory_for_request(
        self,
        request_id: str,
        *,
        observer: RequestTrace | None,
    ) -> None:
        # Flush usage before re-synthesising: the maintenance pass reads the card files that
        # refresh_request may rewrite, so recording hits first keeps the counters intact.
        self._flush_strategy_memory_usage(observer=observer)
        self._strategy_memory.refresh_request(request_id, observer=observer)

    def _flush_strategy_memory_usage(self, *, observer: RequestTrace | None) -> None:
        """Persist which strategy cards this request actually used, then decay unused drafts.

        Hits are collected in the request trace during retrieval and written once here, so the
        query path performs no writes.
        """

        strategy_ids = list(getattr(observer, "strategy_memory_hit_ids", []) or [])
        try:
            self._strategy_memory.run_maintenance(strategy_ids, observer=observer)
        except Exception as exc:  # pragma: no cover - maintenance must never fail a request
            logger.warning("strategy memory maintenance failed error=%s", exc)

    def _reflection_gap_signals(self, state: SummaryState) -> list[str]:
        """Return request-level signals that justify running reflection."""

        signals: list[str] = []
        observer = self._request_trace

        if observer and observer.fallback_count:
            signals.append("request_has_fallback")

        completed_tasks: list[TodoItem] = []
        all_evidence: list[dict[str, Any]] = []
        for task in state.todo_items:
            if task.status == "failed":
                signals.append(f"task_{task.id}_failed")
            elif task.status == "skipped":
                signals.append(f"task_{task.id}_skipped")

            if task.status == "completed":
                completed_tasks.append(task)
                summary = (task.summary or "").strip()
                sources = (task.sources_summary or "").strip()
                if not summary or summary == "暂无可用信息":
                    signals.append(f"task_{task.id}_summary_missing")
                if not sources:
                    signals.append(f"task_{task.id}_sources_missing")
                for item in task.evidence_items or []:
                    if isinstance(item, dict):
                        all_evidence.append(item)

        if all_evidence:
            domains = {
                str(item.get("domain") or "").strip().lower()
                for item in all_evidence
                if str(item.get("domain") or "").strip()
            }
            if len(completed_tasks) >= 2 and len(all_evidence) >= 2 and len(domains) <= 1:
                signals.append("low_global_domain_diversity")

            low_authority_types = {
                "content_farm_or_aggregator",
                "social_general",
                "forum_general",
            }
            typed_evidence = [
                item
                for item in all_evidence
                if str(item.get("source_type") or "").strip()
                or str(item.get("quality_label") or "").strip()
            ]
            if typed_evidence and all(
                str(item.get("quality_label") or "").strip().lower() == "low"
                or str(item.get("source_type") or "").strip().lower() in low_authority_types
                for item in typed_evidence
            ):
                signals.append("low_evidence_authority")

            topic_text = str(state.research_topic or "").strip()
            if topic_text and classify_search_query_bucket(topic_text, None) == "fresh":
                freshness_labels = [
                    str(item.get("freshness_label") or "").strip().lower()
                    for item in all_evidence
                    if str(item.get("freshness_label") or "").strip().lower() not in {"", "unknown"}
                ]
                if freshness_labels and not any(
                    label in {"fresh", "recent"} for label in freshness_labels
                ):
                    signals.append("stale_global_evidence")

        deduped: list[str] = []
        seen: set[str] = set()
        for signal in signals:
            if signal in seen:
                continue
            seen.add(signal)
            deduped.append(signal)
        return deduped

    def _remaining_task_budget(
        self,
        state: SummaryState,
        *,
        stage: str | None = None,
    ) -> int:
        """Return how many additional tasks can still be appended for a stage."""

        total_cap = self.config.resolved_max_total_tasks()
        raw_remaining = max(0, total_cap - len(state.todo_items))
        if stage == "reflection":
            base_cap = max(1, int(self.config.max_agent_tasks or 1))
            if (
                self.config.report_repair_enabled
                and int(self.config.repair_reserve_tasks or 0) > 0
                and total_cap > base_cap
            ):
                repair_cap = max(1, int(self.config.report_repair_max_tasks or 1))
                repair_floor = min(
                    int(self.config.repair_reserve_tasks or 0),
                    repair_cap,
                    max(
                        1,
                        total_cap
                        - self.config.resolved_initial_plan_max_tasks()
                        - max(1, int(self.config.reflection_max_additional_tasks or 1)),
                    ),
                )
                return max(0, raw_remaining - repair_floor)
        return raw_remaining

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

        remaining_budget = self._remaining_task_budget(state, stage="reflection")
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
            strategy_memory_context=self._reflection_strategy_context(state, observer),
            skill_pack_context=self._skill_stage_context("reflection"),
            observer=observer,
        )

        additional_tasks: list[TodoItem] = []
        if assessment.needs_more_research:
            historical_memory_context = self._planning_memory_context(state, observer)
            additional_tasks = self.planner.plan_additional_tasks(
                state,
                missing_angles=assessment.missing_angles,
                existing_tasks=list(state.todo_items),
                max_additional_tasks=max_additional_tasks,
                observer=observer,
                historical_memory_context=historical_memory_context,
                skill_pack_context=self._skill_stage_context("planning"),
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
            skill_pack_context=self._skill_stage_context("review"),
        )

        if review_summary.get("overall_status") == "blocked":
            observer.record_degraded("review_blocked")
        elif review_summary.get("overall_status") == "warning":
            observer.record_degraded("review_warning")

        return review_summary

    def _execute_workflow(
        self,
        topic: str,
        *,
        state: SummaryState,
        observer: RequestTrace,
        resumed: bool,
        resume_phase: str | None,
        emit_stream: bool,
    ) -> Iterator[dict[str, Any]]:
        """Run the unified 6-stage workflow for both sync and streaming callers."""
        report = state.structured_report or ""
        request_completed = False
        self._select_skill_pack(state.research_topic or topic, observer, state=state)

        if emit_stream:
            logger.debug("Starting streaming research: topic=%s", topic)
            yield {"type": "status", "message": "初始化研究流程"}

        try:
            if resumed and resume_phase:
                self._restore_task_counters(observer, state)
                self._persist_request_state(state, phase=resume_phase, status="in_progress")
                if emit_stream:
                    yield {
                        "type": "status",
                        "message": f"已从持久化快照恢复请求：{self.request_id}",
                    }

            if not resumed or not state.todo_items:
                planning_span = observer.start_stage(
                    "planning",
                    scope="request",
                    metadata={"topic": topic} if emit_stream else None,
                )
                if emit_stream:
                    yield planning_span.started_event()

                try:
                    historical_memory_context = self._planning_memory_context(state, observer)
                    state.todo_items = self.planner.plan_todo_list(
                        state,
                        observer=observer,
                        historical_memory_context=historical_memory_context,
                        strategy_memory_context=self._planning_strategy_context(state, observer),
                        skill_pack_context=self._skill_stage_context("planning"),
                    )
                    completed_event = planning_span.complete(
                        status="success",
                        metadata={"task_count": len(state.todo_items)},
                    )
                    if emit_stream:
                        yield completed_event
                except Exception as exc:
                    completed_event = planning_span.complete(status="failed", error=exc)
                    if emit_stream:
                        yield completed_event
                        yield observer.metrics_event()
                    observer.complete_request(status="failed", error=exc)
                    request_completed = True
                    raise

                if not emit_stream:
                    self._drain_tool_events(state)
                self._persist_request_state(state, phase="planned", status="in_progress")

            if emit_stream:
                metrics_event = self._metrics_event()
                if metrics_event:
                    yield metrics_event
                for event in self._drain_tool_events(state, step=0):
                    yield event

            if not state.todo_items:
                logger.info("No TODO items generated; falling back to single task")
                state.todo_items = [self.planner.create_fallback_task(state)]
                fallback_event = observer.record_fallback("planning_returned_no_tasks")
                degraded_event = observer.record_degraded("fallback_task_used")
                if emit_stream:
                    yield fallback_event
                    yield degraded_event

            budget_notice = self._apply_task_budget(state, observer)
            if emit_stream and budget_notice:
                yield {
                    "type": "status",
                    "message": budget_notice,
                    "step": 0,
                }
                yield observer.metrics_event()

            observer.set_task_totals(total_tasks=len(state.todo_items))

            channel_map: dict[int, dict[str, Any]] = {}
            next_step = 1
            if emit_stream:
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
                if emit_stream:
                    for event in self._execute_task_batch_stream(state, pending_tasks, channel_map):
                        yield event
                else:
                    self._execute_task_batch_sync(state, pending_tasks)
                self._persist_request_state(state, phase="task_execution", status="in_progress")

            if not state.reflection_completed:
                reflection_gap_signals = self._reflection_gap_signals(state)
                if self.config.request_reflection_enabled and reflection_gap_signals:
                    reflection_step = next_step
                    remaining_budget = self._remaining_task_budget(state, stage="reflection")
                    if remaining_budget <= 0:
                        reason = "发现覆盖缺口，但任务预算已满，直接生成报告。"
                        observer.record_reflection_skip(
                            reason=reason,
                            gap_signals=reflection_gap_signals,
                        )
                        if emit_stream:
                            yield {
                                "type": "status",
                                "message": reason,
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
                        if emit_stream:
                            reflection_started = reflection_span.started_event()
                            reflection_started["step"] = reflection_step
                            yield reflection_started
                        try:
                            additional_tasks, reflection_notice, assessment, _ = (
                                self._run_reflection_cycle(state, observer)
                            )
                            completed_event = reflection_span.complete(
                                status="success",
                                metadata={
                                    "coverage_status": (
                                        assessment.coverage_status if assessment else "sufficient"
                                    ),
                                    "added_tasks": len(additional_tasks),
                                },
                            )
                            if emit_stream:
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
                            else:
                                self._drain_tool_events(state)

                            if additional_tasks:
                                observer.set_task_totals(total_tasks=len(state.todo_items))
                                if emit_stream:
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
                                else:
                                    self._execute_task_batch_sync(state, additional_tasks)
                        except Exception as exc:
                            completed_event = reflection_span.complete(status="failed", error=exc)
                            observer.record_reflection_call(
                                reason=f"反思阶段失败：{exc}",
                                gap_signals=reflection_gap_signals,
                                added_tasks=0,
                            )
                            if emit_stream:
                                completed_event["step"] = reflection_step
                                yield completed_event
                                yield {
                                    "type": "status",
                                    "message": "反思阶段失败，直接进入报告生成。",
                                    "step": reflection_step,
                                }
                                yield observer.metrics_event()
                                next_step = reflection_step + 1
                            else:
                                logger.warning("Reflection stage failed topic=%s error=%s", topic, exc)
                state.reflection_completed = True
                self._persist_request_state(state, phase="reflection", status="in_progress")

            if not state.review_completed:
                review_step = next_step
                review_span = observer.start_stage(
                    "review",
                    scope="request",
                    metadata={"task_count": len(state.todo_items)} if emit_stream else None,
                )
                if emit_stream:
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
                    if emit_stream:
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
                    if emit_stream:
                        completed_event["step"] = review_step
                        yield completed_event
                    observer.complete_request(status="failed", error=exc)
                    request_completed = True
                    raise
                if emit_stream:
                    next_step = review_step + 1
                self._persist_request_state(state, phase="review", status="in_progress")

            if not state.report_repair_completed:
                repair_candidates = self._select_report_repair_candidates(state)
                if self.config.report_repair_enabled and repair_candidates:
                    repair_step = next_step
                    repair_span = observer.start_stage(
                        "repair",
                        scope="request",
                        metadata={
                            "phase": "report_repair",
                            "candidate_count": len(repair_candidates),
                            "max_tasks": self.config.report_repair_max_tasks,
                        },
                    )
                    if emit_stream:
                        yield {
                            "type": "repair_cycle_started",
                            "cycle": state.report_repair_cycles + 1,
                            "candidate_count": len(repair_candidates),
                            "max_tasks": self.config.report_repair_max_tasks,
                            "step": repair_step,
                        }
                    try:
                        repair_tasks, repair_notice, _ = self._prepare_report_repair_tasks(
                            state, observer
                        )
                        state.report_repair_cycles += 1
                        if emit_stream and repair_notice:
                            yield {
                                "type": "status",
                                "message": repair_notice,
                                "step": repair_step,
                            }

                        if emit_stream:
                            next_step = repair_step + 1

                        if repair_tasks:
                            observer.set_task_totals(total_tasks=len(state.todo_items))
                            if emit_stream:
                                next_step = self._assign_stream_channels(
                                    repair_tasks,
                                    channel_map,
                                    start_step=next_step,
                                )
                                yield {
                                    "type": "todo_list",
                                    "tasks": [self._serialize_task(t) for t in state.todo_items],
                                    "step": repair_step,
                                }
                                yield observer.metrics_event()
                                for event in self._execute_task_batch_stream(
                                    state,
                                    repair_tasks,
                                    channel_map,
                                ):
                                    yield event

                                repair_review_step = next_step
                                repair_review_span = observer.start_stage(
                                    "review",
                                    scope="request",
                                    metadata={
                                        "phase": "report_repair",
                                        "task_count": len(state.todo_items),
                                    },
                                )
                                repair_review_started = repair_review_span.started_event()
                                repair_review_started["step"] = repair_review_step
                                yield repair_review_started
                                review_summary = self._run_review_stage(state, observer)
                                if hasattr(observer, "record_review_summary"):
                                    observer.record_review_summary(review_summary)
                                repair_review_completed = repair_review_span.complete(
                                    status="success",
                                    metadata={
                                        "issue_count": int(review_summary.get("issue_count") or 0),
                                        "overall_status": review_summary.get("overall_status"),
                                    },
                                )
                                repair_review_completed["step"] = repair_review_step
                                yield repair_review_completed
                                yield {
                                    "type": "review_summary",
                                    "step": repair_review_step,
                                    "summary": review_summary,
                                    "tasks": [self._serialize_task(task) for task in state.todo_items],
                                }
                                yield observer.metrics_event()
                                next_step = repair_review_step + 1
                            else:
                                self._execute_task_batch_sync(state, repair_tasks)
                                review_summary = self._run_review_stage(state, observer)
                                if hasattr(observer, "record_review_summary"):
                                    observer.record_review_summary(review_summary)
                        elif repair_notice or emit_stream:
                            observer.record_degraded("report_repair_not_executed")
                            if emit_stream:
                                yield observer.metrics_event()

                        repair_span.complete(
                            status="success",
                            metadata={
                                "added_tasks": len(repair_tasks),
                                "candidate_count": len(repair_candidates),
                                "overall_status": state.review_summary.get("overall_status"),
                            },
                        )
                        if emit_stream:
                            yield {
                                "type": "repair_cycle_completed",
                                "cycle": state.report_repair_cycles,
                                "candidate_count": len(repair_candidates),
                                "added_tasks": len(repair_tasks),
                                "overall_status": state.review_summary.get("overall_status"),
                                "step": repair_step,
                            }
                    except Exception as exc:
                        repair_span.complete(status="failed", error=exc)
                        observer.record_degraded("report_repair_failed")
                        if emit_stream:
                            yield {
                                "type": "status",
                                "message": "报告修补阶段失败，继续输出当前保守报告。",
                                "step": repair_step,
                            }
                            yield observer.metrics_event()
                            logger.warning(
                                "Streaming report repair failed topic=%s error=%s", topic, exc
                            )
                            next_step = repair_step + 1
                        else:
                            logger.warning("Report repair stage failed topic=%s error=%s", topic, exc)
                    finally:
                        state.report_repair_completed = True
                        self._persist_request_state(
                            state, phase="report_repair", status="in_progress"
                        )
                else:
                    state.report_repair_completed = True

            final_step = next_step
            if not report:
                report_span = observer.start_stage(
                    "report",
                    scope="request",
                    metadata={"task_count": len(state.todo_items)} if emit_stream else None,
                )
                if emit_stream:
                    report_started = report_span.started_event()
                    report_started["step"] = final_step
                    yield report_started
                try:
                    report = self.reporting.generate_report(
                        state,
                        observer=observer,
                        skill_pack_context=self._skill_stage_context("report"),
                    )
                    completed_event = report_span.complete(
                        status="success",
                        metadata={"report_length": len(report or "")},
                    )
                    if emit_stream:
                        completed_event["step"] = final_step
                        yield completed_event
                except Exception as exc:
                    completed_event = report_span.complete(status="failed", error=exc)
                    if emit_stream:
                        completed_event["step"] = final_step
                        yield completed_event
                        yield observer.metrics_event()
                    observer.complete_request(status="failed", error=exc)
                    request_completed = True
                    raise
            elif emit_stream:
                yield {
                    "type": "status",
                    "message": "检测到已持久化的报告内容，直接恢复最终结果。",
                    "step": final_step,
                }

            if emit_stream:
                for event in self._drain_tool_events(state, step=final_step):
                    yield event
            else:
                self._drain_tool_events(state)

            state.structured_report = report
            state.running_summary = report

            note_event = self._persist_final_report(state, report)
            if emit_stream and note_event:
                yield note_event

            request_status = self._request_status(state, report)
            if emit_stream and request_status == "partial_success":
                yield observer.record_degraded("partial_result_completed")

            observer.attach_result(
                report_markdown=report,
                todo_items=[self._serialize_task(task) for task in state.todo_items],
                report_note_id=state.report_note_id,
                report_note_path=state.report_note_path,
            )
            observer.complete_request(status=request_status)
            request_completed = True
            self._persist_request_state(
                state,
                phase="completed",
                status=request_status,
                report_markdown=report,
            )
            self._refresh_strategy_memory_for_request(
                self.request_id,
                observer=observer,
            )

            if emit_stream:
                yield observer.metrics_event()
                yield {
                    "type": "final_report",
                    "report": report,
                    "note_id": state.report_note_id,
                    "note_path": state.report_note_path,
                }
                yield {"type": "done"}
        except Exception as exc:
            if not request_completed:
                observer.complete_request(status="failed", error=exc)
            self._persist_request_state(
                state,
                phase="failed",
                status="failed",
                report_markdown=report,
            )
            self._refresh_strategy_memory_for_request(
                self.request_id,
                observer=observer,
            )
            raise
        finally:
            self._restore_request_config()

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
        for _ in self._execute_workflow(
            topic,
            state=state,
            observer=observer,
            resumed=initial_state is not None,
            resume_phase=resume_phase,
            emit_stream=False,
        ):
            pass
        return SummaryStateOutput(
            running_summary=state.running_summary,
            report_markdown=state.structured_report,
            todo_items=state.todo_items,
        )

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
        yield from self._execute_workflow(
            topic,
            state=state,
            observer=observer,
            resumed=initial_state is not None,
            resume_phase=resume_phase,
            emit_stream=True,
        )

    def run_stream_resume(self, request_id: str) -> Iterator[dict[str, Any]]:
        """Resume a persisted request snapshot in streaming mode."""

        snapshot = self._load_resume_snapshot(request_id)
        state, phase = self._state_from_snapshot(snapshot)
        topic = state.research_topic or str(snapshot.get("topic") or "").strip()
        yield from self.run_stream(topic, initial_state=state, resume_phase=phase)

    def _drain_tool_events(
        self,
        state: SummaryState,
        *,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        """Proxy to the shared tool call tracker."""
        events = self._tool_tracker.drain(state, step=step)
        changed_note_ids = {
            str(event.get("note_id") or "").strip()
            for event in events
            if str(event.get("note_id") or "").strip()
        }
        if changed_note_ids:
            self._note_memory.refresh_notes(changed_note_ids, observer=self._request_trace)
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
            "react_rounds": task.react_rounds,
            "react_fetch_count": task.react_fetch_count,
            "react_additional_search_count": task.react_additional_search_count,
            "react_gap_signals": list(task.react_gap_signals),
            "react_last_action": task.react_last_action,
            "react_stop_reason": task.react_stop_reason,
            "react_observation": dict(task.react_observation or {}),
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

        self._note_memory.refresh_notes([note_id], observer=self._request_trace)

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
