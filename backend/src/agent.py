"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import ipaddress
import logging
import re
from pathlib import Path
from threading import Event, Lock, Semaphore, Thread, local
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

import httpx
from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool
from openai import OpenAI

from config import Configuration
from metrics import RequestTrace, metrics_registry
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
from services.claim_verifier import ClaimVerificationOrchestrator
from services.context_builder import (
    ContextPrefixBuilder,
    PromptEnvelope,
    ProviderCacheAdapter,
)
from services.dag_scheduler import build_upstream_context, partition_into_waves
from services.evidence import (
    EvidenceLookupTool,
    EvidenceStore,
    FetchPageTool,
    SearchWebTool,
    render_references,
)
from services.evidence_index import (
    EvidenceMemoryService,
    EvidenceRetrievalService,
    EvidenceRuntimeIndex,
)
from services.fact_ledger import (
    FactLedger,
    FactLedgerCompactor,
    build_evidence_index_cards,
)
from services.hitl import HitlService
from services.mcp_bridge import (
    MCPBridgeManager,
    MCPHitlApprovalRequired,
    ScopedToolRegistry,
)
from services.multimodal_parser import MultimodalEvidenceParser
from services.note_memory import NoteMemoryService
from services.observability import LangfuseTraceAdapter
from services.planner import PlanningService
from services.policy_controller import build_policy_controller
from services.reflection import ReflectionAssessment, ReflectionService
from services.reporter import ReportingService, build_report_workbench_payload
from services.reviewer import ReviewService
from services.sandbox_executor import (
    DockerSandboxBackend,
    PyodideSandboxBackend,
    RunPythonSandboxTool,
    SandboxExecutorService,
)
from services.search import classify_search_query_bucket
from services.skill_pack import (
    SkillPackService,
    SkillSelection,
    apply_skill_pack_overrides,
)
from services.storage_backends import build_request_state_store
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
        self._cache_header_unsupported = False

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
                "cache_read_input_tokens": getattr(raw_usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(raw_usage, "cache_creation_input_tokens", None),
                "prompt_tokens_details": getattr(raw_usage, "prompt_tokens_details", None),
            }
        prompt_tokens = int(usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens") or 0)
        completion_tokens = int(usage_dict.get("completion_tokens") or usage_dict.get("output_tokens") or 0)
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
        elif usage_dict.get("cached_tokens") is not None:
            parsed["cached_tokens"] = max(0, int(usage_dict.get("cached_tokens") or 0))
        elif usage_dict.get("cache_read_input_tokens") is not None:
            parsed["cached_tokens"] = max(0, int(usage_dict.get("cache_read_input_tokens") or 0))

        creation_val = usage_dict.get("cache_creation_tokens")
        if creation_val is None:
            creation_val = usage_dict.get("cache_creation_input_tokens")
        if creation_val is not None:
            parsed["cache_creation_tokens"] = max(0, int(creation_val or 0))
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

    @staticmethod
    def _cache_options_unsupported(exc: Exception) -> bool:
        if isinstance(exc, TypeError):
            return True
        message = str(exc or "").lower()
        return any(
            token in message
            for token in (
                "prompt_cache_key",
                "cache_control",
                "ephemeral",
                "extra_forbidden",
                "extra inputs are not permitted",
                "unknown parameter",
                "unrecognized request argument",
                "unsupported",
            )
        )

    @staticmethod
    def _flatten_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    txt = str(block.get("text") or "").strip()
                    if txt:
                        parts.append(txt)
                elif isinstance(block, str) and block.strip():
                    parts.append(block.strip())
            return "\n\n".join(parts)
        return str(content or "")

    @classmethod
    def _strip_cache_options(
        cls,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        stripped = dict(request_kwargs)
        stripped.pop("prompt_cache_key", None)
        stripped["messages"] = [
            {
                **msg,
                "content": cls._flatten_message_content(msg.get("content")),
            }
            for msg in messages
        ]
        return stripped

    def _open_stream(self, req_kwargs: dict[str, Any]) -> Any:
        try:
            return self._client.chat.completions.create(
                **req_kwargs,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            if not self._stream_options_unsupported(exc):
                raise
            return self._client.chat.completions.create(
                **req_kwargs,
                stream=True,
            )

    def _create_stream_completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        request_kwargs = self._build_request_kwargs(messages, **kwargs)
        has_cache_options = ("prompt_cache_key" in request_kwargs) or any(
            isinstance(msg.get("content"), list) for msg in messages if isinstance(msg, dict)
        )
        if getattr(self, "_cache_header_unsupported", False) and has_cache_options:
            request_kwargs = self._strip_cache_options(messages, request_kwargs)
            has_cache_options = False
        try:
            return self._open_stream(request_kwargs)
        except Exception as exc:
            if has_cache_options and self._cache_options_unsupported(exc):
                self._cache_header_unsupported = True
                stripped_kwargs = self._strip_cache_options(messages, request_kwargs)
                return self._open_stream(stripped_kwargs)
            raise

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


class CacheAwareToolAgent(ToolAwareSimpleAgent):
    """ToolAwareSimpleAgent subclass supporting KV-Cache structured PromptEnvelope dispatch."""

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:  # type: ignore[override]
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return f"❌ 错误：未找到工具 '{tool_name}'"
        try:
            parsed = self._sanitize_parameters(self._parse_tool_parameters(tool_name, parameters))
            result = tool.run(parsed)
            formatted = f"🔧 工具 {tool_name} 执行结果：\n{result}"
        except MCPHitlApprovalRequired:
            raise
        except Exception as exc:
            parsed = {}
            formatted = f"❌ 工具调用失败：{exc}"
        if getattr(self, "_tool_call_listener", None):
            try:
                self._tool_call_listener(
                    {
                        "agent_name": self.name,
                        "tool_name": tool_name,
                        "raw_parameters": parameters,
                        "parsed_parameters": parsed,
                        "result": formatted,
                    }
                )
            except Exception:
                logger.exception("Tool call listener failed")
        return formatted

    def _extract_tool_section(self) -> str:
        if not getattr(self, "enable_tool_calling", False) or getattr(self, "tool_registry", None) is None:
            return ""
        getter = getattr(self, "_get_enhanced_system_prompt", None)
        if callable(getter):
            base_prompt = getattr(self, "system_prompt", None) or "你是一个有用的AI助手。"
            enhanced = str(getter() or "")
            if enhanced.startswith(base_prompt):
                return enhanced[len(base_prompt) :].strip()
            return enhanced.strip()
        registry = getattr(self, "tool_registry", None)
        if registry is not None and hasattr(registry, "get_tools_description"):
            desc = str(registry.get_tools_description() or "").strip()
            if desc and desc != "暂无可用工具":
                return f"## 可用工具\n{desc}"
        return ""

    def _prepare_envelope_messages(
        self,
        envelope: PromptEnvelope,
        *,
        config: Configuration | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        history_messages: list[dict[str, Any]] = []
        for msg in getattr(self, "_history", None) or ():
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            if role:
                history_messages.append({"role": role, "content": content or ""})
        builder = ContextPrefixBuilder(config) if config is not None else None
        provider_mode = builder.resolve_provider_mode() if builder is not None else "openai"
        if builder is not None and not builder.enabled():
            provider_mode = "disabled"
        return ProviderCacheAdapter.render_messages_and_kwargs(
            envelope,
            provider_mode=provider_mode,
            enhanced_tool_section=self._extract_tool_section(),
            history_messages=history_messages,
            config=config,
        )

    def _record_envelope_history(self, user_text: str, assistant_text: str) -> None:
        add_msg = getattr(self, "add_message", None)
        if not callable(add_msg):
            return
        try:
            from hello_agents.core.message import Message

            add_msg(Message(user_text, "user"))
            add_msg(Message(assistant_text, "assistant"))
        except Exception:
            pass

    def run_envelope(
        self,
        envelope: PromptEnvelope,
        *,
        config: Configuration | None = None,
        max_tool_iterations: int = 3,
        **kwargs: Any,
    ) -> str:
        llm = getattr(self, "llm", None)
        if llm is None or not hasattr(llm, "invoke"):
            return self.run(envelope.to_flat_prompt(), **kwargs)

        messages, cache_kwargs = self._prepare_envelope_messages(envelope, config=config)
        invoke_kwargs = {**cache_kwargs, **kwargs}
        current_iteration = 0
        final_response = ""
        tools_enabled = bool(getattr(self, "enable_tool_calling", False)) and callable(
            getattr(self, "_parse_tool_calls", None)
        )

        while current_iteration < max_tool_iterations:
            response = llm.invoke(messages, **invoke_kwargs)
            tool_calls = self._parse_tool_calls(response) if tools_enabled else []
            if tool_calls:
                messages.append({"role": "assistant", "content": response})
                tool_results = [
                    self._execute_tool_call(call["tool_name"], call["parameters"])
                    for call in tool_calls
                ]
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "工具执行结果：\n"
                            + "\n\n".join(tool_results)
                            + "\n\n请基于这些结果给出完整的回答。"
                        ),
                    }
                )
                current_iteration += 1
                continue
            final_response = response
            break

        if current_iteration >= max_tool_iterations and not final_response:
            final_response = llm.invoke(messages, **invoke_kwargs)

        self._record_envelope_history(envelope.user_text(), final_response)
        return final_response

    def stream_run_envelope(
        self,
        envelope: PromptEnvelope,
        *,
        config: Configuration | None = None,
        max_tool_iterations: int = 3,
        **kwargs: Any,
    ) -> Iterator[str]:
        llm = getattr(self, "llm", None)
        if llm is None or not hasattr(llm, "stream_invoke"):
            if hasattr(self, "stream_run"):
                yield from self.stream_run(envelope.to_flat_prompt(), **kwargs)
                return
            yield self.run(envelope.to_flat_prompt(), **kwargs)
            return

        messages, cache_kwargs = self._prepare_envelope_messages(envelope, config=config)
        invoke_kwargs = {**cache_kwargs, **kwargs}
        marker = "[TOOL_CALL:"
        tools_enabled = bool(getattr(self, "enable_tool_calling", False)) and callable(
            getattr(self, "_parse_tool_calls", None)
        )
        find_end = getattr(self, "_find_tool_call_end", None)
        final_segments: list[str] = []
        final_response_text = ""
        current_iteration = 0

        while current_iteration < max_tool_iterations:
            residual = ""
            segments_this_round: list[str] = []
            tool_call_texts: list[str] = []

            def process_residual(final_pass: bool = False) -> Iterator[str]:
                nonlocal residual
                while True:
                    start = residual.find(marker)
                    if start == -1:
                        safe_len = (
                            len(residual)
                            if final_pass
                            else max(0, len(residual) - (len(marker) - 1))
                        )
                        if safe_len > 0:
                            seg = residual[:safe_len]
                            residual = residual[safe_len:]
                            yield seg
                        break
                    if start > 0:
                        seg = residual[:start]
                        residual = residual[start:]
                        if seg:
                            yield seg
                        continue
                    end = find_end(residual, 0) if callable(find_end) else -1
                    if end == -1:
                        if final_pass and residual:
                            yield residual
                            residual = ""
                        break
                    tool_call_texts.append(residual[: end + 1])
                    residual = residual[end + 1 :]

            for chunk in llm.stream_invoke(messages, **invoke_kwargs):
                if not chunk:
                    continue
                residual += chunk
                for seg in process_residual():
                    if not seg:
                        continue
                    segments_this_round.append(seg)
                    final_segments.append(seg)
                    yield seg

            for seg in process_residual(final_pass=True):
                if not seg:
                    continue
                segments_this_round.append(seg)
                final_segments.append(seg)
                yield seg

            clean_response = "".join(segments_this_round)
            tool_calls: list[dict[str, Any]] = []
            if tools_enabled:
                for call_text in tool_call_texts:
                    tool_calls.extend(self._parse_tool_calls(call_text))

            if tool_calls:
                messages.append({"role": "assistant", "content": clean_response})
                tool_results = [
                    self._execute_tool_call(call["tool_name"], call["parameters"])
                    for call in tool_calls
                ]
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "工具执行结果：\n"
                            + "\n\n".join(tool_results)
                            + "\n\n请基于这些结果给出完整的回答。"
                        ),
                    }
                )
                current_iteration += 1
                continue

            final_response_text = clean_response
            break

        if current_iteration >= max_tool_iterations and not final_response_text:
            fallback_response = llm.invoke(messages, **invoke_kwargs)
            final_segments.append(fallback_response)
            final_response_text = fallback_response
            yield fallback_response

        self._record_envelope_history(
            envelope.user_text(),
            final_response_text or "".join(final_segments),
        )


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
        tenant_id: str | None = None,
        mcp_bridge: MCPBridgeManager | None = None,
    ) -> None:
        """Initialise the coordinator with configuration and shared tools."""
        self.config = config or Configuration.from_env()
        self.request_id = request_id or "local-request"
        self.tenant_id = (
            str(
                tenant_id
                or getattr(self.config, "default_tenant_id", "default")
                or "default"
            ).strip()
            or "default"
        )
        self._mcp_bridge = mcp_bridge
        self._owns_mcp_bridge = mcp_bridge is None
        self._active_state: SummaryState | None = None
        self.llm = self._init_llm()
        self._content_only_llm = self._init_llm(allow_reasoning_fallback=False)
        self.fast_llm = self._init_fast_llm(allow_reasoning_fallback=True)
        self._fast_content_only_llm = self._init_fast_llm(allow_reasoning_fallback=False)
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
        self._request_state_store = build_request_state_store(self.config)
        self._request_snapshot_lock = Lock()
        self._policy_controller = build_policy_controller(self.config)
        self._note_memory = NoteMemoryService(self.config)
        self._strategy_synthesizer = StrategySynthesizer(
            lambda: self._create_tool_aware_agent(
                name="策略记忆提炼器",
                system_prompt="你是一名策略记忆提炼器，只输出严格 JSON。",
                llm=self._fast_content_only_llm,
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
        self._observability = LangfuseTraceAdapter(self.config)

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
        self._multimodal_parser = MultimodalEvidenceParser(
            enabled=bool(getattr(self.config, "multimodal_evidence_enabled", False)),
            vlm_enabled=bool(getattr(self.config, "multimodal_vlm_enabled", False)),
            vlm_model=getattr(self.config, "multimodal_vlm_model", None),
            vlm_base_url=self.config.resolved_multimodal_vlm_base_url(),
            vlm_api_key=self.config.resolved_multimodal_vlm_api_key(),
            vlm_timeout_seconds=self.config.multimodal_vlm_timeout_seconds,
            max_vlm_concurrency=int(getattr(self.config, "multimodal_vlm_max_concurrency", 4) or 4),
            max_pdf_pages=int(getattr(self.config, "multimodal_max_pdf_pages", 20) or 20),
            max_pdf_bytes=self.config.multimodal_max_pdf_bytes,
            pdf_render_dpi=self.config.multimodal_pdf_render_dpi,
            max_vlm_pages_per_doc=self.config.multimodal_max_vlm_pages_per_doc,
            max_images_per_page=self.config.multimodal_max_images_per_page,
            max_image_bytes=int(getattr(self.config, "multimodal_image_max_bytes", 2 * 1024 * 1024) or (2 * 1024 * 1024)),
            max_image_pixels=int(getattr(self.config, "multimodal_image_max_pixels", 16_000_000) or 16_000_000),
            metrics=metrics_registry,
        )
        self._sandbox_service = SandboxExecutorService(
            enabled=self.config.resolved_sandbox_effective_enabled(),
            backend_preference=str(getattr(self.config, "sandbox_backend", "auto") or "auto"),
            timeout_seconds=float(getattr(self.config, "sandbox_timeout_seconds", 10.0) or 10.0),
            memory_mb=int(getattr(self.config, "sandbox_memory_mb", 512) or 512),
            max_executions_per_task=self.config.sandbox_max_executions_per_task,
            docker_backend=DockerSandboxBackend(
                image=str(getattr(self.config, "sandbox_docker_image", "helloagents-sandbox:latest") or "helloagents-sandbox:latest"),
                cpus=str(getattr(self.config, "sandbox_cpus", "1.0") or "1.0"),
            ),
            pyodide_backend=PyodideSandboxBackend(
                pyodide_cache_dir=str(getattr(self.config, "sandbox_pyodide_cache_dir", "") or ""),
                node_bin=self.config.sandbox_pyodide_node_bin,
            ),
            metrics=metrics_registry,
        )
        self.fetch_page_tool = FetchPageTool(
            evidence_store=self._evidence_store,
            timeout_seconds=float(self.config.search_tool_timeout_seconds or 10.0),
            multimodal_parser=self._multimodal_parser,
        )
        self.evidence_lookup_tool = EvidenceLookupTool(
            evidence_store=self._evidence_store,
            retrieval_service=self._evidence_retrieval,
            request_id_getter=lambda: self.request_id,
        )
        self.run_python_sandbox_tool = RunPythonSandboxTool(
            sandbox_service=self._sandbox_service,
            evidence_store=self._evidence_store,
            config=self.config,
            tenant_id=self.tenant_id,
        )
        self.tools_registry.register_tool(self.search_web_tool)
        self.tools_registry.register_tool(self.fetch_page_tool)
        self.tools_registry.register_tool(self.evidence_lookup_tool)
        if self.note_tool:
            self.tools_registry.register_tool(self.note_tool)
        self._core_tools = [
            self.search_web_tool,
            self.fetch_page_tool,
            self.evidence_lookup_tool,
            *([self.note_tool] if self.note_tool else []),
        ]
        self._base_tools_registry = self.tools_registry

        self._tool_tracker = ToolCallTracker(
            self.config.notes_workspace if self.config.enable_notes else None
        )
        self._tool_event_sink_enabled = False
        self._state_lock = Lock()

        self.todo_agent = self._create_tool_aware_agent(
            name="研究规划专家",
            system_prompt=todo_planner_system_prompt.strip(),
            llm=self.fast_llm,
        )
        self.report_agent = self._create_tool_aware_agent(
            name="报告撰写专家",
            system_prompt=report_writer_instructions.strip(),
            llm=self._content_only_llm,
        )
        self.reflection_agent = self._create_tool_aware_agent(
            name="研究覆盖评估专家",
            system_prompt=request_reflection_system_prompt.strip(),
            llm=self._fast_content_only_llm,
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
            llm=self._fast_content_only_llm,
            enable_tool_calling=False,
        )

        self._summarizer_factory: Callable[[], ToolAwareSimpleAgent] = lambda: self._create_tool_aware_agent(  # noqa: E501
            name="任务总结专家",
            system_prompt=task_summarizer_instructions.strip(),
            llm=self._fast_content_only_llm,
        )

        self.planner = PlanningService(self.todo_agent, self.config)
        self.reflection = ReflectionService(self.reflection_agent, self.config)
        self._claim_verification_orchestrator = ClaimVerificationOrchestrator(
            self.config,
            retrieval_service=self._evidence_retrieval,
            evidence_store=self._evidence_store,
            fast_llm=self._fast_content_only_llm,
        )
        self.reviewer = ReviewService(
            self.review_agent if self.config.review_agent_enabled else None,
            self.config,
            retrieval_service=self._evidence_retrieval,
            request_id_getter=lambda: self.request_id,
            claim_verification_orchestrator=self._claim_verification_orchestrator,
            evidence_store=self._evidence_store,
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
        if self.config.resolved_sandbox_effective_enabled():
            self._refresh_scoped_tools_for_request()
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

    def _init_fast_llm(self, *, allow_reasoning_fallback: bool = True) -> HelloAgentsLLM:
        """Instantiate an optional fast-tier LLM for planning/summarization/reflection.

        Falls back directly to the primary LLM instance when ``fast_llm_model_id`` is unset.
        """
        if not self.config.resolved_fast_llm_enabled():
            return self.llm if allow_reasoning_fallback else self._content_only_llm

        llm_kwargs: dict[str, Any] = {
            "temperature": float(self.config.fast_llm_temperature or 0.0),
            "model": self.config.resolved_fast_llm_model(),
        }
        provider = (self.config.fast_llm_provider or self.config.llm_provider or "").strip()
        if provider:
            llm_kwargs["provider"] = provider

        base_url = (self.config.fast_llm_base_url or "").strip()
        if not base_url:
            if provider == "ollama":
                base_url = self.config.sanitized_ollama_url()
            elif provider == "lmstudio":
                base_url = self.config.lmstudio_base_url
            elif self.config.llm_base_url:
                base_url = self.config.llm_base_url
        if base_url:
            llm_kwargs["base_url"] = base_url

        api_key = (self.config.fast_llm_api_key or self.config.llm_api_key or "").strip()
        if api_key:
            llm_kwargs["api_key"] = api_key
        elif provider == "ollama":
            llm_kwargs["api_key"] = "ollama"

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

        agent_cls = (
            CacheAwareToolAgent
            if isinstance(ToolAwareSimpleAgent, type) and issubclass(CacheAwareToolAgent, ToolAwareSimpleAgent)
            else ToolAwareSimpleAgent
        )
        return agent_cls(
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
        output_format: str | None = None,
    ) -> None:
        """Pick research skill pack bundle (primary + secondary + output format) before planning."""

        self._skill_selection = None
        if getattr(self.config, "active_mcp_tool_scopes", None):
            self._base_config = getattr(self, "_base_config", None) or self.config
            self._swap_runtime_config(self.config.model_copy(update={"active_mcp_tool_scopes": {}}))
        if not self._skill_pack.enabled:
            self._refresh_scoped_tools_for_request()
            return
        try:
            selection = self._skill_pack.select(
                topic or "",
                output_format=output_format or getattr(self.config, "default_output_format", None),
            )
        except Exception as exc:  # pragma: no cover - selection must never fail a request
            logger.warning("skill pack selection failed error=%s", exc)
            if observer is not None:
                observer.record_degraded("skill_pack_invalid")
            self._refresh_scoped_tools_for_request()
            return
        if selection is None:
            # Even the neutral fallback is missing/invalid; proceed with built-in prompts.
            if observer is not None:
                observer.record_degraded("skill_pack_unavailable")
            self._refresh_scoped_tools_for_request()
            return
        self._skill_selection = selection
        applied_overrides: dict[str, Any] = {}
        secondary_pack = getattr(selection, "secondary_pack", None)
        format_pack = getattr(selection, "format_pack", None)
        if not selection.shadow and (not selection.pack.is_neutral or format_pack is not None):
            self._apply_request_config_overrides(
                selection.pack,
                applied_overrides,
                secondary_pack=secondary_pack,
                format_pack=format_pack,
            )
        if state is not None:
            state.skill_pack_metadata = {
                "pack_id": selection.pack.pack_id,
                "version": str(selection.pack.version),
                "topic_domain": selection.pack.topic_domain,
                "match_score": round(float(selection.match_score), 4),
                "fallback_reason": selection.fallback_reason,
                "shadow": bool(selection.shadow),
                "applied_overrides": applied_overrides,
                "secondary_pack_id": secondary_pack.pack_id if secondary_pack is not None else None,
                "secondary_match_score": round(
                    float(getattr(selection, "secondary_match_score", 0.0) or 0.0),
                    4,
                ),
                "format_pack_id": format_pack.pack_id if format_pack is not None else None,
                "routing_method": str(getattr(selection, "routing_method", "lexical") or "lexical"),
                "active_verifiers": list(getattr(selection, "active_verifiers", []) or []),
            }
        if observer is not None:
            observer.record_skill_pack_selection(
                pack_id=selection.pack.pack_id,
                version=str(selection.pack.version),
                match_score=selection.match_score,
                fallback_reason=selection.fallback_reason,
                shadow=selection.shadow,
            )
        self._refresh_scoped_tools_for_request()

    def _ensure_mcp_bridge(self) -> MCPBridgeManager | None:
        if not bool(getattr(self.config, "mcp_enabled", False)):
            return None
        if getattr(self, "_mcp_bridge", None) is None:
            self._mcp_bridge = MCPBridgeManager(self.config)
        return self._mcp_bridge

    def _build_stage_tool_registry(self, stage: str) -> ToolRegistry | None:
        mcp_enabled = bool(getattr(self.config, "mcp_enabled", False))
        sandbox_enabled = bool(
            getattr(self.config, "resolved_sandbox_effective_enabled", lambda: False)()
        )
        base_reg = getattr(self, "_base_tools_registry", None)
        if base_reg is None:
            base_reg = getattr(self, "tools_registry", None)
        mcp_adapters: list[Any] = []
        if mcp_enabled and getattr(self.config, "mcp_max_tools_per_stage", 6):
            bridge = self._ensure_mcp_bridge()
            if bridge is not None:
                try:
                    mcp_adapters = bridge.build_tool_adapters(
                        evidence_store=self._evidence_store,
                        tenant_id_getter=lambda: self.tenant_id,
                        request_id_getter=lambda: self.request_id,
                        active_scopes_getter=lambda: getattr(self.config, "active_mcp_tool_scopes", None),
                        stage=stage,
                        approved_registry_getter=lambda: getattr(self._active_state, "approved_tool_calls", None),
                        consume_approval=self._consume_mcp_approval if self._request_state_store is not None else None,
                        observer_getter=lambda: self._request_trace,
                    )
                except Exception as exc:
                    logger.warning("MCP adapter discovery failed stage=%s error=%s", stage, exc)
                    mcp_adapters = []
        stage_core_tools = list(getattr(self, "_core_tools", ()))
        if sandbox_enabled and stage in {"task_execution", "review"}:
            sandbox_tool = getattr(self, "run_python_sandbox_tool", None)
            if sandbox_tool is not None and sandbox_tool not in stage_core_tools:
                stage_core_tools.append(sandbox_tool)
        if not mcp_enabled and len(stage_core_tools) == len(getattr(self, "_core_tools", ())):
            return base_reg
        return ScopedToolRegistry(
            core_tools=stage_core_tools,
            mcp_adapters=mcp_adapters,
            active_scopes=getattr(self.config, "active_mcp_tool_scopes", None),
            stage=stage,
            max_mcp_tools=int(getattr(self.config, "mcp_max_tools_per_stage", 6)),
            enforce_readonly=bool(getattr(self.config, "mcp_enforce_readonly", True)),
        )

    def _refresh_scoped_tools_for_request(self) -> None:
        base_reg = getattr(self, "_base_tools_registry", None)
        if base_reg is None:
            base_reg = getattr(self, "tools_registry", None)
        mcp_enabled = bool(getattr(self.config, "mcp_enabled", False))
        sandbox_enabled = bool(
            getattr(self.config, "resolved_sandbox_effective_enabled", lambda: False)()
        )
        if not mcp_enabled and not sandbox_enabled:
            if base_reg is not None:
                self.tools_registry = base_reg
                for stage_agent in (
                    getattr(self, "todo_agent", None),
                    getattr(self, "review_agent", None),
                    getattr(self, "report_agent", None),
                ):
                    if stage_agent is not None and hasattr(stage_agent, "tool_registry"):
                        stage_agent.tool_registry = base_reg
            return
        self.tools_registry = self._build_stage_tool_registry("task_execution") or base_reg
        stage_agents = (
            ("planning", getattr(self, "todo_agent", None)),
            ("review", getattr(self, "review_agent", None)),
            ("report", getattr(self, "report_agent", None)),
        )
        for stage_name, stage_agent in stage_agents:
            if stage_agent is not None and hasattr(stage_agent, "tool_registry"):
                stage_reg = self._build_stage_tool_registry(stage_name)
                if stage_reg is not None:
                    stage_agent.tool_registry = stage_reg

    def _apply_request_config_overrides(
        self,
        pack: Any,
        applied_out: dict[str, Any],
        *,
        secondary_pack: Any = None,
        format_pack: Any = None,
    ) -> None:
        """Apply request-scoped skill pack config overrides and propagate to sub-services."""

        base_config = getattr(self, "_base_config", None) or self.config
        self._base_config = base_config
        effective_config, applied = apply_skill_pack_overrides(
            base_config,
            pack,
            secondary_pack=secondary_pack,
            format_pack=format_pack,
        )
        if effective_config is base_config and not applied:
            return
        applied_out.update(applied)
        self._swap_runtime_config(effective_config)

    def _restore_request_config(self) -> None:
        base_config = getattr(self, "_base_config", None)
        if base_config is not None:
            self._swap_runtime_config(base_config)
            self._base_config = None
        base_reg = getattr(self, "_base_tools_registry", None)
        if base_reg is not None:
            self.tools_registry = base_reg
            for stage_agent in (
                getattr(self, "todo_agent", None),
                getattr(self, "review_agent", None),
                getattr(self, "report_agent", None),
            ):
                if stage_agent is not None and hasattr(stage_agent, "tool_registry"):
                    stage_agent.tool_registry = base_reg

    def _swap_runtime_config(self, target_config: Configuration) -> None:
        self.config = target_config
        for attr in ("planner", "summarizer", "reporting", "reflection", "reviewer"):
            service = getattr(self, attr, None)
            if service is not None and hasattr(service, "_config"):
                service._config = target_config
        sandbox_service = getattr(self, "_sandbox_service", None)
        if sandbox_service is not None:
            sandbox_service.enabled = bool(
                getattr(target_config, "resolved_sandbox_effective_enabled", lambda: False)()
            )
            sandbox_service.timeout_seconds = float(
                getattr(target_config, "sandbox_timeout_seconds", sandbox_service.timeout_seconds)
                or sandbox_service.timeout_seconds
            )
            sandbox_service.memory_mb = int(
                getattr(target_config, "sandbox_memory_mb", sandbox_service.memory_mb)
                or sandbox_service.memory_mb
            )
            sandbox_service.max_executions_per_task = int(
                getattr(
                    target_config,
                    "sandbox_max_executions_per_task",
                    sandbox_service.max_executions_per_task,
                )
                or sandbox_service.max_executions_per_task
            )
        sandbox_tool = getattr(self, "run_python_sandbox_tool", None)
        if sandbox_tool is not None and hasattr(sandbox_tool, "_config"):
            sandbox_tool._config = target_config
        parser = getattr(self, "_multimodal_parser", None)
        if parser is not None:
            parser.enabled = bool(getattr(target_config, "multimodal_evidence_enabled", parser.enabled))
            parser.vlm_enabled = bool(getattr(target_config, "multimodal_vlm_enabled", parser.vlm_enabled))
        orchestrator = getattr(self, "_claim_verification_orchestrator", None)
        if orchestrator is not None:
            orchestrator._config = target_config
            if getattr(orchestrator, "_scorer", None) is not None:
                orchestrator._scorer._config = target_config
            if getattr(orchestrator, "_conflict_builder", None) is not None:
                orchestrator._conflict_builder._config = target_config

    def _refresh_context_checkpoints(self, state: SummaryState) -> None:
        """Advance frozen FactLedger and EvidenceIndexCard checkpoints between waves/stages."""

        changed = False
        evidence_store = getattr(self, "_evidence_store", None)
        if getattr(self.config, "context_compaction_enabled", True):
            valid_ids: set[str] = set()
            for task in state.todo_items:
                for item in task.evidence_items or ():
                    sid = str(item.get("source_id") or "").strip()
                    if sid:
                        valid_ids.add(sid)
            if evidence_store is not None:
                for item in evidence_store.lookup() or ():
                    sid = str(item.get("source_id") or "").strip()
                    if sid:
                        valid_ids.add(sid)
            existing = FactLedger.from_dict(getattr(state, "fact_ledger", None))
            updated = FactLedgerCompactor().update_ledger(
                existing,
                state.todo_items,
                valid_source_ids=valid_ids,
            )
            new_ledger = updated.to_dict()
            if new_ledger != state.fact_ledger:
                state.fact_ledger = new_ledger
                changed = True

        if getattr(self.config, "evidence_pointer_disclosure_enabled", True) and evidence_store is not None:
            runtime_idx = getattr(self, "_evidence_runtime", None)
            cards = build_evidence_index_cards(
                evidence_store,
                runtime_index=runtime_idx,
            )
            if cards != state.evidence_index_cards:
                state.evidence_index_cards = cards
                changed = True

        if changed or not int(getattr(state, "frozen_prefix_version", 0) or 0):
            state.frozen_prefix_version = int(getattr(state, "frozen_prefix_version", 0) or 0) + 1

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
        """Render primary + optional secondary/format skill blocks for one stage."""

        selection = self._skill_selection
        if selection is None or selection.shadow:
            return ""
        parts = [self._skill_pack.render_block(selection.pack, stage)]
        secondary_pack = getattr(selection, "secondary_pack", None)
        if secondary_pack is not None:
            parts.append(self._skill_pack.render_secondary_block(secondary_pack, stage))
        format_pack = getattr(selection, "format_pack", None)
        if format_pack is not None:
            parts.append(self._skill_pack.render_format_block(format_pack, stage))
        return "".join(part for part in parts if part)

    @staticmethod
    def _callable_accepts_kwarg(fn: Any, kwarg_name: str) -> bool:
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD or p.name == kwarg_name
            for p in sig.parameters.values()
        )

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
        def _run_refresh() -> None:
            self._flush_strategy_memory_usage(observer=observer)
            self._strategy_memory.refresh_request(request_id, observer=observer)

        if bool(getattr(self.config, "async_memory_refresh_enabled", False)):
            Thread(
                target=_run_refresh,
                name=f"strategy-memory-refresh-{request_id[:8]}",
                daemon=True,
            ).start()
            return
        _run_refresh()

    def _is_cancelled(self) -> bool:
        cancel_event = getattr(self, "_cancel_event", None)
        return bool(cancel_event is not None and cancel_event.is_set())

    def _export_observability_trace(self, snapshot: dict[str, Any] | None) -> None:
        adapter = getattr(self, "_observability", None)
        if adapter is None or not snapshot:
            return
        adapter.export_request_trace(snapshot)

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

        active_verifiers = list(
            getattr(self._skill_selection, "active_verifiers", None)
            or getattr(self.config, "active_skill_verifiers", None)
            or []
        )
        review_kwargs: dict[str, Any] = {
            "observer": observer,
            "skill_pack_context": self._skill_stage_context("review"),
        }
        if active_verifiers and self._callable_accepts_kwarg(self.reviewer.review_request, "active_verifiers"):
            review_kwargs["active_verifiers"] = active_verifiers
        review_summary = self.reviewer.review_request(state, **review_kwargs)

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
        cancel_event: Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run the unified 6-stage workflow for both sync and streaming callers."""
        self._cancel_event = cancel_event
        self._active_state = state
        pending_approval = getattr(state, "pending_tool_approval", None)
        if resumed and isinstance(pending_approval, dict) and pending_approval.get("status") == "approved_once":
            state.pending_tool_approval = None
        report = state.structured_report or ""
        request_completed = False
        try:
            self._select_skill_pack(state.research_topic or topic, observer, state=state)
            if emit_stream:
                logger.debug("Starting streaming research: topic=%s", topic)
                yield {"type": "status", "message": "初始化研究流程"}
            if self._is_cancelled():
                observer.record_degraded("client_disconnected")
                observer.complete_request(status="cancelled")
                request_completed = True
                self._persist_request_state(state, phase="cancelled", status="cancelled", report_markdown=report)
                return
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
                completed_ids = {
                    int(task.id) for task in state.todo_items if task.status == "completed"
                }
                waves = partition_into_waves(pending_tasks, completed_ids=completed_ids)
                if len(waves) <= 1:
                    if emit_stream:
                        for event in self._execute_task_batch_stream(state, pending_tasks, channel_map):
                            yield event
                    else:
                        self._execute_task_batch_sync(state, pending_tasks)
                    self._refresh_context_checkpoints(state)
                    self._persist_request_state(state, phase="task_execution", status="in_progress")
                else:
                    total_waves = len(waves)
                    for wave_index, wave_tasks in enumerate(waves, start=1):
                        if self._is_cancelled():
                            break
                        if wave_index > 1:
                            compact_enabled = bool(
                                getattr(self.config, "context_compaction_enabled", True)
                            )
                            for downstream_task in wave_tasks:
                                downstream_task.upstream_context = build_upstream_context(
                                    downstream_task,
                                    state.todo_items,
                                    fact_ledger=getattr(state, "fact_ledger", None),
                                    compact_enabled=compact_enabled,
                                )
                        if emit_stream:
                            yield {
                                "type": "wave_started",
                                "wave": wave_index,
                                "total_waves": total_waves,
                                "task_ids": [task.id for task in wave_tasks],
                            }
                            for event in self._execute_task_batch_stream(state, wave_tasks, channel_map):
                                yield event
                            yield {
                                "type": "wave_completed",
                                "wave": wave_index,
                                "total_waves": total_waves,
                                "task_ids": [task.id for task in wave_tasks],
                            }
                        else:
                            self._execute_task_batch_sync(state, wave_tasks)
                        self._refresh_context_checkpoints(state)
                        self._persist_request_state(state, phase="task_execution", status="in_progress")

            if self._is_cancelled():
                observer.record_degraded("client_disconnected")
                observer.complete_request(status="cancelled")
                request_completed = True
                self._persist_request_state(state, phase="cancelled", status="cancelled", report_markdown=report)
                return

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
                                self._refresh_context_checkpoints(state)
                        except MCPHitlApprovalRequired:
                            raise
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
                except MCPHitlApprovalRequired:
                    raise
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
                            self._refresh_context_checkpoints(state)
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
            if self._is_cancelled():
                observer.record_degraded("client_disconnected")
                observer.complete_request(status="cancelled")
                request_completed = True
                self._persist_request_state(state, phase="cancelled", status="cancelled", report_markdown=report)
                return
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
                    if emit_stream and hasattr(self.reporting, "stream_generate_report"):
                        stream_kwargs: dict[str, Any] = {
                            "observer": observer,
                            "skill_pack_context": self._skill_stage_context("report"),
                        }
                        selection = getattr(self, "_skill_selection", None)
                        fmt_pack = getattr(selection, "format_pack", None) if selection is not None and not selection.shadow else None
                        fmt_outline = getattr(fmt_pack, "outline_sections", None) if fmt_pack is not None else None
                        if fmt_outline and self._callable_accepts_kwarg(
                            self.reporting.stream_generate_report,
                            "outline_sections",
                        ):
                            stream_kwargs["outline_sections"] = list(fmt_outline)
                        for report_event in self.reporting.stream_generate_report(
                            state,
                            **stream_kwargs,
                        ):
                            report_event.setdefault("step", final_step)
                            yield report_event
                        report = state.structured_report or ""
                    else:
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
            snapshot = observer.complete_request(status=request_status)
            request_completed = True
            self._export_observability_trace(snapshot)
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
                workbench = build_report_workbench_payload(
                    state,
                    report,
                    evidence_store=self._evidence_store,
                    config=self.config,
                )
                has_workbench = bool(
                    workbench.get("citation_index")
                    or workbench.get("evidence_conflicts")
                    or workbench.get("claim_verification_summary")
                    or (workbench.get("skill_governance") or {}).get("pack_id", "_neutral") != "_neutral"
                )
                yield {
                    "type": "final_report",
                    "report": report,
                    "note_id": state.report_note_id,
                    "note_path": state.report_note_path,
                    **({"charts_spec": list(state.report_charts)} if state.report_charts else {}),
                    "evidence_conflicts": list(getattr(state, "evidence_conflicts", None) or []),
                    "claim_verification_summary": dict(
                        getattr(state, "claim_verification_summary", None) or {}
                    ),
                    **({"workbench": workbench} if has_workbench else {}),
                }
                yield {"type": "done"}
        except MCPHitlApprovalRequired as hitl_exc:
            pending_payload = hitl_exc.to_dict()
            pending_payload.update(request_id=self.request_id, tenant_id=self.tenant_id)
            state.pending_tool_approval = pending_payload
            if not request_completed:
                observer.complete_request(status="awaiting_approval")
                request_completed = True
            self._persist_request_state(
                state,
                phase="awaiting_tool_approval",
                status="awaiting_approval",
                report_markdown=report,
            )
            if emit_stream:
                yield {
                    "type": "tool_approval_required",
                    "request_id": self.request_id,
                    **{key: value for key, value in pending_payload.items() if key != "tenant_id"},
                }
                yield observer.metrics_event()
            return
        except Exception as exc:
            if not request_completed:
                failed_snapshot = observer.complete_request(status="failed", error=exc)
                self._export_observability_trace(failed_snapshot)
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
            self._active_state = None
            self._cancel_event = None
            self._restore_request_config()
            if getattr(self, "_owns_mcp_bridge", False) and self._mcp_bridge is not None:
                bridge, self._mcp_bridge = self._mcp_bridge, None
                bridge.close()

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
            report_charts=list(getattr(state, "report_charts", None) or []),
            evidence_conflicts=list(getattr(state, "evidence_conflicts", None) or []),
            claim_verification_summary=dict(
                getattr(state, "claim_verification_summary", None) or {}
            ),
            todo_items=state.todo_items,
        )

    def run_resume(self, request_id: str) -> SummaryStateOutput:
        """Resume a persisted request snapshot."""

        snapshot = self._load_resume_snapshot(request_id)
        state, phase = self._state_from_snapshot(snapshot)
        topic = state.research_topic or str(snapshot.get("topic") or "").strip()
        return self.run(topic, initial_state=state, resume_phase=phase)

    def execute_single_task(self, state: SummaryState, task_id: int) -> TodoItem:
        """Execute one planned TodoItem in-place for durable step/activity orchestration."""

        target = next((task for task in state.todo_items if task.id == task_id), None)
        if target is None:
            raise ValueError(f"task_id not found: {task_id}")
        for _ in self._execute_task(state, target, emit_stream=False):
            pass
        return target

    def plan_only(self, topic: str) -> dict[str, Any]:
        """Generate an editable research plan and persist it with awaiting_approval status."""

        observer = self._start_request_trace(topic)
        state = SummaryState(research_topic=topic)
        self._select_skill_pack(topic, observer, state=state)
        try:
            planning_span = observer.start_stage("planning", scope="request", metadata={"topic": topic})
            try:
                historical_memory_context = self._planning_memory_context(state, observer)
                state.todo_items = self.planner.plan_todo_list(
                    state,
                    observer=observer,
                    historical_memory_context=historical_memory_context,
                    strategy_memory_context=self._planning_strategy_context(state, observer),
                    skill_pack_context=self._skill_stage_context("planning"),
                )
                planning_span.complete(status="success", metadata={"task_count": len(state.todo_items)})
            except Exception as exc:
                planning_span.complete(status="failed", error=exc)
                observer.complete_request(status="failed", error=exc)
                raise

            if not state.todo_items:
                state.todo_items = [self.planner.create_fallback_task(state)]

            self._apply_task_budget(state, observer)
            observer.set_task_totals(total_tasks=len(state.todo_items))
            observer.complete_request(status="awaiting_approval")
            self._persist_request_state(state, phase="awaiting_approval", status="awaiting_approval")
            return {
                "request_id": self.request_id,
                "topic": topic,
                "status": "awaiting_approval",
                "phase": "awaiting_approval",
                "skill_pack": dict(state.skill_pack_metadata or {}),
                "todo_items": [self._serialize_task(task) for task in state.todo_items],
                "estimated_duration_seconds": HitlService.estimate_duration_seconds(
                    len(state.todo_items), self.config
                ),
            }
        finally:
            self._restore_request_config()

    def drilldown_section(
        self,
        request_id: str,
        *,
        section_title: str,
        question: str,
    ) -> dict[str, Any]:
        """Run a targeted incremental sub-task grounded in existing evidence and merge into a report section."""

        snapshot = self._load_resume_snapshot(request_id)
        state, _ = self._state_from_snapshot(snapshot)
        topic = state.research_topic or str(snapshot.get("topic") or "").strip()
        observer = self._start_request_trace(topic)

        next_id = max((int(task.id) for task in state.todo_items), default=0) + 1
        drilldown_query = " ".join(part for part in (topic, section_title, question) if part.strip()).strip()
        target_task = TodoItem(
            id=next_id,
            title=section_title.strip() or f"深入追问 {next_id}",
            intent=question.strip(),
            query=drilldown_query,
            origin="drilldown",
            round=max((int(getattr(t, "round", 1) or 1) for t in state.todo_items), default=1) + 1,
        )
        state.todo_items.append(target_task)
        observer.set_task_totals(total_tasks=len(state.todo_items))

        for _ in self._execute_task(state, target_task, emit_stream=False):
            pass

        valid_ids = {
            str(item.get("source_id") or "").strip()
            for task in state.todo_items
            for item in (task.evidence_items or [])
            if str(item.get("source_id") or "").strip()
        }
        if self._evidence_store is not None:
            for item in self._evidence_store.list_task_evidence(target_task.id):
                sid = str(item.get("source_id") or "").strip()
                if sid:
                    valid_ids.add(sid)

        addendum = HitlService.sanitize_citations(target_task.summary or "", valid_ids)
        new_references = [
            render_references([item]).splitlines()[-1]
            for item in (target_task.evidence_items or [])
            if item.get("source_id")
        ]
        # Build formatted markdown reference bullets cleanly using existing render_references helper
        rendered_refs = render_references(target_task.evidence_items or [])
        ref_lines = [
            line.strip()
            for line in rendered_refs.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        updated_report = HitlService.merge_section_drilldown(
            state.structured_report or state.running_summary or "",
            section_title=section_title,
            addendum_markdown=addendum or "未检索到可直接补充的新增证据。",
            new_references=ref_lines if addendum else None,
        )
        state.structured_report = updated_report
        state.running_summary = updated_report
        observer.attach_result(
            report_markdown=updated_report,
            todo_items=[self._serialize_task(task) for task in state.todo_items],
            report_note_id=state.report_note_id,
            report_note_path=state.report_note_path,
        )
        completed_status = self._request_status(state, updated_report)
        observer.complete_request(status=completed_status)
        self._persist_request_state(
            state,
            phase="completed",
            status=completed_status,
            report_markdown=updated_report,
        )
        del new_references
        return {
            "request_id": self.request_id,
            "section_title": section_title,
            "question": question,
            "task_id": target_task.id,
            "addendum_markdown": addendum,
            "new_sources": list(target_task.evidence_items or []),
            "updated_report_markdown": updated_report,
        }

    def run_stream(
        self,
        topic: str,
        *,
        initial_state: SummaryState | None = None,
        resume_phase: str | None = None,
        cancel_event: Event | None = None,
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
            cancel_event=cancel_event,
        )

    def run_stream_resume(
        self,
        request_id: str,
        *,
        cancel_event: Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Resume a persisted request snapshot in streaming mode."""

        snapshot = self._load_resume_snapshot(request_id)
        state, phase = self._state_from_snapshot(snapshot)
        topic = state.research_topic or str(snapshot.get("topic") or "").strip()
        yield from self.run_stream(
            topic,
            initial_state=state,
            resume_phase=phase,
            cancel_event=cancel_event,
        )

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
            "depends_on": list(getattr(task, "depends_on", []) or []),
            "fact_claims": list(getattr(task, "fact_claims", []) or []),
            "atomic_claims": list(getattr(task, "atomic_claims", []) or []),
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
