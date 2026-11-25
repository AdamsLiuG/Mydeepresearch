"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
import unicodedata
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, RLock, Semaphore, Thread, local
from time import monotonic
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool
from openai import OpenAI

from config import Configuration
from metrics import (
    RequestTrace,
    _estimate_tokens,
    metrics_registry,
    normalize_reported_usage,
    reported_usage_fields,
)
from models import ReportModelResponse, SummaryState, SummaryStateOutput, TodoItem
from prompts import (
    report_writer_instructions,
    request_reflection_system_prompt,
    request_reviewer_system_prompt,
    task_react_plan_prompt,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from repair_orchestrator import RepairOrchestratorMixin
from services.claim_verifier import (
    ClaimVerificationOrchestrator,
    build_verified_graph_candidates,
)
from services.context_builder import (
    ContextPrefixBuilder,
    PromptEnvelope,
    PromptSegment,
    ProviderCacheAdapter,
)
from services.dag_scheduler import (
    ResearchTreeRePlanner,
    apply_dag_patch,
    build_upstream_context,
    normalize_depends_on,
    partition_into_waves,
    score_wave1_branch,
    select_wave_one,
)
from services.evidence import (
    EvidenceLookupTool,
    EvidenceStore,
    FetchPageTool,
    SearchWebTool,
    public_web_evidence_projection,
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
    append_context_checkpoint,
    build_evidence_index_cards,
)
from services.hitl import HitlService
from services.knowledge_graph import (
    GraphEntity,
    GraphQuery,
    KnowledgeGraphService,
    TerminalGraphSnapshot,
    build_graph_store,
    fact_fingerprint,
    receipt_id_for,
    seal_terminal_batch,
    terminal_snapshot_to_payload,
)
from services.mcp_bridge import (
    MCPBridgeManager,
    MCPHitlApprovalRequired,
    ScopedToolRegistry,
)
from services.memory_sanitizer import sanitize_web_text
from services.memory_scope import MemoryPolicy, valid_internal_id
from services.model_router import CascadeModelRouter, RouteDecision, RouteFeatures
from services.multimodal_parser import MultimodalEvidenceParser
from services.note_memory import NoteMemoryService
from services.notes import ScopedNoteTool, ScopedNoteWorkspace
from services.observability import (
    LangfuseTraceAdapter,
    OTelTraceAdapter,
    parse_traceparent,
)
from services.planner import PlanningService
from services.policy_controller import build_policy_controller
from services.reflection import ReflectionAssessment, ReflectionService
from services.reporter import (
    ReportingService,
    build_publication_bindings,
    build_report_workbench_payload,
)
from services.request_state import SnapshotConflictError
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
from services.storage_backends import (
    JsonStrictSnapshotReader,
    PostgresRequestStateStore,
    build_request_state_store,
    build_strict_snapshot_reader,
)
from services.strategy_memory import StrategyMemoryService
from services.strategy_synthesizer import StrategySynthesizer
from services.summarizer import SummarizationService
from services.tenant import TenantContext, authenticated_memory_scope
from services.tool_events import ToolCallTracker
from services.web_extractor import ExtractionContext
from services.web_fetch_gateway import FetchScope
from state_manager import (
    BudgetDispatchDenied,
    ReportCancelled,
    RequestBudgetController,
    RequestStateGraphAuthority,
    StateManagerMixin,
    TenantSnapshotMemoryPolicy,
    materialize_report_event,
    public_report_workbench,
    public_v4_projection,
    record_v4_graph_ack,
    report_chunks,
)
from stream_coordinator import StreamCoordinatorMixin
from task_executor import TaskExecutorMixin

logger = logging.getLogger(__name__)


class SafeHelloAgentsLLM(HelloAgentsLLM):
    """Provide safer local-vLLM response handling for sync and streaming calls."""

    budget_provider: Callable[[], Any] | None = None
    budget_pricing_catalog: dict[str, Any] | None = None

    def __init__(self, *args: Any, allow_reasoning_fallback: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_reasoning_fallback = allow_reasoning_fallback
        self._usage_local = local()
        self._cache_header_unsupported = False
        self._cache_header_lock = Lock()

    def consume_last_usage(self) -> dict[str, int] | None:
        """Return and clear the most recent API-reported token usage on this thread."""
        usage = getattr(self._usage_local, "last_usage", None)
        self._usage_local.last_usage = None
        return dict(usage) if isinstance(usage, dict) else None

    @staticmethod
    def _extract_chunk_usage(chunk: Any) -> dict[str, int] | None:
        return normalize_reported_usage(getattr(chunk, "usage", None))

    def _capture_stream_usage(self, chunk: Any, counters: dict[str, int]) -> None:
        fields = reported_usage_fields(getattr(chunk, "usage", None))
        if fields:
            counters.update(fields)
            self._usage_local.last_usage = normalize_reported_usage(counters)

    def consume_cache_downgrade(self) -> bool:
        downgraded = bool(getattr(self._usage_local, "cache_downgraded", False))
        self._usage_local.cache_downgraded = False
        return downgraded

    @staticmethod
    def _stream_options_unsupported(exc: Exception) -> bool:
        message = str(exc or "").lower()
        if any(field in message for field in ("prompt_cache_key", "cache_control", "ephemeral")):
            return False
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
        message = str(exc or "").lower()
        return any(token in message for token in ("prompt_cache_key", "cache_control", "ephemeral")) or (
            any(token in message for token in ("content", "messages"))
            and any(token in message for token in ("string", "multipart", "array", "unsupported"))
        )

    @staticmethod
    def _has_cache_breakpoint(content: Any) -> bool:
        return isinstance(content, list) and any(
            isinstance(block, dict) and "cache_control" in block for block in content
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
        if any(cls._has_cache_breakpoint(msg.get("content")) for msg in messages):
            stripped["messages"] = [
                {**msg, "content": cls._flatten_message_content(blocks)}
                if isinstance(blocks := msg.get("content"), list)
                and all(isinstance(block, dict) and block.get("type") == "text" for block in blocks)
                else msg
                for msg in messages
            ]
        return stripped

    def _budget_pricing(self, model: str) -> dict[str, Decimal] | None:
        provider_prices = (self.budget_pricing_catalog or {}).get(self.provider, {})
        if not isinstance(provider_prices, dict):
            return None
        entry = provider_prices.get(model, provider_prices.get("default"))
        if not isinstance(entry, dict):
            return None
        try:
            prompt = Decimal(str(entry["prompt_per_1k_tokens"]))
            completion = Decimal(str(entry["completion_per_1k_tokens"]))
            cached = Decimal(str(entry.get("cached_prompt_per_1k_tokens", prompt)))
            creation = Decimal(str(entry.get("cache_creation_per_1k_tokens", prompt)))
            if not all(rate.is_finite() and rate >= 0 for rate in (prompt, completion, cached, creation)):
                return None
        except (InvalidOperation, KeyError, TypeError, ValueError):
            return None
        return {
            "prompt_per_1k_tokens": prompt,
            "completion_per_1k_tokens": completion,
            "cached_prompt_per_1k_tokens": cached,
            "cache_creation_per_1k_tokens": creation,
        }

    @staticmethod
    def _budget_reported_usage(counters: dict[str, int], pricing: dict[str, Decimal]) -> dict[str, int] | None:
        if not (
            counters.keys() & {"prompt_tokens", "input_tokens"}
            and counters.keys() & {"completion_tokens", "output_tokens"}
        ):
            return None
        usage = normalize_reported_usage(counters)
        if usage is None:
            return None
        prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
        if usage["total_tokens"] > prompt + completion:
            return None
        cached = usage.get("cached_tokens", 0)
        creation = usage.get("cache_creation_tokens", 0)
        cost = (
            (prompt - cached - creation) * pricing["prompt_per_1k_tokens"]
            + cached * pricing["cached_prompt_per_1k_tokens"]
            + creation * pricing["cache_creation_per_1k_tokens"]
            + completion * pricing["completion_per_1k_tokens"]
        ) * 1000
        return {
            "input_tokens": prompt, "output_tokens": completion,
            "cost_microusd": int(cost.to_integral_value(rounding=ROUND_CEILING)),
        }

    @contextmanager
    def _provider_stream(self, req_kwargs: dict[str, Any]) -> Iterator[Any]:
        budget = self.budget_provider() if self.budget_provider else None
        if budget is None:
            yield self._client.chat.completions.create(**req_kwargs, stream=True), None, None, None
            return

        request = dict(req_kwargs)
        messages = request["messages"]
        if request.get("n", 1) != 1 or request.get("extra_body") or request.get("audio"):
            budget.refuse("dispatch_unbounded")
        for message in messages:
            content = message.get("content")
            if content is None or isinstance(content, str):
                continue
            if not isinstance(content, list) or any(
                not isinstance(block, dict) or block.get("type") != "text" for block in content
            ):
                budget.refuse("dispatch_unbounded")
        output_tokens = request.get("max_completion_tokens", request.get("max_tokens"))
        if output_tokens is None:
            output_tokens = {"planning": 1024, "report": 2000}.get(budget.stage, 768)
        if type(output_tokens) is not int or output_tokens <= 0:
            budget.refuse("dispatch_unbounded")
        if "max_completion_tokens" in request:
            request.pop("max_tokens", None)
            request["max_completion_tokens"] = output_tokens
        else:
            request["max_tokens"] = output_tokens
        pricing = self._budget_pricing(request["model"])
        # Bound byte-tokenized input including tool schemas and message framing.
        try:
            input_tokens = len(json.dumps(request, ensure_ascii=True).encode("utf-8")) + 128 + 64 * len(messages)
        except (TypeError, ValueError):
            budget.refuse("dispatch_unbounded")
        provider_timeout = request.get("timeout") or self.timeout
        client = self._client
        if isinstance(client, OpenAI):
            # OpenAI retries and HTTP redirects could launch additional HTTP requests.
            client = client.with_options(max_retries=0, http_client=self._budget_http_client())
        with budget.attempt({}, llm=(input_tokens, output_tokens, pricing)) as attempt:
            timeout = min(attempt.timeout, float(provider_timeout))
            deadline = monotonic() + timeout
            request["timeout"] = timeout
            self._usage_local.provider_attempts += 1
            response = client.chat.completions.create(**request, stream=True)
            try:
                yield response, deadline, attempt, pricing
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    def _budget_http_client(self) -> httpx.Client:
        with self._cache_header_lock:
            client = getattr(self, "_bounded_http_client", None)
            if client is None:
                client = self._bounded_http_client = httpx.Client(
                    follow_redirects=False,
                    trust_env=not self._should_bypass_env_proxy(self.base_url),
                )
            return client

    def _open_stream(self, stack: ExitStack, req_kwargs: dict[str, Any]) -> Any:
        try:
            return stack.enter_context(self._provider_stream({
                **req_kwargs, "stream_options": {"include_usage": True},
            }))
        except Exception as exc:
            if not self._stream_options_unsupported(exc):
                raise
            return stack.enter_context(self._provider_stream(req_kwargs))

    def _create_stream_completion(
        self,
        stack: ExitStack,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        request_kwargs = self._build_request_kwargs(messages, **kwargs)
        has_cache_options = ("prompt_cache_key" in request_kwargs) or any(
            self._has_cache_breakpoint(msg.get("content")) for msg in messages
        )
        if getattr(self, "_cache_header_unsupported", False) and has_cache_options:
            request_kwargs = self._strip_cache_options(messages, request_kwargs)
            has_cache_options = False
        try:
            return self._open_stream(stack, request_kwargs)
        except Exception as exc:
            if has_cache_options and self._cache_options_unsupported(exc):
                cache_lock = getattr(self, "_cache_header_lock", None)
                if cache_lock is None:
                    cache_lock = self._cache_header_lock = Lock()
                with cache_lock:
                    if not getattr(self, "_cache_header_unsupported", False):
                        self._cache_header_unsupported = True
                        self._usage_local.cache_downgraded = True
                stripped_kwargs = self._strip_cache_options(messages, request_kwargs)
                return self._open_stream(stack, stripped_kwargs)
            raise

    def _stream_chunks(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[Any]:
        self._usage_local.last_usage = None
        self._usage_local.provider_attempts = 0
        counters: dict[str, int] = {}
        with ExitStack() as stack:
            response, deadline, attempt, pricing = self._create_stream_completion(stack, messages, **kwargs)
            for chunk in response:
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError("LLM stream exceeded its attempt timeout")
                self._capture_stream_usage(chunk, counters)
                yield chunk
            if attempt is not None:
                if monotonic() >= deadline:
                    raise TimeoutError("LLM stream exceeded its attempt timeout")
                attempt.usage = self._budget_reported_usage(counters, pricing)

    def consume_budget_attempt_count(self) -> int:
        count = getattr(self._usage_local, "provider_attempts", 0)
        self._usage_local.provider_attempts = 0
        return count

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
        for chunk in self._stream_chunks(messages, **kwargs):
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
        for chunk in self._stream_chunks(messages, **kwargs):
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

    _envelope_usage_local = local()

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
        scope = getattr(self, "_request_cache_scope", None)
        if scope is not None:
            envelope = replace(envelope, tenant_id=scope[0], request_id=scope[1])
        role = str(getattr(self, "system_prompt", "") or "").strip()
        if role and not any(segment.name == "role" for segment in envelope.system_segments):
            envelope = replace(envelope, system_segments=[
                PromptSegment(name="role", tier="global_static", text=role), *envelope.system_segments,
            ])
        builder = ContextPrefixBuilder(config) if config is not None else None
        provider_mode = (
            builder.resolve_provider_mode(llm=getattr(self, "llm", None)) if builder is not None else "openai"
        )
        tool_section = self._extract_tool_section()
        request_tools = bool(getattr(getattr(self, "tool_registry", None), "injected_mcp_tools", ()))
        return ProviderCacheAdapter.render_messages_and_kwargs(
            envelope,
            provider_mode=provider_mode,
            enhanced_tool_section="" if request_tools else tool_section,
            request_tool_section=tool_section if request_tools else "",
            config=config,
        )

    def consume_last_usage(self) -> dict[str, int] | None:
        usage = getattr(self._envelope_usage_local, "last_usage", None)
        self._envelope_usage_local.last_usage = None
        return usage[1] if usage is not None and usage[0] is self else None

    @contextmanager
    def _measure_provider_turn(self, messages: list[dict[str, Any]], completion: list[str]) -> Iterator[None]:
        try:
            yield
        finally:
            consume = getattr(self.llm, "consume_last_usage", None)
            usage = normalize_reported_usage(consume()) if callable(consume) else None
            count_attempts = getattr(self.llm, "consume_budget_attempt_count", None)
            provider_attempts = count_attempts() if callable(count_attempts) else 0
            if usage is None:
                rendered_messages = [
                    SafeHelloAgentsLLM._flatten_message_content(message.get("content")) for message in messages
                ]
                prompt_tokens = _estimate_tokens("\n\n".join(rendered_messages))
                completion_tokens = _estimate_tokens("".join(completion))
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated_provider_calls": provider_attempts or 1,
                }
            elif provider_attempts > 1:
                usage["estimated_provider_calls"] = provider_attempts - 1
            usage["provider_calls"] = provider_attempts or 1
            pending = getattr(self._envelope_usage_local, "last_usage", None)
            totals = pending[1] if pending is not None and pending[0] is self else {}
            for key, count in usage.items():
                totals[key] = totals.get(key, 0) + count
            self._envelope_usage_local.last_usage = (self, totals)

            consume_downgrade = getattr(self.llm, "consume_cache_downgrade", None)
            if callable(consume_downgrade) and consume_downgrade():
                get_observer = getattr(self, "_cache_observer_getter", None)
                observer = get_observer() if callable(get_observer) else None
                if observer is not None:
                    observer.record_degraded("kv_cache_header_unsupported")

    def _invoke_provider_turn(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        completion: list[str] = []
        with self._measure_provider_turn(messages, completion):
            response = self.llm.invoke(messages, **kwargs)
            completion.append(response)
        return response

    def _stream_provider_turn(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        completion: list[str] = []
        with self._measure_provider_turn(messages, completion):
            for chunk in self.llm.stream_invoke(messages, **kwargs):
                if chunk:
                    completion.append(chunk)
                yield chunk

    def run_envelope(
        self,
        envelope: PromptEnvelope,
        *,
        config: Configuration | None = None,
        max_tool_iterations: int = 3,
        **kwargs: Any,
    ) -> str:
        self._envelope_usage_local.last_usage = None
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
            response = self._invoke_provider_turn(messages, **invoke_kwargs)
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
            final_response = self._invoke_provider_turn(messages, **invoke_kwargs)

        return final_response

    def stream_run_envelope(
        self,
        envelope: PromptEnvelope,
        *,
        config: Configuration | None = None,
        max_tool_iterations: int = 3,
        **kwargs: Any,
    ) -> Iterator[str]:
        self._envelope_usage_local.last_usage = None
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

            for chunk in self._stream_provider_turn(messages, **invoke_kwargs):
                if not chunk:
                    continue
                residual += chunk
                for seg in process_residual():
                    if not seg:
                        continue
                    segments_this_round.append(seg)
                    yield seg

            for seg in process_residual(final_pass=True):
                if not seg:
                    continue
                segments_this_round.append(seg)
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
            fallback_response = self._invoke_provider_turn(messages, **invoke_kwargs)
            yield fallback_response


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
        web_deployment_provider: Callable[[str, str], Any] | None = None,
        traceparent: str | None = None,
    ) -> None:
        """Initialise the coordinator with configuration and shared tools."""
        self.config = config or Configuration.from_env()
        self.request_id = request_id or "local-request"
        self.traceparent: str | None = (
            traceparent.strip().lower() if parse_traceparent(traceparent) else None
        )
        self.tenant_id = (
            str(
                tenant_id
                or getattr(self.config, "default_tenant_id", "default")
                or "default"
            ).strip()
            or "default"
        )
        self._request_cache_scope = (self.tenant_id, request_id or uuid4().hex)
        self._mcp_bridge = mcp_bridge
        self._owns_mcp_bridge = mcp_bridge is None
        self._active_state: SummaryState | None = None
        self._research_budget: RequestBudgetController | None = None
        self._web_budget: RequestBudgetController | None = None
        self._web_deployment_provider = web_deployment_provider
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
        self._workflow_checkpoint = None
        self._request_snapshot_lock = RLock()
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
        self._otel = OTelTraceAdapter(self.config, metrics=metrics_registry)
        self._model_router = CascadeModelRouter(self.config)

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
            scope_provider=self._web_search_tool_scope,
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
            extraction_context_provider=self._web_fetch_context,
            commit_lock=self._request_snapshot_lock,
            persist_web_evidence=self._commit_web_evidence,
        )
        for tool in (self.fetch_page_tool, self._sandbox_service):
            tool.budget_provider = lambda: self._research_budget
        self.search_web_tool.budget_provider = lambda: self._web_budget or self._research_budget
        self.evidence_lookup_tool = EvidenceLookupTool(
            evidence_store=self._evidence_store,
            retrieval_service=self._evidence_retrieval,
            config=self.config,
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
            retrieval_service=self._evidence_retrieval,
            evidence_store=self._evidence_store,
        )
        if self.config.resolved_sandbox_effective_enabled():
            self._refresh_scoped_tools_for_request()
        self._last_search_notices: list[str] = []
        self._request_trace: RequestTrace | None = None
        self._memory_scope: Any | None = None
        self._memory_policy: MemoryPolicy | None = None
        self._snapshot_reader: Any | None = None
        self._graph_store: Any | None = None
        self._graph_authority: RequestStateGraphAuthority | None = None
        self._graph_service: KnowledgeGraphService | None = None
        self._graph_queries_used: int = 0
        self._graph_deadline: float = 0.0
        self._init_partitioned_memory_and_graph()

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
        client = SafeHelloAgentsLLM(**llm_kwargs)
        client.budget_provider = lambda: getattr(self, "_research_budget", None)
        client.budget_pricing_catalog = self.config.llm_pricing_json
        return client

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
        client = SafeHelloAgentsLLM(**llm_kwargs)
        client.budget_provider = lambda: getattr(self, "_research_budget", None)
        client.budget_pricing_catalog = self.config.llm_pricing_json
        return client

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
        stage_agent = agent_cls(
            name=name,
            llm=llm or self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=tool_calling_enabled,
            tool_registry=self.tools_registry,
            tool_call_listener=self._tool_tracker.record,
        )
        if isinstance(stage_agent, CacheAwareToolAgent):
            stage_agent._request_cache_scope = self._request_cache_scope
            stage_agent._cache_observer_getter = lambda: getattr(self, "_request_trace", None)
        return stage_agent

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
            tenant_id=getattr(self, "tenant_id", "default"),
            topic=topic,
            search_api=search_api,
            provider=self.config.llm_provider,
            model=self.config.resolved_model(),
            pricing_catalog=self.config.llm_pricing_json,
        )
        self._request_trace.memory_shadow_mode = bool(self.config.memory_shadow_mode)
        if getattr(self, "traceparent", None):
            self._request_trace.traceparent = self.traceparent
        return self._request_trace

    def _request_status(self, state: SummaryState, report: str | None = None) -> str:
        """Resolve success state for the overall request."""
        if not report or not report.strip():
            return "failed"

        if not self._request_trace:
            return "success"

        pruned = set(((state.v4 or {}).get("replan") or {}).get("pruned_task_ids", []))
        has_incomplete_tasks = any(task.status in {"skipped", "failed"} and task.id not in pruned for task in state.todo_items)
        review_blocked = str(state.review_summary.get("overall_status") or "").strip() == "blocked"
        if (
            self._request_trace.fallback_count
            or self._request_trace.degraded_reasons
            or has_incomplete_tasks
            or review_blocked
            or (state.v4 or {}).get("degradation_reason")
            or ((state.v4 or {}).get("report") or {}).get("degradation_reason")
            or getattr(getattr(self, "_research_budget", None), "blocked", False)
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
            if bridge is not None and not (self.config.v4_research_budget_enabled and getattr(self, "_active_state", None) is None):
                try:
                    budget_kwargs = {}
                    if self._callable_accepts_kwarg(bridge.build_tool_adapters, "budget_provider"):
                        budget_kwargs["budget_provider"] = lambda: getattr(self, "_research_budget", None)
                    mcp_adapters = bridge.build_tool_adapters(
                        evidence_store=self._evidence_store,
                        tenant_id_getter=lambda: self.tenant_id,
                        request_id_getter=lambda: self.request_id,
                        active_scopes_getter=lambda: getattr(self.config, "active_mcp_tool_scopes", None),
                        stage=stage,
                        approved_registry_getter=lambda: getattr(self._active_state, "approved_tool_calls", None),
                        consume_approval=self._consume_mcp_approval if self._request_state_store is not None else None,
                        observer_getter=lambda: self._request_trace,
                        **budget_kwargs,
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
        if getattr(self, "evidence_lookup_tool", None) is not None:
            self.evidence_lookup_tool._config = target_config
        if getattr(self, "_evidence_retrieval", None) is not None:
            self._evidence_retrieval._config = target_config
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
            if getattr(orchestrator, "_decomposer", None) is not None:
                orchestrator._decomposer._decompose_with_fast_llm = target_config.nli_decompose_with_fast_llm
            if getattr(orchestrator, "_scorer", None) is not None:
                orchestrator._scorer._config = target_config
            if getattr(orchestrator, "_conflict_builder", None) is not None:
                orchestrator._conflict_builder._config = target_config

    def _refresh_context_checkpoints(self, state: SummaryState) -> None:
        """Advance frozen FactLedger and EvidenceIndexCard checkpoints between waves/stages."""

        with getattr(self, "_request_snapshot_lock", nullcontext()):
            changed = False
            evidence_store = getattr(self, "_evidence_store", None)
            if getattr(self.config, "context_compaction_enabled", True):
                valid_ids = {
                    item["source_id"] for item in evidence_store.lookup() if item.get("source_id")
                } if evidence_store is not None else set()
                existing = FactLedger.from_dict(state.fact_ledger)
                updated = FactLedgerCompactor().update_ledger(
                    existing, state.todo_items, valid_source_ids=valid_ids,
                    evidence_store=evidence_store,
                )
                if updated.version != existing.version:
                    state.fact_ledger = updated.to_dict()
                    changed = True

            if getattr(self.config, "evidence_pointer_disclosure_enabled", True) and evidence_store is not None:
                cards = build_evidence_index_cards(
                    evidence_store, runtime_index=getattr(self, "_evidence_runtime", None), max_cards=None,
                )
                committed_ids = {card["source_id"] for card in state.evidence_index_cards}
                new_cards = [card for card in cards if card["source_id"] not in committed_ids]
                if new_cards:
                    state.evidence_index_cards = [*state.evidence_index_cards, *new_cards]
                    changed = True

            checkpoint_changed = append_context_checkpoint(state, self.config)
            if changed or checkpoint_changed:
                state.frozen_prefix_version += 1

    def _prepare_upstream_contexts(self, state: SummaryState, tasks: list[TodoItem]) -> None:
        compact_enabled = getattr(self.config, "context_compaction_enabled", True)
        cards = state.evidence_index_cards if self.config.evidence_pointer_disclosure_enabled else []
        for task in tasks:
            task.upstream_context = build_upstream_context(
                task, state.todo_items, fact_ledger=state.fact_ledger,
                evidence_index_cards=cards, compact_enabled=compact_enabled,
            )

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

    def _web_search_tool_scope(self, task_id: int) -> tuple[str, str] | None:
        state = getattr(self, "_active_state", None)
        if state is not None and any(task.id == task_id for task in state.todo_items):
            return self.tenant_id, self.request_id
        return None

    def _commit_web_evidence(self, task_id: int) -> None:
        state = self._active_state
        if state is not None:
            self._persist_web_evidence(
                state, task_id, self._evidence_store.list_task_evidence(task_id, include_full_content=True),
            )

    def _configure_web_extraction(self, state: SummaryState) -> None:
        from state_manager import web_extraction_adapter

        frozen = (state.v4 or {}).get("web_extraction")
        store = getattr(self, "_evidence_store", None)
        if not isinstance(store, EvidenceStore):
            return
        store.web_extraction_enabled = isinstance(frozen, dict)
        tool = getattr(self, "fetch_page_tool", None)
        if not store.web_extraction_enabled or not isinstance(tool, FetchPageTool):
            return
        extractor = tool.build_web_extractor()
        adapter_name = web_extraction_adapter(frozen, self.config, self.tenant_id)
        provider = getattr(self, "_web_deployment_provider", None)
        if adapter_name != "off" and provider is not None:
            from services.web_dynamic_adapter import (
                DynamicTargetPolicy,
                PlaywrightAdapter,
                ReaderEndpoint,
                ReaderHttpAdapter,
            )

            try:
                deployment = provider(adapter_name, self.tenant_id)
                settings = {
                    "deployment": deployment,
                    "max_concurrency": self.config.v4_web_l2_max_concurrency,
                    "per_tenant": self.config.v4_web_l2_per_tenant,
                    "timeout_seconds": self.config.v4_web_l2_timeout_seconds,
                }
                rules = DynamicTargetPolicy.from_config(self.config.v4_web_l2_origin_rules)
                if adapter_name == "reader_http":
                    endpoint = ReaderEndpoint(
                        self.config.v4_web_reader_url, self.config.v4_web_reader_token,
                        getattr(deployment, "cost_microusd", 0),
                    )
                    extractor.adapter = ReaderHttpAdapter(extractor.gateway, rules, endpoint, **settings)
                else:
                    extractor.adapter = PlaywrightAdapter(extractor.gateway, rules, enabled=True, **settings)
            except Exception:
                logger.warning("Web extraction adapter unavailable (deployment)")
        tool.set_web_extractor(extractor)

    def _web_fetch_context(
        self, task_id: int, *, state: SummaryState | None = None, task: TodoItem | None = None,
        source_id: str | None = None, needs_coverage: bool = False,
    ) -> ExtractionContext | None:
        from state_manager import (
            record_web_extraction_result,
            supported_web_extraction,
            web_extraction_adapter,
        )

        state = state or getattr(self, "_active_state", None)
        if state is None:
            return None
        task = next((item for item in state.todo_items if item.id == task_id), None) if task is None else task
        if task is None or task.id != task_id or task.status in {"completed", "skipped", "failed"}:
            return None
        frozen = (state.v4 or {}).get("web_extraction")
        budget = getattr(self, "_web_budget", None)
        stage = budget.stage if budget is not None and not (state.v4 or {}).get("web_downstream_only") else "research"
        allow_network = (
            budget is not None and not budget.blocked and supported_web_extraction(frozen)
            and self.config.web_extraction_enabled_for(self.tenant_id)
        )
        remaining = budget.governor.remaining_timeout(stage=stage, work_class="optional") if allow_network else 0
        def static_attempt(demand: dict[str, int]):
            return budget.attempt(demand, required=False, stage=stage)

        def record_result(tier: str | None, reason: str | None) -> None:
            with self._request_snapshot_lock:
                current = (state.v4 or {}).get("web_extraction")
                if isinstance(current, dict) and (budget is None or budget.web_owner_is_current()):
                    record_web_extraction_result(current, tier=tier, reason=reason)

        selected_adapter = web_extraction_adapter(frozen, self.config, self.tenant_id)
        return ExtractionContext(
            self.tenant_id, self.request_id, task.id,
            FetchScope(
                monotonic() + (remaining if remaining > 0 else 1.0),
                check_owner=budget.web_owner_is_current if budget is not None and not budget.blocked else lambda: True,
                cancelled=self._is_cancelled,
                attempt=static_attempt,
            ),
            task_terms=f"{task.title} {task.query}", needs_coverage=needs_coverage,
            source_id=source_id, allow_network=allow_network and remaining > 0,
            network_allowed=lambda: self.config.web_extraction_enabled_for(self.tenant_id),
            can_dispatch=(lambda demand: budget.can_dispatch(demand, required=False, stage=stage)) if allow_network else None,
            dynamic_attempt=(
                lambda demand, urls: budget.attempt(demand, required=False, stage=stage, dynamic_urls=urls)
            ) if allow_network else None,
            dynamic_allowed=lambda: selected_adapter != "off" and selected_adapter == web_extraction_adapter(
                frozen, self.config, self.tenant_id,
            ),
            on_result=record_result,
        )

    def bind_workflow_checkpoint(self, checkpoint: Any) -> None:
        previous_store = self._request_state_store
        if previous_store is not checkpoint.store:
            close = getattr(previous_store, "close", None)
            if close is not None:
                close()
        self._workflow_checkpoint = checkpoint
        self._request_state_store = checkpoint.store
        self._init_partitioned_memory_and_graph()

    def _init_partitioned_memory_and_graph(self) -> None:
        """Bind strict tenant partitions, scoped note workspace, and KnowledgeGraphService."""
        config = getattr(self, "config", None)
        if config is None:
            return
        partition_enabled = bool(getattr(config, "v4_tenant_memory_partition_enabled", False))
        graph_enabled = bool(
            getattr(config, "knowledge_graph_enabled_for", lambda _: False)(self.tenant_id)
        )
        if not partition_enabled and not graph_enabled:
            return
        if not valid_internal_id(self.tenant_id) or not valid_internal_id(self.request_id):
            return

        store = getattr(self, "_request_state_store", None)
        if isinstance(store, PostgresRequestStateStore):
            self._snapshot_reader = build_strict_snapshot_reader(config, postgres_store=store)
        elif store is not None and getattr(store, "_directory", None) is not None:
            self._snapshot_reader = JsonStrictSnapshotReader(store._directory)
        else:
            self._snapshot_reader = build_strict_snapshot_reader(config)

        scope = authenticated_memory_scope(
            TenantContext(tenant_id=self.tenant_id, authenticated=True),
            self.request_id,
        )
        if scope is None:
            return
        policy = TenantSnapshotMemoryPolicy(
            tenant_id=self.tenant_id,
            snapshot_reader=self._snapshot_reader,
            active_request_id=self.request_id,
        )
        self._memory_scope = scope
        self._memory_policy = policy

        if partition_enabled:
            self._evidence_memory = EvidenceMemoryService(
                config, scope=scope, policy=policy, frozen_strict=True,
            )
            self._evidence_runtime = EvidenceRuntimeIndex(
                config,
                archive_loader=self._evidence_memory if self._evidence_memory.enabled else None,
            )
            if hasattr(self, "_evidence_store") and self._evidence_store is not None:
                listeners = getattr(self._evidence_store, "_listeners", None)
                if isinstance(listeners, list):
                    listeners[:] = [
                        item for item in listeners
                        if not isinstance(item, (EvidenceMemoryService, EvidenceRuntimeIndex))
                    ]
                    listeners.append(self._evidence_runtime)
                    if self._evidence_memory.enabled:
                        listeners.append(self._evidence_memory)
            self._note_memory = NoteMemoryService(
                config, scope=scope, policy=policy, frozen_strict=True,
            )
            self._strategy_memory = StrategyMemoryService(
                config,
                synthesizer=self._strategy_synthesizer,
                scope=scope,
                policy=policy,
                snapshot_reader=self._snapshot_reader,
                frozen_strict=True,
            )
            if getattr(config, "enable_notes", False):
                self.note_tool = ScopedNoteTool(
                    ScopedNoteWorkspace(config.notes_workspace, scope, policy),
                )
                if getattr(self, "tools_registry", None) is not None:
                    self.tools_registry.register_tool(self.note_tool)
                    self._core_tools = [
                        tool for tool in (
                            self.search_web_tool,
                            self.fetch_page_tool,
                            self.evidence_lookup_tool,
                            self.note_tool,
                        )
                        if tool is not None
                    ]

        if graph_enabled:
            if self._graph_store is None:
                self._graph_store = build_graph_store(config)
            self._graph_authority = RequestStateGraphAuthority(
                snapshot_reader=self._snapshot_reader,
                policy=policy,
                config=config,
                active_evidence_store=getattr(self, "_evidence_store", None),
                active_request_id=self.request_id,
            )
            self._graph_service = KnowledgeGraphService(
                self._graph_store,
                self._graph_authority,
                min_entailment=config.v4_graph_min_entailment,
                nli_entailment_threshold=max(config.nli_entailment_threshold, config.v4_graph_min_entailment),
                max_candidates=config.v4_graph_max_candidates,
                max_projection_tokens=config.v4_graph_max_projection_tokens,
                query_timeout_ms=config.v4_graph_query_timeout_ms,
                tenants=tuple(config.v4_knowledge_graph_tenants) or None,
                metrics=metrics_registry,
            )

    @staticmethod
    def _graph_seed_entities(seed_text: str) -> tuple[GraphEntity, ...]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+-]{1,40}|[\u4e00-\u9fff]{2,20}", str(seed_text or ""))
        seeds: list[GraphEntity] = []
        seen: set[str] = set()
        for raw in tokens:
            cleaned = sanitize_web_text(raw, max_chars=80)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            try:
                seeds.append(GraphEntity(entity_type="organization", name=cleaned))
            except ValueError:
                continue
            if len(seeds) >= 3:
                break
        return tuple(seeds)

    def _query_graph_context(
        self,
        state: SummaryState,
        *,
        purpose: str,
        seed_text: str,
        max_bytes: int,
    ) -> Any | None:
        graph_service = getattr(self, "_graph_service", None)
        memory_scope = getattr(self, "_memory_scope", None)
        if graph_service is None or memory_scope is None:
            return None
        used = int(getattr(self, "_graph_queries_used", 0) or 0)
        if used >= 8:
            return None
        # Reserve the final query slot and >=100ms budget exclusively for reporting.
        if purpose != "report" and used >= 7:
            return None
        now = monotonic()
        deadline = getattr(self, "_graph_deadline", 0.0) or (
            now + max(0.1, float(self.config.v4_graph_request_timeout_ms or 300) / 1000.0)
        )
        remaining = deadline - now
        if purpose != "report" and remaining <= 0.10:
            return None
        if remaining <= 0.01:
            return None
        seeds = self._graph_seed_entities(seed_text)
        if not seeds:
            return None
        self._graph_queries_used = used + 1
        try:
            projection = graph_service.project(
                memory_scope,
                GraphQuery(seed_entities=seeds, max_hops=2, limit=6),
                purpose=purpose,  # type: ignore[arg-type]
                deadline=min(deadline, now + float(self.config.v4_graph_query_timeout_ms or 100) / 1000.0),
                token_limit=max_bytes,
            )
            return projection if getattr(projection, "records", ()) else None
        except Exception:
            return None

    def _graph_projection_for_task(self, state: SummaryState, task: TodoItem) -> Any | None:
        return self._query_graph_context(
            state,
            purpose="task",
            seed_text=f"{task.title} {task.intent} {state.research_topic}",
            max_bytes=256,
        )

    def _seal_v4_knowledge_graph(
        self,
        state: SummaryState,
        *,
        request_status: str,
    ) -> tuple[TerminalGraphSnapshot, Any] | None:
        partition_enabled = bool(getattr(self.config, "v4_tenant_memory_partition_enabled", False))
        if partition_enabled:
            if state.v4 is None:
                state.v4 = {"schema_version": 1}
            state.v4["tenant_memory"] = {
                "schema_version": 1,
                "partition_enabled": True,
                "tenant_id": getattr(self, "tenant_id", "default"),
            }
        graph_service = getattr(self, "_graph_service", None)
        memory_scope = getattr(self, "_memory_scope", None)
        if graph_service is None or memory_scope is None:
            return None
        if request_status not in {"success", "partial_success", "completed"}:
            return None

        ledger = FactLedger.from_dict(state.fact_ledger)
        if not ledger.version and any(getattr(t, "status", "") == "completed" for t in state.todo_items):
            ledger = FactLedgerCompactor().update_ledger(
                ledger, state.todo_items, evidence_store=self._evidence_store,
            )
            state.fact_ledger = ledger.to_dict()

        bindings = build_publication_bindings(
            state,
            evidence_store=self._evidence_store,
            ledger=ledger,
            request_id=self.request_id,
            tenant_id=self.tenant_id,
        )
        candidates = build_verified_graph_candidates(
            state,
            scope=memory_scope,
            evidence_store=self._evidence_store,
            ledger=ledger,
            publication_bindings=bindings,
            min_entailment=self.config.v4_graph_min_entailment,
            nli_entailment_threshold=self.config.nli_entailment_threshold,
        )
        active_receipt_ids = tuple(
            receipt_id_for(
                memory_scope,
                candidate.ledger_version,
                candidate.claim_id,
                fact_fingerprint(candidate.ledger_version, candidate.statement, candidate.fact),
            )
            for candidate in candidates
        )
        source_epoch = max(1, int(getattr(state, "state_version", 0) or 0) + 1)
        ledger_version = max(1, int(ledger.version or 1))
        batch = seal_terminal_batch(
            memory_scope,
            candidates,
            source_epoch=source_epoch,
            ledger_version=ledger_version,
            report_version=1,
        )
        terminal_snapshot = TerminalGraphSnapshot(
            tenant_id=self.tenant_id,
            request_id=self.request_id,
            status="completed" if request_status == "success" else "partial_success",
            policy_version=memory_scope.policy_version,
            owner_epoch=memory_scope.owner_epoch,
            batch=batch,
            active_receipt_ids=active_receipt_ids,
        )
        if state.v4 is None:
            state.v4 = {"schema_version": 1}
        state.v4["knowledge_graph"] = {
            "schema_version": 1,
            "batch_id": batch.batch_id,
            "digest": batch.digest,
            "source_epoch": batch.source_epoch,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "ack_status": "pending",
            "terminal_snapshot": terminal_snapshot_to_payload(terminal_snapshot),
        }
        return terminal_snapshot, batch

    def _commit_v4_knowledge_graph_post_persist(
        self,
        sealed_pair: tuple[TerminalGraphSnapshot, Any] | None,
    ) -> None:
        graph_service = getattr(self, "_graph_service", None)
        memory_scope = getattr(self, "_memory_scope", None)
        if sealed_pair is None or graph_service is None or memory_scope is None:
            return
        terminal_snapshot, batch = sealed_pair
        authority = getattr(self, "_graph_authority", None)
        if authority is not None:
            authority._active_snapshot = terminal_snapshot
            authority._active_evidence_store = getattr(self, "_evidence_store", None)
        deadline = time.monotonic() + max(0.1, float(self.config.v4_graph_request_timeout_ms or 300) / 1000.0)
        result = graph_service.apply_terminal_batch(
            memory_scope, batch.batch_id, deadline=deadline,
        )
        if result.outcome in {"applied", "duplicate", "volatile_applied"} and result.watermark is not None:
            if getattr(self, "_request_state_store", None) is not None:
                record_v4_graph_ack(
                    self._request_state_store,
                    tenant_id=self.tenant_id,
                    request_id=self.request_id,
                    batch_id=batch.batch_id,
                    digest=batch.digest,
                    revision=result.watermark.graph_revision,
                    status=result.outcome,
                )
        else:
            metrics_registry.record_v4_graph_pending(
                "retry_deferred", "<1m", 1, tenant_id=self.tenant_id,
            )

    def close_workflow(self) -> None:
        if self._owns_mcp_bridge and self._mcp_bridge is not None:
            bridge, self._mcp_bridge = self._mcp_bridge, None
            bridge.close()

    def _export_observability_trace(self, snapshot: dict[str, Any] | None) -> None:
        adapter = getattr(self, "_observability", None)
        if adapter is not None and snapshot:
            adapter.export_request_trace(snapshot)
        otel = getattr(self, "_otel", None)
        if otel is not None and snapshot:
            otel.export_request_trace(snapshot, traceparent=getattr(self, "traceparent", None))

    def _select_stage_route(
        self,
        *,
        stage: str,
        topic: str = "",
        state: SummaryState | None = None,
        observer: RequestTrace | None = None,
        retry_index: int = 0,
    ) -> RouteDecision | None:
        router = getattr(self, "_model_router", None)
        if router is None:
            return None
        todo_items = getattr(state, "todo_items", ()) if state is not None else ()
        conflicts = getattr(state, "evidence_conflicts", ()) if state is not None else ()
        budget = getattr(self, "_research_budget", None)
        pressure = 0.0
        if budget is not None:
            try:
                snap = budget.governor.snapshot()
                llm_used = float(getattr(snap, "llm_calls_used", 0) or 0)
                llm_limit = float(budget.governor.budget.limits.get("llm_calls") or 1)
                if llm_limit > 0:
                    pressure = min(1.0, llm_used / llm_limit)
            except Exception:
                pressure = 0.0
        selection = getattr(self, "_skill_selection", None)
        domain = str(getattr(selection, "domain", "") or "general") if selection else "general"
        features = RouteFeatures(
            stage=stage,
            topic=topic or (getattr(state, "research_topic", "") or ""),
            task_count=len(todo_items),
            conflict_count=len(conflicts),
            budget_pressure=pressure,
            retry_index=retry_index,
            domain=domain,
        )
        decision = router.select_tier(features)
        router.record_decision(
            decision,
            metrics=metrics_registry,
            observer=observer,
            tenant_id=getattr(self, "tenant_id", "default"),
        )
        return decision

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
        budget = getattr(self, "_research_budget", None)
        if budget is not None:
            if budget.blocked or budget.replan["plan_locked"] or not budget.can_dispatch({"tasks": 1}):
                return 0
            raw_remaining = min(raw_remaining, budget.governor.snapshot().remaining["tasks"])
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
        self._select_stage_route(stage="review", state=state, observer=observer)
        review_graph = self._query_graph_context(
            state, purpose="task", seed_text=state.research_topic or "", max_bytes=256,
        )
        if review_graph is not None and self._callable_accepts_kwarg(self.reviewer.review_request, "graph_context"):
            review_kwargs["graph_context"] = review_graph
        review_summary = self.reviewer.review_request(state, **review_kwargs)

        if review_summary.get("overall_status") == "blocked":
            observer.record_degraded("review_blocked")
        elif review_summary.get("overall_status") == "warning":
            observer.record_degraded("review_warning")

        return review_summary

    @staticmethod
    def _graph_expectation(replan: dict[str, Any]) -> dict[str, Any]:
        return {
            "expected_dag_revision": replan["dag_revision"],
            "expected_wave_1_committed": replan["wave_1_committed"],
            "expected_plan_locked": replan["plan_locked"],
        }

    @staticmethod
    def _coverage_text(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    def _plan_coverage(self, state: SummaryState) -> tuple[dict[str, str], dict[int, list[str]]]:
        topic = state.research_topic or ""
        explicit = re.search(r"(?:compare|比较|对比).{0,200}?(?:by|across|维度[：:]?|包括)\s*(.+)", topic, re.I)
        dimensions = re.split(r"[,，、;；]|\band\b", explicit[1]) if explicit else []
        selection = getattr(self, "_skill_selection", None)
        if selection and not selection.shadow:
            gates = getattr(selection.pack, "quality_gates", {})
            configured = gates.get("required_coverage", [])
            if isinstance(configured, list):
                dimensions.extend(value for value in configured if isinstance(value, str))
        coverage = {}
        for dimension in dimensions[:24]:
            label = dimension.strip(" .。:：\t\n")[:120]
            if label:
                coverage.setdefault(self._coverage_text(label), label)
        by_task = {}
        for task in state.todo_items:
            wording = self._coverage_text(f"{task.title} {task.intent} {task.query}")
            by_task[task.id] = [key for key in coverage if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", wording)]
        if not coverage:
            return {"research_topic": topic}, {task.id: ["research_topic"] for task in state.todo_items}
        return coverage, by_task

    def _prepare_budget_plan(self, state: SummaryState, budget: RequestBudgetController, *, draft: bool = False, lock: bool = False) -> None:
        replan = budget.replan
        if replan["slots_allocated"]:
            return
        expected = self._graph_expectation(replan)
        required, coverage = self._plan_coverage(state)
        if not (lock or replan["plan_locked"]):
            cap = min(self.config.resolved_initial_plan_max_tasks(), budget.governor.budget.limits["tasks"])
            by_id = {task.id: task for task in state.todo_items}
            chosen: set[int] = set()
            covered: set[str] = set()

            def closure(task_id: int, path: frozenset[int] = frozenset()) -> set[int]:
                if task_id in path or task_id not in by_id:
                    raise ValueError("Invalid plan dependency")
                ids = {task_id}
                for parent in normalize_depends_on(by_id[task_id].depends_on):
                    ids.update(closure(parent, path | {task_id}))
                return ids

            try:
                if len(by_id) != len(state.todo_items) or any(type(task_id) is not int or task_id < 1 for task_id in by_id):
                    raise ValueError("Invalid task IDs")
                candidates = {task_id: closure(task_id) for task_id in by_id}
            except ValueError:
                fallback = self.planner.create_fallback_task(state)
                state.todo_items = [fallback]
                required, coverage = self._plan_coverage(state)
                candidates = {fallback.id: {fallback.id}}
            while candidates:
                ready = [(task_id, ids, set().union(*(set(coverage.get(i, [])) for i in ids)))
                         for task_id, ids in candidates.items() if len(ids | chosen) <= cap]
                if not ready:
                    break
                task_id, ids, keys = min(ready, key=lambda item: (-len(item[2] - covered), item[0]))
                covered.update(keys)
                chosen.update(ids)
                candidates.pop(task_id)
            if len(chosen) < len(state.todo_items):
                budget.degrade("limit_exceeded")
            state.todo_items = [task for task in state.todo_items if task.id in chosen]
        replan.update(
            required_coverage=required, coverage_by_task={str(task.id): coverage.get(task.id, []) for task in state.todo_items},
            next_task_id=max((task.id for task in state.todo_items), default=0) + 1,
        )
        if draft:
            return
        replan["plan_locked"] = lock or replan["plan_locked"]
        if state.todo_items:
            try:
                budget.charge_tasks(len(state.todo_items))
            except BudgetDispatchDenied:
                for task in state.todo_items:
                    if task.status in {"pending", "queued", "awaiting_approval", "in_progress"}:
                        task.status = "skipped"
        replan.update(slots_allocated=True, dag_revision=replan["dag_revision"] + 1)
        self._persist_request_state(state, phase="planned", v4_expected=expected)

    def _budget_event(self, budget: RequestBudgetController, boundary: str) -> dict[str, Any]:
        snapshot = asdict(budget.snapshot())
        budget.observation += 1
        pressure = snapshot["pressure"]
        if budget.last_emitted_pressure is not None and budget.last_emitted_pressure != pressure:
            metrics_registry.record_budget_event("pressure", reason=pressure)
        budget.last_emitted_pressure = pressure
        return {
            "type": "budget_updated", "schema_version": 1, "request_id": self.request_id,
            "event_key": f"{self.request_id}:budget:{budget.state.state_version + 1}:{budget.observation}:{boundary}",
            "stage": budget.stage, "boundary": boundary,
            "budget": {key: value for key, value in snapshot.items() if value is not None},
        }

    def _finish_budget_stage(self, state: SummaryState, budget: RequestBudgetController, phase: str) -> list[dict[str, Any]]:
        return self._persist_request_state(
            state, phase=phase, v4_events=None if budget.blocked else [self._budget_event(budget, phase)],
        )

    def _wave_material(self, state: SummaryState, task_ids: set[int] | None = None) -> dict[str, Any]:
        records = self._evidence_store.lookup(include_full_content=True)
        sources = [record for record in records if (
            (task_ids is None or record.get("task_id") in task_ids)
            and record.get("source_id")
            and self._evidence_store.is_eligible_for_facts(record["source_id"])
            and (record.get("raw_content") or record.get("full_content") or record.get("source_origin") in {"mcp", "sandbox"}
                 or self._evidence_store.web_extraction_enabled and record.get("multimodal_blocks"))
            and record.get("quality_label") != "low"
        )]
        ids = {source["source_id"] for source in sources}
        ledger = FactLedger.from_dict(state.fact_ledger)
        if not getattr(self.config, "context_compaction_enabled", True):
            ledger = FactLedgerCompactor().update_ledger(ledger, state.todo_items, valid_source_ids=ids)
        facts = [
            {**entry.to_dict(), "source_id": source_id}
            for entry in ledger.entries if entry.stance != "gap" for source_id in entry.source_ids if source_id in ids
        ]
        # Persist only scoring inputs, not a second copy of the evidence archive.
        source_fields = ("source_id", "task_id", "url", "domain", "title", "source_origin", "archive_doc_key")
        return {"facts": facts, "sources": [{key: record[key] for key in source_fields if key in record} for record in sources]}

    def _commit_wave_one(self, state: SummaryState, budget: RequestBudgetController, channel_map: dict, next_step: int) -> list[dict[str, Any]]:
        replan = budget.replan
        expected = self._graph_expectation(replan)
        coverage = {int(key): value for key, value in replan["coverage_by_task"].items()}
        baseline = replan.get("wave_1_baseline") or {"facts": [], "sources": []}
        covered_before = {key for source in baseline["sources"] for key in coverage.get(source.get("task_id"), [])}
        scores = []
        for task_id in replan.get("wave_1_task_ids", []):
            material = self._wave_material(state, {task_id})
            scores.append(score_wave1_branch(
                task_id=task_id, required_coverage=coverage.get(task_id, []), baseline_coverage=covered_before,
                wave_coverage=coverage.get(task_id, []) if material["facts"] else [],
                baseline_facts=baseline["facts"], baseline_sources=baseline["sources"],
                wave_facts=material["facts"], wave_sources=material["sources"],
            ))
        remaining_slots = self._remaining_task_budget(state)
        depth = budget.governor.budget.depth
        dispatched = set(replan.get("dispatched_task_ids", []))
        reason = "plan_locked" if replan["plan_locked"] else "replan_disabled"
        events = []
        patch = None
        if self.config.v4_replan_enabled and not replan["plan_locked"] and budget.can_dispatch({"search_calls": 1}):
            decision = ResearchTreeRePlanner().plan(
                state.todo_items, scores, depth=depth, available_task_slots=remaining_slots,
                next_task_id=replan["next_task_id"], current_revision=replan["dag_revision"], dispatched_ids=dispatched,
                pivot_queries={task.id: f"{task.query} independent comparable sources" for task in state.todo_items if task.depends_on},
                needs_cross_check={score.task_id: score.independent_source_clusters == 1 for score in scores},
                optional_task_ids={task.id for task in state.todo_items if not coverage.get(task.id)},
            )
            patch = decision.patch
            reason = decision.reason_code
        elif self.config.v4_replan_enabled and not replan["plan_locked"]:
            reason = "budget_denied"
        else:
            metrics_registry.record_replan_event(reason=reason)
        if patch:
            applied = apply_dag_patch(
                state.todo_items, patch, current_revision=replan["dag_revision"], depth=depth,
                next_task_id=replan["next_task_id"], available_task_slots=remaining_slots,
                dispatched_ids=dispatched, required_coverage=replan["required_coverage"], coverage_by_task=coverage,
            )
            reason = applied.reason_code
            if applied.applied:
                try:
                    budget.charge_tasks(len(applied.added_task_ids))
                except BudgetDispatchDenied:
                    applied = None
                    reason = "budget_denied"
            if applied and applied.applied:
                state.todo_items = applied.tasks
                replan.update(dag_revision=applied.revision, decision_id=patch.decision_id,
                              pruned_task_ids=applied.pruned_task_ids, decision=asdict(patch),
                              next_task_id=replan["next_task_id"] + len(applied.added_task_ids),
                              expansion_count=len(applied.added_task_ids))
                for task in state.todo_items:
                    if task.id in applied.pruned_task_ids:
                        task.status = "skipped"
                        task.notices.append("计划已裁剪：此任务与已覆盖范围重复。")
                    if task.id in applied.added_task_ids:
                        replan["coverage_by_task"][str(task.id)] = list(getattr(task, "coverage_keys", []))
                self._assign_stream_channels([task for task in state.todo_items if task.id in applied.added_task_ids], channel_map, start_step=next_step)
                events.append({
                    "type": "research_replanned", "schema_version": 1, "request_id": self.request_id,
                    "event_key": f"{self.request_id}:replan:{patch.decision_id}", "dag_revision": applied.revision,
                    "decision_id": patch.decision_id, "pruned_task_ids": applied.pruned_task_ids,
                    "actions": [{"action": action.action, "task_id": action.task_id, "parent_task_ids": action.parent_task_ids,
                                 "reason_code": action.reason_code} for action in patch.actions],
                    "scores": [asdict(score) for score in scores],
                })
        replan.update(wave_1_committed=True, reason_code=reason, scores=[asdict(score) for score in scores], score_version=1)
        self._update_pending_task_ids(state, replan)
        events.append(self._budget_event(budget, "wave_1"))
        events.append(self._budget_todo_event(state, replan, boundary="wave_1"))
        return self._persist_request_state(state, phase="task_execution", v4_expected=expected, v4_events=events)

    @staticmethod
    def _update_pending_task_ids(state: SummaryState, replan: dict) -> None:
        replan["pending_task_ids"] = [task.id for task in state.todo_items if task.status not in {"completed", "failed", "skipped"}]

    def _commit_stage_tasks(self, state: SummaryState, budget: RequestBudgetController, tasks: list[TodoItem], *, phase: str) -> None:
        replan = budget.replan
        expected = self._graph_expectation(replan)
        try:
            budget.charge_tasks(len(tasks))
        except BudgetDispatchDenied:
            ids = {task.id for task in tasks}
            state.todo_items = [task for task in state.todo_items if task.id not in ids]
            tasks.clear()
            return
        replan.update(dag_revision=replan["dag_revision"] + 1, next_task_id=max(task.id for task in tasks) + 1)
        self._update_pending_task_ids(state, replan)
        self._persist_request_state(state, phase=phase, v4_expected=expected)

    def _budget_todo_event(self, state: SummaryState, replan: dict, *, boundary: str) -> dict:
        return {
            "type": "todo_list", "schema_version": 1, "request_id": self.request_id,
            "event_key": f"{self.request_id}:todo:{replan['dag_revision']}:{boundary}",
            "dag_revision": replan["dag_revision"],
            "v4": {"schema_version": 1, "replan": public_v4_projection(state.v4)["replan"]},
            "tasks": [self._serialize_task(task) for task in state.todo_items], "step": 0,
        }

    def _run_budget_waves(self, state: SummaryState, budget: RequestBudgetController, channel_map: dict, next_step: int, emit_stream: bool) -> Iterator[dict[str, Any]]:
        replan = budget.replan
        budget.stage = "research"
        if not replan["wave_1_committed"] and "wave_1_task_ids" not in replan:
            if self.config.v4_replan_enabled and not replan["plan_locked"]:
                coverage = {int(key): value for key, value in replan["coverage_by_task"].items()}
                first = select_wave_one(state.todo_items, budget.governor.budget.depth, coverage_by_task=coverage)
            else:
                pending = [task for task in state.todo_items if task.status in {"pending", "queued"}]
                first = next(iter(partition_into_waves(pending)), [])
            replan.update(wave_1_task_ids=[task.id for task in first], wave_1_baseline=self._wave_material(state))
            self._persist_request_state(state, phase="task_execution", v4_expected=self._graph_expectation(replan))

        while not self._is_cancelled():
            pending = [task for task in state.todo_items if task.status not in {"completed", "failed", "skipped"}]
            first_wave = not replan["wave_1_committed"]
            merged = replan.get("wave_1_merged", False)
            waves = partition_into_waves(pending)
            tasks = ([task for task in pending if task.id in replan["wave_1_task_ids"]]
                     if first_wave else next(iter(waves), []))
            if not first_wave and not tasks:
                break
            wave = 1 if first_wave else replan.get("completed_waves", 1) + 1
            total = wave - 1 + max(1, len(waves))
            frame = {"wave": wave, "total_waves": total, "task_ids": [task.id for task in tasks]}
            if tasks and not (first_wave and merged):
                if budget.governor.remaining_timeout(stage="research") <= 0:
                    budget.degrade("deadline_exceeded")
                    for task in pending:
                        task.status = "skipped"
                else:
                    replan["dispatched_task_ids"] = sorted(set(replan.get("dispatched_task_ids", [])) | {task.id for task in tasks})
                    self._update_pending_task_ids(state, replan)
                    self._persist_request_state(state, phase="task_execution", v4_expected=self._graph_expectation(replan))
                    self._prepare_upstream_contexts(state, tasks)
                    if emit_stream:
                        yield {"type": "wave_started", **frame}
                        yield from self._execute_task_batch_stream(state, tasks, channel_map)
                    else:
                        self._execute_task_batch_sync(state, tasks)
                if self._is_cancelled():
                    break
                self._refresh_context_checkpoints(state)
            if first_wave:
                replan["wave_1_merged"] = True
            replan["completed_waves"] = wave
            self._update_pending_task_ids(state, replan)
            self._persist_request_state(state, phase="task_execution", v4_expected=self._graph_expectation(replan))
            if emit_stream and tasks and not merged:
                yield {"type": "wave_completed", **frame}
            if first_wave:
                events = self._commit_wave_one(state, budget, channel_map, next_step)
            elif budget.snapshot().pressure != budget.last_emitted_pressure:
                events = self._persist_request_state(state, phase="task_execution", v4_events=[self._budget_event(budget, "pressure")])
            else:
                events = []
            if emit_stream:
                yield from events

    def _budget_report(self, state: SummaryState, budget: RequestBudgetController) -> str:
        report = ReportingService(None, self.config, evidence_store=self._evidence_store)
        # A local report is restricted to claims verified against registered evidence.
        safe_state = deepcopy(state)
        valid_ids = {
            item["source_id"] for item in self._evidence_store.lookup()
            if item.get("source_id") and self._evidence_store.is_eligible_for_facts(item["source_id"])
        }
        for task in safe_state.todo_items:
            task.claims = []
            task.fact_claims = []
            task.summary_payload = None
            task.summary = None
            task.atomic_claims = [claim for claim in task.atomic_claims if claim.get("nli_label") == "ENTAILED"
                                  and claim.get("best_source_id") in valid_ids]
            task.claims = [{"text": claim["text"], "source_ids": [claim["best_source_id"]]} for claim in task.atomic_claims]
            task.evidence_items = []
        coverage = budget.replan.get("coverage_by_task", {}) if not budget.blocked else {}
        required = budget.replan.get("required_coverage", {}) if not budget.blocked else {}
        covered = {key for task in safe_state.todo_items if task.atomic_claims for key in coverage.get(str(task.id), [])}
        gaps = [label for key, label in required.items() if key not in covered]
        gaps.extend(task.title for task in safe_state.todo_items if task.status in {"skipped", "failed"} and not task.atomic_claims)
        coverage_text = "、".join(required[key] for key in required if key in covered) or "暂无已核验覆盖"
        rendered = report._render_structured_report({"background_overview": f"研究问题：{state.research_topic}\n\n已覆盖范围：{coverage_text}"}, state=safe_state)
        if not gaps and not covered:
            gaps = ["目前没有足够的可核验证据。"]
        return rendered + ("\n\n## 未完成的研究项\n" + "\n".join(f"- {gap}" for gap in dict.fromkeys(gaps)) if gaps else "")

    def _report_transport(self):
        """Capture provider identity and build a separate, retry-free client per call."""
        llm = getattr(self, "llm", None)
        if not isinstance(getattr(llm, "_client", None), OpenAI):
            return None, None, None
        api_key, base_url, model = llm.api_key, llm.base_url, llm.model
        price = llm._budget_pricing(model)
        trust_env = not llm._should_bypass_env_proxy(base_url)

        def send(request):
            if request.model != model:
                raise ValueError("Report model identity changed")
            with OpenAI(
                api_key=api_key, base_url=base_url, max_retries=0,
                http_client=httpx.Client(follow_redirects=False, trust_env=trust_env),
            ) as client:
                output_limit = "max_completion_tokens" if request.provider_mode == "openai" else "max_tokens"
                response = client.chat.completions.create(
                    model=model, messages=request.messages, timeout=request.timeout, temperature=0,
                    **{output_limit: request.max_tokens}, **request.options,
                )
            choice = response.choices[0] if len(response.choices) == 1 else None
            return ReportModelResponse(
                choice.message.content or "" if choice else "",
                choice.finish_reason if choice else "invalid",
                tuple(reported_usage_fields(response.usage).items()),
            )

        return send, lambda request: price if request.model == model else None, (
            lambda response: SafeHelloAgentsLLM._budget_reported_usage(dict(response.usage), price)
            if price is not None else None
        )

    def _run_versioned_report(self, state, budget, observer, step, emit_stream):
        report_state = state.v4["report"]
        cp = self._init_report_checkpoint(state, budget, step=step, emit_stream=emit_stream)
        final = report_state.get("final") or {}
        report = final.get("markdown")
        if final.get("published"):
            report = state.structured_report
        if isinstance(report, str) and sha256(report.encode()).hexdigest() == final.get("markdown_digest"):
            state.report_workbench = final.get("workbench") or state.report_workbench
            state.report_charts = list((state.report_workbench or {}).get("charts_spec") or [])
            return report, cp
        if final:
            raise SnapshotConflictError("Saved report failed integrity validation")

        if not report_state.get("input_digest"):
            self._refresh_context_checkpoints(state)
            self._persist_request_state(state, phase="report", status="in_progress")
        selection = getattr(self, "_skill_selection", None)
        fmt = getattr(selection, "format_pack", None) if selection is not None and not selection.shadow else None
        transport, pricing, usage = self._report_transport()
        report_graph = self._query_graph_context(
            state, purpose="report", seed_text=state.research_topic or "", max_bytes=512,
        )
        coord_kwargs: dict[str, Any] = {}
        if report_graph is not None and self._callable_accepts_kwarg(
            self.reporting.coordinate_report, "graph_context",
        ):
            coord_kwargs["graph_context"] = report_graph
        report = yield from self.reporting.coordinate_report(
            state, observer, self._skill_stage_context("report"), getattr(fmt, "outline_sections", None),
            cp, transport, pricing, usage, **coord_kwargs,
        )
        workbench = public_report_workbench(state.report_workbench)
        cp.commit({"final": {"report_version": 1, "markdown": report, "workbench": workbench,
                             "markdown_digest": sha256(report.encode()).hexdigest()}})
        state.report_workbench = workbench
        return report, cp

    def _finish_versioned_report(self, state, report, cp, report_completed_event):
        final = cp.data["final"]
        state.evidence_conflicts = state.report_workbench["evidence_conflicts"]
        if final.get("published"):
            return []
        descriptors = [
            cp.event("report_chunk", f"final-1-chunk-{index}", index=index, report_version=1)
            for index, _ in enumerate(report_chunks(report), 1)
        ]
        if report_completed_event:
            descriptors.append({
                **report_completed_event, "event_key": "report-final-1-stage-completed", "step": cp.step,
            })
        descriptors.extend(
            cp.event("report_chart", f"final-1-chart-{index}", chart_index=index, report_version=1)
            for index, _ in enumerate(state.report_charts)
        )
        note = cp.data.get("note") or {}
        if not note and self.note_tool:
            cp.commit({"note": {"status": "creating"}})
            try:
                note_event = self._persist_final_report(state, report)
            except Exception:
                logger.warning("Could not confirm research report note write", extra={"request_id": self.request_id})
                note_event = None
            note = {"status": "complete", "note_id": state.report_note_id, "note_path": state.report_note_path}
            if note_event:
                note["title"] = note_event["title"]
            cp.commit({"note": note})
        state.report_note_id = note.get("note_id")
        state.report_note_path = note.get("note_path")
        if note.get("title"):
            descriptors.append(cp.event("report_note", "final-1-note", title=note["title"]))
        # Shared budget and report statistics are projected after the terminal CAS.
        if cp.controller and not cp.controller.blocked and self._research_budget is None:
            state.v4["budget"] = cp.controller.payload()
        cp.data["final"] = {key: value for key, value in final.items() if key not in {"markdown", "workbench"}}
        cp.data["final"]["published"] = True
        return descriptors

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
        report_checkpoint = None
        versioned_report = False
        report_completed_event = None
        try:
            self._graph_queries_used = 0
            self._graph_deadline = monotonic() + max(
                0.1, float(getattr(self.config, "v4_graph_request_timeout_ms", 300) or 300) / 1000.0,
            )
            budget = self._init_request_budget(state, resumed=resumed)
            self._configure_web_extraction(state)
            versioned_report = isinstance((state.v4 or {}).get("report"), dict)
            if budget is not None and resume_phase != "completed":
                report = ""
            if budget is not None and not resumed:
                initial_events = self._persist_request_state(
                    state, phase="planning", v4_expected=self._graph_expectation(budget.replan) if not budget.blocked else None,
                    v4_events=[self._budget_event(budget, "initialized")] if not budget.blocked else None,
                )
            elif not resumed and (versioned_report or (state.v4 or {}).get("web_extraction")):
                self._persist_request_state(state, phase="planning")
            self._select_skill_pack(state.research_topic or topic, observer, state=state)
            if emit_stream:
                logger.debug("Starting streaming research: topic=%s", topic)
                yield {"type": "status", "message": "初始化研究流程"}
                if budget is not None and not resumed:
                    yield from initial_events
            if self._is_cancelled():
                observer.record_degraded("client_disconnected")
                observer.complete_request(status="cancelled")
                request_completed = True
                self._persist_request_state(state, phase="cancelled", status="cancelled", report_markdown=report)
                return
            if resumed and resume_phase:
                self._restore_task_counters(observer, state)
                if not (versioned_report and resume_phase in {"report", "completed"}):
                    self._refresh_context_checkpoints(state)
                    if budget is None or resume_phase != "awaiting_approval":
                        self._persist_request_state(state, phase=resume_phase, status="in_progress")
                if emit_stream:
                    yield {
                        "type": "status",
                        "message": f"已从持久化快照恢复请求：{self.request_id}",
                    }

            needs_plan = (not resumed or not state.todo_items) and not (versioned_report and resume_phase in {"report", "completed"})
            if budget is not None:
                needs_plan = needs_plan and not budget.blocked and not budget.replan["slots_allocated"] and resume_phase != "awaiting_approval"
            if needs_plan:
                planning_span = observer.start_stage(
                    "planning",
                    scope="request",
                    metadata={"topic": topic} if emit_stream else None,
                )
                if emit_stream:
                    yield planning_span.started_event()

                try:
                    if budget is not None and not budget.can_dispatch({"llm_calls": 1}):
                        budget.refuse("limit_exceeded")
                    self._select_stage_route(stage="planning", topic=topic, state=state, observer=observer)
                    historical_memory_context = self._planning_memory_context(state, observer)
                    plan_graph = self._query_graph_context(
                        state, purpose="plan", seed_text=state.research_topic or topic, max_bytes=256,
                    )
                    plan_kwargs: dict[str, Any] = {
                        "observer": observer,
                        "historical_memory_context": historical_memory_context,
                        "strategy_memory_context": self._planning_strategy_context(state, observer),
                        "skill_pack_context": self._skill_stage_context("planning"),
                    }
                    if plan_graph is not None and self._callable_accepts_kwarg(self.planner.plan_todo_list, "graph_context"):
                        plan_kwargs["graph_context"] = plan_graph
                    state.todo_items = self.planner.plan_todo_list(
                        state,
                        **plan_kwargs,
                    )
                    completed_event = planning_span.complete(
                        status="success",
                        metadata={"task_count": len(state.todo_items)},
                    )
                    if emit_stream:
                        yield completed_event
                except BudgetDispatchDenied:
                    completed_event = planning_span.complete(status="partial_success", metadata={"task_count": 0})
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

            if not state.todo_items and not (versioned_report and resume_phase in {"report", "completed"}) and (budget is None or (
                needs_plan and budget.can_dispatch({"tasks": 1, "search_calls": 1})
            )):
                logger.info("No TODO items generated; falling back to single task")
                state.todo_items = [self.planner.create_fallback_task(state)]
                fallback_event = observer.record_fallback("planning_returned_no_tasks")
                degraded_event = observer.record_degraded("fallback_task_used")
                if emit_stream:
                    yield fallback_event
                    yield degraded_event

            budget_notice = self._apply_task_budget(state, observer) if budget is None else None
            if budget is not None and not budget.blocked:
                self._prepare_budget_plan(state, budget, lock=resume_phase == "awaiting_approval")
                if not state.todo_items:
                    budget.degrade("limit_exceeded")
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
                todo_event = {
                    "type": "todo_list",
                    "tasks": [self._serialize_task(t) for t in state.todo_items],
                    "step": 0,
                }
                if budget is not None and not budget.blocked:
                    boundary = f"start:{state.state_version}" if resumed else "start"
                    todo_event = self._budget_todo_event(state, budget.replan, boundary=boundary)
                    if self._request_state_store is not None:
                        todo_event["state_version"] = state.state_version
                yield todo_event

            pending_tasks = [
                task for task in state.todo_items if task.status not in {"completed"}
            ]
            if budget is not None and not budget.blocked:
                yield from self._run_budget_waves(state, budget, channel_map, next_step, emit_stream)
                next_step = max((channel["step"] for channel in channel_map.values()), default=0) + 1
            elif budget is not None:
                for task in pending_tasks:
                    if task.status not in {"failed", "skipped"}:
                        task.status = "skipped"
            elif pending_tasks:
                completed_ids = {
                    int(task.id) for task in state.todo_items if task.status == "completed"
                }
                waves = partition_into_waves(pending_tasks, completed_ids=completed_ids)
                if len(waves) <= 1:
                    self._prepare_upstream_contexts(state, pending_tasks)
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
                        self._prepare_upstream_contexts(state, wave_tasks)
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
                if budget is not None:
                    budget.stage = "reflection"
                reflection_gap_signals = self._reflection_gap_signals(state)
                if self.config.request_reflection_enabled and reflection_gap_signals:
                    reflection_step = next_step
                    remaining_budget = self._remaining_task_budget(state, stage="reflection")
                    if remaining_budget <= 0 or (budget is not None and not budget.can_dispatch({"llm_calls": 1})):
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
                            if budget is not None and additional_tasks:
                                state.reflection_completed = True
                                self._commit_stage_tasks(state, budget, additional_tasks, phase="reflection")
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
                        except (MCPHitlApprovalRequired, SnapshotConflictError):
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
                stage_events = self._finish_budget_stage(state, budget, "reflection") if budget is not None else ()
                if budget is None:
                    self._persist_request_state(state, phase="reflection", status="in_progress")
                if emit_stream:
                    yield from stage_events

            if not state.review_completed:
                if budget is not None:
                    budget.stage = "review"
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
                if budget is None:
                    self._persist_request_state(state, phase="review", status="in_progress")
                else:
                    stage_events = self._finish_budget_stage(state, budget, "review")
                    if emit_stream:
                        yield from stage_events

            if not state.report_repair_completed:
                if budget is not None:
                    budget.stage = "repair"
                repair_candidates = self._select_report_repair_candidates(state)
                if self.config.report_repair_enabled and repair_candidates and (budget is None or (
                    self._remaining_task_budget(state, stage="repair") > 0 and budget.can_dispatch({"llm_calls": 1})
                )):
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
                        if budget is not None and repair_tasks:
                            self._commit_stage_tasks(state, budget, repair_tasks, phase="report_repair")
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

                                self._refresh_context_checkpoints(state)
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
                                self._refresh_context_checkpoints(state)
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
                        if isinstance(exc, SnapshotConflictError) or (budget is not None and isinstance(exc, MCPHitlApprovalRequired)):
                            raise
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
            if versioned_report and report and resume_phase == "completed":
                if emit_stream:
                    saved = self._load_resume_snapshot(self.request_id)
                    for descriptor in ((saved.get("v4") or {}).get("report") or {}).get("final", {}).get("frames", []):
                        yield materialize_report_event(saved, descriptor)
                observer.complete_request(status=self._request_status(state, report))
                return
            if not report or versioned_report:
                if budget is not None:
                    budget.stage = "report"
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
                    if versioned_report:
                        self._select_stage_route(stage="report", topic=topic, state=state, observer=observer)
                        report, report_checkpoint = yield from self._run_versioned_report(
                            state, budget, observer, final_step, emit_stream,
                        )
                    else:
                        self._select_stage_route(stage="report", topic=topic, state=state, observer=observer)
                        if budget is not None and not budget.can_dispatch({"llm_calls": 1}):
                            budget.refuse("limit_exceeded")
                        report_graph = self._query_graph_context(
                            state, purpose="report", seed_text=state.research_topic or topic, max_bytes=512,
                        )
                        if emit_stream and hasattr(self.reporting, "stream_generate_report"):
                            stream_kwargs: dict[str, Any] = {
                                "observer": observer,
                                "skill_pack_context": self._skill_stage_context("report"),
                            }
                            if report_graph is not None and self._callable_accepts_kwarg(
                                self.reporting.stream_generate_report, "graph_context",
                            ):
                                stream_kwargs["graph_context"] = report_graph
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
                                if self._is_cancelled():
                                    break
                                report_event.setdefault("step", final_step)
                                yield report_event
                            report = state.structured_report or ""
                        else:
                            report_kwargs: dict[str, Any] = {
                                "observer": observer,
                                "skill_pack_context": self._skill_stage_context("report"),
                            }
                            if report_graph is not None and self._callable_accepts_kwarg(
                                self.reporting.generate_report, "graph_context",
                            ):
                                report_kwargs["graph_context"] = report_graph
                            report = self.reporting.generate_report(
                                state,
                                **report_kwargs,
                            )
                        if budget is not None and (not report.strip() or report == "报告生成失败，请检查输入。"):
                            budget.refuse("dispatch_unbounded")
                    completed_event = report_span.complete(
                        status="success",
                        metadata={"report_length": len(report or "")},
                    )
                    if versioned_report:
                        report_completed_event = completed_event
                    elif emit_stream:
                        completed_event["step"] = final_step
                        yield completed_event
                except (ReportCancelled, SnapshotConflictError):
                    raise
                except Exception as exc:
                    fallback = not versioned_report and budget is not None and (
                        isinstance(exc, (BudgetDispatchDenied, TimeoutError, httpx.TimeoutException))
                        or type(exc).__name__ == "APITimeoutError"
                    )
                    if fallback:
                        budget.degrade("report_fallback")
                        report = self._budget_report(state, budget)
                    completed_event = report_span.complete(status="partial_success" if fallback else "failed", error=exc)
                    if emit_stream:
                        completed_event["step"] = final_step
                        yield completed_event
                        yield observer.metrics_event()
                    if not fallback:
                        observer.complete_request(status="failed", error=exc)
                        request_completed = True
                        raise
            elif emit_stream:
                yield {
                    "type": "status",
                    "message": "检测到已持久化的报告内容，直接恢复最终结果。",
                    "step": final_step,
                }

            if self._is_cancelled():
                observer.complete_request(status="cancelled")
                request_completed = True
                self._persist_request_state(state, phase="cancelled", status="cancelled")
                return

            if emit_stream:
                for event in self._drain_tool_events(state, step=final_step):
                    yield event
            else:
                self._drain_tool_events(state)

            if report_checkpoint is not None:
                report_frames = self._finish_versioned_report(state, report, report_checkpoint, report_completed_event)
                workbench = state.report_workbench
                if self._is_cancelled():
                    raise ReportCancelled()
            else:
                state.structured_report = report
                state.running_summary = report
                workbench = build_report_workbench_payload(
                    state, report, evidence_store=self._evidence_store, config=self.config,
                )
                state.report_charts = workbench["charts_spec"]
                state.evidence_conflicts = workbench["evidence_conflicts"]
                if (
                    workbench["citation_index"]
                    or workbench["evidence_conflicts"]
                    or workbench["claim_verification_summary"]
                    or workbench["verifier_badges"]
                    or workbench["skill_governance"]["pack_id"] != "_neutral"
                ):
                    state.report_workbench = workbench

                note_event = self._persist_final_report(state, report)
                if emit_stream and note_event:
                    yield note_event

            request_status = self._request_status(state, report)
            if emit_stream and request_status == "partial_success":
                degraded = observer.record_degraded("partial_result_completed")
                if report_checkpoint is not None:
                    report_frames.append({**degraded, "event_key": "report-final-1-degraded"})
                else:
                    yield degraded

            observer.attach_result(
                report_markdown=report,
                todo_items=[self._serialize_task(task) for task in state.todo_items],
                report_note_id=state.report_note_id,
                report_note_path=state.report_note_path,
            )
            snapshot = observer.complete_request(status=request_status)
            request_completed = True
            self._export_observability_trace(snapshot)
            terminal_kwargs = {}
            if budget is not None and not budget.blocked:
                terminal_kwargs["v4_events"] = [self._budget_event(budget, "completed")]
            sealed_graph = self._seal_v4_knowledge_graph(state, request_status=request_status)
            if report_checkpoint is not None:
                cp = report_checkpoint
                report_frames.extend(terminal_kwargs.get("v4_events", ()))
                report_frames.extend((
                    cp.event("metrics_snapshot", "final-1-metrics"),
                    cp.event("final_report", "final-1", report_version=1),
                    cp.event("done", "final-1-done"),
                ))
                state_version = state.state_version + 1
                report_frames = [{**event, "schema_version": 1, "request_id": self.request_id,
                                  "state_version": state_version} for event in report_frames]
                cp.data["final"]["frames"] = [event for event in report_frames if event["event_key"].startswith("report-")]
                cp.data["revision_number"] += 1
                terminal_kwargs = {
                    "v4_expected": {"expected_report_revision": cp.data["revision_number"] - 1},
                    "v4_events": report_frames if emit_stream else [],
                }
            terminal_events = self._persist_request_state(
                state,
                phase="completed",
                status=request_status,
                report_markdown=report,
                **terminal_kwargs,
            )
            self._commit_v4_knowledge_graph_post_persist(sealed_graph)
            if report_checkpoint is not None:
                state.structured_report = report
                state.running_summary = report
            if budget is None:
                self._refresh_strategy_memory_for_request(self.request_id, observer=observer)

            if emit_stream and report_checkpoint is not None:
                saved = self._load_resume_snapshot(self.request_id) if self._request_state_store is not None else {
                    "report_markdown": report, "workbench": workbench, "report_note_id": state.report_note_id,
                    "report_note_path": state.report_note_path, "v4": state.v4, "request_metrics": observer.snapshot(),
                    "claim_verification_summary": state.claim_verification_summary, "evidence_conflicts": state.evidence_conflicts,
                }
                for event in terminal_events:
                    yield materialize_report_event(saved, event)
            elif emit_stream:
                yield from terminal_events or ()
                yield observer.metrics_event()
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
                    **({"v4": projection} if (projection := public_v4_projection(state.v4)) is not None else {}),
                    **({
                        "workbench": state.report_workbench,
                        "citation_drawer_map": workbench["citation_index"],
                        "verifier_badges": workbench["verifier_badges"],
                    } if state.report_workbench else {}),
                }
                yield {"type": "done"}
        except ReportCancelled:
            observer.complete_request(status="cancelled")
            self._persist_request_state(state, phase="cancelled", status="cancelled")
            return
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
        except SnapshotConflictError:
            raise
        except Exception as exc:
            if not request_completed:
                failed_snapshot = observer.complete_request(status="failed", error=exc)
                self._export_observability_trace(failed_snapshot)
            self._persist_request_state(
                state,
                phase="failed",
                status="failed",
                report_markdown=report if not versioned_report else None,
            )
            if getattr(self, "_research_budget", None) is None:
                self._refresh_strategy_memory_for_request(self.request_id, observer=observer)
            raise
        finally:
            self._research_budget = None
            self._web_budget = None
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
            report_workbench=dict(getattr(state, "report_workbench", None) or {}),
            evidence_conflicts=list(getattr(state, "evidence_conflicts", None) or []),
            claim_verification_summary=dict(
                getattr(state, "claim_verification_summary", None) or {}
            ),
            todo_items=state.todo_items,
            v4=public_v4_projection(state.v4),
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
        if target.status == "completed":
            return target
        self._refresh_context_checkpoints(state)
        self._prepare_upstream_contexts(state, [target])
        for _ in self._execute_task(state, target, emit_stream=False):
            pass
        self._refresh_context_checkpoints(state)
        self._persist_request_state(state, phase="task_execution", status="in_progress")
        return target

    def plan_only(self, topic: str) -> dict[str, Any]:
        """Generate an editable research plan and persist it with awaiting_approval status."""

        observer = self._start_request_trace(topic)
        state = SummaryState(research_topic=topic)
        self._active_state = state
        try:
            budget = self._init_request_budget(state, resumed=False)
            self._configure_web_extraction(state)
            if budget is not None and not budget.blocked:
                self._persist_request_state(
                    state, phase="planning", v4_expected=self._graph_expectation(budget.replan),
                    v4_events=[self._budget_event(budget, "initialized")],
                )
            elif (state.v4 or {}).get("web_extraction") or (state.v4 or {}).get("report"):
                self._persist_request_state(state, phase="planning")
            self._select_skill_pack(topic, observer, state=state)
            planning_span = observer.start_stage("planning", scope="request", metadata={"topic": topic})
            try:
                if budget is not None and not budget.can_dispatch({"llm_calls": 1}):
                    budget.refuse("limit_exceeded")
                historical_memory_context = self._planning_memory_context(state, observer)
                state.todo_items = self.planner.plan_todo_list(
                    state,
                    observer=observer,
                    historical_memory_context=historical_memory_context,
                    strategy_memory_context=self._planning_strategy_context(state, observer),
                    skill_pack_context=self._skill_stage_context("planning"),
                )
                planning_span.complete(status="success", metadata={"task_count": len(state.todo_items)})
            except BudgetDispatchDenied:
                planning_span.complete(status="partial_success", metadata={"task_count": 0})
            except Exception as exc:
                planning_span.complete(status="failed", error=exc)
                observer.complete_request(status="failed", error=exc)
                raise

            if not state.todo_items and (budget is None or budget.can_dispatch({"tasks": 1, "search_calls": 1})):
                state.todo_items = [self.planner.create_fallback_task(state)]

            persist_kwargs = {}
            if budget is None:
                self._apply_task_budget(state, observer)
            elif not budget.blocked:
                persist_kwargs["v4_expected"] = self._graph_expectation(budget.replan)
                self._prepare_budget_plan(state, budget, draft=True)
                persist_kwargs["v4_events"] = [self._budget_event(budget, "awaiting_approval")]
            observer.set_task_totals(total_tasks=len(state.todo_items))
            observer.complete_request(status="awaiting_approval")
            self._persist_request_state(state, phase="awaiting_approval", status="awaiting_approval", **persist_kwargs)
            result = {
                "request_id": self.request_id,
                "topic": topic,
                "status": "awaiting_approval",
                "phase": "awaiting_approval",
                "skill_pack": dict(state.skill_pack_metadata or {}),
                "todo_items": [self._serialize_task(task) for task in state.todo_items],
                **({"v4": projection} if (projection := public_v4_projection(state.v4)) is not None else {}),
                **({"plan_revision": 0} if state.v4 and getattr(self, "_workflow_checkpoint", None) is None else {}),
                "estimated_duration_seconds": HitlService.estimate_duration_seconds(
                    len(state.todo_items), self.config
                ),
            }
            return public_web_evidence_projection(result) if (state.v4 or {}).get("web_extraction") else result
        finally:
            self._research_budget = None
            self._web_budget = None
            self._active_state = None
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
        if state.claim_verification_summary:
            observer.restore_review_result(state.review_summary, state.claim_verification_summary)

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

        self._refresh_context_checkpoints(state)
        for _ in self._execute_task(state, target_task, emit_stream=False):
            pass
        self._refresh_context_checkpoints(state)

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
