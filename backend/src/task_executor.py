"""Task execution and task-level ReAct helpers for the deep research workflow."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from queue import Empty, Queue
from threading import Semaphore, Thread
from time import sleep
from typing import Any, Callable, Iterator

from metrics import RequestTrace, extract_llm_usage, metrics_registry
from models import SummaryState, TodoItem
from prompts import web_evidence_prompt_rules
from services.context_builder import ContextPrefixBuilder, render_web_evidence_data
from services.evidence import (
    EvidenceStore,
    build_task_context,
    build_task_context_from_hits,
    format_evidence_sources,
    public_web_evidence_projection,
    public_web_url,
)
from services.evidence_index import EvidenceRetrievalHit
from services.fact_ledger import render_context_checkpoint, render_context_delta
from services.mcp_bridge import MCPHitlApprovalRequired, _scope_allows_tool
from services.memory_sanitizer import sanitize_web_text
from services.policy_controller import (
    ModelPolicyController,
    PolicyDecision,
    PolicyStateFeatures,
    RuleBasedPolicyController,
    ShadowCanaryPolicyController,
    build_policy_controller,
    is_budget_legal_action,
)
from services.reviewer import ReviewService
from services.search import classify_search_query_bucket, dispatch_search
from services.summarizer import TaskSummaryResult
from utils import strip_thinking_tokens

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
_QUERY_NOISE_PATTERNS = (
    re.compile(r"```.*?```", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[TOOL_CALL:[^\]]+\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bnote_[A-Za-z0-9_]+\b", re.IGNORECASE),
    re.compile(r"\b(?:search_web|update_note|note_tool)\s*\([^)]*\)", re.IGNORECASE),
)
_QUERY_NOISE_PHRASES = (
    "search_web",
    "update note",
    "按顺序执行",
    "按任务顺序执行",
    "并更新笔记状态",
    "更新笔记状态",
    "工具调用",
    "工作流",
    "流程编排",
    "进度同步",
)


class ToolExecutionTimeoutError(TimeoutError):
    """Raised when a guarded external tool invocation exceeds its timeout budget."""


@dataclass
class TaskReactObservation:
    """Structured observation used by the bounded task-level evidence loop."""

    gap_signals: list[str]
    source_count: int
    source_diversity: int
    freshness_ok: bool
    evidence_sufficiency: bool
    continue_reason: str = ""
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_signals": list(self.gap_signals),
            "source_count": self.source_count,
            "source_diversity": self.source_diversity,
            "freshness_ok": self.freshness_ok,
            "evidence_sufficiency": self.evidence_sufficiency,
            "continue_reason": self.continue_reason,
            "stop_reason": self.stop_reason,
        }


@dataclass
class TaskReactDecision:
    """Structured next-step decision for task-level ReAct."""

    action: str
    query: str = ""
    source_id: str | None = None
    url: str = ""
    reason: str = ""


class TaskExecutorMixin:
    def _web_search_options(self) -> tuple[Any, dict[str, Any]]:
        store = getattr(self, "_evidence_store", None)
        web_enabled = bool(getattr(store, "web_extraction_enabled", False))
        tenant = getattr(self, "tenant_id", "default")
        if web_enabled:
            return getattr(self, "_web_budget", None), {"web_extraction_enabled": True}
        opts = {"web_extraction_enabled": False} if self.config.web_extraction_enabled_for(tenant) else {}
        return getattr(self, "_research_budget", None), opts
    """Run planned task batches and task-level evidence repair loops."""

    def _execute_task_batch_sync(
        self,
        state: SummaryState,
        tasks: list[TodoItem],
    ) -> None:
        """Execute a task batch for the synchronous request path."""

        self._warm_topic_search_cache(state, observer=self._request_trace)
        for task in tasks:
            for _ in self._execute_task(state, task, emit_stream=False):
                pass
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
        budget, web_options = self._web_search_options()
        cache_context = {
            "research_topic": state.research_topic,
            "task_title": task.title,
            "task_intent": task.intent,
            "request_id": str(getattr(self, "request_id", "") or ""),
            "task_id": str(task.id),
        }
        tenant = str(getattr(self, "tenant_id", "") or "").strip()
        if tenant not in {"", "default"} or web_options.get("web_extraction_enabled"):
            cache_context["tenant_id"] = tenant or "default"

        for retry_index in range(retry_attempts + 1):
            try:
                if budget is not None:
                    return dispatch_search(
                        candidate, self.config, state.research_loop_count,
                        observer=observer, cache_context=cache_context, budget=budget, **web_options,
                    )
                return self._run_with_timeout(
                    lambda: dispatch_search(
                        candidate,
                        self.config,
                        state.research_loop_count,
                        observer=observer,
                        cache_context=cache_context,
                        **web_options,
                    ),
                    timeout_seconds=timeout_seconds,
                    operation_name=operation_name,
                )
            except Exception as exc:
                if getattr(exc, "budget_denied", False):
                    raise
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
                    delay = retry_backoff_seconds
                    if budget is not None:
                        delay = budget.governor.remaining_timeout(delay, stage=budget.stage, work_class="required")
                    sleep(delay)

    def _normalize_query_candidate(self, value: str) -> str:
        """Normalize a candidate search query and strip request-style boilerplate."""

        cleaned = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (value or "").strip())
        for pattern in _QUERY_NOISE_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)
        for phrase in _QUERY_NOISE_PHRASES:
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"(?:^|[\s,，;；])(?:query|search query|检索方向|搜索方向|检索关键词|搜索关键词)\s*[:：]\s*",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.replace("`", " ")

        while cleaned:
            previous = cleaned
            for pattern in _LEADING_SEARCH_QUERY_PATTERNS:
                cleaned = pattern.sub("", cleaned, count=1)
            cleaned = cleaned.lstrip("：:,，。！？、；;[]【】()（）- ")
            if cleaned == previous:
                break

        return " ".join(cleaned.strip().split())

    def _topic_canonical_query(self, state: SummaryState) -> str:
        """Return the stable topic-level query used to warm and probe cache reuse."""

        return self._normalize_query_candidate(state.research_topic or "")

    def _warm_topic_search_cache(
        self,
        state: SummaryState,
        *,
        observer: RequestTrace | None,
    ) -> None:
        """Warm a stable topic-level cache entry before task-specific searches."""

        if state.topic_cache_warmup_completed:
            return

        state.topic_cache_warmup_completed = True
        if not self.config.search_cache_enabled or not self.config.resolved_approximate_cache_enabled():
            return
        if observer is None:
            return

        topic_query = self._topic_canonical_query(state)
        if not topic_query:
            return

        search_span = observer.start_stage(
            "search",
            scope="request",
            metadata={
                "query": topic_query,
                "purpose": "topic_cache_warmup",
            },
        )
        try:
            budget, web_options = self._web_search_options()
            if budget is not None:
                result = dispatch_search(
                    topic_query, self.config, state.research_loop_count, observer=observer,
                    cache_context={"research_topic": state.research_topic, "tenant_id": self.tenant_id},
                    budget=budget, required=False,
                    **web_options,
                )
            else:
                result = self._run_with_timeout(
                    lambda: dispatch_search(
                        topic_query,
                        self.config,
                        state.research_loop_count,
                        observer=observer,
                        cache_context={
                            "research_topic": state.research_topic,
                            **({"tenant_id": self.tenant_id} if web_options.get("web_extraction_enabled") else {}),
                        },
                        **web_options,
                    ),
                    timeout_seconds=self.config.search_tool_timeout_seconds,
                    operation_name=f"主题缓存预热[{topic_query}]",
                )
            (
                search_result,
                notices,
                answer_text,
                backend,
                cache_hit,
                cache_strategy,
            ) = result
        except Exception as exc:
            if getattr(exc, "budget_denied", False):
                search_span.complete(status="skipped")
                return
            observer.record_degraded("topic_cache_warmup_failed")
            search_span.complete(
                status="failed",
                error=exc,
                metadata={
                    "query": topic_query,
                    "purpose": "topic_cache_warmup",
                },
            )
            logger.warning("Topic cache warmup failed topic=%s error=%s", topic_query, exc)
            return

        cache_details = dict(observer.snapshot().get("last_search_cache_details") or {})
        search_span.complete(
            status="success",
            metadata={
                "query": topic_query,
                "purpose": "topic_cache_warmup",
                "backend": backend,
                "result_count": len((search_result or {}).get("results", [])),
                "cache_hit": cache_hit,
                "cache_strategy": cache_strategy,
                "notice_count": len(notices),
                "answer_present": bool(answer_text),
                **cache_details,
            },
        )
        logger.info(
            "Topic cache warmup completed topic=%s cache_hit=%s strategy=%s result_count=%s",
            topic_query,
            cache_hit,
            cache_strategy,
            len((search_result or {}).get("results", [])),
        )

    def _concise_query_intent(self, value: str) -> str:
        cleaned = self._normalize_query_candidate(value)
        if not cleaned:
            return ""
        return re.split(r"[。！？!?；;]+", cleaned, maxsplit=1)[0].strip()

    def _strip_title_already_in_topic(self, topic: str, title: str) -> str:
        normalized_topic = re.sub(r"\s+", "", str(topic or "")).casefold()
        normalized_title = re.sub(r"\s+", "", str(title or "")).casefold()
        if normalized_topic and normalized_title and normalized_title in normalized_topic:
            return ""
        return title

    def _canonical_task_query(self, state: SummaryState, task: TodoItem) -> str:
        topic = self._topic_canonical_query(state)
        raw_title = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip())
        title = self._strip_title_already_in_topic(topic, self._normalize_query_candidate(raw_title))
        intent = self._concise_query_intent(task.intent or "")
        if topic and title:
            return self._normalize_query_candidate(f"{topic} {title}")
        if topic and intent:
            return self._normalize_query_candidate(f"{topic} {intent}")
        return topic or title or intent or self._normalize_query_candidate(task.query or "")

    def _expanded_task_query_with_intent(self, state: SummaryState, task: TodoItem) -> str:
        topic = self._topic_canonical_query(state)
        raw_title = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip())
        title = self._strip_title_already_in_topic(topic, self._normalize_query_candidate(raw_title))
        intent = self._concise_query_intent(task.intent or "")
        return self._normalize_query_candidate(" ".join(part for part in [topic, title, intent] if part))

    def _task_retrieval_query(self, state: SummaryState, task: TodoItem) -> str:
        return self._normalize_query_candidate(
            " ".join(
                part
                for part in [
                    state.research_topic or "",
                    re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip()),
                    task.intent or "",
                    task.query or "",
                ]
                if str(part or "").strip()
            )
        )

    def _grounding_hits_for_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        observer: RequestTrace | None,
    ) -> list[EvidenceRetrievalHit]:
        if not self.config.evidence_runtime_enabled:
            return []
        result = self._ensure_evidence_retrieval().query(
            self._task_retrieval_query(state, task),
            task_id=task.id,
            scope="current",
            mode="grounding",
            top_k=self.config.evidence_runtime_top_k,
            request_id=self.request_id,
            observer=observer,
        )
        store = getattr(self, "_evidence_store", None)
        if getattr(store, "web_extraction_enabled", False):
            return [hit for hit in result.hits if store.is_eligible_for_facts(hit.source_id)]
        return list(result.hits)

    def _repair_hits_for_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        observer: RequestTrace | None,
        query_text: str | None = None,
        top_k: int | None = None,
    ) -> list[EvidenceRetrievalHit]:
        result = self._ensure_evidence_retrieval().query(
            query_text or self._task_retrieval_query(state, task),
            task_id=task.id,
            scope="hybrid",
            mode="repair",
            top_k=top_k or self.config.evidence_memory_top_k,
            request_id=self.request_id,
            observer=observer,
        )
        if getattr(getattr(self, "_evidence_store", None), "web_extraction_enabled", False):
            return [
                replace(hit, title=sanitize_web_text(hit.title, max_chars=500), text="", snippet="")
                if hit.origin == "history" else hit
                for hit in result.hits if hit.origin != "history" or public_web_url(hit.url)
            ]
        return list(result.hits)

    def _query_signal_tokens(self, value: str) -> set[str]:
        cleaned = self._normalize_query_candidate(value).casefold()
        ascii_tokens = set(re.findall(r"[a-z0-9]+", cleaned))
        cjk_tokens = {char for char in cleaned if "\u4e00" <= char <= "\u9fff"}
        return ascii_tokens | cjk_tokens

    def _query_adds_new_signal(self, base_query: str, candidate_query: str) -> bool:
        base_tokens = self._query_signal_tokens(base_query)
        candidate_tokens = self._query_signal_tokens(candidate_query)
        return bool(candidate_tokens - base_tokens)

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

        return self._expanded_task_query_with_intent(state, task)

    def _task_search_queries(
        self,
        state: SummaryState,
        task: TodoItem,
    ) -> list[tuple[str, str]]:
        """Return progressively broader queries for a task-level search retry."""

        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        canonical_query = self._canonical_task_query(state, task)
        rewritten_query = self._expanded_task_query_with_intent(state, task)
        topic_only_query = self._topic_canonical_query(state)
        fresh_bucket = classify_search_query_bucket(
            canonical_query,
            {
                "research_topic": state.research_topic,
                "task_title": task.title,
                "task_intent": task.intent,
            },
        )

        def add(value: str, strategy: str) -> None:
            normalized = self._normalize_query_candidate(value)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append((normalized, strategy))

        add(canonical_query, "original")
        if (
            self.config.task_query_rewrite_enabled
            and rewritten_query
            and self._query_adds_new_signal(canonical_query, rewritten_query)
        ):
            add(rewritten_query, "rewrite")
        add(topic_only_query, "expand")
        if fresh_bucket == "fresh":
            add(f"{canonical_query} 最新进展", "expand")

        fallback = rewritten_query or canonical_query or topic_only_query
        return candidates or [(fallback, "rewrite")]

    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        index = 0
        while index < len(text):
            if text[index] != "{":
                index += 1
                continue
            try:
                payload, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(payload, dict):
                return payload
            index += max(end, 1)
        return None

    @staticmethod
    def _merge_notices(existing: list[str], incoming: list[str]) -> list[str]:
        merged = list(existing)
        for notice in incoming:
            cleaned = str(notice or "").strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged

    @staticmethod
    def _top_evidence_item(evidence_items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not evidence_items:
            return None
        ranked = sorted(
            evidence_items,
            key=lambda item: (
                0 if item.get("full_content") else 1,
                0 if str(item.get("quality_label") or "") == "high" else 1,
                str(item.get("freshness_label") or "") == "stale",
                -int(item.get("provider_count") or 1),
            ),
        )
        return ranked[0]

    def _observe_task_evidence(
        self,
        state: SummaryState,
        task: TodoItem,
        evidence_items: list[dict[str, Any]],
    ) -> TaskReactObservation:
        source_count = len(evidence_items)
        store = getattr(self, "_evidence_store", None)
        web_enabled = bool(getattr(store, "web_extraction_enabled", False))
        supporting_items = [
            item for item in evidence_items if store.is_eligible_for_facts(item.get("source_id"))
        ] if web_enabled else evidence_items
        unique_domains = {
            str(item.get("domain") or "").strip().lower()
            for item in supporting_items
            if str(item.get("domain") or "").strip()
        }
        source_diversity = len(unique_domains)
        freshness_sensitive = ReviewService._is_freshness_sensitive(state.research_topic or "", task)
        freshness_ok = (
            not freshness_sensitive
            or any(
                str(item.get("freshness_label") or "").strip() in {"fresh", "recent", "current"}
                for item in supporting_items
            )
        )
        high_quality_count = sum(
            1
            for item in supporting_items
            if str(item.get("quality_label") or "").strip() == "high"
        )
        high_quality_source = high_quality_count > 0
        full_content_available = bool(supporting_items) if web_enabled else any(
            bool(str(item.get("full_content") or "").strip())
            or len(str(item.get("snippet") or "").strip()) >= 240
            for item in evidence_items
        )

        gap_signals: list[str] = []
        if source_count == 0:
            gap_signals.append("no_results")
        else:
            if source_count < max(int(self.config.review_min_sources_per_task or 1), 1):
                gap_signals.append("low_source_count")
            if source_diversity < max(int(self.config.review_min_domains_per_task or 1), 1):
                gap_signals.append("low_source_diversity")
            if freshness_sensitive and not freshness_ok:
                gap_signals.append("stale_or_unknown_freshness")
            if not high_quality_source:
                gap_signals.append("no_high_quality_source")
            if not full_content_available:
                gap_signals.append("thin_content")

        evidence_sufficiency = (
            len(supporting_items) >= max(int(self.config.review_min_sources_per_task or 1), 1)
            and source_diversity >= max(int(self.config.review_min_domains_per_task or 1), 1)
            and freshness_ok
        )
        high_confidence = (
            evidence_sufficiency
            and source_diversity >= 3
            and high_quality_count >= 2
        )

        if high_confidence:
            stop_reason = "high_confidence_early_stop"
            continue_reason = ""
        elif evidence_sufficiency:
            stop_reason = "evidence_sufficient"
            continue_reason = ""
        elif source_count == 0:
            stop_reason = ""
            continue_reason = "当前任务尚未获得可用结果，值得补一次定向检索。"
        else:
            stop_reason = ""
            continue_reason = "当前任务仍存在来源覆盖或证据质量缺口，值得补一轮证据。"

        return TaskReactObservation(
            gap_signals=gap_signals,
            source_count=source_count,
            source_diversity=source_diversity,
            freshness_ok=freshness_ok,
            evidence_sufficiency=evidence_sufficiency,
            continue_reason=continue_reason,
            stop_reason=stop_reason,
        )

    def _build_broadened_task_query(self, state: SummaryState, task: TodoItem) -> str:
        topic = (state.research_topic or "").strip()
        title = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip())
        intent = (task.intent or "").strip()
        candidates = [
            f"{topic} {title}",
            f"{topic} {intent}",
            f"{title} {intent}",
            topic,
            title,
        ]
        for candidate in candidates:
            normalized = self._normalize_query_candidate(candidate)
            if normalized and normalized != self._normalize_query_candidate(task.query):
                return normalized
        return self._normalize_query_candidate(task.query)

    def _build_diversified_task_query(self, state: SummaryState, task: TodoItem) -> str:
        topic = (state.research_topic or "").strip()
        title = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", (task.title or "").strip())
        intent = (task.intent or "").strip()
        suffix = "official documentation report latest"
        if not ReviewService._is_freshness_sensitive(state.research_topic or "", task):
            suffix = "official documentation whitepaper report"
        return self._normalize_query_candidate(
            " ".join(part for part in [topic, title, intent, suffix] if part)
        )

    def _fallback_task_react_decision(
        self,
        state: SummaryState,
        task: TodoItem,
        observation: TaskReactObservation,
        evidence_items: list[dict[str, Any]],
        repair_hits: list[EvidenceRetrievalHit],
    ) -> TaskReactDecision:
        search_budget_left = (
            task.react_additional_search_count
            < max(int(self.config.task_react_max_additional_searches_per_task or 0), 0)
        )
        fetch_budget_left = (
            task.react_fetch_count < max(int(self.config.task_react_max_fetches_per_task or 0), 0)
        )
        store = getattr(self, "_evidence_store", None)
        top_item = self._top_evidence_item([
            item for item in evidence_items if item.get("source_origin", "web") == "web"
            and not store.is_eligible_for_facts(item.get("source_id"))
        ] if getattr(store, "web_extraction_enabled", False) else evidence_items)
        current_urls = {
            str(item.get("url") or "").strip()
            for item in evidence_items
            if str(item.get("url") or "").strip()
        }
        archive_hit = next(
            (
                hit
                for hit in repair_hits
                if not hit.citation_eligible and hit.url and hit.url not in current_urls
            ),
            None,
        )

        if observation.evidence_sufficiency:
            return TaskReactDecision(action="stop", reason="证据已足够，结束任务级闭环。")

        if "no_results" in observation.gap_signals:
            if not search_budget_left:
                return TaskReactDecision(action="stop", reason="额外搜索预算已耗尽。")
            rewritten = self._rewritten_task_query(state, task)
            if rewritten and rewritten != self._normalize_query_candidate(task.query):
                return TaskReactDecision(
                    action="rewrite_query",
                    query=rewritten,
                    reason="当前任务没有检索结果，先改写为更完整的任务查询。",
                )
            return TaskReactDecision(
                action="broaden_query",
                query=self._build_broadened_task_query(state, task),
                reason="当前任务没有检索结果，尝试更宽泛的相关查询。",
            )

        if (
            "thin_content" in observation.gap_signals
            and top_item
            and str(top_item.get("url") or "").strip()
            and not str(top_item.get("full_content") or "").strip()
            and fetch_budget_left
        ):
            return TaskReactDecision(
                action="fetch_page_for_top_source",
                source_id=str(top_item.get("source_id") or "").strip() or None,
                reason="已有来源但正文不足，先抓取最相关来源的全文。",
            )

        if archive_hit and fetch_budget_left:
            return TaskReactDecision(
                action="fetch_page_for_archive_hit",
                url=archive_hit.url,
                reason="历史证据库命中高相关正文，先抓取该页面纳入当前请求证据。",
            )

        if (
            any(
                signal in observation.gap_signals
                for signal in ("low_source_diversity", "no_high_quality_source", "stale_or_unknown_freshness")
            )
            and search_budget_left
        ):
            return TaskReactDecision(
                action="diversify_source_query",
                query=self._build_diversified_task_query(state, task),
                reason="当前来源偏单一或缺少权威/近期来源，补一轮多样化检索。",
            )

        if "low_source_count" in observation.gap_signals and search_budget_left:
            return TaskReactDecision(
                action="broaden_query",
                query=self._build_broadened_task_query(state, task),
                reason="当前来源数量偏少，补一轮更宽泛检索。",
            )

        return TaskReactDecision(action="stop", reason="继续补证据的收益有限，结束当前任务。")

    def _consult_policy_controller(
        self,
        state: SummaryState,
        task: TodoItem,
        observation: TaskReactObservation,
        evidence_items: list[dict[str, Any]],
        fallback: TaskReactDecision,
        *,
        observer: RequestTrace | None,
        repair_hits: list[EvidenceRetrievalHit] | None = None,
    ) -> TaskReactDecision | None:
        mode = "off" if getattr(self, "_research_budget", None) is not None else self.config.resolved_policy_model_mode()
        controller = getattr(self, "_policy_controller", None)
        if mode == "off":
            controller = RuleBasedPolicyController()
        elif controller is None:
            controller = build_policy_controller(self.config)

        max_rounds = max(int(self.config.task_react_max_rounds or 1), 1)
        round_index = max(1, task.react_rounds)
        searches_left = max(self.config.task_react_max_additional_searches_per_task - task.react_additional_search_count, 0)
        fetches_left = max(self.config.task_react_max_fetches_per_task - task.react_fetch_count, 0)
        high_quality_sources = sum(
            1 for item in evidence_items
            if str(item.get("quality_label") or "").lower() in {"high", "authoritative", "official"}
        )
        fetchable_evidence = [
            item for item in evidence_items
            if item.get("url") and item.get("source_id") and not item.get("full_content")
        ]
        current_targets = {item["url"]: item["source_id"] for item in fetchable_evidence}
        top_item = self._top_evidence_item(fetchable_evidence)
        current_urls = {item.get("url") for item in evidence_items}
        archive_targets = {
            hit.url for hit in repair_hits or ()
            if hit.origin == "history" and not hit.citation_eligible and hit.url and hit.url not in current_urls
        }
        pack_meta = state.skill_pack_metadata or {}
        history_actions = getattr(task, "_policy_action_history", ())
        features = PolicyStateFeatures(
            task_id=task.id,
            round_index=round_index,
            max_rounds=max_rounds,
            evidence_count=len(evidence_items),
            distinct_domains=observation.source_diversity,
            high_quality_sources=high_quality_sources,
            has_gap_signals=bool(observation.gap_signals),
            searches_left=searches_left,
            fetches_left=fetches_left,
            remaining_budget_ratio=round(max(0.0, (max_rounds - round_index) / max_rounds), 4),
            task_title=task.title,
            task_intent=task.intent,
            gap_signals=tuple(observation.gap_signals),
            can_fetch=bool(current_targets or archive_targets),
            skill_pack_id=str(pack_meta.get("pack_id") or "_neutral"),
            topic_domain=str(pack_meta.get("topic_domain") or pack_meta.get("domain") or "general"),
            freshness_ok=observation.freshness_ok,
            evidence_sufficiency=observation.evidence_sufficiency,
            archive_repair_hits=len(archive_targets),
            recent_actions=tuple(history_actions[-3:]),
        )
        context = {
            "request_id": str(getattr(self, "request_id", "") or ""),
            "tenant_id": str(getattr(self, "tenant_id", "default") or "default"),
        }
        allowed_fetch_urls = frozenset(current_targets.keys() | archive_targets)

        def record_shadow(outcome: dict[str, Any]) -> None:
            metrics_registry.record_policy_observation(outcome)
            if observer is not None:
                observer.record_policy_shadow_decision(features.task_id, round_index, outcome)

        if isinstance(controller, ShadowCanaryPolicyController):
            decision, telemetry = controller.decide_with_context(
                features, **context, allowed_fetch_urls=allowed_fetch_urls, on_shadow_result=record_shadow,
            )
        elif isinstance(controller, ModelPolicyController):
            decision, telemetry = controller.decide_with_context(
                features, **context, allowed_fetch_urls=allowed_fetch_urls,
            )
        elif hasattr(controller, "decide_with_context"):
            decision, telemetry = controller.decide_with_context(features, **context)
        else:
            decision, telemetry = controller.decide(features), None
        if telemetry and not telemetry.get("shadow_pending"):
            metrics_registry.record_policy_observation(telemetry)

        routed = None
        if mode in {"canary", "active"} and decision.source == "model" and is_budget_legal_action(features, decision):
            if decision.action == "stop":
                routed = TaskReactDecision(action="stop", reason=decision.reason)
            elif decision.action in {"search", "reflect"}:
                query = self._normalize_query_candidate(decision.query_override or "") or fallback.query or task.query
                action = fallback.action if fallback.action in {
                    "rewrite_query", "broaden_query", "diversify_source_query",
                } else "broaden_query"
                routed = TaskReactDecision(action=action, query=query, reason=decision.reason)
            elif decision.target_url in current_targets:
                routed = TaskReactDecision(
                    action="fetch_page_for_top_source", source_id=current_targets[decision.target_url],
                    reason=decision.reason,
                )
            elif decision.target_url in archive_targets:
                routed = TaskReactDecision(
                    action="fetch_page_for_archive_hit", url=decision.target_url, reason=decision.reason,
                )
            elif decision.target_url is None:
                if fallback.action in {"fetch_page_for_top_source", "fetch_page_for_archive_hit"}:
                    routed = TaskReactDecision(
                        action=fallback.action, source_id=fallback.source_id, url=fallback.url, reason=decision.reason,
                    )
                elif top_item is not None:
                    routed = TaskReactDecision(
                        action="fetch_page_for_top_source", source_id=top_item["source_id"], reason=decision.reason,
                    )
                elif archive_targets:
                    routed = TaskReactDecision(
                        action="fetch_page_for_archive_hit", url=min(archive_targets), reason=decision.reason,
                    )

        trace_decision = decision
        if routed is not None and decision.action in {"search", "reflect"} and decision.query_override:
            trace_decision = PolicyDecision(
                decision.action, decision.confidence, decision.reason, "model", query_override=routed.query,
            )
        if routed is None and mode in {"canary", "active"} and (
            decision.source == "rule_fallback" or decision.source == "model"
            or (telemetry or {}).get("circuit_open") and (mode == "active" or telemetry.get("canary_selected"))
        ):
            routed = fallback
            trace_decision = self._policy_decision_for_task_action(fallback, source=decision.source)
        elif routed is not None and decision.action == "fetch":
            trace_decision = PolicyDecision("fetch", decision.confidence, decision.reason, "model")
        if observer is not None:
            observer.record_policy_decision({
                "task_id": task.id,
                "round": round_index,
                "features": features.to_dict(),
                "decision": trace_decision.to_dict(),
                "applied": routed is not None,
                "policy_mode": mode,
                "telemetry": telemetry or {},
            })
        return routed

    @staticmethod
    def _policy_decision_for_task_action(decision: TaskReactDecision, *, source: str) -> PolicyDecision:
        action = decision.action
        if action.startswith("fetch_page_"):
            action = "fetch"
        elif action != "stop":
            action = "search"
        reason = " ".join(decision.reason.split())[:240] or "Task routing decision"
        return PolicyDecision(action, 0.0, reason, source)

    def _record_baseline_task_action(
        self,
        observer: RequestTrace | None,
        task: TodoItem,
        decision: TaskReactDecision,
        *,
        source: str,
    ) -> TaskReactDecision:
        if observer is not None:
            observer.record_policy_execution(
                task.id, max(1, task.react_rounds),
                self._policy_decision_for_task_action(decision, source=source).to_dict(),
            )
        return decision

    def _plan_task_react_decision(
        self,
        state: SummaryState,
        task: TodoItem,
        observation: TaskReactObservation,
        evidence_items: list[dict[str, Any]],
        repair_hits: list[EvidenceRetrievalHit],
        *,
        observer: RequestTrace | None,
    ) -> TaskReactDecision:
        fallback = self._fallback_task_react_decision(state, task, observation, evidence_items, repair_hits)
        routed = self._consult_policy_controller(
            state,
            task,
            observation,
            evidence_items,
            fallback,
            observer=observer,
            repair_hits=repair_hits,
        )
        if routed is not None:
            return routed

        def baseline(decision: TaskReactDecision, *, source: str = "rule") -> TaskReactDecision:
            return self._record_baseline_task_action(observer, task, decision, source=source)

        if fallback.action == "stop":
            return baseline(fallback)

        top_item = self._top_evidence_item(evidence_items) or {}
        repair_candidates = [
            {
                "origin": hit.origin,
                "citation_eligible": hit.citation_eligible,
                "source_id": hit.source_id,
                "title": hit.title,
                "url": hit.url,
                "quality_label": hit.quality_label,
                "freshness_label": hit.freshness_label,
            }
            for hit in repair_hits[:3]
        ]
        volatile_delta = (
            f"任务标题：{task.title}\n"
            f"任务目标：{task.intent}\n"
            f"当前查询：{task.query}\n"
            f"当前观察：{json.dumps(observation.to_dict(), ensure_ascii=False)}\n"
            f"修补候选：{json.dumps(repair_candidates, ensure_ascii=False)}\n"
            f"预算：{{\"max_rounds\": {self.config.task_react_max_rounds}, "
            f"\"searches_used\": {task.react_additional_search_count}, "
            f"\"searches_left\": {max(int(self.config.task_react_max_additional_searches_per_task or 0), 0) - task.react_additional_search_count}, "
            f"\"fetches_used\": {task.react_fetch_count}, "
            f"\"fetches_left\": {max(int(self.config.task_react_max_fetches_per_task or 0), 0) - task.react_fetch_count}}}\n"
            f"最相关来源：{json.dumps({'source_id': top_item.get('source_id'), 'title': top_item.get('title'), 'url': top_item.get('url')}, ensure_ascii=False)}\n"
            f"建议基线动作：{json.dumps({'action': fallback.action, 'query': fallback.query, 'source_id': fallback.source_id, 'url': fallback.url, 'reason': fallback.reason}, ensure_ascii=False)}"
        )
        builder = ContextPrefixBuilder(self.config)
        frozen = render_context_checkpoint(state, self.config)
        volatile_blocks = [
            ("task_react_delta", volatile_delta),
            ("context_delta", render_context_delta(state, self.config)),
        ]
        role = str(getattr(self.task_react_agent, "system_prompt", "") or "")
        role += web_evidence_prompt_rules(bool(getattr(getattr(self, "_evidence_store", None), "web_extraction_enabled", False)))
        if builder.enabled():
            envelope = builder.build_envelope(
                stage="task_react",
                system_role=role,
                research_topic=state.research_topic or "",
                skill_pack_context=self._skill_stage_context("task_execution"),
                frozen_index_and_ledger=frozen,
                volatile_blocks=volatile_blocks,
            )
        else:
            envelope = builder.build_envelope(
                stage="task_react",
                system_role=role,
                volatile_blocks=[
                    ("topic", f"研究主题：{state.research_topic}"),
                    *volatile_blocks,
                    ("checkpoint", frozen),
                ],
            )
        prompt = envelope.to_flat_prompt()
        run_env = getattr(self.task_react_agent, "run_envelope", None)

        try:
            if callable(run_env):
                response = run_env(envelope, config=self.config)
            else:
                response = self.task_react_agent.run(prompt)
        except Exception as exc:
            if observer:
                observer.record_llm_call(
                    success=False,
                    prompt_text=prompt,
                    completion_text="",
                    error=exc,
                    stage="task_react",
                )
            return baseline(fallback)
        finally:
            self.task_react_agent.clear_history()

        if observer:
            observer.record_llm_call(
                success=True,
                prompt_text=prompt,
                completion_text=response,
                usage=extract_llm_usage(self.task_react_agent),
                stage="task_react",
            )

        cleaned = str(response or "").strip()
        if self.config.strip_thinking_tokens:
            cleaned = strip_thinking_tokens(cleaned)
        payload = self._extract_json_payload(cleaned)
        if not isinstance(payload, dict):
            return baseline(fallback)

        action = str(payload.get("action") or "").strip()
        if action not in {
            "rewrite_query",
            "broaden_query",
            "diversify_source_query",
            "fetch_page_for_top_source",
            "fetch_page_for_archive_hit",
            "stop",
        }:
            return baseline(fallback)

        decision = TaskReactDecision(
            action=action,
            query=self._normalize_query_candidate(str(payload.get("query") or "").strip()),
            source_id=str(payload.get("source_id") or "").strip() or None,
            url=str(payload.get("url") or "").strip(),
            reason=str(payload.get("reason") or "").strip() or fallback.reason,
        )

        if decision.action == "fetch_page_for_top_source":
            if not decision.source_id and fallback.source_id:
                decision.source_id = fallback.source_id
        elif decision.action == "fetch_page_for_archive_hit":
            if not decision.url and fallback.url:
                decision.url = fallback.url
        elif decision.action != "stop" and not decision.query:
            decision.query = fallback.query

        if decision.action == "stop":
            return baseline(decision, source="task_react_llm")
        if decision.action.startswith("fetch_page_") and (
            task.react_fetch_count >= self.config.task_react_max_fetches_per_task
        ):
            return baseline(fallback)
        if not decision.action.startswith("fetch_page_") and (
            task.react_additional_search_count >= self.config.task_react_max_additional_searches_per_task
        ):
            return baseline(fallback)
        if decision.action == "fetch_page_for_top_source":
            if not any(
                item.get("source_id") == decision.source_id and item.get("url") and not item.get("full_content")
                for item in evidence_items
            ):
                return baseline(fallback)
        elif decision.action == "fetch_page_for_archive_hit":
            if not any(
                hit.origin == "history" and not hit.citation_eligible and hit.url == decision.url
                for hit in repair_hits
            ):
                return baseline(fallback)
        elif not decision.query:
            return baseline(fallback)
        return baseline(decision, source="task_react_llm")

    def _apply_task_budget(
        self,
        state: SummaryState,
        observer: RequestTrace | None,
    ) -> str | None:
        """Trim planned tasks to the configured initial planning budget."""

        max_tasks = self.config.resolved_initial_plan_max_tasks()
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
        """Run bounded evidence collection + summarization for a single task."""
        task.status = "in_progress"
        observer = self._request_trace

        evidence_store = getattr(self, "_evidence_store", None)
        if evidence_store is None:
            evidence_store = EvidenceStore()
            self._evidence_store = evidence_store

        react_enabled = bool(self.config.task_react_enabled)
        max_rounds = max(int(self.config.task_react_max_rounds or 1), 1)
        round_index = 0
        answer_text: str | None = None
        backend = ""
        notices = list(task.notices)
        next_action = "initial_search"
        next_query = (task.query or "").strip()
        next_source_id: str | None = None
        next_url = ""
        observation = TaskReactObservation(
            gap_signals=[],
            source_count=0,
            source_diversity=0,
            freshness_ok=True,
            evidence_sufficiency=False,
        )

        while True:
            round_index += 1
            if react_enabled and observer:
                observer.record_task_react_round()
            if react_enabled and emit_stream:
                yield {
                    "type": "task_iteration_started",
                    "task_id": task.id,
                    "round": round_index,
                    "action": next_action,
                    "query": next_query or task.query,
                    "step": step,
                }

            search_result: dict[str, Any] | None = None
            cache_hit = False
            cache_strategy = "miss"
            current_answer = answer_text
            current_backend = backend

            if next_action in {"fetch_page_for_top_source", "fetch_page_for_archive_hit"}:
                target_item = None
                if next_action == "fetch_page_for_top_source" and next_source_id:
                    target_item = evidence_store.get_evidence(next_source_id, include_full_content=True)
                if next_action == "fetch_page_for_top_source" and target_item is None:
                    target_item = self._top_evidence_item(evidence_store.list_task_evidence(task.id))

                target_url = (
                    next_url.strip()
                    if next_action == "fetch_page_for_archive_hit"
                    else str((target_item or {}).get("url") or "").strip()
                )
                target_source_id = (
                    None
                    if next_action == "fetch_page_for_archive_hit"
                    else str((target_item or {}).get("source_id") or "").strip() or None
                )
                if target_url:
                    task.react_fetch_count += 1
                    try:
                        fetch_reason = None
                        if evidence_store.web_extraction_enabled:
                            # Coverage comes from the previous server-side observation and selected task.
                            context = self._web_fetch_context(
                                task.id, state=state, task=task, source_id=target_source_id,
                                needs_coverage=bool(task.react_gap_signals),
                            )
                            if context is None:
                                response, fetch_reason = "{}", "not_eligible"
                            else:
                                response, fetch_reason = self.fetch_page_tool.fetch_tiered(
                                    task.id, target_source_id, target_url, context=context,
                                )
                        else:
                            response = self.fetch_page_tool.run({
                                "task_id": task.id, "source_id": target_source_id, "url": target_url,
                            })
                        payload = self._extract_json_payload(str(response or "").strip()) or {}
                        if evidence_store.web_extraction_enabled:
                            meta = payload.get("provenance_meta", {}).get("web_extraction", {})
                            tier = meta.get("tier")
                            if tier in {"L1", "L2"}:
                                fetch_notice = f"已取得净化正文（{tier}）：{payload.get('source_id') or target_source_id}"
                            elif tier == "L0":
                                fetch_notice = f"保留网页摘要（L0，{fetch_reason or 'not_eligible'}）"
                            else:
                                fetch_notice = f"未取得可用网页内容（{fetch_reason or 'quality_insufficient'}）"
                            notices = self._merge_notices(notices, [fetch_notice])
                        elif payload:
                            fetch_notice = (
                                f"已抓取来源全文：{payload.get('source_id') or target_source_id or target_url}"
                            )
                            notices = self._merge_notices(notices, [fetch_notice])
                    except Exception as exc:
                        if evidence_store.web_extraction_enabled:
                            if type(exc).__name__ in {"SnapshotConflictError", "WorkflowLeaseLost"}:
                                raise
                            notices = self._merge_notices(notices, ["未取得可用网页内容（response_invalid）"])
                        else:
                            notices = self._merge_notices(notices, [f"抓取来源正文失败：{exc}"])
                        logger.warning(
                            "Task fetch-page action failed task_id=%s title=%s error=%s",
                            task.id,
                            task.title,
                            type(exc).__name__ if evidence_store.web_extraction_enabled else exc,
                        )
                        if observer:
                            observer.record_degraded(f"task_fetch_failed:{task.id}")
                else:
                    notices = self._merge_notices(notices, ["未找到可抓取的来源正文，结束补证据。"])
            else:
                search_span = (
                    observer.start_stage(
                        "search",
                        scope="task",
                        task_id=task.id,
                        task_title=task.title,
                        metadata={"query": next_query or task.query, "react_round": round_index},
                    )
                    if observer
                    else None
                )
                if emit_stream and search_span:
                    started = search_span.started_event()
                    started["step"] = step
                    yield started

                try:
                    if round_index == 1:
                        (
                            search_result,
                            round_notices,
                            current_answer,
                            current_backend,
                            cache_hit,
                            cache_strategy,
                        ) = self._search_with_fallback_queries(
                            state,
                            task,
                            observer,
                        )
                    else:
                        task.query = next_query or task.query
                        (
                            search_result,
                            round_notices,
                            current_answer,
                            current_backend,
                            cache_hit,
                            cache_strategy,
                        ) = self._dispatch_search_with_guardrails(
                            task.query,
                            state=state,
                            task=task,
                            observer=observer,
                            notices=[],
                        )
                        task.react_additional_search_count += 1
                        if next_action == "rewrite_query":
                            round_notices = self._merge_notices(
                                round_notices,
                                [f"补证据轮次使用改写检索词：{task.query}"],
                            )
                        elif next_action == "broaden_query":
                            round_notices = self._merge_notices(
                                round_notices,
                                [f"补证据轮次使用更宽泛检索词：{task.query}"],
                            )
                        elif next_action == "diversify_source_query":
                            round_notices = self._merge_notices(
                                round_notices,
                                [f"补证据轮次补充多样化来源检索：{task.query}"],
                            )
                except MCPHitlApprovalRequired:
                    raise
                except Exception as exc:
                    refused = getattr(exc, "budget_denied", False)
                    if search_span:
                        completed = search_span.complete(
                            status="skipped" if refused else "failed",
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
                        skipped=refused,
                    ):
                        yield event
                    self._persist_request_state(state, phase="task_execution", status="in_progress")
                    return

                notices = self._merge_notices(notices, round_notices)
                self._last_search_notices = notices
                task.notices = list(notices)
                answer_text = current_answer or answer_text
                backend = current_backend or backend

                if search_result and search_result.get("results"):
                    evidence_store.record_search_results(
                        task_id=task.id,
                        query=task.query,
                        search_payload=search_result,
                        backend=backend,
                    )
                if round_index == 1:
                    self._dispatch_mcp_scope_prefetch(
                        state,
                        task,
                        evidence_store,
                        observer=observer,
                    )
                    notices = self._merge_notices(notices, task.notices)

                if search_span:
                    cache_details = {}
                    if observer:
                        cache_details = dict(observer.snapshot().get("last_search_cache_details") or {})
                    completed = search_span.complete(
                        status="success",
                        metadata={
                            "query": task.query,
                            "backend": backend,
                            "result_count": len((search_result or {}).get("results", [])),
                            "cache_hit": cache_hit,
                            "cache_strategy": cache_strategy,
                            "notice_count": len(notices),
                            "react_round": round_index,
                            **cache_details,
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

            task.evidence_items = evidence_store.list_task_evidence(task.id, include_full_content=True)
            task.sources_summary = format_evidence_sources(
                public_web_evidence_projection(task.evidence_items) if evidence_store.web_extraction_enabled else task.evidence_items,
            )
            task.react_rounds = round_index
            observation = self._observe_task_evidence(state, task, task.evidence_items)
            task.react_gap_signals = list(observation.gap_signals)
            task.react_observation = observation.to_dict()

            if react_enabled and emit_stream:
                yield {
                    "type": "task_iteration_completed",
                    "task_id": task.id,
                    "round": round_index,
                    "action": next_action,
                    "source_count": observation.source_count,
                    "source_diversity": observation.source_diversity,
                    "freshness_ok": observation.freshness_ok,
                    "evidence_sufficiency": observation.evidence_sufficiency,
                    "step": step,
                }

            if notices and emit_stream:
                for notice in notices:
                    if notice:
                        yield {
                            "type": "status",
                            "message": notice,
                            "task_id": task.id,
                            "step": step,
                        }

            should_continue = (
                react_enabled
                and round_index < max_rounds
                and bool(observation.gap_signals)
                and not observation.evidence_sufficiency
            )
            if should_continue:
                repair_hits = self._repair_hits_for_task(
                    state,
                    task,
                    observer=observer,
                )
                decision = self._plan_task_react_decision(
                    state,
                    task,
                    observation,
                    task.evidence_items,
                    repair_hits,
                    observer=observer,
                )
                executed_action = self._policy_decision_for_task_action(decision, source="rule").action
                task._policy_action_history = (*getattr(task, "_policy_action_history", ()), executed_action)[-3:]
                if decision.action != "stop":
                    task.react_last_action = decision.action
                    if observer:
                        observer.record_task_react_continue()
                    if emit_stream:
                        yield {
                            "type": "task_gap_detected",
                            "task_id": task.id,
                            "round": round_index,
                            "gap_signals": list(observation.gap_signals),
                            "continue_reason": decision.reason or observation.continue_reason,
                            "next_action": decision.action,
                            "step": step,
                        }
                    next_action = decision.action
                    next_query = decision.query
                    next_source_id = decision.source_id
                    next_url = decision.url
                    continue

                task.react_stop_reason = "low_repair_value"
                if observer:
                    observer.record_task_react_stop(task.react_stop_reason)
                if emit_stream:
                    yield {
                        "type": "task_react_stop",
                        "task_id": task.id,
                        "round": round_index,
                        "stop_reason": task.react_stop_reason,
                        "gap_signals": list(observation.gap_signals),
                        "evidence_sufficiency": observation.evidence_sufficiency,
                        "step": step,
                    }
                break

            if observation.evidence_sufficiency:
                task.react_stop_reason = observation.stop_reason or "evidence_sufficient"
            elif observation.gap_signals and round_index >= max_rounds:
                task.react_stop_reason = "max_rounds_reached"
            elif observation.gap_signals:
                task.react_stop_reason = "budget_exhausted"
            else:
                task.react_stop_reason = "no_gap_signals"

            if react_enabled and observer:
                observer.record_task_react_stop(task.react_stop_reason)
                if task.react_stop_reason not in {
                    "evidence_sufficient",
                    "no_gap_signals",
                    "high_confidence_early_stop",
                }:
                    observer.record_degraded(f"task_react_incomplete:{task.id}:{task.react_stop_reason}")
            if react_enabled and emit_stream:
                yield {
                    "type": "task_react_stop",
                    "task_id": task.id,
                    "round": round_index,
                    "stop_reason": task.react_stop_reason,
                    "gap_signals": list(observation.gap_signals),
                    "evidence_sufficiency": observation.evidence_sufficiency,
                    "step": step,
                }
            break

        if not task.evidence_items:
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
                    "react_rounds": task.react_rounds,
                    "react_gap_signals": list(task.react_gap_signals),
                    "react_stop_reason": task.react_stop_reason,
                    "step": step,
                }
                if observer:
                    yield observer.metrics_event()
            else:
                self._drain_tool_events(state)
            self._persist_request_state(state, phase="task_execution", status="in_progress")
            return

        if evidence_store.web_extraction_enabled:
            context = web_evidence_prompt_rules(True) + render_web_evidence_data(task.evidence_items, evidence_store)
            if answer_text:
                context += "\n" + json.dumps({"web_search_answer_lead": answer_text}, ensure_ascii=False)
        else:
            grounding_hits = self._grounding_hits_for_task(state, task, observer=observer)
            build_context = build_task_context_from_hits if grounding_hits else build_task_context
            context = build_context(grounding_hits or task.evidence_items, answer_text=answer_text, config=self.config)
        historical_memory_context = self._task_memory_context(state, task, observer)
        upstream_context = str(getattr(task, "upstream_context", "") or "").strip()
        if upstream_context:
            context = (
                f"### 前置依赖任务摘要（仅作子问题定位线索，不可直接作为本任务独立证据）\n"
                f"{upstream_context}\n\n{context}"
            )

        with self._state_lock:
            state.web_research_results.append(context)
            state.sources_gathered.append(task.sources_summary or "")
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
                    "latest_sources": task.sources_summary,
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
                    historical_memory_context=historical_memory_context,
                    skill_pack_context=self._skill_stage_context("task_execution"),
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
                except MCPHitlApprovalRequired:
                    raise
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
                        skipped=getattr(exc, "budget_denied", False),
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
                            historical_memory_context=historical_memory_context,
                            skill_pack_context=self._skill_stage_context("task_execution"),
                        )
                    )
                except MCPHitlApprovalRequired:
                    raise
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
                        skipped=getattr(exc, "budget_denied", False),
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

        if (
            summary_result is not None
            and bool(getattr(self.config, "resolved_sandbox_effective_enabled", lambda: False)())
            and (
                not self.config.use_tool_calling
                or str(self.config.llm_provider or "").strip().lower() == "vllm"
            )
            and getattr(self, "_sandbox_service", None) is not None
            and isinstance(getattr(summary_result, "payload", None), dict)
            and summary_result.payload.get("computation_requests")
        ):
            self._execute_prompt_only_computations(
                task=task,
                summary_result=summary_result,
                evidence_store=evidence_store,
            )

        task.evidence_items = evidence_store.list_task_evidence(task.id)
        task.sources_summary = format_evidence_sources(
            public_web_evidence_projection(task.evidence_items) if evidence_store.web_extraction_enabled else task.evidence_items,
        )
        task.summary = (summary_result.markdown or "").strip() if summary_result else "暂无可用信息"
        task.summary_payload = dict(summary_result.payload or {}) if summary_result else None
        task.claims = list(summary_result.claims or []) if summary_result else []
        task.notices = list(notices)
        task.status = "completed"
        if observer:
            observer.update_task_status_counts(completed=1)

        self._persist_request_state(state, phase="task_execution", status="in_progress")
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
                "react_rounds": task.react_rounds,
                "react_gap_signals": list(task.react_gap_signals),
                "react_last_action": task.react_last_action,
                "react_stop_reason": task.react_stop_reason,
                "react_observation": dict(task.react_observation or {}),
                "step": step,
            }
        else:
            self._drain_tool_events(state)

    def _execute_prompt_only_computations(
        self,
        *,
        task: TodoItem,
        summary_result: TaskSummaryResult,
        evidence_store: Any,
    ) -> None:
        """Execute post-summary computation_requests in Prompt-Only mode and attach [Tn-Cm] citations."""

        sandbox_service = getattr(self, "_sandbox_service", None)
        if sandbox_service is None:
            return
        requests_list = (summary_result.payload or {}).pop("computation_requests", [])
        if not requests_list:
            return

        from services.memory_sanitizer import sanitize_injection_text
        from services.sandbox_executor import (
            SandboxExecutionRequest,
            record_sandbox_result,
        )

        valid_ids = {
            str(item.get("source_id") or "").strip()
            for item in evidence_store.list_task_evidence(task.id)
            if item.get("source_id") and (
                not evidence_store.web_extraction_enabled or evidence_store.is_eligible_for_facts(item["source_id"])
            )
        }
        max_reqs = max(1, int(getattr(self.config, "sandbox_max_executions_per_task", 4) or 4))
        extra_findings: list[dict[str, Any]] = []
        for req_data in requests_list[:max_reqs]:
            if not isinstance(req_data, dict):
                continue
            try:
                title = sanitize_injection_text(req_data.get("title") or "定量计算验证")[:120] or "定量计算验证"
                request = SandboxExecutionRequest.model_validate({**req_data, "task_id": task.id, "title": title})
                exec_res = sandbox_service.execute(request, valid_source_ids=valid_ids)
            except Exception as exc:
                logger.debug("prompt-only sandbox computation skipped task=%s err=%s", task.id, exc)
                continue

            if not exec_res.ok:
                logger.debug("prompt-only sandbox computation rejected task=%s err=%s", task.id, exec_res.error)
                observer = getattr(self, "_request_trace", None)
                if observer is not None:
                    observer.record_degraded(f"sandbox_computation:{exec_res.error_code or exec_res.backend}")
                continue

            record = record_sandbox_result(
                evidence_store, request, exec_res, tenant_id=getattr(self, "tenant_id", "default"),
            )
            comp_sid = str(record.get("source_id") or "").strip()
            if comp_sid:
                finding_text = f"{title}：{record['snippet']}"
                extra_findings.append({"text": finding_text, "source_ids": [comp_sid]})

        if extra_findings:
            from services.fact_ledger import FactLedgerCompactor

            compactor = FactLedgerCompactor()
            task.fact_claims.extend(
                compactor._structured_from_text(item["text"], item["source_ids"], task)
                for item in extra_findings
            )
            summary_result.claims = list(summary_result.claims or []) + extra_findings
            if isinstance(summary_result.payload, dict):
                existing_kf = list(summary_result.payload.get("key_findings") or [])
                summary_result.payload["key_findings"] = existing_kf + extra_findings
            added_md = "\n".join(
                f"- {item['text']} [{item['source_ids'][0]}]"
                for item in extra_findings
            )
            summary_result.markdown = f"{(summary_result.markdown or '').rstrip()}\n\n{added_md}".strip()

    def _dispatch_mcp_scope_prefetch(
        self,
        state: SummaryState,
        task: TodoItem,
        evidence_store: EvidenceStore,
        *,
        observer: RequestTrace | None = None,
    ) -> None:
        """Deterministically prefetch private MCP evidence when the active skill declares auto_query_tools."""

        if not getattr(self.config, "mcp_enabled", False) or not getattr(self.config, "mcp_max_tools_per_stage", 6):
            return
        scopes = getattr(self.config, "active_mcp_tool_scopes", None) or {}
        auto_tools = scopes.get("auto_query_tools") if isinstance(scopes, dict) else None
        if not isinstance(auto_tools, list) or not auto_tools:
            return
        ensurer = getattr(self, "_ensure_mcp_bridge", None)
        bridge = ensurer() if callable(ensurer) else None
        if bridge is None:
            return

        tenant_id = str(getattr(self, "tenant_id", "default") or "default")
        request_id = str(getattr(self, "request_id", "local") or "local")
        query_text = (task.query or task.intent or task.title or state.research_topic or "").strip()
        if not query_text:
            return

        def allowed(server: str, tool: str) -> bool:
            return _scope_allows_tool(
                scopes, server_id=server, tool_name=tool, stage="task_execution", readonly=True,
            )

        def call(server: str, tool: str, **arguments: Any) -> Any:
            budget = getattr(self, "_research_budget", None)
            result = bridge.call_tool(
                server_id=server, tool_name=tool, arguments=arguments, tenant_id=tenant_id,
                request_id=request_id, task_id=task.id, active_scopes=scopes, readonly_required=True,
                stage="task_execution", observer=observer,
                **({"budget": budget} if budget is not None else {}),
            )
            if not result.ok:
                notice = f"MCP {server}:{tool} unavailable ({result.error_code})."
                if notice not in task.notices:
                    task.notices.append(notice)
            return result

        for ref in dict.fromkeys(ref for ref in auto_tools if isinstance(ref, str)):
            server_id, _, tool_name = ref.partition(":")
            if not tool_name or not allowed(server_id, tool_name):
                continue
            args: dict[str, Any] = {"query": query_text}
            if server_id == "finarag" and tool_name in {"finarag_filings", "finarag_structured_query"}:
                if not allowed("finarag", "finarag_list_companies"):
                    continue
                try:
                    companies_res = call("finarag", "finarag_list_companies", limit=200)
                except Exception as exc:
                    if not getattr(exc, "budget_denied", False):
                        raise
                    continue
                companies = (
                    (companies_res.structured_payload or {}).get("companies") or []
                    if companies_res.ok
                    else []
                )
                names = {
                    item["company_name"].strip() for item in companies if isinstance(item, dict)
                    and isinstance(item.get("company_name"), str) and item["company_name"].strip()
                }
                matches = set()
                for text in (f"{task.title} {query_text}", state.research_topic or ""):
                    matches = {name for name in names if re.search(
                        rf"(?<![a-z0-9]){re.escape(name.casefold())}(?![a-z0-9])", text.casefold(),
                    )}
                    if matches:
                        break
                if len(matches) != 1:
                    continue
                query_key = "question" if tool_name == "finarag_structured_query" else "query"
                args = {query_key: query_text, "company_name": matches.pop()}

            try:
                call_res = call(server_id, tool_name, **args)
            except Exception as exc:
                if not getattr(exc, "budget_denied", False):
                    raise
                continue
            if call_res.ok and call_res.normalized_evidence:
                evidence_store.record_mcp_results(
                    task_id=task.id,
                    query=query_text,
                    server_id=server_id,
                    tool_name=tool_name,
                    items=call_res.normalized_evidence,
                    tenant_id=tenant_id,
                )

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
        skipped: bool = False,
    ) -> list[dict[str, Any]]:
        """Update task state for a recoverable failure and emit stream events when needed."""
        task.status = "skipped" if skipped else "failed"
        task.summary = summary
        task.summary_payload = None
        task.claims = []
        if detail and detail not in task.notices:
            task.notices.append(detail)

        events: list[dict[str, Any]] = []
        if observer:
            observer.update_task_status_counts(skipped=int(skipped), failed=int(not skipped))
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
                    "status": task.status,
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
                    "react_rounds": task.react_rounds,
                    "react_gap_signals": list(task.react_gap_signals),
                    "react_last_action": task.react_last_action,
                    "react_stop_reason": task.react_stop_reason,
                    "react_observation": dict(task.react_observation or {}),
                    "step": step,
                }
            )
            if observer:
                events.append(observer.metrics_event())

        return events
