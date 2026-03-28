"""Task summarization utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace
from models import SummaryState, TodoItem
from services.evidence import extract_citation_ids, format_evidence_sources
from services.notes import build_note_guidance
from services.text_processing import (
    looks_like_meta_reasoning,
    normalize_agent_markdown,
    strip_citation_markers,
    strip_tool_calls,
)
from utils import strip_thinking_tokens, truncate_text


@dataclass
class TaskSummaryResult:
    """Normalized task summary result used by downstream stages."""

    markdown: str
    payload: dict[str, Any]
    claims: list[dict[str, Any]]


class SummarizationService:
    """Handles synchronous and streaming task summarization."""

    def __init__(
        self,
        summarizer_factory: Callable[[], ToolAwareSimpleAgent],
        config: Configuration,
        *,
        evidence_store=None,
    ) -> None:
        self._agent_factory = summarizer_factory
        self._config = config
        self._evidence_store = evidence_store

    def summarize_task(
        self,
        state: SummaryState,
        task: TodoItem,
        context: str,
        observer: RequestTrace | None = None,
    ) -> TaskSummaryResult:
        """Generate a task-specific summary using the summarizer agent."""
        prompt = self._build_prompt(state, task, context)

        agent = self._agent_factory()
        try:
            response = agent.run(prompt)
        except Exception as exc:
            if observer:
                observer.record_llm_call(
                    success=False,
                    prompt_text=prompt,
                    completion_text="",
                    error=exc,
                )
            raise
        finally:
            agent.clear_history()

        if observer:
            observer.record_llm_call(
                success=True,
                prompt_text=prompt,
                completion_text=response,
            )

        return self._finalize_summary(response, task)

    def stream_task_summary(
        self,
        state: SummaryState,
        task: TodoItem,
        context: str,
        observer: RequestTrace | None = None,
    ) -> tuple[Iterator[str], Callable[[], TaskSummaryResult]]:
        """Buffer the raw stream, sanitize it, then emit safe summary chunks."""

        prompt = self._build_prompt(state, task, context)
        raw_parts: list[str] = []
        result_holder: dict[str, TaskSummaryResult] = {}
        call_failed = False
        call_recorded = False
        agent = self._agent_factory()

        def finalize() -> TaskSummaryResult:
            nonlocal call_recorded
            if "result" in result_holder:
                return result_holder["result"]

            raw_text = "".join(raw_parts)
            result = self._finalize_summary(raw_text, task)
            result_holder["result"] = result

            if observer and not call_recorded and not call_failed:
                observer.record_llm_call(
                    success=True,
                    prompt_text=prompt,
                    completion_text=raw_text,
                )
                call_recorded = True
            return result

        def generator() -> Iterator[str]:
            nonlocal call_failed, call_recorded
            try:
                for chunk in agent.stream_run(prompt):
                    if chunk:
                        raw_parts.append(chunk)
                result = finalize()
                for chunk in self._chunk_markdown(result.markdown):
                    yield chunk
            except Exception as exc:
                if observer and not call_recorded:
                    observer.record_llm_call(
                        success=False,
                        prompt_text=prompt,
                        completion_text="".join(raw_parts),
                        error=exc,
                    )
                    call_recorded = True
                call_failed = True
                raise
            finally:
                agent.clear_history()

        return generator(), finalize

    def _build_prompt(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Construct the summarization prompt shared by both modes."""
        trimmed_context = truncate_text(
            context,
            self._config.resolved_task_context_char_limit(),
        )
        evidence_directory = format_evidence_sources(task.evidence_items)
        if not evidence_directory:
            evidence_directory = "- 暂无已登记来源"

        return (
            f"任务主题：{state.research_topic}\n"
            f"任务名称：{task.title}\n"
            f"任务目标：{task.intent}\n"
            f"检索查询：{task.query}\n"
            f"可用来源目录：\n{evidence_directory}\n"
            f"任务上下文：\n{trimmed_context}\n"
            f"{build_note_guidance(task)}\n"
            "请先调用 `evidence_lookup` 读取当前任务的来源目录；当摘要不足以支撑结论时，可调用 "
            "`fetch_page` 读取某个 source_id 对应网页正文，必要时再调用 `search_web` 补充搜索。\n"
            "最终必须只输出 JSON，格式如下：\n"
            "{\n"
            '  "key_findings": [\n'
            '    {"text": "一句完整、面向用户的结论", "source_ids": ["T1-S1", "T1-S2"]}\n'
            "  ],\n"
            '  "evidence_gaps": ["证据不足或待补充说明"]\n'
            "}\n"
            "要求：\n"
            "1. `key_findings` 中每一项都必须带至少一个合法 source_id；\n"
            "2. `text` 只写用户可见结论，不要写你的思考过程、工具计划或提示词复述；\n"
            "3. `evidence_gaps` 仅写证据缺口，不要写工具调用叙述；\n"
            "4. 不允许编造 source_id；\n"
            "5. 最终输出中禁止残留 [TOOL_CALL:...] 指令或自由 Markdown。"
        )

    def _finalize_summary(self, raw_text: str, task: TodoItem) -> TaskSummaryResult:
        cleaned = (raw_text or "").strip()
        if self._config.strip_thinking_tokens:
            cleaned = strip_thinking_tokens(cleaned)
        cleaned = strip_tool_calls(cleaned).strip()

        valid_source_ids = {
            str(item.get("source_id") or "").strip()
            for item in task.evidence_items or []
            if str(item.get("source_id") or "").strip()
        }

        payload = self._extract_json_payload(cleaned)
        normalized = (
            self._normalize_summary_payload(payload, valid_source_ids)
            if isinstance(payload, dict)
            else self._fallback_summary_payload(normalize_agent_markdown(cleaned), valid_source_ids)
        )

        if not normalized["key_findings"]:
            raise ValueError("summary contained no grounded findings")

        markdown = self._render_markdown(normalized)
        claims = [
            {
                "text": item["text"],
                "source_ids": list(item["source_ids"]),
                "support_status": "unreviewed",
            }
            for item in normalized["key_findings"]
        ]
        return TaskSummaryResult(markdown=markdown, payload=normalized, claims=claims)

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
    def _normalize_source_ids(value: Any, valid_source_ids: set[str], *, text: str = "") -> list[str]:
        candidates: list[str] = []
        if isinstance(value, list):
            candidates.extend(str(item or "").strip() for item in value if str(item or "").strip())
        candidates.extend(extract_citation_ids(text))

        normalized: list[str] = []
        seen: set[str] = set()
        for source_id in candidates:
            if source_id not in valid_source_ids or source_id in seen:
                continue
            seen.add(source_id)
            normalized.append(source_id)
        return normalized

    def _normalize_summary_payload(
        self,
        payload: dict[str, Any],
        valid_source_ids: set[str],
    ) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        for item in payload.get("key_findings") or []:
            if isinstance(item, str):
                text = normalize_agent_markdown(item)
                source_ids = self._normalize_source_ids([], valid_source_ids, text=text)
            elif isinstance(item, dict):
                text = normalize_agent_markdown(str(item.get("text") or item.get("claim") or "").strip())
                source_ids = self._normalize_source_ids(item.get("source_ids"), valid_source_ids, text=text)
            else:
                continue

            text = strip_citation_markers(text)
            if not text or looks_like_meta_reasoning(text):
                continue
            if not source_ids:
                continue
            findings.append({"text": text, "source_ids": source_ids})

        evidence_gaps = self._normalize_evidence_gaps(payload.get("evidence_gaps"))
        return {"key_findings": findings, "evidence_gaps": evidence_gaps}

    def _fallback_summary_payload(
        self,
        text: str,
        valid_source_ids: set[str],
    ) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        evidence_gaps: list[str] = []

        normalized = normalize_agent_markdown(text or "")
        for raw_line in normalized.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("#", "|")):
                continue

            cleaned = line
            for pattern in (r"^[-*]\s+", r"^\d+\.\s+", r"^>\s+"):
                cleaned = re.sub(pattern, "", cleaned)
            cleaned = normalize_agent_markdown(cleaned)
            if not cleaned:
                continue
            if looks_like_meta_reasoning(cleaned):
                continue

            source_ids = self._normalize_source_ids([], valid_source_ids, text=cleaned)
            cleaned_text = strip_citation_markers(cleaned)

            if source_ids:
                findings.append({"text": cleaned_text, "source_ids": source_ids})
                continue

            if any(token in cleaned_text for token in ("证据不足", "暂无可用信息", "待补充", "仍需补充")):
                evidence_gaps.append(cleaned_text)

        return {
            "key_findings": findings,
            "evidence_gaps": self._normalize_evidence_gaps(evidence_gaps),
        }

    @staticmethod
    def _normalize_evidence_gaps(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = normalize_agent_markdown(str(item or "").strip())
            text = strip_citation_markers(text)
            if not text or looks_like_meta_reasoning(text) or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @staticmethod
    def _render_markdown(payload: dict[str, Any]) -> str:
        lines = ["# 任务总结", "", "## 关键发现"]
        findings = payload.get("key_findings") or []
        if findings:
            for index, item in enumerate(findings, start=1):
                citations = "".join(f"[{source_id}]" for source_id in item.get("source_ids") or [])
                suffix = f" {citations}" if citations else ""
                lines.append(f"{index}. {item.get('text', '').strip()}{suffix}".rstrip())
        else:
            lines.append("- 暂无经过引用校验的结论")

        evidence_gaps = payload.get("evidence_gaps") or []
        if evidence_gaps:
            lines.extend(["", "## 证据不足"])
            for item in evidence_gaps:
                lines.append(f"- {item}")

        return "\n".join(lines).strip()

    @staticmethod
    def _chunk_markdown(markdown: str, *, chunk_size: int = 160) -> Iterator[str]:
        if not markdown:
            return

        buffer = ""
        for line in markdown.splitlines(keepends=True):
            if buffer and len(buffer) + len(line) > chunk_size:
                yield buffer
                buffer = ""
            if len(line) > chunk_size:
                if buffer:
                    yield buffer
                    buffer = ""
                for start in range(0, len(line), chunk_size):
                    yield line[start : start + chunk_size]
                continue
            buffer += line

        if buffer:
            yield buffer
