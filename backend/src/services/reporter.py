"""Service that consolidates task results into a grounded final report."""

from __future__ import annotations

import json
import re
from typing import Any

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace
from models import SummaryState
from services.evidence import (
    derive_grounded_findings_from_evidence,
    extract_citation_ids,
    looks_like_source_label_text,
    render_references,
)
from services.text_processing import normalize_agent_markdown
from utils import strip_thinking_tokens, truncate_text

_STRICT_JSON_RESPONSE_FORMAT = {"type": "json_object"}


class ReportingService:
    """Generates the final structured report."""

    _FLEXIBLE_LAYOUT_MODE = "flexible"
    _FIXED_LAYOUT_MODE = "fixed"
    _CONTENT_SECTION_DEFAULT_ORDER = (
        "background_overview",
        "key_findings",
        "evidence_and_data",
        "dimension_sections",
        "risks_and_challenges",
    )
    _CORE_SECTION_TITLES = {
        "background_overview": "背景概览",
        "key_findings": "核心洞见",
        "evidence_and_data": "证据与数据",
        "dimension_sections": "分维度展开",
        "risks_and_challenges": "风险与挑战",
        "process_notes": "研究过程说明（系统生成）",
        "references": "参考来源",
    }

    _INTERNAL_PROCESS_PATTERNS = (
        re.compile(r"\b(blocked|warning|passed)\b", re.IGNORECASE),
        re.compile(r"\bsource[_ ]?id\b", re.IGNORECASE),
        re.compile(
            r"\b(missing_angle|weak_evidence|low_source_diversity|stale_evidence|low_quality_mix|missing_citation|invalid_citation)\b",
            re.IGNORECASE,
        ),
        re.compile(r"审查提示"),
        re.compile(r"审查状态"),
        re.compile(r"当前结果存在证据风险"),
        re.compile(r"保守表述"),
        re.compile(r"仅保留通过"),
        re.compile(r"校验"),
        re.compile(r"任务\s*\d+.*状态为"),
    )

    def __init__(
        self,
        report_agent: ToolAwareSimpleAgent,
        config: Configuration,
        *,
        evidence_store,
    ) -> None:
        self._agent = report_agent
        self._config = config
        self._evidence_store = evidence_store

    def generate_report(
        self,
        state: SummaryState,
        observer: RequestTrace | None = None,
    ) -> str:
        """Generate a structured report based on completed tasks."""
        prompt = self._build_prompt(state)

        try:
            try:
                response = self._agent.run(prompt, response_format=_STRICT_JSON_RESPONSE_FORMAT)
            except Exception as exc:
                if not self._response_format_is_unsupported(exc):
                    raise
                response = self._agent.run(prompt)
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
            self._agent.clear_history()

        if observer:
            observer.record_llm_call(
                success=True,
                prompt_text=prompt,
                completion_text=response,
            )

        cleaned = response.strip()
        if self._config.strip_thinking_tokens:
            cleaned = strip_thinking_tokens(cleaned)

        payload = self._extract_json_payload(cleaned)
        if isinstance(payload, dict):
            return self._render_structured_report(
                payload,
                state=state,
                observer=observer,
            )

        report_text = normalize_agent_markdown(cleaned)
        return report_text or "报告生成失败，请检查输入。"

    def _build_prompt(self, state: SummaryState) -> str:
        task_blocks: list[str] = []
        for task in state.todo_items:
            summary_block = truncate_text(
                self._task_prompt_summary(task),
                self._config.resolved_report_summary_char_limit(),
            )
            citations = ", ".join(sorted(extract_citation_ids(summary_block))) or "无"
            source_ids = sorted(extract_citation_ids(summary_block))
            sources_block = self._task_prompt_references(
                task,
                cited_source_ids=source_ids,
            )
            task_blocks.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 已有引用：{citations}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        if self._config.resolved_report_layout_mode() == self._FIXED_LAYOUT_MODE:
            return self._build_fixed_prompt(
                research_topic=state.research_topic,
                task_blocks=task_blocks,
            )
        return self._build_flexible_prompt(
            research_topic=state.research_topic,
            task_blocks=task_blocks,
        )

    @staticmethod
    def _build_fixed_prompt(*, research_topic: str, task_blocks: list[str]) -> str:
        prompt = (
            "报告布局模式：fixed（固定结构备份模式）\n"
            f"研究主题：{research_topic}\n"
            f"任务概览：\n{''.join(task_blocks)}\n"
            "你必须先使用 `evidence_lookup` 检查各任务的 source_id；当任务总结缺少支撑时，"
            "可使用 `fetch_page` 读取某个 source_id 的网页正文。请勿编造不存在的 source_id。\n"
            "请严格输出 JSON，格式如下：\n"
            "{\n"
            '  "background_overview": "背景概览正文",\n'
            '  "key_findings": [{"claim": "结论", "source_ids": ["T1-S1", "T2-S3"]}],\n'
            '  "evidence_and_data": [{"point": "事实或数据", "source_ids": ["T1-S2"]}],\n'
            '  "risks_and_challenges": [{"risk": "风险或限制", "source_ids": ["T3-S1"]}],\n'
            '  "references": [{"source_id": "T1-S1", "title": "来源标题", "url": "https://..."}]\n'
            "}\n"
            "要求：\n"
            "1. 所有 `key_findings` / `evidence_and_data` / `risks_and_challenges` 项都必须携带 source_ids；\n"
            "2. 引用只能使用 evidence_lookup 返回的 source_id；\n"
            "3. `references` 仅保留真正被引用的来源；\n"
            "4. 如果某个观点证据不足，不要强写；\n"
            "5. `背景概览 / 核心洞见 / 证据与数据 / 风险与挑战` 只讨论研究主题本身，不要写审查提示、blocked、warning、source_id 校验、系统保守表述等内部流程话术；\n"
            "6. `风险与挑战` 只写行业、技术、治理、落地层面的真实风险，不要写本次系统执行过程里的审查问题；\n"
            "7. 尽量输出 4-6 条 `key_findings` 和 4-6 条 `evidence_and_data`，保证信息密度，不要只有一句泛泛概述。"
        )
        return prompt

    @staticmethod
    def _build_flexible_prompt(*, research_topic: str, task_blocks: list[str]) -> str:
        prompt = (
            "报告布局模式：flexible（核心必选章节 + 可选自定义章节/动态排序）\n"
            f"研究主题：{research_topic}\n"
            f"任务概览：\n{''.join(task_blocks)}\n"
            "你必须先使用 `evidence_lookup` 检查各任务的 source_id；当任务总结缺少支撑时，"
            "可使用 `fetch_page` 读取某个 source_id 的网页正文。请勿编造不存在的 source_id。\n"
            "请严格输出 JSON，格式如下：\n"
            "{\n"
            '  "background_overview": "背景概览正文",\n'
            '  "key_findings": [{"claim": "结论", "source_ids": ["T1-S1", "T2-S3"]}],\n'
            '  "evidence_and_data": [{"point": "事实或数据", "source_ids": ["T1-S2"]}],\n'
            '  "risks_and_challenges": [{"risk": "风险或限制", "source_ids": ["T3-S1"]}],\n'
            '  "custom_sections": [\n'
            "    {\n"
            '      "section_id": "market_landscape",\n'
            '      "title": "市场格局",\n'
            '      "content_type": "bullets",\n'
            '      "items": [{"text": "补充观察", "source_ids": ["T2-S1"]}]\n'
            "    },\n"
            "    {\n"
            '      "section_id": "timeline",\n'
            '      "title": "关键时间线",\n'
            '      "content_type": "paragraph",\n'
            '      "content": "一段补充正文",\n'
            '      "source_ids": ["T3-S2"]\n'
            "    }\n"
            "  ],\n"
            '  "section_order": ["background_overview", "key_findings", "market_landscape", "evidence_and_data", "risks_and_challenges"],\n'
            '  "references": [{"source_id": "T1-S1", "title": "来源标题", "url": "https://..."}]\n'
            "}\n"
            "要求：\n"
            "1. `背景概览 / 核心洞见 / 证据与数据 / 风险与挑战 / 参考来源` 是核心必选章节，始终要在 JSON 中提供对应字段；\n"
            "2. `custom_sections` 和 `section_order` 为可选字段；只有当主题确实需要时才新增自定义章节，不要为了凑结构而硬加；\n"
            "3. `section_order` 只用于安排正式正文章节顺序；系统会把 `研究过程说明（系统生成）` 和 `参考来源` 放在报告末尾；\n"
            "4. 所有 `key_findings` / `evidence_and_data` / `risks_and_challenges` 项都必须携带 source_ids；\n"
            "5. 自定义章节如果使用 `content_type=bullets`，每条 `items` 都必须携带 source_ids；如果使用 `content_type=paragraph`，章节本身必须携带 source_ids；\n"
            "6. 引用只能使用 evidence_lookup 返回的 source_id；\n"
            "7. `references` 仅保留真正被引用的来源；\n"
            "8. 如果某个观点证据不足，不要强写；\n"
            "9. `背景概览 / 核心洞见 / 证据与数据 / 风险与挑战 / 自定义章节` 只讨论研究主题本身，不要写审查提示、blocked、warning、source_id 校验、系统保守表述等内部流程话术；\n"
            "10. `风险与挑战` 只写行业、技术、治理、落地层面的真实风险，不要写本次系统执行过程里的审查问题；\n"
            "11. 尽量输出 4-6 条 `key_findings` 和 4-6 条 `evidence_and_data`，保证信息密度，不要只有一句泛泛概述。"
        )
        return prompt

    @staticmethod
    def _task_prompt_summary(task: Any) -> str:
        payload = task.summary_payload if isinstance(getattr(task, "summary_payload", None), dict) else {}
        findings = payload.get("key_findings") if isinstance(payload, dict) else None
        if isinstance(findings, list) and findings:
            lines = ["# 任务总结"]
            executive_summary = normalize_agent_markdown(str(payload.get("executive_summary") or "").strip())
            if executive_summary:
                lines.extend(["", "## 任务概述", executive_summary])
            lines.extend(["", "## 关键发现"])
            for index, item in enumerate(findings, start=1):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                citations = "".join(
                    f"[{source_id}]"
                    for source_id in item.get("source_ids") or []
                    if str(source_id).strip()
                )
                lines.append(f"{index}. {text}{(' ' + citations) if citations else ''}".rstrip())
            evidence_gaps = payload.get("evidence_gaps") or []
            if evidence_gaps:
                lines.extend(["", "## 证据边界与待补充点"])
                for item in evidence_gaps:
                    text = str(item or "").strip()
                    if text:
                        lines.append(f"- {text}")
            rendered = "\n".join(lines).strip()
            if rendered:
                return rendered

        claims = getattr(task, "claims", None)
        if isinstance(claims, list) and claims:
            lines = ["# 任务总结", "", "## 关键发现"]
            for index, item in enumerate(claims, start=1):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                citations = "".join(
                    f"[{source_id}]"
                    for source_id in item.get("source_ids") or []
                    if str(source_id).strip()
                )
                lines.append(f"{index}. {text}{(' ' + citations) if citations else ''}".rstrip())
            rendered = "\n".join(lines).strip()
            if rendered:
                return rendered

        return task.summary or "暂无可用信息"

    def _task_prompt_references(
        self,
        task: Any,
        *,
        cited_source_ids: list[str],
    ) -> str:
        evidence_items = list(getattr(task, "evidence_items", None) or [])
        available_source_ids = [
            str(item.get("source_id") or "").strip()
            for item in evidence_items
            if str(item.get("source_id") or "").strip()
        ]

        selected_source_ids: list[str] = []
        for source_id in cited_source_ids:
            if source_id in available_source_ids and source_id not in selected_source_ids:
                selected_source_ids.append(source_id)

        if not selected_source_ids:
            selected_source_ids = available_source_ids[:3]

        if selected_source_ids:
            references = self._evidence_store.build_reference_map(selected_source_ids)
            rendered = render_references(references).strip()
            if rendered:
                return truncate_text(
                    rendered,
                    self._config.resolved_report_sources_char_limit(),
                )

        fallback_sources = str(getattr(task, "sources_summary", "") or "").strip() or "暂无来源"
        return truncate_text(
            fallback_sources,
            self._config.resolved_report_sources_char_limit(),
        )

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
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
    def _response_format_is_unsupported(exc: Exception) -> bool:
        """Return whether the provider rejected the JSON-mode request contract."""

        message = str(exc or "").casefold()
        if "response_format" not in message and "json_object" not in message and "json schema" not in message:
            return isinstance(exc, TypeError) and "response_format" in message

        unsupported_markers = (
            "unsupported",
            "not support",
            "not supported",
            "invalid",
            "unknown",
            "unexpected",
            "extra inputs are not permitted",
            "extra fields not permitted",
        )
        return any(marker in message for marker in unsupported_markers)

    @staticmethod
    def _normalize_source_ids(value: Any, *, valid_source_ids: set[str]) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            source_id = str(item or "").strip()
            if not source_id or source_id in seen or source_id not in valid_source_ids:
                continue
            seen.add(source_id)
            normalized.append(source_id)
        return normalized

    def _normalize_structured_items(
        self,
        value: Any,
        *,
        text_keys: tuple[str, ...],
        valid_source_ids: set[str],
    ) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(value, list):
            return [], 0

        normalized: list[dict[str, Any]] = []
        dropped_items = 0
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    dropped_items += 1
                continue
            if not isinstance(item, dict):
                continue

            text = ""
            for key in text_keys:
                candidate = str(item.get(key) or "").strip()
                if candidate:
                    text = candidate
                    break
            if not text:
                continue
            text = self._strip_internal_process_sentences(text)
            if not text:
                dropped_items += 1
                continue
            if looks_like_source_label_text(text):
                dropped_items += 1
                continue
            source_ids = self._normalize_source_ids(
                item.get("source_ids"),
                valid_source_ids=valid_source_ids,
            )
            if not source_ids:
                dropped_items += 1
                continue
            normalized.append(
                {
                    "text": text,
                    "source_ids": source_ids,
                }
            )
        return normalized, dropped_items

    @staticmethod
    def _normalize_section_id(value: Any, *, fallback: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return fallback
        normalized = re.sub(r"\s+", "_", raw)
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "_", normalized)
        normalized = normalized.strip("_")
        return normalized or fallback

    def _normalize_custom_sections(
        self,
        value: Any,
        *,
        valid_source_ids: set[str],
    ) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(value, list):
            return [], 0

        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set(self._CORE_SECTION_TITLES.keys())
        dropped_sections = 0

        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue

            title = normalize_agent_markdown(str(item.get("title") or item.get("heading") or "").strip())
            if not title:
                dropped_sections += 1
                continue

            section_id = self._normalize_section_id(
                item.get("section_id"),
                fallback=f"custom_section_{index}",
            )
            if section_id in seen_ids:
                section_id = f"{section_id}_{index}"
            seen_ids.add(section_id)

            content_type = str(item.get("content_type") or item.get("type") or "").strip().lower()
            raw_items = item.get("items")
            if raw_items is None:
                raw_items = item.get("points")

            if content_type != "paragraph" and isinstance(raw_items, list):
                section_items, dropped_items = self._normalize_structured_items(
                    raw_items,
                    text_keys=("text", "point", "claim"),
                    valid_source_ids=valid_source_ids,
                )
                if not section_items:
                    dropped_sections += max(1, dropped_items)
                    continue
                normalized.append(
                    {
                        "section_id": section_id,
                        "title": title,
                        "content_type": "bullets",
                        "items": section_items,
                    }
                )
                continue

            content = self._strip_internal_process_sentences(
                item.get("content") or item.get("text") or item.get("body")
            )
            if not content or looks_like_source_label_text(content):
                dropped_sections += 1
                continue

            source_ids = self._normalize_source_ids(
                item.get("source_ids") or extract_citation_ids(content),
                valid_source_ids=valid_source_ids,
            )
            if not source_ids:
                dropped_sections += 1
                continue

            normalized.append(
                {
                    "section_id": section_id,
                    "title": title,
                    "content_type": "paragraph",
                    "text": content,
                    "source_ids": source_ids,
                }
            )

        return normalized, dropped_sections

    @staticmethod
    def _is_placeholder_text(value: Any) -> bool:
        text = normalize_agent_markdown(str(value or "").strip())
        if not text:
            return True

        normalized = text
        for prefix in ("- ", "* "):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break

        return normalized in {
            "暂无相关信息",
            "暂无可用信息",
            "暂无经过引用校验的结论",
        } or looks_like_source_label_text(normalized)

    @staticmethod
    def _task_query_text(task: Any) -> str:
        return " ".join(
            part
            for part in (
                getattr(task, "title", ""),
                getattr(task, "intent", ""),
                getattr(task, "query", ""),
            )
            if str(part or "").strip()
        )

    def _task_grounded_findings(
        self,
        task: Any,
        *,
        valid_source_ids: set[str],
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()

        def append_items(items: Any) -> None:
            if not isinstance(items, list):
                return

            for item in items:
                if not isinstance(item, dict):
                    continue

                raw_text = ""
                for key in ("text", "claim", "point"):
                    candidate = normalize_agent_markdown(str(item.get(key) or "").strip())
                    if candidate:
                        raw_text = candidate
                        break
                if not raw_text:
                    continue
                if looks_like_source_label_text(raw_text):
                    replacements = derive_grounded_findings_from_evidence(
                        list(getattr(task, "evidence_items", None) or []),
                        query_text=self._task_query_text(task),
                        allowed_source_ids=list(item.get("source_ids") or extract_citation_ids(raw_text)),
                        limit=1,
                    )
                    for replacement in replacements:
                        replacement_text = normalize_agent_markdown(str(replacement.get("text") or "").strip())
                        replacement_source_ids = self._normalize_source_ids(
                            replacement.get("source_ids"),
                            valid_source_ids=valid_source_ids,
                        )
                        if not replacement_text or not replacement_source_ids:
                            continue
                        key = (replacement_text, tuple(replacement_source_ids))
                        if key in seen:
                            continue
                        seen.add(key)
                        findings.append({"text": replacement_text, "source_ids": replacement_source_ids})
                    continue

                source_ids = self._normalize_source_ids(
                    item.get("source_ids") or extract_citation_ids(raw_text),
                    valid_source_ids=valid_source_ids,
                )
                if not source_ids:
                    continue

                key = (raw_text, tuple(source_ids))
                if key in seen:
                    continue
                seen.add(key)
                findings.append({"text": raw_text, "source_ids": source_ids})

        payload = getattr(task, "summary_payload", None)
        if isinstance(payload, dict):
            append_items(payload.get("key_findings"))

        append_items(getattr(task, "claims", None))
        if len(findings) < 2:
            supplements = derive_grounded_findings_from_evidence(
                list(getattr(task, "evidence_items", None) or []),
                query_text=self._task_query_text(task),
                limit=4,
            )
            for item in supplements:
                text = normalize_agent_markdown(str(item.get("text") or "").strip())
                source_ids = self._normalize_source_ids(
                    item.get("source_ids"),
                    valid_source_ids=valid_source_ids,
                )
                if not text or not source_ids:
                    continue
                key = (text, tuple(source_ids))
                if key in seen:
                    continue
                seen.add(key)
                findings.append({"text": text, "source_ids": source_ids})
        return findings

    def _build_task_fallback_sections(
        self,
        state: SummaryState,
        *,
        valid_source_ids: set[str],
    ) -> dict[str, Any]:
        task_findings: list[tuple[Any, list[dict[str, Any]]]] = []

        for task in state.todo_items:
            task_status = str(getattr(task, "status", "") or "").strip().lower()
            review_status = str(getattr(task, "review_status", "") or "").strip().lower()

            if task_status != "completed" or review_status == "blocked":
                continue

            findings = self._task_grounded_findings(task, valid_source_ids=valid_source_ids)
            if findings:
                task_findings.append((task, findings))

        covered_titles = "、".join(
            str(getattr(task, "title", "") or "").strip()
            for task, _ in task_findings[:4]
            if str(getattr(task, "title", "") or "").strip()
        )
        if task_findings:
            background = (
                f"本报告围绕“{state.research_topic}”汇总了 {len(task_findings)} 个已完成研究维度中的可追溯结论"
            )
            if covered_titles:
                background += f"，覆盖维度包括：{covered_titles}"
            background += "。以下内容优先保留已绑定 source_id 的信息。"
        else:
            background = ""

        key_findings: list[dict[str, Any]] = []
        evidence_and_data: list[dict[str, Any]] = []
        all_findings: list[dict[str, Any]] = []
        used_texts: set[str] = set()

        def append_unique(bucket: list[dict[str, Any]], item: dict[str, Any], *, limit: int) -> bool:
            text = str(item.get("text") or "").strip()
            if not text or text in used_texts or len(bucket) >= limit:
                return False
            used_texts.add(text)
            bucket.append(
                {
                    "text": text,
                    "source_ids": list(item.get("source_ids") or []),
                }
            )
            return True

        for _, findings in task_findings:
            all_findings.extend(findings)

        for item in all_findings:
            if len(key_findings) >= 4:
                break
            append_unique(key_findings, item, limit=4)

        for item in all_findings:
            if len(evidence_and_data) >= 6:
                break
            append_unique(evidence_and_data, item, limit=6)

        return {
            "background_overview": background,
            "key_findings": key_findings,
            "evidence_and_data": evidence_and_data,
            "risks_and_challenges": [],
        }

    def _merge_reference_items(
        self,
        payload_references: Any,
        *,
        cited_source_ids: list[str],
    ) -> list[dict[str, str]]:
        references_by_id = {
            item["source_id"]: item
            for item in self._evidence_store.build_reference_map(cited_source_ids)
            if item.get("source_id")
        }

        return [references_by_id[source_id] for source_id in cited_source_ids if source_id in references_by_id]

    @staticmethod
    def _render_bullets(items: list[dict[str, Any]]) -> str:
        if not items:
            return "- 暂无相关信息"
        lines: list[str] = []
        for item in items:
            text = str(item.get("text") or "").strip()
            source_ids = [f"[{source_id}]" for source_id in item.get("source_ids") or []]
            suffix = "".join(source_ids)
            lines.append(f"- {text}{(' ' + suffix) if suffix else ''}".rstrip())
        return "\n".join(lines)

    @staticmethod
    def _render_plain_bullets(items: list[str]) -> str:
        if not items:
            return "- 暂无补充说明"
        return "\n".join(f"- {item}" for item in items if str(item).strip())

    def _render_custom_section_body(self, section: dict[str, Any]) -> str:
        if str(section.get("content_type") or "").strip().lower() == "paragraph":
            text = str(section.get("text") or "").strip()
            citations = "".join(f"[{source_id}]" for source_id in section.get("source_ids") or [])
            return f"{text}{(' ' + citations) if citations else ''}".rstrip() or "暂无相关信息"
        return self._render_bullets(section.get("items") or [])

    @staticmethod
    def _render_dimension_body(dimension_sections: list[dict[str, Any]]) -> str:
        if not dimension_sections:
            return ""

        dimension_blocks: list[str] = []
        for item in dimension_sections:
            citations = "".join(f"[{source_id}]" for source_id in item.get("source_ids") or [])
            suffix = f" {citations}" if citations else ""
            dimension_blocks.append(
                f"### {item.get('title', '').strip()}\n"
                f"{str(item.get('text') or '').strip()}{suffix}".rstrip()
            )
        return "\n\n".join(dimension_blocks).strip()

    def _render_process_notes_body(self, process_notes: list[str]) -> str:
        if not process_notes:
            return ""
        return (
            "## 研究过程说明（系统生成）\n"
            "以下内容描述的是本次研究执行、证据覆盖与待补充方向，不属于研究对象本身的行业/技术风险。\n"
            f"{self._render_plain_bullets(process_notes)}"
        )

    @staticmethod
    def _collect_cited_source_ids(*groups: Any) -> list[str]:
        cited_source_ids: list[str] = []
        for group in groups:
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict):
                    continue
                nested_items: list[dict[str, Any]] = [item]
                if isinstance(item.get("items"), list):
                    nested_items.extend(nested for nested in item.get("items") or [] if isinstance(nested, dict))
                for nested in nested_items:
                    if not isinstance(nested, dict):
                        continue
                    for source_id in nested.get("source_ids") or []:
                        if source_id not in cited_source_ids:
                            cited_source_ids.append(source_id)
        return cited_source_ids

    def _resolve_flexible_section_order(
        self,
        requested_order: Any,
        *,
        custom_sections: list[dict[str, Any]],
        include_dimension_sections: bool,
    ) -> list[str]:
        available_ids: list[str] = [
            "background_overview",
            "key_findings",
            "evidence_and_data",
        ]
        available_ids.extend(section["section_id"] for section in custom_sections)
        if include_dimension_sections:
            available_ids.append("dimension_sections")
        available_ids.append("risks_and_challenges")

        requested_ids: list[str] = []
        if isinstance(requested_order, list):
            for item in requested_order:
                section_id = str(item or "").strip()
                if not section_id or section_id in requested_ids or section_id not in available_ids:
                    continue
                requested_ids.append(section_id)

        default_ids = [
            "background_overview",
            "key_findings",
            "evidence_and_data",
        ]
        default_ids.extend(section["section_id"] for section in custom_sections)
        if include_dimension_sections:
            default_ids.append("dimension_sections")
        default_ids.append("risks_and_challenges")
        return requested_ids + [section_id for section_id in default_ids if section_id not in requested_ids]

    @classmethod
    def _contains_internal_process_language(cls, value: Any) -> bool:
        text = normalize_agent_markdown(str(value or "").strip())
        if not text:
            return False
        return any(pattern.search(text) for pattern in cls._INTERNAL_PROCESS_PATTERNS)

    @classmethod
    def _strip_internal_process_sentences(cls, value: Any) -> str:
        text = normalize_agent_markdown(str(value or "").strip())
        if not text:
            return ""

        parts = re.split(r"(?<=[。！？；\n])", text)
        kept: list[str] = []
        for part in parts:
            cleaned = normalize_agent_markdown(part.strip())
            if not cleaned or cls._contains_internal_process_language(cleaned):
                continue
            kept.append(cleaned)

        if not kept and not cls._contains_internal_process_language(text):
            return text
        return "".join(kept).strip()

    @staticmethod
    def _known_source_ids(state: SummaryState) -> set[str]:
        source_ids: set[str] = set()
        for task in state.todo_items:
            for item in getattr(task, "evidence_items", []) or []:
                source_id = str(item.get("source_id") or "").strip()
                if source_id:
                    source_ids.add(source_id)
        return source_ids

    def _task_dimension_sections(
        self,
        state: SummaryState,
        *,
        valid_source_ids: set[str],
    ) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []

        for task in state.todo_items:
            if str(getattr(task, "status", "") or "").strip().lower() != "completed":
                continue

            title = str(getattr(task, "title", "") or "").strip()
            overview = self._task_dimension_overview(task, valid_source_ids=valid_source_ids)
            if not title or not overview:
                continue

            sections.append(
                {
                    "title": title,
                    "text": overview["text"],
                    "source_ids": overview["source_ids"],
                }
            )

        return sections[:4]

    def _task_dimension_overview(
        self,
        task: Any,
        *,
        valid_source_ids: set[str],
    ) -> dict[str, Any] | None:
        payload = task.summary_payload if isinstance(getattr(task, "summary_payload", None), dict) else {}
        findings = self._task_grounded_findings(task, valid_source_ids=valid_source_ids)
        if not findings:
            return None

        executive_summary = normalize_agent_markdown(str(payload.get("executive_summary") or "").strip())
        executive_summary = self._strip_internal_process_sentences(executive_summary)
        if executive_summary and not self._is_placeholder_text(executive_summary):
            source_ids: list[str] = []
            for item in findings[:2]:
                for source_id in item.get("source_ids") or []:
                    if source_id not in source_ids:
                        source_ids.append(source_id)
            if source_ids:
                return {"text": executive_summary, "source_ids": source_ids[:4]}

        text = "；".join(
            str(item.get("text") or "").strip()
            for item in findings[:2]
            if str(item.get("text") or "").strip()
        ).strip()
        if not text:
            return None

        source_ids = []
        for item in findings[:2]:
            for source_id in item.get("source_ids") or []:
                if source_id not in source_ids:
                    source_ids.append(source_id)
        if not source_ids:
            return None
        return {"text": text, "source_ids": source_ids[:4]}

    def _build_process_notes(self, state: SummaryState) -> list[str]:
        notes: list[str] = []
        seen: set[str] = set()

        def append(note: str) -> None:
            cleaned = normalize_agent_markdown(str(note or "").strip())
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            notes.append(cleaned)

        review_summary = state.review_summary if isinstance(getattr(state, "review_summary", None), dict) else {}
        overall_status = str(review_summary.get("overall_status") or "").strip().lower()
        summary_reason = self._strip_internal_process_sentences(review_summary.get("reason"))
        if overall_status == "blocked":
            append("本次研究仍有部分维度未达到理想证据覆盖，正式正文已尽量只保留可追溯内容。")
        elif overall_status == "warning":
            append("本次研究仍有部分维度需要补充复核，正式正文已尽量只保留证据更扎实的内容。")
        if summary_reason:
            append(f"本次研究的补充说明：{summary_reason}")

        for task in state.todo_items:
            title = str(getattr(task, "title", "") or "").strip() or f"任务 {getattr(task, 'id', '')}"
            status = str(getattr(task, "status", "") or "").strip().lower()
            if status in {"failed", "skipped"}:
                append(f"“{title}”这一研究维度本次未成功完成，当前报告对该方向的覆盖仍然不足。")

            payload = task.summary_payload if isinstance(getattr(task, "summary_payload", None), dict) else {}
            for gap in payload.get("evidence_gaps") or []:
                gap_text = normalize_agent_markdown(str(gap or "").strip())
                if gap_text:
                    append(f"在“{title}”这一维度，仍需补充：{gap_text}")

            for issue in getattr(task, "review_issues", []) or []:
                if not isinstance(issue, dict):
                    continue
                note = self._humanize_review_issue(title=title, issue=issue)
                if note:
                    append(note)

        return notes[:8]

    @staticmethod
    def _humanize_review_issue(*, title: str, issue: dict[str, Any]) -> str:
        check = str(issue.get("check") or "").strip().lower()
        if check == "missing_angle":
            return f"“{title}”这一维度当前缺少有效结果支撑，建议后续补做专项调研。"
        if check == "weak_evidence":
            return f"“{title}”这一维度当前登记来源偏少，证据覆盖还不够厚。"
        if check == "low_source_diversity":
            return f"“{title}”这一维度的来源集中在少数站点，交叉验证仍然不足。"
        if check == "stale_evidence":
            return f"“{title}”这一维度的可用来源缺少近期时间支撑，时效性判断仍需补充新资料。"
        if check == "low_quality_mix":
            return f"“{title}”这一维度当前缺少官方文档、论文或权威站点来源，可信度仍有提升空间。"
        if check in {"missing_citation", "invalid_citation"}:
            return f"“{title}”这一维度的部分结论仍需继续核对原始来源绑定。"
        message = normalize_agent_markdown(str(issue.get("message") or "").strip())
        if message:
            return f"“{title}”这一维度仍有待继续核实：{message}"
        return ""

    @staticmethod
    def _supplement_items(
        primary: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
        *,
        target_min: int,
        max_items: int,
    ) -> list[dict[str, Any]]:
        if len(primary) >= target_min:
            return primary[:max_items]

        combined = list(primary)
        seen = {str(item.get("text") or "").strip() for item in primary if str(item.get("text") or "").strip()}
        for item in fallback:
            text = str(item.get("text") or "").strip()
            if not text or text in seen:
                continue
            combined.append(item)
            seen.add(text)
            if len(combined) >= max_items:
                break
        return combined

    def _prepare_structured_report_content(
        self,
        payload: dict[str, Any],
        *,
        state: SummaryState,
        observer: RequestTrace | None = None,
    ) -> dict[str, Any]:
        background = self._strip_internal_process_sentences(payload.get("background_overview")) or "暂无相关信息"
        valid_source_ids = self._known_source_ids(state)
        key_findings, dropped_findings = self._normalize_structured_items(
            payload.get("key_findings"),
            text_keys=("claim", "point", "text"),
            valid_source_ids=valid_source_ids,
        )
        evidence_and_data, dropped_evidence = self._normalize_structured_items(
            payload.get("evidence_and_data"),
            text_keys=("point", "claim", "text"),
            valid_source_ids=valid_source_ids,
        )
        risks_and_challenges, dropped_risks = self._normalize_structured_items(
            payload.get("risks_and_challenges"),
            text_keys=("risk", "point", "text"),
            valid_source_ids=valid_source_ids,
        )
        custom_sections, dropped_custom_sections = self._normalize_custom_sections(
            payload.get("custom_sections"),
            valid_source_ids=valid_source_ids,
        )
        dropped_items = dropped_findings + dropped_evidence + dropped_risks + dropped_custom_sections

        if dropped_items and observer:
            observer.record_degraded(f"report_filtered_ungrounded_items:{dropped_items}")

        fallback_sections = self._build_task_fallback_sections(
            state,
            valid_source_ids=valid_source_ids,
        )
        fallback_applied = False
        dimension_sections = self._task_dimension_sections(
            state,
            valid_source_ids=valid_source_ids,
        )
        process_notes = self._build_process_notes(state)

        if self._is_placeholder_text(background) and fallback_sections["background_overview"]:
            background = fallback_sections["background_overview"]
            fallback_applied = True
        if not key_findings and fallback_sections["key_findings"]:
            key_findings = fallback_sections["key_findings"]
            fallback_applied = True
        elif fallback_sections["key_findings"]:
            supplemented = self._supplement_items(
                key_findings,
                fallback_sections["key_findings"],
                target_min=4,
                max_items=6,
            )
            fallback_applied = fallback_applied or supplemented != key_findings
            key_findings = supplemented
        if not evidence_and_data and fallback_sections["evidence_and_data"]:
            evidence_and_data = fallback_sections["evidence_and_data"]
            fallback_applied = True
        elif fallback_sections["evidence_and_data"]:
            supplemented = self._supplement_items(
                evidence_and_data,
                fallback_sections["evidence_and_data"],
                target_min=4,
                max_items=6,
            )
            fallback_applied = fallback_applied or supplemented != evidence_and_data
            evidence_and_data = supplemented
        if not risks_and_challenges and fallback_sections["risks_and_challenges"]:
            risks_and_challenges = fallback_sections["risks_and_challenges"]
            fallback_applied = True

        if fallback_applied and observer:
            observer.record_degraded("report_task_fallback_applied")

        cited_source_ids = self._collect_cited_source_ids(
            key_findings,
            evidence_and_data,
            risks_and_challenges,
            dimension_sections,
            custom_sections,
        )

        references = self._merge_reference_items(
            payload.get("references"),
            cited_source_ids=cited_source_ids,
        )
        if len(references) < len(cited_source_ids) and observer:
            observer.record_degraded("report_reference_resolution_incomplete")
        return {
            "background": background,
            "key_findings": key_findings,
            "evidence_and_data": evidence_and_data,
            "risks_and_challenges": risks_and_challenges,
            "dimension_sections": dimension_sections,
            "custom_sections": custom_sections,
            "process_notes": process_notes,
            "references": references,
            "section_order": payload.get("section_order"),
        }

    def _render_structured_report_fixed_layout(self, content: dict[str, Any]) -> str:
        dimension_body = self._render_dimension_body(content["dimension_sections"])
        process_notes_body = self._render_process_notes_body(content["process_notes"])

        markdown = (
            "## 背景概览\n"
            f"{content['background']}\n\n"
            "## 核心洞见\n"
            f"{self._render_bullets(content['key_findings'])}\n\n"
            "## 证据与数据\n"
            f"{self._render_bullets(content['evidence_and_data'])}\n\n"
            + (
                "## 分维度展开\n"
                f"{dimension_body}\n\n"
                if dimension_body
                else ""
            )
            + (
                "## 风险与挑战\n"
                f"{self._render_bullets(content['risks_and_challenges'])}\n\n"
            )
            + (
                f"{process_notes_body}\n\n"
                if process_notes_body
                else ""
            )
            + (
                "## 参考来源\n"
                f"{render_references(content['references'])}"
            )
        )
        return normalize_agent_markdown(markdown)

    def _render_structured_report_flexible_layout(self, content: dict[str, Any]) -> str:
        section_registry: dict[str, dict[str, str]] = {
            "background_overview": {
                "title": self._CORE_SECTION_TITLES["background_overview"],
                "body": content["background"],
            },
            "key_findings": {
                "title": self._CORE_SECTION_TITLES["key_findings"],
                "body": self._render_bullets(content["key_findings"]),
            },
            "evidence_and_data": {
                "title": self._CORE_SECTION_TITLES["evidence_and_data"],
                "body": self._render_bullets(content["evidence_and_data"]),
            },
            "risks_and_challenges": {
                "title": self._CORE_SECTION_TITLES["risks_and_challenges"],
                "body": self._render_bullets(content["risks_and_challenges"]),
            },
        }

        dimension_body = self._render_dimension_body(content["dimension_sections"])
        if dimension_body:
            section_registry["dimension_sections"] = {
                "title": self._CORE_SECTION_TITLES["dimension_sections"],
                "body": dimension_body,
            }

        for section in content["custom_sections"]:
            section_registry[section["section_id"]] = {
                "title": str(section.get("title") or "").strip() or "补充章节",
                "body": self._render_custom_section_body(section),
            }

        ordered_section_ids = self._resolve_flexible_section_order(
            content.get("section_order"),
            custom_sections=content["custom_sections"],
            include_dimension_sections=bool(dimension_body),
        )

        blocks = [
            f"## {section_registry[section_id]['title']}\n{section_registry[section_id]['body']}"
            for section_id in ordered_section_ids
            if section_id in section_registry
        ]

        process_notes_body = self._render_process_notes_body(content["process_notes"])
        if process_notes_body:
            blocks.append(process_notes_body)
        blocks.append(f"## 参考来源\n{render_references(content['references'])}")
        return normalize_agent_markdown("\n\n".join(blocks))

    def _render_structured_report(
        self,
        payload: dict[str, Any],
        *,
        state: SummaryState,
        observer: RequestTrace | None = None,
    ) -> str:
        content = self._prepare_structured_report_content(
            payload,
            state=state,
            observer=observer,
        )
        if self._config.resolved_report_layout_mode() == self._FIXED_LAYOUT_MODE:
            return self._render_structured_report_fixed_layout(content)
        return self._render_structured_report_flexible_layout(content)
