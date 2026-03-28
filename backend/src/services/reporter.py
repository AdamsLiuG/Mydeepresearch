"""Service that consolidates task results into a grounded final report."""

from __future__ import annotations

import json
from typing import Any

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace
from models import SummaryState
from services.evidence import extract_citation_ids, render_references
from services.text_processing import normalize_agent_markdown
from utils import strip_thinking_tokens, truncate_text


class ReportingService:
    """Generates the final structured report."""

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
            sources_block = truncate_text(
                task.sources_summary or "暂无来源",
                self._config.resolved_report_sources_char_limit(),
            )
            citations = ", ".join(sorted(extract_citation_ids(summary_block))) or "无"
            review_text = "\n".join(
                f"  - [{issue.get('severity')}] {issue.get('check')}: {issue.get('message')}"
                for issue in task.review_issues
            ) or "  - 无审查问题"
            task_blocks.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 执行状态：{task.status}\n"
                f"- 审查状态：{task.review_status}\n"
                f"- 已有引用：{citations}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
                f"- 审查发现：\n{review_text}\n"
            )

        review_summary = state.review_summary or {}
        review_summary_text = (
            f"- overall_status: {review_summary.get('overall_status', 'unknown')}\n"
            f"- reason: {review_summary.get('reason', '暂无')}\n"
            f"- issue_count: {review_summary.get('issue_count', 0)}"
        )

        prompt = (
            f"研究主题：{state.research_topic}\n"
            f"全局审查摘要：\n{review_summary_text}\n"
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
            "5. 对 review 标记为 blocked/warning 的任务，要在结论中明确保守表述或放入风险与挑战。"
        )
        return prompt

    @staticmethod
    def _task_prompt_summary(task: Any) -> str:
        payload = task.summary_payload if isinstance(getattr(task, "summary_payload", None), dict) else {}
        findings = payload.get("key_findings") if isinstance(payload, dict) else None
        if isinstance(findings, list) and findings:
            lines = ["# 任务总结", "", "## 关键发现"]
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
                lines.extend(["", "## 证据不足"])
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
    def _known_source_ids(state: SummaryState) -> set[str]:
        source_ids: set[str] = set()
        for task in state.todo_items:
            for item in getattr(task, "evidence_items", []) or []:
                source_id = str(item.get("source_id") or "").strip()
                if source_id:
                    source_ids.add(source_id)
        return source_ids

    def _render_structured_report(
        self,
        payload: dict[str, Any],
        *,
        state: SummaryState,
        observer: RequestTrace | None = None,
    ) -> str:
        background = str(payload.get("background_overview") or "").strip() or "暂无相关信息"
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
        dropped_items = dropped_findings + dropped_evidence + dropped_risks

        if dropped_items and observer:
            observer.record_degraded(f"report_filtered_ungrounded_items:{dropped_items}")

        cited_source_ids: list[str] = []
        for item in key_findings + evidence_and_data + risks_and_challenges:
            for source_id in item.get("source_ids") or []:
                if source_id not in cited_source_ids:
                    cited_source_ids.append(source_id)

        references = self._merge_reference_items(
            payload.get("references"),
            cited_source_ids=cited_source_ids,
        )
        if len(references) < len(cited_source_ids) and observer:
            observer.record_degraded("report_reference_resolution_incomplete")

        review_status = str((state.review_summary or {}).get("overall_status") or "").strip().lower()
        review_warning = ""
        if review_status in {"warning", "blocked"}:
            review_warning = (
                "审查提示：当前结果存在证据风险，以下内容仅保留通过 source_id 校验的结论，"
                "仍建议补充官方或高质量来源复核。"
            )

        risks_body = self._render_bullets(risks_and_challenges)
        if review_warning:
            risks_body = f"- {review_warning}\n{risks_body}"

        markdown = (
            "## 背景概览\n"
            f"{background}\n\n"
            "## 核心洞见\n"
            f"{self._render_bullets(key_findings)}\n\n"
            "## 证据与数据\n"
            f"{self._render_bullets(evidence_and_data)}\n\n"
            "## 风险与挑战\n"
            f"{risks_body}\n\n"
            "## 参考来源\n"
            f"{render_references(references)}"
        )
        return normalize_agent_markdown(markdown)
