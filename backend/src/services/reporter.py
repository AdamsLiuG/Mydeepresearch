"""Service that consolidates task results into a grounded final report."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from html import escape
from typing import Any
from urllib.parse import urlsplit

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace, extract_llm_usage
from models import SummaryState
from prompts import web_evidence_prompt_rules
from services.context_builder import (
    ContextPrefixBuilder,
    PromptEnvelope,
    render_web_evidence_data,
)
from services.evidence import (
    derive_grounded_findings_from_evidence,
    extract_citation_ids,
    looks_like_source_label_text,
    render_references,
)
from services.evidence_index import (
    EvidenceRetrievalService,
    render_disputed_source_chunks,
)
from services.fact_ledger import (
    context_covered_task_ids,
    render_context_checkpoint,
    render_context_delta,
)
from services.sandbox_executor import sanitize_chart_spec
from services.text_processing import (
    CITATION_PATTERN,
    iter_markdown_lines,
    normalize_agent_markdown,
    strip_citation_markers,
    truncate_cited_text,
)
from utils import strip_thinking_tokens, truncate_text

_STRICT_JSON_RESPONSE_FORMAT = {"type": "json_object"}


def _safe_workbench_url(value: Any) -> str:
    url = str(value or "").strip()
    if not re.match(r"https?://", url, flags=re.IGNORECASE) or re.search(r"[\x00-\x20\x7f\\]", url):
        return ""
    try:
        return url if urlsplit(url).hostname else ""
    except ValueError:
        return ""


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
        "source_conflicts": "多源口径分歧与证据对照",
        "process_notes": "研究过程说明（系统生成）",
        "references": "参考来源",
    }

    _INTERNAL_PROCESS_PATTERNS = (
        re.compile(
            r"(?:审查|评审|任务|review|status)[\s:：为是]{0,12}\b(blocked|warning|passed)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(blocked|warning|passed)\b\s*(?:或|/|、|,)\s*\b(blocked|warning|passed)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bsource[_ ]?id\b", re.IGNORECASE),
        re.compile(
            r"\b(missing_angle|weak_evidence|low_source_diversity|stale_evidence|low_quality_mix|missing_citation|invalid_citation|claim_evidence_mismatch)\b",
            re.IGNORECASE,
        ),
        re.compile(r"审查提示"),
        re.compile(r"审查状态"),
        re.compile(r"审查发现"),
        re.compile(r"当前结果存在证据风险"),
        re.compile(r"(?:报告结论采取|系统|明确保守)保守表述|保守表述"),
        re.compile(r"仅保留通过.*?(?:校验|验证)"),
        re.compile(r"source[_ ]?id\s*校验|引用校验|审查校验"),
        re.compile(r"系统生成"),
        re.compile(r"任务\s*\d+.*状态为"),
    )

    def __init__(
        self,
        report_agent: ToolAwareSimpleAgent,
        config: Configuration,
        *,
        evidence_store,
        retrieval_service: EvidenceRetrievalService | None = None,
    ) -> None:
        self._agent = report_agent
        self._config = config
        self._evidence_store = evidence_store
        self._retrieval_service = retrieval_service

    def generate_report(
        self,
        state: SummaryState,
        observer: RequestTrace | None = None,
        skill_pack_context: str | None = None,
    ) -> str:
        """Generate a structured report based on completed tasks."""
        envelope = self._build_envelope(state, skill_pack_context=skill_pack_context)
        prompt = envelope.to_flat_prompt()
        run_env = getattr(self._agent, "run_envelope", None)
        use_envelope = callable(run_env)

        try:
            try:
                if use_envelope:
                    response = run_env(
                        envelope,
                        config=self._config,
                        response_format=_STRICT_JSON_RESPONSE_FORMAT,
                    )
                else:
                    response = self._agent.run(prompt, response_format=_STRICT_JSON_RESPONSE_FORMAT)
            except Exception as exc:
                if not self._response_format_is_unsupported(exc):
                    raise
                if use_envelope:
                    response = run_env(envelope, config=self._config)
                else:
                    response = self._agent.run(prompt)
        except Exception as exc:
            if observer:
                observer.record_llm_call(
                    success=False,
                    prompt_text=prompt,
                    completion_text="",
                    error=exc,
                    stage="report",
                )
            raise
        finally:
            self._agent.clear_history()

        if observer:
            observer.record_llm_call(
                success=True,
                prompt_text=prompt,
                completion_text=response,
                usage=extract_llm_usage(self._agent),
                stage="report",
            )

        cleaned = response.strip()
        if self._config.strip_thinking_tokens:
            cleaned = strip_thinking_tokens(cleaned)

        payload = self._extract_json_payload(cleaned)
        if isinstance(payload, dict):
            report = self._render_structured_report(payload, state=state, observer=observer)
        else:
            report = self._revise_markdown_report(cleaned, state)
        return self._sanitize_chunk_citations(report, self._collect_valid_source_ids(state)) or "报告生成失败，请检查输入。"

    def stream_generate_report(
        self,
        state: SummaryState,
        observer: RequestTrace | None = None,
        skill_pack_context: str | None = None,
        outline_sections: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield progressive outline + validated report_chunk events while populating state.structured_report."""

        task_titles = [
            str(task.title).strip()
            for task in state.todo_items
            if str(task.title or "").strip()
        ]
        if outline_sections:
            raw_outline = [
                *(str(sec).strip() for sec in outline_sections if str(sec or "").strip()),
                *task_titles,
                self._CORE_SECTION_TITLES["references"],
            ]
        else:
            raw_outline = [
                self._CORE_SECTION_TITLES["background_overview"],
                self._CORE_SECTION_TITLES["key_findings"],
                self._CORE_SECTION_TITLES["evidence_and_data"],
                *task_titles,
                self._CORE_SECTION_TITLES["risks_and_challenges"],
                self._CORE_SECTION_TITLES["references"],
            ]
        if self._filtered_conflicts(state, self._collect_valid_source_ids(state)):
            raw_outline.insert(-1, self._CORE_SECTION_TITLES["source_conflicts"])
        resolved_outline: list[str] = []
        for title in raw_outline:
            if title and title not in resolved_outline:
                resolved_outline.append(title)
        yield {
            "type": "report_outline",
            "sections": resolved_outline,
        }

        report = self.generate_report(
            state,
            observer=observer,
            skill_pack_context=skill_pack_context,
        )
        state.structured_report = report

        chunks = self._split_report_chunks(report)
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            yield {
                "type": "report_chunk",
                "content": chunk,
                "index": index,
                "total_chunks": total_chunks,
            }
        for chart in getattr(state, "report_charts", None) or []:
            if isinstance(chart, dict):
                yield {
                    "type": "report_chart",
                    "chart": dict(chart),
                }

    def _collect_valid_source_ids(self, state: SummaryState) -> set[str]:
        valid = self._known_source_ids(state)
        if self._evidence_store is not None:
            for task in state.todo_items:
                for item in self._evidence_store.list_task_evidence(task.id):
                    sid = str(item.get("source_id") or "").strip()
                    if sid:
                        valid.add(sid)
        if getattr(self._evidence_store, "web_extraction_enabled", False):
            valid = {sid for sid in valid if self._evidence_store.is_eligible_for_facts(sid)}
        return valid - self._rejected_computation_ids(state)

    @staticmethod
    def _sanitize_chunk_citations(text: str, valid_source_ids: set[str]) -> str:
        """Strip citation tokens absent from the request whitelist."""
        if not text:
            return ""
        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            return match.group(0) if token in valid_source_ids else ""

        return "".join(line if fenced else CITATION_PATTERN.sub(_replace, line) for line, fenced in iter_markdown_lines(text))

    def _revise_markdown_report(self, report: str, state: SummaryState) -> str:
        if report.startswith(("{", '"')):
            report = normalize_agent_markdown(report)
        rejected = self._rejected_claim_texts(state.todo_items)
        for claim in sorted(rejected, key=lambda text: (-len(text), text)):
            pattern = re.compile(re.escape(claim), re.IGNORECASE)
            report = "".join(
                line if fenced else pattern.sub("该陈述与所引证据不符，已省略。", line)
                for line, fenced in iter_markdown_lines(report)
            )

        conflicts = self._filtered_conflicts(state, self._collect_valid_source_ids(state))
        body = self._render_source_conflicts_body(conflicts)
        if not body:
            return report

        lines = list(iter_markdown_lines(report))
        reference_heading = f"## {self._CORE_SECTION_TITLES['references']}"
        reference_position = next(
            (i for i, (line, fenced) in enumerate(lines) if not fenced and line.strip() == reference_heading),
            len(lines),
        )
        before = "".join(line for line, _ in lines[:reference_position]).rstrip()
        after = "".join(line for line, _ in lines[reference_position:]).strip()
        references = self._merge_reference_items(
            None, cited_source_ids=self._collect_conflict_source_ids(conflicts), state=state,
        )
        end_of_references = next(
            (
                i for i in range(reference_position + 1, len(lines))
                if not lines[i][1] and re.match(r"^ {0,3}#{1,2}\s", lines[i][0])
            ),
            len(lines),
        )
        if reference_position < len(lines):
            reference_text = "".join(line for line, _ in lines[reference_position:end_of_references]).strip()
            missing_references = [item for item in references if item["source_id"] not in extract_citation_ids(reference_text)]
            if missing_references:
                trailing = "".join(line for line, _ in lines[end_of_references:]).strip()
                after = "\n\n".join(part for part in (reference_text, render_references(missing_references), trailing) if part)
        else:
            after = f"{reference_heading}\n{render_references(references)}"
        heading = self._CORE_SECTION_TITLES["source_conflicts"]
        return "\n\n".join(part for part in (before, f"## {heading}\n{body}", after) if part)

    @staticmethod
    def _split_report_chunks(report: str, *, max_chunk_chars: int = 800) -> list[str]:
        """Split markdown report into readable section/paragraph chunks without breaking citations."""
        normalized = str(report or "").strip()
        if not normalized:
            return []
        paragraphs = [part for part in re.split(r"(\n{2,})", normalized) if part]
        chunks: list[str] = []
        buffer = ""
        for part in paragraphs:
            if buffer and len(buffer) + len(part) > max_chunk_chars:
                chunks.append(buffer)
                buffer = part
            else:
                buffer += part
        if buffer:
            chunks.append(buffer)
        return chunks

    def _build_envelope(
        self,
        state: SummaryState,
        *,
        skill_pack_context: str | None = None,
    ) -> PromptEnvelope:
        builder = ContextPrefixBuilder(self._config)
        web_enabled = getattr(self._evidence_store, "web_extraction_enabled", False)
        valid_ids = self._collect_valid_source_ids(state)
        covered_task_ids = context_covered_task_ids(state, self._config)
        task_blocks: list[str] = []
        for task in state.todo_items:
            if task.id in covered_task_ids:
                task_blocks.append(f"### 任务 {task.id}: {task.title}\n- 任务目标：{task.intent}\n")
                continue
            summary = self._task_prompt_summary(task) if not web_enabled else self._render_bullets(
                self._task_grounded_findings(task, valid_source_ids=valid_ids),
            )
            summary_block = truncate_cited_text(
                self._sanitize_chunk_citations(summary, valid_ids) if web_enabled else summary,
                self._config.resolved_report_summary_char_limit(),
            )
            citations = ", ".join(sorted(extract_citation_ids(summary_block))) or "无"
            source_ids = sorted(extract_citation_ids(summary_block))
            sources_block = (
                render_web_evidence_data(task.evidence_items, self._evidence_store, max_chars=3000)
                if web_enabled else self._task_prompt_references(task, cited_source_ids=source_ids)
            )
            task_blocks.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 已有引用：{citations}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        tools_enabled = self._tools_enabled()
        layout_mode = self._config.resolved_report_layout_mode()
        build_contract = self._build_fixed_prompt if layout_mode == self._FIXED_LAYOUT_MODE else self._build_flexible_prompt
        static_schema = build_contract(tools_enabled=tools_enabled) + web_evidence_prompt_rules(web_enabled)

        frozen = render_context_checkpoint(state, self._config)
        volatile_blocks: list[tuple[str, str]] = [
            ("report_tasks", f"任务概览：\n{''.join(task_blocks)}"),
            ("context_delta", render_context_delta(state, self._config)),
        ]
        raw_conflicts = self._filtered_conflicts(state, valid_ids)
        if raw_conflicts:
            disputed_ids: list[str] = []
            conflict_hints: list[str] = []
            for cluster in raw_conflicts[:5]:
                if not isinstance(cluster, dict):
                    continue
                disputed_ids.extend(
                    str(side.get("source_id") or "").strip()
                    for side in cluster.get("sides") or []
                    if isinstance(side, dict)
                )
                topic_label = " / ".join(str(cluster[key]) for key in ("entity", "metric", "period") if cluster.get(key))
                rationale = self._conflict_rationale(cluster)
                if topic_label:
                    conflict_hints.append(f"- {topic_label}：{truncate_text(rationale, 140)}")
            if conflict_hints:
                volatile_blocks.append(
                    (
                        "source_conflicts_hint",
                        "以下客观多源口径差异需在正文中如实保留并说明前提边界，不要压缩为单一绝对值：\n"
                        + "\n".join(conflict_hints),
                    )
                )

            if covered_task_ids:
                volatile_blocks.append((
                    "disputed_chunks",
                    render_disputed_source_chunks(
                        self._retrieval_service, [sid for sid in disputed_ids if sid in valid_ids]
                    ),
                ))

        role = str(getattr(self._agent, "system_prompt", "") or "")
        if not builder.enabled():
            return builder.build_envelope(
                stage="report",
                system_role=role,
                volatile_blocks=[
                    ("topic", f"研究主题：{state.research_topic}"),
                    *volatile_blocks,
                    ("output_contract", static_schema),
                    ("checkpoint", frozen),
                    ("skills", str(skill_pack_context or "")),
                ],
            )
        return builder.build_envelope(
            stage="report",
            system_role=role,
            static_schema_and_rules=static_schema,
            research_topic=state.research_topic or "",
            skill_pack_context=str(skill_pack_context or "").strip(),
            frozen_index_and_ledger=frozen,
            volatile_blocks=volatile_blocks,
        )

    def _build_prompt(self, state: SummaryState) -> str:
        return self._build_envelope(state).to_flat_prompt()

    def _tools_enabled(self) -> bool:
        provider = str(getattr(self._config, "llm_provider", "") or "").strip().lower()
        if provider == "vllm" or not getattr(self._config, "use_tool_calling", True):
            return False
        if self._agent is not None and getattr(self._agent, "enable_tool_calling", None) is False:
            return False
        return True

    @staticmethod
    def _build_fixed_prompt(
        *,
        tools_enabled: bool = True,
    ) -> str:
        tool_guidance = (
            "你必须先使用 `evidence_lookup` 检查各任务的 source_id；当任务总结缺少支撑时，"
            "可使用 `evidence_lookup(mode=\"lazy_chunks\")` 按 source_id 读取正文。请勿编造不存在的 source_id。\n"
            if tools_enabled
            else "请直接基于任务概览、事实账本和来源目录中已登记的 source_id 组织报告，"
            "不要尝试调用任何工具，也不要编造不存在的 source_id。\n"
        )
        citation_rule = (
            "2. 引用只能使用 evidence_lookup 返回的 source_id；\n"
            if tools_enabled
            else "2. 引用只能使用任务概览、事实账本和来源目录中已登记的 source_id；\n"
        )
        prompt = (
            "报告布局模式：fixed（固定结构备份模式）\n"
            f"{tool_guidance}"
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
            f"{citation_rule}"
            "3. `references` 仅保留真正被引用的来源；\n"
            "4. 如果某个观点证据不足，不要强写；\n"
            "5. `背景概览 / 核心洞见 / 证据与数据 / 风险与挑战` 只讨论研究主题本身，不要写审查提示、blocked、warning、source_id 校验、系统保守表述等内部流程话术；\n"
            "6. `风险与挑战` 只写行业、技术、治理、落地层面的真实风险，不要写本次系统执行过程里的审查问题；\n"
            "7. 尽量输出 4-6 条 `key_findings` 和 4-6 条 `evidence_and_data`，保证信息密度，不要只有一句泛泛概述。"
        )
        return prompt

    @staticmethod
    def _build_flexible_prompt(
        *,
        tools_enabled: bool = True,
    ) -> str:
        tool_guidance = (
            "你必须先使用 `evidence_lookup` 检查各任务的 source_id；当任务总结缺少支撑时，"
            "可使用 `evidence_lookup(mode=\"lazy_chunks\")` 按 source_id 读取正文。请勿编造不存在的 source_id。\n"
            if tools_enabled
            else "请直接基于任务概览、事实账本和来源目录中已登记的 source_id 组织报告，"
            "不要尝试调用任何工具，也不要编造不存在的 source_id。\n"
        )
        citation_rule = (
            "6. 引用只能使用 evidence_lookup 返回的 source_id；\n"
            if tools_enabled
            else "6. 引用只能使用任务概览、事实账本和来源目录中已登记的 source_id；\n"
        )
        prompt = (
            "报告布局模式：flexible（核心必选章节 + 可选自定义章节/动态排序）\n"
            f"{tool_guidance}"
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
            f"{citation_rule}"
            "7. `references` 仅保留真正被引用的来源；\n"
            "8. 如果某个观点证据不足，不要强写；\n"
            "9. `背景概览 / 核心洞见 / 证据与数据 / 风险与挑战 / 自定义章节` 只讨论研究主题本身，不要写审查提示、blocked、warning、source_id 校验、系统保守表述等内部流程话术；\n"
            "10. `风险与挑战` 只写行业、技术、治理、落地层面的真实风险，不要写本次系统执行过程里的审查问题；\n"
            "11. 尽量输出 4-6 条 `key_findings` 和 4-6 条 `evidence_and_data`，保证信息密度，不要只有一句泛泛概述。"
        )
        return prompt

    @staticmethod
    def _rejected_claim_texts(tasks: list[Any]) -> set[str]:
        return {
            clean
            for task in tasks
            for item in (getattr(task, "claims", None) or []) + (getattr(task, "atomic_claims", None) or [])
            if isinstance(item, dict)
            and (item.get("support_status") == "contradicted" or item.get("nli_label") == "CONTRADICTED")
            and (clean := " ".join(strip_citation_markers(str(item.get("text") or "")).split()).rstrip("。.!?！？"))
        }

    @staticmethod
    def _has_rejected_claim(text: Any, rejected: set[str]) -> bool:
        if not rejected:
            return False
        cleaned = " ".join(strip_citation_markers(str(text or "")).split()).casefold()
        return any(claim.casefold() in cleaned for claim in rejected)

    @staticmethod
    def _task_prompt_summary(task: Any) -> str:
        rejected = ReportingService._rejected_claim_texts([task])
        payload = task.summary_payload if isinstance(getattr(task, "summary_payload", None), dict) else {}
        findings = payload.get("key_findings") if isinstance(payload, dict) else None
        if isinstance(findings, list) and findings:
            lines = ["# 任务总结"]
            executive_summary = normalize_agent_markdown(str(payload.get("executive_summary") or "").strip())
            if executive_summary and not ReportingService._has_rejected_claim(executive_summary, rejected):
                lines.extend(["", "## 任务概述", executive_summary])
            lines.extend(["", "## 关键发现"])
            for index, item in enumerate(findings, start=1):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text or ReportingService._has_rejected_claim(text, rejected):
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
            if rendered and (len(lines) > 3 or not rejected):
                return rendered

        claims = getattr(task, "claims", None)
        if isinstance(claims, list) and claims:
            lines = ["# 任务总结", "", "## 关键发现"]
            for index, item in enumerate(claims, start=1):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text or ReportingService._has_rejected_claim(text, rejected):
                    continue
                citations = "".join(
                    f"[{source_id}]"
                    for source_id in item.get("source_ids") or []
                    if str(source_id).strip() and source_id not in extract_citation_ids(text)
                )
                lines.append(f"{index}. {text}{(' ' + citations) if citations else ''}".rstrip())
            rendered = "\n".join(lines).strip()
            if rendered:
                return rendered

        if ReportingService._has_rejected_claim(task.summary, rejected):
            return "暂无经过引用校验的结论"
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
            if isinstance(payload, dict) and payload.keys() & {
                "background_overview", "key_findings", "evidence_and_data", "risks_and_challenges",
                "custom_sections", "references",
            }:
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
        rejected = self._rejected_claim_texts([task])
        web_enabled = getattr(self._evidence_store, "web_extraction_enabled", False)
        evidence = list(getattr(task, "evidence_items", None) or [])
        if web_enabled:
            evidence = []
            for item in self._evidence_store.list_task_evidence(task.id, include_full_content=True):
                if item["source_id"] in valid_source_ids:
                    if item.get("provenance_meta", {}).get("web_extraction"):
                        item["snippet"] = ""
                    evidence.append(item)

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
                if not raw_text or self._has_rejected_claim(raw_text, rejected):
                    continue
                raw_source_ids = item.get("source_ids") or extract_citation_ids(raw_text)
                if web_enabled and not (
                    raw_source_ids and set(raw_source_ids) | set(extract_citation_ids(raw_text)) <= valid_source_ids
                ):
                    continue
                if looks_like_source_label_text(raw_text):
                    replacements = derive_grounded_findings_from_evidence(
                        evidence,
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
                        if not replacement_text or not replacement_source_ids or self._has_rejected_claim(replacement_text, rejected):
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
                evidence,
                query_text=self._task_query_text(task),
                limit=4,
            )
            for item in supplements:
                text = normalize_agent_markdown(str(item.get("text") or "").strip())
                source_ids = self._normalize_source_ids(
                    item.get("source_ids"),
                    valid_source_ids=valid_source_ids,
                )
                if not text or not source_ids or self._has_rejected_claim(text, rejected):
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
        state: SummaryState,
    ) -> list[dict[str, str]]:
        candidates = self._evidence_store.build_reference_map(cited_source_ids) if self._evidence_store is not None else []
        references_by_id = {
            item["source_id"]: item
            for item in candidates
            if item.get("source_id")
        }
        for task in state.todo_items:
            for item in task.evidence_items or []:
                if isinstance(item, dict) and item.get("source_id"):
                    references_by_id.setdefault(item["source_id"], item)

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
            separator = "\n\n" if "\n" in text else " "
            return f"{text}{(separator + citations) if citations else ''}".rstrip() or "暂无相关信息"
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
        text = str(value or "").strip()
        if text.startswith(("{", '"')):
            text = normalize_agent_markdown(text)
        if not text:
            return ""

        kept: list[str] = []
        for line, fenced in iter_markdown_lines(text):
            if fenced or not cls._contains_internal_process_language(line):
                kept.append(line)
            elif "|" not in line:
                parts = re.split(r"(?<=[。！？；])", line.rstrip("\r\n"))
                clean = "".join(part for part in parts if not cls._contains_internal_process_language(part))
                if clean.strip():
                    kept.append(clean + line[len(line.rstrip("\r\n")):])
        return "".join(kept).strip()

    @staticmethod
    def _rejected_computation_ids(state: SummaryState) -> set[str]:
        return {
            sid for task in state.todo_items for issue in (task.review_issues or [])
            if isinstance(issue, dict) and issue.get("check") == "sandbox_input_ungrounded"
            for sid in (issue.get("source_ids") or [])
            if isinstance(sid, str) and re.fullmatch(r"T[0-9]+-C[0-9]+", sid)
        }

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
        if (
            executive_summary and not self._is_placeholder_text(executive_summary)
            and not self._has_rejected_claim(executive_summary, self._rejected_claim_texts([task]))
        ):
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
        if check == "claim_contradicted":
            return f"“{title}”这一维度的部分陈述与所引证据冲突。"
        if check == "cross_source_conflict":
            return ""
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
        rejected = self._rejected_claim_texts(state.todo_items)
        background = self._strip_internal_process_sentences(payload.get("background_overview")) or "暂无相关信息"
        if self._has_rejected_claim(background, rejected):
            background = "暂无相关信息"
        valid_source_ids = self._collect_valid_source_ids(state)
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
        for items in (key_findings, evidence_and_data, risks_and_challenges):
            items[:] = [item for item in items if not self._has_rejected_claim(item["text"], rejected)]
        for section in custom_sections:
            if section.get("content_type") == "bullets":
                section["items"] = [
                    item for item in section["items"] if not self._has_rejected_claim(item["text"], rejected)
                ]
        custom_sections = [
            section for section in custom_sections
            if not self._has_rejected_claim(section.get("text"), rejected)
            and (section.get("content_type") != "bullets" or section["items"])
        ]
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

        filtered_conflicts = self._filtered_conflicts(state, valid_source_ids)
        cited_source_ids = list(dict.fromkeys([*cited_source_ids, *self._collect_conflict_source_ids(filtered_conflicts)]))

        references = self._merge_reference_items(
            payload.get("references"),
            cited_source_ids=cited_source_ids,
            state=state,
        )
        if len(references) < len(cited_source_ids) and observer:
            observer.record_degraded("report_reference_resolution_incomplete")

        computed_records: dict[str, dict[str, Any]] = {}
        for task in getattr(state, "todo_items", []) or []:
            items = list(task.evidence_items or [])
            if self._evidence_store is not None:
                items.extend(self._evidence_store.list_task_evidence(task.id))
            for item in items:
                sid = str(item.get("source_id") or "") if isinstance(item, dict) else ""
                if sid in valid_source_ids and re.fullmatch(r"T[0-9]+-C[0-9]+", sid):
                    computed_records[sid] = item

        sanitized_charts: list[dict[str, Any]] = []
        seen_chart_ids: set[str] = set()
        for sid, item in computed_records.items():
            structured = item.get("structured_payload")
            charts = structured.get("charts") if isinstance(structured, dict) else None
            for raw_chart in charts if isinstance(charts, list) else []:
                if not isinstance(raw_chart, dict) or not isinstance(raw_chart.get("source_ids", []), list):
                    continue
                candidate = {**raw_chart, "source_ids": [sid, *raw_chart.get("source_ids", [])]}
                chart = sanitize_chart_spec(candidate, valid_source_ids=valid_source_ids)
                if chart is None:
                    continue
                cid = chart["chart_id"]
                if cid in seen_chart_ids:
                    cid = f"{sid}-{cid}"
                    if cid in seen_chart_ids:
                        continue
                    chart["chart_id"] = cid
                    chart["anchor_id"] = cid
                seen_chart_ids.add(cid)
                sanitized_charts.append(chart)
        state.report_charts = list(sanitized_charts)

        return {
            "background": background,
            "key_findings": key_findings,
            "evidence_and_data": evidence_and_data,
            "risks_and_challenges": risks_and_challenges,
            "dimension_sections": dimension_sections,
            "custom_sections": custom_sections,
            "process_notes": process_notes,
            "references": references,
            "charts": sanitized_charts,
            "source_conflicts": filtered_conflicts,
            "section_order": payload.get("section_order"),
        }

    @staticmethod
    def _filtered_conflicts(state: SummaryState, valid_source_ids: set[str]) -> list[dict[str, Any]]:
        conflicts = []
        for cluster in getattr(state, "evidence_conflicts", None) or []:
            if not isinstance(cluster, dict):
                continue
            sides: dict[str, dict[str, Any]] = {}
            for side in cluster.get("sides") or []:
                if not isinstance(side, dict):
                    continue
                sid = str(side.get("source_id") or "").strip()
                if sid in valid_source_ids and CITATION_PATTERN.fullmatch(f"[{sid}]"):
                    sides.setdefault(sid, {**side, "source_id": sid})
            if len(sides) < 2:
                continue
            filtered = {**cluster, "sides": list(sides.values())}
            if isinstance(cluster.get("edges"), list):
                edges = []
                seen_pairs = set()
                for edge in cluster["edges"]:
                    if not isinstance(edge, dict):
                        continue
                    a = str(edge.get("source_id_a") or "")
                    b = str(edge.get("source_id_b") or "")
                    key = tuple(sorted((a, b)))
                    if a != b and a in sides and b in sides and key not in seen_pairs:
                        seen_pairs.add(key)
                        edges.append(dict(edge))
                if not edges:
                    continue
                filtered["edges"] = edges
                connected = {sid for pair in seen_pairs for sid in pair}
                filtered["sides"] = [side for sid, side in sides.items() if sid in connected]
            kept_ids = {side["source_id"] for side in filtered["sides"]}
            if len(kept_ids) != len(cluster.get("sides") or ()):
                filtered["adjudication_rationale"] = ""
            if cluster.get("recommended_source_id") not in kept_ids:
                filtered["recommended_source_id"] = None
                filtered["adjudication_status"] = "unresolved_divergence"
                filtered["adjudication_rationale"] = ""
            conflicts.append(filtered)
        return conflicts

    @staticmethod
    def _collect_conflict_source_ids(conflicts: list[dict[str, Any]]) -> list[str]:
        return list(dict.fromkeys(side["source_id"] for cluster in conflicts for side in cluster["sides"]))

    @staticmethod
    def _escape_table_cell(value: Any) -> str:
        text = escape(" ".join(str(value or "").split()), quote=False)
        return (
            text.replace("\\", "&#92;").replace("|", "\\|").replace("`", "&#96;")
            .replace("[", "&#91;").replace("]", "&#93;").replace("*", "&#42;").replace("_", "&#95;")
        )

    @staticmethod
    def _conflict_rationale(cluster: dict[str, Any]) -> str:
        sid = cluster.get("recommended_source_id")
        status = cluster.get("adjudication_status")
        if status == "resolved_by_authority" and sid:
            return f"依据来源权威性，优先参考 [{sid}]。"
        if status == "resolved_by_recency" and sid:
            return f"依据发布时间，优先参考 [{sid}]。"
        return "这些来源存在口径差异，需结合各自的统计范围与前提阅读。"

    def _render_source_conflicts_body(self, source_conflicts: list[dict[str, Any]]) -> str:
        if not source_conflicts:
            return ""
        lines = [
            "| 分歧议题（主体 / 指标 / 周期） | 来源口径 A | 来源口径 B | 口径采信与差异说明 |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for cluster in source_conflicts:
            sides = {side["source_id"]: side for side in cluster.get("sides") or []}
            if len(sides) < 2:
                continue
            topic_parts = [
                str(cluster.get(k) or "").strip()
                for k in ("entity", "metric", "period")
                if str(cluster.get(k) or "").strip()
            ]
            topic_cell = self._escape_table_cell(" / ".join(topic_parts) or "多源事实")

            def _fmt_side(side: dict[str, Any]) -> str:
                sid = side["source_id"]
                title = truncate_text(strip_citation_markers(str(side.get("title") or sid)), 48)
                val = side.get("normalized_value")
                unit = str(side.get("unit") or "").strip()
                val_txt = f"{val:g}{unit}" if isinstance(val, (int, float)) else ""
                excerpt = truncate_text(
                    strip_citation_markers(str(side.get("claim_text") or side.get("evidence_excerpt") or "")),
                    80,
                )
                detail = f"：{excerpt}" if excerpt else f"（{val_txt}）" if val_txt else ""
                return f"《{self._escape_table_cell(title)}》[{sid}]{self._escape_table_cell(detail)}"

            pairs = [(edge["source_id_a"], edge["source_id_b"]) for edge in cluster.get("edges") or []]
            if not pairs:
                first, *others = sides
                pairs = [(first, other) for other in others]
            rationale = self._conflict_rationale(cluster)
            for a, b in pairs:
                lines.append(f"| {topic_cell} | {_fmt_side(sides[a])} | {_fmt_side(sides[b])} | {rationale} |")
        return "\n".join(lines) if len(lines) > 2 else ""

    def _render_structured_report_fixed_layout(self, content: dict[str, Any]) -> str:
        dimension_body = self._render_dimension_body(content["dimension_sections"])
        conflicts_body = self._render_source_conflicts_body(content.get("source_conflicts") or [])
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
                f"## {self._CORE_SECTION_TITLES['source_conflicts']}\n"
                f"{conflicts_body}\n\n"
                if conflicts_body
                else ""
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
        return markdown.strip()

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

        conflicts_body = self._render_source_conflicts_body(content.get("source_conflicts") or [])
        if conflicts_body:
            blocks.insert(
                ordered_section_ids.index("risks_and_challenges") + 1,
                f"## {self._CORE_SECTION_TITLES['source_conflicts']}\n{conflicts_body}"
            )

        process_notes_body = self._render_process_notes_body(content["process_notes"])
        if process_notes_body:
            blocks.append(process_notes_body)
        blocks.append(f"## 参考来源\n{render_references(content['references'])}")
        return "\n\n".join(blocks).strip()

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

    def build_workbench_payload(
        self,
        state: SummaryState,
        report_markdown: str,
    ) -> dict[str, Any]:
        """Build interactive workbench metadata binding paragraphs, citations, conflicts, and skills."""
        return build_report_workbench_payload(
            state,
            report_markdown,
            evidence_store=self._evidence_store,
            config=self._config,
        )


def build_report_workbench_payload(
    state: Any,
    report_markdown: str,
    *,
    evidence_store: Any | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """Build structured workbench metadata for citation drawer, conflict card, and governance bar."""

    citation_index: dict[str, dict[str, Any]] = {}
    claims_by_source: dict[str, list[dict[str, Any]]] = {}
    rejected_ids = ReportingService._rejected_computation_ids(state)
    web_enabled = getattr(evidence_store, "web_extraction_enabled", False)

    def _ingest_record(rec: dict[str, Any], default_task_id: int | None = None) -> None:
        sid = str(rec.get("source_id") or "").strip()
        if not CITATION_PATTERN.fullmatch(f"[{sid}]") or sid in rejected_ids:
            return
        existing = citation_index.get(sid) or {}
        anchors: list[str] = []
        web_meta = rec.get("provenance_meta", {}).get("web_extraction", {}) if web_enabled else {}
        anchors.extend(
            anchor["anchor_id"] for anchor in web_meta.get("text_anchors", []) if anchor.get("anchor_id")
        )
        modality = str(rec.get("modality") or existing.get("modality") or "text")
        if rec.get("computation_artifact") or "-C" in sid:
            modality = "computed"
            anchors.append("compute-1")
        for idx, block in enumerate(rec.get("multimodal_blocks") or [], start=1):
            if isinstance(block, dict):
                modality = str(block.get("modality") or "multimodal")
                anchor = str(block.get("anchor_id") or f"mm-{idx}")
                if anchor not in anchors:
                    anchors.append(anchor)
        snippet_text = truncate_text(
            str(
                rec.get("snippet")
                or rec.get("full_content")
                or rec.get("raw_content")
                or existing.get("snippet")
                or ""
            ),
            400,
        )
        citation_index[sid] = {
            **existing,
            "source_id": sid,
            "task_id": rec.get("task_id", default_task_id if default_task_id is not None else existing.get("task_id")),
            "title": str(rec.get("title") or existing.get("title") or sid),
            "url": _safe_workbench_url(rec.get("url") or existing.get("url")),
            "domain": str(rec.get("domain") or existing.get("domain") or ""),
            "snippet": snippet_text,
            "excerpt": snippet_text,
            "quality_label": str(rec.get("quality_label") or existing.get("quality_label") or "medium"),
            "source_type": str(rec.get("source_type") or existing.get("source_type") or "web_general"),
            "published_at": rec.get("published_at") or existing.get("published_at"),
            "modality": modality,
            "anchors": list(dict.fromkeys([*(existing.get("anchors") or []), *anchors])),
            "source_origin": str(rec.get("source_origin") or existing.get("source_origin") or "web"),
            **({"page_no": rec["page_no"]} if rec.get("page_no") is not None else {}),
            **({"bbox": rec["bbox"]} if rec.get("bbox") is not None else {}),
            **({"anchor_id": rec["anchor_id"]} if rec.get("anchor_id") else {}),
            **({"freshness_label": rec["freshness_label"]} if rec.get("freshness_label") else {}),
            **({"computation_artifact": rec["computation_artifact"]}
               if isinstance(rec.get("computation_artifact"), dict) else {}),
            **({"derived_from_source_ids": list(rec["derived_from_source_ids"])}
               if isinstance(rec.get("derived_from_source_ids"), list) else {}),
            **({"provenance_meta": {"web_extraction": web_meta}} if web_meta else {}),
        }

    for task in getattr(state, "todo_items", []) or []:
        tid = getattr(task, "id", None)
        for item in [] if web_enabled else getattr(task, "evidence_items", None) or []:
            if isinstance(item, dict):
                _ingest_record(item, default_task_id=tid)
        for claim in getattr(task, "atomic_claims", None) or []:
            if not isinstance(claim, dict) or not claim.get("text") or not claim.get("nli_label"):
                continue
            linked_claim = {
                key: claim[key] for key in ("text", "nli_label", "calibrated_confidence") if key in claim
            }
            for sid in claim.get("source_ids") or []:
                if not isinstance(sid, str) or (web_enabled and not evidence_store.is_eligible_for_facts(sid)):
                    continue
                claims = claims_by_source.setdefault(sid, [])
                if linked_claim not in claims:
                    claims.append(linked_claim)

    if evidence_store is not None and hasattr(evidence_store, "lookup"):
        for item in evidence_store.lookup(include_full_content=False) or []:
            if isinstance(item, dict):
                _ingest_record(item)

    fact_source_ids = {
        sid for sid in citation_index if not web_enabled or evidence_store.is_eligible_for_facts(sid)
    }
    conflicts = ReportingService._filtered_conflicts(state, fact_source_ids)
    for cluster in conflicts:
        for side in cluster["sides"]:
            side["url"] = _safe_workbench_url(side.get("url"))
    conflicts_by_source: dict[str, list[str]] = {}
    # Older snapshots can still link surviving citations to a conflict ID when
    # an evidence record is no longer available to render its side of the graph.
    for cluster in getattr(state, "evidence_conflicts", None) or []:
        if not isinstance(cluster, dict):
            continue
        cid = str(cluster.get("conflict_id") or "").strip()
        if not cid:
            continue
        for side in cluster.get("sides") or []:
            if isinstance(side, dict):
                sid = str(side.get("source_id") or "").strip()
                if sid in fact_source_ids:
                    conflicts_by_source.setdefault(sid, [])
                    if cid not in conflicts_by_source[sid]:
                        conflicts_by_source[sid].append(cid)
    for sid, entry in citation_index.items():
        entry["linked_claims"] = claims_by_source.get(sid, [])
        entry["conflict_ids"] = conflicts_by_source.get(sid, [])

    paragraph_bindings: list[dict[str, Any]] = []
    visible_report = "".join(
        re.sub(r"[^\r\n]", " ", line) if fenced else line
        for line, fenced in iter_markdown_lines(str(report_markdown or ""))
    )
    visible_paragraphs = visible_report.split("\n\n")
    paragraph_index = 0
    raw_paragraphs = str(report_markdown or "").split("\n\n")
    for raw, visible in zip(raw_paragraphs, visible_paragraphs):
        if not raw.strip():
            continue
        idx = paragraph_index
        paragraph_index += 1
        para = raw.strip()
        raw_sids = CITATION_PATTERN.findall(visible)
        if not raw_sids:
            continue
        ordered_sids: list[str] = []
        seen_sids: set[str] = set()
        for sid in raw_sids:
            if sid in fact_source_ids and sid not in seen_sids:
                ordered_sids.append(sid)
                seen_sids.add(sid)
        if web_enabled and not ordered_sids:
            continue
        matched_conflicts: list[str] = []
        for sid in ordered_sids:
            for cid in conflicts_by_source.get(sid, []):
                if cid not in matched_conflicts:
                    matched_conflicts.append(cid)
        paragraph_bindings.append(
            {
                "paragraph_index": idx,
                "excerpt": truncate_text(para, 180),
                "citation_ids": ordered_sids,
                "conflict_ids": matched_conflicts,
            }
        )

    pack_meta = dict(getattr(state, "skill_pack_metadata", None) or {})
    skill_governance = {
        **pack_meta,
        "pack_id": str(pack_meta.get("pack_id") or "_neutral"),
        "version": pack_meta.get("version"),
        "maturity": pack_meta.get("maturity"),
        "active_verifiers": list(
            getattr(config, "active_skill_verifiers", None)
            or pack_meta.get("active_verifiers")
            or []
        ),
        "code_sandbox_enabled": bool(
            getattr(config, "resolved_sandbox_effective_enabled", lambda: False)()
            if config is not None
            else False
        ),
        "nli_verifier_enabled": bool(getattr(config, "nli_verifier_enabled", False) if config else False),
        "mcp_tool_scopes": dict(
            getattr(config, "active_mcp_tool_scopes", None) or {}
            if config is not None
            else {}
        ),
    }
    review = getattr(state, "review_summary", None) or {}
    verifier_badges = [
        {"badge_id": verifier_id, "status": "executed" if review.get("verifier_issue_count") else "passed"}
        for verifier_id in dict.fromkeys(review.get("verifiers_executed") or [])
        if isinstance(verifier_id, str)
    ]
    verification = dict(getattr(state, "claim_verification_summary", None) or {})
    if verification:
        status = "degraded" if verification.get("degraded") else "warning" if verification.get("contradicted_count") else "passed"
        verifier_badges.append({"badge_id": "atomic-nli", "status": status})
    charts = [
        sanitized for chart in getattr(state, "report_charts", None) or []
        if isinstance(chart, dict)
        if (sanitized := sanitize_chart_spec(chart, valid_source_ids=fact_source_ids)) is not None
    ]

    return {
        "citation_index": citation_index,
        "citation_drawer_map": citation_index,
        "paragraph_bindings": paragraph_bindings,
        "evidence_conflicts": conflicts,
        "claim_verification_summary": verification,
        "verifier_badges": verifier_badges,
        "charts_spec": charts,
        "skill_governance": skill_governance,
    }
