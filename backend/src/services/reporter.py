"""Service that consolidates task results into a grounded final report."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
from html import escape
from typing import Any
from urllib.parse import urlsplit

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace, extract_llm_usage
from models import (
    CriticIssue,
    ReportBlock,
    ReportClaim,
    ReportConflict,
    ReportCriticResult,
    ReportEvidencePack,
    ReportEvidenceRef,
    ReportExcerpt,
    ReportInput,
    ReportModelResponse,
    ReportOutline,
    ReportScope,
    ReportSectionPlan,
    SectionDraft,
    SummaryState,
)
from prompts import (
    REPORT_CRITIC_ROLE,
    REPORT_CRITIC_SCHEMA,
    REPORT_LEGACY_ROLE,
    REPORT_LEGACY_SCHEMA,
    REPORT_MAP_ROLE,
    REPORT_MAP_SCHEMA,
    REPORT_OUTLINE_ROLE,
    REPORT_OUTLINE_SCHEMA,
    REPORT_REDUCER_ROLE,
    REPORT_REDUCER_SCHEMA,
    REPORT_REVISION_ROLE,
    REPORT_REVISION_SCHEMA,
    web_evidence_prompt_rules,
)
from services.claim_verifier import (
    normalize_conflict_key,
    numeric_grounding_mismatch,
    report_atomic_sources,
    web_fact_excerpts,
)
from services.context_builder import (
    ContextPrefixBuilder,
    PromptEnvelope,
    ReportModelRequest,
    parse_report_json,
    render_web_evidence_data,
    report_json,
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
    FactLedger,
    context_covered_task_ids,
    render_context_checkpoint,
    render_context_delta,
)
from services.memory_sanitizer import sanitize_injection_text
from services.red_team_critic import AdversarialRedTeamCritic
from services.reviewer import ReviewService
from services.sandbox_executor import sanitize_chart_spec
from services.skill_verifiers import verify_computation_lineage
from services.text_processing import (
    CITATION_PATTERN,
    iter_markdown_lines,
    normalize_agent_markdown,
    strip_citation_markers,
    truncate_cited_text,
)
from state_manager import BudgetDispatchDenied, ReportCancelled
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
        report_transport: Callable[[ReportModelRequest], ReportModelResponse] | None = None,
        critic_transport: Callable[[ReportModelRequest], ReportModelResponse] | None = None,
        report_verifier: Any = None,
    ) -> None:
        self._agent = report_agent
        self._config = config
        self._evidence_store = evidence_store
        self._retrieval_service = retrieval_service
        self._report_transport = report_transport
        self._critic_transport = critic_transport
        self._report_verifier = report_verifier

    def generate_report(
        self,
        state: SummaryState,
        observer: RequestTrace | None = None,
        skill_pack_context: str | None = None,
    ) -> str:
        """Generate a structured report based on completed tasks."""
        settings = self._report_settings(state)
        if settings.parallel_report_enabled or settings.red_team_enabled:
            return ReportCoordinator(self, state, settings, observer, skill_pack_context).generate()
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

    def _report_settings(self, state: SummaryState) -> Configuration:
        frozen = (state.v4 or {}).get("report")
        settings = (frozen.get("frozen_options") or frozen.get("config")) if isinstance(frozen, dict) else None
        if not isinstance(settings, dict):
            return self._config
        keys = (self._config.report_config_projection().keys() - {"version"}) | {"report_layout_mode"}
        return Configuration.model_validate({
            **self._config.model_dump(),
            **{key: settings[key] for key in keys if key in settings},
            "v4_hierarchical_report_enabled": False,
            "v4_red_team_critic_enabled": False,
        })

    def coordinate_report(self, state, observer, skill_context, outline_sections, checkpoint, transport, pricing, usage):
        coordinator = ReportCoordinator(
            self, state, self._report_settings(state), observer, skill_context, outline_sections,
            checkpoint=checkpoint, transport=transport, pricing=pricing, usage=usage,
        )
        return (yield from coordinator.run())

    def stream_generate_report(
        self,
        state: SummaryState,
        observer: RequestTrace | None = None,
        skill_pack_context: str | None = None,
        outline_sections: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield progressive outline + validated report_chunk events while populating state.structured_report."""

        settings = self._report_settings(state)
        if settings.parallel_report_enabled or settings.red_team_enabled:
            coordinator = ReportCoordinator(self, state, settings, observer, skill_pack_context, outline_sections)
            report = coordinator.generate()
            state.structured_report = report
            sections = state.report_workbench["report_sections"]
            yield {"type": "report_outline", "sections": [section["title"] for section in sections]}
            chunks = re.split(r"(?<=\n\n)(?=## )", report)
            for index, content in enumerate(chunks, 1):
                yield {"type": "report_chunk", "content": content, "index": index, "total_chunks": len(chunks)}
            for chart in state.report_charts:
                yield {"type": "report_chart", "chart": chart}
            return

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

    if getattr(state, "report_workbench_fingerprint", None) == _report_fingerprint(report_markdown or ""):
        return state.report_workbench

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


_REPORT_INPUT_LIMITS = {"outline": 2048, "map": 4096, "reduce": 3072, "critic": 4096, "revision": 2048, "legacy": 8192}
_MAX_SECTION_CLAIMS = 12
_MAX_SECTION_SOURCES = 8
_MAX_SECTION_EXCERPTS = 16
_MAX_EXCERPT_BYTES = 1024
_MAX_SECTION_EXCERPT_BYTES = 8192
_MAX_BLOCKS = 10
_TRANSITIONS = {"scope": "本节按已核验的研究范围组织。", "limitations": "以下结果受来源和测量范围限制。"}
_UNSAFE_REPORT_TEXT = re.compile(r"https?://|www\.|[<>\[\]`\\\r\n]|^(?:[#>*+-]|\d+[.)])\s|!\[", re.I)


def _report_fingerprint(value: str) -> str:
    return hashlib.sha256(value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def _report_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _display_report_text(value: str) -> str:
    # Escaping brackets as entities also prevents literal user text from matching CITATION_PATTERN.
    return re.sub(r"([\`*_~|])", r"\\\1", escape(" ".join(value.split())).replace("[", "&#91;").replace("]", "&#93;"))


def _report_statement(value: Any) -> str:
    return str(value or "").strip()


def _report_scope(data: dict) -> ReportScope:
    return ReportScope(**{
        name: _report_key(str(data.get(name) or "unknown")) for name in ReportScope.__dataclass_fields__
    })


def _report_ids(data: dict) -> set[str]:
    values = data.get("source_ids")
    return {sid for sid in values if isinstance(sid, str)} if isinstance(values, (list, tuple)) else set()


def _bounded_excerpts(record: dict[str, Any]) -> tuple[ReportExcerpt, ...]:
    meta = record.get("provenance_meta", {}).get("web_extraction", {})
    sid = record["source_id"]
    version = str(meta["artifact_version"]) if meta.get("artifact_version") else None
    body = str(record.get("full_content") or "")
    fragments = []
    if meta.get("tier") in {"L1", "L2"}:
        if not version:
            return ()
        for anchor in meta.get("text_anchors") or ():
            start, end = anchor.get("start"), anchor.get("end")
            if type(start) is int and type(end) is int and 0 <= start < end <= len(body) and anchor.get("anchor_id"):
                fragments.append((anchor["anchor_id"], body[start:end]))
        for block in record.get("multimodal_blocks") or ():
            payload = block.get("structured_payload") or {}
            anchor_id = block.get("anchor_id")
            if not anchor_id:
                continue
            if block.get("modality") == "table" and payload.get("schema_version") == 1:
                rows = web_fact_excerpts({
                    "provenance_meta": {"web_extraction": meta}, "multimodal_blocks": [block],
                }) or []
                heading = " | ".join(str(item) for item in (
                    payload.get("caption"), payload.get("unit_text"), *(payload.get("headers") or ()),
                ) if item)
                fragments.extend((anchor_id, f"{heading}\n{row}") for row in rows)
            elif block.get("modality") != "table" and (text := block.get("text")):
                fragments.append((anchor_id, text))
    else:
        text = body or record.get("raw_content") or record.get("snippet") or record.get("content") or ""
        fragments.extend(
            (record.get("anchor_id"), fragment) for fragment in re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        )
        for block in record.get("multimodal_blocks") or ():
            if block.get("anchor_id") and (text := block.get("text") or block.get("markdown")):
                fragments.append((block["anchor_id"], text))
    result = []
    for anchor, fragment in fragments:
        text = sanitize_injection_text(fragment).strip()
        if text and len(text.encode("utf-8")) <= _MAX_EXCERPT_BYTES:
            result.append(ReportExcerpt(ReportEvidenceRef(sid, anchor, version, _report_fingerprint(text)), text))
    return tuple(dict.fromkeys(result))


class OutlinePlanner:
    """Plan ordered scopes from request-local reviewed facts and registered evidence."""

    def __init__(self, reporter: ReportingService, state: SummaryState, tenant_id: str, request_id: str) -> None:
        self.reporter = reporter
        self.state = state
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.ledger = FactLedger.from_dict(state.fact_ledger)
        replan = (state.v4 or {}).get("replan") or {}
        required = replan.get("required_coverage") or {}
        self.required = {
            _report_key(key): str(label)[:120] for key, label in required.items()
            if isinstance(key, str) and _report_key(key)
        } if isinstance(required, dict) else {}
        if isinstance(required, (list, tuple)):
            self.required = {_report_key(key): key[:120] for key in required if isinstance(key, str) and key.strip()}
        raw_coverage = replan.get("coverage_by_task") or {}
        self.coverage = {
            task.id: tuple(dict.fromkeys(
                _report_key(key) for key in raw_coverage.get(str(task.id), getattr(task, "coverage_keys", ()))
                if isinstance(key, str) and _report_key(key) in self.required
            ))
            for task in state.todo_items
        }
        self.records = self.current_records()
        self.excerpts = {
            sid: _bounded_excerpts(record) for sid, record in self.records.items()
        }
        self.rejected = {_report_key(text) for text in reporter._rejected_claim_texts(state.todo_items)}
        self.entry_data = {
            entry["claim_id"]: entry for entry in (state.fact_ledger or {}).get("entries", ())
            if isinstance(entry, dict) and isinstance(entry.get("claim_id"), str)
        }

    def owns_record(self, record: dict) -> bool:
        explicit_tenant = record.get("tenant_id")
        if explicit_tenant:
            return explicit_tenant == self.tenant_id
        # Ordinary web records inherit ownership from their request-scoped EvidenceStore.
        return record.get("source_origin") == "web" or self.tenant_id == "default"

    def current_records(self) -> dict[str, dict]:
        store = self.reporter._evidence_store
        if store is None:
            return {}
        valid = self.reporter._collect_valid_source_ids(self.state)
        records = {
            sid: record for record in store.lookup(source_ids=sorted(valid), include_full_content=True)
            if (sid := record.get("source_id")) in valid
            and CITATION_PATTERN.fullmatch(f"[{sid}]")
            and self.owns_record(record)
            and store.is_eligible_for_facts(sid)
            and record.get("provenance_meta", {}).get("web_extraction", {}).get("tier") != "L0"
        }
        # Removing an invalid dependency can invalidate a computation farther downstream.
        while rejected := {
            sid for sid, record in records.items()
            if (record.get("source_origin") == "sandbox" or "-C" in sid)
            and verify_computation_lineage(record, records)
        }:
            for sid in rejected:
                del records[sid]
        return records

    def _refs(self, ids: set[str], text: str, proof: dict | None = None) -> tuple[ReportEvidenceRef, ...]:
        refs = []
        proof = proof or {}
        for sid in sorted(ids.intersection(self.excerpts)):
            allowed = self.excerpts[sid]
            bindings = proof.get("evidence_refs")
            binding = None
            if isinstance(bindings, list) and bindings:
                binding = next((item for item in bindings if isinstance(item, dict) and item.get("source_id") == sid), None)
                if binding is None:
                    continue
            binding = binding or proof
            matched_excerpt = _report_statement(proof.get("matched_excerpt"))
            permitted = [
                item for item in allowed if (
                    (not binding.get("anchor_id") or item.ref.anchor_id == binding["anchor_id"])
                    and (not binding.get("artifact_version") or item.ref.artifact_version == str(binding["artifact_version"]))
                    and (not binding.get("excerpt_fingerprint")
                         or item.ref.excerpt_fingerprint == binding["excerpt_fingerprint"])
                )
            ]
            if permitted:
                # For verified artifacts the verification must bind to a current excerpt.
                exact = next((
                    item for item in permitted
                    if ReviewService.verify_report_claim(text, (item.text,)) == "rule_passed"
                ), None)
                matched = next((
                    item for item in permitted if proof.get("nli_label") == "ENTAILED" and matched_excerpt
                    and _report_key(matched_excerpt) in _report_key(item.text)
                    and not numeric_grounding_mismatch(text, [item.text])
                ), None)
                if chosen := exact or matched:
                    refs.append(chosen.ref)
        return tuple(refs)

    def _claims(self) -> tuple[dict[str, ReportClaim], dict[str, ReportClaim]]:
        by_text = defaultdict(list)
        atomics = []
        for task in self.state.todo_items:
            if task.status != "completed" or task.review_status == "blocked":
                continue
            for item in task.claims or ():
                if isinstance(item, dict) and item.get("support_status") == "supported" and task.review_status in {"passed", "warning"}:
                    by_text[(task.id, _report_key(_report_statement(item.get("text"))))].append(item)
            for item in task.atomic_claims or ():
                if isinstance(item, dict) and report_atomic_sources(item, self.records.keys()):
                    atomics.append((task.id, item))

        claims = {}
        candidates = {}
        bound_atomics = set()
        for entry in sorted(self.ledger.entries, key=lambda item: item.claim_id):
            data = self.entry_data.get(entry.claim_id, {})
            text = entry.statement
            if not entry.claim_id or not text or entry.stance == "gap" or not self._safe_fact(text):
                continue
            task_ids = tuple(dict.fromkeys((entry.task_id, *entry.supporting_task_ids)))
            ids = set(entry.source_ids)
            matches = [
                (task_id, atomic) for task_id, atomic in atomics
                if task_id in task_ids and _report_fingerprint(_report_statement(atomic.get("text"))) == _report_fingerprint(text)
                and ids.intersection(report_atomic_sources(atomic, self.records.keys()))
                and self._matching_proposition(entry, atomic)
            ]
            proof = matches[0][1] if len(matches) == 1 else None
            allowed = set(report_atomic_sources(proof, self.records.keys())) & ids if proof else set()
            refs = self._refs(allowed, text, proof) if proof else ()
            if refs:
                bound_atomics.add((matches[0][0], proof.get("atomic_id")))
            else:
                proof = None
                for task_id in task_ids:
                    for reviewed in by_text[task_id, _report_key(text)]:
                        if _report_statement(reviewed.get("text")) == text:
                            refs += self._refs(_report_ids(reviewed) & ids, text, reviewed)
            claim = ReportClaim(
                entry.claim_id, text, tuple(dict.fromkeys(refs)), task_ids, _report_scope({
                    **(proof.get("proposition") or {} if proof else {}), **data,
                }), str(proof["atomic_id"]) if proof and proof.get("atomic_id") else None,
                entry.value, entry.period,
            )
            if refs and entry.stance == "asserted":
                claims[entry.claim_id] = claim
            if entry.conflict_group_id or entry.stance == "disputed" or any(
                entry.claim_id in members for members in self.ledger.conflict_groups.values()
            ):
                excerpts = {item.ref: item.text for sid in ids for item in self.excerpts.get(sid, ())}
                side_refs = tuple(ref for ref, body in excerpts.items()
                                  if ReviewService.verify_report_claim(text, (body,)) == "rule_passed")
                candidates[entry.claim_id] = replace(claim, support_refs=side_refs)
        disputed = {
            (entry.statement, sid)
            for entry in self.ledger.entries if entry.claim_id in candidates
            for sid in entry.source_ids
        }
        for task_id, item in atomics:
            text = _report_statement(item.get("text"))
            atomic_id = _report_statement(item.get("atomic_id"))
            if not atomic_id or (task_id, atomic_id) in bound_atomics or not self._safe_fact(text):
                continue
            refs = self._refs({
                sid for sid in report_atomic_sources(item, self.records.keys()) if (text, sid) not in disputed
            }, text, item)
            if refs:
                claim_id = atomic_id if atomic_id not in claims else "atomic-" + _report_fingerprint(
                    report_json([task_id, atomic_id, text])
                )[:24]
                proposition = item.get("proposition") or {}
                claims[claim_id] = ReportClaim(
                    claim_id, text, refs, (task_id,), _report_scope(proposition), atomic_id,
                    str(proposition.get("numeric_value") or ""), str(proposition.get("period") or ""),
                )
        # Legacy Reviewer outputs can predate both ledgers. Exact source matches remain usable.
        prior = {(claim.text, sid) for claim in claims.values() for sid in (ref.source_id for ref in claim.support_refs)}
        for (task_id, _), rows in by_text.items():
            for row in rows:
                text = _report_statement(row.get("text"))
                if not self._safe_fact(text) or any((text, sid) in prior | disputed for sid in _report_ids(row)):
                    continue
                refs = self._refs(_report_ids(row), text, row)
                if refs:
                    claim_id = "review-" + _report_fingerprint(report_json([task_id, text, sorted(_report_ids(row))]))[:24]
                    claims[claim_id] = ReportClaim(claim_id, text, refs, (task_id,), _report_scope(row))
        return claims, candidates

    def _safe_fact(self, text: str) -> bool:
        normalized = _report_key(text).rstrip("。.!?！？")
        return bool(
            normalized and normalized not in self.rejected and len(text) <= 700
            and not _UNSAFE_REPORT_TEXT.search(text)
        )

    def _matching_proposition(self, entry, atomic: dict) -> bool:
        if not (proposition := atomic.get("proposition")):
            return False
        return normalize_conflict_key(entry.entity, entry.metric, entry.period) == normalize_conflict_key(
            str(proposition.get("entity") or ""), str(proposition.get("metric") or ""), str(proposition.get("period") or ""),
        )

    def _conflicts(self, claims: dict[str, ReportClaim], candidates: dict[str, ReportClaim]) -> tuple[ReportConflict, ...]:
        groups = []
        ledger_groups = {key: list(ids) for key, ids in self.ledger.conflict_groups.items()}
        for entry in self.ledger.entries:
            if entry.conflict_group_id:
                ledger_groups.setdefault(entry.conflict_group_id, []).append(entry.claim_id)
        for cid, ids in sorted(ledger_groups.items()):
            members = tuple(dict.fromkeys(ids))
            groups.append(ReportConflict(cid, members, tuple(candidates[key] for key in members if key in candidates), "unresolved"))

        for cluster in self.state.evidence_conflicts or ():
            if not isinstance(cluster, dict) or not isinstance(cluster.get("conflict_id"), str):
                continue
            cid = cluster["conflict_id"]
            matched_ids = set()
            sides = []
            for index, side in enumerate(cluster.get("sides") or ()):
                sid = side.get("source_id")
                text = _report_statement(side.get("claim_text"))
                same_claim = [
                    claim for claim in (claims | candidates).values()
                    if claim.text == text and any(ref.source_id == sid for ref in claim.support_refs)
                ]
                matched_ids.update(claim.claim_id for claim in same_claim)
                text_ref = next((
                    excerpt.ref for excerpt in self.excerpts.get(sid, ())
                    if self._safe_fact(text) and ReviewService.verify_report_claim(text, (excerpt.text,)) == "rule_passed"
                ), None)
                sides.append(ReportClaim(
                    same_claim[0].claim_id if len(same_claim) == 1 else f"{cid}-{index}",
                    text, (text_ref,) if text_ref else (), (side.get("task_id") or 0,),
                    _report_scope(cluster | side), observed_period=str(cluster.get("period") or ""),
                ))
            entry = ReportConflict(cid, tuple(sorted(matched_ids)), tuple(sides), str(cluster.get("adjudication_status") or "unresolved"))
            equivalent = next((
                group for group in groups if group.sides and self._side_identity(group) == self._side_identity(entry)
            ), None)
            if equivalent:
                groups.remove(equivalent)
                entry = replace(entry, aliases=(equivalent.conflict_id,))
            groups.append(entry)
        return tuple(groups)

    @staticmethod
    def _side_identity(group: ReportConflict) -> frozenset:
        return frozenset((side.text, side.observed_period, side.scope, side.support_refs) for side in group.sides)

    def build(self, settings: Configuration, format_titles: tuple[str, ...] = ()) -> ReportInput:
        claims, candidates = self._claims()
        conflicts = (*self._conflicts(claims, candidates), *CoherenceReducer.scope_conflicts(tuple(claims.values())))
        disputed_ids = {cid for group in conflicts for cid in group.claim_ids}
        # Known disputes are attributed only, even when a Reviewer passed either side earlier.
        active_claims = {cid: claim for cid, claim in claims.items() if cid not in disputed_ids}
        by_task = {task.id: task for task in self.state.todo_items}
        task_keys = {
            task_id: keys or (f"task-{task_id}",) for task_id, keys in self.coverage.items()
        }
        required_by_slot = [key for key in self.required if key != "research_topic"]
        ordered_keys = list(dict.fromkeys((*required_by_slot, *(key for keys in task_keys.values() for key in keys))))
        if len(ordered_keys) > 1 and "research_topic" in ordered_keys:
            ordered_keys.remove("research_topic")
        if not ordered_keys:
            ordered_keys = ["research_topic"]
        labels = {
            **{f"task-{tid}": task.title for tid, task in by_task.items()},
            **self.required,
            "research_topic": self.required.get("research_topic") or str(self.state.research_topic or "研究主题"),
        }
        fixed = settings.resolved_report_layout_mode() == "fixed"
        max_body = 4 if conflicts else 5
        body_limit = min(max_body, len(ordered_keys))
        if fixed and not format_titles:
            format_titles = tuple(self.reporter._CORE_SECTION_TITLES[key] for key in self.reporter._CONTENT_SECTION_DEFAULT_ORDER)
        count = min(max_body, max(body_limit, len(format_titles)))
        scopes = [
            tuple(ordered_keys[index * len(ordered_keys) // count:(index + 1) * len(ordered_keys) // count])
            for index in range(count)
        ]
        if format_titles:
            # Match explicit Format titles before assigning remaining keys in data order.
            matched = {key: index for key in ordered_keys for index, title in enumerate(format_titles[:count])
                       if _report_key(title) in {_report_key(labels.get(key, key)), key}}
            scopes = [
                tuple(dict.fromkeys((*[key for key in ordered_keys if matched.get(key) == index],
                                     *(key for key in keys if key not in matched))))
                for index, keys in enumerate(scopes)
            ]
        unique_slots = set()
        sections = []
        packs = []
        remaining = dict(active_claims)
        retained = set()
        all_excerpts = {item.ref: item for rows in self.excerpts.values() for item in rows}

        def add(title, kind, slot, keys=(), scoped=(), groups=(), refs=()):
            digest = _report_fingerprint(report_json([self.request_id, 1, slot]))
            length = 16
            while digest[:length] in unique_slots:
                length += 1
            unique_slots.add(digest[:length])
            section = ReportSectionPlan(
                "sec-" + digest[:length], len(sections), title[:120],
                " / ".join(labels.get(key, key) for key in keys) or title, keys,
                tuple(item.claim_id for item in scoped), tuple(sorted({ref.source_id for ref in refs})),
                tuple(group.conflict_id for group in groups), kind, slot,
            )
            sections.append(section)
            return section

        owners = {}
        claim_coverage = {}
        for claim in active_claims.values():
            options = []
            for index, keys in enumerate(scopes):
                for key in keys:
                    matches = [pos for pos, tid in enumerate(claim.task_ids) if key in task_keys.get(tid, ())]
                    if matches:
                        options.append((key not in required_by_slot, matches[0], -len(key), index, key))
            if options:
                owner = min(options)[3]
                owners[claim.claim_id] = owner
                claim_coverage[claim.claim_id] = tuple(dict.fromkeys(
                    option[4] for option in sorted(options) if option[3] == owner
                ))

        add("摘要", "summary", "summary")
        for index, keys in enumerate(scopes):
            title = format_titles[index] if index < len(format_titles) else labels.get(keys[0], keys[0]) if keys else "研究发现"
            scoped = []
            refs = set()
            groups = []
            for group in conflicts:
                relevant = any(
                    set(keys).intersection(task_keys.get(tid, ())) for side in group.sides for tid in side.task_ids
                )
                group_refs = {ref for side in group.sides for ref in side.support_refs}
                if relevant and self._complete_conflict(group) and self._within_pack(refs | group_refs, all_excerpts):
                    refs.update(group_refs)
                    groups.append(group)
            for claim_id in sorted(remaining, key=lambda key: (
                not bool(set(keys).intersection(required_by_slot)), -len(remaining[key].support_refs), key,
            )):
                if owners.get(claim_id) != index:
                    continue
                claim = remaining[claim_id]
                candidate_refs = refs | set(claim.support_refs)
                if len(scoped) < _MAX_SECTION_CLAIMS and self._within_pack(candidate_refs, all_excerpts):
                    scoped.append(claim)
                    retained.add(claim_id)
                    refs = candidate_refs
            section = add(
                title, "body", f"format-{index}" if index < len(format_titles) else f"coverage-{keys[0] if keys else index}",
                keys, scoped, groups, refs,
            )
            packs.append(ReportEvidencePack(
                section, tuple(scoped), tuple(all_excerpts[ref] for ref in sorted(refs, key=_ref_key)),
                tuple(groups), tuple((key, labels.get(key, key)) for key in keys),
            ))
        if conflicts:
            add(
                self.reporter._CORE_SECTION_TITLES["source_conflicts"], "conflicts", "conflicts", groups=conflicts,
                refs=tuple(ref for group in conflicts if self._complete_conflict(group) for side in group.sides for ref in side.support_refs),
            )
        add("结论与局限", "conclusion", "conclusion")
        add(self.reporter._CORE_SECTION_TITLES["references"], "references", "references")
        return ReportInput(
            ReportOutline(1, self.ledger.version, self.state.frozen_prefix_version, tuple(sections)),
            tuple(packs), tuple(all_excerpts.values()), conflicts, tuple(self.required.items()), frozenset(self.records),
            tuple(sorted(active_claims.keys() - retained)), tuple(sorted(claim_coverage.items())),
        )

    @staticmethod
    def _complete_conflict(group: ReportConflict) -> bool:
        return len(group.sides) >= 2 and len({ref.source_id for side in group.sides for ref in side.support_refs}) >= 2 and all(
            side.support_refs for side in group.sides
        ) and (not group.claim_ids or set(group.claim_ids) <= {side.claim_id for side in group.sides})

    @staticmethod
    def _within_pack(refs: set[ReportEvidenceRef], excerpts: dict[ReportEvidenceRef, ReportExcerpt]) -> bool:
        return (
            len(refs) <= _MAX_SECTION_EXCERPTS
            and len({ref.source_id for ref in refs}) <= _MAX_SECTION_SOURCES
            and sum(len(excerpts[ref].text.encode("utf-8")) for ref in refs) <= _MAX_SECTION_EXCERPT_BYTES
        )


def _ref_key(ref: ReportEvidenceRef) -> tuple[str, str, str]:
    return ref.source_id, ref.anchor_id or "", ref.excerpt_fingerprint


def _report_block(section: ReportSectionPlan, kind, text: str, **fields) -> ReportBlock:
    identity = report_json([section.section_id, kind, fields.get("claim_id"), fields.get("conflict_id"),
                            fields.get("coverage_key"), fields.get("gap_code"), text])
    return ReportBlock("block-" + _report_fingerprint(identity)[:24], kind, text, **fields)


def _render_block(block: ReportBlock) -> str:
    citations = " ".join(f"[{sid}]" for sid in dict.fromkeys(ref.source_id for ref in block.evidence_refs))
    return f"- {_display_report_text(block.text)}" + (f" {citations}" if citations else "")


def _section_draft(section: ReportSectionPlan, blocks) -> SectionDraft:
    items = tuple(dict.fromkeys(blocks))
    return SectionDraft(
        section.section_id, "\n\n".join(_render_block(block) for block in items),
        tuple(dict.fromkeys(block.claim_id for block in items if block.claim_id)),
        tuple(dict.fromkeys(ref.source_id for block in items for ref in block.evidence_refs)),
        tuple(dict.fromkeys(block.gap_code for block in items if block.gap_code)), items,
    )


def _fact_block(section: ReportSectionPlan, claim: ReportClaim, refs=None) -> ReportBlock:
    return _report_block(section, "fact", claim.text, claim_id=claim.claim_id,
                         evidence_refs=claim.support_refs if refs is None else refs)


def _gap_block(section: ReportSectionPlan, label: str, key=None, code="missing_scope") -> ReportBlock:
    return _report_block(section, "gap", f"{label}：现有证据不足以给出可核验结论。", coverage_key=key, gap_code=code)


def _conflict_blocks(section: ReportSectionPlan, group: ReportConflict) -> tuple[ReportBlock, ...]:
    if not OutlinePlanner._complete_conflict(group):
        return (_report_block(
            section, "gap", "部分分歧来源缺少可核验的正文，暂不能确定结论。",
            conflict_id=group.conflict_id, gap_code="incomplete_conflict",
        ),)
    context = "口径或观测期不同，数值不可直接比较。" if group.status == "incomparable" else {
        "resolved_by_authority": "已有权威来源裁决，仍保留两方原始口径。",
        "resolved_by_recency": "已有时效性裁决，仍保留两方原始口径。",
    }.get(group.status, "存在来源分歧，分别列示原始陈述。")
    return (_report_block(section, "transition", context, conflict_id=group.conflict_id), *(
        _report_block(section, "conflict", f"来源报告：{side.text}", conflict_id=group.conflict_id,
                      claim_id=side.claim_id, evidence_refs=side.support_refs)
        for side in group.sides
    ))


class CoherenceReducer:
    """Reconcile evidence-bound drafts and render the final citation projection."""

    @staticmethod
    def scope_conflicts(claims: tuple[ReportClaim, ...]) -> tuple[ReportConflict, ...]:
        by_metric = defaultdict(list)
        for claim in claims:
            identity = normalize_conflict_key(claim.scope.entity, claim.scope.metric, "")
            if "unknown" not in identity[:2] and claim.value:
                by_metric[identity[:2]].append(claim)
        result = []
        for rows in by_metric.values():
            if len(rows) < 2:
                continue
            signatures = {(row.scope, row.observed_period) for row in rows}
            if len(signatures) > 1:
                status = "incomparable"
            else:
                values = {_report_quantity(row) for row in rows}
                if len(values) < 2:
                    continue
                scope, observed = next(iter(signatures))
                complete = all(value != "unknown" for value in asdict(scope).values()) or (
                    bool(observed and observed not in {"unknown", "unspecified"})
                    and all(getattr(scope, key) != "unknown" for key in (
                        "period_basis", "unit", "denominator", "population", "methodology",
                    ))
                    and (scope.currency != "unknown" or scope.unit in {"%", "percent", "percentage point", "count"})
                )
                status = "unresolved" if complete else "incomparable"
            ids = tuple(sorted(row.claim_id for row in rows))
            result.append(ReportConflict("scope-" + _report_fingerprint(report_json(ids))[:20], ids, tuple(rows), status))
        return tuple(result)

    @staticmethod
    def fallback(pack: ReportEvidencePack) -> SectionDraft:
        blocks = [_fact_block(pack.section, claim) for claim in pack.claims[:_MAX_BLOCKS]]
        for group in pack.conflicts:
            rows = _conflict_blocks(pack.section, group)
            if len(blocks) + len(rows) <= _MAX_BLOCKS:
                blocks.extend(rows)
        while blocks and (sum(len(item.text) for item in blocks) > 2100 or
                          sum(len(_render_block(item).encode("utf-8")) + 2 for item in blocks) > 3700):
            cid = blocks[-1].conflict_id
            blocks = [item for item in blocks if item.conflict_id != cid] if cid else blocks[:-1]
        return _section_draft(pack.section, blocks)

    def merge(
        self, frozen: ReportInput, drafts: dict[str, SectionDraft], live_refs: set[ReportEvidenceRef],
        *, withheld: frozenset[str] = frozenset(), summary_order: tuple[str, ...] = (),
        replacements: dict[str, str] | None = None, include_local_facts: bool = False,
    ) -> tuple[SectionDraft, ...]:
        replacements = replacements or {}
        published = {}
        identities = {}
        covered = set()
        coverage_by_claim = dict(frozen.coverage_by_claim)
        merged = {}
        packs = {pack.section.section_id: pack for pack in frozen.packs}
        for section in frozen.outline.sections:
            if section.kind != "body":
                continue
            pack = packs[section.section_id]
            claims = {claim.claim_id: claim for claim in pack.claims}
            draft = drafts.get(section.section_id) or self.fallback(pack)
            blocks = []
            # The global conflict section renders complete groups atomically.
            inputs = list(draft.blocks)
            existing = {block.claim_id for block in inputs if block.kind == "fact"}
            if include_local_facts:
                inputs.extend(_fact_block(section, claim) for claim in pack.claims if claim.claim_id not in existing)
            for block in dict.fromkeys(inputs):
                if block.kind in {"transition", "gap"} and not block.conflict_id:
                    blocks.append(block)
                    continue
                claim = claims.get(block.claim_id)
                if block.kind != "fact" or not claim:
                    continue
                if claim.claim_id in withheld and claim.claim_id not in replacements:
                    blocks.append(_gap_block(section, "本节的一项陈述需要补充核验", claim.claim_id, "critic_withheld"))
                    continue
                refs = tuple(ref for ref in block.evidence_refs if (
                    ref in live_refs and ref in claim.support_refs and ref.source_id in section.allowed_source_ids
                ))
                if block.text != claim.text or refs != block.evidence_refs or not refs:
                    blocks.append(_gap_block(section, "本节的一项证据已失效", claim.claim_id, "stale_anchor"))
                    continue
                text = replacements.get(claim.claim_id, claim.text)
                key = (_report_key(text), claim.scope, claim.observed_period)
                if key in identities:
                    owner = identities[key]
                    if owner.section_id != section.section_id:
                        blocks.append(_report_block(section, "transition", f"同一事实参见「{owner.title}」。"))
                    covered.update(coverage_by_claim.get(claim.claim_id, ()))
                    continue
                issued = _fact_block(section, replace(claim, text=text), refs)
                candidate = (*blocks, issued)
                if len(candidate) > _MAX_BLOCKS or sum(len(item.text) for item in candidate) > 2200 or (
                    sum(len(_render_block(item).encode("utf-8")) + 2 for item in candidate) > 3700
                ):
                    continue
                identities[key] = section
                published[claim.claim_id] = (claim, issued)
                blocks.append(issued)
                covered.update(coverage_by_claim.get(claim.claim_id, ()))
            if not blocks and not pack.required_coverage:
                blocks.append(_gap_block(section, section.scope))
            merged[section.section_id] = _section_draft(section, blocks)

        # Shared conflicts survive as attributed statements, including scope-incompatible quantities.
        conflicted = [
            replace(group, sides=tuple(
                replace(side, support_refs=tuple(ref for ref in side.support_refs if ref in live_refs))
                for side in group.sides
            ))
            for group in frozen.conflicts
        ]
        for section in frozen.outline.sections:
            if section.kind == "conflicts":
                merged[section.section_id] = _section_draft(section, (
                    block for group in conflicted for block in _conflict_blocks(section, group)
                ))
            if section.kind in {"summary", "conclusion"}:
                ranked = list(dict.fromkeys((*summary_order, *published))) if section.kind == "summary" else list(published)
                blocks = [
                    _fact_block(section, replace(published[cid][0], text=published[cid][1].text), published[cid][1].evidence_refs)
                    for cid in ranked if cid in published
                ][:4]
                if conflicted:
                    blocks.append(_report_block(section, "transition", "结论受分歧或数据口径限制，详见分歧与证据对照。"))
                if section.kind == "conclusion":
                    blocks.extend(_gap_block(section, label, key) for key, label in frozen.required_coverage if key not in covered)
                    if frozen.omitted_claim_ids:
                        blocks.append(_gap_block(section, "部分研究项超出本次报告的证据额度", code="coverage_limit"))
                if not blocks:
                    blocks.append(_gap_block(section, "研究范围"))
                merged[section.section_id] = _section_draft(section, blocks)
        return tuple(merged.get(section.section_id, _section_draft(section, ())) for section in frozen.outline.sections)


def _report_quantity(claim: ReportClaim) -> Decimal | str:
    match = re.fullmatch(r"([+\-]?\d+(?:\.\d+)?)\s*(thousand|million|billion|万|亿)?", claim.value.replace(",", ""), re.I)
    if not match:
        return claim.value
    try:
        scale = {None: 1, "thousand": 1000, "million": 1000000, "billion": 1000000000, "万": 10000, "亿": 100000000}
        return Decimal(match[1]) * scale[match[2].casefold() if match[2] else None]
    except InvalidOperation:
        return claim.value


class ReportCoordinator:
    """Run bounded report operations against the frozen section packs."""

    def __init__(
        self, reporter, state, settings, observer, skill_context, outline_sections=(), *,
        checkpoint=None, transport=None, pricing=None, usage=None,
    ):
        self.reporter = reporter
        self.state = state
        self.settings = settings
        self.observer = observer
        self.started = time.monotonic()
        frozen = (state.v4 or {}).get("report") or {}
        tenant = frozen.get("tenant_id") or getattr(observer, "tenant_id", None) or settings.default_tenant_id
        request = frozen.get("request_id") or getattr(observer, "request_id", None) or "local"
        self.planner = OutlinePlanner(reporter, state, tenant, request)
        meta = state.skill_pack_metadata or {}
        self.skill_context = "" if meta.get("shadow") else (skill_context or "")
        self.titles = () if meta.get("shadow") else tuple(outline_sections or meta.get("outline_sections") or ())
        self.checkpoint = checkpoint
        self.transport = reporter._report_transport or transport
        self.critic_transport = reporter._critic_transport or self.transport
        self.pricing = pricing
        self.usage = usage
        self.attempt = None
        self.local_only = False
        self.reducer = CoherenceReducer()
        self.rejections = defaultdict(int)
        self.terminology = tuple(sorted(
            (str(alias), str(name)) for alias, name in (meta.get("terminology") or {}).items()
            if isinstance(alias, str) and isinstance(name, str) and alias.strip() and name.strip()
        ))[:32] if not meta.get("shadow") else ()
        self.frozen = self.planner.build(settings, self.titles)

    def _prefix(self, stage, role, schema):
        return ContextPrefixBuilder(self.settings).freeze_report_prefix(
            tenant_id=self.planner.tenant_id, request_id=self.planner.request_id,
            topic=self.state.research_topic or "", ledger_version=self.frozen.outline.ledger_version,
            prefix_version=self.frozen.outline.frozen_prefix_version, format_context=self.skill_context,
            layout=self.settings.resolved_report_layout_mode(), role=role, schema=schema, stage=f"report_{stage}",
            llm=getattr(self.reporter._agent, "llm", None), terminology=self.terminology,
        )

    @staticmethod
    def _outbound(transport, request):
        response = transport(request)
        if not isinstance(response, ReportModelResponse):
            raise ValueError("report transport must return ReportModelResponse")
        return response

    def _record(self, kind, key, reason, started, *, response=None, request=None):
        if self.observer is None:
            return
        attempted = request is not None
        usage = dict(response.usage) if response else None
        if attempted:
            self.observer.record_llm_call(
                success=response is not None, prompt_text=request.messages_json,
                completion_text=response.text if response else "", usage=usage, error=None if response else reason,
                stage="report",
            )
        outcome = {
            "ok": "ok", "timeout": "timeout", "invalid_output": "invalid", "disabled": "skipped",
            "no_model": "skipped", "no_candidates": "skipped",
        }.get(reason, "fallback")
        self.observer.record_report_operation(
            key, kind=kind, outcome=outcome, reason=reason, elapsed_ms=(time.monotonic() - started) * 1000,
            usage=usage, attempted=attempted,
        )

    def _recover(self, kind, section_id=""):
        if not self.checkpoint or self.local_only:
            return None, None
        previous = self.checkpoint.previous(kind, section_id)
        if previous.get("status") == "dispatched":
            return "outcome_unknown", None
        if previous.get("status") in {"completed", "degraded"}:
            return previous["reason"], previous.get("result")
        return None, None

    def _start(self, request, section_id="", *, required=False):
        if not self.checkpoint:
            return request, None
        if self.checkpoint.agent._is_cancelled():
            raise ReportCancelled()
        keep = None
        successor = {"outline": (4096, 768), "reduce": (4096, 768)}.get(request.kind)
        if request.kind == "reduce" and not self.settings.red_team_enabled:
            successor = None
        pricing = self.pricing(request) if self.pricing else None
        if successor:
            from services.budget import estimate_llm_cost_microusd

            input_tokens, output_tokens = successor
            cost = estimate_llm_cost_microusd(input_tokens=input_tokens, output_tokens=output_tokens, pricing=pricing)
            if cost is None:
                raise BudgetDispatchDenied("pricing_unavailable")
            keep = {"llm_calls": 1, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_microusd": cost}
            if self.checkpoint.controller and self.checkpoint.controller.governor.remaining_timeout(stage="report") <= (
                request.timeout + 3 + (self.settings.v4_red_team_timeout_seconds if request.kind == "reduce" else 1)
            ):
                raise BudgetDispatchDenied("deadline_exceeded")
        timeout, settle = self.checkpoint.start(
            request.kind, section_id, request, pricing, required=required, keep_available=keep,
        )
        return replace(request, timeout=timeout), settle

    def _complete(self, kind, reason, result, *, section_id="", settle=None, response=None, update=None):
        if self.checkpoint:
            self.checkpoint.complete(
                kind, section_id, reason, result, settle=settle,
                usage=self.usage(response) if settle and response and self.usage else None, update=update,
            )

    @staticmethod
    def _restore_draft(data):
        return SectionDraft(
            data["section_id"], data["markdown"], tuple(data["claim_ids"]), tuple(data["source_ids"]),
            tuple(data["gap_codes"]), tuple(
                ReportBlock(**{**block, "evidence_refs": tuple(ReportEvidenceRef(**ref) for ref in block["evidence_refs"])})
                for block in data["blocks"]
            ),
        )

    def _progress(self):
        if self.checkpoint:
            yield from self.checkpoint.drain()

    def _wait_result(self, future, timeout):
        if not self.checkpoint:
            return future.result(timeout=timeout)
        deadline = time.monotonic() + timeout
        while True:
            if self.checkpoint.agent._is_cancelled():
                raise ReportCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                return future.result(timeout=min(0.1, remaining))
            except TimeoutError:
                if future.done():
                    raise

    def _one_call(self, kind, transport, role, schema, payload, max_tokens, timeout):
        self.attempt = None
        if self.local_only or transport is None:
            return None, "no_model", None
        request = self._prefix(kind, role, schema).request(kind, payload, max_tokens=max_tokens, timeout=timeout)
        if request.input_upper_bound > _REPORT_INPUT_LIMITS[kind]:
            return None, "input_limit", None
        try:
            request, self.attempt = self._start(request, required=kind == "legacy")
        except BudgetDispatchDenied as exc:
            return None, "deadline" if str(exc) == "deadline_exceeded" else "budget_denied", None
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"report-{kind}")
        try:
            future = pool.submit(self._outbound, transport, request)
            return self._wait_result(future, request.timeout), "ok", request
        except ReportCancelled:
            raise
        except TimeoutError:
            return None, "timeout", request
        except Exception:
            return None, "provider_error", request
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _payload(pack):
        return {
            "section_id": pack.section.section_id,
            "title": pack.section.title,
            "scope": pack.section.scope,
            "coverage_keys": pack.section.coverage_keys,
            "claims": [asdict(claim) for claim in pack.claims],
            "conflicts": [asdict(group) for group in pack.conflicts],
            "excerpts": [asdict(excerpt) for excerpt in pack.excerpts],
        }

    def _map_request(self, prefix, pack):
        while pack.claims or pack.conflicts:
            request = prefix.request(
                "map", self._payload(pack), max_tokens=768, timeout=self.settings.v4_report_map_timeout_seconds,
            )
            if request.input_upper_bound <= _REPORT_INPUT_LIMITS["map"]:
                return request, pack
            claims = pack.claims[:-1] if pack.claims else ()
            conflicts = pack.conflicts if pack.claims else pack.conflicts[:-1]
            refs = {ref for claim in claims for ref in claim.support_refs}
            refs.update(ref for group in conflicts for side in group.sides for ref in side.support_refs)
            pack = replace(pack, claims=claims, conflicts=conflicts,
                           excerpts=tuple(item for item in pack.excerpts if item.ref in refs))
        return None, pack

    def _guard_fact(self, row, pack, claims):
        claim_id = row["claim_id"]
        claim = claims.get(claim_id) if isinstance(claim_id, str) else None
        if not claim:
            self.rejections["unsupported_claim"] += 1
            return None
        text = row["text"]
        if not isinstance(text, str) or _UNSAFE_REPORT_TEXT.search(text) or not text.strip() or len(text) > 700:
            self.rejections["unsupported_claim"] += 1
            return None
        rows = row["evidence_refs"]
        if not isinstance(rows, list) or not rows or len(rows) > _MAX_SECTION_EXCERPTS:
            self.rejections["unsupported_claim"] += 1
            return None
        selected = set()
        for raw in rows:
            if not isinstance(raw, dict) or not {"source_id", "anchor_id"} <= raw.keys() <= {
                "source_id", "anchor_id", "artifact_version", "excerpt_fingerprint",
            } or not isinstance(raw["source_id"], str):
                self.rejections["unsupported_claim"] += 1
                return None
            sid = raw["source_id"]
            if sid not in self.frozen.eligible_source_ids:
                reason = "ineligible" if self.reporter._evidence_store and self.reporter._evidence_store.get_evidence(sid) else "unknown_source"
            elif sid not in pack.section.allowed_source_ids:
                reason = "cross_section"
            else:
                matches = [
                    ref for ref in claim.support_refs
                    if ref.source_id == sid and ref.anchor_id == raw["anchor_id"]
                    and all(raw.get(key, getattr(ref, key)) == getattr(ref, key)
                            for key in ("artifact_version", "excerpt_fingerprint"))
                ]
                if len(matches) == 1:
                    selected.add(matches[0])
                    continue
                reason = "stale_anchor" if any(ref.source_id == sid for ref in claim.support_refs) else "unsupported_claim"
            self.rejections[reason] += 1
            return None
        # Only the original statement or a verified orthographic equivalent can inherit its receipt.
        if _report_key(text) != _report_key(claim.text):
            self.rejections["unsupported_claim"] += 1
            return None
        return _fact_block(pack.section, claim, tuple(sorted(selected, key=_ref_key)))

    def _guard_map(self, response, pack):
        payload = parse_report_json(response.text, byte_limit=12288) if response.finish_reason in {"stop", "end_turn"} else None
        if not payload or set(payload) != {"section_id", "blocks"} or payload["section_id"] != pack.section.section_id:
            return None
        rows = payload["blocks"]
        if not isinstance(rows, list) or len(rows) > _MAX_BLOCKS:
            return None
        claims = {claim.claim_id: claim for claim in pack.claims}
        groups = {group.conflict_id: group for group in pack.conflicts}
        labels = dict(pack.required_coverage)
        blocks = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
                self.rejections["unsupported_claim"] += 1
                continue
            if row["kind"] == "fact" and set(row) == {"kind", "claim_id", "text", "evidence_refs"}:
                if block := self._guard_fact(row, pack, claims):
                    blocks.append(block)
            elif row["kind"] == "conflict" and set(row) == {"kind", "conflict_id"} and isinstance(row["conflict_id"], str):
                if group := groups.get(row["conflict_id"]):
                    blocks.extend(_conflict_blocks(pack.section, group))
                else:
                    self.rejections["unsupported_claim"] += 1
            elif (
                row["kind"] == "gap" and set(row) == {"kind", "coverage_key", "gap_code"}
                and isinstance(row["coverage_key"], str) and row["coverage_key"] in labels
                and row["gap_code"] == "missing_scope"
            ):
                key = row["coverage_key"]
                blocks.append(_gap_block(pack.section, labels[key], key))
            elif row["kind"] == "transition" and set(row) == {"kind", "template"} and isinstance(row["template"], str) and row["template"] in _TRANSITIONS:
                blocks.append(_report_block(pack.section, "transition", _TRANSITIONS[row["template"]]))
            else:
                self.rejections["unsupported_claim"] += 1
        draft = _section_draft(pack.section, blocks)
        if len(draft.blocks) > _MAX_BLOCKS or len(draft.markdown) > 2400 or len(draft.markdown.encode("utf-8")) > 4096:
            return None
        return draft

    def _map(self):
        prefix = self._prefix("map", REPORT_MAP_ROLE, REPORT_MAP_SCHEMA)
        required = {key for key, _ in self.frozen.required_coverage}
        packs = sorted(self.frozen.packs, key=lambda pack: (
            not required.intersection(pack.section.coverage_keys), not bool(pack.conflicts), pack.section.order,
        ))
        drafts = {}
        ready = []
        queued = time.monotonic()
        for pack in packs:
            sid = pack.section.section_id
            reason, saved = self._recover("map", sid)
            if reason is not None:
                drafts[sid] = self._restore_draft(saved) if saved else self.reducer.fallback(pack)
                if reason == "outcome_unknown" and saved is None:
                    self._complete("map", reason, asdict(drafts[sid]), section_id=sid)
                    self._record("map", sid, reason, queued)
                continue
            request, sent = self._map_request(prefix, pack) if self.transport and not self.local_only else (None, pack)
            if request:
                ready.append((pack, sent, request))
            else:
                reason = "no_model" if not self.transport else "input_limit" if pack.claims or pack.conflicts else "missing_scope"
                drafts[sid] = self.reducer.fallback(pack)
                self._complete("map", reason, asdict(drafts[sid]), section_id=sid)
                self._record("map", sid, reason, queued)
        yield from self._progress()
        if not ready:
            return drafts
        workers = 1 if len(ready) <= 2 else min(len(ready), self.settings.v4_report_map_max_concurrency)
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="report-map")
        pending = {}
        next_job = iter(ready)
        # A timed-out worker retains capacity until it actually exits.
        abandoned = set()
        try:
            while True:
                abandoned = {job for job in abandoned if not job.done()}
                while len(pending) + len(abandoned) < workers and (job := next(next_job, None)):
                    pack, sent, request = job
                    started = time.monotonic()
                    if self.observer:
                        self.observer.record_report_timing("report_section_queue_ms", (started - queued) * 1000, label="map")
                    sid = pack.section.section_id
                    try:
                        request, settle = self._start(request, sid, required=bool(required.intersection(pack.section.coverage_keys)))
                    except BudgetDispatchDenied as exc:
                        reason = "deadline" if str(exc) == "deadline_exceeded" else "budget_denied"
                        drafts[sid] = self.reducer.fallback(pack)
                        self._complete("map", reason, asdict(drafts[sid]), section_id=sid)
                        self._record("map", sid, reason, started)
                        continue
                    pending[pool.submit(self._outbound, self.transport, request)] = (pack, sent, request, started, settle)
                yield from self._progress()
                if not pending:
                    break
                if self.checkpoint and self.checkpoint.agent._is_cancelled():
                    raise ReportCancelled()
                deadline = min(started + request.timeout for _, _, request, started, _ in pending.values())
                timeout = max(0, deadline - time.monotonic())
                done, _ = wait(pending, timeout=min(0.1, timeout) if self.checkpoint else timeout, return_when=FIRST_COMPLETED)
                now = time.monotonic()
                for job, (pack, sent, request, started, settle) in list(pending.items()):
                    if job not in done and now < started + request.timeout:
                        continue
                    response = None
                    draft = None
                    if job not in done or now > started + request.timeout:
                        reason = "timeout"
                        if not job.done():
                            abandoned.add(job)
                    else:
                        try:
                            response = job.result()
                        except TimeoutError:
                            reason = "timeout"
                        except Exception:
                            reason = "provider_error"
                        else:
                            draft = self._guard_map(response, sent)
                            reason = "invalid_output" if draft is None else "ok"
                    sid = pack.section.section_id
                    drafts[sid] = draft if draft and draft.blocks else self.reducer.fallback(pack)
                    self._complete("map", reason, asdict(drafts[sid]), section_id=sid, settle=settle, response=response)
                    self._record("map", sid, reason, started, response=response, request=request)
                    if self.observer and len(drafts) == len(packs) - len(ready) + 1:
                        self.observer.record_report_timing("report_first_section_ms", (now - self.started) * 1000)
                    del pending[job]
            for pack, _, _ in next_job:
                sid = pack.section.section_id
                drafts[sid] = self.reducer.fallback(pack)
                self._complete("map", "deadline", asdict(drafts[sid]), section_id=sid)
                self._record("map", sid, "deadline", queued)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        yield from self._progress()
        return drafts

    def _outline(self):
        start = time.monotonic()
        packs = self.frozen.packs
        required = {key for key, _ in self.frozen.required_coverage}
        uncertain = required - {key for pack in packs if pack.claims for key in pack.section.coverage_keys}
        previous, titles = self._recover("outline")
        self.attempt = None
        response = request = None
        if previous is not None:
            reason = previous
        elif not self.settings.v4_report_outline_llm_enabled:
            reason = "ok"
            response = request = None
        elif len(packs) < 3 or len(uncertain) < 2 or self.settings.resolved_report_layout_mode() == "fixed":
            reason = "no_candidates"
            response = request = None
        else:
            rows = [
                {"slot_key": pack.section.slot_key, "title": pack.section.title, "coverage_keys": list(pack.section.coverage_keys)}
                for pack in packs
            ]
            response, reason, request = self._one_call(
                "outline", self.transport, REPORT_OUTLINE_ROLE, REPORT_OUTLINE_SCHEMA, {"sections": rows}, 384, 6,
            )
            if response:
                parsed = self._json_response(response, 5000)
                proposed = parsed.get("sections") if parsed and set(parsed) == {"sections"} else None
                if isinstance(proposed, list) and len(proposed) == len(rows) and all(
                    isinstance(row, dict) and set(row) == set(original)
                    and row["slot_key"] == original["slot_key"] and row["coverage_keys"] == original["coverage_keys"]
                    and isinstance(row["title"], str) and 0 < len(row["title"].strip()) <= 120
                    and not _UNSAFE_REPORT_TEXT.search(row["title"])
                    for row, original in zip(proposed, rows)
                ):
                    titles = {row["slot_key"]: row["title"].strip() for row in proposed}
                else:
                    reason = "invalid_output"
        if titles:
            sections = tuple(
                replace(section, title=titles.get(section.slot_key, section.title))
                for section in self.frozen.outline.sections
            )
            by_id = {section.section_id: section for section in sections}
            self.frozen = replace(
                self.frozen, outline=replace(self.frozen.outline, sections=sections),
                packs=tuple(replace(pack, section=by_id[pack.section.section_id]) for pack in packs),
            )
        if previous is None or (previous == "outcome_unknown" and titles is None):
            self._complete("outline", reason, titles, settle=self.attempt, response=response)
        self._record("outline", "outline", reason, start, response=response, request=request)

    @staticmethod
    def _json_response(response, byte_limit):
        if response.finish_reason in {"stop", "end_turn"}:
            return parse_report_json(response.text, byte_limit=byte_limit)
        return None

    def _legacy_draft(self):
        start = time.monotonic()
        previous, saved = self._recover("legacy")
        if saved is not None:
            return {sid: self._restore_draft(draft) for sid, draft in saved.items()}
        data = {
            "layout": self.settings.resolved_report_layout_mode(),
            "sections": [self._payload(pack) for pack in self.frozen.packs],
        }
        self.attempt = None
        response, reason, request = None, previous, None
        if previous is None:
            response, reason, request = self._one_call(
                "legacy", self.transport, REPORT_LEGACY_ROLE, REPORT_LEGACY_SCHEMA, data, 2048, 20,
            )
        if response and (response.finish_reason not in {"stop", "end_turn"} or len(response.text.encode("utf-8")) > 16384):
            reason = "invalid_output"
        drafts = {}
        if response and reason == "ok":
            drafts = self._bind_legacy_draft(response.text)
        self._complete("legacy", reason, {sid: asdict(draft) for sid, draft in drafts.items()},
                       settle=self.attempt, response=response)
        self._record("legacy", "legacy", reason, start, response=response, request=request)
        return drafts

    def _bind_legacy_draft(self, markdown):
        # Markdown is only an ordering and selection hint. Each paragraph needs an exact authorized claim.
        lines = [line.strip() for line, fenced in iter_markdown_lines(markdown) if not fenced]
        by_identity = {}
        for index, line in enumerate(lines):
            cited = set(CITATION_PATTERN.findall(line))
            if cited:
                plain = strip_citation_markers(line).strip().removeprefix("- ").strip()
                by_identity.setdefault(plain, []).append((index, cited))
        drafts = {}
        for pack in self.frozen.packs:
            matched = []
            for claim in pack.claims:
                for index, cited in by_identity.get(claim.text, ()):
                    refs = tuple(ref for ref in claim.support_refs if ref.source_id in cited)
                    if cited and {ref.source_id for ref in refs} == cited:
                        matched.append((index, _fact_block(pack.section, claim, refs)))
                        break
            if matched:
                drafts[pack.section.section_id] = _section_draft(
                    pack.section, (block for _, block in sorted(matched, key=lambda pair: pair[0])[:_MAX_BLOCKS]),
                )
        return drafts

    def _live_refs(self) -> set[ReportEvidenceRef]:
        current = {item.ref for record in self.planner.current_records().values() for item in _bounded_excerpts(record)}
        return current.intersection(item.ref for item in self.frozen.excerpts)

    def _published_claims(self, drafts):
        by_id = {claim.claim_id: claim for pack in self.frozen.packs for claim in pack.claims}
        body_ids = {pack.section.section_id for pack in self.frozen.packs}
        return tuple(
            (draft.section_id, replace(by_id[block.claim_id], support_refs=block.evidence_refs))
            for draft in drafts if draft.section_id in body_ids
            for block in draft.blocks if block.kind == "fact"
        )

    def _summary_order(self, drafts):
        claims = self._published_claims(drafts)
        if len({section for section, _ in claims}) < 3:
            return (), "no_candidates", None, None
        ids = tuple(claim.claim_id for _, claim in claims[:24])
        response, reason, request = self._one_call(
            "reduce", self.transport, REPORT_REDUCER_ROLE, REPORT_REDUCER_SCHEMA,
            {"claim_ids": ids}, 512, 6,
        )
        if response is None:
            return (), reason, response, request
        result = self._json_response(response, 5000)
        rows = result.get("summary_claim_ids") if result and set(result) == {"summary_claim_ids"} else None
        if not isinstance(rows, list) or len(rows) != len(ids) or any(not isinstance(item, str) for item in rows) or set(rows) != set(ids):
            return (), "invalid_output", response, request
        return tuple(rows), "ok", response, request

    def _critique(self, drafts, live_refs):
        start = time.monotonic()
        claims = self._published_claims(drafts)
        visible = tuple(dict.fromkeys(
            block.conflict_id for draft in drafts for block in draft.blocks if block.kind == "conflict" and block.conflict_id
        ))
        excerpts = tuple(item for item in self.frozen.excerpts if item.ref in live_refs)
        gaps = tuple(
            (draft.section_id, block.coverage_key) for draft in drafts for block in draft.blocks
            if block.kind == "gap" and block.coverage_key in dict(self.frozen.required_coverage)
        )
        critic = AdversarialRedTeamCritic()
        previous, saved = self._recover("critic")
        if saved is not None:
            return ReportCriticResult(
                saved["status"], tuple(CriticIssue(**{**issue, "source_ids": tuple(issue["source_ids"])})
                                      for issue in saved["issues"]), saved["reason"],
            )
        confirmed = critic.confirm(claims, excerpts, self.frozen.conflicts, visible)
        response = request = None
        self.attempt = None
        if previous:
            reason, sample = previous, {}
        elif claims or self.frozen.conflicts:
            sample = critic.sample(claims, excerpts, self.frozen.conflicts, gaps)
            for limit in (24, 12, 6, 2, 1):
                sample = critic.sample(claims[:limit], excerpts, self.frozen.conflicts[:min(limit, 8)], gaps)
                response, reason, request = self._one_call(
                    "critic", self.critic_transport, REPORT_CRITIC_ROLE, REPORT_CRITIC_SCHEMA, sample,
                    768, self.settings.v4_red_team_timeout_seconds,
                )
                if reason != "input_limit":
                    break
        else:
            reason = "no_candidates"
            sample = {}
        result = critic.evaluate(response, sample=sample, confirmed=confirmed, reason=reason)
        self._complete("critic", reason if previous else result.reason, asdict(result), settle=self.attempt, response=response,
                       update={"critique": {"status": result.status, "complete": True}})
        self._record("critic", "critic", result.reason, start, response=response, request=request)
        if self.observer:
            self.observer.record_report_event("report_critic_status_total", result.status)
        return result

    def _revise(self, drafts, critic):
        digest = _report_fingerprint(report_json([asdict(draft) for draft in drafts]))
        plan = AdversarialRedTeamCritic.plan_revision(digest, critic.issues, round_number=0)
        high = frozenset(issue.claim_id for issue in critic.issues if issue.severity == "high" and issue.claim_id)
        if plan is None:
            return high, {}
        previous, saved = self._recover("revision")
        if saved is not None:
            return frozenset(saved["withheld"]), saved["replacements"]
        if self.checkpoint and not self.checkpoint.data.get("revision"):
            self.checkpoint.commit({"revision": {"draft_digest": digest, "plan": asdict(plan)}})
        start = time.monotonic()
        self.attempt = None
        replacements = {}
        reason = previous or "ok"
        response = request = None
        published = {claim.claim_id: claim for _, claim in self._published_claims(drafts)}
        targets = {
            issue.claim_id: published[issue.claim_id] for issue in plan.issues
            if issue.claim_id in published and issue.proposed_action in {"qualify", "reverify"}
        }
        if not previous and self.settings.v4_report_revision_llm_enabled and targets:
            snippets = {
                cid: tuple(item.text for item in self.frozen.excerpts if item.ref in claim.support_refs)[:2]
                for cid, claim in targets.items()
            }
            response, reason, request = self._one_call(
                "revision", self.transport, REPORT_REVISION_ROLE, REPORT_REVISION_SCHEMA, {
                    "claims": [{"claim_id": cid, "text": claim.text, "excerpts": snippets[cid]}
                               for cid, claim in targets.items()],
                }, 512, 6,
            )
            if response:
                payload = self._json_response(response, 6000)
                rows = payload.get("revisions") if payload and set(payload) == {"revisions"} else None
                if not isinstance(rows, list) or len(rows) > 4:
                    reason = "invalid_output"
                else:
                    seen = set()
                    for row in rows:
                        if not isinstance(row, dict) or set(row) != {"claim_id", "text"}:
                            continue
                        cid, text = row["claim_id"], row["text"]
                        if not isinstance(cid, str) or cid in seen or cid not in targets or not isinstance(text, str):
                            continue
                        seen.add(cid)
                        if self.planner._safe_fact(text) and text == targets[cid].text and AdversarialRedTeamCritic.qualifies(
                            targets[cid], text, snippets[cid],
                        ):
                            replacements[cid] = text
        self._complete(
            "revision", reason, {"withheld": sorted(high), "replacements": replacements},
            settle=self.attempt, response=response,
            update={"revision": {"draft_digest": digest, "plan": asdict(plan), "complete": True}},
        )
        self._record("revision", "revision", reason, start, response=response, request=request)
        if self.observer:
            self.observer.record_report_event("report_revision_batch_total", "applied")
        return high, replacements

    def _publish(self, drafts, critic):
        sections = self.frozen.outline.sections
        store = self.reporter._evidence_store
        cited = tuple(dict.fromkeys(sid for draft in drafts for sid in draft.source_ids))
        references = store.build_reference_map(list(cited)) if store else []
        refs_by_id = {
            item["source_id"]: {
                **item, "title": _display_report_text(item["title"]),
                "url": _safe_workbench_url(item.get("url")),
                "backend": _display_report_text(str(item.get("backend") or "sandbox")),
                "derived_from_source_ids": [
                    sid for sid in item.get("derived_from_source_ids") or () if sid in self.frozen.eligible_source_ids
                ],
            }
            for item in references if item["source_id"] in cited
        }
        lines = [f"# {_display_report_text(self.state.research_topic or '研究报告')}"]
        bindings = []
        visible_sections = []
        published = {}
        for section, draft in zip(sections, drafts):
            visible_sections.append({
                "section_id": section.section_id, "order": section.order, "title": section.title, "kind": section.kind,
                "coverage_keys": list(section.coverage_keys), "paragraph_index": len(lines),
            })
            lines.append(f"## {_display_report_text(section.title)}")
            if section.kind == "references":
                lines.append(render_references(list(refs_by_id.values())))
                continue
            for block in draft.blocks:
                if any(ref.source_id not in refs_by_id for ref in block.evidence_refs):
                    block = _gap_block(section, "本节的一项证据已失效", block.claim_id, "stale_anchor")
                rendered = _render_block(block)
                source_ids = list(dict.fromkeys(ref.source_id for ref in block.evidence_refs))
                if set(CITATION_PATTERN.findall(rendered)) != set(source_ids):
                    continue
                paragraph = len(lines)
                lines.append(rendered)
                bindings.append({
                    "section_id": section.section_id, "block_id": block.block_id, "kind": block.kind,
                    "claim_id": block.claim_id, "conflict_id": block.conflict_id, "gap_code": block.gap_code,
                    "coverage_key": block.coverage_key, "paragraph_index": paragraph, "source_ids": source_ids,
                    "text_fingerprint": _report_fingerprint(block.text),
                    "evidence_refs": [asdict(ref) for ref in block.evidence_refs],
                })
                if block.kind == "fact" and section.kind == "body":
                    published[block.claim_id] = (block, source_ids)
        report = "\n\n".join(lines)
        workbench = build_report_workbench_payload(
            replace(self.state, report_charts=[], evidence_conflicts=[], report_workbench_fingerprint=None), report,
            evidence_store=store, config=self.settings,
        )
        records = {item["source_id"]: item for item in store.lookup()} if store else {}
        index = {
            sid: item for sid, item in workbench["citation_index"].items()
            if sid in records and self.planner.owns_record(records[sid])
        }
        for sid, item in index.items():
            if sid not in self.frozen.eligible_source_ids:
                item["linked_claims"] = []
        workbench["citation_index"] = workbench["citation_drawer_map"] = index
        workbench["paragraph_bindings"] = [
            {
                "paragraph_index": row["paragraph_index"], "excerpt": truncate_text(lines[row["paragraph_index"]], 180),
                "citation_ids": row["source_ids"], "conflict_ids": [row["conflict_id"]] if row["conflict_id"] else [],
                "section_id": row["section_id"], "block_id": row["block_id"], "claim_id": row["claim_id"],
            }
            for row in bindings if row["source_ids"]
        ]
        self._workbench_conflicts(workbench, bindings, drafts)
        charts = self._grounded_charts(published)
        workbench.update(
            report_version=1, outline_version=self.frozen.outline.outline_version,
            report_sections=visible_sections, report_blocks=bindings, charts_spec=charts,
        )
        if critic:
            workbench["red_team"] = {
                "status": critic.status, "reason_code": critic.reason,
                "issues": [{
                    **asdict(issue), "source_ids": [sid for sid in issue.source_ids if sid in index],
                    "resolution": "withheld" if issue.claim_id not in published else "open",
                } for issue in critic.issues],
            }
        self.state.report_workbench = workbench
        self.state.report_workbench_fingerprint = _report_fingerprint(report)
        self.state.report_charts = charts
        return report

    def _workbench_conflicts(self, workbench, bindings, drafts):
        index = workbench["citation_index"]
        rows = {row["block_id"]: row for row in bindings if row["kind"] == "conflict"}
        groups = defaultdict(list)
        for draft in drafts:
            for block in draft.blocks:
                if row := rows.get(block.block_id):
                    for sid in row["source_ids"]:
                        groups[block.conflict_id].append({
                            "source_id": sid, "claim_id": block.claim_id, "claim_text": block.text,
                            "url": index.get(sid, {}).get("url", ""), "title": index.get(sid, {}).get("title", sid),
                            "anchor_id": next((ref.anchor_id for ref in block.evidence_refs if ref.source_id == sid), None),
                        })
        workbench["evidence_conflicts"] = [
            {
                "conflict_id": group.conflict_id, "sides": groups[group.conflict_id],
                "adjudication_status": group.status, "aliases": list(group.aliases),
            }
            for group in self.frozen.conflicts if len({side["source_id"] for side in groups[group.conflict_id]}) >= 2
        ]
        for group in workbench["evidence_conflicts"]:
            for side in group["sides"]:
                entry = index.get(side["source_id"])
                if entry is not None and group["conflict_id"] not in entry["conflict_ids"]:
                    entry["conflict_ids"].append(group["conflict_id"])

    def _grounded_charts(self, published):
        results = []
        eligible = set(self.frozen.eligible_source_ids)
        for chart in self.state.report_charts:
            sanitized = sanitize_chart_spec(chart, valid_source_ids=eligible)
            if not sanitized or sanitized["library"] != "vega-lite":
                continue
            data = sanitized["spec"].get("data")
            values = data.get("values") if isinstance(data, dict) else None
            if not isinstance(values, list) or not values:
                continue
            # The first supported chart format is inline data with explicit per-point fact bindings.
            def grounded(point):
                if not isinstance(point, dict) or not isinstance(point.get("claim_id"), str):
                    return False
                binding = published.get(point["claim_id"])
                if binding is None or len(point) != 4 or set(point) != {"claim_id", "source_id", "label", "value"}:
                    return False
                claim, sources = binding
                value = point["value"]
                return (
                    isinstance(value, (float, int)) and not isinstance(value, bool)
                    and isinstance(point["label"], str) and _report_key(point["label"]) in _report_key(claim.text)
                    and point["source_id"] in sources and point["source_id"] in sanitized["source_ids"]
                    and not numeric_grounding_mismatch(str(value), [claim.text])
                    and set(sanitized["spec"]) <= {"data", "mark", "encoding", "title", "width", "height", "$schema"}
                )
            if all(grounded(point) for point in values):
                results.append(sanitized)
        return results

    def generate(self) -> str:
        run = self.run()
        while True:
            try:
                next(run)
            except StopIteration as completed:
                return completed.value

    def run(self):
        if self.checkpoint:
            material = asdict(self.frozen)
            material["eligible_source_ids"] = sorted(self.frozen.eligible_source_ids)
            digest = _report_fingerprint(report_json({
                "input": material, "skill_context": self.skill_context, "terminology": self.terminology,
                "dag_revision": ((self.state.v4 or {}).get("replan") or {}).get("dag_revision", 0),
            }))
            saved_digest = self.checkpoint.data.get("input_digest")
            self.local_only = (
                self.checkpoint.data.get("schema_version") != 1
                or (saved_digest is not None and saved_digest != digest)
            )
            if saved_digest is None:
                self.checkpoint.commit({"input_digest": digest, "outline_version": 1,
                                        "evidence_manifest": sorted(self.frozen.eligible_source_ids)})
            if self.local_only:
                self.checkpoint.commit({"degradation_reason": "input_changed"})
        if self.settings.parallel_report_enabled:
            self._outline()
        if self.checkpoint and (not self.checkpoint.data.get("outline") or self.local_only):
            cp = self.checkpoint
            sections = self.frozen.outline.sections
            cp.commit({"outline": asdict(self.frozen.outline)}, cp.event(
                "report_outline", "outline-1", outline_version=1,
                sections=[section.title for section in sections], section_ids=[section.section_id for section in sections],
            ))
            yield from self._progress()
        if self.settings.parallel_report_enabled:
            drafts = yield from self._map()
        else:
            drafts = self._legacy_draft()
        live_refs = self._live_refs()
        started = time.monotonic()
        current = self.reducer.merge(self.frozen, drafts, live_refs, include_local_facts=True)
        order = ()
        reason = "ok"
        response = request = None
        previous, saved = self._recover("reduce")
        self.attempt = None
        if previous is not None:
            order, reason = tuple(saved or ()), previous
        elif self.settings.parallel_report_enabled and self.settings.v4_report_reducer_llm_enabled:
            order, reason, response, request = self._summary_order(current)
        if saved is None:
            self._complete("reduce", reason, order, settle=self.attempt, response=response,
                           update={"reduce": {"complete": True}})
        self._record("reduce", "reduce", reason, started, response=response, request=request)
        critic = self._critique(current, live_refs) if self.settings.red_team_enabled else None
        withheld, replacements = self._revise(current, critic) if critic else (frozenset(), {})
        if self.checkpoint and critic and not self.checkpoint.data.get("critic_event_committed"):
            counts = defaultdict(int)
            for issue in critic.issues:
                counts[issue.kind] += 1
            cp = self.checkpoint
            cp.commit({"critic_event_committed": True}, cp.event(
                "report_critic_completed", "critic-1", outline_version=1, status=critic.status,
                issue_counts=dict(counts), patch_applied=bool(withheld or replacements),
            ))
            yield from self._progress()
        merged = self.reducer.merge(
            self.frozen, drafts, self._live_refs(), withheld=withheld, summary_order=order,
            replacements=replacements, include_local_facts=True,
        )
        report = self._publish(merged, critic)
        if self.checkpoint:
            missing_required = any(
                block.kind == "gap" and block.coverage_key in dict(self.frozen.required_coverage)
                for draft in merged for block in draft.blocks
            )
            sections = self.checkpoint.data["sections_by_id"]
            all_map_unavailable = self.settings.parallel_report_enabled and not any(
                row.get("status") == "completed" for row in sections.values()
            )
            partial_reason = "input_changed" if self.local_only else (
                "missing_scope" if missing_required or not self._published_claims(merged) or withheld else (
                    "no_model" if all_map_unavailable else None
                )
            )
            status = {
                draft.section_id: {
                    "status": sections.get(draft.section_id, {}).get("status", "completed"),
                    "claim_ids": list(draft.claim_ids),
                }
                for draft in merged
            }
            self.checkpoint.commit({"sections_by_id": status, "degradation_reason": partial_reason})
        if self.observer:
            for code, count in self.rejections.items():
                self.observer.record_report_event("report_citation_rejected_total", code, amount=count)
            for draft in merged:
                for block in draft.blocks:
                    if block.kind == "gap":
                        required = block.coverage_key in dict(self.frozen.required_coverage)
                        self.observer.record_report_event("report_coverage_gap_total", "required" if required else "uncovered")
            if critic:
                for item in self.state.report_workbench["red_team"]["issues"]:
                    self.observer.record_report_event(
                        "report_critic_issue_total", f'{item["kind"]}:{item["severity"]}:{item["resolution"]}',
                    )
            elapsed = (time.monotonic() - self.started) * 1000
            self.observer.record_report_timing("report_stage_ms", elapsed)
            self.observer.record_report_timing("report_first_verified_chunk_ms", elapsed)
        return report
