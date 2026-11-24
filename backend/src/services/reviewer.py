"""Review-stage helpers for claim grounding and evidence quality checks."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace, extract_llm_usage
from models import SummaryState, TodoItem
from prompts import get_current_date
from services.context_builder import ContextPrefixBuilder, PromptEnvelope
from services.evidence import extract_citation_ids, format_evidence_sources
from services.evidence_index import EvidenceRetrievalService
from services.fact_ledger import FactLedger, render_evidence_index_cards_block
from services.skill_verifiers import run_skill_verifiers
from services.text_processing import (
    looks_like_meta_reasoning,
    normalize_agent_markdown,
    strip_citation_markers,
)
from utils import strip_thinking_tokens, truncate_text

_FRESHNESS_PATTERNS = (
    "最新",
    "近期",
    "最近",
    "today",
    "this week",
    "this month",
    "current",
    "current state",
    "latest",
    "newest",
    "recent",
)


def _dynamic_freshness_years(*, now: datetime | None = None) -> tuple[str, ...]:
    current_year = (now or datetime.now(timezone.utc)).year
    return (str(current_year - 1), str(current_year), str(current_year + 1))


@dataclass
class ReviewIssue:
    """Single review finding attached to a task or request."""

    task_id: int | None
    severity: str
    check: str
    message: str
    source_ids: list[str] = field(default_factory=list)
    origin: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
            "source_ids": list(self.source_ids),
            "origin": self.origin,
        }


_SALIENT_CLAIM_NUMBER_PATTERN = re.compile(
    r"(?:"
    r"\b(?:19|20)\d{2}\b"
    r"|\b\d+(?:\.\d+)?\s*(?:%|ms|s|GB|MB|TB|KB|x|倍|亿|万|千|美元|元|B|M|K)\b"
    r"|\d+(?:\.\d+)?%"
    r")",
    re.IGNORECASE,
)


def _extract_salient_numbers(text: str) -> list[str]:
    """Extract salient quantitative figures (years, percentages, unit numbers) from a claim."""
    if not text:
        return []
    found: list[str] = []
    for match in _SALIENT_CLAIM_NUMBER_PATTERN.finditer(text):
        token = re.sub(r"\s+", "", match.group(0)).lower()
        if token and token not in found:
            found.append(token)
    return found


def _evidence_body_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("title") or "").strip(),
        str(item.get("snippet") or "").strip(),
        str(item.get("raw_content") or "").strip(),
        str(item.get("full_content") or "").strip(),
        str(item.get("content") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _number_token_supported_in_text(token: str, corpus_lower: str) -> bool:
    if token in corpus_lower:
        return True
    digits_match = re.match(r"(\d+(?:\.\d+)?)", token)
    if digits_match:
        bare = digits_match.group(1)
        # Allow formatted variants like "45 %" or "45.0" or "45,"
        if re.search(rf"(?<![0-9.]){re.escape(bare)}(?![0-9])", corpus_lower):
            return True
    return False


class ReviewService:
    """Run deterministic and optional LLM-assisted review checks."""

    def __init__(
        self,
        review_agent: ToolAwareSimpleAgent | None,
        config: Configuration,
        *,
        retrieval_service: EvidenceRetrievalService | None = None,
        request_id_getter: Callable[[], str | None] | None = None,
        claim_verifier: Callable[[str, list[dict[str, Any]]], tuple[bool, str] | None] | None = None,
        claim_verification_orchestrator: Any | None = None,
        evidence_store: Any | None = None,
    ) -> None:
        self._agent = review_agent
        self._config = config
        self._retrieval_service = retrieval_service
        self._request_id_getter = request_id_getter
        self._claim_verifier = claim_verifier
        self._claim_verification_orchestrator = claim_verification_orchestrator
        self._evidence_store = evidence_store

    def review_request(
        self,
        state: SummaryState,
        *,
        observer: RequestTrace | None = None,
        skill_pack_context: str | None = None,
        active_verifiers: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Review task summaries and evidence before final report generation."""

        issues, claims_by_task = self._rule_based_review(state)
        resolved_verifiers = (
            list(active_verifiers)
            if active_verifiers is not None
            else list(getattr(self._config, "active_skill_verifiers", []) or [])
        )
        verifier_issue_dicts, executed_verifiers = run_skill_verifiers(
            state,
            self._config,
            resolved_verifiers,
        )
        verifier_issues = [
            ReviewIssue(
                task_id=item.get("task_id"),
                severity=str(item.get("severity") or "medium"),
                check=str(item.get("check") or "skill_verifier"),
                message=str(item.get("message") or ""),
                source_ids=list(item.get("source_ids") or []),
                origin=str(item.get("origin") or "skill_verifier"),
            )
            for item in verifier_issue_dicts
        ]
        issues.extend(verifier_issues)
        claim_verification_summary: dict[str, Any] = {}
        evidence_conflict_count = 0
        unresolved_conflict_count = 0

        orchestrator = getattr(self, "_claim_verification_orchestrator", None)
        if orchestrator is not None and bool(getattr(self._config, "nli_verifier_enabled", False)):
            nli_out = orchestrator.verify_request(
                state,
                claims_by_task=claims_by_task,
                observer=observer,
                fallback_claim_verifier=self._numeric_fallback_only,
            )
            claim_verification_summary = dict(nli_out.get("summary") or {})
            conflicts = nli_out.get("conflicts") or []
            evidence_conflict_count = len(conflicts)
            unresolved_conflict_count = sum(
                1
                for c in conflicts
                if isinstance(c, dict) and c.get("adjudication_status") == "unresolved_divergence"
            )
            for item in nli_out.get("issues") or []:
                if not isinstance(item, dict):
                    continue
                issues.append(
                    ReviewIssue(
                        task_id=item.get("task_id"),
                        severity=str(item.get("severity") or "medium"),
                        check=str(item.get("check") or "claim_evidence_mismatch"),
                        message=str(item.get("message") or ""),
                        source_ids=list(item.get("source_ids") or []),
                        origin=str(item.get("origin") or "claim_verifier"),
                    )
                )

        llm_summary: dict[str, Any] = {}

        if self._config.review_agent_enabled and self._agent is not None:
            envelope = self._build_envelope(state, issues, skill_pack_context=skill_pack_context)
            prompt = envelope.to_flat_prompt()
            run_env = getattr(self._agent, "run_envelope", None)
            try:
                if callable(run_env):
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
                        stage="review",
                    )
            else:
                if observer:
                    observer.record_llm_call(
                        success=True,
                        prompt_text=prompt,
                        completion_text=response,
                        usage=extract_llm_usage(self._agent),
                        stage="review",
                    )
                llm_issues, llm_summary = self._parse_response(response)
                issues = self._merge_issues(issues, llm_issues)
            finally:
                self._agent.clear_history()

        summary = self._apply_review_result(
            state,
            issues=issues,
            claims_by_task=claims_by_task,
            llm_summary=llm_summary,
            executed_verifiers=executed_verifiers,
            verifier_issue_count=len(verifier_issues),
            claim_verification_summary=claim_verification_summary,
            evidence_conflict_count=evidence_conflict_count,
            unresolved_conflict_count=unresolved_conflict_count,
        )
        return summary

    def _build_envelope(
        self,
        state: SummaryState,
        issues: list[ReviewIssue],
        *,
        skill_pack_context: str | None = None,
    ) -> PromptEnvelope:
        task_blocks: list[str] = []
        for task in state.todo_items:
            sources = format_evidence_sources(task.evidence_items) or "- 暂无来源"
            excerpt_lines: list[str] = []
            for item in (task.evidence_items or [])[:6]:
                sid = str(item.get("source_id") or "").strip()
                snippet = str(
                    item.get("snippet")
                    or item.get("raw_content")
                    or item.get("full_content")
                    or item.get("content")
                    or ""
                ).strip()
                if sid and snippet:
                    excerpt_lines.append(f"  - [{sid}] {truncate_text(snippet, 240)}")
            excerpt_text = "\n".join(excerpt_lines) or "  - 暂无正文摘录"
            repair_hits = self._repair_hits_for_task(state, task)
            repair_text = "\n".join(
                f"  - [{hit['origin']}] {hit['title']} ({hit['quality_label']}/{hit['freshness_label']}) {hit['url']}"
                for hit in repair_hits
            ) or "  - 暂无额外修补候选"
            issue_text = "\n".join(
                f"  - [{issue.severity}] {issue.check}: {issue.message}"
                for issue in issues
                if issue.task_id == task.id
            ) or "  - 暂无规则审查问题"
            task_blocks.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 目标：{task.intent}\n"
                f"- 状态：{task.status}\n"
                f"- 总结：\n{truncate_text(task.summary or '暂无可用信息', 1200)}\n"
                f"- 来源目录：\n{truncate_text(sources, 1200)}\n"
                f"- 关键证据摘录：\n{excerpt_text}\n"
                f"- 修补候选：\n{repair_text}\n"
                f"- 规则审查结果：\n{issue_text}\n"
            )

        tools_enabled = self._tools_enabled()
        tool_guidance = (
            "你可以调用 evidence_lookup；若需要检索历史修补线索，请使用 `scope=hybrid, mode=repair`。"
            "历史命中只能作为线索，不能直接当作当前 source_id 引用。"
            "你也可以调用 fetch_page 重新抓取 URL，把线索提升为当前证据。"
            if tools_enabled
            else "当前运行模式不启用工具调用，请直接基于下方任务总结、来源目录、修补候选和规则审查结果补充指出证据问题。"
            "历史修补候选只能作为线索，不能直接当作当前 source_id 引用。"
        )
        static_rules = (
            "请基于以下任务结果补充指出仍然存在的证据问题。"
            f"{tool_guidance}"
            "不要重写报告正文，也不要重复输出规则已明确指出的问题。"
        )

        frozen_parts: list[str] = []
        raw_ledger = getattr(state, "fact_ledger", None)
        if getattr(self._config, "context_compaction_enabled", True) and raw_ledger:
            ledger_md = FactLedger.from_dict(raw_ledger).render_frozen_markdown(
                max_entries=int(getattr(self._config, "fact_ledger_max_entries", 40) or 40),
            )
            if ledger_md:
                frozen_parts.append(ledger_md)
        raw_cards = getattr(state, "evidence_index_cards", None) or []
        if getattr(self._config, "evidence_pointer_disclosure_enabled", True) and raw_cards:
            cards_md = render_evidence_index_cards_block(raw_cards)
            if cards_md:
                frozen_parts.append(cards_md)

        volatile_delta = f"当前日期：{get_current_date()}\n\n{''.join(task_blocks)}"
        builder = ContextPrefixBuilder(self._config)
        return builder.build_envelope(
            stage="review",
            system_role=str(getattr(self._agent, "system_prompt", "") or ""),
            static_schema_and_rules=static_rules,
            research_topic=state.research_topic or "",
            skill_pack_context=str(skill_pack_context or "").strip(),
            frozen_index_and_ledger="\n\n".join(frozen_parts),
            volatile_blocks=[("review_tasks", volatile_delta)],
            cache_key_scope="request",
        )

    def _build_prompt(
        self,
        state: SummaryState,
        issues: list[ReviewIssue],
        *,
        skill_pack_context: str | None = None,
    ) -> str:
        return self._build_envelope(
            state,
            issues,
            skill_pack_context=skill_pack_context,
        ).to_flat_prompt()

    def _tools_enabled(self) -> bool:
        provider = str(getattr(self._config, "llm_provider", "") or "").strip().lower()
        if provider == "vllm" or not getattr(self._config, "use_tool_calling", True):
            return False
        if self._agent is not None and getattr(self._agent, "enable_tool_calling", None) is False:
            return False
        return True

    def _rule_based_review(self, state: SummaryState) -> tuple[list[ReviewIssue], dict[int, list[dict[str, Any]]]]:
        issues: list[ReviewIssue] = []
        claims_by_task: dict[int, list[dict[str, Any]]] = {}

        request_evidence_by_id: dict[str, dict[str, Any]] = {}
        for t in state.todo_items:
            for item in t.evidence_items or []:
                if isinstance(item, dict) and item.get("source_id"):
                    request_evidence_by_id[str(item["source_id"]).strip()] = item
        evidence_store = getattr(self, "_evidence_store", None)
        if evidence_store is not None and hasattr(evidence_store, "lookup"):
            for item in evidence_store.lookup(include_full_content=True) or []:
                if isinstance(item, dict) and item.get("source_id"):
                    request_evidence_by_id[str(item["source_id"]).strip()] = item
        request_valid_ids = set(request_evidence_by_id.keys())

        for task in state.todo_items:
            task_issues, task_claims = self._review_task(
                state,
                task,
                request_valid_ids=request_valid_ids,
                request_evidence_by_id=request_evidence_by_id,
            )
            issues.extend(task_issues)
            claims_by_task[task.id] = task_claims

        return issues, claims_by_task

    def _repair_hits_for_task(
        self,
        state: SummaryState,
        task: TodoItem,
    ) -> list[dict[str, Any]]:
        if self._retrieval_service is None:
            return []
        query_text = " ".join(
            part.strip()
            for part in [state.research_topic or "", task.title or "", task.intent or "", task.query or ""]
            if str(part or "").strip()
        )
        result = self._retrieval_service.query(
            query_text,
            task_id=task.id,
            scope="hybrid",
            mode="repair",
            top_k=3,
            request_id=self._request_id_getter() if self._request_id_getter else None,
        )
        return [hit.to_dict() for hit in result.hits[:3]]

    def _review_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        request_valid_ids: set[str] | None = None,
        request_evidence_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[ReviewIssue], list[dict[str, Any]]]:
        issues: list[ReviewIssue] = []
        claims: list[dict[str, Any]] = []

        evidence_items = list(task.evidence_items or [])
        valid_ids = {
            str(item.get("source_id") or "").strip()
            for item in evidence_items
            if str(item.get("source_id") or "").strip()
        }
        effective_valid_ids = valid_ids | (request_valid_ids or set())
        unique_domains = {
            str(item.get("domain") or "").strip().lower()
            for item in evidence_items
            if str(item.get("domain") or "").strip()
        }
        freshness_sensitive = self._is_freshness_sensitive(state.research_topic, task)

        if task.status in {"failed", "skipped"}:
            issues.append(
                ReviewIssue(
                    task_id=task.id,
                    severity="high",
                    check="missing_angle",
                    message=f"任务 {task.id} 未成功完成，可能导致该维度覆盖不足。",
                )
            )
            return issues, claims

        if task.status != "completed":
            return issues, claims

        if len(evidence_items) < self._config.review_min_sources_per_task:
            issues.append(
                ReviewIssue(
                    task_id=task.id,
                    severity="medium",
                    check="weak_evidence",
                    message=(
                        f"任务 {task.id} 仅登记 {len(evidence_items)} 个来源，"
                        "证据覆盖偏薄。"
                    ),
                    source_ids=sorted(valid_ids),
                )
            )

        if evidence_items and len(unique_domains) < self._config.review_min_domains_per_task:
            issues.append(
                ReviewIssue(
                    task_id=task.id,
                    severity="medium",
                    check="low_source_diversity",
                    message=(
                        f"任务 {task.id} 仅覆盖 {len(unique_domains)} 个域名，"
                        "来源多样性不足。"
                    ),
                    source_ids=sorted(valid_ids),
                )
            )

        dated_sources = [item for item in evidence_items if item.get("published_at")]
        recent_sources = [
            item
            for item in evidence_items
            if str(item.get("freshness_label") or "").strip() in {"fresh", "recent", "current"}
        ]
        if freshness_sensitive and evidence_items and not recent_sources:
            issues.append(
                ReviewIssue(
                    task_id=task.id,
                    severity="high" if dated_sources else "medium",
                    check="stale_evidence",
                    message=(
                        "该任务具有明显时效性，但当前来源缺少近期发布时间支撑。"
                        if dated_sources
                        else "该任务具有明显时效性，但当前来源缺少可解析发布时间。"
                    ),
                    source_ids=sorted(valid_ids),
                )
            )

        high_quality_sources = [
            item for item in evidence_items if str(item.get("quality_label") or "") == "high"
        ]
        if evidence_items and not high_quality_sources:
            issues.append(
                ReviewIssue(
                    task_id=task.id,
                    severity="low",
                    check="low_quality_mix",
                    message="当前任务没有明显高质量来源，建议补充官方文档、论文或权威站点。",
                    source_ids=sorted(valid_ids),
                )
            )

        for claim_candidate in self._claim_candidates(task):
            claim_text = str(claim_candidate.get("text") or "").strip()
            if not claim_text:
                continue
            citation_ids = [
                source_id
                for source_id in claim_candidate.get("source_ids") or extract_citation_ids(claim_text)
                if source_id
            ]
            missing = not citation_ids
            invalid_source_ids = [
                source_id for source_id in citation_ids if source_id not in effective_valid_ids
            ]
            if missing:
                issues.append(
                    ReviewIssue(
                        task_id=task.id,
                        severity="high",
                        check="missing_citation",
                        message=f"任务 {task.id} 存在未绑定 source_id 的结论：{truncate_text(claim_text, 140)}",
                    )
                )
                support_status = "missing_citation"
            elif invalid_source_ids:
                issues.append(
                    ReviewIssue(
                        task_id=task.id,
                        severity="high",
                        check="invalid_citation",
                        message=(
                            f"任务 {task.id} 使用了不存在的 source_id："
                            f"{', '.join(invalid_source_ids)}"
                        ),
                        source_ids=invalid_source_ids,
                    )
                )
                support_status = "invalid_citation"
            else:
                support_status = "supported"
                evidence_by_id = dict(request_evidence_by_id or {})
                for item in evidence_items:
                    sid_key = str(item.get("source_id") or "").strip()
                    if sid_key:
                        evidence_by_id[sid_key] = item
                cited_records = [
                    evidence_by_id[sid] for sid in citation_ids if sid in evidence_by_id
                ]
                mismatch_reason = self._verify_claim_against_evidence(claim_text, cited_records)
                if mismatch_reason:
                    issues.append(
                        ReviewIssue(
                            task_id=task.id,
                            severity="medium",
                            check="claim_evidence_mismatch",
                            message=f"任务 {task.id} 存在结论与引用证据不一致风险：{mismatch_reason}",
                            source_ids=citation_ids,
                        )
                    )
                    support_status = "unverified_claim"

            claims.append(
                {
                    "text": claim_text.strip(),
                    "source_ids": citation_ids,
                    "support_status": support_status,
                }
            )

        evidence_by_id = {
            str(item.get("source_id") or "").strip(): item
            for item in evidence_items
            if str(item.get("source_id") or "").strip()
        }
        for item in evidence_items:
            sid = str(item.get("source_id") or "").strip()
            is_sandbox = str(item.get("source_origin") or "").strip().lower() == "sandbox" or "-C" in sid
            if not is_sandbox:
                continue
            lineage_reason = self._verify_computation_lineage(item, evidence_by_id)
            if lineage_reason:
                issues.append(
                    ReviewIssue(
                        task_id=task.id,
                        severity="high",
                        check="sandbox_input_ungrounded",
                        message=f"任务 {task.id} 沙箱计算证据 [{sid}] 溯源校验失败：{lineage_reason}",
                        source_ids=[sid] if sid else [],
                    )
                )

        return issues, claims

    @staticmethod
    def _claim_candidates(task: TodoItem) -> list[dict[str, Any]]:
        claims = getattr(task, "claims", None)
        if isinstance(claims, list) and claims:
            normalized: list[dict[str, Any]] = []
            for item in claims:
                if not isinstance(item, dict):
                    continue
                text = normalize_agent_markdown(str(item.get("text") or "").strip())
                text = strip_citation_markers(text)
                if not text or looks_like_meta_reasoning(text):
                    continue
                normalized.append(
                    {
                        "text": text,
                        "source_ids": [
                            str(source_id).strip()
                            for source_id in item.get("source_ids") or []
                            if str(source_id).strip()
                        ],
                    }
                )
            if normalized:
                return normalized

        payload = getattr(task, "summary_payload", None)
        if isinstance(payload, dict):
            findings = payload.get("key_findings")
            if isinstance(findings, list):
                normalized = []
                for item in findings:
                    if not isinstance(item, dict):
                        continue
                    text = normalize_agent_markdown(str(item.get("text") or "").strip())
                    text = strip_citation_markers(text)
                    if not text or looks_like_meta_reasoning(text):
                        continue
                    source_ids = [
                        str(source_id).strip()
                        for source_id in item.get("source_ids") or []
                        if str(source_id).strip()
                    ]
                    normalized.append({"text": text, "source_ids": source_ids})
                if normalized:
                    return normalized

        return [
            {"text": claim_text, "source_ids": extract_citation_ids(claim_text)}
            for claim_text in ReviewService._extract_claim_texts(task.summary or "")
        ]

    @staticmethod
    def _extract_claim_texts(summary: str) -> list[str]:
        normalized = normalize_agent_markdown(summary or "")
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]

        claims: list[str] = []
        for line in lines:
            if re.match(r"^#{1,6}\s+", line):
                continue
            if re.match(r"^\|(?:[-:\s|]+)\|?$", line):
                continue
            cleaned = re.sub(r"^[-*]\s+", "", line)
            cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
            claim_text = cleaned.strip()
            visible_text = strip_citation_markers(claim_text)
            if len(visible_text) < 8:
                continue
            if looks_like_meta_reasoning(visible_text):
                continue
            claims.append(claim_text)

        if claims:
            return claims

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", normalized)
            if paragraph.strip()
        ]
        return [
            paragraph
            for paragraph in paragraphs
            if len(strip_citation_markers(paragraph)) >= 8
            and not looks_like_meta_reasoning(strip_citation_markers(paragraph))
        ]

    @staticmethod
    def _is_freshness_sensitive(topic: str, task: TodoItem) -> bool:
        combined = " ".join(
            part.strip().lower()
            for part in [topic or "", task.title or "", task.intent or "", task.query or ""]
            if part and part.strip()
        )
        keywords = (*_FRESHNESS_PATTERNS, *_dynamic_freshness_years())
        return any(keyword in combined for keyword in keywords)

    def _parse_response(
        self,
        raw_response: str,
    ) -> tuple[list[ReviewIssue], dict[str, Any]]:
        text = (raw_response or "").strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)
        text = normalize_agent_markdown(text)

        payload = self._extract_json_payload(text)
        if not isinstance(payload, dict):
            return [], {}

        issues: list[ReviewIssue] = []
        for item in payload.get("issues") or []:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "medium").strip().lower()
            if severity not in {"high", "medium", "low"}:
                severity = "medium"
            message = str(item.get("message") or "").strip()
            if not message:
                continue
            source_ids = item.get("source_ids")
            if not isinstance(source_ids, list):
                source_ids = []
            issues.append(
                ReviewIssue(
                    task_id=int(item["task_id"]) if str(item.get("task_id") or "").strip() else None,
                    severity=severity,
                    check=str(item.get("check") or "llm_review").strip() or "llm_review",
                    message=message,
                    source_ids=[str(source_id).strip() for source_id in source_ids if str(source_id).strip()],
                    origin="llm",
                )
            )

        summary = payload.get("summary")
        return issues, summary if isinstance(summary, dict) else {}

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
    def _merge_issues(left: list[ReviewIssue], right: list[ReviewIssue]) -> list[ReviewIssue]:
        merged: list[ReviewIssue] = []
        seen: set[tuple[int | None, str, str]] = set()
        for item in [*left, *right]:
            key = (item.task_id, item.check, item.message)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _verify_claim_against_evidence(
        self,
        claim_text: str,
        cited_records: list[dict[str, Any]],
    ) -> str | None:
        """Check claim consistency against cited evidence excerpts via pluggable verifier + numeric grounding."""
        if not cited_records:
            return None

        if (
            getattr(self, "_claim_verification_orchestrator", None) is not None
            and bool(getattr(self._config, "nli_verifier_enabled", False))
        ):
            # Single-track guard: defer to batch ClaimVerificationOrchestrator when active.
            return None

        if self._claim_verifier is not None:
            verdict = self._claim_verifier(claim_text, cited_records)
            if verdict is not None:
                supported, reason = verdict
                if not supported:
                    return reason or truncate_text(claim_text, 120)

        return self._numeric_fallback_only(claim_text, cited_records)

    @staticmethod
    def _numeric_fallback_only(
        claim_text: str,
        cited_records: list[dict[str, Any]],
    ) -> str | None:
        if not cited_records:
            return None
        corpus = " ".join(_evidence_body_text(rec) for rec in cited_records).strip()
        # Only run numeric grounding when cited records actually carry substantive excerpt text.
        body_without_titles = " ".join(
            str(rec.get(field) or "").strip()
            for rec in cited_records
            for field in ("snippet", "raw_content", "full_content", "content")
        ).strip()
        if len(body_without_titles) < 20:
            return None

        salient_numbers = _extract_salient_numbers(claim_text)
        if not salient_numbers:
            return None

        corpus_lower = corpus.lower()
        unsupported = [
            token
            for token in salient_numbers
            if not _number_token_supported_in_text(token, corpus_lower)
        ]
        if unsupported and len(unsupported) == len(salient_numbers):
            return (
                f"结论中的关键数值/年份 ({', '.join(unsupported[:3])}) "
                f"未在所引证据摘录中出现：{truncate_text(claim_text, 100)}"
            )
        return None

    @staticmethod
    def _extract_salient_numbers(text: str) -> list[float]:
        ignored_ints = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 24.0, 52.0, 60.0, 100.0, 365.0, 1000.0}
        values: list[float] = []
        for token in re.findall(r"(?<![A-Za-z0-9_])(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)", str(text or "")):
            normalized = token.replace(",", "")
            try:
                num = float(normalized)
            except ValueError:
                continue
            if num in ignored_ints:
                continue
            if num.is_integer() and 0 <= num < 10:
                continue
            values.append(num)
        return values

    def _verify_computation_lineage(
        self,
        compute_record: dict[str, Any],
        evidence_by_id: dict[str, dict[str, Any]],
    ) -> str | None:
        derived_ids = [
            str(sid).strip().strip("[]")
            for sid in (compute_record.get("derived_from_source_ids") or [])
            if str(sid).strip()
        ]
        if not derived_ids:
            return "未声明上游支撑来源 derived_from_source_ids"

        upstream_records = [evidence_by_id[sid] for sid in derived_ids if sid in evidence_by_id]
        if not upstream_records:
            return f"引用的上游来源 {', '.join(derived_ids)} 不在当前任务证据目录中"

        artifact = compute_record.get("computation_artifact")
        if not isinstance(artifact, dict):
            return "缺少 computation_artifact 计算产物记录"

        input_corpus_parts = [str(artifact.get("code") or "")]
        if isinstance(artifact.get("tables"), list):
            input_corpus_parts.append(str(artifact.get("tables")))
        salient_inputs = self._extract_salient_numbers("\n".join(input_corpus_parts))
        if len(salient_inputs) < 2:
            return None

        upstream_parts: list[str] = []
        for rec in upstream_records:
            upstream_parts.append(str(rec.get("title") or ""))
            upstream_parts.append(str(rec.get("snippet") or ""))
            upstream_parts.append(str(rec.get("raw_content") or ""))
            upstream_parts.append(str(rec.get("full_content") or ""))
            for block in rec.get("multimodal_blocks") or []:
                if isinstance(block, dict):
                    upstream_parts.append(str(block.get("markdown") or ""))
                    upstream_parts.append(str(block.get("text") or ""))
                    upstream_parts.append(str(block.get("summary") or ""))
                    if isinstance(block.get("structured_payload"), dict):
                        upstream_parts.append(str(block.get("structured_payload")))
        upstream_nums = [
            float(tok.replace(",", ""))
            for tok in re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)", "\n".join(upstream_parts))
            if tok not in {".", ""}
        ]
        if not upstream_nums:
            return "上游来源缺乏可核对的数值输入"

        missing: list[float] = []
        for val in salient_inputs:
            matched = any(
                abs(val - u) <= max(1e-6, 0.01 * max(abs(val), abs(u)))
                for u in upstream_nums
            )
            if not matched:
                missing.append(val)
        if len(missing) / len(salient_inputs) > 0.30:
            sample = ", ".join(f"{v:g}" for v in missing[:4])
            return f"超过 30% 的计算关键输入数字未出现在上游来源中: {sample}"
        return None

    @staticmethod
    def _normalize_repair_check(check: str) -> str | None:
        normalized = str(check or "").strip().lower()
        if normalized in {"missing_angle", "weak_evidence", "stale_evidence", "invalid_citation"}:
            return normalized
        if normalized == "claim_contradicted":
            return "invalid_citation"
        if normalized == "timeline_inconsistency":
            return "stale_evidence"
        if normalized in {
            "low_source_diversity",
            "low_quality_mix",
            "missing_citation",
            "claim_evidence_mismatch",
            "numeric_audit_mismatch",
            "comparison_asymmetry",
            "sandbox_input_ungrounded",
            "cross_source_conflict",
        }:
            return "weak_evidence"
        return None

    def _build_repair_candidates(self, issues: list[ReviewIssue]) -> list[dict[str, Any]]:
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int | None, str, str]] = set()

        for issue in sorted(issues, key=lambda item: severity_rank.get(item.severity, 99)):
            normalized_check = self._normalize_repair_check(issue.check)
            if normalized_check is None:
                continue

            key = (issue.task_id, normalized_check, issue.message)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "task_id": issue.task_id,
                    "severity": issue.severity,
                    "check": normalized_check,
                    "message": issue.message,
                    "source_ids": list(issue.source_ids),
                    "origin_check": issue.check,
                }
            )

        return candidates

    def _apply_review_result(
        self,
        state: SummaryState,
        *,
        issues: list[ReviewIssue],
        claims_by_task: dict[int, list[dict[str, Any]]],
        llm_summary: dict[str, Any],
        executed_verifiers: list[str] | None = None,
        verifier_issue_count: int = 0,
        claim_verification_summary: dict[str, Any] | None = None,
        evidence_conflict_count: int = 0,
        unresolved_conflict_count: int = 0,
    ) -> dict[str, Any]:
        issues_by_task: dict[int, list[dict[str, Any]]] = {}
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            if issue.task_id is not None:
                issues_by_task.setdefault(issue.task_id, []).append(issue.to_dict())

        for task in state.todo_items:
            task.claims = claims_by_task.get(task.id, [])
            task.review_issues = issues_by_task.get(task.id, [])
            if task.review_issues:
                task.review_status = (
                    "blocked"
                    if any(issue.get("severity") == "high" for issue in task.review_issues)
                    else "warning"
                )
                for issue in task.review_issues:
                    notice = f"[review] {issue['message']}"
                    if notice not in task.notices:
                        task.notices.append(notice)
            elif task.status == "completed":
                task.review_status = "passed"
            else:
                task.review_status = "skipped"

        overall_status = str(llm_summary.get("overall_status") or "").strip().lower()
        if overall_status not in {"passed", "warning", "blocked"}:
            if severity_counts["high"] > 0:
                overall_status = "blocked"
            elif sum(severity_counts.values()) > 0:
                overall_status = "warning"
            else:
                overall_status = "passed"

        reason = str(llm_summary.get("reason") or "").strip()
        if not reason:
            if overall_status == "blocked":
                reason = "审查阶段发现高风险证据问题，报告需要明确保守表述。"
            elif overall_status == "warning":
                reason = "审查阶段发现中低风险问题，报告已保守吸收这些限制。"
            else:
                reason = "审查阶段未发现明显证据问题。"

        summary = {
            "overall_status": overall_status,
            "reason": reason,
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "issues": [issue.to_dict() for issue in issues],
            "repair_candidates": self._build_repair_candidates(issues),
            "verifiers_executed": list(executed_verifiers or []),
            "verifier_issue_count": int(verifier_issue_count or 0),
        }
        if claim_verification_summary:
            summary["claim_verification_summary"] = dict(claim_verification_summary)
            summary["evidence_conflict_count"] = int(evidence_conflict_count or 0)
            summary["unresolved_conflict_count"] = int(unresolved_conflict_count or 0)
        state.review_summary = summary
        state.review_completed = True
        return summary
