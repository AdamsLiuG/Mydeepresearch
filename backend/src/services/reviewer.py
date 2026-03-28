"""Review-stage helpers for claim grounding and evidence quality checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace
from models import SummaryState, TodoItem
from prompts import get_current_date
from services.evidence import extract_citation_ids, format_evidence_sources
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
    "2025",
    "2026",
)


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


class ReviewService:
    """Run deterministic and optional LLM-assisted review checks."""

    def __init__(
        self,
        review_agent: ToolAwareSimpleAgent | None,
        config: Configuration,
    ) -> None:
        self._agent = review_agent
        self._config = config

    def review_request(
        self,
        state: SummaryState,
        *,
        observer: RequestTrace | None = None,
    ) -> dict[str, Any]:
        """Review task summaries and evidence before final report generation."""

        issues, claims_by_task = self._rule_based_review(state)
        llm_summary: dict[str, Any] = {}

        if self._config.review_agent_enabled and self._agent is not None:
            prompt = self._build_prompt(state, issues)
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
            else:
                if observer:
                    observer.record_llm_call(
                        success=True,
                        prompt_text=prompt,
                        completion_text=response,
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
        )
        return summary

    def _build_prompt(
        self,
        state: SummaryState,
        issues: list[ReviewIssue],
    ) -> str:
        task_blocks: list[str] = []
        for task in state.todo_items:
            sources = format_evidence_sources(task.evidence_items) or "- 暂无来源"
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
                f"- 规则审查结果：\n{issue_text}\n"
            )

        return (
            f"当前日期：{get_current_date()}\n"
            f"研究主题：{state.research_topic}\n"
            "请基于以下任务结果补充指出仍然存在的证据问题。"
            "你可以调用 evidence_lookup 或 fetch_page，但不要重写报告正文，也不要重复输出规则已明确指出的问题。\n\n"
            f"{''.join(task_blocks)}"
        )

    def _rule_based_review(self, state: SummaryState) -> tuple[list[ReviewIssue], dict[int, list[dict[str, Any]]]]:
        issues: list[ReviewIssue] = []
        claims_by_task: dict[int, list[dict[str, Any]]] = {}

        for task in state.todo_items:
            task_issues, task_claims = self._review_task(state, task)
            issues.extend(task_issues)
            claims_by_task[task.id] = task_claims

        return issues, claims_by_task

    def _review_task(
        self,
        state: SummaryState,
        task: TodoItem,
    ) -> tuple[list[ReviewIssue], list[dict[str, Any]]]:
        issues: list[ReviewIssue] = []
        claims: list[dict[str, Any]] = []

        evidence_items = list(task.evidence_items or [])
        valid_ids = {
            str(item.get("source_id") or "").strip()
            for item in evidence_items
            if str(item.get("source_id") or "").strip()
        }
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
            invalid_source_ids = [source_id for source_id in citation_ids if source_id not in valid_ids]
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

            claims.append(
                {
                    "text": claim_text.strip(),
                    "source_ids": citation_ids,
                    "support_status": support_status,
                }
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
        return any(keyword in combined for keyword in _FRESHNESS_PATTERNS)

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

    def _apply_review_result(
        self,
        state: SummaryState,
        *,
        issues: list[ReviewIssue],
        claims_by_task: dict[int, list[dict[str, Any]]],
        llm_summary: dict[str, Any],
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
        }
        state.review_summary = summary
        state.review_completed = True
        return summary
