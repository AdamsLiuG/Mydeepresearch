"""Deterministic executable verification skills (whitelisted Python registry)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

REGISTERED_VERIFIER_IDS: tuple[str, ...] = (
    "numeric-audit",
    "timeline-audit",
    "comparison-matrix",
)

_GROWTH_ARITHMETIC_PATTERN = re.compile(
    r"(?:从|由)\s*(\d+(?:\.\d+)?)\s*(?:万|亿|%|元|美元|倍|ms|s|GB|MB|TB|K|M|B)?\s*"
    r"(?:增长|增加|提升|上升|下降|降低|减少)(?:到|至)\s*(\d+(?:\.\d+)?)\s*"
    r"(?:万|亿|%|元|美元|倍|ms|s|GB|MB|TB|K|M|B)?[^。；\n]{0,40}?"
    r"(?:增幅|降幅|增长率|下降幅度|变化|增长|下降)[^\d]{0,8}(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_SHARE_SENTENCE_HINT_PATTERN = re.compile(r"(?:占比|份额|比重|share)", re.IGNORECASE)
_PERCENT_TOKEN_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_YEAR_TOKEN_PATTERN = re.compile(r"\b(20\d{2})\b")
_RECENCY_HINT_PATTERN = re.compile(
    r"(?:最新|近期|最近|今年|current|latest|recent)",
    re.IGNORECASE,
)
_COMPARISON_HINT_PATTERN = re.compile(
    r"(?:\bvs\.?\b|对比|比较|选型|竞品|compare|comparison|versus|trade-?off)",
    re.IGNORECASE,
)


def _source_ids_for_task(task: Any) -> list[str]:
    return sorted(
        {
            str(item.get("source_id") or "").strip()
            for item in (getattr(task, "evidence_items", None) or [])
            if isinstance(item, dict) and str(item.get("source_id") or "").strip()
        }
    )


def _numeric_audit(state: Any, config: Any) -> list[dict[str, Any]]:
    """Verify arithmetic growth claims and percentage share sums per completed task."""

    del config
    issues: list[dict[str, Any]] = []
    for task in getattr(state, "todo_items", []) or []:
        if getattr(task, "status", "") != "completed":
            continue
        texts: list[str] = []
        summary = str(getattr(task, "summary", "") or "").strip()
        if summary:
            texts.append(summary)
        for claim in getattr(task, "claims", None) or []:
            if isinstance(claim, dict) and str(claim.get("text") or "").strip():
                texts.append(str(claim.get("text")).strip())
        combined = "\n".join(texts)
        if not combined:
            continue

        source_ids = _source_ids_for_task(task)
        for match in _GROWTH_ARITHMETIC_PATTERN.finditer(combined):
            base_val = float(match.group(1))
            target_val = float(match.group(2))
            claimed_pct = float(match.group(3))
            if base_val <= 0:
                continue
            expected_pct = abs(target_val - base_val) / base_val * 100.0
            if abs(expected_pct - claimed_pct) > 5.0:
                issues.append(
                    {
                        "task_id": task.id,
                        "severity": "high",
                        "check": "numeric_audit_mismatch",
                        "message": (
                            f"任务 {task.id} 存在算术增幅自洽异常：从 {base_val:g} 到 {target_val:g} "
                            f"实际变化幅度约 {expected_pct:.1f}%，但文中表述为 {claimed_pct:g}%。"
                        ),
                        "source_ids": source_ids,
                        "origin": "skill_verifier:numeric-audit",
                    }
                )
                break

        for line in re.split(r"[。\n；;]+", combined):
            if not _SHARE_SENTENCE_HINT_PATTERN.search(line):
                continue
            pcts = [float(val) for val in _PERCENT_TOKEN_PATTERN.findall(line)]
            if len(pcts) >= 3 and sum(pcts) > 105.0:
                issues.append(
                    {
                        "task_id": task.id,
                        "severity": "medium",
                        "check": "numeric_audit_mismatch",
                        "message": (
                            f"任务 {task.id} 单句中多组份额/占比合计达 {sum(pcts):.1f}%（超过 105%），"
                            "请核实统计口径是否重叠或存在计算误差。"
                        ),
                        "source_ids": source_ids,
                        "origin": "skill_verifier:numeric-audit",
                    }
                )
                break
    return issues


def _timeline_audit(state: Any, config: Any) -> list[dict[str, Any]]:
    """Flag recency assertions backed only by stale dated evidence items."""

    del config
    current_year = datetime.now(timezone.utc).year
    stale_ceiling = current_year - 2
    recent_floor = current_year - 1
    issues: list[dict[str, Any]] = []

    for task in getattr(state, "todo_items", []) or []:
        if getattr(task, "status", "") != "completed":
            continue
        summary = str(getattr(task, "summary", "") or "").strip()
        if not summary:
            continue

        summary_years = [int(y) for y in _YEAR_TOKEN_PATTERN.findall(summary)]
        claims_recent = bool(_RECENCY_HINT_PATTERN.search(summary)) or any(
            year >= recent_floor for year in summary_years
        )
        if not claims_recent:
            continue

        evidence_years: list[int] = []
        for item in getattr(task, "evidence_items", None) or []:
            if not isinstance(item, dict):
                continue
            published = str(item.get("published_at") or "").strip()
            for y in _YEAR_TOKEN_PATTERN.findall(published):
                evidence_years.append(int(y))
            corpus = " ".join(
                str(item.get(k) or "")
                for k in ("title", "snippet", "raw_content", "full_content")
            )
            for y in _YEAR_TOKEN_PATTERN.findall(corpus):
                year_int = int(y)
                if 2000 <= year_int <= current_year + 1:
                    evidence_years.append(year_int)

        if evidence_years and max(evidence_years) <= stale_ceiling:
            issues.append(
                {
                    "task_id": task.id,
                    "severity": "medium",
                    "check": "timeline_inconsistency",
                    "message": (
                        f"任务 {task.id} 涉及近期/年份表述（{max(summary_years) if summary_years else current_year}），"
                        f"但所引证据标注年份均停留在 {max(evidence_years)} 年及以前，存在时间线滞后风险。"
                    ),
                    "source_ids": _source_ids_for_task(task),
                    "origin": "skill_verifier:timeline-audit",
                }
            )
    return issues


def _comparison_audit(state: Any, config: Any) -> list[dict[str, Any]]:
    """Detect asymmetric evidence coverage across compared options/subtasks."""

    del config
    completed = [
        task
        for task in (getattr(state, "todo_items", []) or [])
        if getattr(task, "status", "") == "completed"
    ]
    if len(completed) < 2:
        return []

    topic_text = str(getattr(state, "research_topic", "") or "")
    combined_meta = " ".join(
        [
            topic_text,
            *(f"{getattr(t, 'title', '')} {getattr(t, 'intent', '')}" for t in completed),
        ]
    )
    if not _COMPARISON_HINT_PATTERN.search(combined_meta):
        return []

    counts = [(task, len(getattr(task, "evidence_items", None) or [])) for task in completed]
    total_evidence = sum(count for _, count in counts)
    if total_evidence == 0:
        return []

    max_count = max(count for _, count in counts)
    issues: list[dict[str, Any]] = []
    for task, count in counts:
        is_empty_starved = count == 0 and max_count >= 2
        is_dominant_skew = max_count / total_evidence > 0.80 and count < max_count
        if is_empty_starved or is_dominant_skew:
            issues.append(
                {
                    "task_id": task.id,
                    "severity": "medium",
                    "check": "comparison_asymmetry",
                    "message": (
                        f"横向对比存在证据失衡：任务 {task.id}（{getattr(task, 'title', '')}）仅 {count} 条来源，"
                        f"而对照组最高 {max_count} 条，请补充该维度证据或在矩阵中显式标明证据缺口。"
                    ),
                    "source_ids": _source_ids_for_task(task),
                    "origin": "skill_verifier:comparison-matrix",
                }
            )
    return issues


_VERIFIER_REGISTRY: dict[str, Callable[[Any, Any], list[dict[str, Any]]]] = {
    "numeric-audit": _numeric_audit,
    "timeline-audit": _timeline_audit,
    "comparison-matrix": _comparison_audit,
}


def run_skill_verifiers(
    state: Any,
    config: Any,
    verifier_ids: Sequence[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run all whitelisted skill verifiers safely, returning (issue_dicts, executed_ids)."""

    collected: list[dict[str, Any]] = []
    executed: list[str] = []
    seen: set[str] = set()

    for raw_id in verifier_ids or ():
        vid = str(raw_id or "").strip().lower()
        if not vid or vid in seen:
            continue
        fn = _VERIFIER_REGISTRY.get(vid)
        if fn is None:
            logger.warning("Skipping unknown skill verifier id=%r", vid)
            continue
        seen.add(vid)
        try:
            produced = fn(state, config)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.warning("Skill verifier '%s' crashed: %s", vid, exc)
            continue
        executed.append(vid)
        for item in produced or ():
            if isinstance(item, dict):
                collected.append(item)

    return collected, executed
