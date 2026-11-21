"""Heuristic metrics used by the offline benchmark runner."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from evals.judges.base import Judge
from evals.schema import BenchmarkCase

_CITATION_PATTERN = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)
_SOURCE_ID_PATTERN = re.compile(r"\[(T\d+-S\d+)\]")
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _contains_section(report_text: str, section: str) -> bool:
    if not section:
        return False

    normalized_section = _normalize(section)
    if not report_text or not normalized_section:
        return False

    for line in report_text.splitlines():
        heading = _HEADING_PATTERN.match(line)
        if not heading:
            continue
        title = _normalize(heading.group(1))
        if title == normalized_section or normalized_section in title:
            return True
    return False


def _extract_reference_section(report_text: str) -> str:
    lines = report_text.splitlines()
    collected: list[str] = []
    in_section = False
    for line in lines:
        heading = _HEADING_PATTERN.match(line)
        if heading:
            title = _normalize(heading.group(1))
            if title == "参考来源":
                in_section = True
                continue
            if in_section:
                break
        if in_section:
            collected.append(line)
    return "\n".join(collected).strip()


def _collect_bullet_lines(report_text: str, section_titles: tuple[str, ...]) -> list[str]:
    lines = report_text.splitlines()
    bullets: list[str] = []
    current_section = ""
    for line in lines:
        heading = _HEADING_PATTERN.match(line)
        if heading:
            current_section = _normalize(heading.group(1))
            continue
        if current_section not in section_titles:
            continue
        stripped = line.strip()
        if stripped.startswith(("-", "*")) or re.match(r"^\d+\.", stripped):
            bullets.append(stripped)
    return bullets


def _trace_int(trace: Mapping[str, Any], key: str) -> int:
    try:
        return max(0, int(trace.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _memory_metrics(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Extract L1 memory exposure metrics already present in the request trace.

    ``*_prompt_injections`` counts blocks that were *rendered*. Under shadow mode the
    agent withholds them, so the effective injection count is reported separately to
    keep "would have injected" distinguishable from "did inject".
    """

    shadow_mode = bool(trace.get("memory_shadow_mode"))
    rendered_injections = _trace_int(trace, "note_memory_prompt_injections") + _trace_int(
        trace,
        "strategy_memory_prompt_injections",
    )
    match_kinds = [
        str(value).strip()
        for value in list(trace.get("note_memory_last_match_types") or [])
        + list(trace.get("strategy_memory_last_match_kinds") or [])
        if str(value).strip()
    ]

    return {
        "memory_query_count": _trace_int(trace, "note_memory_queries")
        + _trace_int(trace, "strategy_memory_queries"),
        "memory_hit_count": _trace_int(trace, "note_memory_hits")
        + _trace_int(trace, "strategy_memory_hits"),
        "memory_rendered_injection_count": rendered_injections,
        "memory_effective_injection_count": 0 if shadow_mode else rendered_injections,
        "note_memory_hit_count": _trace_int(trace, "note_memory_hits"),
        "strategy_memory_hit_count": _trace_int(trace, "strategy_memory_hits"),
        "memory_shadow_mode": shadow_mode,
        "memory_shadow_suppressions": _trace_int(trace, "memory_shadow_suppressions"),
        "memory_match_kinds": sorted(set(match_kinds)),
    }


class HeuristicJudge(Judge):
    """Deterministic text-based judge for early benchmark iterations."""

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        report_markdown: str,
        todo_items: Sequence[Any],
        trace_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        report_text = report_markdown or ""
        normalized_report = _normalize(report_text)
        trace = dict(trace_snapshot or {})

        matched_sections = [
            section for section in case.expected_sections if _contains_section(report_text, section)
        ]
        matched_keywords = [
            keyword for keyword in case.expected_keywords if _normalize(keyword) in normalized_report
        ]
        unique_citations = sorted(set(_CITATION_PATTERN.findall(report_text)))
        cited_source_ids = sorted(set(_SOURCE_ID_PATTERN.findall(report_text)))
        reference_section = _extract_reference_section(report_text)
        referenced_source_ids = sorted(set(_SOURCE_ID_PATTERN.findall(reference_section)))
        grounded_bullets = _collect_bullet_lines(
            report_text,
            section_titles=("核心洞见", "证据与数据", "风险与挑战"),
        )
        grounded_with_citations = [
            bullet for bullet in grounded_bullets if _SOURCE_ID_PATTERN.search(bullet)
        ]

        total_sections = len(case.expected_sections)
        total_keywords = len(case.expected_keywords)

        degraded_flag = bool(
            trace.get("degraded")
            or trace.get("fallback_triggered")
            or trace.get("status") == "partial_success"
        )

        return {
            "report_generated": bool(report_text.strip()),
            "degraded_flag": degraded_flag,
            "section_completeness": round(
                len(matched_sections) / total_sections,
                4,
            )
            if total_sections
            else 1.0,
            "keyword_coverage": round(
                len(matched_keywords) / total_keywords,
                4,
            )
            if total_keywords
            else 1.0,
            "citation_count": len(unique_citations),
            "citation_marker_count": len(cited_source_ids),
            "reference_section_present": bool(reference_section),
            "reference_match_rate": round(
                len(set(cited_source_ids) & set(referenced_source_ids)) / len(cited_source_ids),
                4,
            )
            if cited_source_ids
            else 0.0,
            "grounded_bullet_ratio": round(
                len(grounded_with_citations) / len(grounded_bullets),
                4,
            )
            if grounded_bullets
            else 0.0,
            "total_latency_ms": round(float(trace.get("elapsed_ms") or 0.0), 2),
            "estimated_cost": round(float(trace.get("estimated_cost") or 0.0), 6),
            "matched_sections": matched_sections,
            "missing_sections": [
                section for section in case.expected_sections if section not in matched_sections
            ],
            "matched_keywords": matched_keywords,
            "missing_keywords": [
                keyword for keyword in case.expected_keywords if keyword not in matched_keywords
            ],
            "todo_item_count": len(todo_items),
            "completed_task_count": sum(
                1 for item in todo_items if getattr(item, "status", None) == "completed"
            ),
            **_memory_metrics(trace),
        }
