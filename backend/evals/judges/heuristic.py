"""Heuristic metrics used by the offline benchmark runner."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from evals.judges.base import Judge
from evals.schema import BenchmarkCase

_CITATION_PATTERN = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _contains_section(report_text: str, section: str) -> bool:
    if not section:
        return False

    normalized_report = _normalize(report_text)
    normalized_section = _normalize(section)
    if not normalized_report or not normalized_section:
        return False

    heading_markers = (
        f"# {normalized_section}",
        f"## {normalized_section}",
        f"### {normalized_section}",
        normalized_section,
    )
    return any(marker in normalized_report for marker in heading_markers)


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
        }
