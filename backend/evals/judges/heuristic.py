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
        }
