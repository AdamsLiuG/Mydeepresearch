"""Human-in-the-Loop (HITL) helpers for clarification, editable plans, and section drill-down."""

from __future__ import annotations

import re
from typing import Any

from config import Configuration
from models import TodoItem
from services.dag_scheduler import normalize_depends_on, sanitize_task_dependencies
from services.skill_pack import SkillPackService
from services.text_processing import CITATION_PATTERN as _CITATION_TOKEN_PATTERN

_BROAD_COMPARISON_MARKERS = (
    "对比",
    "比较",
    "选型",
    "vs",
    "compare",
    "comparison",
    "trade-off",
    "tradeoff",
)
_TIMEFRAME_MARKERS = (
    "202",
    "最新",
    "近期",
    "今年",
    "去年",
    "current",
    "latest",
    "recent",
)
_SPECIFIC_SCOPE_MARKERS = (
    "生产",
    "企业",
    "本地",
    "开源",
    "闭源",
    "定价",
    "成本",
    "合规",
    "监管",
    "论文",
    "benchmark",
    "production",
    "pricing",
    "compliance",
)


class HitlService:
    """Stateless HITL orchestration utilities for pre-flight clarify/plan and post-report drill-down."""

    @staticmethod
    def estimate_duration_seconds(task_count: int, config: Configuration) -> int:
        count = max(1, int(task_count or 1))
        rounds = max(1, int(getattr(config, "task_react_max_rounds", 2) or 1))
        concurrency = max(1, int(getattr(config, "task_summary_max_concurrency", 1) or 1))
        base_per_task = 18 + (rounds * 8)
        parallel_factor = min(count, max(1, concurrency + 1))
        return max(15, int(round((count / parallel_factor) * base_per_task + 20)))

    @classmethod
    def clarify_topic(
        cls,
        topic: str,
        *,
        config: Configuration,
        skill_pack_service: SkillPackService | None = None,
    ) -> dict[str, Any]:
        """Determine whether a research topic is underspecified and propose up to 3 clarifying questions."""

        cleaned = " ".join(str(topic or "").strip().split())
        if not cleaned:
            raise ValueError("research topic must not be empty")

        lowered = cleaned.lower()
        questions: list[str] = []

        # Heuristic ambiguity checks (zero LLM latency)
        char_len = len(cleaned)
        word_count = len(cleaned.split())
        is_very_short = char_len < 12 or (word_count <= 2 and char_len < 20)

        if is_very_short:
            questions.append("你希望重点关注哪几个核心角度（如技术原理、工程实践、成本对比或风险限制）？")

        has_comparison = any(token in lowered for token in _BROAD_COMPARISON_MARKERS)
        has_specific_scope = any(token in lowered for token in _SPECIFIC_SCOPE_MARKERS)
        has_timeframe = any(token in lowered for token in _TIMEFRAME_MARKERS)

        if not has_timeframe:
            questions.append("是否需要限定具体的时间范围或版本口径（例如近一年最新动态还是历史演进综述）？")

        if has_comparison and not has_specific_scope:
            questions.append("本次选型或对比的主要业务场景/约束条件是什么（如高并发低延迟、私有化部署或预算敏感）？")
        elif not is_very_short and not has_specific_scope and char_len < 24:
            questions.append("报告的目标读者与深度侧重是什么（决策摘要、架构选型评估还是学术文献梳理）？")

        questions = questions[:3]

        suggested_pack = "_neutral"
        if skill_pack_service is not None and getattr(skill_pack_service, "enabled", False):
            selection = skill_pack_service.select(cleaned)
            if selection is not None:
                suggested_pack = selection.pack.pack_id

        estimated_tasks = max(3, min(config.resolved_initial_plan_max_tasks(), 4))
        return {
            "topic": cleaned,
            "clarification_needed": bool(is_very_short or len(questions) >= 2),
            "questions": questions,
            "suggested_skill_pack": suggested_pack,
            "estimated_duration_seconds": cls.estimate_duration_seconds(estimated_tasks, config),
        }

    @staticmethod
    def validate_editable_tasks(
        raw_tasks: list[dict[str, Any]],
        *,
        max_tasks: int,
    ) -> list[dict[str, Any]]:
        """Validate user-edited plan tasks and normalize DAG dependencies."""

        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("plan must contain at least one task")

        cap = max(1, int(max_tasks or 7))
        if len(raw_tasks) > cap:
            raise ValueError(f"plan exceeds max_tasks limit ({cap})")

        todo_models: list[TodoItem] = []
        used_ids: set[int] = set()
        for idx, item in enumerate(raw_tasks, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"task at index {idx - 1} must be an object")
            title = str(item.get("title") or "").strip()
            intent = str(item.get("intent") or "").strip()
            query = str(item.get("query") or "").strip()
            if not title:
                raise ValueError(f"task at index {idx - 1} is missing title")
            if not query:
                query = f"{title} {intent}".strip()

            raw_id = item.get("id")
            try:
                task_id = int(raw_id) if raw_id is not None and not isinstance(raw_id, bool) else idx
            except (TypeError, ValueError):
                task_id = idx
            if task_id <= 0 or task_id in used_ids:
                task_id = idx
                while task_id in used_ids:
                    task_id += 1
            used_ids.add(task_id)

            todo_models.append(
                TodoItem(
                    id=task_id,
                    title=title,
                    intent=intent or "聚焦主题的关键问题",
                    query=query,
                    origin=str(item.get("origin") or "planned").strip() or "planned",
                    round=max(1, int(item.get("round") or 1)),
                    depends_on=normalize_depends_on(item.get("depends_on")),
                )
            )

        sanitize_task_dependencies(todo_models)
        return [
            {
                "id": t.id,
                "title": t.title,
                "intent": t.intent,
                "query": t.query,
                "status": "pending",
                "origin": t.origin,
                "round": t.round,
                "depends_on": list(t.depends_on),
            }
            for t in todo_models
        ]

    @staticmethod
    def sanitize_citations(text: str, valid_source_ids: set[str]) -> str:
        """Strip hallucinated [T{task}-S{seq}] tokens not present in the whitelist."""
        cleaned = str(text or "").strip()
        if not cleaned or not valid_source_ids:
            return cleaned

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            return match.group(0) if token in valid_source_ids else ""

        result = _CITATION_TOKEN_PATTERN.sub(_replace, cleaned)
        return re.sub(r"[ \t]{2,}", " ", result).strip()

    @staticmethod
    def merge_section_drilldown(
        report_markdown: str,
        *,
        section_title: str,
        addendum_markdown: str,
        new_references: list[str] | None = None,
    ) -> str:
        """Insert a drill-down addendum under the matching section or before references."""

        base = str(report_markdown or "").strip()
        addendum_body = str(addendum_markdown or "").strip()
        if not addendum_body:
            return base

        block = f"### 深入补充：{section_title}\n\n{addendum_body}"
        if not base:
            return block

        lines = base.splitlines()
        normalized_target = re.sub(r"\s+", "", section_title).casefold()
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

        insert_at: int | None = None
        matched_level: int | None = None
        references_idx: int | None = None

        for idx, line in enumerate(lines):
            match = heading_pattern.match(line.strip())
            if not match:
                continue
            level = len(match.group(1))
            title_text = match.group(2).strip()
            normalized_heading = re.sub(r"\s+", "", title_text).casefold()

            if references_idx is None and ("参考来源" in title_text or "references" in normalized_heading):
                references_idx = idx

            if matched_level is None:
                if normalized_target and (
                    normalized_target == normalized_heading
                    or normalized_target in normalized_heading
                    or normalized_heading in normalized_target
                ):
                    matched_level = level
            elif level <= matched_level:
                insert_at = idx
                break

        if matched_level is not None and insert_at is None:
            insert_at = references_idx if references_idx is not None else len(lines)
        elif insert_at is None:
            insert_at = references_idx if references_idx is not None else len(lines)

        before = "\n".join(lines[:insert_at]).rstrip()
        after = "\n".join(lines[insert_at:]).lstrip()
        parts = [part for part in (before, block, after) if part]
        merged = "\n\n".join(parts)

        refs = [ref.strip() for ref in (new_references or []) if str(ref or "").strip()]
        if refs:
            if references_idx is not None or "## 参考来源" in merged:
                merged = merged.rstrip() + "\n" + "\n".join(refs)
            else:
                merged = merged.rstrip() + "\n\n## 参考来源\n" + "\n".join(refs)

        return merged
