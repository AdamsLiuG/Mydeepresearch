"""Review-driven report repair helpers for the deep research workflow."""

from __future__ import annotations

from typing import Any

from metrics import RequestTrace
from models import SummaryState, TodoItem


class RepairOrchestratorMixin:
    """Select review repair candidates and plan focused repair tasks."""

    def _select_report_repair_candidates(self, state: SummaryState) -> list[dict[str, Any]]:
        summary = state.review_summary or {}
        raw_candidates = summary.get("repair_candidates")
        if not isinstance(raw_candidates, list):
            return []

        severity_rank = {"high": 0, "medium": 1, "low": 2}
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int | None, str, str]] = set()

        for item in sorted(
            [candidate for candidate in raw_candidates if isinstance(candidate, dict)],
            key=lambda candidate: severity_rank.get(str(candidate.get("severity") or "").strip().lower(), 99),
        ):
            severity = str(item.get("severity") or "").strip().lower()
            if severity not in {"high", "medium"}:
                continue

            check = str(item.get("check") or "").strip()
            message = str(item.get("message") or "").strip()
            task_id_value = item.get("task_id")
            task_id = None
            if str(task_id_value or "").strip():
                try:
                    task_id = int(task_id_value)
                except (TypeError, ValueError):
                    task_id = None

            key = (task_id, check, message)
            if not message or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "task_id": task_id,
                    "severity": severity,
                    "check": check,
                    "message": message,
                    "source_ids": [
                        str(source_id).strip()
                        for source_id in item.get("source_ids") or []
                        if str(source_id).strip()
                    ],
                }
            )
        return candidates

    def _prepare_report_repair_tasks(
        self,
        state: SummaryState,
        observer: RequestTrace,
    ) -> tuple[list[TodoItem], str | None, list[dict[str, Any]]]:
        if not self.config.report_repair_enabled:
            return [], None, []
        if state.report_repair_completed:
            return [], None, []
        if state.report_repair_cycles >= max(int(self.config.report_repair_max_cycles or 1), 1):
            return [], "报告修补循环预算已耗尽，直接输出保守报告。", []

        repair_candidates = self._select_report_repair_candidates(state)
        if not repair_candidates:
            return [], None, []

        remaining_budget = self._remaining_task_budget(state)
        if remaining_budget <= 0:
            observer.record_report_repair(added_tasks=0, triggered=True)
            observer.record_degraded("report_repair_budget_exhausted")
            return [], "审查发现证据缺口，但任务预算已满，直接输出保守报告。", repair_candidates

        max_repair_tasks = min(
            remaining_budget,
            max(int(self.config.report_repair_max_tasks or 1), 1),
        )
        repair_tasks = self.planner.plan_repair_tasks(
            state,
            repair_candidates=repair_candidates,
            existing_tasks=list(state.todo_items),
            max_additional_tasks=max_repair_tasks,
            observer=observer,
            historical_memory_context=self._planning_memory_context(state, observer),
        )
        observer.record_report_repair(added_tasks=len(repair_tasks), triggered=True)

        if repair_tasks:
            state.todo_items.extend(repair_tasks)
            notice = f"审查阶段识别到证据缺口，新增 {len(repair_tasks)} 个定向修补任务。"
        else:
            notice = "审查阶段识别到证据缺口，但未生成有效修补任务，直接输出保守报告。"

        return repair_tasks, notice, repair_candidates
