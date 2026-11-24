"""DAG dependency sanitization and multi-wave execution scheduler."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from models import TodoItem
from utils import truncate_text


def normalize_depends_on(raw_value: Any) -> list[int]:
    """Coerce raw dependency metadata into a sorted-order deduplicated list of positive ints."""

    if raw_value is None:
        return []
    if isinstance(raw_value, (int, str)):
        candidates = [raw_value]
    elif isinstance(raw_value, Iterable):
        candidates = list(raw_value)
    else:
        return []

    seen: set[int] = set()
    normalized: list[int] = []
    for item in candidates:
        if isinstance(item, bool):
            continue
        try:
            task_id = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if task_id <= 0 or task_id in seen:
            continue
        seen.add(task_id)
        normalized.append(task_id)
    return normalized


def sanitize_task_dependencies(tasks: list[TodoItem]) -> list[TodoItem]:
    """Remove invalid, self-referential, duplicate, or cyclic dependencies from tasks.

    Keeps the edge set strictly acyclic by processing tasks in order and dropping any edge
    that creates a cycle reachable in the current DAG.
    """

    valid_ids = {int(task.id) for task in tasks if int(task.id) > 0}
    accepted_edges: dict[int, list[int]] = {task_id: [] for task_id in valid_ids}

    def _can_reach(start_id: int, target_id: int, visited: set[int] | None = None) -> bool:
        if start_id == target_id:
            return True
        seen = visited if visited is not None else set()
        if start_id in seen:
            return False
        seen.add(start_id)
        for dep_id in accepted_edges.get(start_id, ()):
            if _can_reach(dep_id, target_id, seen):
                return True
        return False

    for task in tasks:
        task_id = int(task.id)
        raw_deps = normalize_depends_on(getattr(task, "depends_on", None))
        clean_deps: list[int] = []
        for dep_id in raw_deps:
            if dep_id == task_id or dep_id not in valid_ids:
                continue
            # Adding edge task_id -> dep_id creates a cycle iff dep_id can already reach task_id.
            if _can_reach(dep_id, task_id):
                continue
            clean_deps.append(dep_id)
            accepted_edges[task_id].append(dep_id)
        task.depends_on = clean_deps
    return tasks


def partition_into_waves(
    tasks: Sequence[TodoItem],
    *,
    completed_ids: set[int] | None = None,
) -> list[list[TodoItem]]:
    """Topologically partition pending tasks into parallelizable execution waves.

    When no pending task declares an active dependency, returns ``[list(tasks)]`` so
    non-DAG workflows incur zero behavioral change.
    """

    pending = list(tasks)
    if not pending:
        return []

    has_dependencies = any(bool(getattr(task, "depends_on", None)) for task in pending)
    if not has_dependencies:
        return [pending]

    pending_ids = {int(task.id) for task in pending}
    resolved_ids: set[int] = set(completed_ids or ())
    # Any dependency pointing to a task outside pending & completed (e.g. skipped/missing)
    # should not deadlock downstream tasks.
    known_active = pending_ids | resolved_ids

    deps_map: dict[int, set[int]] = {}
    for task in pending:
        raw = normalize_depends_on(getattr(task, "depends_on", None))
        deps_map[int(task.id)] = {
            dep_id for dep_id in raw if dep_id in known_active and dep_id != int(task.id)
        }

    waves: list[list[TodoItem]] = []
    remaining = list(pending)
    while remaining:
        ready = [
            task
            for task in remaining
            if deps_map.get(int(task.id), set()).issubset(resolved_ids)
        ]
        if not ready:
            # Defensive cycle-breaker in case unsanitized tasks are passed directly.
            ready = [remaining[0]]
        waves.append(ready)
        ready_ids = {int(task.id) for task in ready}
        resolved_ids.update(ready_ids)
        remaining = [task for task in remaining if int(task.id) not in ready_ids]

    return waves


def build_upstream_context(
    task: TodoItem,
    all_tasks: Sequence[TodoItem],
    *,
    max_chars: int = 800,
    fact_ledger: dict[str, Any] | Any | None = None,
    compact_enabled: bool = True,
) -> str:
    """Build concise upstream dependency findings injected into downstream wave execution."""

    dep_ids = normalize_depends_on(getattr(task, "depends_on", None))
    if not dep_ids:
        return ""

    entries_by_task: dict[int, list[str]] = {}
    if compact_enabled and fact_ledger:
        raw_entries = (
            fact_ledger.get("entries")
            if isinstance(fact_ledger, dict)
            else getattr(fact_ledger, "entries", None)
        )
        for item in raw_entries or ():
            if isinstance(item, dict):
                tid = int(item.get("task_id") or 0)
                cid = str(item.get("claim_id") or "").strip()
                stmt = str(item.get("statement") or "").strip()
                cites = ",".join(str(s) for s in (item.get("source_ids") or []) if s)
                conflict = str(item.get("conflict_group_id") or "").strip()
            else:
                tid = int(getattr(item, "task_id", 0) or 0)
                cid = str(getattr(item, "claim_id", "") or "").strip()
                stmt = str(getattr(item, "statement", "") or "").strip()
                cites = ",".join(str(s) for s in (getattr(item, "source_ids", ()) or ()) if s)
                conflict = str(getattr(item, "conflict_group_id", "") or "").strip()
            if tid and stmt:
                tag = f" [CONFLICT:{conflict}]" if conflict else ""
                entries_by_task.setdefault(tid, []).append(
                    f"  - {cid}{tag} ({cites}): {truncate_text(stmt, 140)}"
                )

    by_id = {int(item.id): item for item in all_tasks}
    blocks: list[str] = []
    for dep_id in dep_ids:
        upstream = by_id.get(dep_id)
        if upstream is None:
            continue
        summary_text = str(upstream.summary or "").strip()
        if not summary_text and isinstance(upstream.summary_payload, dict):
            summary_text = str(upstream.summary_payload.get("executive_summary") or "").strip()
        ledger_lines = entries_by_task.get(int(upstream.id), [])
        if not summary_text and not ledger_lines:
            continue
        if summary_text:
            summary_cap = 180 if ledger_lines else 280
            blocks.append(
                f"- 前置任务 {upstream.id}（{upstream.title}）结论：{truncate_text(summary_text, summary_cap)}"
            )
        else:
            blocks.append(f"- 前置任务 {upstream.id}（{upstream.title}）结构化事实：")
        if ledger_lines:
            blocks.extend(ledger_lines[:4])

    if not blocks:
        return ""
    joined = "\n".join(blocks)
    return truncate_text(joined, max(120, int(max_chars or 800)))
