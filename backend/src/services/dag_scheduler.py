"""DAG dependency sanitization and multi-wave execution scheduler."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from models import TodoItem
from services.fact_ledger import FactLedger, render_evidence_index_card
from services.text_processing import truncate_cited_text


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
    fact_ledger: dict[str, Any] | FactLedger | None = None,
    evidence_index_cards: Sequence[dict[str, Any]] = (),
    compact_enabled: bool = True,
) -> str:
    """Render dependency-local facts and pointer cards within the supplied budget."""
    dep_ids = normalize_depends_on(task.depends_on)
    ledger = fact_ledger if isinstance(fact_ledger, FactLedger) else FactLedger.from_dict(fact_ledger)
    by_id = {int(item.id): item for item in all_tasks}
    lines: list[str] = []
    remaining = max_chars

    def append(line: str) -> bool:
        nonlocal remaining
        cost = len(line) + bool(lines)
        if line and cost <= remaining:
            lines.append(line)
            remaining -= cost
            return True
        return False

    for dep_id in dep_ids:
        upstream = by_id.get(dep_id)
        if upstream is None or upstream.status != "completed":
            continue
        rows = [
            entry for entry in ledger.entries
            if compact_enabled and (entry.task_id == dep_id or dep_id in entry.supporting_task_ids)
        ]
        rows.sort(key=lambda entry: not bool(entry.conflict_group_id))
        cards = [
            card for card in evidence_index_cards if compact_enabled and card.get("task_id") == dep_id
        ]
        summary = str(upstream.summary or (upstream.summary_payload or {}).get("executive_summary") or "").strip()
        if not (rows or cards or summary):
            continue
        header = f"- 前置任务 {dep_id}（{' '.join(upstream.title.split())[:80]}）结论："
        if not (rows or cards):
            append(truncate_cited_text(f"{header}{summary}", min(len(header) + 280, remaining - bool(lines))))
            continue
        if not append(header):
            continue
        # Guarantee room for a compact pointer before attempting full fact lines.
        if cards:
            append(render_evidence_index_card(cards[0]))
        for entry in rows[:4]:
            append(entry.to_compact_line())
        for card in cards[1:3]:
            append(render_evidence_index_card(card))
    return "\n".join(lines)
