"""DAG dependency sanitization and multi-wave execution scheduler."""

from __future__ import annotations

import copy
import unicodedata
from collections.abc import Iterable, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from metrics import metrics_registry
from models import (
    BranchScore,
    DagAction,
    DagPatch,
    DagPatchResult,
    ReplanResult,
    TodoItem,
)
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


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _coverage_for_task(
    task: TodoItem,
    coverage_by_task: dict[int, Iterable[str]] | None,
) -> set[str]:
    values = (
        coverage_by_task.get(int(task.id), getattr(task, "coverage_keys", ()))
        if coverage_by_task is not None
        else getattr(task, "coverage_keys", ())
    )
    return {_normalized_text(value) for value in values if _normalized_text(value)}


def select_wave_one(
    tasks: Sequence[TodoItem],
    depth: str,
    *,
    coverage_by_task: dict[int, Iterable[str]] | None = None,
    completed_ids: set[int] | None = None,
) -> list[TodoItem]:
    """Choose a deterministic, coverage-diverse set of currently legal roots."""
    from services.budget import profile_for_depth

    cap = profile_for_depth(depth).wave_one_limit
    completed = set(completed_ids or ())
    by_id = {int(task.id): task for task in tasks}
    completed.update(task_id for task_id, task in by_id.items() if task.status == "completed")
    roots = []
    for task in tasks:
        if task.status not in {"pending", "queued"}:
            continue
        dependencies = normalize_depends_on(task.depends_on)
        if all(parent_id in completed for parent_id in dependencies):
            roots.append(task)

    selected: list[TodoItem] = []
    covered: set[str] = set()
    while roots and len(selected) < cap:
        roots.sort(
            key=lambda task: (
                -len(_coverage_for_task(task, coverage_by_task).difference(covered)),
                int(task.id),
            )
        )
        next_task = roots.pop(0)
        selected.append(next_task)
        covered.update(_coverage_for_task(next_task, coverage_by_task))
    return selected


_UNIT_TABLE: dict[str, tuple[str, Decimal]] = {
    "m": ("length", Decimal("1")),
    "meter": ("length", Decimal("1")),
    "meters": ("length", Decimal("1")),
    "km": ("length", Decimal("1000")),
    "kilometer": ("length", Decimal("1000")),
    "kilometers": ("length", Decimal("1000")),
    "cm": ("length", Decimal("0.01")),
    "mm": ("length", Decimal("0.001")),
    "kg": ("mass", Decimal("1")),
    "kilogram": ("mass", Decimal("1")),
    "g": ("mass", Decimal("0.001")),
    "gram": ("mass", Decimal("0.001")),
    "mg": ("mass", Decimal("0.000001")),
    "w": ("power", Decimal("1")),
    "kw": ("power", Decimal("1000")),
    "wh": ("energy", Decimal("1")),
    "kwh": ("energy", Decimal("1000")),
    "usd": ("currency:usd", Decimal("1")),
    "$": ("currency:usd", Decimal("1")),
    "eur": ("currency:eur", Decimal("1")),
    "%": ("percent", Decimal("1")),
    "percent": ("percent", Decimal("1")),
}


def _unit_info(value: Any) -> tuple[str, Decimal]:
    unit = _normalized_text(value)
    if unit in _UNIT_TABLE:
        return _UNIT_TABLE[unit]
    return f"unit:{unit}", Decimal("1")


def _decimal_value(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _fact_parts(fact: Any) -> tuple[tuple[str, ...], tuple[str, ...], Decimal | None, Decimal | None] | None:
    if isinstance(fact, (tuple, list)) and len(fact) >= 6:
        record = dict(zip(("entity", "metric", "scope", "period", "unit", "value"), fact[:6]))
    elif isinstance(fact, dict):
        record = fact
    else:
        record = {"fact_key": fact}
    if record.get("valid") is False or record.get("authorized") is False or record.get("anchor_valid") is False:
        return None
    tier = _normalized_text(record.get("tier") or record.get("level") or "")
    if tier in {"l0", "summary", "search_summary"}:
        return None
    if "quality_score" in record:
        score = _decimal_value(record.get("quality_score"))
        if score is not None and score < Decimal("0.5"):
            return None

    raw_value = record.get("value")
    unit_dimension, factor = _unit_info(record.get("unit", ""))
    number = _decimal_value(raw_value) if raw_value is not None else None
    canonical_value = number * factor if number is not None else None
    if canonical_value is not None:
        try:
            quantum = Decimal(1).scaleb(number.as_tuple().exponent)
        except (AttributeError, TypeError):
            quantum = Decimal("1")
        half_width = abs(quantum * factor) / 2
        value_key = format(canonical_value.normalize(), "f")
    else:
        half_width = None
        raw_key = record.get("fact_key", raw_value)
        value_key = _normalized_text(raw_key)
    comparison = tuple(
        _normalized_text(record.get(name, ""))
        for name in ("entity", "metric", "scope", "period")
    ) + (unit_dimension,)
    key = comparison + (value_key,)
    return key, comparison, canonical_value, half_width


def _record_source_ids(records: Sequence[Any]) -> list[dict[str, Any]]:
    result = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            source_id = str(record)
            result.append({"source_id": source_id})
            continue
        if record.get("valid") is False or record.get("authorized") is False or record.get("anchor_valid") is False:
            continue
        if _normalized_text(record.get("tier") or record.get("level") or "") in {"l0", "summary", "search_summary"}:
            continue
        copied = dict(record)
        copied.setdefault("source_id", str(copied.get("id") or f"source-{index}"))
        result.append(copied)
    return result


def _source_clusters(records: Sequence[dict[str, Any]]) -> tuple[dict[str, str], set[str]]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    identity_fields = (
        "source_cluster",
        "origin_id",
        "original_source_id",
        "canonical_url",
        "url",
        "publisher",
        "content_fingerprint",
        "text_fingerprint",
        "table_fingerprint",
    )
    identity_owners: dict[tuple[str, str], str] = {}
    source_ids: set[str] = set()
    for record in records:
        source_id = _normalized_text(record.get("source_id") or record.get("id"))
        if not source_id:
            continue
        source_ids.add(source_id)
        find(source_id)
        for field_name in identity_fields:
            identity = _normalized_text(record.get(field_name))
            if not identity:
                continue
            identity_key = (field_name, identity)
            existing = identity_owners.get(identity_key)
            if existing is not None:
                union(source_id, existing)
            else:
                identity_owners[identity_key] = source_id
    resolved = {source_id: find(source_id) for source_id in source_ids}
    clusters: dict[str, set[str]] = {}
    for source_id, root in resolved.items():
        clusters.setdefault(root, set()).add(source_id)
    canonical = {
        source_id: min(members)
        for source_id, root in resolved.items()
        for members in (clusters[root],)
    }
    return canonical, {min(members) for members in clusters.values()}


def _has_numeric_conflict(left: tuple[Decimal | None, Decimal | None], right: tuple[Decimal | None, Decimal | None]) -> bool:
    left_value, left_width = left
    right_value, right_width = right
    if left_value is None or right_value is None or left_width is None or right_width is None:
        return False
    return abs(left_value - right_value) > left_width + right_width


def score_wave1_branch(
    *,
    task_id: int,
    required_coverage: Iterable[str],
    baseline_coverage: Iterable[str],
    wave_coverage: Iterable[str],
    wave_facts: Sequence[Any],
    baseline_facts: Sequence[Any] = (),
    wave_sources: Sequence[Any] = (),
    baseline_sources: Sequence[Any] = (),
    comparable_pairs: Sequence[dict[str, Any]] = (),
) -> BranchScore:
    """Compute deterministic coverage, evidence-gain, and conflict signals."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise ValueError("task_id must be a positive integer")
    required = {_normalized_text(value) for value in required_coverage if _normalized_text(value)}
    baseline_cov = {_normalized_text(value) for value in baseline_coverage if _normalized_text(value)}
    wave_cov = {_normalized_text(value) for value in wave_coverage if _normalized_text(value)}
    missing = required.difference(baseline_cov)
    covered = missing.intersection(wave_cov)

    valid_wave_facts: list[tuple[tuple[str, ...], tuple[str, ...], Decimal | None, Decimal | None, str]] = []
    valid_baseline_facts: list[tuple[tuple[str, ...], tuple[str, ...], Decimal | None, Decimal | None, str]] = []
    for fact, target in [(fact, valid_wave_facts) for fact in wave_facts] + [
        (fact, valid_baseline_facts) for fact in baseline_facts
    ]:
        parts = _fact_parts(fact)
        if parts is None:
            continue
        source_id = str(fact.get("source_id") or fact.get("evidence_id") or "") if isinstance(fact, dict) else ""
        target.append((*parts, _normalized_text(source_id)))
    wave_fact_keys = {fact[0] for fact in valid_wave_facts}
    baseline_fact_keys = {fact[0] for fact in valid_baseline_facts}
    new_facts = wave_fact_keys.difference(baseline_fact_keys)

    all_source_records = _record_source_ids([*baseline_sources, *wave_sources])
    for fact in [*wave_facts, *baseline_facts]:
        if not isinstance(fact, dict):
            continue
        source_id = fact.get("source_id") or fact.get("evidence_id")
        if source_id:
            all_source_records.append({"source_id": str(source_id)})
    source_lookup, all_clusters = _source_clusters(all_source_records)
    baseline_ids = {
        _normalized_text(record.get("source_id") or record.get("id"))
        for record in _record_source_ids(baseline_sources)
    }
    wave_ids = {
        _normalized_text(record.get("source_id") or record.get("id"))
        for record in _record_source_ids(wave_sources)
    }
    if not wave_ids:
        wave_ids = {fact[4] for fact in valid_wave_facts if fact[4]}
    baseline_clusters = {source_lookup[source_id] for source_id in baseline_ids if source_id in source_lookup}
    wave_clusters = {source_lookup[source_id] for source_id in wave_ids if source_id in source_lookup}
    new_clusters = wave_clusters.difference(baseline_clusters)

    pair_records: dict[tuple[tuple[str, ...], tuple[str, str]], bool] = {}
    fact_groups: dict[tuple[str, ...], list[tuple[str, Decimal | None, Decimal | None, bool]]] = {}
    for fact, is_wave in [*((item, True) for item in valid_wave_facts), *((item, False) for item in valid_baseline_facts)]:
        source_id = fact[4]
        cluster = source_lookup.get(source_id, source_id)
        fact_groups.setdefault(fact[1], []).append((cluster, fact[2], fact[3], is_wave))
    for dimension, values in fact_groups.items():
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                if left[0] == right[0] or not (left[3] or right[3]):
                    continue
                sources = tuple(sorted((left[0], right[0])))
                conflict = _has_numeric_conflict((left[1], left[2]), (right[1], right[2]))
                pair_records[(dimension, sources)] = pair_records.get((dimension, sources), False) or conflict
    for pair in comparable_pairs:
        if not isinstance(pair, dict):
            continue
        if (
            pair.get("valid") is False
            or pair.get("authorized") is False
            or pair.get("anchor_valid") is False
            or _normalized_text(pair.get("tier") or pair.get("level") or "")
            in {"l0", "summary", "search_summary"}
        ):
            continue
        dimension_value = pair.get("dimension")
        if isinstance(dimension_value, (tuple, list)):
            dimension = tuple(_normalized_text(value) for value in dimension_value)
        else:
            dimension = (_normalized_text(dimension_value),)
        left_source = _normalized_text(pair.get("left_source"))
        right_source = _normalized_text(pair.get("right_source"))
        if not left_source or not right_source:
            continue
        left_cluster = source_lookup.get(left_source, left_source)
        right_cluster = source_lookup.get(right_source, right_source)
        if left_cluster == right_cluster:
            continue
        sources = tuple(sorted((left_cluster, right_cluster)))
        key = (dimension, sources)
        pair_records[key] = pair_records.get(key, False) or pair.get("conflict") is True

    comparable_dimensions = {dimension for dimension, _ in pair_records}
    conflicting_dimensions = {
        dimension for (dimension, _), conflict in pair_records.items() if conflict
    }
    denominators = (len(missing), len(wave_fact_keys), len(wave_clusters))
    numerators = (
        len(covered),
        len(new_facts),
        len(new_clusters),
    )
    ig = (
        0.5 * (numerators[0] / denominators[0] if denominators[0] else 0.0)
        + 0.3 * (numerators[1] / denominators[1] if denominators[1] else 0.0)
        + 0.2 * (numerators[2] / denominators[2] if denominators[2] else 0.0)
    )
    cd = len(conflicting_dimensions) / max(1, len(comparable_dimensions))
    insufficient = len(new_clusters) < 2 or not wave_fact_keys or not comparable_dimensions
    return BranchScore(
        task_id=task_id,
        information_gain=ig,
        conflict_density=cd,
        ig_numerators=numerators,
        ig_denominators=denominators,
        covered_dimensions=tuple(sorted(covered)),
        effective_fact_count=len(wave_fact_keys),
        independent_source_clusters=len(new_clusters),
        comparable_dimensions=len(comparable_dimensions),
        conflicting_dimensions=len(conflicting_dimensions),
        comparable_pair_count=len(pair_records),
        insufficient_samples=insufficient,
        score_version=1,
        missing_coverage_keys=tuple(sorted(missing.difference(covered))),
    )


def _exact_ig(score: BranchScore) -> Fraction:
    total = Fraction(0, 1)
    weights = (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5))
    for weight, numerator, denominator in zip(weights, score.ig_numerators, score.ig_denominators):
        if denominator:
            total += weight * Fraction(numerator, denominator)
    return total


def _exact_cd(score: BranchScore) -> Fraction:
    if score.comparable_dimensions:
        return Fraction(score.conflicting_dimensions, score.comparable_dimensions)
    return Fraction(str(score.conflict_density))


def _descendant_ids(tasks: Sequence[TodoItem], root_id: int) -> set[int]:
    descendants = {root_id}
    changed = True
    while changed:
        changed = False
        for task in tasks:
            task_id = int(task.id)
            if task_id in descendants:
                continue
            if descendants.intersection(normalize_depends_on(task.depends_on)):
                descendants.add(task_id)
                changed = True
    return descendants


class ResearchTreeRePlanner:
    """Rule-based Wave 1 decision engine with stable priority and bounded actions."""

    def plan(
        self,
        tasks: Sequence[TodoItem],
        scores: Sequence[BranchScore],
        *,
        depth: str,
        next_task_id: int,
        available_task_slots: int,
        current_revision: int = 0,
        expansion_count: int = 0,
        missing_coverage_by_branch: dict[int, Sequence[str]] | None = None,
        needs_cross_check: dict[int, bool] | None = None,
        pivot_queries: dict[int, str] | None = None,
        deepen_queries: dict[int, str] | None = None,
        optional_task_ids: set[int] | None = None,
        dispatched_ids: set[int] | None = None,
        locked_ids: set[int] | None = None,
        awaiting_approval_ids: set[int] | None = None,
        failure_signatures_by_branch: dict[int, Sequence[dict[str, Any]]] | None = None,
        plan_locked: bool = False,
        wave_1_committed: bool = False,
    ) -> ReplanResult:
        from services.budget import profile_for_depth

        profile = profile_for_depth(depth)
        if plan_locked:
            return self._no_patch("plan_locked")
        if wave_1_committed:
            return self._no_patch("replan_disabled")
        if not tasks or not scores:
            return self._no_patch("no_eligible_signal")
        if isinstance(available_task_slots, bool) or available_task_slots < 0:
            raise ValueError("available_task_slots must be a non-negative integer")
        by_id = {int(task.id): task for task in tasks}
        dispatched = set(dispatched_ids or ())
        locked = set(locked_ids or ())
        awaiting = set(awaiting_approval_ids or ())
        score_by_id = {score.task_id: score for score in scores}
        ordered_scores = sorted(
            score_by_id.values(),
        key=lambda score: (-_exact_cd(score), -_exact_ig(score), score.task_id),
        )
        actions: list[DagAction] = []
        used_targets: set[int] = set()
        expansions = max(0, expansion_count)
        remaining_slots = available_task_slots
        next_id = next_task_id
        max_expansions = max(0, profile.deepen_limit - expansions)
        skip_reason = "no_eligible_signal"

        for score in ordered_scores:
            root = by_id.get(score.task_id)
            if root is None:
                continue
            branch = _descendant_ids(tasks, score.task_id)
            candidates = sorted(
                task_id
                for task_id in branch.difference({score.task_id})
                if task_id in by_id
                and by_id[task_id].status in {"pending", "queued"}
                and task_id not in dispatched
                and task_id not in locked
                and task_id not in awaiting
                and task_id not in used_targets
            )
            failures = (failure_signatures_by_branch or {}).get(score.task_id, ())
            repeated_failure = False
            if len(failures) >= 2:
                previous, latest = failures[-2:]
                repeated_failure = (
                    previous.get("terminal") is True
                    and latest.get("terminal") is True
                    and previous.get("signature")
                    and previous.get("signature") == latest.get("signature")
                    and previous.get("coverage_key") == latest.get("coverage_key")
                    and previous.get("reason") not in {"budget_denied", "cancelled", "approval_pending"}
                    and latest.get("reason") not in {"budget_denied", "cancelled", "approval_pending"}
                )
            pivot_eligible = (
                score.comparable_dimensions > 0 and _exact_cd(score) >= Fraction(35, 100)
            ) or repeated_failure
            if pivot_eligible:
                if score.insufficient_samples and not repeated_failure and score.comparable_dimensions == 0:
                    skip_reason = "insufficient_samples"
                    continue
                target_id = next(
                    (task_id for task_id in candidates if task_id in (pivot_queries or {})),
                    None,
                )
                if target_id is None:
                    skip_reason = "no_successor" if not candidates else "budget_denied"
                    continue
                target = by_id[target_id]
                alternative_query = str((pivot_queries or {}).get(target_id, "")).strip()
                if not alternative_query:
                    skip_reason = "no_successor"
                    continue
                actions.append(
                    DagAction(
                        action="PIVOT",
                        task_id=target_id,
                        parent_task_ids=[score.task_id],
                        reason_code="conflict_density" if not repeated_failure else "repeated_terminal_failure",
                        before={
                            "query_summary": target.query[:200],
                            "intent_summary": target.intent[:200],
                            "depends_on": list(target.depends_on),
                        },
                        after={
                            "query": alternative_query,
                            "intent": target.intent,
                            "depends_on": list(target.depends_on),
                        },
                    )
                )
                used_targets.add(target_id)
                skip_reason = "no_eligible_signal"
                continue

            missing = list((missing_coverage_by_branch or {}).get(score.task_id, score.missing_coverage_keys))
            deepen_eligible = _exact_ig(score) >= Fraction(45, 100) and (
                bool(missing) or (needs_cross_check or {}).get(score.task_id, False)
            )
            if deepen_eligible:
                if max_expansions <= 0 or remaining_slots <= 0:
                    skip_reason = "budget_denied"
                    continue
                if root.status != "completed":
                    skip_reason = "no_successor"
                    continue
                query = str((deepen_queries or {}).get(score.task_id) or f"{root.query} independent corroboration").strip()
                action = DagAction(
                    action="DEEPEN",
                    task_id=next_id,
                    parent_task_ids=[score.task_id],
                    reason_code="missing_coverage" if missing else "cross_check",
                    after={
                        "title": f"补充验证：{root.title}"[:120],
                        "intent": f"补充覆盖 {', '.join(missing)}" if missing else f"独立交叉确认：{root.intent}",
                        "query": query,
                        "depends_on": [score.task_id],
                        "coverage_keys": missing,
                    },
                )
                actions.append(action)
                next_id += 1
                remaining_slots -= 1
                max_expansions -= 1
                skip_reason = "no_eligible_signal"
                continue

            prune_eligible = (
                _exact_ig(score) <= Fraction(15, 100)
                and not score.insufficient_samples
                and score.conflicting_dimensions == 0
                and not missing
            )
            if prune_eligible:
                prune_target = next(
                    (
                        task_id
                        for task_id in candidates
                        if task_id in (optional_task_ids or set())
                    ),
                    None,
                )
                if prune_target is None:
                    skip_reason = "coverage_required" if candidates else "no_successor"
                    continue
                actions.append(
                    DagAction(
                        action="PRUNE",
                        task_id=prune_target,
                        parent_task_ids=[score.task_id],
                        reason_code="redundant_branch",
                    )
                )
                used_targets.add(prune_target)

        if not actions:
            return self._no_patch(skip_reason)
        patch = DagPatch(
            base_revision=current_revision,
            decision_id=f"wave1-r{current_revision + 1}",
            actions=actions,
            score_version=1,
            scores=list(ordered_scores),
        )
        metrics_registry.record_replan_event(
            patched=True,
            action=actions[0].action if len(actions) == 1 else None,
            reason=actions[0].reason_code if len(actions) == 1 else None,
        )
        return ReplanResult(patch=patch, reason_code="patched")

    @staticmethod
    def _no_patch(reason: str) -> ReplanResult:
        metrics_registry.record_replan_event(reason=reason)
        return ReplanResult(patch=None, reason_code=reason)


def _is_positive_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _graph_is_acyclic(tasks: Sequence[TodoItem]) -> bool:
    by_id = {int(task.id): task for task in tasks}
    indegree = {task_id: 0 for task_id in by_id}
    children: dict[int, list[int]] = {task_id: [] for task_id in by_id}
    for task in tasks:
        task_id = int(task.id)
        dependencies = task.depends_on or []
        if not isinstance(dependencies, (list, tuple)):
            return False
        if any(not _is_positive_id(parent_id) for parent_id in dependencies):
            return False
        if len(dependencies) != len(set(dependencies)):
            return False
        for parent_id in dependencies:
            if parent_id not in by_id or parent_id == task_id:
                return False
            indegree[task_id] += 1
            children[parent_id].append(task_id)
    ready = sorted(task_id for task_id, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        task_id = ready.pop(0)
        visited += 1
        for child_id in sorted(children[task_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
                ready.sort()
    return visited == len(by_id)


def _rewave_remaining_tasks(
    tasks: Sequence[TodoItem],
    *,
    pruned_ids: set[int],
) -> list[list[TodoItem]]:
    by_id = {int(task.id): task for task in tasks}
    completed_ids = {
        task_id for task_id, task in by_id.items() if task.status == "completed"
    }
    pending_ids = {
        task_id
        for task_id, task in by_id.items()
        if task.status in {"pending", "queued"} and task_id not in pruned_ids
    }
    runnable_ids = set(completed_ids)
    remaining = set(pending_ids)
    while remaining:
        ready = {
            task_id
            for task_id in remaining
            if set(normalize_depends_on(by_id[task_id].depends_on)).issubset(runnable_ids)
        }
        if not ready:
            break
        runnable_ids.update(ready)
        remaining.difference_update(ready)
    runnable = [by_id[task_id] for task_id in sorted(pending_ids.difference(remaining))]
    return partition_into_waves(runnable, completed_ids=completed_ids)


def _reachable_coverage(
    tasks: Sequence[TodoItem],
    *,
    pruned_ids: set[int],
    coverage_by_task: dict[int, Iterable[str]] | None,
) -> set[str]:
    return {
        key
        for task in tasks
        if int(task.id) not in pruned_ids
        for key in _coverage_for_task(task, coverage_by_task)
    }


def apply_dag_patch(
    tasks: Sequence[TodoItem],
    patch: DagPatch,
    *,
    current_revision: int,
    next_task_id: int | None = None,
    historical_task_ids: set[int] | None = None,
    dispatched_ids: set[int] | None = None,
    awaiting_approval_ids: set[int] | None = None,
    locked_ids: set[int] | None = None,
    required_coverage: Iterable[str] = (),
    coverage_by_task: dict[int, Iterable[str]] | None = None,
    query_scope_by_task: dict[int, Iterable[str]] | None = None,
    available_task_slots: int | None = None,
    depth: str = "standard",
    expansion_count: int = 0,
    wave_1_committed: bool = False,
) -> DagPatchResult:
    """Validate and atomically apply a candidate graph patch on a deep copy."""
    from services.budget import profile_for_depth

    original = list(tasks)
    base_failure = current_revision if isinstance(current_revision, int) and current_revision >= 0 else 0

    def reject(reason: str = "invalid_patch") -> DagPatchResult:
        metrics_registry.record_replan_event(rejected=True, reason=reason, decision=False)
        return DagPatchResult(applied=False, reason_code=reason, revision=base_failure, tasks=original)

    if not isinstance(patch, DagPatch):
        return reject()
    if isinstance(current_revision, bool) or not isinstance(current_revision, int) or current_revision < 0:
        return reject()
    if isinstance(patch.base_revision, bool) or patch.base_revision != current_revision:
        return reject("stale_revision")
    if wave_1_committed:
        return reject("replan_disabled")
    if patch.score_version != 1 or patch.decision_id != f"wave1-r{current_revision + 1}":
        return reject()
    if not isinstance(patch.actions, list) or any(
        not isinstance(action, DagAction) for action in patch.actions
    ):
        return reject()
    if any(not _is_positive_id(task.id) for task in original):
        return reject()
    original_ids = [int(task.id) for task in original]
    if len(original_ids) != len(set(original_ids)) or not _graph_is_acyclic(original):
        return reject()
    normalized_queries = [_normalized_text(task.query) for task in original]
    if any(not query for query in normalized_queries) or len(normalized_queries) != len(set(normalized_queries)):
        return reject()
    if not patch.actions:
        return DagPatchResult(
            applied=True,
            reason_code="no_eligible_signal",
            revision=current_revision,
            tasks=copy.deepcopy(original),
            waves=_rewave_remaining_tasks(copy.deepcopy(original), pruned_ids=set()),
        )
    if any(not _is_positive_id(action.task_id) for action in patch.actions):
        return reject()
    if len({action.task_id for action in patch.actions}) != len(patch.actions):
        return reject()

    by_id = {int(task.id): task for task in copy.deepcopy(original)}
    history = set(historical_task_ids or ())
    if any(not _is_positive_id(task_id) for task_id in history):
        return reject()
    dispatched = set(dispatched_ids or ())
    awaiting = set(awaiting_approval_ids or ())
    locked = set(locked_ids or ())
    required = {_normalized_text(value) for value in required_coverage if _normalized_text(value)}
    before_coverage = _reachable_coverage(
        list(by_id.values()), pruned_ids=set(), coverage_by_task=coverage_by_task
    )
    expected_next = next_task_id if next_task_id is not None else max([*original_ids, *history], default=0) + 1
    if isinstance(expected_next, bool) or not isinstance(expected_next, int) or expected_next <= max(
        [*original_ids, *history], default=0
    ):
        return reject()
    added_ids: list[int] = []
    pruned_ids: set[int] = set()
    profile = profile_for_depth(depth)
    deepen_count = sum(1 for action in patch.actions if action.action == "DEEPEN")
    if deepen_count:
        if available_task_slots is None or isinstance(available_task_slots, bool) or available_task_slots < deepen_count:
            return reject("budget_denied")
        if expansion_count + deepen_count > profile.deepen_limit:
            return reject("budget_denied")

    for action in patch.actions:
        if action.action not in {"PIVOT", "DEEPEN", "PRUNE"} or not _is_positive_id(action.task_id):
            return reject()
        if not isinstance(action.parent_task_ids, list):
            return reject()
        if action.before is not None and not isinstance(action.before, dict):
            return reject()
        if action.after is not None and not isinstance(action.after, dict):
            return reject()
        if any(not _is_positive_id(parent_id) for parent_id in action.parent_task_ids):
            return reject()
        if len(action.parent_task_ids) != len(set(action.parent_task_ids)):
            return reject()
        if action.action == "DEEPEN":
            if action.task_id in by_id or action.task_id in history or action.task_id != expected_next:
                return reject()
            after = action.after or {}
            if set(after).difference({"title", "intent", "query", "depends_on", "coverage_keys"}):
                return reject()
            title = str(after.get("title") or "").strip()
            intent = str(after.get("intent") or "").strip()
            query = str(after.get("query") or "").strip()
            dependencies = after.get("depends_on")
            if not title or not intent or not query or not isinstance(dependencies, list):
                return reject()
            if dependencies != action.parent_task_ids or not dependencies:
                return reject()
            if any(parent_id not in by_id or by_id[parent_id].status != "completed" for parent_id in dependencies):
                return reject()
            if any(parent_id in dispatched and by_id[parent_id].status != "completed" for parent_id in dependencies):
                return reject()
            normalized_query = _normalized_text(query)
            if any(_normalized_text(task.query) == normalized_query for task in by_id.values()):
                return reject()
            task = TodoItem(
                id=action.task_id,
                title=title,
                intent=intent,
                query=query,
                status="pending",
                origin="replan",
                round=2,
                depends_on=list(dependencies),
            )
            task.coverage_keys = list(after.get("coverage_keys") or [])
            by_id[action.task_id] = task
            added_ids.append(action.task_id)
            expected_next += 1
            continue

        task = by_id.get(action.task_id)
        if task is None or task.status not in {"pending", "queued"}:
            return reject()
        if action.task_id in dispatched or action.task_id in awaiting or action.task_id in locked:
            return reject("plan_locked")
        ancestors = set(normalize_depends_on(task.depends_on))
        frontier = list(ancestors)
        while frontier:
            parent = by_id.get(frontier.pop())
            if parent is None:
                continue
            for parent_id in normalize_depends_on(parent.depends_on):
                if parent_id not in ancestors:
                    ancestors.add(parent_id)
                    frontier.append(parent_id)
        if any(parent_id not in ancestors for parent_id in action.parent_task_ids):
            return reject()
        if action.action == "PIVOT":
            after = action.after or {}
            if not action.parent_task_ids or set(after).difference(
                {"query", "intent", "depends_on", "scope_keys"}
            ):
                return reject()
            query = str(after.get("query") or "").strip()
            if not query or _normalized_text(query) == _normalized_text(task.query):
                return reject()
            if "depends_on" in after and normalize_depends_on(after["depends_on"]) != normalize_depends_on(task.depends_on):
                return reject()
            normalized_query = _normalized_text(query)
            if any(
                other_id != action.task_id and _normalized_text(other.query) == normalized_query
                for other_id, other in by_id.items()
            ):
                return reject()
            scope_values = (query_scope_by_task or {}).get(
                action.task_id,
                getattr(task, "scope_keys", ()),
            )
            scope_before = {
                _normalized_text(value) for value in scope_values if _normalized_text(value)
            }
            scope_after = set(_normalized_text(value) for value in after.get("scope_keys", scope_before) if _normalized_text(value))
            if not scope_after.issubset(scope_before):
                return reject()
            task.query = query
            task.intent = str(after.get("intent") or task.intent).strip()
        elif action.action == "PRUNE":
            if not action.parent_task_ids or action.after:
                return reject()
            if _coverage_for_task(task, coverage_by_task).intersection(required):
                return reject("coverage_required")
            pruned_ids.add(action.task_id)

    for task in by_id.values():
        task_id = int(task.id)
        if task_id in pruned_ids:
            continue
        dependencies = normalize_depends_on(task.depends_on)
        if any(parent_id in pruned_ids for parent_id in dependencies):
            return reject("coverage_required")
        task.depends_on = dependencies
    if not _graph_is_acyclic(list(by_id.values())):
        return reject()

    after_coverage = _reachable_coverage(
        list(by_id.values()), pruned_ids=pruned_ids, coverage_by_task=coverage_by_task
    )
    required_to_preserve = before_coverage.intersection(required)
    if not required_to_preserve.issubset(after_coverage):
        return reject("coverage_required")
    task_list = [by_id[task_id] for task_id in sorted(by_id)]
    waves = _rewave_remaining_tasks(task_list, pruned_ids=pruned_ids)
    metrics_registry.record_replan_event(
        patched=True,
        action=patch.actions[0].action if len(patch.actions) == 1 else None,
        decision=False,
    )
    return DagPatchResult(
        applied=True,
        reason_code="patched",
        revision=current_revision + 1,
        tasks=task_list,
        waves=waves,
        added_task_ids=added_ids,
        pruned_task_ids=sorted(pruned_ids),
    )
