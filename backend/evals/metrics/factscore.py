"""Offline FActScore and cross-source conflict resolution evaluation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_LABEL_COUNT_KEYS = ("entailed_count", "contradicted_count", "neutral_count", "skipped_count")


def _nonnegative_count(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def compute_atomic_factscore(
    todo_items: Sequence[Any],
    claim_verification_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute atomic-level FActScore / Faithfulness across task atomic_claims or summary counts."""

    entailed = contradicted = neutral = skipped = 0
    saw_atomics = False
    for task in todo_items or ():
        atomic_claims = task.get("atomic_claims") if isinstance(task, Mapping) else getattr(task, "atomic_claims", None)
        for item in atomic_claims or ():
            if not isinstance(item, Mapping):
                continue
            saw_atomics = True
            label = str(item.get("nli_label") or "").strip().upper()
            if label == "ENTAILED":
                entailed += 1
            elif label == "CONTRADICTED":
                contradicted += 1
            elif label == "NEUTRAL":
                neutral += 1
            else:
                skipped += 1

    if not saw_atomics and isinstance(claim_verification_summary, Mapping) and claim_verification_summary:
        entailed, contradicted, neutral, skipped = (
            _nonnegative_count(claim_verification_summary.get(key)) for key in _LABEL_COUNT_KEYS
        )

    denom = entailed + contradicted + neutral
    factscore = round(entailed / denom, 4) if denom > 0 else None
    contradiction_rate = round(contradicted / denom, 4) if denom > 0 else None
    return {
        "total_atomic_claims": denom + skipped,
        "evaluated_atomic_claims": denom,
        "entailed_count": entailed,
        "contradicted_count": contradicted,
        "neutral_count": neutral,
        "skipped_count": skipped,
        "atomic_factscore": factscore,
        "contradiction_rate": contradiction_rate,
    }


def aggregate_atomic_factscores(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(_nonnegative_count(score.get(key)) for score in scores) for key in _LABEL_COUNT_KEYS}
    return compute_atomic_factscore([], counts)


def summarize_conflict_resolution(
    evidence_conflicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize cross-source conflict clusters by adjudication outcome and severity."""

    total = 0
    seen_ids: set[str] = set()
    resolved_authority = resolved_recency = unresolved = 0
    high_sev = medium_sev = low_sev = 0
    for cluster in evidence_conflicts or ():
        if not isinstance(cluster, Mapping):
            continue
        conflict_id = str(cluster.get("conflict_id") or "").strip()
        if conflict_id:
            if conflict_id in seen_ids:
                continue
            seen_ids.add(conflict_id)
        total += 1
        status = str(cluster.get("adjudication_status") or "").strip().lower()
        if status == "resolved_by_authority":
            resolved_authority += 1
        elif status == "resolved_by_recency":
            resolved_recency += 1
        else:
            unresolved += 1

        sev = str(cluster.get("severity") or "").strip().lower()
        if sev == "high":
            high_sev += 1
        elif sev == "medium":
            medium_sev += 1
        else:
            low_sev += 1

    return {
        "conflict_count": total,
        "resolved_by_authority": resolved_authority,
        "resolved_by_recency": resolved_recency,
        "unresolved_divergence": unresolved,
        "resolution_rate": round((resolved_authority + resolved_recency) / total, 4) if total > 0 else None,
        "severity_counts": {
            "high": high_sev,
            "medium": medium_sev,
            "low": low_sev,
        },
    }


def compute_conflict_recall(
    evidence_conflicts: Sequence[Mapping[str, Any]],
    expected_conflicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Match distinct gold entity/metric/period clusters to the graph for one case."""
    from services.claim_verifier import normalize_conflict_key

    def keys(conflicts: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
        normalized = set()
        for cluster in conflicts:
            if not isinstance(cluster, Mapping):
                continue
            entity = str(cluster.get("entity") or "").strip()
            metric = str(cluster.get("metric") or "").strip()
            period = str(cluster.get("period") or "").strip()
            if entity and metric:
                key = normalize_conflict_key(entity, metric, period or "general")
                if key[0] and key[1]:
                    normalized.add(key)
        return normalized

    expected = keys(expected_conflicts)
    return _conflict_recall_result(len(keys(evidence_conflicts) & expected), len(expected))


def aggregate_conflict_recall(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _conflict_recall_result(
        sum(_nonnegative_count(score.get("matched_conflict_count")) for score in scores),
        sum(_nonnegative_count(score.get("expected_conflict_count")) for score in scores),
    )


def _conflict_recall_result(matched: int, expected: int) -> dict[str, Any]:
    return {
        "expected_conflict_count": expected,
        "matched_conflict_count": matched,
        "conflict_recall": round(matched / expected, 4) if expected else None,
    }
