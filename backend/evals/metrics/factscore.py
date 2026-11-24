"""Offline FActScore and cross-source conflict resolution evaluation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compute_atomic_factscore(
    todo_items: Sequence[Mapping[str, Any]],
    claim_verification_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute atomic-level FActScore / Faithfulness across task atomic_claims or summary counts."""

    entailed = contradicted = neutral = skipped = 0
    saw_atomics = False
    for task in todo_items or ():
        if not isinstance(task, Mapping):
            continue
        for item in task.get("atomic_claims") or ():
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
        entailed = int(claim_verification_summary.get("entailed_count") or 0)
        contradicted = int(claim_verification_summary.get("contradicted_count") or 0)
        neutral = int(claim_verification_summary.get("neutral_count") or 0)
        skipped = int(claim_verification_summary.get("skipped_count") or 0)

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


def summarize_conflict_resolution(
    evidence_conflicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize cross-source conflict clusters by adjudication outcome and severity."""

    total = len(evidence_conflicts or ())
    resolved_authority = resolved_recency = unresolved = 0
    high_sev = medium_sev = low_sev = 0
    for cluster in evidence_conflicts or ():
        if not isinstance(cluster, Mapping):
            continue
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
