"""Hybrid benchmark judge blending deterministic heuristic checks with LLM rubric scoring."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evals.judges.base import Judge
from evals.judges.heuristic import HeuristicJudge
from evals.judges.llm import LLMJudge
from evals.metrics.factscore import compute_atomic_factscore, compute_conflict_recall
from evals.schema import BenchmarkCase


class HybridJudge(Judge):
    """Combine deterministic HeuristicJudge metrics with optional LLMJudge semantic evaluation."""

    _LLM_RUBRIC_KEYS = (
        "factuality_score",
        "coverage_score",
        "citation_grounding_score",
        "freshness_score",
        "conservativeness_score",
    )

    def __init__(
        self,
        *,
        heuristic_judge: HeuristicJudge | None = None,
        llm_judge: LLMJudge | None = None,
        heuristic_weight: float = 0.5,
        llm_weight: float = 0.5,
    ) -> None:
        self._heuristic = heuristic_judge or HeuristicJudge()
        self._llm = llm_judge
        total = max(float(heuristic_weight) + float(llm_weight), 1e-6)
        self._heuristic_weight = float(heuristic_weight) / total
        self._llm_weight = float(llm_weight) / total

    @staticmethod
    def _heuristic_composite(metrics: Mapping[str, Any]) -> float:
        citation_integrity = (
            metrics.get("citation_integrity")
            if "citation_integrity" in metrics
            else metrics.get("reference_match_rate", 0.0)
        )
        citation_density = (
            metrics.get("citation_density_score")
            if "citation_density_score" in metrics
            else metrics.get("grounded_bullet_ratio", 0.0)
        )
        parts = [
            float(metrics.get("keyword_coverage") or 0.0),
            float(metrics.get("section_completeness") or 0.0),
            float(citation_integrity or 0.0),
            float(citation_density or 0.0),
        ]
        return round(sum(parts) / len(parts), 4)

    def _llm_composite(self, llm_metrics: Mapping[str, Any]) -> float | None:
        scores = [
            float(llm_metrics[key])
            for key in self._LLM_RUBRIC_KEYS
            if key in llm_metrics and isinstance(llm_metrics[key], (int, float))
        ]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 4)

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        report_markdown: str,
        todo_items: Sequence[Any],
        trace_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(
            self._heuristic.evaluate(
                case=case,
                report_markdown=report_markdown,
                todo_items=todo_items,
                trace_snapshot=trace_snapshot,
            )
        )
        trace = trace_snapshot or {}
        claim_summary = trace.get("claim_verification_summary")
        review = trace.get("review_summary")
        if not claim_summary and isinstance(review, Mapping):
            claim_summary = review.get("claim_verification_summary")
        atomic_score = compute_atomic_factscore(todo_items, claim_summary)
        if atomic_score["total_atomic_claims"]:
            merged.update(atomic_score)
            merged["atomic_faithfulness_rate"] = atomic_score["atomic_factscore"]
        elif isinstance(claim_summary, Mapping) and claim_summary.get("total_atomic_claims"):
            # Older snapshots can carry only the precomputed rate.
            legacy_rate = claim_summary.get("atomic_faithfulness_rate")
            if isinstance(legacy_rate, (int, float)) and 0 <= legacy_rate <= 1:
                merged["atomic_faithfulness_rate"] = float(legacy_rate)

        expected_conflicts = case.metadata.get("expected_conflicts")
        if isinstance(expected_conflicts, list) and expected_conflicts:
            merged.update(compute_conflict_recall(trace.get("evidence_conflicts") or [], expected_conflicts))

        heuristic_score = self._heuristic_composite(merged)
        merged["heuristic_composite_score"] = heuristic_score

        if self._llm is None:
            merged["judge_type"] = "hybrid_heuristic_only"
            merged["llm_judge_status"] = "skipped"
            merged["hybrid_score"] = heuristic_score
            return merged

        try:
            llm_metrics = dict(
                self._llm.evaluate(
                    case=case,
                    report_markdown=report_markdown,
                    todo_items=todo_items,
                    trace_snapshot=trace_snapshot,
                )
            )
        except Exception as exc:
            merged["judge_type"] = "hybrid_fallback_heuristic"
            merged["llm_judge_status"] = "degraded"
            merged["llm_judge_error"] = str(exc)
            merged["hybrid_score"] = heuristic_score
            return merged

        for key, value in llm_metrics.items():
            if key in {"report_generated", "degraded_flag", "total_latency_ms", "estimated_cost"}:
                continue
            target_key = key if key not in merged else f"llm_{key}"
            merged[target_key] = value

        llm_status = str(llm_metrics.get("judge_status") or "ok").strip().lower()
        llm_composite = self._llm_composite(llm_metrics)
        if llm_composite is not None and llm_status in {"ok", "completed", "success", ""}:
            merged["judge_type"] = "hybrid"
            merged["llm_judge_status"] = "ok"
            merged["llm_composite_score"] = llm_composite
            merged["hybrid_score"] = round(
                (self._heuristic_weight * heuristic_score) + (self._llm_weight * llm_composite),
                4,
            )
        else:
            merged["judge_type"] = "hybrid_fallback_heuristic"
            merged["llm_judge_status"] = llm_status or "degraded"
            merged["hybrid_score"] = heuristic_score

        return merged
