"""Hybrid benchmark judge blending deterministic heuristic checks with LLM rubric scoring."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evals.judges.base import Judge
from evals.judges.heuristic import HeuristicJudge
from evals.judges.llm import LLMJudge
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
        parts = [
            float(metrics.get("keyword_coverage") or 0.0),
            float(metrics.get("section_completeness") or 0.0),
            float(metrics.get("citation_integrity") or 0.0),
            float(metrics.get("citation_density_score") or 0.0),
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
