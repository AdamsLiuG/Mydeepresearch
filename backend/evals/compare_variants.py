"""Case-aligned differential comparison between two benchmark variants (L2 attribution).

The offline benchmark is deterministic and replayable, which makes it possible to
attribute an outcome change to memory *causally*: run the same cases twice, once with
memory disabled (baseline) and once with it enabled (candidate), then diff the
artefacts case by case. Unlike an LLM self-report, this needs no model cooperation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Bounded [0, 1] quality ratios (weight 1.0 in normalized quality gain).
RATIO_QUALITY_METRICS = (
    "section_completeness",
    "keyword_coverage",
    "reference_match_rate",
    "grounded_bullet_ratio",
)

# Optional semantic quality scores from LLMJudge (also [0, 1], weight 1.0 when present).
LLM_JUDGE_QUALITY_METRICS = (
    "factuality_score",
    "coverage_score",
    "citation_grounding_score",
    "freshness_score",
    "conservativeness_score",
)

# Optional composite & verifier quality metrics (additive when present in either variant).
OPTIONAL_VERIFIER_QUALITY_METRICS = (
    "hybrid_score",
    "verifier_clean_rate",
    "citation_validity_rate",
)

DEFAULT_MAX_ADVERSARIAL_REGRESSION = 0.02

# Unbounded count metrics (relative-normalized and down-weighted so raw counts never drown ratios).
COUNT_QUALITY_METRICS = (
    "citation_count",
    "citation_marker_count",
    "completed_task_count",
)

COUNT_METRIC_WEIGHT = 0.25

# Metrics where a higher value is better; the sign of the delta is interpreted with this.
HIGHER_IS_BETTER = RATIO_QUALITY_METRICS + COUNT_QUALITY_METRICS

# Metrics that represent a cost paid; a positive delta is a regression.
LOWER_IS_BETTER = (
    "total_latency_ms",
    "estimated_cost",
)

DEFAULT_MAX_LATENCY_REGRESSION_RATIO = 0.50
DEFAULT_MAX_COST_REGRESSION_RATIO = 0.50

# Reported for context only, never scored.
INFORMATIONAL = (
    "memory_query_count",
    "memory_hit_count",
    "memory_effective_injection_count",
    "memory_rendered_injection_count",
    "memory_shadow_suppressions",
)


def _load_payload(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark result must be a JSON object: {resolved}")
    return payload


def _index_results(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            continue
        case_id = str(result.get("id") or "").strip()
        if case_id:
            indexed[case_id] = result
    return indexed


def _metric_value(result: Mapping[str, Any], name: str) -> float:
    metrics = result.get("metrics") or {}
    try:
        return float(metrics.get(name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _task_titles(result: Mapping[str, Any]) -> set[str]:
    titles: set[str] = set()
    for item in result.get("todo_items") or []:
        if isinstance(item, Mapping):
            title = str(item.get("title") or "").strip().casefold()
            if title:
                titles.add(title)
    return titles


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return round(len(left & right) / len(union), 4)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _case_normalized_quality_delta(
    base_result: Mapping[str, Any],
    cand_result: Mapping[str, Any],
) -> float:
    """Compute a scale-safe weighted quality delta for a single case."""
    base_metrics = base_result.get("metrics") or {}
    cand_metrics = cand_result.get("metrics") or {}

    weighted_sum = 0.0
    for name in RATIO_QUALITY_METRICS:
        weighted_sum += _metric_value(cand_result, name) - _metric_value(base_result, name)

    for name in LLM_JUDGE_QUALITY_METRICS + OPTIONAL_VERIFIER_QUALITY_METRICS:
        if name in base_metrics or name in cand_metrics:
            weighted_sum += _metric_value(cand_result, name) - _metric_value(base_result, name)

    for name in COUNT_QUALITY_METRICS:
        base_val = _metric_value(base_result, name)
        cand_val = _metric_value(cand_result, name)
        if base_val == 0.0 and cand_val == 0.0:
            continue
        rel_delta = _clamp((cand_val - base_val) / max(abs(base_val), 1.0))
        weighted_sum += COUNT_METRIC_WEIGHT * rel_delta

    return round(weighted_sum, 6)


def compare_cases(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    metric_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Diff every case present in both runs."""
    baseline_index = _index_results(baseline)
    candidate_index = _index_results(candidate)
    shared_ids = sorted(set(baseline_index) & set(candidate_index))
    default_names = list(HIGHER_IS_BETTER + LOWER_IS_BETTER + INFORMATIONAL)
    for extra_metric in LLM_JUDGE_QUALITY_METRICS + OPTIONAL_VERIFIER_QUALITY_METRICS:
        if any(
            extra_metric in (baseline_index[cid].get("metrics") or {})
            or extra_metric in (candidate_index[cid].get("metrics") or {})
            for cid in shared_ids
        ):
            default_names.append(extra_metric)
    names = tuple(metric_names or default_names)

    comparisons: list[dict[str, Any]] = []
    for case_id in shared_ids:
        base_result = baseline_index[case_id]
        cand_result = candidate_index[case_id]
        deltas = {
            name: round(_metric_value(cand_result, name) - _metric_value(base_result, name), 6)
            for name in names
        }
        base_titles = _task_titles(base_result)
        cand_titles = _task_titles(cand_result)
        net_quality_delta = _case_normalized_quality_delta(base_result, cand_result)
        comparisons.append(
            {
                "id": case_id,
                "topic": str(cand_result.get("topic") or base_result.get("topic") or ""),
                "task_title_jaccard": _jaccard(base_titles, cand_titles),
                "plan_changed": base_titles != cand_titles,
                "baseline_error": base_result.get("error"),
                "candidate_error": cand_result.get("error"),
                "net_quality_delta": net_quality_delta,
                "deltas": deltas,
            }
        )
    return comparisons


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def build_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_label: str | None = None,
    candidate_label: str | None = None,
    max_latency_regression_ratio: float = DEFAULT_MAX_LATENCY_REGRESSION_RATIO,
    max_cost_regression_ratio: float = DEFAULT_MAX_COST_REGRESSION_RATIO,
    max_adversarial_regression: float = DEFAULT_MAX_ADVERSARIAL_REGRESSION,
) -> dict[str, Any]:
    """Build the full comparison payload, including normalized quality & cost guardrails."""
    comparisons = compare_cases(baseline, candidate)
    baseline_index = _index_results(baseline)
    candidate_index = _index_results(candidate)
    shared_ids = [case["id"] for case in comparisons]

    metric_names = list(HIGHER_IS_BETTER + LOWER_IS_BETTER + INFORMATIONAL)
    for extra_metric in LLM_JUDGE_QUALITY_METRICS + OPTIONAL_VERIFIER_QUALITY_METRICS:
        if any(extra_metric in case["deltas"] for case in comparisons):
            metric_names.append(extra_metric)

    aggregate = {
        name: _mean([case["deltas"].get(name, 0.0) for case in comparisons])
        for name in metric_names
    }

    improved = sum(1 for case in comparisons if case["net_quality_delta"] > 1e-6)
    regressed = sum(1 for case in comparisons if case["net_quality_delta"] < -1e-6)
    normalized_quality_gain = _mean([case["net_quality_delta"] for case in comparisons])

    regression_guardrails_tripped: list[str] = []
    if shared_ids:
        base_latency_mean = _mean([_metric_value(baseline_index[cid], "total_latency_ms") for cid in shared_ids])
        cand_latency_mean = _mean([_metric_value(candidate_index[cid], "total_latency_ms") for cid in shared_ids])
        if (
            base_latency_mean > 0
            and max_latency_regression_ratio >= 0
            and (cand_latency_mean - base_latency_mean) / base_latency_mean > max_latency_regression_ratio
        ):
            regression_guardrails_tripped.append("latency_regression_exceeded")

        base_cost_mean = _mean([_metric_value(baseline_index[cid], "estimated_cost") for cid in shared_ids])
        cand_cost_mean = _mean([_metric_value(candidate_index[cid], "estimated_cost") for cid in shared_ids])
        if (
            base_cost_mean > 0
            and max_cost_regression_ratio >= 0
            and (cand_cost_mean - base_cost_mean) / base_cost_mean > max_cost_regression_ratio
        ):
            regression_guardrails_tripped.append("cost_regression_exceeded")

        for cid in shared_ids:
            if not cid.lower().startswith("adv_"):
                continue
            b_metrics = baseline_index[cid].get("metrics") or {}
            c_metrics = candidate_index[cid].get("metrics") or {}
            if "hybrid_score" in b_metrics or "hybrid_score" in c_metrics:
                drop = float(b_metrics.get("hybrid_score") or 0.0) - float(c_metrics.get("hybrid_score") or 0.0)
                if drop > float(max_adversarial_regression):
                    regression_guardrails_tripped.append("adversarial_regression_exceeded")
                    break

    if not comparisons:
        verdict = "inconclusive"
    elif normalized_quality_gain < -1e-6 or regression_guardrails_tripped:
        verdict = "fail"
    elif normalized_quality_gain > 1e-6 and improved >= regressed:
        verdict = "pass"
    else:
        verdict = "inconclusive"

    return {
        "baseline_label": baseline_label or baseline.get("variant") or "baseline",
        "candidate_label": candidate_label or candidate.get("variant") or "candidate",
        "compared_cases": len(comparisons),
        "baseline_only_cases": sorted(set(baseline_index) - set(candidate_index)),
        "candidate_only_cases": sorted(set(candidate_index) - set(baseline_index)),
        "plan_changed_cases": sum(1 for case in comparisons if case["plan_changed"]),
        "improved_cases": improved,
        "regressed_cases": regressed,
        "normalized_quality_gain": normalized_quality_gain,
        "regression_guardrails_tripped": regression_guardrails_tripped,
        "verdict": verdict,
        "aggregate_deltas": aggregate,
        "cases": comparisons,
    }


def _format_delta(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:+g}"


def render_markdown(comparison: Mapping[str, Any]) -> str:
    """Render the variant delta table used as the P0 acceptance artefact."""
    lines = [
        "# Variant Comparison",
        "",
        f"- Baseline: `{comparison.get('baseline_label')}`",
        f"- Candidate: `{comparison.get('candidate_label')}`",
        f"- Compared cases: {comparison.get('compared_cases')}",
        f"- Plan changed cases: {comparison.get('plan_changed_cases')}",
        f"- Improved / regressed cases: {comparison.get('improved_cases')} / {comparison.get('regressed_cases')}",
        f"- Verdict: **{comparison.get('verdict')}**",
        "",
        "## Aggregate deltas (candidate - baseline)",
        "",
        "| Metric | Delta | Direction |",
        "| --- | --- | --- |",
    ]
    aggregate = comparison.get("aggregate_deltas") or {}
    for name in HIGHER_IS_BETTER:
        lines.append(f"| {name} | {_format_delta(aggregate.get(name, 0.0))} | higher is better |")
    for name in LOWER_IS_BETTER:
        lines.append(f"| {name} | {_format_delta(aggregate.get(name, 0.0))} | lower is better |")
    for name in INFORMATIONAL:
        lines.append(f"| {name} | {_format_delta(aggregate.get(name, 0.0))} | informational |")

    lines.extend(
        [
            "",
            "## Per-case",
            "",
            "| Case | Plan Jaccard | reference_match_rate | grounded_bullet_ratio | estimated_cost |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in comparison.get("cases") or []:
        deltas = case.get("deltas") or {}
        lines.append(
            "| {id} | {jaccard} | {ref} | {grounded} | {cost} |".format(
                id=case.get("id"),
                jaccard=case.get("task_title_jaccard"),
                ref=_format_delta(deltas.get("reference_match_rate", 0.0)),
                grounded=_format_delta(deltas.get("grounded_bullet_ratio", 0.0)),
                cost=_format_delta(deltas.get("estimated_cost", 0.0)),
            )
        )

    unmatched = list(comparison.get("baseline_only_cases") or []) + list(
        comparison.get("candidate_only_cases") or []
    )
    if unmatched:
        lines.extend(["", f"> Unmatched cases skipped: {', '.join(unmatched)}"])
    return "\n".join(lines) + "\n"


def compare_result_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    """Load two result files, compare them and optionally persist the artefacts."""
    baseline = _load_payload(baseline_path)
    candidate = _load_payload(candidate_path)
    comparison = build_comparison(baseline, candidate)

    if output_json is not None:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        comparison["output_json"] = str(destination)

    if output_md is not None:
        destination = Path(output_md).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_markdown(comparison), encoding="utf-8")
        comparison["output_md"] = str(destination)

    return comparison


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result files case by case.",
    )
    parser.add_argument("--baseline", required=True, help="Baseline result JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate result JSON path.")
    parser.add_argument("--output-json", default=None, help="Optional comparison JSON output path.")
    parser.add_argument("--output-md", default=None, help="Optional comparison Markdown output path.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    comparison = compare_result_files(
        args.baseline,
        args.candidate,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    sys.stdout.write(render_markdown(comparison) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
