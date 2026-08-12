"""Case-aligned differential comparison between two benchmark variants (L2 attribution).

The offline benchmark is deterministic and replayable, which makes it possible to
attribute an outcome change to memory *causally*: run the same cases twice, once with
memory disabled (baseline) and once with it enabled (candidate), then diff the
artefacts case by case. Unlike an LLM self-report, this needs no model cooperation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Metrics where a higher value is better; the sign of the delta is interpreted with this.
HIGHER_IS_BETTER = (
    "section_completeness",
    "keyword_coverage",
    "citation_count",
    "citation_marker_count",
    "reference_match_rate",
    "grounded_bullet_ratio",
    "completed_task_count",
)

# Metrics that represent a cost paid; a positive delta is a regression.
LOWER_IS_BETTER = (
    "total_latency_ms",
    "estimated_cost",
)

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
    names = tuple(metric_names or (HIGHER_IS_BETTER + LOWER_IS_BETTER + INFORMATIONAL))

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
        comparisons.append(
            {
                "id": case_id,
                "topic": str(cand_result.get("topic") or base_result.get("topic") or ""),
                "task_title_jaccard": _jaccard(base_titles, cand_titles),
                "plan_changed": base_titles != cand_titles,
                "baseline_error": base_result.get("error"),
                "candidate_error": cand_result.get("error"),
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
) -> dict[str, Any]:
    """Build the full comparison payload, including an aggregate verdict."""
    comparisons = compare_cases(baseline, candidate)
    metric_names = tuple(HIGHER_IS_BETTER + LOWER_IS_BETTER + INFORMATIONAL)
    aggregate = {
        name: _mean([case["deltas"].get(name, 0.0) for case in comparisons])
        for name in metric_names
    }

    improved = sum(
        1
        for case in comparisons
        if any(case["deltas"].get(name, 0.0) > 0 for name in HIGHER_IS_BETTER)
    )
    regressed = sum(
        1
        for case in comparisons
        if any(case["deltas"].get(name, 0.0) < 0 for name in HIGHER_IS_BETTER)
    )
    quality_gain = sum(aggregate.get(name, 0.0) for name in HIGHER_IS_BETTER)

    if not comparisons:
        verdict = "inconclusive"
    elif quality_gain > 0 and improved >= regressed:
        verdict = "pass"
    elif quality_gain < 0:
        verdict = "fail"
    else:
        verdict = "inconclusive"

    baseline_index = _index_results(baseline)
    candidate_index = _index_results(candidate)
    return {
        "baseline_label": baseline_label or baseline.get("variant") or "baseline",
        "candidate_label": candidate_label or candidate.get("variant") or "candidate",
        "compared_cases": len(comparisons),
        "baseline_only_cases": sorted(set(baseline_index) - set(candidate_index)),
        "candidate_only_cases": sorted(set(candidate_index) - set(baseline_index)),
        "plan_changed_cases": sum(1 for case in comparisons if case["plan_changed"]),
        "improved_cases": improved,
        "regressed_cases": regressed,
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
    print(render_markdown(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
