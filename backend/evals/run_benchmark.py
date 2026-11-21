"""CLI entrypoint for offline benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from evals.compare_variants import compare_result_files, render_markdown
from evals.judges import HeuristicJudge, HybridJudge, LLMJudge
from evals.loader import load_benchmark_cases
from evals.runner import run_benchmark_suite

VARIANT_DIR = BACKEND_ROOT / "evals" / "variants"


def load_variant(name: str) -> dict:
    """Load a named variant definition from evals/variants."""
    path = VARIANT_DIR / f"{name}.json"
    if not path.is_file():
        available = sorted(item.stem for item in VARIANT_DIR.glob("*.json"))
        raise SystemExit(
            f"unknown variant {name!r}; available: {', '.join(available) or 'none'}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = payload.get("config_overrides")
    if not isinstance(overrides, dict):
        raise SystemExit(f"variant {name!r} is missing a config_overrides object")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run offline benchmark cases.")
    parser.add_argument(
        "--input",
        default=str(BACKEND_ROOT / "evals" / "benchmarks" / "sample_benchmark.jsonl"),
        help="Path to benchmark .json or .jsonl file.",
    )
    parser.add_argument(
        "--output",
        default=str(BACKEND_ROOT / "evals" / "results" / "latest_results.json"),
        help="Path to the result JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional case limit for quick smoke runs.",
    )
    parser.add_argument(
        "--search-api",
        default=None,
        help="Optional search backend override for the entire suite.",
    )
    parser.add_argument(
        "--request-id-prefix",
        default="eval",
        help="Prefix added to per-case request ids.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional named variant from evals/variants applied as config overrides.",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Optional baseline result JSON path; emits a case-aligned delta report.",
    )
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Optional path for the comparison Markdown report (implies --compare-to).",
    )
    parser.add_argument(
        "--judge",
        default="heuristic",
        choices=("heuristic", "llm", "hybrid"),
        help="Benchmark judge to use for result scoring.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional model override for --judge llm.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Optional provider override for --judge llm.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL override for --judge llm.",
    )
    parser.add_argument(
        "--judge-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout override for --judge llm.",
    )
    parser.add_argument(
        "--judge-version",
        default=None,
        help="Optional prompt / rubric version label for --judge llm.",
    )
    return parser.parse_args(argv)


def build_judge(args: argparse.Namespace, *, config: Configuration):
    """Construct the selected benchmark judge implementation."""
    if args.judge == "llm":
        return LLMJudge(
            config=config,
            model=args.judge_model,
            provider=args.judge_provider,
            base_url=args.judge_base_url,
            timeout_seconds=args.judge_timeout_seconds,
            judge_version=args.judge_version,
            load_env_file=False,
        )
    if args.judge == "hybrid":
        llm_judge = LLMJudge(
            config=config,
            model=args.judge_model,
            provider=args.judge_provider,
            base_url=args.judge_base_url,
            timeout_seconds=args.judge_timeout_seconds,
            judge_version=args.judge_version,
            load_env_file=False,
        )
        return HybridJudge(heuristic_judge=HeuristicJudge(), llm_judge=llm_judge)
    return HeuristicJudge()


def main() -> int:
    """Run the benchmark suite and print a compact summary."""
    args = parse_args()
    cases = load_benchmark_cases(args.input)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    overrides = {}
    variant_name: str | None = None
    if args.variant:
        variant = load_variant(args.variant)
        variant_name = str(variant.get("name") or args.variant)
        overrides.update(variant["config_overrides"])
    if args.search_api:
        overrides["search_api"] = args.search_api

    config = Configuration.from_env(
        overrides=overrides or None,
        load_env_file=True,
    )
    judge = build_judge(args, config=config)
    result = run_benchmark_suite(
        cases,
        config=config,
        judge=judge,
        output_path=args.output,
        benchmark_path=args.input,
        request_id_prefix=args.request_id_prefix,
        variant=variant_name,
    )

    summary = result["summary"]
    print(f"Judge: {args.judge}")
    if variant_name:
        print(f"Variant: {variant_name}")
    print(f"Benchmark cases: {summary['total_cases']}")
    print(f"Reports generated: {summary['reports_generated']}")
    print(f"Degraded cases: {summary['degraded_cases']}")
    print(f"Error cases: {summary['error_cases']}")
    if args.judge == "llm":
        print(f"Judge success cases: {summary['judge_success_cases']}")
        print(f"Judge error cases: {summary['judge_error_cases']}")
        print(f"Judge skipped cases: {summary['judge_skipped_cases']}")
        print(f"Judge pass / warning / fail: {summary['judge_pass_cases']} / {summary['judge_warning_cases']} / {summary['judge_fail_cases']}")
        print(f"Average factuality score: {summary['average_factuality_score']}")
        print(f"Average coverage score: {summary['average_coverage_score']}")
        print(f"Average citation grounding score: {summary['average_citation_grounding_score']}")
        print(f"Average freshness score: {summary['average_freshness_score']}")
        print(f"Average conservativeness score: {summary['average_conservativeness_score']}")
    else:
        print(f"Average keyword coverage: {summary['average_keyword_coverage']}")
        print(f"Average section completeness: {summary['average_section_completeness']}")
    print(f"Average latency ms: {summary['average_latency_ms']}")
    print(f"Total estimated cost: {summary['total_estimated_cost']}")
    print(f"Average memory hits: {summary['average_memory_hit_count']}")
    print(f"Cases with memory injected: {summary['cases_with_memory_injection']}")
    print(f"Shadow-mode cases: {summary['memory_shadow_cases']}")
    print(f"Results written to: {result.get('output_path', args.output)}")

    if args.compare_to:
        comparison = compare_result_files(
            args.compare_to,
            result.get("output_path", args.output),
            output_md=args.comparison_output,
        )
        print()
        print(render_markdown(comparison))
        if args.comparison_output:
            print(f"Comparison written to: {comparison.get('output_md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
