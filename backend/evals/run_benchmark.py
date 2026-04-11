"""CLI entrypoint for offline benchmark runs."""

from __future__ import annotations

import argparse
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
from evals.judges import HeuristicJudge, LLMJudge
from evals.loader import load_benchmark_cases
from evals.runner import run_benchmark_suite


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
        "--judge",
        default="heuristic",
        choices=("heuristic", "llm"),
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
    return HeuristicJudge()


def main() -> int:
    """Run the benchmark suite and print a compact summary."""
    args = parse_args()
    cases = load_benchmark_cases(args.input)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    overrides = {}
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
    )

    summary = result["summary"]
    print(f"Judge: {args.judge}")
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
    print(f"Results written to: {result.get('output_path', args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
