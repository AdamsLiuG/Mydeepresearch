"""CLI entrypoint for offline benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from evals.loader import load_benchmark_cases
from evals.runner import run_benchmark_suite


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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
    result = run_benchmark_suite(
        cases,
        config=config,
        output_path=args.output,
        benchmark_path=args.input,
        request_id_prefix=args.request_id_prefix,
    )

    summary = result["summary"]
    print(f"Benchmark cases: {summary['total_cases']}")
    print(f"Reports generated: {summary['reports_generated']}")
    print(f"Degraded cases: {summary['degraded_cases']}")
    print(f"Error cases: {summary['error_cases']}")
    print(f"Average keyword coverage: {summary['average_keyword_coverage']}")
    print(f"Average section completeness: {summary['average_section_completeness']}")
    print(f"Average latency ms: {summary['average_latency_ms']}")
    print(f"Total estimated cost: {summary['total_estimated_cost']}")
    print(f"Results written to: {result.get('output_path', args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
