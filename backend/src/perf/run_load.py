"""CLI entrypoint for Locust-based engineering load benchmarks."""

from __future__ import annotations

import argparse
import json
import sys

from perf.common import load_scenario, resolve_baseline_path, resolve_output_path
from perf.runtime import execute_locust_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Locust engineering load benchmark.")
    parser.add_argument("--profile", default="stub", choices=["stub", "real_local"])
    parser.add_argument("--scenario", default=None, help="Optional path to a load scenario JSON file.")
    parser.add_argument("--baseline", default=None, help="Optional path to a baseline JSON file.")
    parser.add_argument("--output", default=None, help="Optional path to the output JSON file.")
    parser.add_argument("--users", type=int, default=None, help="Override the number of Locust users.")
    parser.add_argument("--spawn-rate", type=float, default=None, help="Override the Locust spawn rate.")
    parser.add_argument("--duration", default=None, help="Override the Locust run duration (for example 10s).")
    parser.add_argument(
        "--enforce-baseline",
        action="store_true",
        help="Fail the command when regression checks fail.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = load_scenario("load.json", args.scenario)
    if args.users is not None:
        scenario["users"] = args.users
    if args.spawn_rate is not None:
        scenario["spawn_rate"] = args.spawn_rate
    if args.duration is not None:
        scenario["duration"] = args.duration

    payload = execute_locust_benchmark(
        profile=args.profile,
        scenario=scenario,
        output_path=resolve_output_path("load", args.profile, args.output),
        baseline_path=resolve_baseline_path(args.profile, args.baseline),
        enforce_thresholds=args.enforce_baseline,
    )
    sys.stdout.write(
        json.dumps(
            {
                "output_path": payload["output_path"],
                "summary": payload["summary"],
                "baseline_comparison": payload.get("baseline_comparison"),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

