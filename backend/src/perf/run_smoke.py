"""CLI entrypoint for the lightweight engineering smoke benchmark."""

from __future__ import annotations

import argparse
import json
import sys

from perf.common import load_scenario, resolve_baseline_path, resolve_output_path
from perf.runtime import execute_http_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the engineering smoke benchmark.")
    parser.add_argument("--profile", default="stub", choices=["stub", "real_local"])
    parser.add_argument("--scenario", default=None, help="Optional path to a smoke scenario JSON file.")
    parser.add_argument("--baseline", default=None, help="Optional path to a baseline JSON file.")
    parser.add_argument("--output", default=None, help="Optional path to the output JSON file.")
    parser.add_argument("--no-enforce", action="store_true", help="Do not fail when baseline checks fail.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = load_scenario("smoke.json", args.scenario)
    payload = execute_http_benchmark(
        mode="smoke",
        profile=args.profile,
        scenario=scenario,
        output_path=resolve_output_path("smoke", args.profile, args.output),
        baseline_path=resolve_baseline_path(args.profile, args.baseline),
        enforce_thresholds=not args.no_enforce,
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

