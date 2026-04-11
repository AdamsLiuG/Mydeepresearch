"""Reproduce the full-system eval result artifacts under a controlled output layout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"
REPO_ROOT = BACKEND_ROOT.parent
FRONTEND_ROOT = REPO_ROOT / "frontend"
HTTP_SUITE_PATH = BACKEND_ROOT / "evals" / "run_http_suite.py"
REPORT_METRICS_PATH = BACKEND_ROOT / "evals" / "report_metrics.py"
DEFAULT_RESULTS_DIR = BACKEND_ROOT / "backend" / "evals" / "results"
DEFAULT_REQUEST_STATE_DIR = BACKEND_ROOT / ".state" / "requests"
DEFAULT_BENCHMARK_PATH = BACKEND_ROOT / "evals" / "benchmarks" / "full_system_12cases.jsonl"
DEFAULT_PERF_RESULTS_DIR = BACKEND_ROOT / "perf" / "results"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from perf.runtime import ManagedServer  # noqa: E402


@dataclass(frozen=True)
class ReproductionPaths:
    results_dir: Path
    perf_results_dir: Path
    http_results_json: Path
    interview_summary_md: Path
    metrics_report_json: Path
    metrics_report_md: Path
    smoke_json: Path
    regression_json: Path
    load_json: Path


def build_output_paths(
    *,
    results_dir: Path,
    perf_results_dir: Path,
    profile: str,
    tag: str,
) -> ReproductionPaths:
    safe_tag = str(tag or "").strip() or "run"
    safe_profile = str(profile or "").strip() or "real_local"
    return ReproductionPaths(
        results_dir=results_dir,
        perf_results_dir=perf_results_dir,
        http_results_json=results_dir / f"full_system_http_results_{safe_profile}_{safe_tag}.json",
        interview_summary_md=results_dir / f"full_system_interview_summary_{safe_profile}_{safe_tag}.md",
        metrics_report_json=results_dir / f"project_metrics_report_{safe_profile}_{safe_tag}.json",
        metrics_report_md=results_dir / f"project_metrics_report_{safe_profile}_{safe_tag}.md",
        smoke_json=perf_results_dir / f"smoke_{safe_profile}_{safe_tag}.json",
        regression_json=perf_results_dir / f"regression_{safe_profile}_{safe_tag}.json",
        load_json=perf_results_dir / f"load_{safe_profile}_{safe_tag}.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the full-system eval result files and optional project metrics report.",
    )
    parser.add_argument("--profile", default="real_local", choices=["stub", "real_local"])
    parser.add_argument("--tag", required=True, help="Run tag used in output filenames, for example 20260329.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Target directory for the reproduced eval result files.",
    )
    parser.add_argument(
        "--perf-results-dir",
        default=str(DEFAULT_PERF_RESULTS_DIR),
        help="Target directory for generated perf JSON files.",
    )
    parser.add_argument(
        "--request-state-dir",
        default=str(DEFAULT_REQUEST_STATE_DIR),
        help="Request snapshot directory used by the managed backend process.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Benchmark JSONL used by the HTTP suite.",
    )
    parser.add_argument("--search-api", default=None, help="Optional search backend override.")
    parser.add_argument("--limit", type=int, default=0, help="Optional benchmark case limit.")
    parser.add_argument(
        "--request-id-prefix",
        default=None,
        help="Optional explicit request id prefix. Defaults to <profile>-<tag>.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Use an already running backend instead of starting a managed local server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional port for the managed local backend. Defaults to an auto-selected free port.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=60.0,
        help="How long to wait for the managed backend to become healthy.",
    )
    parser.add_argument("--skip-frontend-build", action="store_true", help="Skip frontend npm ci + npm run build.")
    parser.add_argument("--skip-perf", action="store_true", help="Skip smoke/regression/load perf runs.")
    parser.add_argument("--skip-metrics-report", action="store_true", help="Skip project metrics report generation.")
    return parser.parse_args()


def _resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = ((base or REPO_ROOT) / candidate).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _run(cmd: list[str], *, workdir: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(workdir), env=env, check=True)


def _build_frontend() -> None:
    _run(["npm", "ci"], workdir=FRONTEND_ROOT)
    _run(["npm", "run", "build"], workdir=FRONTEND_ROOT)


def _run_perf_suite(*, profile: str, paths: ReproductionPaths) -> list[Path]:
    commands = [
        (
            [
                sys.executable,
                "-m",
                "perf.run_smoke",
                "--profile",
                profile,
                "--output",
                str(paths.smoke_json),
            ],
            paths.smoke_json,
        ),
        (
            [
                sys.executable,
                "-m",
                "perf.run_regression",
                "--profile",
                profile,
                "--output",
                str(paths.regression_json),
            ],
            paths.regression_json,
        ),
        (
            [
                sys.executable,
                "-m",
                "perf.run_load",
                "--profile",
                profile,
                "--output",
                str(paths.load_json),
                "--users",
                "2",
                "--spawn-rate",
                "1",
                "--duration",
                "10s",
            ],
            paths.load_json,
        ),
    ]

    generated: list[Path] = []
    for cmd, output_path in commands:
        _run(cmd, workdir=BACKEND_ROOT)
        if output_path.exists():
            generated.append(output_path)
    return generated


def _run_http_suite(
    *,
    base_url: str,
    profile: str,
    benchmark_path: Path,
    request_id_prefix: str,
    paths: ReproductionPaths,
    perf_outputs: Sequence[Path],
    search_api: str | None,
    limit: int,
) -> None:
    cmd = [
        sys.executable,
        str(HTTP_SUITE_PATH),
        "--base-url",
        base_url,
        "--mode",
        "both",
        "--perf-profile",
        profile,
        "--request-id-prefix",
        request_id_prefix,
        "--input",
        str(benchmark_path),
        "--output",
        str(paths.http_results_json),
        "--summary-md",
        str(paths.interview_summary_md),
    ]
    if search_api:
        cmd.extend(["--search-api", search_api])
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    for perf_output in perf_outputs:
        cmd.extend(["--perf-result", str(perf_output)])

    _run(cmd, workdir=BACKEND_ROOT)


def _run_metrics_report(
    *,
    request_state_dir: Path,
    eval_result_path: Path,
    paths: ReproductionPaths,
) -> None:
    cmd = [
        sys.executable,
        str(REPORT_METRICS_PATH),
        "--request-state-dir",
        str(request_state_dir),
        "--eval-result",
        str(eval_result_path),
        "--output-json",
        str(paths.metrics_report_json),
        "--output-md",
        str(paths.metrics_report_md),
    ]
    _run(cmd, workdir=BACKEND_ROOT)


def main() -> int:
    args = parse_args()

    results_dir = _resolve_path(args.results_dir)
    perf_results_dir = _resolve_path(args.perf_results_dir)
    request_state_dir = _resolve_path(args.request_state_dir)
    benchmark_path = _resolve_path(args.input)
    paths = build_output_paths(
        results_dir=results_dir,
        perf_results_dir=perf_results_dir,
        profile=args.profile,
        tag=args.tag,
    )
    request_id_prefix = args.request_id_prefix or f"{args.profile}-{args.tag}"

    if not args.skip_frontend_build:
        _build_frontend()

    perf_outputs: list[Path] = []
    if not args.skip_perf:
        perf_outputs = _run_perf_suite(profile=args.profile, paths=paths)

    if args.base_url:
        base_url = args.base_url
        _run_http_suite(
            base_url=base_url,
            profile=args.profile,
            benchmark_path=benchmark_path,
            request_id_prefix=request_id_prefix,
            paths=paths,
            perf_outputs=perf_outputs,
            search_api=args.search_api,
            limit=args.limit,
        )
    else:
        extra_env = {
            "REQUEST_STATE_ENABLED": "True",
            "REQUEST_STATE_DIR": str(request_state_dir),
            "METRICS_RECENT_REQUESTS_LIMIT": "256",
            "BENCHMARK_PROFILE": args.profile,
            "BENCHMARK_STUB_ENABLED": "True" if args.profile == "stub" else "False",
        }
        with ManagedServer(
            profile=args.profile,
            port=args.port,
            extra_env=extra_env,
            startup_timeout_seconds=args.startup_timeout_seconds,
        ) as server:
            _run_http_suite(
                base_url=server.base_url,
                profile=args.profile,
                benchmark_path=benchmark_path,
                request_id_prefix=request_id_prefix,
                paths=paths,
                perf_outputs=perf_outputs,
                search_api=args.search_api,
                limit=args.limit,
            )

    if not args.skip_metrics_report:
        _run_metrics_report(
            request_state_dir=request_state_dir,
            eval_result_path=paths.http_results_json,
            paths=paths,
        )

    sys.stdout.write(f"HTTP suite JSON: {paths.http_results_json}\n")
    sys.stdout.write(f"Interview summary: {paths.interview_summary_md}\n")
    if perf_outputs:
        sys.stdout.write("Perf outputs:\n")
        for perf_output in perf_outputs:
            sys.stdout.write(f"- {perf_output}\n")
    if not args.skip_metrics_report:
        sys.stdout.write(f"Metrics report JSON: {paths.metrics_report_json}\n")
        sys.stdout.write(f"Metrics report MD: {paths.metrics_report_md}\n")
    if args.base_url:
        sys.stdout.write(f"Used existing backend: {args.base_url}\n")
    else:
        sys.stdout.write(f"Managed request state dir: {request_state_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
