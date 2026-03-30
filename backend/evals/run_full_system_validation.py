"""Convenience wrapper for the real-path full-system validation workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
FRONTEND_ROOT = REPO_ROOT / "frontend"
HTTP_SUITE_PATH = BACKEND_ROOT / "evals" / "run_http_suite.py"


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the orchestration wrapper."""
    parser = argparse.ArgumentParser(description="Run the full-system validation workflow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Running backend base URL.")
    parser.add_argument("--input", default=None, help="Optional benchmark file path passed to run_http_suite.py.")
    parser.add_argument("--output", default=None, help="Optional JSON output path passed to run_http_suite.py.")
    parser.add_argument("--summary-md", default=None, help="Optional Markdown summary path passed to run_http_suite.py.")
    parser.add_argument("--search-api", default=None, help="Optional search_api override for HTTP suite requests.")
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for quick runs.")
    parser.add_argument("--request-id-prefix", default="full-system", help="X-Request-ID prefix.")
    parser.add_argument("--perf-profile", default="real_local", help="Perf profile for smoke/regression/load.")
    parser.add_argument("--skip-frontend-build", action="store_true", help="Skip frontend npm ci + npm run build.")
    parser.add_argument("--skip-perf", action="store_true", help="Skip perf smoke/regression/load commands.")
    return parser.parse_args()


def _run(cmd: list[str], *, workdir: Path) -> None:
    subprocess.run(cmd, cwd=str(workdir), check=True)


def main() -> int:
    """Run frontend build, perf checks, and the HTTP suite in order."""
    args = parse_args()

    if not args.skip_frontend_build:
        _run(["npm", "ci"], workdir=FRONTEND_ROOT)
        _run(["npm", "run", "build"], workdir=FRONTEND_ROOT)

    if not args.skip_perf:
        _run([sys.executable, "-m", "perf.run_smoke", "--profile", args.perf_profile], workdir=BACKEND_ROOT)
        _run([sys.executable, "-m", "perf.run_regression", "--profile", args.perf_profile], workdir=BACKEND_ROOT)
        _run(
            [
                sys.executable,
                "-m",
                "perf.run_load",
                "--profile",
                args.perf_profile,
                "--users",
                "2",
                "--spawn-rate",
                "1",
                "--duration",
                "10s",
            ],
            workdir=BACKEND_ROOT,
        )

    cmd = [
        sys.executable,
        str(HTTP_SUITE_PATH),
        "--base-url",
        args.base_url,
        "--mode",
        "both",
        "--perf-profile",
        args.perf_profile,
        "--request-id-prefix",
        args.request_id_prefix,
    ]
    if args.input:
        cmd.extend(["--input", args.input])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.summary_md:
        cmd.extend(["--summary-md", args.summary_md])
    if args.search_api:
        cmd.extend(["--search-api", args.search_api])
    if args.limit and args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    _run(cmd, workdir=BACKEND_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
