"""Runtime helpers used by perf CLI entrypoints."""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Thread
from typing import Any

import requests

from config import backend_root
from perf.common import (
    build_result_payload,
    build_summary,
    compare_against_baseline,
    load_json,
    resolve_baseline_path,
    resolve_output_path,
    update_baseline,
    write_json,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_local(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Issue a localhost request without inheriting proxy settings from the environment."""
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, url, **kwargs)


class ResourceSampler:
    """Collect coarse CPU and RSS samples for the benchmarked server process."""

    def __init__(self, pid: int, *, interval_seconds: float) -> None:
        self.pid = pid
        self.interval_seconds = max(interval_seconds, 0.05)
        self.samples: list[dict[str, Any]] = []
        self.available = False
        self._process: Any | None = None
        self._psutil: Any | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._started_at = 0.0

        try:
            import psutil  # type: ignore
        except ImportError:
            return

        self._psutil = psutil
        self._process = psutil.Process(pid)
        self.available = True

    def start(self) -> None:
        if not self.available or self._process is None:
            return

        self._started_at = time.perf_counter()
        self._process.cpu_percent(interval=None)
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        if not self.available:
            return []

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

        self._sample_once()
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        if not self.available or self._process is None:
            return

        try:
            cpu_percent = float(self._process.cpu_percent(interval=None))
            rss_mb = float(self._process.memory_info().rss) / (1024 * 1024)
        except Exception:
            return

        self.samples.append(
            {
                "elapsed_s": round(time.perf_counter() - self._started_at, 3),
                "cpu_percent": round(cpu_percent, 2),
                "rss_mb": round(rss_mb, 2),
            }
        )


class ManagedServer:
    """Launch and manage a disposable uvicorn subprocess for benchmarks."""

    def __init__(
        self,
        *,
        profile: str,
        port: int | None = None,
        extra_env: dict[str, str] | None = None,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        self.profile = profile
        self.requested_port = port
        self.port = port or 0
        self.extra_env = extra_env or {}
        self.startup_timeout_seconds = startup_timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self.log_file: tempfile.NamedTemporaryFile[str] | None = None
        self.startup_attempts = 1 if port is not None else 3

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def log_path(self) -> str | None:
        return self.log_file.name if self.log_file is not None else None

    def __enter__(self) -> ManagedServer:
        last_error: Exception | None = None
        for _attempt in range(self.startup_attempts):
            self.port = self.requested_port or _find_free_port()

            env = os.environ.copy()
            env["PYTHONPATH"] = str(backend_root() / "src")
            if os.environ.get("PYTHONPATH"):
                env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + os.environ["PYTHONPATH"]
            env["HOST"] = "127.0.0.1"
            env["PORT"] = str(self.port)
            env["LOG_LEVEL"] = "WARNING"
            env["ENABLE_NOTES"] = self.extra_env.get("ENABLE_NOTES", "False")
            env["BENCHMARK_PROFILE"] = self.profile
            env["BENCHMARK_STUB_ENABLED"] = "True" if self.profile == "stub" else "False"
            env.update(self.extra_env)

            self.log_file = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix=f"perf-server-{self.profile}-",
                suffix=".log",
                delete=False,
            )
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "main:app",
                    "--app-dir",
                    str(backend_root() / "src"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.port),
                    "--log-level",
                    "warning",
                ],
                cwd=str(backend_root()),
                env=env,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                self._wait_until_ready()
                return self
            except Exception as exc:
                last_error = exc
                self.__exit__(None, None, None)

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"Benchmark server failed to start for profile={self.profile}")

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self.process is None:
            return

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_seconds
        last_error: str | None = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                break
            try:
                response = _request_local("GET", f"{self.base_url}/healthz", timeout=1)
                if response.ok:
                    return
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(0.2)

        raise RuntimeError(
            f"Benchmark server failed to start for profile={self.profile}. "
            f"Last error: {last_error or 'unavailable'}; log={self.log_path}"
        )


def _invoke_research_request(
    *,
    base_url: str,
    topic: str,
    timeout_seconds: float,
    search_api: str | None,
    request_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"topic": topic}
    if search_api:
        payload["search_api"] = search_api

    started_at = time.perf_counter()
    try:
        response = _request_local(
            "POST",
            f"{base_url}/research",
            json=payload,
            timeout=timeout_seconds,
            headers={"X-Request-ID": f"perf-{request_index:04d}"},
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "request_index": request_index,
            "topic": topic,
            "status_code": response.status_code,
            "success": response.ok,
            "latency_ms": latency_ms,
            "report_length": len(str(body.get("report_markdown") or "")),
            "todo_count": len(body.get("todo_items") or []),
            "error": None if response.ok else str(body or response.text[:300]),
        }
    except requests.RequestException as exc:
        return {
            "request_index": request_index,
            "topic": topic,
            "status_code": 0,
            "success": False,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "report_length": 0,
            "todo_count": 0,
            "error": str(exc),
        }


def run_http_requests(
    *,
    base_url: str,
    topics: list[str],
    requests_total: int,
    concurrency: int,
    timeout_seconds: float,
    search_api: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    if not topics:
        topics = ["Engineering benchmark smoke test"]

    request_payloads = [
        {
            "topic": topics[index % len(topics)],
            "request_index": index,
        }
        for index in range(requests_total)
    ]

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [
            executor.submit(
                _invoke_research_request,
                base_url=base_url,
                topic=payload["topic"],
                timeout_seconds=timeout_seconds,
                search_api=search_api,
                request_index=payload["request_index"],
            )
            for payload in request_payloads
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_duration_s = time.perf_counter() - started_at
    results.sort(key=lambda item: int(item.get("request_index") or 0))
    return results, total_duration_s


def validate_stream_endpoint(
    *,
    base_url: str,
    topic: str,
    timeout_seconds: float,
    search_api: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"topic": topic}
    if search_api:
        payload["search_api"] = search_api

    event_types: list[str] = []
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.post(
                f"{base_url}/research/stream",
                json=payload,
                stream=True,
                timeout=timeout_seconds,
                headers={"X-Request-ID": "perf-stream-check"},
            ) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
                    if not text.startswith("data: "):
                        continue
                    event = json.loads(text[6:])
                    event_type = str(event.get("type") or "")
                    if event_type:
                        event_types.append(event_type)
        success = "final_report" in event_types and "done" in event_types
        return {
            "success": success,
            "event_types": event_types,
        }
    except Exception as exc:
        return {
            "success": False,
            "event_types": event_types,
            "error": str(exc),
        }


def fetch_metrics_snapshot(base_url: str) -> dict[str, Any]:
    try:
        response = _request_local("GET", f"{base_url}/metrics/json", timeout=5)
        if response.ok:
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
    except requests.RequestException:
        return {}
    return {}


def _maybe_start_tracemalloc(enabled: bool) -> None:
    if enabled:
        tracemalloc.start()


def _maybe_stop_tracemalloc(enabled: bool) -> float | None:
    if not enabled:
        return None

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak / (1024 * 1024), 4)


def execute_http_benchmark(
    *,
    mode: str,
    profile: str,
    scenario: dict[str, Any],
    output_path: str,
    baseline_path: str | None = None,
    enforce_thresholds: bool = False,
    write_baseline: bool = False,
    enable_tracemalloc: bool = False,
) -> dict[str, Any]:
    concurrency = int(scenario.get("concurrency") or 1)
    requests_total = int(scenario.get("requests_total") or 1)
    timeout_seconds = float(scenario.get("timeout_seconds") or 120.0)
    endpoint = str(scenario.get("endpoint") or "/research")
    search_api = scenario.get("search_api")
    topics = list(scenario.get("topics") or [])
    extra_env = {str(key): str(value) for key, value in (scenario.get("server_env") or {}).items()}
    sample_interval = float(scenario.get("sample_interval_seconds") or 0.5)

    with ManagedServer(profile=profile, extra_env=extra_env) as server:
        stream_validation = None
        if scenario.get("validate_stream"):
            stream_validation = validate_stream_endpoint(
                base_url=server.base_url,
                topic=topics[0] if topics else "Engineering benchmark stream validation",
                timeout_seconds=timeout_seconds,
                search_api=search_api,
            )

        sampler = ResourceSampler(server.process.pid if server.process else 0, interval_seconds=sample_interval)
        sampler.start()
        _maybe_start_tracemalloc(enable_tracemalloc)
        request_records, total_duration_s = run_http_requests(
            base_url=server.base_url,
            topics=topics,
            requests_total=requests_total,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            search_api=search_api,
        )
        tracemalloc_peak_mb = _maybe_stop_tracemalloc(enable_tracemalloc)
        resource_samples = sampler.stop()
        metrics_snapshot = fetch_metrics_snapshot(server.base_url)

        cpu_samples = [float(sample.get("cpu_percent") or 0.0) for sample in resource_samples]
        rss_samples = [float(sample.get("rss_mb") or 0.0) for sample in resource_samples]
        summary = build_summary(
            mode=mode,
            profile=profile,
            endpoint=endpoint,
            concurrency=concurrency,
            request_records=request_records,
            total_duration_s=total_duration_s,
            cpu_samples=cpu_samples,
            rss_samples=rss_samples,
            estimated_cost_total=float(metrics_snapshot.get("estimated_cost") or 0.0),
            extra={"runner_tracemalloc_peak_mb": tracemalloc_peak_mb},
        )
        baseline_payload = load_json(resolve_baseline_path(profile, baseline_path)) if baseline_path else None
        comparison = compare_against_baseline(
            summary=summary,
            baseline_payload=baseline_payload,
            mode=mode,
        )
        payload = build_result_payload(
            mode=mode,
            profile=profile,
            endpoint=endpoint,
            concurrency=concurrency,
            summary=summary,
            scenario=scenario,
            request_records=request_records,
            resource_samples=resource_samples,
            metrics_snapshot=metrics_snapshot,
            stream_validation=stream_validation,
            baseline_comparison=comparison,
            server_log_path=server.log_path,
        )

    output_file = write_json(resolve_output_path(mode, profile, output_path), payload)
    payload["output_path"] = output_file

    if write_baseline:
        baseline_file = update_baseline(resolve_baseline_path(profile, baseline_path), payload)
        payload["baseline_output_path"] = baseline_file

    if enforce_thresholds and comparison and not comparison.get("passed", True):
        failed_metrics = ", ".join(comparison.get("failed_metrics") or [])
        raise RuntimeError(f"{mode} benchmark failed baseline checks: {failed_metrics or 'unknown'}")

    return payload


def parse_locust_stats_csv(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}

    aggregated = next(
        (
            row
            for row in rows
            if str(row.get("Name") or "").lower() == "aggregated"
        ),
        rows[-1],
    )

    request_total = int(float(aggregated.get("Request Count") or 0))
    failure_total = int(float(aggregated.get("Failure Count") or 0))
    return {
        "requests_total": request_total,
        "failure_total": failure_total,
        "success_rate": round((request_total - failure_total) / request_total, 4)
        if request_total
        else 0.0,
        "error_rate": round(failure_total / request_total, 4) if request_total else 0.0,
        "rps": round(float(aggregated.get("Requests/s") or 0.0), 4),
        "avg_latency_ms": round(float(aggregated.get("Average Response Time") or 0.0), 2),
        "p50_latency_ms": round(
            float(aggregated.get("50%") or aggregated.get("Median Response Time") or 0.0),
            2,
        ),
        "p95_latency_ms": round(float(aggregated.get("95%") or 0.0), 2),
        "p99_latency_ms": round(float(aggregated.get("99%") or 0.0), 2),
    }


def execute_locust_benchmark(
    *,
    profile: str,
    scenario: dict[str, Any],
    output_path: str,
    baseline_path: str | None = None,
    enforce_thresholds: bool = False,
) -> dict[str, Any]:
    users = int(scenario.get("users") or 2)
    spawn_rate = float(scenario.get("spawn_rate") or 1.0)
    duration = str(scenario.get("duration") or "10s")
    timeout_seconds = float(scenario.get("timeout_seconds") or 120.0)
    extra_env = {str(key): str(value) for key, value in (scenario.get("server_env") or {}).items()}
    sample_interval = float(scenario.get("sample_interval_seconds") or 0.5)
    topic = str((scenario.get("topics") or ["Engineering benchmark load test"])[0])

    with ManagedServer(profile=profile, extra_env=extra_env) as server:
        sampler = ResourceSampler(server.process.pid if server.process else 0, interval_seconds=sample_interval)
        sampler.start()
        temp_dir = Path(tempfile.mkdtemp(prefix="perf-locust-"))
        csv_prefix = temp_dir / "locust"
        locust_env = os.environ.copy()
        locust_env["LOCUST_RESEARCH_TOPIC"] = topic
        if scenario.get("search_api"):
            locust_env["LOCUST_SEARCH_API"] = str(scenario["search_api"])

        command = [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(backend_root() / "src" / "perf" / "locustfile.py"),
            "--host",
            server.base_url,
            "--headless",
            "--users",
            str(users),
            "--spawn-rate",
            str(spawn_rate),
            "--run-time",
            duration,
            "--csv",
            str(csv_prefix),
            "--only-summary",
        ]
        completed = subprocess.run(
            command,
            cwd=str(backend_root()),
            env=locust_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_seconds), 30),
        )
        resource_samples = sampler.stop()
        metrics_snapshot = fetch_metrics_snapshot(server.base_url)

        if completed.returncode != 0:
            raise RuntimeError(
                f"Locust benchmark failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        stats = parse_locust_stats_csv(csv_prefix.with_name(csv_prefix.name + "_stats.csv"))
        cpu_samples = [float(sample.get("cpu_percent") or 0.0) for sample in resource_samples]
        rss_samples = [float(sample.get("rss_mb") or 0.0) for sample in resource_samples]
        cpu_summary = build_summary(
            mode="load",
            profile=profile,
            endpoint="/research",
            concurrency=users,
            request_records=[],
            total_duration_s=max(float(stats.get("requests_total") or 0) / max(float(stats.get("rps") or 0.001), 0.001), 0.001),
            cpu_samples=cpu_samples,
            rss_samples=rss_samples,
            estimated_cost_total=float(metrics_snapshot.get("estimated_cost") or 0.0),
        )
        summary = {
            **cpu_summary,
            **stats,
            "mode": "load",
            "profile": profile,
            "endpoint": "/research",
            "concurrency": users,
            "cpu_percent_avg": cpu_summary.get("cpu_percent_avg"),
            "cpu_percent_peak": cpu_summary.get("cpu_percent_peak"),
            "rss_mb_avg": cpu_summary.get("rss_mb_avg"),
            "rss_mb_peak": cpu_summary.get("rss_mb_peak"),
            "estimated_cost_total": round(float(metrics_snapshot.get("estimated_cost") or 0.0), 6),
            "estimated_cost_per_request": round(
                float(metrics_snapshot.get("estimated_cost") or 0.0) / max(int(stats.get("requests_total") or 0), 1),
                6,
            )
            if stats.get("requests_total")
            else 0.0,
        }
        baseline_payload = load_json(resolve_baseline_path(profile, baseline_path)) if baseline_path else None
        comparison = compare_against_baseline(
            summary=summary,
            baseline_payload=baseline_payload,
            mode="regression",
        )
        payload = build_result_payload(
            mode="load",
            profile=profile,
            endpoint="/research",
            concurrency=users,
            summary=summary,
            scenario=scenario,
            request_records=[],
            resource_samples=resource_samples,
            metrics_snapshot=metrics_snapshot,
            baseline_comparison=comparison,
            server_log_path=server.log_path,
            extra={
                "locust_stdout": completed.stdout.strip(),
                "locust_stderr": completed.stderr.strip(),
                "locust_stats_csv": str(csv_prefix.with_name(csv_prefix.name + "_stats.csv")),
            },
        )

    output_file = write_json(resolve_output_path("load", profile, output_path), payload)
    payload["output_path"] = output_file

    if enforce_thresholds and comparison and not comparison.get("passed", True):
        failed_metrics = ", ".join(comparison.get("failed_metrics") or [])
        raise RuntimeError(f"load benchmark failed baseline checks: {failed_metrics or 'unknown'}")

    return payload
