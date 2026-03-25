"""Shared helpers for engineering benchmark runners."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import backend_root


def perf_root() -> Path:
    return backend_root() / "perf"


def scenarios_dir() -> Path:
    return perf_root() / "scenarios"


def baselines_dir() -> Path:
    return perf_root() / "baselines"


def results_dir() -> Path:
    path = perf_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: str | Path, payload: dict[str, Any]) -> str:
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = backend_root() / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(destination.resolve())


def resolve_output_path(mode: str, profile: str, output_path: str | None = None) -> str:
    if output_path:
        return str(output_path)
    return str(results_dir() / f"{mode}-{profile}.json")


def resolve_baseline_path(profile: str, baseline_path: str | None = None) -> str:
    if baseline_path:
        return str(baseline_path)
    return str(baselines_dir() / f"{profile}_baseline.json")


def load_scenario(default_name: str, scenario_path: str | None = None) -> dict[str, Any]:
    source = Path(scenario_path).expanduser() if scenario_path else scenarios_dir() / default_name
    if not source.is_absolute():
        source = backend_root() / source
    return load_json(source)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)

    rank = ((pct or 0.0) / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[int(rank)], 2)

    fraction = rank - lower
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(interpolated, 2)


def summarize_series(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "peak": None}

    avg_value = round(sum(values) / len(values), 2)
    peak_value = round(max(values), 2)
    return {"avg": avg_value, "peak": peak_value}


def build_summary(
    *,
    mode: str,
    profile: str,
    endpoint: str,
    concurrency: int,
    request_records: list[dict[str, Any]],
    total_duration_s: float,
    cpu_samples: list[float],
    rss_samples: list[float],
    estimated_cost_total: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latencies = [float(record.get("latency_ms") or 0.0) for record in request_records]
    request_total = len(request_records)
    success_total = sum(1 for record in request_records if record.get("success"))
    error_total = request_total - success_total
    safe_duration = max(total_duration_s, 0.001)
    cpu_summary = summarize_series(cpu_samples)
    rss_summary = summarize_series(rss_samples)

    summary = {
        "mode": mode,
        "profile": profile,
        "endpoint": endpoint,
        "concurrency": concurrency,
        "requests_total": request_total,
        "success_rate": round((success_total / request_total), 4) if request_total else 0.0,
        "error_rate": round((error_total / request_total), 4) if request_total else 0.0,
        "rps": round((request_total / safe_duration), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "cpu_percent_avg": cpu_summary["avg"],
        "cpu_percent_peak": cpu_summary["peak"],
        "rss_mb_avg": rss_summary["avg"],
        "rss_mb_peak": rss_summary["peak"],
        "estimated_cost_total": round(float(estimated_cost_total or 0.0), 6),
        "estimated_cost_per_request": round(
            float(estimated_cost_total or 0.0) / request_total,
            6,
        )
        if request_total
        else 0.0,
    }

    if extra:
        summary.update(extra)

    return summary


def build_result_payload(
    *,
    mode: str,
    profile: str,
    endpoint: str,
    concurrency: int,
    summary: dict[str, Any],
    scenario: dict[str, Any],
    request_records: list[dict[str, Any]] | None = None,
    resource_samples: list[dict[str, Any]] | None = None,
    metrics_snapshot: dict[str, Any] | None = None,
    stream_validation: dict[str, Any] | None = None,
    baseline_comparison: dict[str, Any] | None = None,
    server_log_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "generated_at": utc_now(),
        "mode": mode,
        "profile": profile,
        "endpoint": endpoint,
        "concurrency": concurrency,
        "summary": summary,
        "scenario": scenario,
        "request_records": request_records or [],
        "resource_samples": resource_samples or [],
        "metrics_snapshot": metrics_snapshot or {},
        "stream_validation": stream_validation,
        "baseline_comparison": baseline_comparison,
        "server_log_path": server_log_path,
    }
    if extra:
        payload.update(extra)
    return payload


def _delta_pct(current: Any, baseline: Any) -> float | None:
    try:
        current_value = float(current)
        baseline_value = float(baseline)
    except (TypeError, ValueError):
        return None

    if baseline_value == 0:
        return None

    return round(((current_value - baseline_value) / baseline_value) * 100, 2)


def compare_against_baseline(
    *,
    summary: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any] | None:
    if not baseline_payload:
        return None

    baseline_summary = baseline_payload.get("summary") or {}
    thresholds = (baseline_payload.get("thresholds") or {}).get(mode) or {}

    metrics: dict[str, Any] = {}
    passed = True
    failed_metrics: list[str] = []

    for metric_name, rules in thresholds.items():
        current_value = summary.get(metric_name)
        baseline_value = baseline_summary.get(metric_name)
        checks: list[dict[str, Any]] = []

        min_absolute = rules.get("min_absolute")
        if min_absolute is not None and current_value is not None:
            check_passed = float(current_value) >= float(min_absolute)
            checks.append(
                {
                    "type": "min_absolute",
                    "threshold": float(min_absolute),
                    "passed": check_passed,
                }
            )

        max_absolute = rules.get("max_absolute")
        if max_absolute is not None and current_value is not None:
            check_passed = float(current_value) <= float(max_absolute)
            checks.append(
                {
                    "type": "max_absolute",
                    "threshold": float(max_absolute),
                    "passed": check_passed,
                }
            )

        min_ratio = rules.get("min_ratio")
        if min_ratio is not None and current_value is not None and baseline_value not in (None, 0):
            threshold_value = float(baseline_value) * float(min_ratio)
            check_passed = float(current_value) >= threshold_value
            checks.append(
                {
                    "type": "min_ratio",
                    "threshold": round(threshold_value, 4),
                    "ratio": float(min_ratio),
                    "passed": check_passed,
                }
            )

        max_ratio = rules.get("max_ratio")
        if max_ratio is not None and current_value is not None and baseline_value not in (None, 0):
            threshold_value = float(baseline_value) * float(max_ratio)
            check_passed = float(current_value) <= threshold_value
            checks.append(
                {
                    "type": "max_ratio",
                    "threshold": round(threshold_value, 4),
                    "ratio": float(max_ratio),
                    "passed": check_passed,
                }
            )

        metric_passed = all(check["passed"] for check in checks) if checks else True
        if not metric_passed:
            passed = False
            failed_metrics.append(metric_name)

        metrics[metric_name] = {
            "value": current_value,
            "baseline": baseline_value,
            "delta_pct": _delta_pct(current_value, baseline_value),
            "checks": checks,
            "passed": metric_passed,
        }

    return {
        "mode": mode,
        "baseline_ready": bool(baseline_payload.get("baseline_ready", True)),
        "baseline_profile": baseline_payload.get("profile"),
        "passed": passed,
        "failed_metrics": failed_metrics,
        "metrics": metrics,
    }


def update_baseline(path: str | Path, payload: dict[str, Any]) -> str:
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = backend_root() / destination

    existing = load_json(destination) if destination.exists() else {}
    existing["profile"] = payload.get("profile")
    existing["baseline_ready"] = True
    existing["summary"] = payload.get("summary") or {}
    existing["last_updated_at"] = utc_now()
    existing.setdefault("thresholds", {})
    return write_json(destination, existing)

