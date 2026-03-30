"""CLI entrypoint for HTTP-level full-system validation suites."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import requests

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from evals.judges.heuristic import HeuristicJudge  # noqa: E402
from evals.loader import load_benchmark_cases  # noqa: E402
from evals.schema import BenchmarkCase  # noqa: E402
from perf.common import load_json  # noqa: E402

DEFAULT_BENCHMARK_PATH = BACKEND_ROOT / "evals" / "benchmarks" / "full_system_12cases.jsonl"
DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "evals" / "results" / "full_system_http_results.json"
DEFAULT_SUMMARY_PATH = BACKEND_ROOT / "evals" / "results" / "full_system_interview_summary.md"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUIRED_STREAM_EVENTS = ("status", "todo_list", "metrics_snapshot", "final_report", "done")
KNOWN_ENGINEERING_NOTE = (
    "本轮总结聚焦系统功能与性能。若当前工作区存在独立的 lint/ruff 失败，应单独归档为工程质量问题，"
    "不要与真实链路功能结果混为一谈。"
)
FRONTEND_MANUAL_CHECKS = (
    "首页提交后切到双栏布局",
    "任务规划卡出现",
    "流程日志持续追加",
    "metrics strip 有值",
    "最终 Markdown 报告渲染",
    "引用可见",
    "无错误横幅",
)
FRONTEND_MANUAL_CASE_IDS = (
    "agent_engineering_practice",
    "inference_service_ops",
    "fresh_multimodal_2025",
    "enterprise_tradeoffs_2025",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the HTTP suite runner."""
    parser = argparse.ArgumentParser(description="Run HTTP-level full-system research suites.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to a benchmark .json or .jsonl file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the JSON result file.",
    )
    parser.add_argument(
        "--summary-md",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to the interview-style Markdown summary.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the running backend server.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=("sync", "stream", "both"),
        help="Which HTTP suite to run.",
    )
    parser.add_argument(
        "--search-api",
        default=None,
        help="Optional search_api override sent with every request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional case limit for quick validation.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=300.0,
        help="HTTP timeout for a single sync or stream request.",
    )
    parser.add_argument(
        "--trace-timeout-seconds",
        type=float,
        default=15.0,
        help="How long to wait for /metrics/json recent_requests to contain the request trace.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.5,
        help="Polling interval used while waiting for metrics traces.",
    )
    parser.add_argument(
        "--request-id-prefix",
        default="full-system",
        help="Stable prefix used for X-Request-ID headers.",
    )
    parser.add_argument(
        "--perf-profile",
        default="real_local",
        help="Perf profile used when auto-discovering perf result files.",
    )
    parser.add_argument(
        "--perf-result",
        action="append",
        default=[],
        help="Optional perf result JSON file. Can be passed multiple times.",
    )
    return parser.parse_args()


def _resolve_output_path(path: str | Path) -> Path:
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = BACKEND_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.resolve()


def _request_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _request_json(
    session: requests.Session,
    *,
    method: str,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[requests.Response, dict[str, Any] | None, str]:
    response = session.request(
        method=method,
        url=url,
        json=payload,
        headers=headers,
        timeout=timeout_seconds,
    )
    body_text = response.text
    try:
        parsed = response.json()
    except ValueError:
        parsed = None
    return response, parsed if isinstance(parsed, dict) else None, body_text


def _health_check(base_url: str, timeout_seconds: float) -> None:
    with _request_session() as session:
        response, payload, body_text = _request_json(
            session,
            method="GET",
            url=f"{base_url}/healthz",
            timeout_seconds=timeout_seconds,
        )
    if not response.ok:
        raise RuntimeError(f"/healthz returned HTTP {response.status_code}: {body_text}")
    if payload != {"status": "ok"}:
        raise RuntimeError(f"/healthz returned unexpected payload: {payload!r}")


def _safe_float(value: Any, *, digits: int = 2) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_namespace_items(todo_items: Sequence[Any] | None) -> list[Any]:
    payload: list[Any] = []
    for item in todo_items or []:
        if isinstance(item, dict):
            payload.append(SimpleNamespace(**item))
        else:
            payload.append(item)
    return payload


def _fallback_trace(
    *,
    request_id: str,
    case: BenchmarkCase,
    search_api: str | None,
    status: str,
    elapsed_ms: float | None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "topic": case.topic,
        "search_api": search_api,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "error": error,
        "fallback_triggered": False,
        "fallback_reasons": [],
        "degraded": False,
        "degraded_reasons": [],
        "cache_hits": 0,
        "cache_exact_hits": 0,
        "cache_semantic_hits": 0,
        "cache_misses": 0,
        "estimated_cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "todo_items": [],
        "stages": [],
    }


def _wait_for_request_trace(
    session: requests.Session,
    *,
    base_url: str,
    request_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    deadline = time.perf_counter() + max(timeout_seconds, 0.1)
    last_metrics: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        try:
            response, payload, _ = _request_json(
                session,
                method="GET",
                url=f"{base_url}/metrics/json",
                timeout_seconds=max(poll_interval_seconds * 2, 1.0),
            )
        except requests.RequestException:
            time.sleep(max(poll_interval_seconds, 0.05))
            continue
        if response.ok and payload:
            last_metrics = payload
            for item in payload.get("recent_requests") or []:
                if isinstance(item, dict) and item.get("request_id") == request_id:
                    return item, payload, True
        time.sleep(max(poll_interval_seconds, 0.05))
    return {}, last_metrics, False


def _judge_report(
    *,
    case: BenchmarkCase,
    report_markdown: str,
    todo_items: Sequence[Any] | None,
    trace_snapshot: dict[str, Any],
) -> dict[str, Any]:
    return HeuristicJudge().evaluate(
        case=case,
        report_markdown=report_markdown,
        todo_items=_to_namespace_items(todo_items),
        trace_snapshot=trace_snapshot,
    )


def _send_sync_request(
    session: requests.Session,
    *,
    base_url: str,
    case: BenchmarkCase,
    search_api: str | None,
    request_id: str,
    timeout_seconds: float,
    trace_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    payload = {"topic": case.topic}
    if search_api:
        payload["search_api"] = search_api

    started_at = time.perf_counter()
    response: requests.Response | None = None
    body: dict[str, Any] | None = None
    raw_body = ""
    error: str | None = None

    try:
        response, body, raw_body = _request_json(
            session,
            method="POST",
            url=f"{base_url}/research",
            timeout_seconds=timeout_seconds,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            payload=payload,
        )
    except requests.RequestException as exc:
        error = str(exc)

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    header_request_id = response.headers.get("X-Request-ID") if response is not None else None
    matched_trace: dict[str, Any] = {}
    metrics_snapshot: dict[str, Any] = {}
    trace_found = False
    if error is None:
        matched_trace, metrics_snapshot, trace_found = _wait_for_request_trace(
            session,
            base_url=base_url,
            request_id=request_id,
            timeout_seconds=trace_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    http_ok = bool(response is not None and response.ok and isinstance(body, dict))
    report_markdown = _safe_text(body.get("report_markdown")) if body else ""
    todo_items = list(body.get("todo_items") or []) if body else []
    trace_snapshot = matched_trace or _fallback_trace(
        request_id=request_id,
        case=case,
        search_api=search_api,
        status="success" if http_ok else "failed",
        elapsed_ms=latency_ms,
        error=error or (raw_body[:240] if raw_body else None),
    )
    judge_metrics = _judge_report(
        case=case,
        report_markdown=report_markdown,
        todo_items=todo_items,
        trace_snapshot=trace_snapshot,
    )

    return {
        "mode": "sync",
        "id": case.id,
        "topic": case.topic,
        "level": case.metadata.get("level"),
        "category": case.metadata.get("category"),
        "freshness_sensitive": case.freshness_sensitive,
        "expected_sections": list(case.expected_sections),
        "expected_keywords": list(case.expected_keywords),
        "request_id": request_id,
        "header_request_id": header_request_id,
        "request_id_matches_header": not header_request_id or header_request_id == request_id,
        "status_code": response.status_code if response is not None else None,
        "http_ok": http_ok,
        "client_latency_ms": latency_ms,
        "trace_lookup_found": trace_found,
        "trace": trace_snapshot,
        "metrics_snapshot": metrics_snapshot,
        "report_markdown": report_markdown,
        "report_length": len(report_markdown),
        "todo_items": todo_items,
        "todo_item_count": len(todo_items),
        "judge_metrics": judge_metrics,
        "case_passed": bool(http_ok and judge_metrics.get("report_generated")),
        "error": error,
        "response_text_excerpt": raw_body[:400] if raw_body else "",
    }


def _flush_sse_event(raw_block: str) -> dict[str, Any] | None:
    if not raw_block:
        return None

    lines = raw_block.splitlines()
    data_lines = [line[5:].lstrip() for line in lines if line.startswith("data:")]
    if not data_lines:
        return None

    raw_payload = "\n".join(data_lines).strip()
    if not raw_payload:
        return None

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return {"type": "invalid_json", "raw_payload": raw_payload}
    return payload if isinstance(payload, dict) else {"type": "invalid_payload", "raw_payload": raw_payload}


def _iter_sse_events(response: requests.Response) -> Iterable[dict[str, Any]]:
    buffer = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
        if not chunk:
            continue
        text = chunk.decode("utf-8", errors="replace")
        buffer += text

        boundary = buffer.find("\n\n")
        while boundary != -1:
            raw_block = buffer[:boundary].strip()
            buffer = buffer[boundary + 2 :]
            payload = _flush_sse_event(raw_block)
            if payload is not None:
                yield payload
            boundary = buffer.find("\n\n")

    payload = _flush_sse_event(buffer.strip())
    if payload is not None:
        yield payload


def _stream_request(
    session: requests.Session,
    *,
    base_url: str,
    case: BenchmarkCase,
    search_api: str | None,
    request_id: str,
    timeout_seconds: float,
    trace_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    payload = {"topic": case.topic}
    if search_api:
        payload["search_api"] = search_api

    started_at = time.perf_counter()
    response: requests.Response | None = None
    raw_error_text = ""
    request_error: str | None = None
    event_counts: dict[str, int] = {}
    first_seen_ms: dict[str, float] = {}
    event_sequence: list[str] = []
    error_events: list[dict[str, Any]] = []
    last_todo_list: list[dict[str, Any]] = []
    final_report = ""
    final_report_index: int | None = None
    done_index: int | None = None
    metrics_event_payload: dict[str, Any] | None = None

    try:
        response = session.post(
            f"{base_url}/research/stream",
            json=payload,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            timeout=timeout_seconds,
            stream=True,
        )

        if response.ok:
            for event_index, event in enumerate(_iter_sse_events(response), start=1):
                now_ms = round((time.perf_counter() - started_at) * 1000, 2)
                event_type = _safe_text(event.get("type")) or "unknown"
                event_sequence.append(event_type)
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                first_seen_ms.setdefault(event_type, now_ms)

                if event_type == "error":
                    error_events.append(event)
                elif event_type == "todo_list":
                    last_todo_list = list(event.get("tasks") or [])
                elif event_type == "final_report":
                    final_report = _safe_text(event.get("report"))
                    final_report_index = event_index
                elif event_type == "done":
                    done_index = event_index
                elif event_type == "metrics_snapshot":
                    metrics_event_payload = event
        else:
            raw_error_text = response.text[:400]
    except requests.RequestException as exc:
        request_error = str(exc)
    finally:
        if response is not None:
            response.close()

    total_stream_duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    header_request_id = response.headers.get("X-Request-ID") if response is not None else None
    matched_trace, metrics_snapshot, trace_found = _wait_for_request_trace(
        session,
        base_url=base_url,
        request_id=request_id,
        timeout_seconds=trace_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    trace_snapshot = matched_trace or _fallback_trace(
        request_id=request_id,
        case=case,
        search_api=search_api,
        status="success" if response is not None and response.ok and not request_error else "failed",
        elapsed_ms=total_stream_duration_ms,
        error=request_error or raw_error_text or None,
    )
    todo_items = (
        list((metrics_event_payload or {}).get("request_metrics", {}).get("todo_items") or [])
        or list(trace_snapshot.get("todo_items") or [])
        or last_todo_list
    )

    missing_required_events = [event for event in REQUIRED_STREAM_EVENTS if event_counts.get(event, 0) <= 0]
    final_report_before_done = (
        final_report_index is not None and done_index is not None and final_report_index < done_index
    )
    no_error_event = event_counts.get("error", 0) == 0
    http_ok = bool(response is not None and response.ok and request_error is None)
    case_passed = bool(
        http_ok
        and no_error_event
        and not missing_required_events
        and final_report_before_done
    )

    return {
        "mode": "stream",
        "id": case.id,
        "topic": case.topic,
        "level": case.metadata.get("level"),
        "category": case.metadata.get("category"),
        "freshness_sensitive": case.freshness_sensitive,
        "expected_sections": list(case.expected_sections),
        "expected_keywords": list(case.expected_keywords),
        "request_id": request_id,
        "header_request_id": header_request_id,
        "request_id_matches_header": not header_request_id or header_request_id == request_id,
        "status_code": response.status_code if response is not None else None,
        "http_ok": http_ok,
        "trace_lookup_found": trace_found,
        "trace": trace_snapshot,
        "metrics_snapshot": metrics_snapshot,
        "metrics_event_request_metrics": (metrics_event_payload or {}).get("request_metrics") or {},
        "report_markdown": final_report,
        "report_length": len(final_report),
        "todo_items": todo_items,
        "todo_item_count": len(todo_items),
        "event_counts": event_counts,
        "event_sequence": event_sequence,
        "total_event_count": sum(event_counts.values()),
        "missing_required_events": missing_required_events,
        "no_error_event": no_error_event,
        "final_report_before_done": final_report_before_done,
        "time_to_first_event_ms": first_seen_ms.get(event_sequence[0]) if event_sequence else None,
        "time_to_todo_list_ms": first_seen_ms.get("todo_list"),
        "time_to_first_tool_call_ms": first_seen_ms.get("tool_call"),
        "time_to_final_report_ms": first_seen_ms.get("final_report"),
        "total_stream_duration_ms": total_stream_duration_ms,
        "case_passed": case_passed,
        "error": request_error,
        "error_events": error_events,
        "response_text_excerpt": raw_error_text,
    }


def _average(values: Sequence[float | int | None], *, digits: int = 4) -> float:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return 0.0
    return round(sum(usable) / len(usable), digits)


def _status_counts(results: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = _safe_text(result.get("trace", {}).get("status")) or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_sync_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(results)
    metrics_list = [result.get("judge_metrics") or {} for result in results]
    reference_match_values = [
        float(metrics.get("reference_match_rate") or 0.0)
        for metrics in metrics_list
        if int(metrics.get("citation_marker_count") or 0) > 0
    ]

    http_ok_count = sum(1 for result in results if result.get("http_ok"))
    report_generated_count = sum(1 for metrics in metrics_list if metrics.get("report_generated"))
    reference_section_present_count = sum(1 for metrics in metrics_list if metrics.get("reference_section_present"))
    degraded_case_count = sum(1 for result in results if result.get("trace", {}).get("degraded"))
    fallback_case_count = sum(1 for result in results if result.get("trace", {}).get("fallback_triggered"))
    error_case_count = sum(1 for result in results if result.get("error"))

    average_section = _average(
        [metrics.get("section_completeness") for metrics in metrics_list],
    )
    average_keyword = _average(
        [metrics.get("keyword_coverage") for metrics in metrics_list],
    )
    average_citation = _average(
        [metrics.get("citation_count") for metrics in metrics_list],
    )
    average_reference_match = _average(reference_match_values)
    average_grounded_ratio = _average(
        [metrics.get("grounded_bullet_ratio") for metrics in metrics_list],
    )
    average_latency_ms = _average(
        [result.get("trace", {}).get("elapsed_ms") or result.get("client_latency_ms") for result in results],
        digits=2,
    )
    total_estimated_cost = round(
        sum(float(result.get("trace", {}).get("estimated_cost") or 0.0) for result in results),
        6,
    )

    acceptance = {
        "http_200_and_report_generated": http_ok_count == total_cases and report_generated_count == total_cases,
        "average_section_completeness_gte_0_75": average_section >= 0.75,
        "average_keyword_coverage_gte_0_60": average_keyword >= 0.60,
        "average_citation_count_gte_3": average_citation >= 3.0,
        "reference_section_present_at_least_10": reference_section_present_count >= min(total_cases, 10),
        "average_reference_match_rate_gte_0_80_when_referenced": (
            average_reference_match >= 0.80 if reference_match_values else True
        ),
    }

    return {
        "mode": "sync",
        "total_cases": total_cases,
        "http_ok_count": http_ok_count,
        "report_generated_count": report_generated_count,
        "reference_section_present_count": reference_section_present_count,
        "degraded_case_count": degraded_case_count,
        "fallback_case_count": fallback_case_count,
        "error_case_count": error_case_count,
        "status_counts": _status_counts(results),
        "average_section_completeness": average_section,
        "average_keyword_coverage": average_keyword,
        "average_citation_count": average_citation,
        "average_reference_match_rate": average_reference_match,
        "reference_match_case_count": len(reference_match_values),
        "average_grounded_bullet_ratio": average_grounded_ratio,
        "average_latency_ms": average_latency_ms,
        "total_estimated_cost": total_estimated_cost,
        "acceptance": {
            **acceptance,
            "passed": all(acceptance.values()),
        },
    }


def build_stream_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(results)
    http_ok_count = sum(1 for result in results if result.get("http_ok"))
    no_error_event_count = sum(1 for result in results if result.get("no_error_event"))
    required_event_bundle_count = sum(1 for result in results if not result.get("missing_required_events"))
    final_report_before_done_count = sum(1 for result in results if result.get("final_report_before_done"))
    degraded_case_count = sum(1 for result in results if result.get("trace", {}).get("degraded"))
    error_case_count = sum(
        1
        for result in results
        if result.get("error") or result.get("event_counts", {}).get("error", 0) > 0
    )

    acceptance = {
        "http_200_all_cases": http_ok_count == total_cases,
        "no_error_event_all_cases": no_error_event_count == total_cases,
        "required_event_bundle_all_cases": required_event_bundle_count == total_cases,
        "final_report_before_done_all_cases": final_report_before_done_count == total_cases,
    }

    return {
        "mode": "stream",
        "total_cases": total_cases,
        "http_ok_count": http_ok_count,
        "no_error_event_count": no_error_event_count,
        "required_event_bundle_count": required_event_bundle_count,
        "final_report_before_done_count": final_report_before_done_count,
        "degraded_case_count": degraded_case_count,
        "error_case_count": error_case_count,
        "status_counts": _status_counts(results),
        "average_time_to_first_event_ms": _average(
            [result.get("time_to_first_event_ms") for result in results],
            digits=2,
        ),
        "average_time_to_todo_list_ms": _average(
            [result.get("time_to_todo_list_ms") for result in results],
            digits=2,
        ),
        "average_time_to_first_tool_call_ms": _average(
            [result.get("time_to_first_tool_call_ms") for result in results if result.get("time_to_first_tool_call_ms") is not None],
            digits=2,
        ),
        "average_time_to_final_report_ms": _average(
            [result.get("time_to_final_report_ms") for result in results],
            digits=2,
        ),
        "average_total_stream_duration_ms": _average(
            [result.get("total_stream_duration_ms") for result in results],
            digits=2,
        ),
        "average_total_event_count": _average(
            [result.get("total_event_count") for result in results],
            digits=2,
        ),
        "total_estimated_cost": round(
            sum(float(result.get("trace", {}).get("estimated_cost") or 0.0) for result in results),
            6,
        ),
        "acceptance": {
            **acceptance,
            "passed": all(acceptance.values()),
        },
    }


def _discover_perf_results(profile: str, explicit_paths: Sequence[str]) -> list[dict[str, Any]]:
    result_paths = [Path(path).expanduser() for path in explicit_paths]
    if not result_paths:
        result_paths = [
            BACKEND_ROOT / "perf" / "results" / f"smoke-{profile}.json",
            BACKEND_ROOT / "perf" / "results" / f"regression-{profile}.json",
            BACKEND_ROOT / "perf" / "results" / f"load-{profile}.json",
        ]

    perf_payloads: list[dict[str, Any]] = []
    for path in result_paths:
        if not path.is_absolute():
            path = (BACKEND_ROOT / path).resolve()
        if not path.exists():
            continue
        payload = load_json(path)
        summary = payload.get("summary") or {}
        perf_payloads.append(
            {
                "path": str(path.resolve()),
                "mode": payload.get("mode") or summary.get("mode"),
                "profile": payload.get("profile") or summary.get("profile"),
                "summary": summary,
                "baseline_comparison": payload.get("baseline_comparison") or {},
            }
        )
    return perf_payloads


def _case_matrix(
    cases: Sequence[BenchmarkCase],
    *,
    sync_results: Sequence[dict[str, Any]] | None,
    stream_results: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    sync_by_id = {result["id"]: result for result in sync_results or []}
    stream_by_id = {result["id"]: result for result in stream_results or []}
    rows: list[dict[str, Any]] = []

    for case in cases:
        sync_result = sync_by_id.get(case.id, {})
        stream_result = stream_by_id.get(case.id, {})
        trace = sync_result.get("trace") or stream_result.get("trace") or {}

        rows.append(
            {
                "id": case.id,
                "topic": case.topic,
                "level": case.metadata.get("level"),
                "category": case.metadata.get("category"),
                "sync_case_passed": bool(sync_result.get("case_passed")) if sync_result else None,
                "stream_case_passed": bool(stream_result.get("case_passed")) if stream_result else None,
                "request_status": trace.get("status"),
                "degraded": bool(trace.get("degraded")),
                "fallback_triggered": bool(trace.get("fallback_triggered")),
                "elapsed_ms": _safe_float(trace.get("elapsed_ms") or sync_result.get("client_latency_ms") or stream_result.get("total_stream_duration_ms")),
                "estimated_cost": round(float(trace.get("estimated_cost") or 0.0), 6),
                "degraded_reasons": list(trace.get("degraded_reasons") or []),
                "stream_missing_events": list(stream_result.get("missing_required_events") or []),
            }
        )
    return rows


def _frontend_manual_cases(cases: Sequence[BenchmarkCase]) -> list[dict[str, Any]]:
    case_map = {case.id: case for case in cases}
    payload: list[dict[str, Any]] = []
    for case_id in FRONTEND_MANUAL_CASE_IDS:
        case = case_map.get(case_id)
        if case is None:
            continue
        payload.append(
            {
                "id": case.id,
                "topic": case.topic,
                "level": case.metadata.get("level"),
                "checks": list(FRONTEND_MANUAL_CHECKS),
                "status": "pending_manual_validation",
            }
        )
    return payload


def build_payload(
    *,
    cases: Sequence[BenchmarkCase],
    benchmark_path: str | Path,
    base_url: str,
    mode: str,
    search_api: str | None,
    sync_results: Sequence[dict[str, Any]] | None,
    stream_results: Sequence[dict[str, Any]] | None,
    perf_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    sync_summary = build_sync_summary(sync_results or []) if sync_results is not None else None
    stream_summary = build_stream_summary(stream_results or []) if stream_results is not None else None
    case_matrix = _case_matrix(
        cases,
        sync_results=sync_results,
        stream_results=stream_results,
    )
    degraded_case_ids = sorted({row["id"] for row in case_matrix if row.get("degraded")})

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_path": str(Path(benchmark_path).expanduser().resolve()),
        "base_url": base_url,
        "mode": mode,
        "search_api": search_api,
        "known_engineering_note": KNOWN_ENGINEERING_NOTE,
        "frontend_manual_cases": _frontend_manual_cases(cases),
        "sync_suite": {
            "summary": sync_summary,
            "results": list(sync_results or []),
        }
        if sync_results is not None
        else None,
        "stream_suite": {
            "summary": stream_summary,
            "results": list(stream_results or []),
        }
        if stream_results is not None
        else None,
        "case_matrix": case_matrix,
        "perf_results": list(perf_payloads),
        "overall": {
            "sync_passed": bool(sync_summary and sync_summary.get("acceptance", {}).get("passed")),
            "stream_passed": bool(stream_summary and stream_summary.get("acceptance", {}).get("passed")),
            "degraded_case_ids": degraded_case_ids,
            "total_estimated_cost": round(
                sum(float(row.get("estimated_cost") or 0.0) for row in case_matrix),
                6,
            ),
        },
    }


def _acceptance_text(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "通过" if value else "未通过"


def _case_status_text(value: bool | None) -> str:
    if value is None:
        return "未运行"
    return "通过" if value else "失败"


def _failure_bullets(payload: dict[str, Any]) -> list[str]:
    case_topics = {row["id"]: row["topic"] for row in payload.get("case_matrix") or []}
    bullets: list[str] = []

    for result in (payload.get("sync_suite") or {}).get("results") or []:
        if result.get("case_passed") and not result.get("trace", {}).get("degraded"):
            continue
        details: list[str] = []
        if not result.get("http_ok"):
            details.append(f"sync HTTP={result.get('status_code')}")
        if result.get("error"):
            details.append(f"sync error={result['error']}")
        if result.get("trace", {}).get("degraded"):
            details.append(f"降级={','.join(result['trace'].get('degraded_reasons') or [])}")
        if details:
            bullets.append(f"{result['id']} | {case_topics.get(result['id'], result['id'])} | " + "；".join(details))

    for result in (payload.get("stream_suite") or {}).get("results") or []:
        if result.get("case_passed") and not result.get("trace", {}).get("degraded"):
            continue
        details = []
        if not result.get("http_ok"):
            details.append(f"stream HTTP={result.get('status_code')}")
        if result.get("missing_required_events"):
            details.append(f"缺少事件={','.join(result['missing_required_events'])}")
        if not result.get("final_report_before_done"):
            details.append("final_report 与 done 顺序异常")
        if result.get("error"):
            details.append(f"stream error={result['error']}")
        if result.get("trace", {}).get("degraded"):
            details.append(f"降级={','.join(result['trace'].get('degraded_reasons') or [])}")
        if details:
            bullets.append(f"{result['id']} | {case_topics.get(result['id'], result['id'])} | " + "；".join(details))

    deduped: list[str] = []
    seen: set[str] = set()
    for bullet in bullets:
        if bullet in seen:
            continue
        seen.add(bullet)
        deduped.append(bullet)
    return deduped


def render_interview_summary(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 全系统真实链路测试总结")
    lines.append("")
    lines.append("## 测试范围")
    lines.append(f"- 生成时间：{payload.get('generated_at')}")
    lines.append(f"- 后端地址：{payload.get('base_url')}")
    lines.append(f"- Benchmark 文件：`{payload.get('benchmark_path')}`")
    lines.append(f"- 运行模式：`{payload.get('mode')}`")
    lines.append(f"- 搜索覆盖：`{payload.get('search_api') or '沿用后端配置'}`")
    lines.append(f"- 说明：{payload.get('known_engineering_note')}")
    lines.append("")

    sync_suite = payload.get("sync_suite") or {}
    sync_summary = sync_suite.get("summary") or {}
    if sync_summary:
        lines.append("## 同步功能结果")
        lines.append(f"- 整体结论：{_acceptance_text(sync_summary.get('acceptance', {}).get('passed'))}")
        lines.append(
            "- 通过情况："
            f"{sync_summary.get('http_ok_count', 0)}/{sync_summary.get('total_cases', 0)} HTTP 200，"
            f"{sync_summary.get('report_generated_count', 0)}/{sync_summary.get('total_cases', 0)} 生成报告"
        )
        lines.append(
            "- 平均指标："
            f"章节完整度 {sync_summary.get('average_section_completeness', 0.0)}，"
            f"关键词覆盖 {sync_summary.get('average_keyword_coverage', 0.0)}，"
            f"引用数 {sync_summary.get('average_citation_count', 0.0)}，"
            f"引用匹配率 {sync_summary.get('average_reference_match_rate', 0.0)}"
        )
        lines.append(
            "- 请求表现："
            f"平均耗时 {sync_summary.get('average_latency_ms', 0.0)} ms，"
            f"降级 {sync_summary.get('degraded_case_count', 0)} 例，"
            f"fallback {sync_summary.get('fallback_case_count', 0)} 例，"
            f"估算总成本 {sync_summary.get('total_estimated_cost', 0.0)}"
        )
        lines.append("")

    stream_suite = payload.get("stream_suite") or {}
    stream_summary = stream_suite.get("summary") or {}
    if stream_summary:
        lines.append("## 流式结果")
        lines.append(f"- 整体结论：{_acceptance_text(stream_summary.get('acceptance', {}).get('passed'))}")
        lines.append(
            "- 事件完整性："
            f"{stream_summary.get('required_event_bundle_count', 0)}/{stream_summary.get('total_cases', 0)} 包含完整事件集，"
            f"{stream_summary.get('no_error_event_count', 0)}/{stream_summary.get('total_cases', 0)} 无 error 事件，"
            f"{stream_summary.get('final_report_before_done_count', 0)}/{stream_summary.get('total_cases', 0)} 顺序正确"
        )
        lines.append(
            "- 时序指标："
            f"首事件 {stream_summary.get('average_time_to_first_event_ms', 0.0)} ms，"
            f"todo_list {stream_summary.get('average_time_to_todo_list_ms', 0.0)} ms，"
            f"final_report {stream_summary.get('average_time_to_final_report_ms', 0.0)} ms，"
            f"总时长 {stream_summary.get('average_total_stream_duration_ms', 0.0)} ms"
        )
        lines.append(
            "- 请求表现："
            f"平均事件数 {stream_summary.get('average_total_event_count', 0.0)}，"
            f"降级 {stream_summary.get('degraded_case_count', 0)} 例，"
            f"估算总成本 {stream_summary.get('total_estimated_cost', 0.0)}"
        )
        lines.append("")

    lines.append("## 用例总表")
    lines.append("| ID | 层次 | Sync | Stream | 请求状态 | 降级 | 耗时ms | 成本 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload.get("case_matrix") or []:
        lines.append(
            f"| {row['id']} | {row.get('level') or '-'} | {_case_status_text(row.get('sync_case_passed'))} | "
            f"{_case_status_text(row.get('stream_case_passed'))} | {row.get('request_status') or '-'} | "
            f"{'是' if row.get('degraded') else '否'} | {row.get('elapsed_ms') or 0} | {row.get('estimated_cost') or 0.0} |"
        )
    lines.append("")

    lines.append("## 失败与降级案例")
    bullets = _failure_bullets(payload)
    if bullets:
        for bullet in bullets:
            lines.append(f"- {bullet}")
    else:
        lines.append("- 本轮 HTTP 套件未发现失败或降级案例。")
    lines.append("")

    lines.append("## 性能结论")
    perf_results = payload.get("perf_results") or []
    if perf_results:
        lines.append("| 模式 | Profile | RPS | P95 ms | P99 ms | 成功率 | 错误率 | CPU 峰值 | RSS 峰值 | 成本/请求 | Baseline |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in perf_results:
            summary = item.get("summary") or {}
            baseline = item.get("baseline_comparison") or {}
            lines.append(
                f"| {item.get('mode') or '-'} | {item.get('profile') or '-'} | {summary.get('rps', 0.0)} | "
                f"{summary.get('p95_latency_ms', 0.0)} | {summary.get('p99_latency_ms', 0.0)} | "
                f"{summary.get('success_rate', 0.0)} | {summary.get('error_rate', 0.0)} | "
                f"{summary.get('cpu_percent_peak', 0.0)} | {summary.get('rss_mb_peak', 0.0)} | "
                f"{summary.get('estimated_cost_per_request', 0.0)} | {_acceptance_text(baseline.get('passed'))} |"
            )
    else:
        lines.append("- 当前未发现可复用的 perf 结果文件；需要额外运行 `perf.run_smoke`、`perf.run_regression` 和 `perf.run_load`。")
    lines.append("")

    lines.append("## 前端手工验收")
    for item in payload.get("frontend_manual_cases") or []:
        lines.append(
            f"- {item['id']} | {item.get('level') or '-'} | 状态：{item['status']} | 主题：{item['topic']}"
        )
    lines.append(f"- 共用检查项：{'；'.join(FRONTEND_MANUAL_CHECKS)}")
    lines.append("")

    lines.append("## 可讲亮点")
    lines.append("- 这轮验证不是只跑单元测试，而是同时覆盖同步研究接口、SSE 流式过程、内容质量、trace 指标和轻负载性能。")
    lines.append("- 每个请求都通过固定 `X-Request-ID` 关联到 `/metrics/json` 的 recent trace，可以把“最终报告质量”和“内部阶段行为”一起讲清楚。")
    lines.append("- 鲁棒性主题允许 `partial_success` 或 `degraded`，但不接受 500，有助于在面试里强调 Agent 系统的降级设计而不是只讲 happy path。")
    return "\n".join(lines).strip() + "\n"


def _write_payload(path: str | Path, payload: dict[str, Any]) -> str:
    destination = _resolve_output_path(path)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(destination)


def _write_text(path: str | Path, content: str) -> str:
    destination = _resolve_output_path(path)
    destination.write_text(content, encoding="utf-8")
    return str(destination)


def _print_summary(payload: dict[str, Any], *, output_path: str, summary_path: str) -> None:
    lines = [f"Benchmark cases: {len(payload.get('case_matrix') or [])}"]
    sync_summary = ((payload.get("sync_suite") or {}).get("summary") or {})
    stream_summary = ((payload.get("stream_suite") or {}).get("summary") or {})
    if sync_summary:
        lines.append(f"Sync acceptance: {sync_summary.get('acceptance', {}).get('passed')}")
        lines.append(f"Sync average section completeness: {sync_summary.get('average_section_completeness')}")
        lines.append(f"Sync average keyword coverage: {sync_summary.get('average_keyword_coverage')}")
    if stream_summary:
        lines.append(f"Stream acceptance: {stream_summary.get('acceptance', {}).get('passed')}")
        lines.append(
            f"Stream average final_report latency ms: {stream_summary.get('average_time_to_final_report_ms')}"
        )
    lines.append(f"Degraded case ids: {payload.get('overall', {}).get('degraded_case_ids')}")
    lines.append(f"Perf files discovered: {len(payload.get('perf_results') or [])}")
    lines.append(f"JSON results written to: {output_path}")
    lines.append(f"Markdown summary written to: {summary_path}")
    sys.stdout.write("\n".join(lines) + "\n")


def main() -> int:
    """Run sync and/or streaming HTTP suites and write JSON + Markdown outputs."""
    args = parse_args()
    cases = load_benchmark_cases(args.input)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    base_url = args.base_url.rstrip("/")
    _health_check(base_url, timeout_seconds=max(args.poll_interval_seconds * 2, 5.0))

    sync_results: list[dict[str, Any]] | None = None
    stream_results: list[dict[str, Any]] | None = None

    with _request_session() as session:
        if args.mode in {"sync", "both"}:
            sync_results = []
            for index, case in enumerate(cases, start=1):
                request_id = f"{args.request_id_prefix}-sync-{index:02d}-{case.id}"
                sync_results.append(
                    _send_sync_request(
                        session,
                        base_url=base_url,
                        case=case,
                        search_api=args.search_api,
                        request_id=request_id,
                        timeout_seconds=args.request_timeout_seconds,
                        trace_timeout_seconds=args.trace_timeout_seconds,
                        poll_interval_seconds=args.poll_interval_seconds,
                    )
                )

        if args.mode in {"stream", "both"}:
            stream_results = []
            for index, case in enumerate(cases, start=1):
                request_id = f"{args.request_id_prefix}-stream-{index:02d}-{case.id}"
                stream_results.append(
                    _stream_request(
                        session,
                        base_url=base_url,
                        case=case,
                        search_api=args.search_api,
                        request_id=request_id,
                        timeout_seconds=args.request_timeout_seconds,
                        trace_timeout_seconds=args.trace_timeout_seconds,
                        poll_interval_seconds=args.poll_interval_seconds,
                    )
                )

    perf_payloads = _discover_perf_results(args.perf_profile, args.perf_result)
    payload = build_payload(
        cases=cases,
        benchmark_path=args.input,
        base_url=base_url,
        mode=args.mode,
        search_api=args.search_api,
        sync_results=sync_results,
        stream_results=stream_results,
        perf_payloads=perf_payloads,
    )
    output_path = _write_payload(args.output, payload)
    summary_path = _write_text(args.summary_md, render_interview_summary(payload))
    _print_summary(payload, output_path=output_path, summary_path=summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
