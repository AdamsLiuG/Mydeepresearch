"""Batch runner for benchmark cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

from config import Configuration
from evals.judges.base import Judge
from evals.judges.heuristic import HeuristicJudge
from evals.schema import BenchmarkCase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_agent_factory():
    from agent import DeepResearchAgent

    return DeepResearchAgent


def _trace_snapshot(agent: Any) -> dict[str, Any]:
    trace = getattr(agent, "_request_trace", None)
    if trace is None:
        return {}
    if hasattr(trace, "snapshot"):
        return trace.snapshot()
    return {}


def _serialize_todo_items(todo_items: Sequence[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in todo_items:
        payload.append(
            {
                "id": getattr(item, "id", None),
                "title": getattr(item, "title", None),
                "intent": getattr(item, "intent", None),
                "query": getattr(item, "query", None),
                "status": getattr(item, "status", None),
                "note_id": getattr(item, "note_id", None),
                "note_path": getattr(item, "note_path", None),
            }
        )
    return payload


def evaluate_case(
    case: BenchmarkCase,
    *,
    config: Configuration,
    agent_factory: Callable[..., Any] | None = None,
    judge: Judge | None = None,
    request_id_prefix: str = "eval",
) -> dict[str, Any]:
    """Run a single benchmark case and return structured metrics."""
    judge_impl = judge or HeuristicJudge()

    report_markdown = ""
    todo_items: Sequence[Any] = []
    trace_snapshot: dict[str, Any] = {}
    error: str | None = None
    agent = None
    started_at = perf_counter()

    try:
        factory = agent_factory or _default_agent_factory()
        agent = factory(config=config, request_id=f"{request_id_prefix}-{case.id}")
        result = agent.run(case.topic)
        report_markdown = (
            getattr(result, "report_markdown", None)
            or getattr(result, "running_summary", None)
            or ""
        )
        todo_items = getattr(result, "todo_items", []) or []
    except Exception as exc:  # pragma: no cover - exercised through batch runner tests
        error = str(exc)
    finally:
        if agent is not None:
            trace_snapshot = _trace_snapshot(agent)
        if not trace_snapshot.get("elapsed_ms"):
            trace_snapshot["elapsed_ms"] = round((perf_counter() - started_at) * 1000, 2)

    metrics = judge_impl.evaluate(
        case=case,
        report_markdown=report_markdown,
        todo_items=todo_items,
        trace_snapshot=trace_snapshot,
    )

    return {
        "id": case.id,
        "topic": case.topic,
        "freshness_sensitive": case.freshness_sensitive,
        "metrics": metrics,
        "error": error,
        "report_markdown": report_markdown,
        "todo_items": _serialize_todo_items(todo_items),
        "trace": trace_snapshot,
        "metadata": case.metadata,
    }


def build_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case results into a compact benchmark summary."""
    total = len(results)
    if total == 0:
        return {
            "total_cases": 0,
            "reports_generated": 0,
            "degraded_cases": 0,
            "average_section_completeness": 0.0,
            "average_keyword_coverage": 0.0,
            "average_citation_count": 0.0,
            "average_latency_ms": 0.0,
            "total_estimated_cost": 0.0,
            "error_cases": 0,
        }

    def average(metric_name: str) -> float:
        values = [
            float(result["metrics"].get(metric_name) or 0.0)
            for result in results
        ]
        return round(sum(values) / len(values), 4)

    report_generated = sum(1 for result in results if result["metrics"].get("report_generated"))
    degraded_cases = sum(1 for result in results if result["metrics"].get("degraded_flag"))
    error_cases = sum(1 for result in results if result.get("error"))
    total_estimated_cost = round(
        sum(float(result["metrics"].get("estimated_cost") or 0.0) for result in results),
        6,
    )

    return {
        "total_cases": total,
        "reports_generated": report_generated,
        "degraded_cases": degraded_cases,
        "error_cases": error_cases,
        "report_generation_rate": round(report_generated / total, 4),
        "degraded_rate": round(degraded_cases / total, 4),
        "average_section_completeness": average("section_completeness"),
        "average_keyword_coverage": average("keyword_coverage"),
        "average_citation_count": average("citation_count"),
        "average_latency_ms": average("total_latency_ms"),
        "total_estimated_cost": total_estimated_cost,
    }


def run_benchmark_suite(
    cases: Sequence[BenchmarkCase],
    *,
    config: Configuration,
    agent_factory: Callable[..., Any] | None = None,
    judge: Judge | None = None,
    output_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
    request_id_prefix: str = "eval",
) -> dict[str, Any]:
    """Run a full benchmark suite and optionally write the result file."""
    results = [
        evaluate_case(
            case,
            config=config,
            agent_factory=agent_factory,
            judge=judge,
            request_id_prefix=request_id_prefix,
        )
        for case in cases
    ]

    payload = {
        "generated_at": _utc_now(),
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "summary": build_summary(results),
        "results": results,
    }

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload["output_path"] = str(destination)

    return payload
