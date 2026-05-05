"""Aggregate project-level metrics from request snapshots and eval result files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_STATE_DIR = BACKEND_ROOT / ".state" / "requests"
DEFAULT_EVAL_RESULTS_DIRS = (
    BACKEND_ROOT / "evals" / "results",
    BACKEND_ROOT / "backend" / "evals" / "results",
)
DEFAULT_OUTPUT_JSON = BACKEND_ROOT / "evals" / "results" / "project_metrics_report.json"
DEFAULT_OUTPUT_MD = BACKEND_ROOT / "evals" / "results" / "project_metrics_report.md"
TERMINAL_REQUEST_STATUSES = {"success", "partial_success", "failed"}
TERMINAL_REQUEST_PHASES = {"completed", "failed"}
TERMINAL_TASK_STATUSES = {"completed", "failed", "skipped"}
REVIEWED_CLAIM_STATUSES = {"supported", "missing_citation", "invalid_citation"}


@dataclass(frozen=True)
class MetricRecord:
    request_id: str
    status: str
    phase: str
    elapsed_ms: float | None
    cache_hits: int
    cache_misses: int
    cache_exact_hits: int
    cache_semantic_hits: int
    cache_approximate_hits: int
    cache_approximate_dense_hits: int
    cache_approximate_sparse_hits: int
    cache_approximate_hybrid_hits: int
    todo_items: list[dict[str, Any]]
    source_label: str


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen_digests: set[str] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        deduped.append(path)
    return sorted(deduped)


def _default_eval_result_dirs() -> list[Path]:
    return [path for path in DEFAULT_EVAL_RESULTS_DIRS if path.exists() and path.is_dir()]


def _normalize_todo_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _first_float(*values: Any) -> float | None:
    for value in values:
        resolved = _safe_float(value)
        if resolved is not None:
            return resolved
    return None


def _is_terminal_request(status: str, phase: str) -> bool:
    normalized_status = str(status or "").strip().lower()
    normalized_phase = str(phase or "").strip().lower()
    return normalized_status in TERMINAL_REQUEST_STATUSES or normalized_phase in TERMINAL_REQUEST_PHASES


def normalize_request_snapshot(
    payload: dict[str, Any],
    *,
    source_label: str,
) -> MetricRecord | None:
    request_metrics = payload.get("request_metrics")
    metrics = request_metrics if isinstance(request_metrics, dict) else payload

    request_id = str(payload.get("request_id") or metrics.get("request_id") or "").strip()
    if not request_id:
        return None

    status = str(metrics.get("status") or payload.get("status") or "").strip().lower()
    phase = str(payload.get("phase") or "").strip().lower()
    todo_items = _normalize_todo_items(payload.get("todo_items") or metrics.get("todo_items"))
    cache_diagnostics = payload.get("cache_diagnostics") if isinstance(payload.get("cache_diagnostics"), dict) else {}

    return MetricRecord(
        request_id=request_id,
        status=status,
        phase=phase,
        elapsed_ms=_first_float(metrics.get("elapsed_ms"), payload.get("elapsed_ms")),
        cache_hits=_safe_int(metrics.get("cache_hits")) or _safe_int(cache_diagnostics.get("cache_hits")),
        cache_misses=_safe_int(metrics.get("cache_misses")) or _safe_int(cache_diagnostics.get("cache_misses")),
        cache_exact_hits=(
            _safe_int(metrics.get("cache_exact_hits"))
            or _safe_int(cache_diagnostics.get("cache_exact_hits"))
        ),
        cache_semantic_hits=(
            _safe_int(metrics.get("cache_semantic_hits"))
            or _safe_int(cache_diagnostics.get("cache_semantic_hits"))
        ),
        cache_approximate_hits=(
            _safe_int(metrics.get("cache_approximate_hits"))
            or _safe_int(cache_diagnostics.get("cache_approximate_hits"))
            or _safe_int(metrics.get("cache_semantic_hits"))
            or _safe_int(cache_diagnostics.get("cache_semantic_hits"))
        ),
        cache_approximate_dense_hits=(
            _safe_int(metrics.get("cache_approximate_dense_hits"))
            or _safe_int(cache_diagnostics.get("cache_approximate_dense_hits"))
        ),
        cache_approximate_sparse_hits=(
            _safe_int(metrics.get("cache_approximate_sparse_hits"))
            or _safe_int(cache_diagnostics.get("cache_approximate_sparse_hits"))
        ),
        cache_approximate_hybrid_hits=(
            _safe_int(metrics.get("cache_approximate_hybrid_hits"))
            or _safe_int(cache_diagnostics.get("cache_approximate_hybrid_hits"))
        ),
        todo_items=todo_items,
        source_label=source_label,
    )


def normalize_eval_result(
    payload: dict[str, Any],
    *,
    source_label: str,
    fallback_latency_keys: Sequence[str],
) -> MetricRecord | None:
    trace = payload.get("trace")
    trace_payload = trace if isinstance(trace, dict) else {}

    request_id = str(payload.get("request_id") or trace_payload.get("request_id") or "").strip()
    if not request_id:
        return None

    latency_candidates = [trace_payload.get("elapsed_ms")]
    latency_candidates.extend(payload.get(key) for key in fallback_latency_keys)

    return MetricRecord(
        request_id=request_id,
        status=str(trace_payload.get("status") or "").strip().lower(),
        phase="completed",
        elapsed_ms=_first_float(*latency_candidates),
        cache_hits=_safe_int(trace_payload.get("cache_hits")),
        cache_misses=_safe_int(trace_payload.get("cache_misses")),
        cache_exact_hits=_safe_int(trace_payload.get("cache_exact_hits")),
        cache_semantic_hits=_safe_int(trace_payload.get("cache_semantic_hits")),
        cache_approximate_hits=(
            _safe_int(trace_payload.get("cache_approximate_hits"))
            or _safe_int(trace_payload.get("cache_semantic_hits"))
        ),
        cache_approximate_dense_hits=_safe_int(trace_payload.get("cache_approximate_dense_hits")),
        cache_approximate_sparse_hits=_safe_int(trace_payload.get("cache_approximate_sparse_hits")),
        cache_approximate_hybrid_hits=_safe_int(trace_payload.get("cache_approximate_hybrid_hits")),
        todo_items=_normalize_todo_items(payload.get("todo_items") or trace_payload.get("todo_items")),
        source_label=source_label,
    )


def aggregate_records(records: Sequence[MetricRecord]) -> dict[str, Any]:
    request_status_counts: Counter[str] = Counter()
    task_status_counts: Counter[str] = Counter()
    claim_support_counts: Counter[str] = Counter()
    latencies: list[float] = []
    cache_hits = 0
    cache_misses = 0
    cache_exact_hits = 0
    cache_semantic_hits = 0
    cache_approximate_hits = 0
    cache_approximate_dense_hits = 0
    cache_approximate_sparse_hits = 0
    cache_approximate_hybrid_hits = 0

    for record in records:
        if record.status:
            request_status_counts[record.status] += 1
        if record.elapsed_ms is not None:
            latencies.append(record.elapsed_ms)

        cache_hits += max(record.cache_hits, 0)
        cache_misses += max(record.cache_misses, 0)
        cache_exact_hits += max(record.cache_exact_hits, 0)
        cache_semantic_hits += max(record.cache_semantic_hits, 0)
        cache_approximate_hits += max(record.cache_approximate_hits, 0)
        cache_approximate_dense_hits += max(record.cache_approximate_dense_hits, 0)
        cache_approximate_sparse_hits += max(record.cache_approximate_sparse_hits, 0)
        cache_approximate_hybrid_hits += max(record.cache_approximate_hybrid_hits, 0)

        for task in record.todo_items:
            task_status = str(task.get("status") or "").strip().lower() or "unknown"
            task_status_counts[task_status] += 1
            for claim in task.get("claims") or []:
                if not isinstance(claim, dict):
                    continue
                support_status = str(claim.get("support_status") or "").strip().lower() or "unknown"
                claim_support_counts[support_status] += 1

    completed_tasks = task_status_counts.get("completed", 0)
    failed_tasks = task_status_counts.get("failed", 0)
    skipped_tasks = task_status_counts.get("skipped", 0)
    terminal_task_total = completed_tasks + failed_tasks + skipped_tasks

    supported_claims = claim_support_counts.get("supported", 0)
    missing_citation_claims = claim_support_counts.get("missing_citation", 0)
    invalid_citation_claims = claim_support_counts.get("invalid_citation", 0)
    reviewed_claim_total = supported_claims + missing_citation_claims + invalid_citation_claims

    cache_total = cache_hits + cache_misses

    return {
        "request_count": len(records),
        "request_status_counts": dict(sorted(request_status_counts.items())),
        "task_counts": dict(sorted(task_status_counts.items())),
        "task_completion_rate": round(completed_tasks / terminal_task_total, 4)
        if terminal_task_total
        else None,
        "task_terminal_total": terminal_task_total,
        "claim_support_counts": dict(sorted(claim_support_counts.items())),
        "reviewed_claim_total": reviewed_claim_total,
        "citation_validity_rate": round(supported_claims / reviewed_claim_total, 4)
        if reviewed_claim_total
        else None,
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "exact_hits": cache_exact_hits,
            "semantic_hits": cache_semantic_hits,
            "approximate_hits": cache_approximate_hits,
            "approximate_dense_hits": cache_approximate_dense_hits,
            "approximate_sparse_hits": cache_approximate_sparse_hits,
            "approximate_hybrid_hits": cache_approximate_hybrid_hits,
            "hit_rate": round(cache_hits / cache_total, 4) if cache_total else None,
        },
        "latency_ms": {
            "average": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "min": round(min(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
    }


def load_request_state_records(
    *,
    directories: Sequence[Path],
    include_nonterminal: bool,
) -> tuple[list[MetricRecord], dict[str, Any]]:
    candidates: list[Path] = []
    for directory in directories:
        if directory.exists() and directory.is_dir():
            candidates.extend(sorted(directory.glob("*.json")))
    paths = _dedupe_paths(candidates)

    records: list[MetricRecord] = []
    skipped_nonterminal = 0
    invalid_files = 0
    for path in paths:
        payload = _read_json(path)
        if not payload:
            invalid_files += 1
            continue
        record = normalize_request_snapshot(payload, source_label=path.name)
        if record is None:
            invalid_files += 1
            continue
        if not include_nonterminal and not _is_terminal_request(record.status, record.phase):
            skipped_nonterminal += 1
            continue
        records.append(record)

    metadata = {
        "source_dirs": [str(path.resolve()) for path in directories if path.exists()],
        "discovered_files": len(paths),
        "included_records": len(records),
        "skipped_nonterminal_records": skipped_nonterminal,
        "invalid_files": invalid_files,
    }
    return records, metadata


def load_eval_result_summaries(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[MetricRecord], set[str]]:
    file_summaries: list[dict[str, Any]] = []
    all_records: list[MetricRecord] = []
    request_ids: set[str] = set()

    for path in _dedupe_paths(paths):
        payload = _read_json(path)
        if not payload:
            file_summaries.append(
                {
                    "path": str(path.resolve()),
                    "kind": "invalid_json",
                }
            )
            continue

        if payload.get("sync_suite") or payload.get("stream_suite"):
            sync_results = ((payload.get("sync_suite") or {}).get("results") or [])
            stream_results = ((payload.get("stream_suite") or {}).get("results") or [])

            sync_records = [
                record
                for record in (
                    normalize_eval_result(
                        item,
                        source_label=f"{path.name}#sync",
                        fallback_latency_keys=("client_latency_ms",),
                    )
                    for item in sync_results
                    if isinstance(item, dict)
                )
                if record is not None
            ]
            stream_records = [
                record
                for record in (
                    normalize_eval_result(
                        item,
                        source_label=f"{path.name}#stream",
                        fallback_latency_keys=("total_stream_duration_ms",),
                    )
                    for item in stream_results
                    if isinstance(item, dict)
                )
                if record is not None
            ]

            file_records = [*sync_records, *stream_records]
            for record in file_records:
                request_ids.add(record.request_id)
            all_records.extend(file_records)

            file_summaries.append(
                {
                    "path": str(path.resolve()),
                    "kind": "http_suite",
                    "generated_at": payload.get("generated_at"),
                    "sync_summary": aggregate_records(sync_records),
                    "stream_summary": aggregate_records(stream_records),
                    "combined_summary": aggregate_records(file_records),
                    "embedded_sync_summary": ((payload.get("sync_suite") or {}).get("summary") or {}),
                    "embedded_stream_summary": ((payload.get("stream_suite") or {}).get("summary") or {}),
                }
            )
            continue

        if isinstance(payload.get("results"), list) and isinstance(payload.get("summary"), dict):
            benchmark_records = [
                record
                for record in (
                    normalize_eval_result(
                        item,
                        source_label=path.name,
                        fallback_latency_keys=(),
                    )
                    for item in payload.get("results") or []
                    if isinstance(item, dict)
                )
                if record is not None
            ]
            for record in benchmark_records:
                request_ids.add(record.request_id)
            all_records.extend(benchmark_records)

            file_summaries.append(
                {
                    "path": str(path.resolve()),
                    "kind": "benchmark",
                    "generated_at": payload.get("generated_at"),
                    "summary": aggregate_records(benchmark_records),
                    "embedded_summary": payload.get("summary") or {},
                }
            )
            continue

        file_summaries.append(
            {
                "path": str(path.resolve()),
                "kind": "unsupported",
            }
        )

    return file_summaries, all_records, request_ids


def build_report_payload(
    *,
    request_state_dirs: Sequence[Path],
    eval_result_paths: Sequence[Path],
    include_nonterminal: bool,
) -> dict[str, Any]:
    request_records, request_state_meta = load_request_state_records(
        directories=request_state_dirs,
        include_nonterminal=include_nonterminal,
    )
    eval_summaries, eval_records, eval_request_ids = load_eval_result_summaries(eval_result_paths)
    matched_request_records = [
        record for record in request_records if record.request_id in eval_request_ids
    ]

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_state": {
            **request_state_meta,
            "summary_all": aggregate_records(request_records),
            "summary_matched_eval_request_ids": (
                aggregate_records(matched_request_records) if matched_request_records else None
            ),
        },
        "eval_results": {
            "discovered_files": len(eval_result_paths),
            "summaries": eval_summaries,
            "aggregate_summary": aggregate_records(eval_records),
            "request_id_count": len(eval_request_ids),
        },
    }


def _format_metric(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def _render_summary_block(title: str, summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return [f"### {title}", "", "- 无可用数据", ""]

    task_counts = summary.get("task_counts") or {}
    cache = summary.get("cache") or {}
    latency = summary.get("latency_ms") or {}
    claim_counts = summary.get("claim_support_counts") or {}

    return [
        f"### {title}",
        "",
        f"- 请求数：{summary.get('request_count', 0)}",
        f"- 任务完成率：{_format_metric(summary.get('task_completion_rate'))}",
        (
            "- 任务状态："
            f"completed={task_counts.get('completed', 0)}, "
            f"failed={task_counts.get('failed', 0)}, "
            f"skipped={task_counts.get('skipped', 0)}, "
            f"other={sum(count for status, count in task_counts.items() if status not in TERMINAL_TASK_STATUSES)}"
        ),
        (
            "- 引用有效率："
            f"{_format_metric(summary.get('citation_validity_rate'))} "
            f"(supported={claim_counts.get('supported', 0)}, "
            f"missing={claim_counts.get('missing_citation', 0)}, "
            f"invalid={claim_counts.get('invalid_citation', 0)})"
        ),
        (
            "- 缓存命中率："
            f"{_format_metric(cache.get('hit_rate'))} "
            f"(hits={cache.get('hits', 0)}, misses={cache.get('misses', 0)}, "
            f"exact={cache.get('exact_hits', 0)}, approximate={cache.get('approximate_hits', 0)}, "
            f"semantic_compat={cache.get('semantic_hits', 0)})"
        ),
        f"- 平均耗时：{_format_metric(latency.get('average'), suffix=' ms')}",
        "",
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Project Metrics Report",
        "",
        "## 口径",
        "",
        "- 任务完成率 = completed / (completed + failed + skipped)",
        "- 引用有效率 = supported / (supported + missing_citation + invalid_citation)",
        "- 缓存命中率 = cache_hits / (cache_hits + cache_misses)",
        "- 平均耗时 = 平均 request elapsed_ms",
        "",
        f"- 生成时间：{payload.get('generated_at')}",
        "",
        "## Request State",
        "",
    ]

    request_state = payload.get("request_state") or {}
    lines.extend(
        [
            f"- 请求快照目录：{', '.join(request_state.get('source_dirs') or []) or 'N/A'}",
            f"- 发现文件数：{request_state.get('discovered_files', 0)}",
            f"- 纳入统计记录数：{request_state.get('included_records', 0)}",
            f"- 跳过非终态记录数：{request_state.get('skipped_nonterminal_records', 0)}",
            "",
        ]
    )
    lines.extend(_render_summary_block("全部请求快照", request_state.get("summary_all")))
    lines.extend(
        _render_summary_block(
            "与 Eval Request ID 匹配的请求快照",
            request_state.get("summary_matched_eval_request_ids"),
        )
    )

    eval_results = payload.get("eval_results") or {}
    lines.extend(
        [
            "## Eval Results",
            "",
            f"- 发现 JSON 文件数：{eval_results.get('discovered_files', 0)}",
            f"- 提取到的 request_id 数：{eval_results.get('request_id_count', 0)}",
            "",
        ]
    )
    lines.extend(_render_summary_block("全部 Eval 请求记录", eval_results.get("aggregate_summary")))

    for item in eval_results.get("summaries") or []:
        path = item.get("path") or ""
        kind = item.get("kind") or "unknown"
        lines.append(f"### Eval File: {Path(path).name if path else 'unknown'}")
        lines.append("")
        lines.append(f"- 类型：{kind}")
        lines.append(f"- 路径：{path}")
        if kind == "http_suite":
            lines.append("")
            lines.extend(_render_summary_block("Sync", item.get("sync_summary")))
            lines.extend(_render_summary_block("Stream", item.get("stream_summary")))
        elif kind == "benchmark":
            lines.append("")
            lines.extend(_render_summary_block("Benchmark", item.get("summary")))
        else:
            lines.append("")
            lines.append("- 当前文件未识别为可聚合的 eval 结果结构")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _resolve_output_path(path: str | Path) -> Path:
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = (BACKEND_ROOT / destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _write_outputs(*, payload: dict[str, Any], output_json: str | Path, output_md: str | Path) -> tuple[str, str]:
    json_path = _resolve_output_path(output_json)
    md_path = _resolve_output_path(output_md)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return str(json_path), str(md_path)


def _collect_eval_paths(*, explicit_files: Sequence[str], explicit_dirs: Sequence[str]) -> list[Path]:
    files: list[Path] = [Path(path).expanduser() for path in explicit_files]

    directories = [Path(path).expanduser() for path in explicit_dirs]
    if not directories and not files:
        directories = _default_eval_result_dirs()

    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            continue
        files.extend(sorted(directory.glob("*.json")))

    return _dedupe_paths(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate task completion, citation validity, cache hit rate, and latency metrics.",
    )
    parser.add_argument(
        "--request-state-dir",
        action="append",
        default=[],
        help="Request snapshot directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--eval-result",
        action="append",
        default=[],
        help="Explicit eval JSON file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--eval-results-dir",
        action="append",
        default=[],
        help="Directory scanned for eval JSON files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-nonterminal",
        action="store_true",
        help="Include in-progress request snapshots in the request-state summary.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    request_state_dirs = [Path(path).expanduser() for path in args.request_state_dir] or [DEFAULT_REQUEST_STATE_DIR]
    eval_result_paths = _collect_eval_paths(
        explicit_files=args.eval_result,
        explicit_dirs=args.eval_results_dir,
    )

    payload = build_report_payload(
        request_state_dirs=request_state_dirs,
        eval_result_paths=eval_result_paths,
        include_nonterminal=args.include_nonterminal,
    )
    json_path, md_path = _write_outputs(
        payload=payload,
        output_json=args.output_json,
        output_md=args.output_md,
    )

    sys.stdout.write(f"Request-state records: {payload['request_state']['included_records']}\n")
    sys.stdout.write(f"Eval JSON files: {payload['eval_results']['discovered_files']}\n")
    sys.stdout.write(f"JSON report written to: {json_path}\n")
    sys.stdout.write(f"Markdown report written to: {md_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
