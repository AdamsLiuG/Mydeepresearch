"""Offline RL trajectory scorer and SFT/DPO dataset exporter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _unwrap_metrics_and_state(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nested = snapshot.get("request_metrics")
    metrics = dict(nested) if isinstance(nested, dict) else dict(snapshot)
    return snapshot, metrics


def compute_trajectory_reward(
    snapshot: dict[str, Any],
    *,
    max_expected_tokens: int = 20000,
    max_expected_elapsed_ms: float = 120000.0,
) -> dict[str, Any]:
    """Compute a composite scalar reward and component breakdown from a request snapshot."""

    raw, metrics = _unwrap_metrics_and_state(snapshot)
    total_tasks = int(metrics.get("total_tasks") or len(raw.get("todo_items") or []) or 0)
    completed_tasks = int(metrics.get("completed_tasks") or 0)
    if completed_tasks == 0 and isinstance(raw.get("todo_items"), list):
        completed_tasks = sum(
            1
            for item in raw["todo_items"]
            if isinstance(item, dict) and str(item.get("status") or "").lower() == "completed"
        )

    status = str(metrics.get("status") or raw.get("status") or "").lower()
    report_markdown = str(
        raw.get("report_markdown")
        or metrics.get("report_markdown")
        or raw.get("running_summary")
        or ""
    ).strip()

    if total_tasks > 0:
        completion_score = min(1.0, max(0.0, completed_tasks / total_tasks))
    elif status in {"success", "completed"} or report_markdown:
        completion_score = 1.0
    else:
        completion_score = 0.0

    if len(report_markdown) >= 120:
        report_score = 1.0
    elif report_markdown:
        report_score = 0.5
    else:
        report_score = 0.0

    review_summary = metrics.get("review_summary") or raw.get("review_summary") or {}
    if not isinstance(review_summary, dict):
        review_summary = {}
    critical_issues = int(
        review_summary.get("critical_issues")
        or review_summary.get("error_count")
        or 0
    )
    warning_issues = int(
        review_summary.get("warning_issues")
        or review_summary.get("warning_count")
        or 0
    )
    review_score = max(0.0, min(1.0, 1.0 - 0.3 * critical_issues - 0.05 * warning_issues))
    if status in {"failed", "error"}:
        review_score = min(review_score, 0.25)

    base_quality = (
        0.4 * completion_score
        + 0.3 * report_score
        + 0.3 * review_score
    )

    total_tokens = max(0, int(metrics.get("total_tokens") or 0))
    elapsed_ms = max(0.0, float(metrics.get("elapsed_ms") or 0.0))
    degraded = bool(metrics.get("degraded", False))
    fallback_reasons = metrics.get("fallback_reasons") or []
    fallback_count = len(fallback_reasons) if isinstance(fallback_reasons, list) else int(metrics.get("fallback_triggered") or 0)

    token_penalty = 0.15 * min(1.0, total_tokens / max(1, int(max_expected_tokens)))
    latency_penalty = 0.1 * min(1.0, elapsed_ms / max(1.0, float(max_expected_elapsed_ms)))
    degraded_penalty = (0.2 if degraded else 0.0) + min(0.15, 0.05 * fallback_count)

    reward = max(-1.0, min(1.0, base_quality - token_penalty - latency_penalty - degraded_penalty))
    return {
        "reward": round(reward, 6),
        "base_quality": round(base_quality, 6),
        "completion_score": round(completion_score, 4),
        "report_score": round(report_score, 4),
        "review_score": round(review_score, 4),
        "token_penalty": round(token_penalty, 6),
        "latency_penalty": round(latency_penalty, 6),
        "degraded_penalty": round(degraded_penalty, 6),
    }


def _format_trajectory_response(snapshot: dict[str, Any], metrics: dict[str, Any]) -> str:
    policy_decisions = metrics.get("policy_decisions") or snapshot.get("policy_decisions") or []
    report_markdown = str(
        snapshot.get("report_markdown")
        or metrics.get("report_markdown")
        or snapshot.get("running_summary")
        or ""
    ).strip()
    payload = {
        "policy_decisions": policy_decisions if isinstance(policy_decisions, list) else [],
        "completed_tasks": int(metrics.get("completed_tasks") or 0),
        "total_tasks": int(metrics.get("total_tasks") or len(snapshot.get("todo_items") or []) or 0),
        "report_excerpt": report_markdown[:1500],
    }
    return json.dumps(payload, ensure_ascii=False)


def export_sft_records(
    snapshots: list[dict[str, Any]],
    *,
    min_reward: float = 0.5,
) -> list[dict[str, Any]]:
    """Export high-reward research trajectories as OpenAI/ShareGPT SFT JSONL records."""

    records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        raw, metrics = _unwrap_metrics_and_state(snapshot)
        topic = str(raw.get("topic") or metrics.get("topic") or "").strip()
        if not topic:
            continue
        scored = compute_trajectory_reward(snapshot)
        if scored["reward"] < float(min_reward):
            continue

        request_id = str(raw.get("request_id") or metrics.get("request_id") or "")
        records.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the DeepResearch RL policy and synthesis coordinator. "
                            "Optimize grounded coverage while controlling search/token budget."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Research topic: {topic}",
                    },
                    {
                        "role": "assistant",
                        "content": _format_trajectory_response(raw, metrics),
                    },
                ],
                "metadata": {
                    "request_id": request_id,
                    "topic": topic,
                    "reward_breakdown": scored,
                },
            }
        )
    return records


def export_dpo_pairs(
    snapshots: list[dict[str, Any]],
    *,
    min_reward_margin: float = 0.15,
) -> list[dict[str, Any]]:
    """Pair highest- and lowest-reward trajectories per topic when margin exceeds threshold."""

    grouped: dict[str, list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        raw, metrics = _unwrap_metrics_and_state(snapshot)
        topic = str(raw.get("topic") or metrics.get("topic") or "").strip()
        if not topic:
            continue
        scored = compute_trajectory_reward(snapshot)
        grouped[topic.lower()].append((scored["reward"], raw, metrics, scored))

    pairs: list[dict[str, Any]] = []
    for entries in grouped.values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda item: item[0])
        worst_reward, worst_raw, worst_metrics, _ = entries[0]
        best_reward, best_raw, best_metrics, _ = entries[-1]
        margin = round(best_reward - worst_reward, 6)
        if margin < float(min_reward_margin):
            continue

        topic = str(best_raw.get("topic") or best_metrics.get("topic") or "").strip()
        pairs.append(
            {
                "prompt": f"Research topic: {topic}",
                "chosen": _format_trajectory_response(best_raw, best_metrics),
                "rejected": _format_trajectory_response(worst_raw, worst_metrics),
                "chosen_reward": best_reward,
                "rejected_reward": worst_reward,
                "reward_margin": margin,
                "topic": topic,
                "chosen_request_id": str(best_raw.get("request_id") or best_metrics.get("request_id") or ""),
                "rejected_request_id": str(worst_raw.get("request_id") or worst_metrics.get("request_id") or ""),
            }
        )
    return pairs


def load_snapshots_from_dir(input_dir: Path) -> list[dict[str, Any]]:
    """Load persisted JSON snapshots from a directory."""

    if not input_dir.exists() or not input_dir.is_dir():
        return []
    snapshots: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            snapshots.append(payload)
    return snapshots


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export RL SFT/DPO datasets from DeepResearch snapshots.")
    parser.add_argument("--input-dir", default="backend/.state", help="Directory containing JSON request snapshots")
    parser.add_argument("--output-dir", default="backend/evals/rl_exports", help="Destination directory for JSONL exports")
    parser.add_argument("--min-reward", type=float, default=0.5, help="Minimum composite reward for SFT export")
    parser.add_argument("--min-margin", type=float, default=0.15, help="Minimum reward margin for DPO pairing")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    snapshots = load_snapshots_from_dir(input_dir)

    sft_records = export_sft_records(snapshots, min_reward=args.min_reward)
    dpo_pairs = export_dpo_pairs(snapshots, min_reward_margin=args.min_margin)

    sft_path = output_dir / "sft_trajectories.jsonl"
    dpo_path = output_dir / "dpo_pairs.jsonl"
    _write_jsonl(sft_path, sft_records)
    _write_jsonl(dpo_path, dpo_pairs)

    summary = {
        "input_snapshots": len(snapshots),
        "sft_exported": len(sft_records),
        "dpo_pairs_exported": len(dpo_pairs),
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
