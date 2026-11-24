"""Offline RL trajectory scorer, GRPO group builder, and SFT/DPO dataset exporter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from services.policy_controller import (  # noqa: E402
    ALLOWED_POLICY_ACTIONS,
    PolicyDecision,
    PolicyStateFeatures,
    format_policy_messages,
    is_budget_legal_action,
)


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
    severity_counts = review_summary.get("severity_counts")
    if not isinstance(severity_counts, dict):
        severity_counts = {}
    critical_issues = int(
        severity_counts.get("high")
        if severity_counts.get("high") is not None
        else (
            review_summary.get("critical_issues")
            or review_summary.get("error_count")
            or 0
        )
    )
    warning_issues = int(
        severity_counts.get("medium")
        if severity_counts.get("medium") is not None
        else (
            review_summary.get("warning_issues")
            or review_summary.get("warning_count")
            or 0
        )
    )
    verifier_issues = int(review_summary.get("verifier_issue_count") or 0)
    review_score = max(
        0.0,
        min(1.0, 1.0 - 0.3 * critical_issues - 0.05 * warning_issues - 0.08 * verifier_issues),
    )
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


def _features_from_decision_record(raw_features: dict[str, Any]) -> PolicyStateFeatures:
    return PolicyStateFeatures(
        task_id=int(raw_features.get("task_id") or 1),
        round_index=int(raw_features.get("round_index") or 1),
        max_rounds=max(1, int(raw_features.get("max_rounds") or 3)),
        evidence_count=int(raw_features.get("evidence_count") or 0),
        distinct_domains=int(raw_features.get("distinct_domains") or 0),
        high_quality_sources=int(raw_features.get("high_quality_sources") or 0),
        has_gap_signals=bool(raw_features.get("has_gap_signals", False)),
        searches_left=int(raw_features.get("searches_left") or 0),
        fetches_left=int(raw_features.get("fetches_left") or 0),
        remaining_budget_ratio=float(raw_features.get("remaining_budget_ratio") or 0.0),
        task_title=str(raw_features.get("task_title") or ""),
        task_intent=str(raw_features.get("task_intent") or ""),
        gap_signals=tuple(str(g) for g in (raw_features.get("gap_signals") or ())),
        can_fetch=bool(raw_features.get("can_fetch", False)),
        skill_pack_id=str(raw_features.get("skill_pack_id") or "_neutral"),
        topic_domain=str(raw_features.get("topic_domain") or "general"),
        freshness_ok=bool(raw_features.get("freshness_ok", True)),
        evidence_sufficiency=bool(raw_features.get("evidence_sufficiency", False)),
        archive_repair_hits=int(raw_features.get("archive_repair_hits") or 0),
        recent_actions=tuple(str(a) for a in (raw_features.get("recent_actions") or ())),
    )


def compute_verifiable_reward(
    response_text: str,
    features_dict: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Compute deterministic RLVR reward combining format, budget legality, and trajectory outcome."""

    traj_breakdown = compute_trajectory_reward(snapshot)
    traj_reward = float(traj_breakdown["reward"])

    parsed: dict[str, Any] | None = None
    try:
        loaded = json.loads(str(response_text or "").strip())
        if isinstance(loaded, dict):
            parsed = loaded
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        return {
            "reward": round(-0.9 + 0.1 * traj_reward, 6),
            "format_ok": False,
            "budget_legal": False,
            "trajectory_reward": traj_reward,
            "penalty_reason": "invalid_json",
        }

    action = str(parsed.get("action") or "").strip().lower()
    if action not in ALLOWED_POLICY_ACTIONS:
        return {
            "reward": round(-0.75 + 0.1 * traj_reward, 6),
            "format_ok": False,
            "budget_legal": False,
            "trajectory_reward": traj_reward,
            "penalty_reason": f"invalid_action:{action}",
        }

    try:
        conf = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = -1.0
    if not (0.0 <= conf <= 1.0):
        return {
            "reward": round(-0.6 + 0.1 * traj_reward, 6),
            "format_ok": False,
            "budget_legal": False,
            "trajectory_reward": traj_reward,
            "penalty_reason": "invalid_confidence",
        }

    features = _features_from_decision_record(features_dict)
    decision = PolicyDecision(
        action=action,
        confidence=conf,
        reason=str(parsed.get("reason") or ""),
        source="model",
        query_override=parsed.get("query_override"),
        target_url=parsed.get("target_url"),
    )
    budget_ok = is_budget_legal_action(features, decision)
    if not budget_ok:
        return {
            "reward": -0.5,
            "format_ok": True,
            "budget_legal": False,
            "trajectory_reward": traj_reward,
            "penalty_reason": f"budget_illegal:{action}",
        }

    # Alignment bonus for grounded stop vs continuing when needed
    action_bonus = 0.0
    if not features.has_gap_signals and features.evidence_sufficiency and action == "stop":
        action_bonus += 0.15
    elif features.has_gap_signals and action in {"search", "fetch"}:
        action_bonus += 0.10
    elif features.has_gap_signals and features.evidence_count == 0 and action == "stop":
        action_bonus -= 0.25

    combined = max(-1.0, min(1.0, 0.75 * traj_reward + 0.15 + action_bonus))
    return {
        "reward": round(combined, 6),
        "format_ok": True,
        "budget_legal": True,
        "trajectory_reward": traj_reward,
        "action_bonus": round(action_bonus, 4),
        "penalty_reason": None,
    }


def export_grpo_records(
    snapshots: list[dict[str, Any]],
    *,
    prompt_version: str = "policy_prompt_v2",
    min_group_size: int = 4,
    min_reward_std: float = 0.03,
) -> list[dict[str, Any]]:
    """Export grouped prompt+candidates+rewards records for GRPO verifiable RL training."""

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        raw, metrics = _unwrap_metrics_and_state(snapshot)
        topic = str(raw.get("topic") or metrics.get("topic") or "").strip()
        request_id = str(raw.get("request_id") or metrics.get("request_id") or "")
        decisions = metrics.get("policy_decisions") or raw.get("policy_decisions") or []
        if not isinstance(decisions, list):
            continue
        for entry in decisions:
            if not isinstance(entry, dict):
                continue
            raw_features = entry.get("features")
            raw_decision = entry.get("decision")
            if not isinstance(raw_features, dict) or not isinstance(raw_decision, dict):
                continue
            task_id = int(entry.get("task_id") or raw_features.get("task_id") or 1)
            round_idx = int(entry.get("round") or raw_features.get("round_index") or 1)
            response_json = json.dumps(
                {
                    "action": raw_decision.get("action"),
                    "confidence": raw_decision.get("confidence", 0.5),
                    "reason": raw_decision.get("reason", ""),
                    "query_override": raw_decision.get("query_override"),
                    "target_url": raw_decision.get("target_url"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            reward_info = compute_verifiable_reward(response_json, raw_features, snapshot)
            group_key = (topic.lower() or request_id, task_id, round_idx)
            grouped[group_key].append(
                {
                    "request_id": request_id,
                    "features": raw_features,
                    "response": response_json,
                    "reward": float(reward_info["reward"]),
                    "reward_breakdown": reward_info,
                }
            )

    records: list[dict[str, Any]] = []
    for (topic_key, task_id, round_idx), candidates in sorted(grouped.items()):
        if len(candidates) < max(1, int(min_group_size)):
            continue
        rewards = [c["reward"] for c in candidates]
        mean_r = sum(rewards) / len(rewards)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
        if std_r < float(min_reward_std):
            continue

        rep_features = _features_from_decision_record(candidates[0]["features"])
        messages = format_policy_messages(rep_features, prompt_version=prompt_version)
        records.append(
            {
                "prompt": messages[1]["content"],
                "messages": messages,
                "state_features": rep_features.to_dict(),
                "candidate_responses": [c["response"] for c in candidates],
                "rewards": rewards,
                "group_mean_reward": round(mean_r, 6),
                "group_reward_std": round(std_r, 6),
                "metadata": {
                    "topic_key": topic_key,
                    "task_id": task_id,
                    "round": round_idx,
                    "request_ids": [c["request_id"] for c in candidates],
                    "prompt_version": prompt_version,
                },
            }
        )
    return records


def load_snapshots_from_dir(input_dir: Path) -> list[dict[str, Any]]:
    """Load persisted JSON snapshots from a directory tree (recursively)."""

    if not input_dir.exists() or not input_dir.is_dir():
        return []
    snapshots: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
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
    parser = argparse.ArgumentParser(description="Export RL SFT/DPO/GRPO datasets from DeepResearch snapshots.")
    parser.add_argument("--input-dir", default="backend/.state/requests", help="Directory containing JSON request snapshots")
    parser.add_argument("--output-dir", default="backend/evals/rl_exports", help="Destination directory for JSONL exports")
    parser.add_argument("--mode", choices=["sft-dpo", "grpo-group"], default="sft-dpo", help="Export format mode")
    parser.add_argument("--min-reward", type=float, default=0.5, help="Minimum composite reward for SFT export")
    parser.add_argument("--min-margin", type=float, default=0.15, help="Minimum reward margin for DPO pairing")
    parser.add_argument("--min-group-size", type=int, default=4, help="Minimum group size for GRPO export")
    parser.add_argument("--min-reward-std", type=float, default=0.03, help="Minimum reward std-dev for GRPO export")
    parser.add_argument("--prompt-version", default="policy_prompt_v2", help="Prompt template version for GRPO export")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    snapshots = load_snapshots_from_dir(input_dir)

    if args.mode == "grpo-group":
        grpo_records = export_grpo_records(
            snapshots,
            prompt_version=args.prompt_version,
            min_group_size=args.min_group_size,
            min_reward_std=args.min_reward_std,
        )
        grpo_path = output_dir / "grpo_groups.jsonl"
        _write_jsonl(grpo_path, grpo_records)
        summary = {
            "mode": "grpo-group",
            "input_snapshots": len(snapshots),
            "grpo_groups_exported": len(grpo_records),
            "grpo_path": str(grpo_path),
        }
        sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return 0

    sft_records = export_sft_records(snapshots, min_reward=args.min_reward)
    dpo_pairs = export_dpo_pairs(snapshots, min_reward_margin=args.min_margin)

    sft_path = output_dir / "sft_trajectories.jsonl"
    dpo_path = output_dir / "dpo_pairs.jsonl"
    _write_jsonl(sft_path, sft_records)
    _write_jsonl(dpo_path, dpo_pairs)

    summary = {
        "mode": "sft-dpo",
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
