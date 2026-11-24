"""Offline RL trajectory scorer, GRPO group builder, and SFT/DPO dataset exporter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _BACKEND_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from services.policy_controller import (  # noqa: E402
    PolicyStateFeatures,
    format_policy_messages,
    is_budget_legal_action,
    parse_policy_decision,
)


def _unwrap_metrics_and_state(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nested = snapshot.get("request_metrics", snapshot.get("trace"))
    metrics = dict(nested) if isinstance(nested, dict) else snapshot
    return snapshot, metrics


def compute_trajectory_reward(
    snapshot: dict[str, Any],
    *,
    max_expected_tokens: int = 20000,
    max_expected_elapsed_ms: float = 120000.0,
) -> dict[str, Any]:
    """Compute a composite scalar reward and component breakdown from a request snapshot."""

    raw, metrics = _unwrap_metrics_and_state(snapshot)
    tasks = _tasks(raw, metrics)
    total_tasks = _count(metrics.get("total_tasks")) or len(tasks)
    completed_tasks = metrics.get("completed_tasks")
    if completed_tasks is None:
        completed_tasks = sum(_as_dict(task).get("status") == "completed" for task in tasks)

    status = str(metrics.get("status") or raw.get("status") or "").lower()
    report_markdown = str(
        raw.get("report_markdown")
        or metrics.get("report_markdown")
        or raw.get("running_summary")
        or ""
    ).strip()

    if total_tasks > 0:
        completion_score = min(1.0, max(0.0, _count(completed_tasks) / total_tasks))
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

    review_summary = _as_dict(metrics.get("review_summary")) or _as_dict(raw.get("review_summary"))
    critical_issues = min(4, _issue_count(review_summary, "high", "critical_issues", "error_count"))
    warning_issues = min(20, _issue_count(review_summary, "medium", "warning_issues", "warning_count"))
    verifier_issues = min(13, _count(review_summary.get("verifier_issue_count")))
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

    token_penalty, latency_penalty, degraded_penalty = _operating_penalties(
        metrics, max_expected_tokens, max_expected_elapsed_ms, use_cache=False,
    )

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
        "policy_decisions": [
            {key: value for key, value in entry.items() if key not in {"telemetry", "shadow_decision"}}
            for entry in _as_list(policy_decisions)
            if isinstance(entry, dict) and entry.get("applied") is not False
        ],
        "completed_tasks": _count(metrics.get("completed_tasks")),
        "total_tasks": _count(metrics.get("total_tasks")) or len(_tasks(snapshot, metrics)),
        "report_excerpt": report_markdown[:1500],
    }
    return json.dumps(payload, ensure_ascii=True)


def export_sft_records(
    snapshots: list[dict[str, Any]],
    *,
    min_reward: float = 0.5,
) -> list[dict[str, Any]]:
    """Export high-reward research trajectories as OpenAI/ShareGPT SFT JSONL records."""

    if not -1 <= _finite(min_reward, -2) <= 1:
        raise ValueError("min_reward must be finite and within [-1, 1]")
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

    if not 0 <= _finite(min_reward_margin, -1) <= 2:
        raise ValueError("min_reward_margin must be finite and within [0, 2]")
    grouped: dict[tuple[str, str], list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        raw, metrics = _unwrap_metrics_and_state(snapshot)
        topic = str(raw.get("topic") or metrics.get("topic") or "").strip()
        if not topic:
            continue
        scored = compute_trajectory_reward(snapshot)
        tenant = str(raw.get("tenant_id") or metrics.get("tenant_id") or "default")
        grouped[(tenant, topic.casefold())].append((scored["reward"], raw, metrics, scored))

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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _count(value: Any) -> int:
    return int(max(0.0, _finite(value)))


def _rate(value: Any, default: float = 0.0) -> float:
    rate = _finite(value, default)
    return rate if 0 <= rate <= 1 else default


def _tasks(raw: dict[str, Any], metrics: dict[str, Any]) -> list[Any]:
    return _as_list(raw.get("todo_items")) or _as_list(metrics.get("todo_items"))


def _claim_statuses(tasks: list[Any]) -> list[str]:
    # A repeated supported sentence must not drown out unverified or invalid citations.
    priorities = {
        "supported": 0, "unverified_claim": 1, "contradicted": 2,
        "missing_citation": 3, "invalid_citation": 4,
    }
    claims: dict[str, str] = {}
    for task in tasks:
        for claim in _as_list(_as_dict(task).get("claims")):
            if not isinstance(claim, dict):
                continue
            status = str(claim.get("support_status", "")).casefold()
            if status not in priorities:
                status = "unverified_claim"
            key = " ".join(str(claim.get("text") or "").casefold().split())
            if not key:
                key = f"\0{status}"
            if priorities[status] >= priorities[claims.get(key, "supported")]:
                claims[key] = status
    return list(claims.values())


def _issue_count(review: dict[str, Any], severity: str, *legacy_keys: str) -> int:
    counts = _as_dict(review.get("severity_counts"))
    issues = _as_list(review.get("issues"))
    logged_count = sum(_as_dict(issue).get("severity") == severity for issue in issues)
    reported_count = counts.get(severity)
    if reported_count is None:
        reported_count = next((review[key] for key in legacy_keys if key in review), 0)
    return max(logged_count, _count(reported_count))


def _operating_penalties(
    metrics: dict[str, Any],
    max_expected_tokens: int,
    max_expected_elapsed_ms: float,
    *,
    use_cache: bool,
) -> tuple[float, float, float]:
    token_limit = max(1, _finite(max_expected_tokens, 20000))
    time_limit = max(1.0, _finite(max_expected_elapsed_ms, 120000.0))
    tokens = max(0.0, _finite(metrics.get("total_tokens", 0), token_limit))
    if use_cache:
        tokens -= min(tokens, _count(metrics.get("prompt_tokens")), _count(metrics.get("cached_prompt_tokens")))
    elapsed = max(0.0, _finite(metrics.get("elapsed_ms", 0), time_limit))
    fallback_count = len(_as_list(metrics.get("fallback_reasons"))) or bool(metrics.get("fallback_triggered"))
    return (
        0.15 * min(1.0, tokens / token_limit),
        0.10 * min(1.0, elapsed / time_limit),
        (0.20 if metrics.get("degraded") else 0.0) + min(0.15, 0.05 * fallback_count),
    )


def compute_verifiable_reward(
    snapshot: dict[str, Any],
    *,
    judge_metrics: dict[str, Any] | None = None,
    max_expected_tokens: int = 20000,
    max_expected_elapsed_ms: float = 120000.0,
) -> dict[str, Any]:
    """Score persisted review evidence and operational cost without rewarding report length."""

    raw, metrics = _unwrap_metrics_and_state(snapshot)
    tasks = _tasks(raw, metrics)
    review = _as_dict(metrics.get("review_summary")) or _as_dict(raw.get("review_summary"))
    high = _issue_count(review, "high", "critical_issues", "error_count")
    medium = _issue_count(review, "medium", "warning_issues", "warning_count")
    statuses = _claim_statuses(tasks)
    invalid = "invalid_citation" in statuses
    missing = "missing_citation" in statuses
    citation_rate = statuses.count("supported") / len(statuses) if statuses else 0.0
    nli = (
        _as_dict(raw.get("claim_verification_summary"))
        or _as_dict(metrics.get("claim_verification_summary"))
        or _as_dict(review.get("claim_verification_summary"))
    )
    atomic_rate = _rate(nli.get("atomic_faithfulness_rate"), -1)
    nli_used = (
        nli.get("degraded", False) is False
        and _count(nli.get("total_atomic_claims")) > 0
        and atomic_rate >= 0
    )
    effective_rate = atomic_rate if nli_used else citation_rate
    if invalid or missing:
        effective_rate = min(effective_rate, citation_rate)
    judge = (
        _as_dict(judge_metrics) if judge_metrics is not None
        else _as_dict(raw.get("judge_metrics")) or _as_dict(metrics.get("judge_metrics")) or _as_dict(raw.get("metrics"))
    )
    reference_match = min(effective_rate, _rate(judge.get("reference_match_rate"), citation_rate))
    grounded_ratio = min(effective_rate, _rate(judge.get("grounded_bullet_ratio"), citation_rate))
    faithfulness = 0.50 * effective_rate + 0.25 * reference_match + 0.25 * grounded_ratio
    heuristic = _rate(judge.get("heuristic_composite_score"), citation_rate)
    hybrid = _rate(judge.get("hybrid_score"), heuristic)
    if effective_rate == 0 or invalid or missing:
        hybrid = min(hybrid, effective_rate)

    pack = _as_dict(raw.get("skill_pack")) or _as_dict(metrics.get("skill_pack"))
    expected_verifiers = {v for v in _as_list(pack.get("active_verifiers")) if isinstance(v, str)}
    executed = {v for v in _as_list(review.get("verifiers_executed")) if isinstance(v, str)}
    raw_verifier_count = _finite(review.get("verifier_issue_count"), -1)
    audited = raw_verifier_count >= 0 and raw_verifier_count.is_integer() and expected_verifiers <= executed
    conflict = (
        "contradicted" in statuses
        or _count(review.get("unresolved_conflict_count")) > 0
        or (nli_used and (
            _count(nli.get("contradicted_count")) > 0 or _count(nli.get("unresolved_conflict_count")) > 0
        ))
    )
    verifier_clean = 0.0
    if audited and not (high or invalid or missing or review.get("overall_status") == "blocked"):
        verifier_clean = max(0.0, 1.0 - 0.35 * min(raw_verifier_count, 3) - 0.10 * min(medium, 10) - 0.25 * conflict)

    total = max(1, _count(metrics.get("total_tasks")) or len(tasks))
    done = metrics.get("completed_tasks")
    if done is None:
        done = sum(_as_dict(task).get("status") == "completed" for task in tasks)
    completion = min(1.0, _count(done) / total)
    quality = 0.35 * faithfulness + 0.30 * hybrid + 0.20 * verifier_clean + 0.15 * completion
    tokens, latency, degraded = _operating_penalties(metrics, max_expected_tokens, max_expected_elapsed_ms, use_cache=True)
    return {
        "reward": round(max(-1.0, min(1.0, quality - tokens - latency - degraded)), 6),
        "outcome_quality": round(quality, 6),
        "citation_validity_rate": round(citation_rate, 6),
        "citation_faithfulness": round(faithfulness, 6),
        "hybrid_judge_score": round(hybrid, 6),
        "verifier_clean_rate": round(verifier_clean, 6),
        "completion_score": round(completion, 6),
        "high_severity_issues": high,
        "token_penalty": round(tokens, 6),
        "latency_penalty": round(latency, 6),
        "degraded_penalty": round(degraded, 6),
        "nli_signal_used": nli_used,
    }


def features_from_record(data: Any) -> PolicyStateFeatures | None:
    """Reconstruct usable replay state, retaining defaults for fields absent from old traces."""

    if isinstance(data, PolicyStateFeatures):
        data = data.to_dict()
    if not isinstance(data, dict) or not data:
        return None
    defaults = dict(PolicyStateFeatures(
        task_id=1, round_index=1, max_rounds=3, evidence_count=0,
        distinct_domains=0, high_quality_sources=0, has_gap_signals=False,
        searches_left=0, fetches_left=0, remaining_budget_ratio=0.0,
    ).to_dict())
    for key, default in defaults.items():
        value = data.get(key, default)
        if isinstance(default, tuple | list):
            if not isinstance(value, tuple | list) or not all(isinstance(item, str) for item in value):
                return None
            defaults[key] = tuple(value)
        elif isinstance(default, float):
            if type(value) not in (int, float) or _rate(value, -1) < 0:
                return None
            defaults[key] = float(value)
        elif type(value) is not type(default) or (type(value) is int and value < 0):
            return None
        else:
            defaults[key] = value
    if min(defaults["task_id"], defaults["max_rounds"]) < 1:
        return None
    return PolicyStateFeatures(**defaults)


def step_shaping(features: PolicyStateFeatures, action: str, *, clean_rate: float, high_issues: int) -> float:
    """Score budget-aware action alignment independently of the recorded action."""

    can_continue = features.round_index < features.max_rounds and (
        features.searches_left > 0 or (features.fetches_left > 0 and features.can_fetch)
    )
    unresolved = features.has_gap_signals or not features.freshness_ok or high_issues > 0 or features.evidence_count == 0
    if action == "stop" and can_continue and unresolved:
        return -0.35 if high_issues > 0 or clean_rate < 1 or features.evidence_count == 0 else -0.15
    if action == "stop":
        return 0.30 if features.evidence_sufficiency and not unresolved else 0.0
    if action == "fetch" and "thin_content" in features.gap_signals:
        return 0.25
    if features.evidence_sufficiency and not unresolved:
        return -0.15
    return 0.10


def _decision_steps(snapshot: dict[str, Any]) -> list[tuple[PolicyStateFeatures, str, str]]:
    raw, metrics = _unwrap_metrics_and_state(snapshot)
    decisions = _as_list(metrics.get("policy_decisions")) or _as_list(raw.get("policy_decisions"))
    steps: dict[tuple[int, int], tuple[PolicyStateFeatures, str, str]] = {}
    for record in decisions:
        entry = _as_dict(record)
        if entry.get("applied") is False:
            continue
        features = features_from_record(entry.get("features"))
        fields = _as_dict(entry.get("decision"))
        if features is None or not fields or fields.get("source") in ("shadow", "rule_fallback"):
            continue
        try:
            completion = json.dumps(
                {key: fields.get(key) for key in ("action", "confidence", "reason", "query_override", "target_url")},
                ensure_ascii=False, sort_keys=True, allow_nan=False,
            )
        except (TypeError, ValueError):
            continue
        decision = parse_policy_decision(completion)
        if decision is not None and is_budget_legal_action(features, decision):
            steps[(features.task_id, features.round_index)] = (features, completion, decision.action)
    return list(steps.values())


def _rollout_groups(snapshots: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        raw, metrics = _unwrap_metrics_and_state(snapshot)
        status = str(raw.get("status") or metrics.get("status") or "").casefold()
        if status and status not in {"success", "partial_success", "completed", "failed", "error"}:
            continue
        request_id = str(raw.get("request_id") or metrics.get("request_id") or "").strip()
        topic = str(raw.get("topic") or metrics.get("topic") or "").strip().casefold()
        case_id = str(raw.get("case_id") or metrics.get("case_id") or "").strip()
        tenant = str(raw.get("tenant_id") or metrics.get("tenant_id") or "default")
        explicit_group = str(raw.get("group_id") or metrics.get("group_id") or "")
        if not request_id or not (case_id or topic):
            continue
        rollout_id = str(raw.get("rollout_id") or metrics.get("rollout_id") or request_id).strip()
        steps = _decision_steps(snapshot)
        if not steps or not rollout_id:
            continue
        members = groups[(tenant, case_id or topic, explicit_group)]
        if request_id in members or any(member["rollout_id"] == rollout_id for member in members.values()):
            continue
        members[request_id] = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "case_id": case_id,
            "topic": topic,
            "outcome": compute_verifiable_reward(snapshot),
            "steps": steps,
        }
    return {key: list(members.values()) for key, members in groups.items()}


def _checked_group_limits(min_group_size: int, min_reward_std: float) -> None:
    if type(min_group_size) is not int or min_group_size < 4:
        raise ValueError("min_group_size must be at least 4")
    if _finite(min_reward_std, -1) < 0:
        raise ValueError("min_reward_std must be finite and nonnegative")


def _reward_stats(rewards: list[float]) -> tuple[float, float]:
    mean = math.fsum(rewards) / len(rewards)
    std = math.sqrt(math.fsum((reward - mean) ** 2 for reward in rewards) / len(rewards))
    return mean, std


def collect_group_rollouts(
    case_id: str,
    run_rollout: Callable[[int], dict[str, Any]],
    *,
    group_id: str,
    group_size: int = 4,
) -> list[dict[str, Any]]:
    """Collect separate terminal snapshots using an offline runner supplied by the caller."""

    if not case_id.strip() or not group_id.strip() or type(group_size) is not int or not 4 <= group_size <= 8:
        raise ValueError("A case ID, group ID and 4 to 8 rollouts are required")
    rollouts = [
        dict(run_rollout(index), case_id=case_id, group_id=group_id)
        for index in range(group_size)
    ]
    retained = sum(map(len, _rollout_groups(rollouts).values()))
    if retained != group_size:
        raise ValueError("Every rollout must have a unique request ID and a usable terminal policy step")
    return rollouts


def export_grpo_step_records(
    snapshots: list[dict[str, Any]],
    *,
    prompt_version: str = "policy_prompt_v2",
    min_group_size: int = 4,
    min_reward_std: float = 0.03,
) -> list[dict[str, Any]]:
    """Export a training row for each actual decision in a distinct, informative rollout."""

    _checked_group_limits(min_group_size, min_reward_std)
    records: list[dict[str, Any]] = []
    for identity, rollouts in sorted(_rollout_groups(snapshots).items()):
        if len(rollouts) < min_group_size:
            continue
        rewards = [item["outcome"]["reward"] for item in rollouts]
        mean, std = _reward_stats(rewards)
        if std <= 0 or std < min_reward_std:
            continue
        group_id = "grp_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
        for rollout in rollouts:
            outcome = rollout["outcome"]
            advantage = (outcome["reward"] - mean) / std
            for index, (features, completion, action) in enumerate(rollout["steps"]):
                shaping = step_shaping(
                    features, action, clean_rate=outcome["verifier_clean_rate"],
                    high_issues=outcome["high_severity_issues"],
                )
                records.append({
                    "group_id": group_id,
                    "case_id": rollout["case_id"],
                    "rollout_id": rollout["rollout_id"],
                    "request_id": rollout["request_id"],
                    "skill_pack_id": features.skill_pack_id,
                    "task_id": features.task_id,
                    "step_index": index,
                    "prompt": format_policy_messages(features, prompt_version=prompt_version),
                    "state_features": features.to_dict(),
                    "reference_completion": completion,
                    "trajectory_reward": outcome["reward"],
                    "group_mean_reward": round(mean, 6),
                    "group_std_reward": round(std, 6),
                    "group_advantage": round(advantage, 6),
                    "step_shaping": shaping,
                    "step_advantage": round(advantage + shaping, 6),
                    "verifiable_oracle": {
                        **features.to_dict(),
                        "final_verifier_clean_rate": outcome["verifier_clean_rate"],
                        "final_citation_faithfulness": outcome["citation_faithfulness"],
                        "final_hybrid_score": outcome["hybrid_judge_score"],
                        "high_severity_issues_after_step": outcome["high_severity_issues"],
                    },
                })
    return records


def export_grpo_records(
    snapshots: list[dict[str, Any]],
    *,
    prompt_version: str = "policy_prompt_v2",
    min_group_size: int = 4,
    min_reward_std: float = 0.03,
) -> list[dict[str, Any]]:
    """Export the legacy grouped view when candidates share exactly the same prompt."""

    _checked_group_limits(min_group_size, min_reward_std)
    records: list[dict[str, Any]] = []
    for _, rollouts in sorted(_rollout_groups(snapshots).items()):
        by_state: dict[str, list[tuple[dict[str, Any], PolicyStateFeatures, str]]] = defaultdict(list)
        for rollout in rollouts:
            for features, completion, _ in rollout["steps"]:
                by_state[json.dumps(features.to_dict(), sort_keys=True)].append((rollout, features, completion))
        for candidates in by_state.values():
            if len(candidates) < min_group_size:
                continue
            rewards = [rollout["outcome"]["reward"] for rollout, _, _ in candidates]
            mean, std = _reward_stats(rewards)
            if std <= 0 or std < min_reward_std:
                continue
            rollout, features, _ = candidates[0]
            messages = format_policy_messages(features, prompt_version=prompt_version)
            records.append({
                "prompt": messages[1]["content"],
                "messages": messages,
                "state_features": features.to_dict(),
                "candidate_responses": [completion for _, _, completion in candidates],
                "rewards": rewards,
                "group_mean_reward": round(mean, 6),
                "group_reward_std": round(std, 6),
                "metadata": {
                    "topic_key": rollout["topic"],
                    "task_id": features.task_id,
                    "round": features.round_index,
                    "request_ids": [item["request_id"] for item, _, _ in candidates],
                    "prompt_version": prompt_version,
                },
            })
    return records


def load_snapshots_from_dir(input_dir: Path) -> list[dict[str, Any]]:
    """Load persisted JSON snapshots from a directory tree (recursively)."""

    if not input_dir.exists() or not input_dir.is_dir():
        return []
    snapshots: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        if any(part.startswith(".") for part in path.relative_to(input_dir).parts) or path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, RecursionError):
            continue
        if isinstance(payload, dict):
            snapshots.append(payload)
    return snapshots


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export RL SFT/DPO/GRPO datasets from DeepResearch snapshots.")
    parser.add_argument("--input-dir", default=str(_BACKEND_DIR / ".state" / "requests"), help="Directory containing JSON request snapshots")
    parser.add_argument("--output-dir", default=str(_BACKEND_DIR / "evals" / "rl_exports"), help="Destination directory for JSONL exports")
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
        step_records = export_grpo_step_records(
            snapshots, prompt_version=args.prompt_version,
            min_group_size=args.min_group_size, min_reward_std=args.min_reward_std,
        )
        step_path = output_dir / "grpo_step_records.jsonl"
        _write_jsonl(step_path, step_records)
        grpo_path = output_dir / "grpo_groups.jsonl"
        _write_jsonl(grpo_path, grpo_records)
        summary = {
            "mode": "grpo-group",
            "input_snapshots": len(snapshots),
            "grpo_groups_exported": len(grpo_records),
            "grpo_path": str(grpo_path),
            "grpo_steps_exported": len(step_records),
            "grpo_step_path": str(step_path),
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
