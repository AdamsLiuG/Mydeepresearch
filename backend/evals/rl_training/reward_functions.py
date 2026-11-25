"""Deterministic RLVR rewards for TRL replay and the veRL reward manager."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for entry in (_BACKEND_DIR, _BACKEND_DIR / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evals.export_rl_trajectories import (  # noqa: E402
    _as_dict,
    _count,
    _finite,
    _rate,
    compute_verifiable_reward,
    features_from_record,
    step_shaping,
)
from services.policy_controller import (  # noqa: E402
    PolicyDecision,
    is_budget_legal_action,
    parse_policy_decision,
)


def _completion_decision(completion: Any) -> PolicyDecision | None:
    if isinstance(completion, list):
        if len(completion) != 1:
            return None
        completion = completion[0]
    if isinstance(completion, dict):
        if completion.get("role", "assistant") != "assistant":
            return None
        completion = completion.get("content")
    return parse_policy_decision(completion) if isinstance(completion, str) else None


def _at(items: list[Any] | None, index: int) -> Any:
    return items[index] if isinstance(items, (list, tuple)) and index < len(items) else None


def json_format_reward_fn(completions: list[Any], **kwargs: Any) -> list[float]:
    """Reward only schema-valid, unfenced policy completions."""

    return [1.0 if _completion_decision(completion) is not None else -1.0 for completion in completions]


def budget_legality_reward_fn(
    completions: list[Any],
    *,
    state_features: list[Any] | None = None,
    verifiable_oracle: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    scores: list[float] = []
    for index, completion in enumerate(completions):
        decision = _completion_decision(completion)
        features = features_from_record(_at(state_features, index) or _at(verifiable_oracle, index))
        if decision is None or features is None:
            scores.append(-1.0)
        else:
            scores.append(0.5 if is_budget_legal_action(features, decision) else -0.85)
    return scores


def outcome_alignment_reward_fn(
    completions: list[Any],
    *,
    state_features: list[Any] | None = None,
    snapshots: list[Any] | None = None,
    verifiable_oracle: list[Any] | None = None,
    reference_completion: list[Any] | None = None,
    group_advantage: list[Any] | None = None,
    step_advantage: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Apply hard format/budget penalties, then bounded replay outcome and action shaping."""

    scores: list[float] = []
    for index, completion in enumerate(completions):
        decision = _completion_decision(completion)
        oracle = _as_dict(_at(verifiable_oracle, index))
        features = features_from_record(_at(state_features, index) or oracle)
        if decision is None or features is None:
            scores.append(-1.0)
            continue
        if not is_budget_legal_action(features, decision):
            scores.append(-0.85)
            continue

        if not oracle:
            result = compute_verifiable_reward(_as_dict(_at(snapshots, index)))
            oracle = {
                "final_citation_faithfulness": result["citation_faithfulness"],
                "final_hybrid_score": result["hybrid_judge_score"],
                "final_verifier_clean_rate": result["verifier_clean_rate"],
                "high_severity_issues_after_step": result["high_severity_issues"],
            }
        clean = _rate(oracle.get("final_verifier_clean_rate"))
        align = step_shaping(
            features, decision.action, clean_rate=clean,
            high_issues=_count(oracle.get("high_severity_issues_after_step")),
        )
        if align == -0.35:
            scores.append(align)
            continue

        reference = _completion_decision(_at(reference_completion, index))
        replay = 0.0
        # Outcome belongs to the action taken in the captured state. A different generated
        # action receives its own state alignment score, with no counterfactual outcome credit.
        if reference is not None and (
            decision.action, decision.query_override, decision.target_url
        ) == (reference.action, reference.query_override, reference.target_url):
            outcome = (
                0.40 * _rate(oracle.get("final_citation_faithfulness"))
                + 0.35 * _rate(oracle.get("final_hybrid_score"))
                + 0.25 * clean
            )
            advantage = _at(step_advantage, index)
            if advantage is None:
                advantage = _at(group_advantage, index)
            clipped_advantage = max(-1.5, min(1.5, _finite(advantage))) / 1.5
            replay = 0.45 * outcome + 0.25 * clipped_advantage
        scores.append(round(max(-1.0, min(1.0, 0.30 * align + replay)), 6))
    return scores


def verifiable_policy_reward_fn(completions: list[Any], **kwargs: Any) -> list[float]:
    """Composite reward with hard penalties applied once; use as the sole trainer reward."""

    alignment = outcome_alignment_reward_fn(completions, **kwargs)
    return [
        score if score <= -0.35 else round(min(1.0, 0.30 + score), 6)
        for score in alignment
    ]


def multiturn_trajectory_reward_fn(completions: list[Any], **kwargs: Any) -> list[float]:
    """Multi-turn RLVR reward incorporating bounded discounted turn credit when provided."""
    base_scores = verifiable_policy_reward_fn(completions, **kwargs)
    turns_col = kwargs.get("turns")
    gamma = max(0.01, min(1.0, _finite(kwargs.get("discount_gamma"), 0.9)))
    adjusted: list[float] = []
    for index, base in enumerate(base_scores):
        if base <= -0.35 or not isinstance(turns_col, (list, tuple)) or index >= len(turns_col):
            adjusted.append(base)
            continue
        row_turns = turns_col[index]
        if not isinstance(row_turns, list) or not row_turns:
            adjusted.append(base)
            continue
        weighted_returns: list[float] = []
        for offset, turn in enumerate(reversed(row_turns)):
            if not isinstance(turn, dict):
                continue
            ret = _finite(turn.get("discounted_return"), 0.0)
            weighted_returns.append(ret * (gamma ** offset))
        if not weighted_returns:
            adjusted.append(base)
            continue
        turn_bonus = 0.10 * max(-1.0, min(1.0, sum(weighted_returns) / len(weighted_returns)))
        adjusted.append(round(max(-1.0, min(1.0, base + turn_bonus)), 6))
    return adjusted


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Adapt the replay fields preserved in veRL's extra_info column."""

    info = {**_as_dict(ground_truth), **_as_dict(extra_info)}
    return verifiable_policy_reward_fn(
        [solution_str],
        state_features=[info.get("state_features")],
        snapshots=[info.get("snapshot")],
        verifiable_oracle=[info.get("verifiable_oracle")],
        reference_completion=[info.get("reference_completion")],
        group_advantage=[info.get("group_advantage")],
        step_advantage=[info.get("step_advantage")],
    )[0]
