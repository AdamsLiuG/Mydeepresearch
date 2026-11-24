"""Deterministic RLVR reward functions shared by TRL GRPOTrainer and verl."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evals.export_rl_trajectories import compute_verifiable_reward  # noqa: E402
from services.policy_controller import (  # noqa: E402
    ALLOWED_POLICY_ACTIONS,
    PolicyDecision,
    PolicyStateFeatures,
    is_budget_legal_action,
)


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "")
    return str(completion or "")


def _parse_policy_json(raw_text: str) -> tuple[dict[str, Any] | None, bool]:
    text = str(raw_text or "").strip()
    if not text:
        return None, False
    had_fence = False
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        had_fence = True
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, had_fence
    if not isinstance(parsed, dict):
        return None, had_fence
    return parsed, had_fence


def _coerce_features(item: Any) -> PolicyStateFeatures:
    if isinstance(item, PolicyStateFeatures):
        return item
    data = item if isinstance(item, dict) else {}
    return PolicyStateFeatures(
        task_id=int(data.get("task_id") or 1),
        round_index=int(data.get("round_index") or 1),
        max_rounds=max(1, int(data.get("max_rounds") or 3)),
        evidence_count=int(data.get("evidence_count") or 0),
        distinct_domains=int(data.get("distinct_domains") or 0),
        high_quality_sources=int(data.get("high_quality_sources") or 0),
        has_gap_signals=bool(data.get("has_gap_signals", False)),
        searches_left=int(data.get("searches_left") or 0),
        fetches_left=int(data.get("fetches_left") or 0),
        remaining_budget_ratio=float(data.get("remaining_budget_ratio") or 0.0),
        task_title=str(data.get("task_title") or ""),
        task_intent=str(data.get("task_intent") or ""),
        gap_signals=tuple(str(g) for g in (data.get("gap_signals") or ())),
        can_fetch=bool(data.get("can_fetch", False)),
        skill_pack_id=str(data.get("skill_pack_id") or "_neutral"),
        topic_domain=str(data.get("topic_domain") or "general"),
        freshness_ok=bool(data.get("freshness_ok", True)),
        evidence_sufficiency=bool(data.get("evidence_sufficiency", False)),
        archive_repair_hits=int(data.get("archive_repair_hits") or 0),
        recent_actions=tuple(str(a) for a in (data.get("recent_actions") or ())),
    )


def json_format_reward_fn(completions: list[Any], **kwargs: Any) -> list[float]:
    """Reward strict JSON adherence and valid action/confidence schema (+1.0 max, -1.0 min)."""

    del kwargs
    scores: list[float] = []
    for item in completions:
        text = _extract_completion_text(item)
        parsed, had_fence = _parse_policy_json(text)
        if parsed is None:
            scores.append(-1.0)
            continue
        action = str(parsed.get("action") or "").strip().lower()
        if action not in ALLOWED_POLICY_ACTIONS:
            scores.append(-0.75)
            continue
        try:
            conf = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            scores.append(-0.6)
            continue
        if not (0.0 <= conf <= 1.0):
            scores.append(-0.6)
            continue
        reason = str(parsed.get("reason") or "").strip()
        score = 1.0 if reason else 0.7
        if had_fence:
            score -= 0.1
        scores.append(round(score, 4))
    return scores


def budget_legality_reward_fn(
    completions: list[Any],
    *,
    state_features: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward strict compliance with searches_left / fetches_left budgets."""

    del kwargs
    feats_list = state_features or [{} for _ in completions]
    scores: list[float] = []
    for idx, item in enumerate(completions):
        raw_feat = feats_list[idx] if idx < len(feats_list) else {}
        features = _coerce_features(raw_feat)
        parsed, _ = _parse_policy_json(_extract_completion_text(item))
        if parsed is None:
            scores.append(-0.8)
            continue
        action = str(parsed.get("action") or "").strip().lower()
        decision = PolicyDecision(
            action=action,
            confidence=0.5,
            reason=str(parsed.get("reason") or ""),
            source="model",
            query_override=parsed.get("query_override"),
            target_url=parsed.get("target_url"),
        )
        scores.append(0.5 if is_budget_legal_action(features, decision) else -0.8)
    return scores


def outcome_alignment_reward_fn(
    completions: list[Any],
    *,
    state_features: list[Any] | None = None,
    snapshots: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Combine trajectory quality reward and state-action efficiency alignment."""

    del kwargs
    feats_list = state_features or [{} for _ in completions]
    snaps_list = snapshots or [{} for _ in completions]
    scores: list[float] = []
    for idx, item in enumerate(completions):
        raw_feat = feats_list[idx] if idx < len(feats_list) else {}
        feat_dict = raw_feat.to_dict() if isinstance(raw_feat, PolicyStateFeatures) else dict(raw_feat or {})
        snap = snaps_list[idx] if idx < len(snaps_list) else {}
        parsed, _ = _parse_policy_json(_extract_completion_text(item))
        normalized_text = json.dumps(parsed, ensure_ascii=False) if parsed is not None else _extract_completion_text(item)
        verifiable = compute_verifiable_reward(normalized_text, feat_dict, snap)
        scores.append(float(verifiable["reward"]))
    return scores


def verifiable_policy_reward_fn(
    completions: list[Any],
    *,
    state_features: list[Any] | None = None,
    snapshots: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Primary composite RLVR reward weighting schema legality, budget safety, and outcome."""

    fmt = json_format_reward_fn(completions, **kwargs)
    bud = budget_legality_reward_fn(completions, state_features=state_features, **kwargs)
    out = outcome_alignment_reward_fn(
        completions,
        state_features=state_features,
        snapshots=snapshots,
        **kwargs,
    )
    combined: list[float] = []
    for f_score, b_score, o_score in zip(fmt, bud, out):
        val = 0.25 * f_score + 0.30 * b_score + 0.45 * o_score
        combined.append(round(max(-1.0, min(1.0, val)), 6))
    return combined


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """verl-compatible scalar reward entrypoint (`custom_reward_function.name=compute_score`)."""

    del data_source, kwargs
    info = extra_info if isinstance(extra_info, dict) else {}
    gt = ground_truth if isinstance(ground_truth, dict) else {}
    features = info.get("state_features") or gt.get("state_features") or {}
    snapshot = info.get("snapshot") or gt.get("snapshot") or {}
    return verifiable_policy_reward_fn(
        [solution_str],
        state_features=[features],
        snapshots=[snapshot],
    )[0]
