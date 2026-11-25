"""Difficulty-aware cascade model router with bounded escalation and budget pressure control."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

ModelTier = Literal["fast", "standard", "frontier"]

_TIER_ORDER: tuple[ModelTier, ...] = ("fast", "standard", "frontier")
_STAGE_BASE_WEIGHT: dict[str, float] = {
    "planning": 0.45,
    "task_react": 0.25,
    "summarization": 0.28,
    "reflection": 0.35,
    "review": 0.50,
    "report_outline": 0.45,
    "report_map": 0.42,
    "report_reduce": 0.62,
    "report": 0.58,
    "report_critic": 0.68,
    "report_revision": 0.64,
}
_COMPLEX_CUES = re.compile(
    r"(?:compare|comparison|trade-off|versus|vs\.?|benchmark|conflict|contrarian|"
    r"对比|比较|权衡|架构|演进|并发|冲突|归因|财务|监管|合规|综述)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteFeatures:
    """Deterministic routing signals gathered before dispatching an LLM turn."""

    stage: str
    topic: str = ""
    task_count: int = 0
    conflict_count: int = 0
    evidence_gap_count: int = 0
    budget_pressure: float = 0.0
    retry_index: int = 0
    domain: str = "general"


@dataclass(frozen=True)
class RouteDecision:
    """Routing choice and telemetry metadata produced by CascadeModelRouter."""

    stage: str
    tier: ModelTier
    model_name: str
    reason: str
    complexity_score: float
    escalated_from: ModelTier | None = None
    shadow_only: bool = True
    estimated_cost_saving_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "tier": self.tier,
            "model_name": self.model_name,
            "reason": self.reason,
            "complexity_score": round(float(self.complexity_score), 4),
            "shadow_only": bool(self.shadow_only),
            "estimated_cost_saving_ratio": round(float(self.estimated_cost_saving_ratio), 4),
        }
        if self.escalated_from is not None:
            payload["escalated_from"] = self.escalated_from
        return payload


class CascadeModelRouter:
    """Route turns across fast/standard/frontier tiers with bounded escalation."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._enabled = bool(getattr(config, "v4_model_router_enabled", False))
        raw_mode = str(getattr(config, "v4_model_router_mode", "off") or "off").strip().lower()
        self._mode: Literal["off", "shadow", "active"] = (
            raw_mode if raw_mode in {"off", "shadow", "active"} else "off"
        )
        self._max_escalations = max(
            0, min(2, int(getattr(config, "v4_model_router_max_escalations_per_stage", 1) or 1)),
        )
        raw_thresh = float(getattr(config, "v4_model_router_confidence_threshold", 0.72) or 0.72)
        self._confidence_threshold = min(1.0, max(0.0, raw_thresh)) if math.isfinite(raw_thresh) else 0.72

    @property
    def enabled(self) -> bool:
        return self._enabled and self._mode != "off"

    @property
    def mode(self) -> str:
        return self._mode if self._enabled else "off"

    @property
    def max_escalations(self) -> int:
        return self._max_escalations

    def model_for_tier(self, tier: ModelTier) -> str:
        default_model = ""
        if hasattr(self._config, "resolved_model"):
            default_model = str(self._config.resolved_model() or "").strip()
        fast_default = (
            str(getattr(self._config, "resolved_fast_llm_model", lambda: "")() or "").strip()
            or default_model
        )
        if tier == "fast":
            return str(getattr(self._config, "v4_model_router_fast_model", "") or "").strip() or fast_default
        if tier == "frontier":
            return str(getattr(self._config, "v4_model_router_frontier_model", "") or "").strip() or default_model
        return str(getattr(self._config, "v4_model_router_standard_model", "") or "").strip() or default_model

    @staticmethod
    def _cost_saving_ratio(tier: ModelTier) -> float:
        return {"fast": 0.55, "standard": 0.25, "frontier": 0.0}[tier]

    @staticmethod
    def score_complexity(features: RouteFeatures) -> float:
        stage_key = str(features.stage or "task_react").strip().lower()
        score = _STAGE_BASE_WEIGHT.get(stage_key, 0.35)
        topic = str(features.topic or "").strip()
        if len(topic) >= 60:
            score += 0.08
        cue_hits = len(_COMPLEX_CUES.findall(topic))
        score += min(0.18, cue_hits * 0.06)
        if int(features.task_count or 0) >= 5:
            score += 0.08
        if int(features.conflict_count or 0) > 0:
            score += min(0.22, int(features.conflict_count) * 0.11)
        if int(features.evidence_gap_count or 0) > 0:
            score += min(0.14, int(features.evidence_gap_count) * 0.07)
        if int(features.retry_index or 0) > 0:
            score += min(0.15, int(features.retry_index) * 0.08)
        domain = str(features.domain or "general").strip().lower()
        if domain in {"finance", "finance_filing", "legal", "legal_compliance", "clinical_biomed", "medical"}:
            score += 0.12
        return min(1.0, max(0.0, round(score, 4)))

    def select_tier(self, features: RouteFeatures) -> RouteDecision:
        """Select fast/standard/frontier deterministically for the incoming stage."""
        complexity = self.score_complexity(features)
        shadow_only = self.mode != "active"
        if not self.enabled:
            default_model = self.model_for_tier("standard")
            return RouteDecision(
                stage=features.stage,
                tier="standard",
                model_name=default_model,
                reason="router_disabled",
                complexity_score=complexity,
                shadow_only=True,
                estimated_cost_saving_ratio=0.0,
            )

        pressure = float(features.budget_pressure or 0.0)
        if not math.isfinite(pressure):
            pressure = 0.0
        pressure = min(1.0, max(0.0, pressure))

        if complexity < 0.42:
            tier: ModelTier = "fast"
            reason = "low_complexity"
        elif complexity < 0.72:
            tier = "standard"
            reason = "moderate_complexity"
        else:
            tier = "frontier"
            reason = "high_complexity"

        if pressure >= 0.90:
            tier = "fast"
            reason = "budget_pressure_fast"
        elif pressure >= 0.80 and tier == "frontier":
            tier = "standard"
            reason = "budget_pressure_cap"

        return RouteDecision(
            stage=features.stage,
            tier=tier,
            model_name=self.model_for_tier(tier),
            reason=reason,
            complexity_score=complexity,
            shadow_only=shadow_only,
            estimated_cost_saving_ratio=self._cost_saving_ratio(tier),
        )

    def evaluate_escalation(
        self,
        decision: RouteDecision,
        *,
        parse_ok: bool = True,
        confidence: float = 1.0,
        conflict_count: int = 0,
        citation_ok: bool = True,
        escalation_count: int = 0,
        budget_allows: bool = True,
    ) -> RouteDecision | None:
        """Escalate one tier up within the configured per-stage cap when validation degrades."""
        if not self.enabled or not budget_allows:
            return None
        if escalation_count >= self._max_escalations:
            return None
        idx = _TIER_ORDER.index(decision.tier)
        if idx >= len(_TIER_ORDER) - 1:
            return None

        reason = ""
        if not parse_ok:
            reason = "parse_failure_escalation"
        elif not citation_ok:
            reason = "citation_failure_escalation"
        elif isinstance(confidence, (int, float)) and math.isfinite(float(confidence)) and float(confidence) < self._confidence_threshold:
            reason = "low_confidence_escalation"
        elif int(conflict_count or 0) > 0 and decision.tier == "fast":
            reason = "conflict_escalation"
        if not reason:
            return None

        next_tier = _TIER_ORDER[idx + 1]
        return RouteDecision(
            stage=decision.stage,
            tier=next_tier,
            model_name=self.model_for_tier(next_tier),
            reason=reason,
            complexity_score=decision.complexity_score,
            escalated_from=decision.tier,
            shadow_only=decision.shadow_only,
            estimated_cost_saving_ratio=self._cost_saving_ratio(next_tier),
        )

    def record_decision(
        self,
        decision: RouteDecision,
        *,
        metrics: Any | None = None,
        observer: Any | None = None,
        tenant_id: str | None = None,
        outcome: str = "selected",
    ) -> None:
        if not self.enabled:
            return
        if metrics is not None and hasattr(metrics, "record_v4_model_route"):
            metrics.record_v4_model_route(
                decision.stage,
                decision.tier,
                "shadow" if decision.shadow_only and outcome == "selected" else outcome,
                escalated=decision.escalated_from is not None,
                tenant_id=tenant_id,
            )
        if observer is not None and hasattr(observer, "record_model_route"):
            observer.record_model_route(decision.to_dict())
