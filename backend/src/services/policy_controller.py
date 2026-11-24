"""Pluggable policy controller for ReAct decision routing (rule-based or learned model)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ALLOWED_POLICY_ACTIONS = frozenset({"stop", "search", "fetch", "reflect"})

POLICY_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action", "confidence", "reason"],
    "properties": {
        "action": {"type": "string", "enum": ["stop", "search", "fetch", "reflect"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "query_override": {"type": ["string", "null"]},
        "target_url": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PolicyStateFeatures:
    """Structured state features extracted at each ReAct round for policy routing."""

    task_id: int
    round_index: int
    max_rounds: int
    evidence_count: int
    distinct_domains: int
    high_quality_sources: int
    has_gap_signals: bool
    searches_left: int
    fetches_left: int
    remaining_budget_ratio: float
    task_title: str = ""
    task_intent: str = ""
    gap_signals: tuple[str, ...] = ()
    can_fetch: bool = False
    skill_pack_id: str = "_neutral"
    topic_domain: str = "general"
    freshness_ok: bool = True
    evidence_sufficiency: bool = False
    archive_repair_hits: int = 0
    recent_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gap_signals"] = list(self.gap_signals)
        payload["recent_actions"] = list(self.recent_actions)
        return payload


def format_policy_messages(
    features: PolicyStateFeatures,
    *,
    prompt_version: str = "policy_prompt_v2",
) -> list[dict[str, str]]:
    """Build deterministic training/inference chat messages shared by GRPO export & online policy."""

    system_content = (
        "You are the RL routing policy for DeepResearch. "
        "Return strict JSON with keys: action (stop|search|fetch|reflect), "
        "confidence (float 0..1), reason (string), query_override (string|null), "
        "target_url (string|null). Respect remaining searches_left/fetches_left budgets."
    )
    user_payload = {
        "prompt_version": str(prompt_version or "policy_prompt_v2"),
        "state": features.to_dict(),
    }
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def is_budget_legal_action(features: PolicyStateFeatures, decision: PolicyDecision) -> bool:
    """Verify that a policy decision respects action allowlist and remaining ReAct budgets."""

    action = str(decision.action or "").strip().lower()
    if action not in ALLOWED_POLICY_ACTIONS:
        return False
    if action == "fetch" and int(features.fetches_left) <= 0:
        return False
    if action in {"search", "reflect"} and int(features.searches_left) <= 0:
        return False
    return True


@dataclass(frozen=True)
class PolicyDecision:
    """Normalized action decision returned by a PolicyController."""

    action: str
    confidence: float
    reason: str
    source: str
    query_override: str | None = None
    target_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "source": self.source,
            "query_override": self.query_override,
            "target_url": self.target_url,
        }


class PolicyController(Protocol):
    """Protocol implemented by rule-based and learned RL policy controllers."""

    def decide(self, features: PolicyStateFeatures) -> PolicyDecision:
        """Select the next action given current task/budget features."""


class RuleBasedPolicyController:
    """Deterministic budget- and coverage-aware baseline policy controller."""

    def decide(self, features: PolicyStateFeatures) -> PolicyDecision:
        if features.round_index >= max(1, features.max_rounds):
            return PolicyDecision(
                action="stop",
                confidence=0.95,
                reason="达到任务级最大轮次上限，停止继续扩展。",
                source="rule",
            )

        if not features.has_gap_signals and features.evidence_count > 0:
            return PolicyDecision(
                action="stop",
                confidence=0.9,
                reason="当前证据覆盖充分且无明显缺口，停止追加轮次。",
                source="rule",
            )

        if features.evidence_count == 0 and features.searches_left > 0:
            return PolicyDecision(
                action="search",
                confidence=0.88,
                reason="当前任务尚未获取有效证据，需要补充定向搜索。",
                source="rule",
            )

        if (
            ("thin_content" in features.gap_signals or features.can_fetch)
            and features.fetches_left > 0
        ):
            return PolicyDecision(
                action="fetch",
                confidence=0.82,
                reason="已有来源正文偏薄且抓取预算充足，优先抓取全文。",
                source="rule",
            )

        if features.has_gap_signals and features.searches_left > 0:
            return PolicyDecision(
                action="search",
                confidence=0.78,
                reason="仍存在来源覆盖或多样性缺口，继续补充检索。",
                source="rule",
            )

        return PolicyDecision(
            action="stop",
            confidence=0.75,
            reason="可用搜索/抓取预算已接近上限，结束任务级闭环。",
            source="rule",
        )

    def decide_with_context(
        self,
        features: PolicyStateFeatures,
        *,
        request_id: str = "",
        tenant_id: str = "default",
    ) -> tuple[PolicyDecision, dict[str, Any]]:
        del request_id, tenant_id
        decision = self.decide(features)
        return decision, {
            "mode": "off",
            "executed": "rule",
            "canary_selected": False,
            "circuit_open": False,
            "shadow_agreement": None,
            "latency_ms": 0.0,
        }


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace_match:
            text = brace_match.group(0).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


class ModelPolicyController:
    """Learned policy client calling an OpenAI-compatible small model with strict rule fallback."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        prompt_version: str = "policy_prompt_v2",
        fallback: PolicyController | None = None,
        transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]] | None = None,
    ) -> None:
        self._model_id = str(model_id or "").strip()
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._api_key = str(api_key or "").strip() or None
        self._timeout_seconds = max(1.0, float(timeout_seconds or 15.0))
        self._prompt_version = str(prompt_version or "policy_prompt_v2")
        self._fallback = fallback or RuleBasedPolicyController()
        self._transport = transport or self._default_post_json

    @staticmethod
    def _default_post_json(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _fallback_decision(self, features: PolicyStateFeatures, note: str) -> PolicyDecision:
        base = self._fallback.decide(features)
        return PolicyDecision(
            action=base.action,
            confidence=base.confidence,
            reason=f"{base.reason} ({note})",
            source="rule_fallback",
            query_override=base.query_override,
            target_url=base.target_url,
        )

    def decide(self, features: PolicyStateFeatures) -> PolicyDecision:
        endpoint = (
            self._base_url
            if self._base_url.endswith("/chat/completions")
            else f"{self._base_url}/chat/completions"
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request_payload = {
            "model": self._model_id,
            "temperature": 0.0,
            "max_tokens": 200,
            "messages": format_policy_messages(
                features,
                prompt_version=self._prompt_version,
            ),
        }

        try:
            response = self._transport(
                endpoint,
                request_payload,
                headers,
                self._timeout_seconds,
            )
            choices = response.get("choices") if isinstance(response, dict) else None
            if not isinstance(choices, list) or not choices:
                return self._fallback_decision(features, "empty model choices")
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            parsed = _extract_json_object(str(message.get("content") or ""))
            if not parsed:
                return self._fallback_decision(features, "malformed policy JSON")

            action = str(parsed.get("action") or "").strip().lower()
            if action not in ALLOWED_POLICY_ACTIONS:
                return self._fallback_decision(features, f"unsupported action={action!r}")

            raw_conf = parsed.get("confidence", 0.5)
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                return self._fallback_decision(features, "invalid confidence value")
            if not (0.0 <= confidence <= 1.0):
                return self._fallback_decision(features, "confidence out of [0, 1]")

            query_override = parsed.get("query_override")
            target_url = parsed.get("target_url")
            candidate = PolicyDecision(
                action=action,
                confidence=confidence,
                reason=str(parsed.get("reason") or "model policy decision").strip() or "model policy decision",
                source="model",
                query_override=str(query_override).strip() if query_override else None,
                target_url=str(target_url).strip() if target_url else None,
            )
            if not is_budget_legal_action(features, candidate):
                return self._fallback_decision(features, f"budget-illegal action={action}")
            return candidate
        except Exception as exc:
            logger.warning("Policy model failed (%s); using rule fallback", exc)
            return self._fallback_decision(features, "policy request failed")

    def decide_with_context(
        self,
        features: PolicyStateFeatures,
        *,
        request_id: str = "",
        tenant_id: str = "default",
    ) -> tuple[PolicyDecision, dict[str, Any]]:
        del request_id, tenant_id
        started = time.perf_counter()
        decision = self.decide(features)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return decision, {
            "mode": "active",
            "executed": decision.source,
            "canary_selected": True,
            "circuit_open": False,
            "shadow_agreement": None,
            "latency_ms": elapsed_ms,
        }


class ShadowCanaryPolicyController:
    """Gradual rollout wrapper supporting shadow evaluation, deterministic canarying, and circuit breaker."""

    def __init__(
        self,
        *,
        mode: str = "shadow",
        model_controller: ModelPolicyController | None = None,
        rule_controller: RuleBasedPolicyController | None = None,
        canary_ratio: float = 0.0,
        canary_tenants: tuple[str, ...] | list[str] = (),
        circuit_breaker_window: int = 20,
        circuit_breaker_fallback_rate: float = 0.25,
        max_p95_ms: float = 250.0,
        consecutive_failure_limit: int = 3,
    ) -> None:
        self._mode = str(mode or "shadow").strip().lower()
        self._model = model_controller
        self._rule = rule_controller or RuleBasedPolicyController()
        self._canary_ratio = max(0.0, min(1.0, float(canary_ratio or 0.0)))
        self._canary_tenants = {str(t).strip() for t in canary_tenants if str(t).strip()}
        self._window_size = max(1, int(circuit_breaker_window or 20))
        self._max_fallback_rate = max(0.0, min(1.0, float(circuit_breaker_fallback_rate or 0.25)))
        self._max_p95_ms = max(1.0, float(max_p95_ms or 250.0))
        self._consecutive_failure_limit = max(1, int(consecutive_failure_limit or 3))
        self._history: deque[tuple[bool, float]] = deque(maxlen=self._window_size)
        self._consecutive_failures = 0
        self._circuit_open = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_circuit_open(self) -> bool:
        return self._circuit_open

    def reset_circuit_breaker(self) -> None:
        self._history.clear()
        self._consecutive_failures = 0
        self._circuit_open = False

    def _in_canary_bucket(self, request_id: str, tenant_id: str) -> bool:
        if tenant_id and tenant_id in self._canary_tenants:
            return True
        if self._canary_ratio <= 0.0:
            return False
        if self._canary_ratio >= 1.0:
            return True
        digest = hashlib.sha256(f"{tenant_id}:{request_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return bucket < self._canary_ratio

    def _record_outcome(self, *, is_fallback: bool, latency_ms: float) -> None:
        self._history.append((is_fallback, latency_ms))
        if is_fallback:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        if self._consecutive_failures >= self._consecutive_failure_limit:
            self._circuit_open = True
            return

        min_samples = min(self._window_size, max(4, self._window_size // 2))
        if len(self._history) >= min_samples:
            fallback_rate = sum(1 for fb, _ in self._history if fb) / len(self._history)
            if fallback_rate > self._max_fallback_rate:
                self._circuit_open = True

    def decide_with_context(
        self,
        features: PolicyStateFeatures,
        *,
        request_id: str = "",
        tenant_id: str = "default",
    ) -> tuple[PolicyDecision, dict[str, Any]]:
        rule_decision = self._rule.decide(features)
        if self._mode == "off" or self._model is None or self._circuit_open:
            return rule_decision, {
                "mode": self._mode,
                "executed": "rule",
                "canary_selected": False,
                "circuit_open": self._circuit_open,
                "shadow_agreement": None,
                "latency_ms": 0.0,
            }

        if self._mode == "canary" and not self._in_canary_bucket(request_id, tenant_id):
            return rule_decision, {
                "mode": "canary",
                "executed": "rule",
                "canary_selected": False,
                "circuit_open": self._circuit_open,
                "shadow_agreement": None,
                "latency_ms": 0.0,
            }

        started = time.perf_counter()
        model_decision = self._model.decide(features)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        is_legal = model_decision.source == "model" and is_budget_legal_action(features, model_decision)
        is_fallback = not is_legal or model_decision.source != "model"
        self._record_outcome(is_fallback=is_fallback, latency_ms=latency_ms)

        agreement = model_decision.action == rule_decision.action if model_decision.source == "model" else False
        telemetry = {
            "mode": self._mode,
            "canary_selected": self._mode in {"canary", "active"},
            "rule_action": rule_decision.action,
            "model_action": model_decision.action,
            "model_source": model_decision.source,
            "model_confidence": round(float(model_decision.confidence), 4),
            "shadow_agreement": agreement,
            "circuit_open": self._circuit_open,
            "latency_ms": latency_ms,
            "slo_ok": latency_ms <= self._max_p95_ms,
        }

        if self._mode == "shadow":
            telemetry["executed"] = "rule"
            return rule_decision, telemetry

        if is_legal:
            telemetry["executed"] = "model"
            return model_decision, telemetry

        telemetry["executed"] = "rule_fallback"
        return (
            PolicyDecision(
                action=rule_decision.action,
                confidence=rule_decision.confidence,
                reason=f"{rule_decision.reason} (canary fallback)",
                source="rule_fallback",
                query_override=rule_decision.query_override,
                target_url=rule_decision.target_url,
            ),
            telemetry,
        )

    def decide(self, features: PolicyStateFeatures) -> PolicyDecision:
        return self.decide_with_context(features)[0]


def build_policy_controller(config: Any) -> PolicyController:
    """Return RuleBasedPolicyController, ModelPolicyController, or ShadowCanaryPolicyController."""

    mode = (
        config.resolved_policy_model_mode()
        if hasattr(config, "resolved_policy_model_mode")
        else ("active" if bool(getattr(config, "policy_model_enabled", False)) else "off")
    )
    model_id = str(getattr(config, "policy_model_id", "") or "").strip()
    base_url = str(getattr(config, "policy_model_base_url", "") or "").strip()
    if mode == "off" or not model_id or not base_url:
        return RuleBasedPolicyController()

    model_ctrl = ModelPolicyController(
        model_id=model_id,
        base_url=base_url,
        api_key=getattr(config, "policy_model_api_key", None),
        timeout_seconds=float(getattr(config, "policy_model_timeout_seconds", 15.0) or 15.0),
        prompt_version=str(getattr(config, "policy_model_prompt_version", "policy_prompt_v2") or "policy_prompt_v2"),
    )
    if mode == "active":
        return model_ctrl

    return ShadowCanaryPolicyController(
        mode=mode,
        model_controller=model_ctrl,
        canary_ratio=float(getattr(config, "policy_model_canary_ratio", 0.0) or 0.0),
        canary_tenants=tuple(getattr(config, "policy_model_canary_tenants", ()) or ()),
        circuit_breaker_window=int(getattr(config, "policy_model_circuit_breaker_window", 20) or 20),
        circuit_breaker_fallback_rate=float(
            getattr(config, "policy_model_circuit_breaker_fallback_rate", 0.25) or 0.25
        ),
        max_p95_ms=float(getattr(config, "policy_model_max_p95_ms", 250.0) or 250.0),
    )
