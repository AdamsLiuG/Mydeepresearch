"""Pluggable policy controller for ReAct decision routing (rule-based or learned model)."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ALLOWED_POLICY_ACTIONS = frozenset({"stop", "search", "fetch", "reflect"})


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gap_signals"] = list(self.gap_signals)
        return payload


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
        fallback: PolicyController | None = None,
        transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]] | None = None,
    ) -> None:
        self._model_id = str(model_id or "").strip()
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._api_key = str(api_key or "").strip() or None
        self._timeout_seconds = max(1.0, float(timeout_seconds or 15.0))
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
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the RL routing policy for DeepResearch. "
                        "Return strict JSON with keys: action (stop|search|fetch|reflect), "
                        "confidence (float 0..1), reason (string), query_override (string|null), "
                        "target_url (string|null)."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(features.to_dict(), ensure_ascii=False),
                },
            ],
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
            return PolicyDecision(
                action=action,
                confidence=confidence,
                reason=str(parsed.get("reason") or "model policy decision").strip() or "model policy decision",
                source="model",
                query_override=str(query_override).strip() if query_override else None,
                target_url=str(target_url).strip() if target_url else None,
            )
        except Exception as exc:
            logger.warning("Policy model failed (%s); using rule fallback", exc)
            return self._fallback_decision(features, "policy request failed")


def build_policy_controller(config: Any) -> PolicyController:
    """Return ModelPolicyController when configured and enabled, else RuleBasedPolicyController."""

    enabled = bool(getattr(config, "policy_model_enabled", False))
    model_id = str(getattr(config, "policy_model_id", "") or "").strip()
    base_url = str(getattr(config, "policy_model_base_url", "") or "").strip()
    if enabled and model_id and base_url:
        return ModelPolicyController(
            model_id=model_id,
            base_url=base_url,
            api_key=getattr(config, "policy_model_api_key", None),
            timeout_seconds=float(getattr(config, "policy_model_timeout_seconds", 15.0) or 15.0),
        )
    return RuleBasedPolicyController()
