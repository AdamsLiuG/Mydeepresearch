"""LLM-based synthesis of reusable strategy cards from completed request snapshots."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from utils import strip_thinking_tokens, truncate_text

logger = logging.getLogger(__name__)

_ALLOWED_KINDS = {"planning_pattern", "reflection_pattern", "anti_pattern"}
_KIND_TO_STAGE = {
    "planning_pattern": "planning",
    "reflection_pattern": "reflection",
    # Anti-patterns are queried in both stages; keep a stable canonical stage
    # while allowing the memory service to reuse them more broadly.
    "anti_pattern": "reflection",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any, *, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _normalize_list(value: Any, *, max_items: int) -> list[str]:
    if isinstance(value, list):
        candidates = value
    elif value in (None, ""):
        candidates = []
    else:
        candidates = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = " ".join(str(item or "").strip().split())
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


@dataclass
class StrategySourceRequest:
    request_id: str
    topic: str
    status: str
    review_status: str
    report_available: bool
    completed_task_count: int
    failed_task_count: int
    repair_cycles: int
    reflection_gap_signals: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    request_metrics: dict[str, Any] = field(default_factory=dict)
    requested_kinds: list[str] = field(default_factory=list)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "topic": self.topic,
            "status": self.status,
            "review_status": self.review_status,
            "report_available": self.report_available,
            "completed_task_count": self.completed_task_count,
            "failed_task_count": self.failed_task_count,
            "repair_cycles": self.repair_cycles,
            "reflection_gap_signals": list(self.reflection_gap_signals),
            "degraded_reasons": list(self.degraded_reasons),
            "requested_kinds": list(self.requested_kinds),
            "tasks": list(self.tasks),
            "request_metrics": dict(self.request_metrics),
        }


@dataclass
class StrategyCard:
    strategy_id: str
    strategy_kind: str
    stage_scope: str
    title: str
    applicable_when: str
    match_signals: list[str]
    recommended_actions: list[str]
    query_templates: list[str]
    preferred_sources: list[str]
    pitfalls_to_avoid: list[str]
    origin_request_id: str
    origin_status: str
    origin_review_status: str
    origin_task_ids: list[int]
    created_at: str

    def to_document(self) -> str:
        sections = [
            f"kind: {self.strategy_kind}",
            self.title,
            self.applicable_when,
            "match_signals: " + "; ".join(self.match_signals),
            "recommended_actions: " + "; ".join(self.recommended_actions),
            "query_templates: " + "; ".join(self.query_templates),
            "preferred_sources: " + "; ".join(self.preferred_sources),
            "pitfalls_to_avoid: " + "; ".join(self.pitfalls_to_avoid),
        ]
        return "\n".join(section for section in sections if str(section or "").strip())

    def to_metadata(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_kind": self.strategy_kind,
            "stage_scope": self.stage_scope,
            "title": self.title,
            "applicable_when": self.applicable_when,
            "match_signals": json.dumps(self.match_signals, ensure_ascii=False),
            "recommended_actions": json.dumps(self.recommended_actions, ensure_ascii=False),
            "query_templates": json.dumps(self.query_templates, ensure_ascii=False),
            "preferred_sources": json.dumps(self.preferred_sources, ensure_ascii=False),
            "pitfalls_to_avoid": json.dumps(self.pitfalls_to_avoid, ensure_ascii=False),
            "origin_request_id": self.origin_request_id,
            "origin_status": self.origin_status,
            "origin_review_status": self.origin_review_status,
            "origin_task_ids": json.dumps(self.origin_task_ids, ensure_ascii=False),
            "created_at": self.created_at,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_kind": self.strategy_kind,
            "stage_scope": self.stage_scope,
            "title": self.title,
            "applicable_when": self.applicable_when,
            "match_signals": list(self.match_signals),
            "recommended_actions": list(self.recommended_actions),
            "query_templates": list(self.query_templates),
            "preferred_sources": list(self.preferred_sources),
            "pitfalls_to_avoid": list(self.pitfalls_to_avoid),
            "origin_request_id": self.origin_request_id,
            "origin_status": self.origin_status,
            "origin_review_status": self.origin_review_status,
            "origin_task_ids": list(self.origin_task_ids),
            "created_at": self.created_at,
        }


class StrategySynthesizer:
    """Use the content-only LLM to distill completed requests into strategy cards."""

    def __init__(
        self,
        agent_factory: Callable[[], ToolAwareSimpleAgent],
        config: Configuration,
    ) -> None:
        self._agent_factory = agent_factory
        self._config = config

    def synthesize(self, source_request: StrategySourceRequest) -> list[StrategyCard]:
        if not source_request.requested_kinds:
            return []

        prompt = self._build_prompt(source_request)
        agent = self._agent_factory()
        try:
            response = agent.run(prompt)
        finally:
            agent.clear_history()
        return self._parse_response(response, source_request)

    def _build_prompt(self, source_request: StrategySourceRequest) -> str:
        payload_json = json.dumps(
            source_request.to_prompt_payload(),
            ensure_ascii=False,
            indent=2,
        )
        requested_kinds = ", ".join(source_request.requested_kinds)
        return (
            "你是一名 Strategy Memory 提炼器，只负责把历史请求压缩成可复用的方法经验卡片。\n\n"
            "<GOAL>\n"
            f"- 仅为这些 strategy_kind 生成结果：{requested_kinds}\n"
            "- 只提炼任务拆解经验、覆盖修补经验或失败反模式，不要提炼主题事实结论；\n"
            "- 禁止输出 source_id、禁止复述报告正文、禁止把单次主题事实包装成经验；\n"
            "- query_templates 必须是可迁移的检索模式，而不是某个主题的具体答案；\n"
            "- anti_pattern 重点描述在哪些信号下容易出错、应避免什么；\n"
            "- planning_pattern / reflection_pattern 重点描述何时适用、推荐动作、推荐来源偏好。\n"
            "</GOAL>\n\n"
            "<OUTPUT_RULES>\n"
            "- 只能输出一个 JSON array；\n"
            "- 不要输出 Markdown 代码块；\n"
            "- 每个元素必须包含以下字段：\n"
            '  "strategy_kind", "title", "applicable_when", "match_signals", '
            '"recommended_actions", "query_templates", "preferred_sources", "pitfalls_to_avoid"\n'
            "- 不要输出其他解释文字。\n"
            "</OUTPUT_RULES>\n\n"
            "<JSON_CONTEXT>\n"
            f"{payload_json}\n"
            "</JSON_CONTEXT>\n"
        )

    def _parse_response(
        self,
        raw_response: str,
        source_request: StrategySourceRequest,
    ) -> list[StrategyCard]:
        text = str(raw_response or "").strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text).strip()
        if not text or "```" in text:
            return []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "strategy synthesizer returned invalid JSON request_id=%s response=%s",
                source_request.request_id,
                truncate_text(text, 400),
            )
            return []

        if not isinstance(payload, list):
            logger.warning(
                "strategy synthesizer expected JSON array request_id=%s payload_type=%s",
                source_request.request_id,
                type(payload).__name__,
            )
            return []

        cards: list[StrategyCard] = []
        seen_ids: set[str] = set()
        task_ids = sorted(
            {
                int(task.get("id"))
                for task in source_request.tasks
                if str(task.get("id") or "").strip().isdigit()
            }
        )
        created_at = _utc_now()
        for item in payload:
            card = self._normalize_card(
                item,
                source_request=source_request,
                origin_task_ids=task_ids,
                created_at=created_at,
            )
            if card is None or card.strategy_id in seen_ids:
                continue
            seen_ids.add(card.strategy_id)
            cards.append(card)
        return cards

    def _normalize_card(
        self,
        payload: Any,
        *,
        source_request: StrategySourceRequest,
        origin_task_ids: list[int],
        created_at: str,
    ) -> StrategyCard | None:
        if not isinstance(payload, dict):
            return None

        strategy_kind = _normalize_text(payload.get("strategy_kind"))
        if strategy_kind not in _ALLOWED_KINDS:
            return None
        if strategy_kind not in source_request.requested_kinds:
            return None

        title = truncate_text(_normalize_text(payload.get("title")), 120)
        applicable_when = truncate_text(_normalize_text(payload.get("applicable_when")), 320)
        if not title or not applicable_when:
            return None

        match_signals = _normalize_list(payload.get("match_signals"), max_items=4)
        recommended_actions = _normalize_list(payload.get("recommended_actions"), max_items=4)
        query_templates = _normalize_list(payload.get("query_templates"), max_items=3)
        preferred_sources = _normalize_list(payload.get("preferred_sources"), max_items=3)
        pitfalls_to_avoid = _normalize_list(payload.get("pitfalls_to_avoid"), max_items=3)

        if strategy_kind == "anti_pattern":
            if not match_signals and not pitfalls_to_avoid:
                return None
        elif not recommended_actions:
            return None

        strategy_id = f"{source_request.request_id}::{strategy_kind}"
        return StrategyCard(
            strategy_id=strategy_id,
            strategy_kind=strategy_kind,
            stage_scope=_KIND_TO_STAGE[strategy_kind],
            title=title,
            applicable_when=applicable_when,
            match_signals=match_signals,
            recommended_actions=recommended_actions,
            query_templates=query_templates,
            preferred_sources=preferred_sources,
            pitfalls_to_avoid=pitfalls_to_avoid,
            origin_request_id=source_request.request_id,
            origin_status=source_request.status,
            origin_review_status=source_request.review_status,
            origin_task_ids=origin_task_ids,
            created_at=created_at,
        )
