"""Request-level reflection helpers for coverage assessment and lightweight replanning."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import extract_llm_usage
from models import SummaryState, TodoItem
from prompts import (
    get_current_date,
    reflection_output_contract,
    request_reflection_instructions,
)
from services.context_builder import ContextPrefixBuilder, PromptEnvelope
from services.fact_ledger import (
    context_covered_task_ids,
    render_context_checkpoint,
    render_context_delta,
)
from utils import strip_thinking_tokens, truncate_text

logger = logging.getLogger(__name__)

_STRICT_JSON_INVALID_REASON = "reflection 输出不符合严格 JSON，已跳过补充研究。"
_STRICT_JSON_DEGRADED_REASON = "reflection_invalid_output"
_MAX_MISSING_ANGLES = 3
_STRICT_JSON_RESPONSE_FORMAT = {"type": "json_object"}
_REFLECTION_SUMMARY_CHAR_LIMIT = 260
_REFLECTION_SOURCES_CHAR_LIMIT = 160


@dataclass
class ReflectionAssessment:
    """Structured result returned by the reflection stage."""

    coverage_status: str = "sufficient"
    reason: str = "首轮研究覆盖充分，直接生成报告。"
    gap_signals: list[str] = field(default_factory=list)
    missing_angles: list[str] = field(default_factory=list)

    @property
    def needs_more_research(self) -> bool:
        return self.coverage_status == "needs_more_research"


class ReflectionService:
    """Assess whether a completed request needs supplemental research."""

    def __init__(self, reflection_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = reflection_agent
        self._config = config

    def assess_request(
        self,
        state: SummaryState,
        *,
        gap_signals: list[str],
        strategy_memory_context: str | None = None,
        skill_pack_context: str | None = None,
        observer=None,
    ) -> ReflectionAssessment:
        """Judge whether the current request needs more research coverage."""
        context = self._build_reflection_context(state, gap_signals=gap_signals)
        envelope = self._build_envelope(
            state,
            context,
            strategy_memory_context=strategy_memory_context,
            skill_pack_context=skill_pack_context,
        )
        prompt = envelope.to_flat_prompt()
        existing_task_titles = {
            str(task.get("title") or "").strip().casefold()
            for task in context.get("task_snapshots", [])
            if str(task.get("title") or "").strip()
        }

        try:
            response = self._run_agent_with_strict_json(prompt, envelope=envelope)
        except Exception as exc:
            if observer:
                observer.record_llm_call(
                    success=False,
                    prompt_text=prompt,
                    completion_text="",
                    error=exc,
                    stage="reflection",
                )
            raise
        finally:
            self._agent.clear_history()

        if observer:
            observer.record_llm_call(
                success=True,
                prompt_text=prompt,
                completion_text=response,
                usage=extract_llm_usage(self._agent),
                stage="reflection",
            )

        return self._parse_response(
            response,
            gap_signals=gap_signals,
            observer=observer,
            existing_task_titles=existing_task_titles,
        )

    def _run_agent_with_strict_json(
        self,
        prompt: str,
        *,
        envelope: PromptEnvelope | None = None,
    ) -> str:
        """Prefer provider-level JSON mode and fall back when unsupported."""
        run_env = getattr(self._agent, "run_envelope", None)
        use_envelope = envelope is not None and callable(run_env)
        try:
            if use_envelope:
                return run_env(envelope, config=self._config, response_format=_STRICT_JSON_RESPONSE_FORMAT)
            return self._agent.run(prompt, response_format=_STRICT_JSON_RESPONSE_FORMAT)
        except Exception as exc:
            if not self._response_format_is_unsupported(exc):
                raise

            logger.info("Reflection JSON mode unsupported; retrying without response_format: %s", exc)
            if use_envelope:
                return run_env(envelope, config=self._config)
            return self._agent.run(prompt)

    def _build_task_snapshot(self, task: TodoItem) -> dict[str, Any]:
        """Return a compact task snapshot used by reflection prompting."""
        summary_text = (task.summary or "").strip()
        sources_text = (task.sources_summary or "").strip()
        evidence_items = list(task.evidence_items or [])
        unique_domains = sorted(
            {
                str(item.get("domain") or "").strip().lower()
                for item in evidence_items
                if str(item.get("domain") or "").strip()
            }
        )

        summary_missing = not summary_text or summary_text == "暂无可用信息"
        sources_missing = not sources_text

        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "status": task.status,
            "origin": task.origin,
            "round": task.round,
            "summary_excerpt": truncate_text(summary_text or "暂无可用信息", _REFLECTION_SUMMARY_CHAR_LIMIT),
            "sources_excerpt": truncate_text(sources_text or "暂无来源", _REFLECTION_SOURCES_CHAR_LIMIT),
            "evidence_count": len(evidence_items),
            "unique_domain_count": len(unique_domains),
            "notice_count": len(task.notices or []),
            "summary_missing": summary_missing,
            "sources_missing": sources_missing,
        }

    def _build_reflection_context(
        self,
        state: SummaryState,
        *,
        gap_signals: list[str],
    ) -> dict[str, Any]:
        """Build a structured reflection context that can be rendered verbatim."""
        task_snapshots = [self._build_task_snapshot(task) for task in state.todo_items]
        task_counts = {
            "total": len(task_snapshots),
            "completed": sum(1 for task in task_snapshots if task.get("status") == "completed"),
            "failed": sum(1 for task in task_snapshots if task.get("status") == "failed"),
            "skipped": sum(1 for task in task_snapshots if task.get("status") == "skipped"),
            "pending": sum(1 for task in task_snapshots if task.get("status") not in {"completed", "failed", "skipped"}),
        }
        return {
            "current_date": get_current_date(),
            "research_topic": state.research_topic,
            "research_loop_count": int(state.research_loop_count or 0),
            "gap_signals": list(gap_signals),
            "task_counts": task_counts,
            "task_snapshots": task_snapshots,
            "hard_constraints": {
                "allow_tools": False,
                "must_output_single_json_object": True,
                "must_not_output_markdown_fences": True,
                "must_not_repeat_existing_task_titles": True,
                "max_missing_angles": _MAX_MISSING_ANGLES,
            },
        }

    def _build_envelope(
        self,
        state: SummaryState | None,
        context: dict[str, Any],
        *,
        strategy_memory_context: str | None = None,
        skill_pack_context: str | None = None,
    ) -> PromptEnvelope:
        task_counts = context.get("task_counts", {})
        task_count_summary = (
            f"共 {task_counts.get('total', 0)} 个任务，"
            f"已完成 {task_counts.get('completed', 0)} 个，"
            f"失败 {task_counts.get('failed', 0)} 个，"
            f"跳过 {task_counts.get('skipped', 0)} 个，"
            f"待处理 {task_counts.get('pending', 0)} 个，"
            f"研究轮次计数 {context.get('research_loop_count', 0)}。"
        )
        gap_signal_lines = "\n".join(f"- {signal}" for signal in context.get("gap_signals", [])) or "- 无显式缺口信号"

        builder = ContextPrefixBuilder(self._config)
        compact = builder.enabled() and getattr(self._config, "context_compaction_enabled", True)
        task_overview_lines: list[str] = []
        for task in context.get("task_snapshots", []):
            task_overview_lines.append(
                f"- 任务 {task['id']} | {task['title']} | 状态={task['status']} | "
                f"来源={task['evidence_count']} / 域名={task['unique_domain_count']} / notice={task['notice_count']} | "
                f"摘要缺失={'是' if task['summary_missing'] else '否'} | 来源缺失={'是' if task['sources_missing'] else '否'}\n"
                + (f"  摘要：{task['summary_excerpt']}\n" if not compact else "")
            )
        task_overview = "\n".join(task_overview_lines) or "- 暂无任务结果"
        covered_task_ids = context_covered_task_ids(state, self._config) if state is not None else set()
        compact_context = {
            **context,
            "task_snapshots": [
                {key: value for key, value in task.items() if key not in {"summary_excerpt", "sources_excerpt"}}
                if task["id"] in covered_task_ids else task
                for task in context.get("task_snapshots", [])
            ],
        }
        compact_context_json = json.dumps(compact_context, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        volatile_body = request_reflection_instructions.format(
            current_date=context.get("current_date"),
            research_topic=context.get("research_topic"),
            task_count_summary=task_count_summary,
            gap_signals=gap_signal_lines,
            task_overview=task_overview,
            reflection_context_json=compact_context_json,
        )

        strategy_block = self._strategy_memory_block(strategy_memory_context)
        frozen = render_context_checkpoint(state, self._config) if state is not None else ""
        delta = render_context_delta(state, self._config) if state is not None else ""
        role = str(getattr(self._agent, "system_prompt", "") or "")
        if not builder.enabled():
            return builder.build_envelope(
                stage="reflection",
                system_role=role,
                volatile_blocks=[
                    ("reflection_delta", volatile_body),
                    ("output_contract", reflection_output_contract),
                    ("checkpoint", frozen),
                    ("context_delta", delta),
                    ("strategy_memory", strategy_block),
                    ("skills", str(skill_pack_context or "")),
                ],
            )
        return builder.build_envelope(
            stage="reflection",
            system_role=role,
            static_schema_and_rules=reflection_output_contract,
            research_topic=str(context.get("research_topic") or ""),
            skill_pack_context="\n\n".join(
                part.strip() for part in (str(skill_pack_context or ""), strategy_block) if part.strip()
            ),
            frozen_index_and_ledger=frozen,
            volatile_blocks=[("reflection_delta", volatile_body), ("context_delta", delta)],
        )

    def _build_prompt(
        self,
        context: dict[str, Any],
        *,
        strategy_memory_context: str | None = None,
        skill_pack_context: str | None = None,
    ) -> str:
        """Render the reflection prompt from a structured context object."""
        return self._build_envelope(
            None,
            context,
            strategy_memory_context=strategy_memory_context,
            skill_pack_context=skill_pack_context,
        ).to_flat_prompt()

    @staticmethod
    def _strategy_memory_block(strategy_memory_context: str | None) -> str:
        context = str(strategy_memory_context or "").strip()
        if not context:
            return ""
        return (
            "\n\n<STRATEGY_MEMORY>\n"
            "以下内容来自历史请求提炼出的策略记忆，只能用于帮助你识别覆盖缺口、修补思路、查询模式与风险信号。\n"
            "这些内容不是当前主题事实，不是当前证据，也不能直接复用历史报告结论。\n"
            f"{context}\n"
            "</STRATEGY_MEMORY>\n"
        )

    def _parse_response(
        self,
        raw_response: str,
        *,
        gap_signals: list[str],
        observer=None,
        existing_task_titles: set[str] | None = None,
    ) -> ReflectionAssessment:
        """Parse a strict JSON reflection response and reject any wrapped output."""
        text = (raw_response or "").strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text).strip()

        if not text:
            return self._invalid_assessment(gap_signals, observer, raw_response)
        if "```" in text or "[TOOL_CALL:" in text:
            return self._invalid_assessment(gap_signals, observer, raw_response)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return self._invalid_assessment(gap_signals, observer, raw_response)

        if not isinstance(payload, dict):
            return self._invalid_assessment(gap_signals, observer, raw_response)

        required_fields = {"coverage_status", "reason", "gap_signals", "missing_angles"}
        if not required_fields.issubset(payload):
            return self._invalid_assessment(gap_signals, observer, raw_response)

        if not isinstance(payload.get("gap_signals"), list) or not isinstance(payload.get("missing_angles"), list):
            return self._invalid_assessment(gap_signals, observer, raw_response)

        coverage_status = str(payload.get("coverage_status") or "").strip().lower()
        if coverage_status not in {"sufficient", "needs_more_research"}:
            return self._invalid_assessment(gap_signals, observer, raw_response)

        reason = str(payload.get("reason") or "").strip()
        if not reason:
            return self._invalid_assessment(gap_signals, observer, raw_response)

        normalized_gap_signals = self._normalize_text_list(payload.get("gap_signals")) or list(gap_signals)
        missing_angles = self._normalize_text_list(payload.get("missing_angles"))

        if len(missing_angles) > _MAX_MISSING_ANGLES:
            return self._invalid_assessment(gap_signals, observer, raw_response)
        if coverage_status == "needs_more_research" and not missing_angles:
            return self._invalid_assessment(gap_signals, observer, raw_response)

        if existing_task_titles and any(angle.casefold() in existing_task_titles for angle in missing_angles):
            return self._invalid_assessment(gap_signals, observer, raw_response)

        if coverage_status == "sufficient":
            missing_angles = []

        return ReflectionAssessment(
            coverage_status=coverage_status,
            reason=reason,
            gap_signals=normalized_gap_signals,
            missing_angles=missing_angles,
        )

    def _invalid_assessment(
        self,
        gap_signals: list[str],
        observer,
        raw_response: str,
    ) -> ReflectionAssessment:
        """Return the conservative fallback used for any strict JSON violation."""
        if observer:
            observer.record_degraded(_STRICT_JSON_DEGRADED_REASON)

        logger.warning("Reflection returned invalid strict JSON output: %r", raw_response)
        return ReflectionAssessment(
            coverage_status="sufficient",
            reason=_STRICT_JSON_INVALID_REASON,
            gap_signals=list(gap_signals),
            missing_angles=[],
        )

    @staticmethod
    def _response_format_is_unsupported(exc: Exception) -> bool:
        """Return whether the provider rejected the JSON-mode request contract."""
        message = str(exc or "").casefold()
        if "response_format" not in message and "json_object" not in message and "json schema" not in message:
            return isinstance(exc, TypeError) and "response_format" in message

        unsupported_markers = (
            "unsupported",
            "not support",
            "not supported",
            "invalid",
            "unknown",
            "unexpected",
            "extra inputs are not permitted",
            "extra fields not permitted",
        )
        return any(marker in message for marker in unsupported_markers)

    def _normalize_text_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized
