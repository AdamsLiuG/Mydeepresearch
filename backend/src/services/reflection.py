"""Request-level reflection helpers for coverage assessment and lightweight replanning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import SummaryState
from prompts import get_current_date, request_reflection_instructions
from services.text_processing import normalize_agent_markdown
from utils import strip_thinking_tokens, truncate_text


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
        observer=None,
    ) -> ReflectionAssessment:
        """Judge whether the current request needs more research coverage."""
        prompt = self._build_prompt(state, gap_signals=gap_signals)
        try:
            response = self._agent.run(prompt)
        except Exception as exc:
            if observer:
                observer.record_llm_call(
                    success=False,
                    prompt_text=prompt,
                    completion_text="",
                    error=exc,
                )
            raise
        finally:
            self._agent.clear_history()

        if observer:
            observer.record_llm_call(
                success=True,
                prompt_text=prompt,
                completion_text=response,
            )

        return self._parse_response(response, gap_signals=gap_signals)

    def _build_prompt(self, state: SummaryState, *, gap_signals: list[str]) -> str:
        task_blocks: list[str] = []
        for task in state.todo_items:
            summary = truncate_text(task.summary or "暂无可用信息", 500)
            sources = truncate_text(task.sources_summary or "暂无来源", 300)
            task_blocks.append(
                f"- 任务 {task.id} ({task.origin}, round={task.round})\n"
                f"  标题：{task.title}\n"
                f"  目标：{task.intent}\n"
                f"  状态：{task.status}\n"
                f"  总结：{summary}\n"
                f"  来源：{sources}\n"
            )

        gap_text = "\n".join(f"- {signal}" for signal in gap_signals) or "- 无显式缺口信号"
        tasks_text = "\n".join(task_blocks) or "- 暂无任务结果"
        return request_reflection_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
            gap_signals=gap_text,
            task_results=tasks_text,
        )

    def _parse_response(
        self,
        raw_response: str,
        *,
        gap_signals: list[str],
    ) -> ReflectionAssessment:
        text = (raw_response or "").strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)
        text = normalize_agent_markdown(text)

        payload = self._extract_json_payload(text)
        if not isinstance(payload, dict):
            return ReflectionAssessment(
                coverage_status="sufficient",
                reason="反思输出不可解析，跳过补充研究。",
                gap_signals=list(gap_signals),
                missing_angles=[],
            )

        coverage_status = str(payload.get("coverage_status") or "sufficient").strip().lower()
        if coverage_status not in {"sufficient", "needs_more_research"}:
            coverage_status = "sufficient"

        reason = str(payload.get("reason") or "").strip()
        normalized_gap_signals = self._normalize_text_list(payload.get("gap_signals")) or list(gap_signals)
        missing_angles = self._normalize_text_list(payload.get("missing_angles"))

        if coverage_status == "needs_more_research" and not missing_angles:
            coverage_status = "sufficient"
            if not reason:
                reason = "未识别到明确缺失维度，直接生成报告。"

        if not reason:
            reason = (
                "发现首轮研究仍有明显缺口，补充执行新增任务。"
                if coverage_status == "needs_more_research"
                else "首轮研究覆盖充分，直接生成报告。"
            )

        return ReflectionAssessment(
            coverage_status=coverage_status,
            reason=reason,
            gap_signals=normalized_gap_signals,
            missing_angles=missing_angles,
        )

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        index = 0
        while index < len(text):
            if text[index] != "{":
                index += 1
                continue
            try:
                payload, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(payload, dict):
                return payload
            index += max(end, 1)
        return None

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
