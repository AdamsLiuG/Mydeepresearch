"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from metrics import RequestTrace
from models import SummaryState, TodoItem
from prompts import get_current_date, todo_planner_instructions
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)

NUMBERED_TASK_PATTERN = re.compile(
    r"^(?:[-*]\s*)?(?P<index>\d+)[\.\)、]\s*(?P<title>[^：:]{1,40})(?:[：:]\s*(?P<intent>.+))?$"
)
MARKDOWN_TABLE_TASK_PATTERN = re.compile(
    r"^\|\s*(?P<index>\d+)\s*\|\s*(?P<title>[^|]+?)\s*\|\s*(?P<intent>[^|]+?)\s*\|(?:\s*[^|]+?\s*\|)?$"
)
META_WORKFLOW_KEYWORDS = (
    "启动检索",
    "进度同步",
    "交叉验证",
    "综合报告",
    "并行执行",
    "依赖关系",
    "信息整合",
    "标签追踪",
    "任务分配",
    "执行顺序",
    "流程编排",
    "结果输出",
    "笔记同步",
)
META_WORKFLOW_HINTS = (
    "agent",
    "query字段",
    "note",
    "笔记",
    "工具调用",
    "工作流",
    "流程",
    "同步任务",
)

class PlanningService:
    """Wraps the planner agent to produce structured TODO items."""

    def __init__(self, planner_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._config = config

    def plan_todo_list(
        self,
        state: SummaryState,
        observer: RequestTrace | None = None,
    ) -> List[TodoItem]:
        """Ask the planner agent to break the topic into actionable tasks."""
        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )
        response = self._invoke_planner(prompt, observer=observer)
        logger.info("Planner raw output (truncated): %s", response[:500])

        raw_tasks = self._extract_tasks(response)
        tasks_payload, rejected_count = self._sanitize_tasks(raw_tasks, research_topic=state.research_topic)
        if self._should_repair_plan(tasks_payload, rejected_count, response):
            logger.warning(
                "Planner produced workflow-like tasks; requesting repaired task plan topic=%s rejected=%s",
                state.research_topic,
                rejected_count,
            )
            repair_prompt = self._build_repair_prompt(state.research_topic, response)
            repaired_response = self._invoke_planner(repair_prompt, observer=observer)
            logger.info("Planner repaired output (truncated): %s", repaired_response[:500])
            repaired_tasks, _ = self._sanitize_tasks(
                self._extract_tasks(repaired_response),
                research_topic=state.research_topic,
            )
            if repaired_tasks:
                tasks_payload = repaired_tasks

        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            query = str(item.get("query") or "").strip()

            if not query:
                query = self._default_query_for_task(
                    research_topic=state.research_topic,
                    title=title,
                    intent=intent,
                )

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    def _default_query_for_task(
        self,
        *,
        research_topic: str,
        title: str,
        intent: str,
    ) -> str:
        """Build a differentiated fallback query when the planner omits one."""

        topic = (research_topic or "").strip()
        task_title = self._normalize_task_title(title)
        task_intent = (intent or "").strip()

        if topic and task_title:
            return f"{topic} {task_title}".strip()
        if topic:
            return topic
        if task_title and task_intent:
            return f"{task_title} {task_intent}".strip()
        return task_title or task_intent

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning failed."""
        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=f"{state.research_topic} 最新进展" if state.research_topic else "基础背景梳理",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""
        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        for json_payload in self._iter_json_payloads(text):
            tasks = self._tasks_from_payload(json_payload)
            if tasks:
                return tasks

        tool_payload_tasks = self._extract_tasks_from_tool_payloads(text)
        if tool_payload_tasks:
            logger.warning(
                "Planner response did not contain standalone task JSON; recovered %d tasks from tool payloads",
                len(tool_payload_tasks),
            )
            return tool_payload_tasks

        numbered_tasks = self._extract_tasks_from_numbered_text(text)
        if numbered_tasks:
            logger.warning(
                "Planner response did not contain parseable task JSON; recovered %d tasks from numbered text",
                len(numbered_tasks),
            )
            return numbered_tasks

        markdown_table_tasks = self._extract_tasks_from_markdown_table(text)
        if markdown_table_tasks:
            logger.warning(
                "Planner response did not contain parseable task JSON; recovered %d tasks from markdown table",
                len(markdown_table_tasks),
            )
            return markdown_table_tasks

        return []

    def _extract_json_payload(self, text: str) -> dict[str, Any] | list | None:
        """Try to locate and parse a JSON object or array from the text."""
        for payload in self._iter_json_payloads(text):
            return payload
        return None

    def _extract_tool_payload(self, text: str) -> dict[str, Any] | None:
        """Parse the first TOOL_CALL expression in the output."""
        payloads = self._extract_tool_payloads(text)
        return payloads[0] if payloads else None

    def _iter_json_payloads(self, text: str):
        """Yield JSON payloads embedded anywhere in the planner response."""

        decoder = json.JSONDecoder()
        index = 0
        length = len(text)

        while index < length:
            if text[index] not in "{[":
                index += 1
                continue

            try:
                payload, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                index += 1
                continue

            yield payload
            index += max(end, 1)

    def _tasks_from_payload(self, payload: dict[str, Any] | list | None) -> List[dict[str, Any]]:
        """Normalize a JSON payload into planner task dictionaries."""

        tasks: List[dict[str, Any]] = []
        if isinstance(payload, dict):
            candidate = payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    tasks.append(item)
        return tasks

    def _extract_tool_payloads(self, text: str) -> List[dict[str, Any]]:
        """Parse all TOOL_CALL payloads embedded in the response."""

        payloads: List[dict[str, Any]] = []
        for body in self._iter_tool_call_bodies(text):
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = self._parse_key_value_payload(body)

            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def _iter_tool_call_bodies(self, text: str):
        """Yield TOOL_CALL bodies, tolerating nested JSON arrays and objects."""

        marker = "[TOOL_CALL:"
        index = 0
        length = len(text)

        while index < length:
            start = text.find(marker, index)
            if start == -1:
                return

            cursor = start + len(marker)
            separator = text.find(":", cursor)
            if separator == -1:
                return

            cursor = separator + 1
            body_start = cursor
            brace_depth = 0
            bracket_depth = 1
            in_string = False
            escaped = False

            while cursor < length:
                char = text[cursor]

                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    cursor += 1
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth = max(brace_depth - 1, 0)
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                    if bracket_depth == 0 and brace_depth == 0:
                        yield text[body_start:cursor].strip()
                        cursor += 1
                        break

                cursor += 1

            index = cursor

    def _parse_key_value_payload(self, body: str) -> dict[str, Any]:
        """Fallback parser for simple key=value tool payloads."""

        parts = [segment.strip() for segment in body.split(",") if segment.strip()]
        payload: dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip().strip('"').strip("'")
        return payload

    def _extract_tasks_from_tool_payloads(self, text: str) -> List[dict[str, Any]]:
        """Recover planner tasks from emitted TOOL_CALL payloads."""

        payloads = self._extract_tool_payloads(text)
        for payload in payloads:
            tasks = self._tasks_from_payload(payload)
            if tasks:
                return tasks

        recovered: dict[int, dict[str, Any]] = {}
        for payload in payloads:
            task_id = payload.get("task_id")
            if task_id is None:
                continue

            try:
                normalized_id = int(task_id)
            except (TypeError, ValueError):
                continue

            title = self._normalize_task_title(str(payload.get("title") or f"任务{normalized_id}"))
            content = str(payload.get("content") or "").strip()
            intent = self._extract_intent_from_text(content) or "聚焦主题的关键问题"
            query = str(
                payload.get("query")
                or payload.get("search_query")
                or payload.get("search")
                or ""
            ).strip()

            recovered[normalized_id] = {
                "title": title or f"任务{normalized_id}",
                "intent": intent,
                "query": query,
            }

        return [recovered[idx] for idx in sorted(recovered)]

    def _extract_tasks_from_numbered_text(self, text: str) -> List[dict[str, Any]]:
        """Recover tasks from a numbered natural-language list."""

        tasks: List[dict[str, Any]] = []
        for line in text.splitlines():
            match = NUMBERED_TASK_PATTERN.match(line.strip())
            if not match:
                continue

            title = self._normalize_task_title(match.group("title"))
            intent = (match.group("intent") or "").strip() or "聚焦主题的关键问题"
            if not title:
                continue

            tasks.append(
                {
                    "title": title,
                    "intent": intent,
                    "query": "",
                }
            )

        return tasks if len(tasks) >= 2 else []

    def _extract_tasks_from_markdown_table(self, text: str) -> List[dict[str, Any]]:
        """Recover tasks from a markdown table with title and intent columns."""

        tasks: List[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            if set(stripped.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue

            match = MARKDOWN_TABLE_TASK_PATTERN.match(stripped)
            if not match:
                continue

            title = self._normalize_task_title(match.group("title"))
            intent = match.group("intent").strip()
            if not title:
                continue

            tasks.append(
                {
                    "title": title,
                    "intent": intent or "聚焦主题的关键问题",
                    "query": "",
                }
            )

        return tasks if len(tasks) >= 2 else []

    def _normalize_task_title(self, title: str) -> str:
        """Strip common task numbering prefixes from recovered titles."""

        cleaned = re.sub(r"^任务\s*\d+\s*[:：\-]\s*", "", title.strip())
        cleaned = cleaned.strip().strip('"').strip("'").strip("*").strip("`")
        return cleaned.strip()

    def _extract_intent_from_text(self, text: str) -> str:
        """Extract a concise intent line from free-form task text."""

        for raw_line in text.splitlines():
            line = raw_line.strip().strip("-*#` ")
            if not line:
                continue
            if line.startswith("任务目标"):
                _, _, suffix = line.partition("：")
                if suffix.strip():
                    return suffix.strip()
                _, _, suffix = line.partition(":")
                if suffix.strip():
                    return suffix.strip()
            if "记录任务概览" in line:
                continue
            return line
        return ""

    def _invoke_planner(
        self,
        prompt: str,
        *,
        observer: RequestTrace | None = None,
    ) -> str:
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
        else:
            if observer:
                observer.record_llm_call(
                    success=True,
                    prompt_text=prompt,
                    completion_text=response,
                )
            return response
        finally:
            self._agent.clear_history()

    def _sanitize_tasks(
        self,
        tasks: List[dict[str, Any]],
        *,
        research_topic: str,
    ) -> tuple[List[dict[str, Any]], int]:
        sanitized: List[dict[str, Any]] = []
        seen_titles: set[str] = set()
        rejected_count = 0

        for item in tasks:
            title = self._normalize_task_title(str(item.get("title") or ""))
            intent = str(item.get("intent") or "").strip()
            query = str(item.get("query") or "").strip()
            if not title:
                rejected_count += 1
                continue

            if self._is_meta_workflow_task(title=title, intent=intent, query=query):
                rejected_count += 1
                continue

            title_key = title.casefold()
            if title_key in seen_titles:
                continue

            seen_titles.add(title_key)
            sanitized.append(
                {
                    "title": title,
                    "intent": intent or "聚焦主题的关键问题",
                    "query": query,
                }
            )

        return sanitized, rejected_count

    def _should_repair_plan(
        self,
        tasks: List[dict[str, Any]],
        rejected_count: int,
        raw_response: str,
    ) -> bool:
        if rejected_count > 0:
            return True
        if len(tasks) >= 2:
            return False

        normalized_response = raw_response.casefold()
        keyword_hits = sum(1 for keyword in META_WORKFLOW_KEYWORDS if keyword.casefold() in normalized_response)
        return keyword_hits >= 2

    def _is_meta_workflow_task(self, *, title: str, intent: str, query: str) -> bool:
        normalized_title = self._normalize_task_title(title).replace(" ", "").casefold()
        normalized_intent = intent.replace(" ", "").casefold()
        normalized_query = query.replace(" ", "").casefold()

        if any(keyword.casefold() in normalized_title for keyword in META_WORKFLOW_KEYWORDS):
            return True

        combined = f"{normalized_title} {normalized_intent} {normalized_query}"
        hint_hits = sum(1 for hint in META_WORKFLOW_HINTS if hint.casefold() in combined)
        return hint_hits >= 2

    def _build_repair_prompt(self, research_topic: str, previous_response: str) -> str:
        return f"""
你上一轮返回的是系统执行步骤或协作流程，而不是围绕研究主题的调研任务。
禁止输出这类标题：启动检索、进度同步、交叉验证、综合报告、并行执行、依赖关系、信息整合、标签追踪。

请重新规划 3~5 个真正面向主题内容的研究任务，任务标题要体现主题维度，例如“技术发展脉络”“主流模型能力对比”“典型应用场景”“技术瓶颈与未来方向”，而不是系统工作步骤。

研究主题：{research_topic}

你上一轮的错误输出如下，仅供参考，请不要复用其中的流程标题：
{previous_response}

请严格只输出 JSON：
{{
  "tasks": [
    {{
      "title": "任务名称（10字内，突出主题内容）",
      "intent": "任务要解决的核心问题，用1-2句描述",
      "query": "建议使用的检索关键词"
    }}
  ]
}}
"""
