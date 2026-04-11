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
from prompts import (
    get_current_date,
    report_repair_task_prompt,
    supplemental_planner_instructions,
    todo_planner_instructions,
)
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)
_STRICT_JSON_RESPONSE_FORMAT = {"type": "json_object"}

NUMBERED_TASK_PATTERN = re.compile(
    r"^(?:[-*]\s*)?(?P<index>\d+)[\.\)、]\s*(?P<title>[^：:]{1,40})(?:[：:]\s*(?P<intent>.+))?$"
)
MARKDOWN_TABLE_TASK_PATTERN = re.compile(
    r"^\|\s*(?P<index>\d+)\s*\|\s*(?P<title>[^|]+?)\s*\|\s*(?P<intent>[^|]+?)\s*\|(?:\s*[^|]+?\s*\|)?$"
)
MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|\s*[:\-| ]+\|\s*$")
META_WORKFLOW_KEYWORDS = (
    "启动检索",
    "进度同步",
    "交叉验证",
    "综合报告",
    "迭代更新",
    "风险前置",
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
META_PHASE_PREFIXES = (
    "优先",
    "并行",
    "跟进",
    "最后",
    "首先",
    "随后",
    "先做",
    "后做",
)
QUERY_NOISE_PATTERNS = (
    re.compile(r"```.*?```", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[TOOL_CALL:[^\]]+\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bnote_[A-Za-z0-9_]+\b", re.IGNORECASE),
    re.compile(r"\b(?:search_web|update_note|note_tool)\s*\([^)]*\)", re.IGNORECASE),
)
QUERY_NOISE_PHRASES = (
    "search_web",
    "update note",
    "按顺序执行",
    "按任务顺序执行",
    "并更新笔记状态",
    "更新笔记状态",
    "工具调用",
    "工作流",
    "流程编排",
    "进度同步",
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
        historical_memory_context: str | None = None,
        strategy_memory_context: str | None = None,
    ) -> List[TodoItem]:
        """Ask the planner agent to break the topic into actionable tasks."""
        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )
        prompt += self._historical_memory_block(historical_memory_context)
        prompt += self._strategy_memory_block(strategy_memory_context)
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
            query = self._canonical_query_for_task(
                research_topic=state.research_topic,
                title=title,
                intent=intent,
                raw_query=str(item.get("query") or "").strip(),
            )

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
                origin="planned",
                round=1,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    def plan_additional_tasks(
        self,
        state: SummaryState,
        *,
        missing_angles: list[str],
        existing_tasks: list[TodoItem],
        max_additional_tasks: int,
        observer: RequestTrace | None = None,
        historical_memory_context: str | None = None,
    ) -> List[TodoItem]:
        """Generate supplemental tasks that cover missing angles without duplicating existing work."""

        if max_additional_tasks <= 0:
            return []

        next_id = max((task.id for task in existing_tasks), default=0) + 1
        prompt = supplemental_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
            starting_task_id=next_id,
            existing_tasks="\n".join(
                f"- 任务 {task.id}: {task.title} | 目标：{task.intent} | 状态：{task.status}"
                for task in existing_tasks
            )
            or "- 暂无已规划任务",
            missing_angles="\n".join(f"- {angle}" for angle in missing_angles)
            or "- 无明确缺失维度",
            max_additional_tasks=max_additional_tasks,
        )
        prompt += self._historical_memory_block(historical_memory_context)
        response = self._invoke_planner(prompt, observer=observer)
        logger.info("Supplemental planner raw output (truncated): %s", response[:500])

        raw_tasks = self._extract_tasks(response)
        sanitized_tasks, _ = self._sanitize_tasks(raw_tasks, research_topic=state.research_topic)
        unique_tasks = self._filter_duplicate_candidates(sanitized_tasks, existing_tasks=existing_tasks)

        supplemental_items: List[TodoItem] = []
        for item in unique_tasks[:max_additional_tasks]:
            title = str(item.get("title") or f"任务{next_id}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            query = self._canonical_query_for_task(
                research_topic=state.research_topic,
                title=title,
                intent=intent,
                raw_query=str(item.get("query") or "").strip(),
            )

            supplemental_items.append(
                TodoItem(
                    id=next_id,
                    title=title,
                    intent=intent,
                    query=query,
                    origin="replanned",
                    round=2,
                )
            )
            next_id += 1

        return supplemental_items

    def plan_repair_tasks(
        self,
        state: SummaryState,
        *,
        repair_candidates: list[dict[str, Any]],
        existing_tasks: list[TodoItem],
        max_additional_tasks: int,
        observer: RequestTrace | None = None,
        historical_memory_context: str | None = None,
    ) -> List[TodoItem]:
        """Generate targeted repair tasks from high-priority review findings."""

        if max_additional_tasks <= 0 or not repair_candidates:
            return []

        next_id = max((task.id for task in existing_tasks), default=0) + 1
        next_round = max((task.round for task in existing_tasks), default=1) + 1
        review_summary = state.review_summary or {}
        prompt = report_repair_task_prompt.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
            starting_task_id=next_id,
            existing_tasks="\n".join(
                f"- 任务 {task.id}: {task.title} | 目标：{task.intent} | 状态：{task.status}"
                for task in existing_tasks
            )
            or "- 暂无已执行任务",
            review_summary=json.dumps(
                {
                    "overall_status": review_summary.get("overall_status"),
                    "reason": review_summary.get("reason"),
                    "issue_count": review_summary.get("issue_count"),
                },
                ensure_ascii=False,
            ),
            repair_candidates="\n".join(
                (
                    f"- task_id={item.get('task_id')}, severity={item.get('severity')}, "
                    f"check={item.get('check')}, message={item.get('message')}"
                )
                for item in repair_candidates
            )
            or "- 无待修补问题",
            max_additional_tasks=max_additional_tasks,
        )
        prompt += self._historical_memory_block(historical_memory_context)
        response = self._invoke_planner(prompt, observer=observer)
        logger.info("Repair planner raw output (truncated): %s", response[:500])

        raw_tasks = self._extract_tasks(response)
        sanitized_tasks, _ = self._sanitize_tasks(raw_tasks, research_topic=state.research_topic)
        unique_tasks = self._filter_duplicate_candidates(sanitized_tasks, existing_tasks=existing_tasks)

        repair_items: List[TodoItem] = []
        for item in unique_tasks[:max_additional_tasks]:
            title = str(item.get("title") or f"任务{next_id}").strip()
            intent = str(item.get("intent") or "修补当前审查暴露的证据缺口").strip()
            query = self._canonical_query_for_task(
                research_topic=state.research_topic,
                title=title,
                intent=intent,
                raw_query=str(item.get("query") or "").strip(),
            )

            repair_items.append(
                TodoItem(
                    id=next_id,
                    title=title,
                    intent=intent,
                    query=query,
                    origin="repair",
                    round=next_round,
                )
            )
            next_id += 1

        return repair_items

    @staticmethod
    def _historical_memory_block(historical_memory_context: str | None) -> str:
        context = str(historical_memory_context or "").strip()
        if not context:
            return ""
        return (
            "\n\n<HISTORICAL_MEMORY>\n"
            "以下内容来自历史研究笔记，仅用于启发任务拆解与覆盖检查，不代表本轮已经验证过的事实。\n"
            "你仍需要为当前研究主题独立规划任务，并通过本轮搜索重新获得证据。\n"
            f"{context}\n"
            "</HISTORICAL_MEMORY>\n"
        )

    @staticmethod
    def _strategy_memory_block(strategy_memory_context: str | None) -> str:
        context = str(strategy_memory_context or "").strip()
        if not context:
            return ""
        return (
            "\n\n<STRATEGY_MEMORY>\n"
            "以下内容来自历史请求提炼出的策略记忆，只能作为任务拆解、检索构造、来源偏好和风险规避提示。\n"
            "这些内容不是当前主题事实，不是本轮证据，也不能直接复用历史报告结论。\n"
            f"{context}\n"
            "</STRATEGY_MEMORY>\n"
        )

    def _default_query_for_task(
        self,
        *,
        research_topic: str,
        title: str,
        intent: str,
    ) -> str:
        """Backward-compatible alias for the deterministic task query canonicalizer."""

        return self._canonical_query_for_task(
            research_topic=research_topic,
            title=title,
            intent=intent,
            raw_query="",
        )

    def _canonical_query_for_task(
        self,
        *,
        research_topic: str,
        title: str,
        intent: str,
        raw_query: str,
    ) -> str:
        """Build the deterministic final search query for a task."""

        topic = self._clean_query_noise(research_topic)
        task_title = self._strip_title_already_in_topic(topic, self._normalize_task_title(title))
        task_intent = self._concise_intent(intent)
        cleaned_raw_query = self._clean_query_noise(raw_query)

        if topic and task_title:
            return f"{topic} {task_title}".strip()
        if topic and task_intent:
            return f"{topic} {task_intent}".strip()
        if topic:
            return topic
        if task_title:
            return task_title
        if task_intent:
            return task_intent
        return cleaned_raw_query

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning failed."""
        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=state.research_topic if state.research_topic else "基础背景梳理",
            origin="planned",
            round=1,
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

        markdown_table_tasks = self._extract_tasks_from_markdown_table(text)
        if markdown_table_tasks:
            logger.warning(
                "Planner response did not contain parseable task JSON; recovered %d tasks from markdown table",
                len(markdown_table_tasks),
            )
            return markdown_table_tasks

        numbered_tasks = self._extract_tasks_from_numbered_text(text)
        if numbered_tasks:
            logger.warning(
                "Planner response did not contain parseable task JSON; recovered %d tasks from numbered text",
                len(numbered_tasks),
            )
            return numbered_tasks

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

            content = str(payload.get("content") or "").strip()
            raw_title = str(
                payload.get("title")
                or payload.get("task_title")
                or payload.get("name")
                or f"任务{normalized_id}"
            ).strip()
            title = self._normalize_task_title(raw_title)
            if not title or self._is_placeholder_task_title(title):
                title = self._extract_task_title_from_text(content, task_id=normalized_id) or title

            intent = str(payload.get("intent") or payload.get("objective") or "").strip()
            if not intent:
                intent = self._extract_intent_from_text(content) or "聚焦主题的关键问题"
            query = str(
                payload.get("query")
                or payload.get("search_query")
                or payload.get("search")
                or payload.get("keywords")
                or ""
            ).strip()
            if not query:
                query = self._extract_query_from_text(content)

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

        for block in self._iter_markdown_table_blocks(text):
            structured_tasks = self._extract_tasks_from_markdown_table_block(block)
            if structured_tasks:
                return structured_tasks

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

    def _iter_markdown_table_blocks(self, text: str):
        """Yield contiguous markdown table blocks."""

        current_block: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("|"):
                current_block.append(stripped)
                continue

            if current_block:
                yield current_block
                current_block = []

        if current_block:
            yield current_block

    def _extract_tasks_from_markdown_table_block(self, lines: list[str]) -> List[dict[str, Any]]:
        """Recover tasks from a single markdown table block with flexible headers."""

        if len(lines) < 3:
            return []

        header_cells = self._split_markdown_table_row(lines[0])
        if not header_cells:
            return []

        title_index = self._find_markdown_table_column(
            header_cells,
            ("任务名称", "任务标题", "标题", "名称"),
        )
        if title_index is None:
            return []

        intent_index = self._find_markdown_table_column(
            header_cells,
            ("任务目标", "核心关注点", "核心问题", "目标意图", "任务意图", "关注点", "研究重点"),
        )
        query_index = self._find_markdown_table_column(
            header_cells,
            ("检索方向", "检索关键词", "建议检索", "查询", "query"),
        )

        start_index = 1
        if len(lines) > 1 and self._is_markdown_table_separator(lines[1]):
            start_index = 2

        tasks: List[dict[str, Any]] = []
        for line in lines[start_index:]:
            if self._is_markdown_table_separator(line):
                continue

            cells = self._split_markdown_table_row(line)
            if title_index >= len(cells):
                continue

            title = self._normalize_task_title(cells[title_index])
            if not title:
                continue

            intent = cells[intent_index].strip() if intent_index is not None and intent_index < len(cells) else ""
            query = cells[query_index].strip() if query_index is not None and query_index < len(cells) else ""
            tasks.append(
                {
                    "title": title,
                    "intent": intent or "聚焦主题的关键问题",
                    "query": query,
                }
            )

        return tasks if len(tasks) >= 2 else []

    def _split_markdown_table_row(self, line: str) -> list[str]:
        """Split a markdown table row into normalized cell strings."""

        stripped = line.strip()
        if not stripped.startswith("|"):
            return []
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    def _is_markdown_table_separator(self, line: str) -> bool:
        """Return whether the line is a markdown table separator row."""

        return bool(MARKDOWN_TABLE_SEPARATOR_PATTERN.match(line.strip()))

    def _find_markdown_table_column(
        self,
        headers: list[str],
        candidates: tuple[str, ...],
    ) -> int | None:
        """Find the first header column whose normalized text contains a candidate token."""

        normalized_headers = [re.sub(r"\s+", "", header).casefold() for header in headers]
        for candidate in candidates:
            token = candidate.casefold()
            for index, header in enumerate(normalized_headers):
                if token in header:
                    return index
        return None

    def _normalize_task_title(self, title: str) -> str:
        """Strip common task numbering prefixes from recovered titles."""

        cleaned = title.strip()
        arrow_match = re.search(r"(?:→|->|=>|➜)\s*(.+)$", cleaned)
        if arrow_match and arrow_match.group(1).strip():
            cleaned = arrow_match.group(1).strip()

        cleaned = re.sub(r"^(?:[-*#]\s*)?(?:任务\s*)?\d+\s*[\.\)、:：\-]\s*", "", cleaned)
        cleaned = re.sub(r"[（(]\s*任务\s*[\d、,，\s]+\s*[）)]", "", cleaned)
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = cleaned.strip().strip('"').strip("'").strip("*").strip("`")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" -:：|/").strip()
        return cleaned.strip()

    def _is_placeholder_task_title(self, title: str) -> bool:
        """Return whether the title is still a bare task placeholder."""

        normalized = self._normalize_task_title(title)
        return bool(re.fullmatch(r"(?:任务|task)\s*_?\s*\d+", normalized, flags=re.IGNORECASE))

    def _extract_task_title_from_text(self, text: str, *, task_id: int | None = None) -> str:
        """Recover a concrete task title from note-like free-form content."""

        numbered_prefix = r"\d+" if task_id is None else re.escape(str(task_id))
        inline_pattern = re.compile(
            rf"^(?:任务|task)\s*_?\s*{numbered_prefix}\s*[:：\-]\s*(?P<title>.+)$",
            flags=re.IGNORECASE,
        )

        for raw_line in text.splitlines():
            line = raw_line.strip().strip("-*#` ")
            if not line:
                continue

            for label in ("任务标题", "任务名称", "标题", "名称"):
                if line.startswith(label):
                    _, _, suffix = line.partition("：")
                    if suffix.strip():
                        candidate = self._normalize_task_title(suffix)
                        if candidate and not self._is_placeholder_task_title(candidate):
                            return candidate
                    _, _, suffix = line.partition(":")
                    if suffix.strip():
                        candidate = self._normalize_task_title(suffix)
                        if candidate and not self._is_placeholder_task_title(candidate):
                            return candidate

            match = inline_pattern.match(line)
            if not match:
                continue

            candidate = self._normalize_task_title(match.group("title"))
            if candidate and not self._is_placeholder_task_title(candidate):
                return candidate

        return ""

    def _extract_query_from_text(self, text: str) -> str:
        """Extract a suggested search query from note-like free-form content."""

        for raw_line in text.splitlines():
            line = raw_line.strip().strip("-*#` ")
            if not line:
                continue

            for label in ("检索方向", "搜索方向", "检索关键词", "搜索关键词", "查询", "Query", "query"):
                if line.startswith(label):
                    _, _, suffix = line.partition("：")
                    if suffix.strip():
                        return self._clean_query_noise(suffix)
                    _, _, suffix = line.partition(":")
                    if suffix.strip():
                        return self._clean_query_noise(suffix)

        return ""

    def _clean_query_noise(self, value: str) -> str:
        cleaned = str(value or "").strip()
        for pattern in QUERY_NOISE_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)
        for phrase in QUERY_NOISE_PHRASES:
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"(?:^|[\s,，;；])(?:query|search query|检索方向|搜索方向|检索关键词|搜索关键词)\s*[:：]\s*",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.replace("`", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" -:：,，;；|/")

    def _concise_intent(self, intent: str) -> str:
        cleaned = self._clean_query_noise(intent)
        if not cleaned:
            return ""
        parts = re.split(r"[。！？!?；;]+", cleaned, maxsplit=1)
        return parts[0].strip()

    def _strip_title_already_in_topic(self, research_topic: str, title: str) -> str:
        normalized_topic = re.sub(r"\s+", "", str(research_topic or "")).casefold()
        normalized_title = re.sub(r"\s+", "", str(title or "")).casefold()
        if normalized_topic and normalized_title and normalized_title in normalized_topic:
            return ""
        return title

    def _filter_duplicate_candidates(
        self,
        tasks: List[dict[str, Any]],
        *,
        existing_tasks: list[TodoItem],
    ) -> List[dict[str, Any]]:
        """Drop supplemental tasks that substantially overlap with existing work."""

        unique_tasks: list[dict[str, Any]] = []
        seen_titles = {self._normalize_task_title(task.title).casefold() for task in existing_tasks}

        for item in tasks:
            title = self._normalize_task_title(str(item.get("title") or ""))
            intent = str(item.get("intent") or "").strip()
            title_key = title.casefold()
            if not title or title_key in seen_titles:
                continue

            if any(
                self._is_similar_task(
                    title=title,
                    intent=intent,
                    existing_title=task.title,
                    existing_intent=task.intent,
                )
                for task in existing_tasks
            ):
                continue

            seen_titles.add(title_key)
            unique_tasks.append(
                {
                    "title": title,
                    "intent": intent or "聚焦主题的关键问题",
                    "query": str(item.get("query") or "").strip(),
                }
            )

        return unique_tasks

    def _is_similar_task(
        self,
        *,
        title: str,
        intent: str,
        existing_title: str,
        existing_intent: str,
    ) -> bool:
        """Heuristic duplicate check for supplemental tasks."""

        normalized_title = self._normalize_similarity_text(title)
        normalized_existing_title = self._normalize_similarity_text(existing_title)

        if normalized_title == normalized_existing_title:
            return True
        if normalized_title and normalized_existing_title:
            if normalized_title in normalized_existing_title or normalized_existing_title in normalized_title:
                return True

        title_similarity = self._char_jaccard_similarity(normalized_title, normalized_existing_title)
        if title_similarity >= 0.75:
            return True

        normalized_intent = self._normalize_similarity_text(intent)
        normalized_existing_intent = self._normalize_similarity_text(existing_intent)
        intent_similarity = self._char_jaccard_similarity(normalized_intent, normalized_existing_intent)
        return title_similarity >= 0.5 and intent_similarity >= 0.6

    def _normalize_similarity_text(self, value: str) -> str:
        cleaned = self._normalize_task_title(value).casefold()
        return re.sub(r"[\s\-_/:：|，,。；;（）()]+", "", cleaned)

    def _char_jaccard_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0

        def grams(text: str) -> set[str]:
            if len(text) <= 2:
                return {text}
            return {text[index : index + 2] for index in range(len(text) - 1)}

        left_grams = grams(left)
        right_grams = grams(right)
        union = left_grams | right_grams
        if not union:
            return 0.0
        return len(left_grams & right_grams) / len(union)

    def _extract_intent_from_text(self, text: str) -> str:
        """Extract a concise intent line from free-form task text."""

        for raw_line in text.splitlines():
            line = raw_line.strip().strip("-*#` ")
            if not line:
                continue
            if self._extract_task_title_from_text(line):
                continue
            if line.startswith("任务目标"):
                _, _, suffix = line.partition("：")
                if suffix.strip():
                    return suffix.strip()
                _, _, suffix = line.partition(":")
                if suffix.strip():
                    return suffix.strip()
            if line.startswith("目标意图"):
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
            try:
                response = self._agent.run(prompt, response_format=_STRICT_JSON_RESPONSE_FORMAT)
            except Exception as exc:
                if not self._response_format_is_unsupported(exc):
                    raise
                logger.info("Planner JSON mode unsupported; retrying without response_format: %s", exc)
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
            original_title = str(item.get("title") or "")
            title = self._normalize_task_title(original_title)
            intent = self._clean_query_noise(str(item.get("intent") or ""))
            query = self._clean_query_noise(str(item.get("query") or ""))
            if not title:
                rejected_count += 1
                continue

            if self._is_meta_workflow_task(
                title=title,
                intent=intent,
                query=query,
                raw_title=original_title,
            ):
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

    def _is_meta_workflow_task(
        self,
        *,
        title: str,
        intent: str,
        query: str,
        raw_title: str | None = None,
    ) -> bool:
        raw_title = raw_title or title
        normalized_title = self._normalize_task_title(title).replace(" ", "").casefold()
        normalized_intent = intent.replace(" ", "").casefold()
        normalized_query = query.replace(" ", "").casefold()

        if any(token in raw_title for token in ("→", "->", "=>", "➜")):
            return True

        if re.search(r"[（(]\s*任务\s*[\d、,，\s]+\s*[）)]", raw_title):
            return True

        if any(normalized_title.startswith(prefix.casefold()) for prefix in META_PHASE_PREFIXES):
            return True

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
