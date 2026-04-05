"""State models used by the deep research workflow."""

import operator
from dataclasses import dataclass, field
from typing import Any, List

from typing_extensions import Annotated


@dataclass(kw_only=True)
class TodoItem:
    """单个待办任务项。"""

    id: int
    title: str
    intent: str
    query: str
    status: str = field(default="pending")
    summary: str | None = field(default=None)
    summary_payload: dict | None = field(default=None)
    sources_summary: str | None = field(default=None)
    notices: list[str] = field(default_factory=list)
    evidence_items: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    review_issues: list[dict] = field(default_factory=list)
    review_status: str = field(default="pending")
    note_id: str | None = field(default=None)
    note_path: str | None = field(default=None)
    stream_token: str | None = field(default=None)
    origin: str = field(default="planned")
    round: int = field(default=1)
    react_rounds: int = field(default=0)
    react_fetch_count: int = field(default=0)
    react_additional_search_count: int = field(default=0)
    react_gap_signals: list[str] = field(default_factory=list)
    react_last_action: str | None = field(default=None)
    react_stop_reason: str | None = field(default=None)
    react_observation: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)  # Report topic
    search_query: str = field(default=None)  # Deprecated placeholder
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)  # Research loop count
    running_summary: str = field(default=None)  # Legacy summary field
    todo_items: Annotated[list, operator.add] = field(default_factory=list)
    structured_report: str | None = field(default=None)
    report_note_id: str | None = field(default=None)
    report_note_path: str | None = field(default=None)
    review_summary: dict = field(default_factory=dict)
    reflection_completed: bool = field(default=False)
    review_completed: bool = field(default=False)
    report_repair_completed: bool = field(default=False)
    report_repair_cycles: int = field(default=0)


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)  # Backward-compatible文本
    report_markdown: str | None = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)
