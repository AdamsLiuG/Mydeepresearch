"""State models used by the deep research workflow."""

import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Literal

from typing_extensions import Annotated

ResearchDepth = Literal["quick", "standard", "deep"]
WorkClass = Literal["required", "optional"]
Resource = Literal[
    "tasks",
    "search_calls",
    "web_fetches",
    "dynamic_fetches",
    "sandbox_calls",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cost_microusd",
]
DagActionName = Literal["PIVOT", "DEEPEN", "PRUNE"]
BudgetPressure = Literal["normal", "high", "exhausted"]

RESOURCE_NAMES: tuple[str, ...] = (
    "tasks",
    "search_calls",
    "web_fetches",
    "dynamic_fetches",
    "sandbox_calls",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cost_microusd",
)
FINALIZATION_RESOURCES: tuple[str, ...] = (
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cost_microusd",
)


def _validated_resource_map(value: dict[str, int], *, sparse: bool) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("resource values must be a dictionary")
    unknown = set(value).difference(RESOURCE_NAMES)
    if unknown:
        raise ValueError(f"unknown resources: {', '.join(sorted(unknown))}")
    result = {name: 0 for name in RESOURCE_NAMES} if not sparse else {}
    for name, amount in value.items():
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f"resource {name} must be a non-negative integer")
        result[name] = amount
    return result


@dataclass(frozen=True)
class RequestBudget:
    """Immutable request budget values with defensive copies at the API boundary."""

    depth: ResearchDepth
    limits: dict[Resource, int]
    finalization_reserve: dict[Resource, int]
    deadline_at: str
    finalization_reserve_ms: int
    profile_version: int = 1

    def __post_init__(self) -> None:
        if self.depth not in {"quick", "standard", "deep"}:
            raise ValueError("depth must be quick, standard, or deep")
        limits = _validated_resource_map(self.limits, sparse=False)
        reserve = _validated_resource_map(self.finalization_reserve, sparse=True)
        if (
            isinstance(self.finalization_reserve_ms, bool)
            or not isinstance(self.finalization_reserve_ms, int)
            or self.finalization_reserve_ms < 0
        ):
            raise ValueError("finalization_reserve_ms must be non-negative")
        if (
            isinstance(self.profile_version, bool)
            or not isinstance(self.profile_version, int)
            or self.profile_version < 1
        ):
            raise ValueError("profile_version must be a positive integer")
        deadline = str(self.deadline_at).strip()
        try:
            parsed_deadline = datetime.fromisoformat(
                deadline[:-1] + "+00:00" if deadline.endswith("Z") else deadline
            )
        except ValueError as exc:
            raise ValueError("deadline_at must be an RFC 3339 timestamp") from exc
        if not deadline or parsed_deadline.tzinfo is None or parsed_deadline.utcoffset() is None:
            raise ValueError("deadline_at must be an RFC 3339 timestamp")
        for resource, amount in reserve.items():
            if amount > limits[resource]:
                raise ValueError(f"reserve for {resource} exceeds its limit")
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "finalization_reserve", reserve)
        object.__setattr__(self, "deadline_at", deadline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "limits": dict(self.limits),
            "finalization_reserve": dict(self.finalization_reserve),
            "deadline_at": self.deadline_at,
            "finalization_reserve_ms": self.finalization_reserve_ms,
            "profile_version": self.profile_version,
        }


@dataclass(frozen=True)
class BudgetLease:
    operation_id: str
    reserved: dict[Resource, int]
    deadline_at: str
    work_class: WorkClass
    stage: str = "research"

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        if self.work_class not in {"required", "optional"}:
            raise ValueError("work_class must be required or optional")
        object.__setattr__(self, "reserved", _validated_resource_map(self.reserved, sparse=True))

    @property
    def allowed(self) -> bool:
        return True


@dataclass(frozen=True)
class BudgetDenial:
    reason: str
    resource: Resource | None = None

    @property
    def allowed(self) -> bool:
        return False


@dataclass(frozen=True)
class BudgetSettlement:
    operation_id: str
    usage: dict[Resource, int]
    usage_source: Literal["reported", "estimated", "unknown"]
    charged_cost_microusd: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _validated_resource_map(self.usage, sparse=True))


@dataclass(frozen=True)
class BudgetSnapshot:
    limit: RequestBudget
    spent: dict[Resource, int]
    reserved: dict[Resource, int]
    remaining: dict[Resource, int]
    pressure: BudgetPressure
    last_denial_reason: str | None = None
    external_attempts: int = 0
    overrun: dict[Resource, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DagRevision:
    revision: int = 0
    wave_1_committed: bool = False
    decision_id: str | None = None
    pruned_task_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if not isinstance(self.wave_1_committed, bool):
            raise ValueError("wave_1_committed must be a boolean")
        if any(isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0 for task_id in self.pruned_task_ids):
            raise ValueError("pruned task IDs must be positive integers")
        object.__setattr__(self, "pruned_task_ids", tuple(sorted(set(self.pruned_task_ids))))


@dataclass(frozen=True)
class BranchScore:
    task_id: int
    information_gain: float
    conflict_density: float
    ig_numerators: tuple[int, int, int] = (0, 0, 0)
    ig_denominators: tuple[int, int, int] = (0, 0, 0)
    covered_dimensions: tuple[str, ...] = ()
    effective_fact_count: int = 0
    independent_source_clusters: int = 0
    comparable_dimensions: int = 0
    conflicting_dimensions: int = 0
    comparable_pair_count: int = 0
    insufficient_samples: bool = True
    score_version: int = 1
    missing_coverage_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class DagAction:
    action: DagActionName
    task_id: int
    parent_task_ids: list[int] = field(default_factory=list)
    reason_code: str = ""
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


@dataclass(frozen=True)
class DagPatch:
    base_revision: int
    decision_id: str
    actions: list[DagAction] = field(default_factory=list)
    score_version: int = 1
    scores: list[BranchScore] = field(default_factory=list)


@dataclass(frozen=True)
class ReplanResult:
    patch: DagPatch | None
    reason_code: str


@dataclass(frozen=True)
class DagPatchResult:
    applied: bool
    reason_code: str
    revision: int
    tasks: list["TodoItem"] = field(default_factory=list)
    waves: list[list["TodoItem"]] = field(default_factory=list)
    added_task_ids: list[int] = field(default_factory=list)
    pruned_task_ids: list[int] = field(default_factory=list)


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
    depends_on: list[int] = field(default_factory=list)
    upstream_context: str = field(default="")
    fact_claims: list[dict[str, Any]] = field(default_factory=list)
    atomic_claims: list[dict[str, Any]] = field(default_factory=list)


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
    topic_cache_warmup_completed: bool = field(default=False)
    skill_pack_metadata: dict = field(default_factory=dict)
    fact_ledger: dict[str, Any] = field(default_factory=dict)
    evidence_index_cards: list[dict[str, Any]] = field(default_factory=list)
    frozen_prefix_version: int = field(default=0)
    context_checkpoint_blocks: list[dict[str, dict[str, str]]] = field(default_factory=list)
    pending_tool_approval: dict[str, Any] | None = field(default=None)
    approved_tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_charts: list[dict[str, Any]] = field(default_factory=list)
    report_workbench: dict[str, Any] = field(default_factory=dict)
    report_workbench_fingerprint: str | None = None
    evidence_conflicts: list[dict[str, Any]] = field(default_factory=list)
    claim_verification_summary: dict[str, Any] = field(default_factory=dict)
    state_version: int = field(default=0)
    runner_backend: str = field(default="inline")
    workflow_run_id: str | None = field(default=None)
    v4: dict[str, Any] | None = field(default=None)


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)  # Backward-compatible文本
    report_markdown: str | None = field(default=None)
    report_charts: list[dict[str, Any]] = field(default_factory=list)
    report_workbench: dict[str, Any] = field(default_factory=dict)
    evidence_conflicts: list[dict[str, Any]] = field(default_factory=list)
    claim_verification_summary: dict[str, Any] = field(default_factory=dict)
    state_version: int = field(default=0)
    runner_backend: str = field(default="inline")
    workflow_run_id: str | None = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)
    v4: dict[str, Any] | None = field(default=None)


@dataclass(frozen=True)
class ReportEvidenceRef:
    source_id: str
    anchor_id: str | None
    artifact_version: str | None
    excerpt_fingerprint: str


@dataclass(frozen=True)
class ReportExcerpt:
    ref: ReportEvidenceRef
    text: str


@dataclass(frozen=True)
class ReportScope:
    entity: str = "unknown"
    metric: str = "unknown"
    period_start: str = "unknown"
    period_end: str = "unknown"
    period_basis: str = "unknown"
    unit: str = "unknown"
    currency: str = "unknown"
    denominator: str = "unknown"
    population: str = "unknown"
    methodology: str = "unknown"


@dataclass(frozen=True)
class ReportClaim:
    claim_id: str
    text: str
    support_refs: tuple[ReportEvidenceRef, ...]
    task_ids: tuple[int, ...]
    scope: ReportScope = field(default_factory=ReportScope)
    atomic_claim_id: str | None = None
    value: str = ""
    observed_period: str = ""


@dataclass(frozen=True)
class ReportConflict:
    conflict_id: str
    claim_ids: tuple[str, ...]
    sides: tuple[ReportClaim, ...]
    status: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportSectionPlan:
    section_id: str
    order: int
    title: str
    scope: str
    coverage_keys: tuple[str, ...]
    claim_ids: tuple[str, ...]
    allowed_source_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    kind: Literal["body", "summary", "conclusion", "conflicts", "references"]
    slot_key: str = ""


@dataclass(frozen=True)
class ReportOutline:
    outline_version: int
    ledger_version: int
    frozen_prefix_version: int
    sections: tuple[ReportSectionPlan, ...]


@dataclass(frozen=True)
class ReportBlock:
    block_id: str
    kind: Literal["fact", "conflict", "gap", "transition"]
    text: str
    claim_id: str | None = None
    evidence_refs: tuple[ReportEvidenceRef, ...] = ()
    conflict_id: str | None = None
    gap_code: str | None = None
    coverage_key: str | None = None


@dataclass(frozen=True)
class SectionDraft:
    section_id: str
    markdown: str
    claim_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    gap_codes: tuple[str, ...]
    blocks: tuple[ReportBlock, ...]


@dataclass(frozen=True)
class ReportEvidencePack:
    section: ReportSectionPlan
    claims: tuple[ReportClaim, ...]
    excerpts: tuple[ReportExcerpt, ...]
    conflicts: tuple[ReportConflict, ...]
    required_coverage: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReportInput:
    outline: ReportOutline
    packs: tuple[ReportEvidencePack, ...]
    excerpts: tuple[ReportExcerpt, ...]
    conflicts: tuple[ReportConflict, ...]
    required_coverage: tuple[tuple[str, str], ...]
    eligible_source_ids: frozenset[str]
    omitted_claim_ids: tuple[str, ...] = ()
    coverage_by_claim: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class ReportModelResponse:
    text: str
    finish_reason: str
    usage: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class CriticIssue:
    section_id: str
    claim_id: str | None
    kind: Literal["counter_evidence_gap", "survivorship_bias", "temporal_skew", "causal_overreach"]
    severity: Literal["medium", "high"]
    source_ids: tuple[str, ...]
    proposed_action: Literal["qualify", "surface_conflict", "mark_gap", "reverify"]
    rationale: str
    conflict_id: str | None = None
    reason_code: str = ""
    issue_id: str = ""


@dataclass(frozen=True)
class ReportRevisionPlan:
    draft_digest: str
    issues: tuple[CriticIssue, ...]
    round_number: int = 1


@dataclass(frozen=True)
class ReportCriticResult:
    status: Literal["ok", "skipped", "timeout", "invalid"]
    issues: tuple[CriticIssue, ...] = ()
    reason: str = "ok"
