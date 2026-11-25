export type ResearchDepth = "quick" | "standard" | "deep";

export interface ResearchRequest {
  topic: string;
  search_api?: string;
  output_format?: string;
  mode?: "auto" | "plan_first";
  research_depth?: ResearchDepth;
}

export interface BudgetSnapshot {
  limit: {
    depth: ResearchDepth;
    limits: Record<string, number>;
    finalization_reserve: Record<string, number>;
    deadline_at: string;
    finalization_reserve_ms: number;
    profile_version: number;
  };
  spent: Record<string, number>;
  reserved: Record<string, number>;
  remaining: Record<string, number>;
  pressure: "normal" | "high" | "exhausted";
  last_denial_reason?: string;
  external_attempts?: number;
  overrun?: Record<string, number>;
}

export type ReportSectionStatus = "pending" | "started" | "completed" | "degraded";

export interface ReportProgress {
  outline_version: number;
  report_version?: number;
  critic_status: "ok" | "skipped" | "timeout" | "invalid";
  degradation_reason?: string;
  sections: {
    section_id: string;
    title: string;
    order: number;
    status: ReportSectionStatus;
    claim_ids?: string[];
  }[];
}

export interface V4ResearchProjection {
  schema_version: 1;
  budget?: BudgetSnapshot;
  replan?: {
    dag_revision: number;
    decision_id?: string;
    pruned_task_ids: number[];
  };
  report?: ReportProgress;
}

interface V4ResearchEvent {
  schema_version: 1;
  request_id: string;
  event_key: string;
  state_version?: number;
}

export interface BudgetUpdatedEvent extends V4ResearchEvent {
  type: "budget_updated";
  budget: BudgetSnapshot;
  stage: string;
}

export interface ResearchReplannedEvent extends V4ResearchEvent {
  type: "research_replanned";
  dag_revision: number;
  decision_id: string;
  actions: Record<string, unknown>[];
  scores: Record<string, unknown>[];
  pruned_task_ids: number[];
}

export interface SkillsCatalogResponse {
  enabled: boolean;
  domain_packs: Array<{
    pack_id: string;
    title: string;
    topic_domain: string;
    applicable_when: string;
    verifiers: string[];
    composable_with: string[];
    version: number;
    status: string;
    code_sandbox?: boolean;
  }>;
  format_packs: Array<{
    pack_id: string;
    title: string;
    applicable_when: string;
    outline_sections: string[];
    version: number;
    status: string;
  }>;
  registered_verifiers: string[];
}

export interface ResumeRequest {
  search_api?: string;
}

export interface ResearchStreamEvent {
  type: string;
  v4?: V4ResearchProjection;
  [key: string]: unknown;
}

export interface StreamOptions {
  signal?: AbortSignal;
  apiKey?: string;
  lastEventId?: string;
  onEventId?: (eventId: string) => void;
  onStreamStart?: (metadata: { requestId?: string; runner?: string }) => void;
}

export type ResearchEventHandler = (event: ResearchStreamEvent) => void;

export interface MetricsSnapshot {
  generated_at?: string;
  cache_hit_total?: number;
  cache_miss_total?: number;
  cache_approximate_hit_total?: number;
  cache_approximate_dense_hit_total?: number;
  cache_approximate_sparse_hit_total?: number;
  cache_approximate_hybrid_hit_total?: number;
  counters?: Record<string, number>;
  recent_requests?: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface ResearchTaskPlanItem {
  id?: number;
  title: string;
  intent: string;
  query: string;
  depends_on?: number[];
  status?: string;
  summary?: string;
  sources_summary?: string;
  note_id?: string;
  note_path?: string;
  origin?: string;
  round?: number;
  [key: string]: unknown;
}

export interface ClarifyResponse {
  topic: string;
  needs_clarification?: boolean;
  detected_domains?: string[];
  questions: string[];
  suggested_angles?: string[];
  [key: string]: unknown;
}

export interface EditablePlanResponse {
  request_id: string;
  topic: string;
  status: string;
  phase: string;
  todo_items: ResearchTaskPlanItem[];
  estimated_duration_seconds: number;
  plan_revision?: number;
  v4?: V4ResearchProjection;
  [key: string]: unknown;
}

export interface SectionDrilldownResponse {
  request_id: string;
  section_title: string;
  question: string;
  answer_markdown: string;
  citations?: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface PersistedRequestSummary {
  request_id: string;
  topic: string;
  status?: string;
  phase?: string;
  updated_at?: string;
  completed?: boolean;
  error?: string;
  v4?: V4ResearchProjection;
  [key: string]: unknown;
}

export interface ReplayEventItem {
  id: string;
  data: ResearchStreamEvent;
}

export interface SourceItem {
  sourceId: string;
  title: string;
  url: string;
  snippet: string;
  raw: string;
  domain: string;
  sourceType: string;
  qualityLabel: string;
  freshnessLabel: string;
  publishedAt: string;
  providerCount: number;
}

export interface ReviewIssue {
  task_id?: number | null;
  severity: string;
  check: string;
  message: string;
  source_ids: string[];
  origin?: string;
}

export interface ClaimItem {
  text: string;
  source_ids: string[];
  support_status: string;
}

export interface ToolCallLog {
  eventId: number;
  agent: string;
  tool: string;
  parameters: Record<string, unknown>;
  result: string;
  noteId: string | null;
  notePath: string | null;
  timestamp: number;
}

export interface TodoTaskView {
  id: number;
  title: string;
  intent: string;
  query: string;
  status: string;
  summary: string;
  sourcesSummary: string;
  sourceItems: SourceItem[];
  notices: string[];
  claims: ClaimItem[];
  reviewIssues: ReviewIssue[];
  reviewStatus: string;
  noteId: string | null;
  notePath: string | null;
  toolCalls: ToolCallLog[];
  origin: string;
  round: number;
  atomic_claims?: Array<Record<string, unknown>>;
}

export interface WorkbenchCitationEntry {
  source_id: string;
  task_id?: number | null;
  title: string;
  url: string;
  domain: string;
  snippet: string;
  quality_label: string;
  source_type?: string;
  published_at?: string | null;
  modality?: string;
  anchors?: string[];
  source_origin?: string;
  excerpt?: string;
  page_no?: number | null;
  bbox?: number[] | null;
  anchor_id?: string | null;
  freshness_label?: string;
  derived_from_source_ids?: string[];
  computation_artifact?: Record<string, unknown> | null;
  linked_claims?: Array<{ text: string; nli_label: string; calibrated_confidence?: number }>;
  conflict_ids?: string[];
}

export interface WorkbenchParagraphBinding {
  paragraph_index: number;
  excerpt: string;
  citation_ids: string[];
  conflict_ids: string[];
}

export interface ConflictSide {
  source_id: string;
  task_id: number;
  title: string;
  url: string;
  domain: string;
  source_type: string;
  quality_label: string;
  published_at?: string | null;
  claim_text: string;
  normalized_value?: number | null;
  unit?: string;
  evidence_excerpt?: string;
  authority_rank?: number;
  nli_support_score?: number;
}

export interface EvidenceConflictCluster {
  conflict_id: string;
  entity: string;
  metric: string;
  period: string;
  conflict_type: string;
  severity: string;
  contradiction_score: number;
  sides: ConflictSide[];
  adjudication_status: string;
  recommended_source_id?: string | null;
  adjudication_rationale: string;
  affected_task_ids?: number[];
}

export interface ClaimVerificationSummary {
  total_parent_claims?: number;
  total_atomic_claims?: number;
  entailed_count?: number;
  contradicted_count?: number;
  neutral_count?: number;
  skipped_count?: number;
  atomic_faithfulness_rate?: number;
  conflict_count?: number;
  unresolved_conflict_count?: number;
  backend_used?: string;
  elapsed_ms?: number;
  degraded?: boolean;
}

export interface SkillGovernanceInfo {
  pack_id: string;
  version?: number | string | null;
  maturity?: string | null;
  active_verifiers?: string[];
  code_sandbox_enabled?: boolean;
  nli_verifier_enabled?: boolean;
  mcp_tool_scopes?: Record<string, unknown>;
  secondary_pack_id?: string | null;
  format_pack_id?: string | null;
  routing_method?: string;
  match_score?: number;
  status?: string;
  shadow?: boolean;
}

export interface ChartSpecArtifact {
  chart_id: string;
  title: string;
  library: "echarts" | "vega-lite";
  spec: Record<string, unknown>;
  source_ids: string[];
  caption?: string;
}

export interface VerifierBadge {
  badge_id: string;
  label?: string;
  status: string;
}

export interface ReportSectionBinding {
  section_id: string;
  title: string;
  paragraph_index: number;
}

export interface ReportClaimBinding {
  section_id: string;
  block_id: string;
  claim_id: string;
  paragraph_index: number;
  text_fingerprint: string;
  citation_ids: string[];
  evidence_anchors: { source_id: string; artifact_id?: string; anchor_id?: string }[];
}

export interface ReportWorkbenchPayload {
  citation_index: Record<string, WorkbenchCitationEntry>;
  paragraph_bindings: WorkbenchParagraphBinding[];
  evidence_conflicts: EvidenceConflictCluster[];
  claim_verification_summary: ClaimVerificationSummary;
  skill_governance: SkillGovernanceInfo;
  verifier_badges?: VerifierBadge[];
  charts_spec?: unknown[];
  section_bindings?: ReportSectionBinding[];
  claim_bindings?: ReportClaimBinding[];
  critic_issues?: Record<string, unknown>[];
}

export interface AsyncResearchSubmitResponse {
  request_id: string;
  workflow_run_id: string;
  runner_backend: string;
  status: string;
  tenant_id: string;
  events_url: string;
  runner?: string;
  stream_url?: string;
}

export interface WorkflowSignalResponse {
  request_id: string;
  tenant_id: string;
  runner_backend: string;
  signal_name: string;
  accepted: boolean;
  status?: string;
  detail?: string;
}

export interface RecentResearchView {
  requestId: string;
  topic: string;
  status: string;
  elapsedMs: number;
  searchApi: string;
  reportMarkdown: string;
  todoItems: TodoTaskView[];
  requestMetrics: Record<string, unknown>;
  error?: string;
  canViewContent: boolean;
  canResume: boolean;
  phase?: string;
  updatedAt?: string;
}
