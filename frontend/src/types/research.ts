export interface ResearchRequest {
  topic: string;
  search_api?: string;
  output_format?: string;
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
  [key: string]: unknown;
}

export interface StreamOptions {
  signal?: AbortSignal;
  apiKey?: string;
  lastEventId?: string;
  onEventId?: (eventId: string) => void;
}

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
