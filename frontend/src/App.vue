<template>
  <main class="app-shell" :class="{ expanded: isExpanded }">
    <div class="aurora" aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </div>

    <!-- 初始状态：居中输入卡片 -->
    <div v-if="!isExpanded" class="layout layout-centered">
      <section class="panel panel-form panel-centered">
        <header class="panel-head">
          <div class="logo">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M12 2.5c-.7 0-1.4.2-2 .6L4.6 7C3.6 7.6 3 8.7 3 9.9v4.2c0 1.2.6 2.3 1.6 2.9l5.4 3.9c1.2.8 2.8.8 4 0l5.4-3.9c1-.7 1.6-1.7 1.6-2.9V9.9c0-1.2-.6-2.3-1.6-2.9L14 3.1a3.6 3.6 0 0 0-2-.6Z"
              />
            </svg>
          </div>
          <div>
            <h1>深度研究助手</h1>
            <p>结合多轮智能检索与总结，实时呈现洞见与引用。</p>
          </div>
        </header>

        <form class="form" @submit.prevent="handleSubmit">
          <label class="field">
            <span>研究主题</span>
            <textarea
              v-model="form.topic"
              placeholder="例如：探索多模态模型在 2025 年的关键突破"
              rows="4"
              required
            ></textarea>
          </label>

          <section class="options">
            <label class="field option">
              <span>搜索引擎</span>
              <select v-model="form.searchApi">
                <option value="">沿用后端配置</option>
                <option
                  v-for="option in searchOptions"
                  :key="option"
                  :value="option"
                >
                  {{ option }}
                </option>
              </select>
            </label>
          </section>

          <div class="form-actions">
            <button class="submit" type="submit" :disabled="loading">
              <span class="submit-label">
                <svg
                  v-if="loading"
                  class="spinner"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <circle cx="12" cy="12" r="9" stroke-width="3" />
                </svg>
                {{ loading ? "研究进行中..." : "开始研究" }}
              </span>
            </button>
            <button
              v-if="loading"
              type="button"
              class="secondary-btn"
              @click="cancelResearch"
            >
              取消研究
            </button>
          </div>
        </form>

        <p v-if="error" class="error-chip">
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path
              d="M10 3.2c-.3 0-.6.2-.8.5L3.4 15c-.4.7.1 1.6.8 1.6h11.6c.7 0 1.2-.9.8-1.6L10.8 3.7c-.2-.3-.5-.5-.8-.5Zm0 4.3c.4 0 .7.3.7.7v4c0 .4-.3.7-.7.7s-.7-.3-.7-.7V8.2c0-.4.3-.7.7-.7Zm0 6.6a1 1 0 1 1 0 2 1 1 0 0 1 0-2Z"
            />
          </svg>
          {{ error }}
        </p>
        <p v-else-if="loading" class="hint muted">
          正在收集线索与证据，实时进展见右侧区域。
        </p>
      </section>
    </div>

    <!-- 全屏状态：左右分栏布局 -->
    <div v-else class="layout layout-fullscreen">
      <!-- 左侧：研究信息 -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <button class="back-btn" @click="goBack" :disabled="loading">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path d="M19 12H5M12 19l-7-7 7-7" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            返回
          </button>
          <h2>🔍 深度研究助手</h2>
        </div>

        <div class="research-info">
          <div class="info-item">
            <label>研究主题</label>
            <p class="topic-display">{{ form.topic }}</p>
          </div>

          <div class="info-item" v-if="form.searchApi">
            <label>搜索引擎</label>
            <p>{{ form.searchApi }}</p>
          </div>

          <div class="info-item" v-if="totalTasks > 0">
            <label>研究进度</label>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: `${(completedTasks / totalTasks) * 100}%` }"></div>
            </div>
            <p class="progress-text">{{ completedTasks }} / {{ totalTasks }} 任务完成</p>
          </div>
        </div>

        <div class="sidebar-actions">
          <button class="new-research-btn" @click="startNewResearch">
            <svg viewBox="0 0 24 24" width="18" height="18">
              <path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>
            </svg>
            开始新研究
          </button>
        </div>

        <section class="sidebar-history">
          <div class="sidebar-history-head">
            <h3>最近研究历史</h3>
            <span class="history-count">{{ recentResearchHistory.length }} 条</span>
          </div>
          <ul v-if="recentResearchHistory.length" class="sidebar-history-list">
            <li
              v-for="item in recentResearchHistory"
              :key="item.requestId"
              :class="['sidebar-history-item', { selected: selectedHistoryRequestId === item.requestId }]"
            >
              <div class="sidebar-history-main">
                <p class="sidebar-history-topic" :title="item.topic">
                  {{ item.topic }}
                </p>
                <div class="sidebar-history-meta">
                  <span class="history-status-chip" :class="item.status">
                    {{ formatRequestStatus(item.status) }}
                  </span>
                  <span>{{ formatElapsed(item.elapsedMs) }}</span>
                </div>
              </div>
              <div class="sidebar-history-actions">
                <button
                  type="button"
                  class="history-view-btn"
                  @click="viewHistoryResearch(item)"
                  :disabled="loading || !item.canViewContent"
                >
                  查看内容
                </button>
                <button
                  type="button"
                  class="history-repeat-btn"
                  @click="repeatResearch(item)"
                  :disabled="loading"
                >
                  再次研究
                </button>
              </div>
            </li>
          </ul>
          <p v-else class="sidebar-history-empty">暂无历史研究记录</p>
        </section>
      </aside>

      <!-- 右侧：研究结果 -->
      <section
        ref="resultPanelRef"
        class="panel panel-result"
        v-if="todoTasks.length || reportMarkdown || progressLogs.length"
      >
        <header class="status-bar">
          <div class="status-main">
            <div class="status-chip" :class="{ active: loading }">
              <span class="dot"></span>
              {{ loading ? "研究进行中" : "研究流程完成" }}
            </div>
            <span class="status-meta">
              任务进度：{{ completedTasks }} / {{ totalTasks || todoTasks.length || 1 }}
              · 阶段记录 {{ progressLogs.length }} 条
            </span>
          </div>
          <div class="status-controls">
            <button class="secondary-btn" @click="toggleLogsVisibility">
              {{ logsCollapsed ? "展开流程" : "收起流程" }}
            </button>
          </div>
        </header>

        <section v-if="hasMetrics" class="metrics-strip">
          <div class="metric-card">
            <span class="metric-label">请求状态</span>
            <strong class="metric-value">{{ requestMetricStatus }}</strong>
          </div>
          <div class="metric-card">
            <span class="metric-label">总耗时</span>
            <strong class="metric-value">{{ formatElapsed(requestMetricElapsedMs) }}</strong>
          </div>
          <div class="metric-card">
            <span class="metric-label">搜索缓存</span>
            <strong class="metric-value">
              {{ requestCacheHits }} 命中 / {{ requestCacheMisses }} 未命中
            </strong>
            <span class="metric-hint">
              全局累计：{{ globalCacheHits }} 命中 / {{ globalCacheMisses }} 未命中
            </span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Token / 成本</span>
            <strong class="metric-value">
              {{ requestTotalTokens }} / {{ formatCost(requestEstimatedCost) }}
            </strong>
            <span class="metric-hint">来源：{{ requestTokenSource }}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">聚合成功率</span>
            <strong class="metric-value">{{ aggregateSuccessRateText }}</strong>
            <span class="metric-hint">Fallback {{ aggregateFallbackTotal }} 次</span>
          </div>
        </section>

        <section v-if="todoTasks.length" class="planning-summary">
          <div class="planning-summary-head">
            <div>
              <h3>任务规划概览</h3>
              <p>始终按结构化任务展示当前主题的规划结果。</p>
            </div>
            <span class="planning-count">{{ todoTasks.length }} 个任务</span>
          </div>
          <div class="planning-grid">
            <article
              v-for="task in todoTasks"
              :key="`plan-${task.id}`"
              class="planning-card"
            >
              <div class="planning-card-head">
                <strong>{{ task.title }}</strong>
                <span class="task-status" :class="task.status">
                  {{ formatTaskStatus(task.status) }}
                </span>
              </div>
              <p class="planning-intent">{{ task.intent }}</p>
              <p class="planning-query">查询：{{ task.query || form.topic }}</p>
              <p v-if="task.noteId" class="planning-note">笔记：{{ task.noteId }}</p>
            </article>
          </div>
        </section>

        <section
          v-if="progressLogs.length"
          ref="timelineSectionRef"
          class="timeline-section"
        >
          <div class="timeline-head">
            <div>
              <h3>流程记录</h3>
              <p>展示规划、搜索、总结和报告阶段的实时过程。</p>
            </div>
            <span class="timeline-count">{{ progressLogs.length }} 条</span>
          </div>

          <div
            ref="timelineWrapperRef"
            class="timeline-wrapper"
            v-show="!logsCollapsed"
          >
            <transition-group name="timeline" tag="ul" class="timeline">
              <li v-for="(log, index) in progressLogs" :key="`${log}-${index}`">
                <span class="timeline-node"></span>
                <p>{{ log }}</p>
              </li>
            </transition-group>
          </div>
        </section>

        <div class="tasks-section" v-if="todoTasks.length">
          <aside class="tasks-list">
            <h3>任务清单</h3>
            <ul>
              <li
                v-for="task in todoTasks"
                :key="task.id"
                :class="['task-item', { active: task.id === activeTaskId, completed: task.status === 'completed' }]"
              >
                <button
                  type="button"
                  class="task-button"
                  @click="activeTaskId = task.id"
                >
                  <span class="task-title">{{ task.title }}</span>
                  <span class="task-status" :class="task.status">
                    {{ formatTaskStatus(task.status) }}
                  </span>
                </button>
                <p class="task-intent">{{ task.intent }}</p>
              </li>
            </ul>
          </aside>

          <article class="task-detail" v-if="currentTask">
            <header class="task-header">
              <div>
                <h3>{{ currentTaskTitle || "当前任务" }}</h3>
                <p class="muted" v-if="currentTaskIntent">
                  {{ currentTaskIntent }}
                </p>
              </div>
              <div class="task-chip-group">
                <span class="task-label">查询：{{ currentTaskQuery || "" }}</span>
                <span
                  v-if="currentTaskNoteId"
                  class="task-label note-chip"
                  :title="currentTaskNoteId"
                >
                  笔记：{{ currentTaskNoteId }}
                </span>
                <span
                  v-if="currentTaskNotePath"
                  class="task-label note-chip path-chip"
                  :title="currentTaskNotePath"
                >
                  <span class="path-label">路径：</span>
                  <span class="path-text">{{ currentTaskNotePath }}</span>
                  <button
                    class="chip-action"
                    type="button"
                    @click="copyNotePath(currentTaskNotePath)"
                  >
                    复制
                  </button>
                </span>
              </div>
            </header>

            <section v-if="currentTask && currentTask.notices.length" class="task-notices">
              <h4>系统提示</h4>
              <ul>
                <li v-for="(notice, idx) in currentTask.notices" :key="`${notice}-${idx}`">
                  {{ notice }}
                </li>
              </ul>
            </section>

            <section
              class="sources-block"
              :class="{ 'block-highlight': sourcesHighlight }"
            >
              <h3>最新来源</h3>
              <template v-if="currentTaskSources.length">
                <ul class="sources-list">
                  <li
                    v-for="(item, index) in currentTaskSources"
                    :key="`${item.title}-${index}`"
                    class="source-item"
                  >
                    <a
                      class="source-link"
                      :href="item.url || '#'"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {{ item.title || item.url || `来源 ${index + 1}` }}
                    </a>
                    <div v-if="item.snippet || item.raw" class="source-tooltip">
                      <p v-if="item.snippet">{{ item.snippet }}</p>
                      <p v-if="item.raw" class="muted-text">{{ item.raw }}</p>
                    </div>
                  </li>
                </ul>
              </template>
              <p v-else class="muted">暂无可用来源</p>
            </section>

            <section
              class="summary-block"
              :class="{ 'block-highlight': summaryHighlight }"
            >
              <h3>任务总结</h3>
              <div
                class="markdown-block"
                v-html="currentTaskSummaryHtml"
              ></div>
            </section>

            <section
              class="tools-block"
              :class="{ 'block-highlight': toolHighlight }"
              v-if="currentTaskToolCalls.length"
            >
              <h3>工具调用记录</h3>
              <ul class="tool-list">
                <li
                  v-for="entry in currentTaskToolCalls"
                  :key="`${entry.eventId}-${entry.timestamp}`"
                  class="tool-entry"
                >
                  <div class="tool-entry-header">
                    <span class="tool-entry-title">
                      #{{ entry.eventId }} {{ entry.agent }} → {{ entry.tool }}
                    </span>
                    <span
                      v-if="entry.noteId"
                      class="tool-entry-note"
                    >
                      笔记：{{ entry.noteId }}
                    </span>
                  </div>
                  <p v-if="entry.notePath" class="tool-entry-path">
                    笔记路径：
                    <button
                      class="link-btn"
                      type="button"
                      @click="copyNotePath(entry.notePath)"
                    >
                      复制
                    </button>
                    <span class="path-text">{{ entry.notePath }}</span>
                  </p>
                  <p class="tool-subtitle">参数</p>
                  <pre class="tool-pre">{{ formatToolParameters(entry.parameters) }}</pre>
                  <template v-if="entry.result">
                    <p class="tool-subtitle">执行结果</p>
                    <pre class="tool-pre">{{ formatToolResult(entry.result) }}</pre>
                  </template>
                </li>
              </ul>
            </section>
          </article>

          <article class="task-detail" v-else>
            <p class="muted">等待任务规划或执行结果。</p>
          </article>
        </div>

        <div
          v-if="reportMarkdown"
          class="report-block"
          :class="{ 'block-highlight': reportHighlight }"
        >
          <h3>最终报告</h3>
          <div
            class="markdown-block"
            v-html="reportMarkdownHtml"
          ></div>
        </div>
      </section>

    </div>
  </main>
</template>

<script lang="ts" setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { marked } from "marked";

import {
  fetchMetricsSnapshot,
  runResearchStream,
  type ResearchStreamEvent
} from "./services/api";

interface SourceItem {
  title: string;
  url: string;
  snippet: string;
  raw: string;
}

interface ToolCallLog {
  eventId: number;
  agent: string;
  tool: string;
  parameters: Record<string, unknown>;
  result: string;
  noteId: string | null;
  notePath: string | null;
  timestamp: number;
}

interface TodoTaskView {
  id: number;
  title: string;
  intent: string;
  query: string;
  status: string;
  summary: string;
  sourcesSummary: string;
  sourceItems: SourceItem[];
  notices: string[];
  noteId: string | null;
  notePath: string | null;
  toolCalls: ToolCallLog[];
}

interface RecentResearchView {
  requestId: string;
  topic: string;
  status: string;
  elapsedMs: number;
  searchApi: string;
  reportMarkdown: string;
  todoItems: TodoTaskView[];
  requestMetrics: UnknownRecord;
  canViewContent: boolean;
}

type UnknownRecord = Record<string, unknown>;

const form = reactive({
  topic: "",
  searchApi: ""
});

const loading = ref(false);
const error = ref("");
const progressLogs = ref<string[]>([]);
const logsCollapsed = ref(false);
const isExpanded = ref(false);

const todoTasks = ref<TodoTaskView[]>([]);
const activeTaskId = ref<number | null>(null);
const reportMarkdown = ref("");
const latestRequestMetrics = ref<UnknownRecord | null>(null);
const latestAggregateMetrics = ref<UnknownRecord | null>(null);
const selectedHistoryRequestId = ref<string | null>(null);

const summaryHighlight = ref(false);
const sourcesHighlight = ref(false);
const reportHighlight = ref(false);
const toolHighlight = ref(false);
const resultPanelRef = ref<HTMLElement | null>(null);
const timelineSectionRef = ref<HTMLElement | null>(null);
const timelineWrapperRef = ref<HTMLElement | null>(null);

let currentController: AbortController | null = null;
let globalMetricsTimer: number | null = null;

const searchOptions = [
  "advanced",
  "duckduckgo",
  "tavily",
  "perplexity",
  "searxng"
];

const TASK_STATUS_LABEL: Record<string, string> = {
  pending: "待执行",
  in_progress: "进行中",
  completed: "已完成",
  skipped: "已跳过"
};
const REQUEST_STATUS_LABEL: Record<string, string> = {
  success: "成功",
  partial_success: "部分成功",
  failed: "失败",
  in_progress: "进行中"
};

function formatTaskStatus(status: string): string {
  return TASK_STATUS_LABEL[status] ?? status;
}

function formatRequestStatus(status: string): string {
  return REQUEST_STATUS_LABEL[status] ?? status;
}

const totalTasks = computed(() => todoTasks.value.length);
const completedTasks = computed(() =>
  todoTasks.value.filter((task) => task.status === "completed").length
);

const currentTask = computed(() => {
  if (activeTaskId.value !== null) {
    return todoTasks.value.find((task) => task.id === activeTaskId.value) ?? null;
  }
  return todoTasks.value[0] ?? null;
});

const currentTaskSources = computed(() => currentTask.value?.sourceItems ?? []);
const currentTaskSummary = computed(() => currentTask.value?.summary ?? "");
const currentTaskTitle = computed(() => currentTask.value?.title ?? "");
const currentTaskIntent = computed(() => currentTask.value?.intent ?? "");
const currentTaskQuery = computed(() => currentTask.value?.query ?? "");
const currentTaskNoteId = computed(() => currentTask.value?.noteId ?? "");
const currentTaskNotePath = computed(() => currentTask.value?.notePath ?? "");
const currentTaskToolCalls = computed(
  () => currentTask.value?.toolCalls ?? []
);
const hasMetrics = computed(
  () => latestRequestMetrics.value !== null || latestAggregateMetrics.value !== null
);
const currentTaskSummaryHtml = computed(() =>
  renderMarkdownBlock(currentTaskSummary.value || "暂无可用信息")
);
const reportMarkdownHtml = computed(() =>
  renderMarkdownBlock(reportMarkdown.value || "暂无生成的报告")
);

function getNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function getString(value: unknown, fallback = "-"): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMarkdownBlock(value: string): string {
  const text = typeof value === "string" ? value.trim() : "";
  if (!text) {
    return "<p>暂无可用信息</p>";
  }

  return marked.parse(escapeHtml(text), {
    breaks: true,
    gfm: true
  }) as string;
}

const requestMetricElapsedMs = computed(() =>
  getNumber(latestRequestMetrics.value?.elapsed_ms)
);
const requestMetricStatus = computed(() =>
  getString(latestRequestMetrics.value?.status, loading.value ? "in_progress" : "unknown")
);
const requestEstimatedCost = computed(() =>
  getNumber(latestRequestMetrics.value?.estimated_cost)
);
const requestTotalTokens = computed(() =>
  getNumber(latestRequestMetrics.value?.total_tokens)
);
const requestTokenSource = computed(() =>
  getString(latestRequestMetrics.value?.token_source, "unknown")
);
const requestCacheHits = computed(() =>
  getNumber(latestRequestMetrics.value?.cache_hits)
);
const requestCacheMisses = computed(() =>
  getNumber(latestRequestMetrics.value?.cache_misses)
);
const globalCacheHits = computed(() => {
  const direct = getNumber(latestAggregateMetrics.value?.cache_hit_total);
  if (direct) {
    return direct;
  }
  return getNumber(
    ensureRecord(latestAggregateMetrics.value?.counters).cache_hit_total
  );
});
const globalCacheMisses = computed(() => {
  const direct = getNumber(latestAggregateMetrics.value?.cache_miss_total);
  if (direct) {
    return direct;
  }
  return getNumber(
    ensureRecord(latestAggregateMetrics.value?.counters).cache_miss_total
  );
});
const recentResearchHistory = computed<RecentResearchView[]>(() => {
  const items = latestAggregateMetrics.value?.recent_requests;
  if (!Array.isArray(items)) {
    return [];
  }

  return items
    .map((entry) => ensureRecord(entry))
    .map((entry, index) => {
      const topic = getString(entry.topic, "");
      if (!topic) {
        return null;
      }
      const historyTasks = mapHistoryTasks(entry.todo_items);
      const reportMarkdown =
        typeof entry.report_markdown === "string" ? entry.report_markdown.trim() : "";

      return {
        requestId: getString(entry.request_id, `recent-${index}`),
        topic,
        status: getString(entry.status, "unknown"),
        elapsedMs: getNumber(entry.elapsed_ms),
        searchApi: getString(entry.search_api, ""),
        reportMarkdown,
        todoItems: historyTasks,
        requestMetrics: entry,
        canViewContent: Boolean(reportMarkdown) || historyTasks.length > 0
      };
    })
    .filter((entry): entry is RecentResearchView => entry !== null);
});
const aggregateSuccessRateText = computed(() => {
  const rate = getNumber(latestAggregateMetrics.value?.success_rate);
  return `${(rate * 100).toFixed(1)}%`;
});
const aggregateFallbackTotal = computed(() =>
  getNumber(
    ensureRecord(latestAggregateMetrics.value?.counters).fallback_trigger_total
  )
);

const pulse = (flag: typeof summaryHighlight) => {
  flag.value = false;
  requestAnimationFrame(() => {
    flag.value = true;
    window.setTimeout(() => {
      flag.value = false;
    }, 1200);
  });
};

function parseSources(raw: string): SourceItem[] {
  if (!raw) {
    return [];
  }

  const items: SourceItem[] = [];
  const lines = raw.split("\n");

  let current: SourceItem | null = null;
  const truncate = (value: string, max = 360) => {
    const trimmed = value.trim();
    return trimmed.length > max ? `${trimmed.slice(0, max)}…` : trimmed;
  };

  const flush = () => {
    if (!current) {
      return;
    }
    const normalized: SourceItem = {
      title: current.title?.trim() || "",
      url: current.url?.trim() || "",
      snippet: current.snippet ? truncate(current.snippet) : "",
      raw: current.raw ? truncate(current.raw, 420) : ""
    };

    if (
      normalized.title ||
      normalized.url ||
      normalized.snippet ||
      normalized.raw
    ) {
      if (!normalized.title && normalized.url) {
        normalized.title = normalized.url;
      }
      items.push(normalized);
    }
    current = null;
  };

  const ensureCurrent = () => {
    if (!current) {
      current = { title: "", url: "", snippet: "", raw: "" };
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    if (/^\*/.test(trimmed) && trimmed.includes(" : ")) {
      flush();
      const withoutBullet = trimmed.replace(/^\*\s*/, "");
      const [titlePart, urlPart] = withoutBullet.split(" : ");
      current = {
        title: titlePart?.trim() || "",
        url: urlPart?.trim() || "",
        snippet: "",
        raw: ""
      };
      continue;
    }

    if (/^(Source|信息来源)\s*:/.test(trimmed)) {
      flush();
      const [, titlePart = ""] = trimmed.split(/:\s*(.+)/);
      current = {
        title: titlePart.trim(),
        url: "",
        snippet: "",
        raw: ""
      };
      continue;
    }

    if (/^URL\s*:/.test(trimmed)) {
      ensureCurrent();
      const [, urlPart = ""] = trimmed.split(/:\s*(.+)/);
      current!.url = urlPart.trim();
      continue;
    }

    if (
      /^(Most relevant content from source|信息内容)\s*:/.test(trimmed)
    ) {
      ensureCurrent();
      const [, contentPart = ""] = trimmed.split(/:\s*(.+)/);
      current!.snippet = contentPart.trim();
      continue;
    }

    if (
      /^(Full source content limited to|信息内容限制为)\s*:/.test(trimmed)
    ) {
      ensureCurrent();
      const [, rawPart = ""] = trimmed.split(/:\s*(.+)/);
      current!.raw = rawPart.trim();
      continue;
    }

    if (/^https?:\/\//.test(trimmed)) {
      ensureCurrent();
      if (!current!.url) {
        current!.url = trimmed;
        continue;
      }
    }

    ensureCurrent();
    current!.raw = current!.raw ? `${current!.raw}\n${trimmed}` : trimmed;
  }

  flush();
  return items;
}

function mapHistoryTasks(value: unknown): TodoTaskView[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item, index) => {
    const payload = ensureRecord(item);
    const rawId =
      typeof payload.id === "number"
        ? payload.id
        : typeof payload.id === "string"
        ? Number(payload.id)
        : index + 1;
    const id = Number.isFinite(rawId) ? Number(rawId) : index + 1;
    const sourcesSummary =
      typeof payload.sources_summary === "string" ? payload.sources_summary.trim() : "";

    return {
      id,
      title:
        typeof payload.title === "string" && payload.title.trim()
          ? payload.title.trim()
          : `任务${id}`,
      intent:
        typeof payload.intent === "string" && payload.intent.trim()
          ? payload.intent.trim()
          : "历史任务",
      query:
        typeof payload.query === "string" && payload.query.trim()
          ? payload.query.trim()
          : "",
      status:
        typeof payload.status === "string" && payload.status.trim()
          ? payload.status.trim()
          : "completed",
      summary:
        typeof payload.summary === "string" && payload.summary.trim()
          ? payload.summary.trim()
          : "暂无可用信息",
      sourcesSummary,
      sourceItems: parseSources(sourcesSummary),
      notices: Array.isArray(payload.notices)
        ? payload.notices.filter((notice): notice is string => typeof notice === "string")
        : [],
      noteId: extractOptionalString(payload.note_id),
      notePath: extractOptionalString(payload.note_path),
      toolCalls: []
    };
  });
}

function extractOptionalString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function ensureRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function applyNoteMetadata(
  task: TodoTaskView,
  payload: Record<string, unknown>
): void {
  const noteId = extractOptionalString(payload.note_id);
  if (noteId) {
    task.noteId = noteId;
  }
  const notePath = extractOptionalString(payload.note_path);
  if (notePath) {
    task.notePath = notePath;
  }
}

function formatToolParameters(parameters: Record<string, unknown>): string {
  try {
    return JSON.stringify(parameters, null, 2);
  } catch (error) {
    console.warn("无法格式化工具参数", error, parameters);
    return Object.entries(parameters)
      .map(([key, value]) => `${key}: ${String(value)}`)
      .join("\n");
  }
}

function formatToolResult(result: string): string {
  const trimmed = result.trim();
  const limit = 900;
  if (trimmed.length > limit) {
    return `${trimmed.slice(0, limit)}…`;
  }
  return trimmed;
}

async function copyNotePath(path: string | null | undefined) {
  if (!path) {
    return;
  }

  try {
    await navigator.clipboard.writeText(path);
    progressLogs.value.push(`已复制笔记路径：${path}`);
  } catch (error) {
    console.warn("无法直接复制到剪贴板", error);
    window.prompt("复制以下笔记路径", path);
    progressLogs.value.push("请手动复制笔记路径");
  }
}

async function refreshGlobalMetrics(signal?: AbortSignal) {
  try {
    const snapshot = await fetchMetricsSnapshot({ signal });
    latestAggregateMetrics.value = {
      ...ensureRecord(latestAggregateMetrics.value),
      ...ensureRecord(snapshot)
    };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return;
    }
    console.warn("获取全局缓存指标失败", error);
  }
}

function resetWorkflowState() {
  todoTasks.value = [];
  activeTaskId.value = null;
  reportMarkdown.value = "";
  progressLogs.value = [];
  latestRequestMetrics.value = null;
   selectedHistoryRequestId.value = null;
  summaryHighlight.value = false;
  sourcesHighlight.value = false;
  reportHighlight.value = false;
  toolHighlight.value = false;
  logsCollapsed.value = false;
}

async function repeatResearch(item: RecentResearchView) {
  if (loading.value) {
    return;
  }

  form.topic = item.topic;
  form.searchApi = item.searchApi;
  await handleSubmit();
}

function viewHistoryResearch(item: RecentResearchView) {
  if (loading.value) {
    return;
  }

  resetWorkflowState();
  selectedHistoryRequestId.value = item.requestId;
  form.topic = item.topic;
  form.searchApi = item.searchApi;
  todoTasks.value = item.todoItems.map((task) => ({
    ...task,
    sourceItems: [...task.sourceItems],
    notices: [...task.notices],
    toolCalls: []
  }));
  activeTaskId.value = todoTasks.value[0]?.id ?? null;
  reportMarkdown.value = item.reportMarkdown || "该历史研究未保存最终报告";
  latestRequestMetrics.value = { ...item.requestMetrics };
  progressLogs.value = [`已载入历史研究：${item.topic}`];
  isExpanded.value = true;
}

function findTask(taskId: unknown): TodoTaskView | undefined {
  const numeric =
    typeof taskId === "number"
      ? taskId
      : typeof taskId === "string"
      ? Number(taskId)
      : NaN;
  if (Number.isNaN(numeric)) {
    return undefined;
  }
  return todoTasks.value.find((task) => task.id === numeric);
}

function upsertTaskMetadata(task: TodoTaskView, payload: Record<string, unknown>) {
  if (typeof payload.title === "string" && payload.title.trim()) {
    task.title = payload.title.trim();
  }
  if (typeof payload.intent === "string" && payload.intent.trim()) {
    task.intent = payload.intent.trim();
  }
  if (typeof payload.query === "string" && payload.query.trim()) {
    task.query = payload.query.trim();
  }
}

function formatElapsed(value: number): string {
  if (!value) {
    return "-";
  }
  return `${value.toFixed(0)} ms`;
}

function formatCost(value: number): string {
  if (!value) {
    return "$0.000000";
  }
  return `$${value.toFixed(6)}`;
}

function formatStageLabel(stage: string): string {
  const labels: Record<string, string> = {
    planning: "规划",
    search: "搜索",
    summarization: "总结",
    report: "报告"
  };
  return labels[stage] ?? stage;
}

async function toggleLogsVisibility() {
  const nextCollapsed = !logsCollapsed.value;
  logsCollapsed.value = nextCollapsed;

  await nextTick();

  const panel = resultPanelRef.value;
  const timelineSection = timelineSectionRef.value;
  if (!panel || !timelineSection) {
    return;
  }

  const targetTop = Math.max(0, timelineSection.offsetTop - 24);

  if (nextCollapsed) {
    const maxScrollTop = Math.max(0, panel.scrollHeight - panel.clientHeight);
    const safeScrollTop = Math.min(targetTop, maxScrollTop);
    if (panel.scrollTop > safeScrollTop) {
      panel.scrollTo({
        top: safeScrollTop,
        behavior: "smooth"
      });
    }
    return;
  }

  panel.scrollTo({
    top: targetTop,
    behavior: "smooth"
  });
}

const handleSubmit = async () => {
  if (!form.topic.trim()) {
    error.value = "请输入研究主题";
    return;
  }

  if (currentController) {
    currentController.abort();
    currentController = null;
  }

  loading.value = true;
  error.value = "";
  isExpanded.value = true;
  resetWorkflowState();

  const controller = new AbortController();
  currentController = controller;

  const payload = {
    topic: form.topic.trim(),
    search_api: form.searchApi || undefined
  };

  try {
    await runResearchStream(
      payload,
      (event: ResearchStreamEvent) => {
        if (event.type === "status") {
          const message =
            typeof event.message === "string" && event.message.trim()
              ? event.message
              : "流程状态更新";
          progressLogs.value.push(message);

          const payload = event as Record<string, unknown>;
          const task = findTask(payload.task_id);
          if (task && message) {
            task.notices.push(message);
            applyNoteMetadata(task, payload);
          }
          return;
        }

        if (event.type === "todo_list") {
          const tasks = Array.isArray(event.tasks)
            ? (event.tasks as Record<string, unknown>[])
            : [];

          todoTasks.value = tasks.map((item, index) => {
            const rawId =
              typeof item.id === "number"
                ? item.id
                : typeof item.id === "string"
                ? Number(item.id)
                : index + 1;
            const id = Number.isFinite(rawId) ? Number(rawId) : index + 1;
            const noteId =
              typeof item.note_id === "string" && item.note_id.trim()
                ? item.note_id.trim()
                : null;
            const notePath =
              typeof item.note_path === "string" && item.note_path.trim()
                ? item.note_path.trim()
                : null;

            return {
              id,
              title:
                typeof item.title === "string" && item.title.trim()
                  ? item.title.trim()
                  : `任务${id}`,
              intent:
                typeof item.intent === "string" && item.intent.trim()
                  ? item.intent.trim()
                  : "探索与主题相关的关键信息",
              query:
                typeof item.query === "string" && item.query.trim()
                  ? item.query.trim()
                  : form.topic.trim(),
              status:
                typeof item.status === "string" && item.status.trim()
                  ? item.status.trim()
                  : "pending",
              summary: "",
              sourcesSummary: "",
              sourceItems: [],
              notices: [],
              noteId,
              notePath,
              toolCalls: []
            } as TodoTaskView;
          });

          if (todoTasks.value.length) {
            activeTaskId.value = todoTasks.value[0].id;
            progressLogs.value.push("已生成任务清单");
          } else {
            progressLogs.value.push("未生成任务清单，使用默认任务继续");
          }
          return;
        }

        if (event.type === "task_status") {
          const payload = event as Record<string, unknown>;
          const task = findTask(event.task_id);
          if (!task) {
            return;
          }

          upsertTaskMetadata(task, payload);
          applyNoteMetadata(task, payload);
          const status =
            typeof event.status === "string" && event.status.trim()
              ? event.status.trim()
              : task.status;
          task.status = status;

          if (status === "in_progress") {
            task.summary = "";
            task.sourcesSummary = "";
            task.sourceItems = [];
            task.notices = [];
            activeTaskId.value = task.id;
            progressLogs.value.push(`开始执行任务：${task.title}`);
          } else if (status === "completed") {
            if (typeof event.summary === "string" && event.summary.trim()) {
              task.summary = event.summary.trim();
            }
            if (
              typeof event.sources_summary === "string" &&
              event.sources_summary.trim()
            ) {
              task.sourcesSummary = event.sources_summary.trim();
              task.sourceItems = parseSources(task.sourcesSummary);
            }
            progressLogs.value.push(`完成任务：${task.title}`);
            if (activeTaskId.value === task.id) {
              pulse(summaryHighlight);
              pulse(sourcesHighlight);
            }
          } else if (status === "skipped") {
            progressLogs.value.push(`任务跳过：${task.title}`);
          }
          return;
        }

        if (event.type === "sources") {
          const payload = event as Record<string, unknown>;
          const task = findTask(event.task_id);
          if (!task) {
            return;
          }

          const textCandidates = [
            payload.latest_sources,
            payload.sources_summary,
            payload.raw_context
          ];
          const latestText = textCandidates
            .map((value) => (typeof value === "string" ? value.trim() : ""))
            .find((value) => value);

          if (latestText) {
            task.sourcesSummary = latestText;
            task.sourceItems = parseSources(latestText);
            if (activeTaskId.value === task.id) {
              pulse(sourcesHighlight);
            }
            progressLogs.value.push(`已更新任务来源：${task.title}`);
          }

          if (typeof payload.backend === "string") {
            progressLogs.value.push(
              `当前使用搜索后端：${payload.backend}`
            );
          }

          applyNoteMetadata(task, payload);

          return;
        }

        if (event.type === "task_summary_chunk") {
          const payload = event as Record<string, unknown>;
          const task = findTask(event.task_id);
          if (!task) {
            return;
          }
          const chunk =
            typeof event.content === "string" ? event.content : "";
          task.summary += chunk;
          applyNoteMetadata(task, payload);
          if (activeTaskId.value === task.id) {
            pulse(summaryHighlight);
          }
          return;
        }

        if (event.type === "tool_call") {
          const payload = event as Record<string, unknown>;
          const eventId =
            typeof payload.event_id === "number"
              ? payload.event_id
              : Date.now();
          const agent =
            typeof payload.agent === "string" && payload.agent.trim()
              ? payload.agent.trim()
              : "Agent";
          const tool =
            typeof payload.tool === "string" && payload.tool.trim()
              ? payload.tool.trim()
              : "tool";
          const parameters = ensureRecord(payload.parameters);
          const result =
            typeof payload.result === "string" ? payload.result : "";
          const noteId = extractOptionalString(payload.note_id);
          const notePath = extractOptionalString(payload.note_path);

          const task = findTask(payload.task_id);
          if (task) {
            task.toolCalls.push({
              eventId,
              agent,
              tool,
              parameters,
              result,
              noteId,
              notePath,
              timestamp: Date.now()
            });
            if (noteId) {
              task.noteId = noteId;
            }
            if (notePath) {
              task.notePath = notePath;
            }
            const logSummary = noteId
              ? `${agent} 调用了 ${tool}（任务 ${task.id}，笔记 ${noteId}）`
              : `${agent} 调用了 ${tool}（任务 ${task.id}）`;
            progressLogs.value.push(logSummary);
            if (activeTaskId.value === task.id) {
              pulse(toolHighlight);
            }
          } else {
            progressLogs.value.push(`${agent} 调用了 ${tool}`);
          }
          return;
        }

        if (event.type === "stage_started") {
          const payload = event as UnknownRecord;
          const stage = formatStageLabel(getString(payload.stage, "stage"));
          const taskId = getNumber(payload.task_id);
          progressLogs.value.push(
            taskId ? `阶段开始：${stage}（任务 ${taskId}）` : `阶段开始：${stage}`
          );
          return;
        }

        if (event.type === "stage_completed") {
          const payload = event as UnknownRecord;
          const stage = formatStageLabel(getString(payload.stage, "stage"));
          const taskId = getNumber(payload.task_id);
          const elapsed = formatElapsed(getNumber(payload.elapsed_ms));
          const status = getString(payload.status, "unknown");
          progressLogs.value.push(
            taskId
              ? `阶段完成：${stage}（任务 ${taskId}，${status}，${elapsed}）`
              : `阶段完成：${stage}（${status}，${elapsed}）`
          );
          return;
        }

        if (event.type === "fallback_triggered") {
          const payload = event as UnknownRecord;
          progressLogs.value.push(
            `触发兜底流程：${getString(payload.reason, "unknown")}`
          );
          return;
        }

        if (event.type === "degraded_response") {
          const payload = event as UnknownRecord;
          progressLogs.value.push(
            `降级响应：${getString(payload.reason, "unknown")}`
          );
          return;
        }

        if (event.type === "metrics_snapshot") {
          const payload = event as UnknownRecord;
          latestRequestMetrics.value = ensureRecord(payload.request_metrics);
          const nextAggregate = ensureRecord(payload.aggregate_metrics);
          const previousAggregate = ensureRecord(latestAggregateMetrics.value);
          if (
            !Array.isArray(nextAggregate.recent_requests) &&
            Array.isArray(previousAggregate.recent_requests)
          ) {
            nextAggregate.recent_requests = previousAggregate.recent_requests;
          }
          latestAggregateMetrics.value = nextAggregate;
          return;
        }

        if (event.type === "final_report") {
          const report =
            typeof event.report === "string" && event.report.trim()
              ? event.report.trim()
              : "";
          reportMarkdown.value = report || "报告生成失败，未获得有效内容";
          pulse(reportHighlight);
          progressLogs.value.push("最终报告已生成");
          return;
        }

        if (event.type === "error") {
          const detail =
            typeof event.detail === "string" && event.detail.trim()
              ? event.detail
              : "研究过程中发生错误";
          error.value = detail;
          progressLogs.value.push("研究失败，已停止流程");
        }
      },
      { signal: controller.signal }
    );

    if (!reportMarkdown.value) {
      reportMarkdown.value = "暂无生成的报告";
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      progressLogs.value.push("已取消当前研究任务");
    } else {
      error.value = err instanceof Error ? err.message : "请求失败";
    }
  } finally {
    loading.value = false;
    void refreshGlobalMetrics();
    if (currentController === controller) {
      currentController = null;
    }
  }
};

const cancelResearch = () => {
  if (!loading.value || !currentController) {
    return;
  }
  progressLogs.value.push("正在尝试取消当前研究任务…");
  currentController.abort();
};

const goBack = () => {
  if (loading.value) {
    return; // 研究进行中不允许返回
  }
  isExpanded.value = false;
};

const startNewResearch = () => {
  if (loading.value) {
    cancelResearch();
  }
  resetWorkflowState();
  isExpanded.value = false;
  form.topic = "";
  form.searchApi = "";
};

onMounted(() => {
  void refreshGlobalMetrics();
  globalMetricsTimer = window.setInterval(() => {
    void refreshGlobalMetrics();
  }, 15000);
});

onBeforeUnmount(() => {
  if (currentController) {
    currentController.abort();
    currentController = null;
  }
  if (globalMetricsTimer !== null) {
    window.clearInterval(globalMetricsTimer);
    globalMetricsTimer = null;
  }
});
</script>


<style scoped>
.app-shell {
  position: relative;
  min-height: 100vh;
  padding: 72px 24px;
  display: flex;
  justify-content: center;
  align-items: center;
  background: radial-gradient(circle at 20% 20%, #f8fafc, #dbeafe 60%);
  color: #1f2937;
  overflow: hidden;
  box-sizing: border-box;
  transition: padding 0.4s ease;
}

.app-shell.expanded {
  padding: 0;
  align-items: stretch;
}

.aurora {
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0.55;
}

.aurora span {
  position: absolute;
  width: 45vw;
  height: 45vw;
  max-width: 520px;
  max-height: 520px;
  background: radial-gradient(circle, rgba(148, 197, 255, 0.35), transparent 60%);
  filter: blur(90px);
  animation: float 26s infinite linear;
}

.aurora span:nth-child(1) {
  top: -20%;
  left: -18%;
  animation-delay: 0s;
}

.aurora span:nth-child(2) {
  bottom: -25%;
  right: -20%;
  background: radial-gradient(circle, rgba(166, 139, 255, 0.28), transparent 60%);
  animation-delay: -9s;
}

.aurora span:nth-child(3) {
  top: 35%;
  left: 45%;
  background: radial-gradient(circle, rgba(164, 219, 216, 0.26), transparent 60%);
  animation-delay: -16s;
}

.layout {
  position: relative;
  width: 100%;
  display: flex;
  gap: 24px;
  z-index: 1;
  transition: all 0.4s ease;
}

.layout-centered {
  max-width: 600px;
  justify-content: center;
  align-items: center;
}

.layout-fullscreen {
  height: 100vh;
  max-width: 100%;
  gap: 0;
  align-items: stretch;
}

.panel {
  position: relative;
  flex: 1 1 360px;
  padding: 24px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 24px 48px rgba(15, 23, 42, 0.12);
  backdrop-filter: blur(8px);
  overflow: hidden;
}

.panel-form {
  max-width: 420px;
}

.panel-centered {
  width: 100%;
  max-width: 600px;
  padding: 40px;
  box-shadow: 0 32px 64px rgba(15, 23, 42, 0.15);
  transform: scale(1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.panel-centered:hover {
  transform: scale(1.02);
  box-shadow: 0 40px 80px rgba(15, 23, 42, 0.2);
}

.panel-result {
  min-width: 360px;
  flex: 2 1 420px;
}

.panel::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(125, 86, 255, 0.1));
  opacity: 0;
  transition: opacity 0.35s ease;
  z-index: 0;
}

.panel:hover::before {
  opacity: 1;
}

.panel > * {
  position: relative;
  z-index: 1;
}

.panel-form h1 {
  margin: 0;
  font-size: 26px;
  letter-spacing: 0.01em;
}

.panel-form p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 13px;
}

.panel-head {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.logo {
  width: 52px;
  height: 52px;
  display: grid;
  place-items: center;
  border-radius: 16px;
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  box-shadow: 0 12px 28px rgba(59, 130, 246, 0.4);
}

.logo svg {
  width: 28px;
  height: 28px;
  fill: #f8fafc;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.field span {
  font-weight: 600;
  color: #475569;
}

textarea,
input,
select {
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: rgba(255, 255, 255, 0.92);
  color: #1f2937;
  font-size: 14px;
  transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
}

textarea:focus,
input:focus,
select:focus {
  outline: none;
  border-color: rgba(37, 99, 235, 0.65);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
  background: #ffffff;
}

.options {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.option {
  flex: 1;
  min-width: 140px;
}

.form-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.submit {
  align-self: flex-start;
  padding: 12px 24px;
  border-radius: 16px;
  border: none;
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  color: #ffffff;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  position: relative;
}

.submit-label {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.submit .spinner {
  width: 18px;
  height: 18px;
  fill: none;
  stroke: rgba(255, 255, 255, 0.85);
  stroke-linecap: round;
  animation: spin 1s linear infinite;
}

.submit:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.submit:not(:disabled):hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 28px rgba(37, 99, 235, 0.28);
}

.secondary-btn {
  padding: 10px 18px;
  border-radius: 14px;
  background: rgba(148, 163, 184, 0.12);
  border: 1px solid rgba(148, 163, 184, 0.28);
  color: #1f2937;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}

.secondary-btn:hover {
  background: rgba(148, 163, 184, 0.2);
  border-color: rgba(148, 163, 184, 0.35);
  color: #0f172a;
}

.error-chip {
  margin-top: 16px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(248, 113, 113, 0.12);
  border: 1px solid rgba(248, 113, 113, 0.35);
  border-radius: 14px;
  color: #b91c1c;
  font-size: 14px;
}

.error-chip svg {
  width: 18px;
  height: 18px;
  fill: currentColor;
}

.panel-result {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.status-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.status-main {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.status-controls {
  display: flex;
  gap: 8px;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(191, 219, 254, 0.28);
  padding: 8px 14px;
  border-radius: 999px;
  font-size: 13px;
  color: #1f2937;
  border: 1px solid rgba(59, 130, 246, 0.35);
  transition: background 0.3s ease, color 0.3s ease;
}

.status-chip.active {
  background: rgba(129, 140, 248, 0.2);
  border-color: rgba(129, 140, 248, 0.4);
  color: #1e293b;
}

.status-chip .dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #2563eb;
  box-shadow: 0 0 12px rgba(37, 99, 235, 0.45);
  animation: pulse 1.8s ease-in-out infinite;
}

.status-meta {
  color: #64748b;
  font-size: 13px;
}

.metrics-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}

.metric-card {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(248, 250, 252, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-label {
  font-size: 12px;
  color: #64748b;
}

.metric-value {
  font-size: 15px;
  color: #0f172a;
}

.metric-hint {
  font-size: 12px;
  color: #475569;
}

.planning-summary {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.22);
  box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.35);
}

.planning-summary-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.planning-summary-head h3 {
  margin: 0;
  font-size: 16px;
  color: #0f172a;
}

.planning-summary-head p {
  margin: 4px 0 0;
  font-size: 13px;
  color: #64748b;
}

.planning-count {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(191, 219, 254, 0.34);
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 600;
}

.planning-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}

.planning-card {
  padding: 14px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
  border: 1px solid rgba(148, 163, 184, 0.2);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.planning-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.planning-card-head strong {
  font-size: 14px;
  color: #0f172a;
}

.planning-intent,
.planning-query,
.planning-note {
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
}

.planning-intent {
  color: #334155;
}

.planning-query,
.planning-note {
  color: #475569;
}

.timeline-section {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px;
  border-radius: 18px;
  background: rgba(248, 250, 252, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.28);
  overflow-anchor: none;
}

.timeline-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.timeline-head h3 {
  margin: 0;
  font-size: 16px;
  color: #0f172a;
}

.timeline-head p {
  margin: 4px 0 0;
  font-size: 13px;
  color: #64748b;
}

.timeline-count {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(191, 219, 254, 0.34);
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 600;
}

.timeline-wrapper {
  scroll-margin-top: 18px;
  max-height: 220px;
  overflow-y: auto;
  padding: 4px 8px 4px 0;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.84);
  padding-right: 8px;
  scrollbar-width: thin;
  scrollbar-color: rgba(129, 140, 248, 0.45) rgba(226, 232, 240, 0.6);
}

.timeline-wrapper::-webkit-scrollbar {
  width: 6px;
}

.timeline-wrapper::-webkit-scrollbar-track {
  background: rgba(226, 232, 240, 0.6);
  border-radius: 999px;
}

.timeline-wrapper::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(129, 140, 248, 0.8), rgba(59, 130, 246, 0.7));
  border-radius: 999px;
}

.timeline-wrapper::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.9), rgba(37, 99, 235, 0.8));
}

.timeline {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  position: relative;
  padding-left: 12px;
}

.timeline::before {
  content: "";
  position: absolute;
  top: 8px;
  bottom: 8px;
  left: 0;
  width: 2px;
  background: linear-gradient(180deg, rgba(59, 130, 246, 0.35), rgba(129, 140, 248, 0.15));
}

.timeline li {
  position: relative;
  padding-left: 24px;
  color: #1e293b;
  font-size: 14px;
  line-height: 1.5;
}

.timeline li p {
  margin: 0;
}

.timeline-node {
  position: absolute;
  left: -12px;
  top: 6px;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(135deg, #38bdf8, #7c3aed);
  box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.22);
}

.timeline-enter-active,
.timeline-leave-active {
  transition: all 0.35s ease, opacity 0.35s ease;
}

.timeline-enter-from,
.timeline-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.tasks-section {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 20px;
  align-items: start;
}

@media (max-width: 960px) {
  .tasks-section {
    grid-template-columns: 1fr;
  }
}

.tasks-list {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.26);
  border-radius: 18px;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.4);
}

.tasks-list h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.tasks-list ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.task-item {
  border-radius: 14px;
  border: 1px solid transparent;
  transition: border-color 0.2s ease, background 0.2s ease;
}

.task-item.completed {
  border-color: rgba(56, 189, 248, 0.35);
  background: rgba(191, 219, 254, 0.28);
}

.task-item.active {
  border-color: rgba(129, 140, 248, 0.5);
  background: rgba(224, 231, 255, 0.5);
}

.task-button {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px 6px;
  background: transparent;
  border: none;
  color: inherit;
  cursor: pointer;
  text-align: left;
}

.task-title {
  font-weight: 600;
  font-size: 14px;
  color: #1e293b;
}

.task-status {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  color: #1f2937;
  background: rgba(148, 163, 184, 0.2);
}

.task-status.pending {
  background: rgba(148, 163, 184, 0.18);
  color: #475569;
}

.task-status.in_progress {
  background: rgba(129, 140, 248, 0.24);
  color: #312e81;
}

.task-status.completed {
  background: rgba(34, 197, 94, 0.2);
  color: #15803d;
}

.task-status.skipped {
  background: rgba(248, 113, 113, 0.18);
  color: #b91c1c;
}

.task-intent {
  margin: 0;
  padding: 0 14px 12px 14px;
  font-size: 13px;
  color: #64748b;
}

.task-detail {
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.26);
  border-radius: 18px;
  padding: 22px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.5);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
}

.task-chip-group {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.task-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
}

.task-header .muted {
  margin: 6px 0 0;
}

.task-label {
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(191, 219, 254, 0.32);
  border: 1px solid rgba(59, 130, 246, 0.35);
  font-size: 12px;
  color: #1e3a8a;
}

.task-label.note-chip {
  background: rgba(34, 197, 94, 0.2);
  border-color: rgba(34, 197, 94, 0.35);
  color: #15803d;
}

.task-label.path-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 360px;
  background: rgba(56, 189, 248, 0.2);
  border-color: rgba(56, 189, 248, 0.35);
  color: #0369a1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.path-label {
  font-weight: 500;
}

.path-text {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chip-action {
  border: none;
  background: rgba(56, 189, 248, 0.2);
  color: #0369a1;
  padding: 3px 8px;
  border-radius: 10px;
  font-size: 11px;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease;
}

.chip-action:hover {
  background: rgba(14, 165, 233, 0.28);
  color: #0f172a;
}

.task-notices {
  background: rgba(191, 219, 254, 0.28);
  border: 1px solid rgba(96, 165, 250, 0.35);
  border-radius: 16px;
  padding: 14px 18px;
  color: #1f2937;
}

.task-notices h4 {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 600;
}

.task-notices ul {
  list-style: disc;
  margin: 0 0 0 18px;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.task-notices li {
  font-size: 13px;
}

.report-block {
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.26);
  border-radius: 18px;
  padding: 22px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.report-block h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
}

.block-pre {
  font-family: "JetBrains Mono", "Fira Code", ui-monospace, SFMono-Regular,
    Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  color: #1f2937;
  background: rgba(248, 250, 252, 0.9);
  padding: 16px;
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  overflow: auto;
  max-height: 420px;
  scrollbar-width: thin;
  scrollbar-color: rgba(129, 140, 248, 0.6) rgba(226, 232, 240, 0.7);
}

.block-pre::-webkit-scrollbar {
  width: 6px;
}

.block-pre::-webkit-scrollbar-track {
  background: rgba(226, 232, 240, 0.7);
  border-radius: 999px;
}

.block-pre::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.75), rgba(59, 130, 246, 0.65));
  border-radius: 999px;
}

.block-pre::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(79, 70, 229, 0.8), rgba(37, 99, 235, 0.75));
}

.markdown-block {
  font-size: 14px;
  line-height: 1.75;
  color: #1f2937;
  background: rgba(248, 250, 252, 0.9);
  padding: 16px 18px;
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  overflow: auto;
  max-height: 420px;
  scrollbar-width: thin;
  scrollbar-color: rgba(129, 140, 248, 0.6) rgba(226, 232, 240, 0.7);
}

.markdown-block::-webkit-scrollbar {
  width: 6px;
}

.markdown-block::-webkit-scrollbar-track {
  background: rgba(226, 232, 240, 0.7);
  border-radius: 999px;
}

.markdown-block::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.75), rgba(59, 130, 246, 0.65));
  border-radius: 999px;
}

.markdown-block::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(79, 70, 229, 0.8), rgba(37, 99, 235, 0.75));
}

.summary-block .markdown-block,
.report-block .markdown-block {
  max-height: 360px;
}

.markdown-block :deep(h1),
.markdown-block :deep(h2),
.markdown-block :deep(h3),
.markdown-block :deep(h4) {
  margin: 0 0 12px;
  color: #0f172a;
  line-height: 1.35;
}

.markdown-block :deep(h1) {
  font-size: 22px;
}

.markdown-block :deep(h2) {
  font-size: 18px;
}

.markdown-block :deep(h3) {
  font-size: 16px;
}

.markdown-block :deep(p),
.markdown-block :deep(ul),
.markdown-block :deep(ol),
.markdown-block :deep(blockquote) {
  margin: 0 0 12px;
}

.markdown-block :deep(ul),
.markdown-block :deep(ol) {
  padding-left: 20px;
}

.markdown-block :deep(li + li) {
  margin-top: 6px;
}

.markdown-block :deep(code) {
  font-family: "JetBrains Mono", "Fira Code", ui-monospace, SFMono-Regular,
    Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  background: rgba(224, 231, 255, 0.65);
  padding: 2px 6px;
  border-radius: 6px;
}

.markdown-block :deep(pre) {
  margin: 0 0 12px;
  padding: 14px 16px;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.92);
  color: #e2e8f0;
  overflow: auto;
}

.markdown-block :deep(pre code) {
  background: transparent;
  padding: 0;
  color: inherit;
}

.markdown-block :deep(a) {
  color: #2563eb;
  text-decoration: none;
}

.markdown-block :deep(a:hover) {
  text-decoration: underline;
}

.markdown-block :deep(blockquote) {
  padding-left: 12px;
  border-left: 3px solid rgba(59, 130, 246, 0.35);
  color: #475569;
}


.tools-block {
  position: relative;
  margin-top: 16px;
  padding: 20px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.4);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tools-block h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  letter-spacing: 0.02em;
}

.tool-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tool-entry {
  background: rgba(248, 250, 252, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 14px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.tool-entry-header {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
}

.tool-entry-title {
  font-weight: 600;
  color: #1f2937;
}

.tool-entry-note {
  font-size: 12px;
  color: #0f766e;
}

.tool-entry-path {
  margin: 0;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  color: #2563eb;
}

.tool-subtitle {
  margin: 0;
  font-size: 13px;
  color: #475569;
  font-weight: 500;
}

.tool-pre {
  font-family: "JetBrains Mono", "Fira Code", ui-monospace, SFMono-Regular,
    Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: #1f2937;
  background: rgba(248, 250, 252, 0.9);
  padding: 12px;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  overflow: auto;
  max-height: 260px;
  scrollbar-width: thin;
  scrollbar-color: rgba(129, 140, 248, 0.6) rgba(226, 232, 240, 0.7);
}

.tool-pre::-webkit-scrollbar {
  width: 6px;
}

.tool-pre::-webkit-scrollbar-track {
  background: rgba(226, 232, 240, 0.7);
}

.tool-pre::-webkit-scrollbar-thumb {
  background: rgba(99, 102, 241, 0.7);
  border-radius: 10px;
}

.link-btn {
  background: none;
  border: none;
  color: #0369a1;
  cursor: pointer;
  padding: 0 4px;
  font-size: 12px;
  border-radius: 8px;
  transition: color 0.2s ease, background 0.2s ease;
}

.link-btn:hover {
  color: #0ea5e9;
  background: rgba(14, 165, 233, 0.16);
}


.sources-block,
.summary-block {
  position: relative;
  margin-top: 16px;
  padding: 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.4);
}

.sources-history {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.sources-history h4 {
  margin: 0;
  color: #1f2937;
  font-size: 14px;
  letter-spacing: 0.01em;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.history-list details {
  background: rgba(248, 250, 252, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 14px;
  padding: 12px 16px;
  color: #1f2937;
  transition: border-color 0.2s ease, background 0.2s ease;
}

.history-list details[open] {
  background: rgba(224, 231, 255, 0.55);
  border-color: rgba(129, 140, 248, 0.4);
}

.history-list summary {
  cursor: pointer;
  font-weight: 600;
  outline: none;
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.history-list summary::-webkit-details-marker {
  display: none;
}

.history-list summary::after {
  content: "▾";
  margin-left: 6px;
  font-size: 12px;
  opacity: 0.7;
  transition: transform 0.2s ease;
}

.history-list details[open] summary::after {
  transform: rotate(180deg);
}

.block-highlight {
  animation: glow 1.2s ease;
}

.sources-block h3,
.summary-block h3 {
  margin: 0 0 14px;
  color: #1f2937;
  letter-spacing: 0.02em;
}

.sources-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.source-item {
  position: relative;
  display: inline-flex;
  flex-direction: column;
  gap: 6px;
}

.source-link {
  color: #2563eb;
  text-decoration: none;
  font-weight: 600;
  letter-spacing: 0.01em;
  transition: color 0.2s ease;
}

.source-link::after {
  content: " ↗";
  font-size: 12px;
  opacity: 0.6;
}

.source-link:hover {
  color: #0f172a;
}

.source-tooltip {
  display: none;
  position: absolute;
  bottom: calc(100% + 12px);
  left: 50%;
  transform: translateX(-50%);
  background: rgba(255, 255, 255, 0.98);
  color: #1f2937;
  padding: 14px 16px;
  border-radius: 16px;
  box-shadow: 0 18px 32px rgba(15, 23, 42, 0.18);
  width: min(420px, 90vw);
  z-index: 20;
  border: 1px solid rgba(148, 163, 184, 0.24);
}

.source-tooltip::after {
  content: "";
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border-width: 10px;
  border-style: solid;
  border-color: rgba(255, 255, 255, 0.98) transparent transparent transparent;
}

.source-tooltip::before {
  content: "";
  position: absolute;
  bottom: -12px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 12px 10px 0 10px;
  border-style: solid;
  border-color: rgba(255, 255, 255, 0.98) transparent transparent transparent;
  filter: drop-shadow(0 -2px 4px rgba(15, 23, 42, 0.12));
}

.source-tooltip p {
  margin: 0 0 8px;
  font-size: 13px;
  line-height: 1.6;
}

.source-tooltip p:last-child {
  margin-bottom: 0;
}

.muted-text {
  color: #64748b;
}

.source-item:hover .source-tooltip,
.source-item:focus-within .source-tooltip {
  display: block;
}

.hint.muted {
  color: #64748b;
}

@keyframes float {
  0% {
    transform: translate3d(0, 0, 0) rotate(0deg);
  }
  50% {
    transform: translate3d(10%, 6%, 0) rotate(3deg);
  }
  100% {
    transform: translate3d(0, 0, 0) rotate(0deg);
  }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes pulse {
  0%,
  100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.3);
    opacity: 0.5;
  }
}

@keyframes glow {
  0% {
    box-shadow: 0 0 0 rgba(59, 130, 246, 0.3);
    border-color: rgba(59, 130, 246, 0.5);
  }
  100% {
    box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.12);
    border-color: rgba(148, 163, 184, 0.2);
  }
}

@media (max-width: 960px) {
  .app-shell {
    padding: 56px 16px;
  }

  .layout {
    flex-direction: column;
    align-items: stretch;
  }

  .panel {
    padding: 22px;
  }

  .panel-form,
  .panel-result {
    max-width: none;
  }

  .status-bar {
    flex-direction: column;
    align-items: flex-start;
  }

  .status-main,
  .status-controls {
    width: 100%;
  }

  .status-controls {
    justify-content: flex-start;
  }
}

@media (max-width: 600px) {
  .options {
    flex-direction: column;
  }

  .status-meta {
    font-size: 12px;
  }

  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .panel-form h1 {
    font-size: 24px;
  }
}

/* 侧边栏样式 */
.sidebar {
  width: 400px;
  min-width: 400px;
  height: 100vh;
  background: rgba(255, 255, 255, 0.98);
  border-right: 1px solid rgba(148, 163, 184, 0.2);
  padding: 32px 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  overflow-y: auto;
  box-shadow: 4px 0 24px rgba(15, 23, 42, 0.08);
}

.sidebar-header {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.sidebar-header h2 {
  font-size: 24px;
  font-weight: 700;
  margin: 0;
  color: #1f2937;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: transparent;
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 12px;
  color: #64748b;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  width: fit-content;
}

.back-btn:hover:not(:disabled) {
  background: rgba(59, 130, 246, 0.1);
  border-color: #3b82f6;
  color: #3b82f6;
}

.back-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.research-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-item label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
}

.info-item p {
  margin: 0;
  font-size: 14px;
  color: #1f2937;
  line-height: 1.6;
}

.topic-display {
  font-size: 16px !important;
  font-weight: 600;
  color: #0f172a !important;
  padding: 12px;
  background: rgba(59, 130, 246, 0.05);
  border-radius: 8px;
  border-left: 3px solid #3b82f6;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: rgba(148, 163, 184, 0.2);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  border-radius: 4px;
  transition: width 0.5s ease;
}

.progress-text {
  font-size: 13px !important;
  color: #64748b !important;
  font-weight: 500;
}

.sidebar-actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-top: 16px;
  border-top: 1px solid rgba(148, 163, 184, 0.2);
}

.new-research-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 14px 20px;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  border: none;
  border-radius: 12px;
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

.new-research-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

.new-research-btn:active {
  transform: translateY(0);
}

.sidebar-history {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px;
  border-radius: 20px;
  background: rgba(248, 250, 252, 0.88);
  border: 1px solid rgba(148, 163, 184, 0.2);
}

.sidebar-history-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.sidebar-history-head h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
  color: #0f172a;
}

.history-count {
  font-size: 12px;
  color: #64748b;
}

.sidebar-history-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.sidebar-history-item {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.sidebar-history-item.selected {
  border-color: rgba(59, 130, 246, 0.34);
  box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.12);
}

.sidebar-history-main {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sidebar-history-topic {
  margin: 0;
  color: #1f2937;
  font-size: 14px;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.sidebar-history-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
  color: #64748b;
}

.history-status-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  color: #475569;
  background: rgba(148, 163, 184, 0.16);
}

.history-status-chip.success {
  color: #166534;
  background: rgba(34, 197, 94, 0.16);
}

.history-status-chip.partial_success {
  color: #b45309;
  background: rgba(245, 158, 11, 0.18);
}

.history-status-chip.failed {
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.16);
}

.history-status-chip.in_progress {
  color: #1d4ed8;
  background: rgba(59, 130, 246, 0.16);
}

.history-repeat-btn {
  align-self: flex-start;
  border: 1px solid rgba(59, 130, 246, 0.24);
  background: rgba(59, 130, 246, 0.08);
  color: #1d4ed8;
  border-radius: 12px;
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}

.history-repeat-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  background: rgba(59, 130, 246, 0.14);
  box-shadow: 0 10px 20px rgba(59, 130, 246, 0.12);
}

.history-repeat-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.sidebar-history-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.history-view-btn {
  align-self: flex-start;
  border: 1px solid rgba(15, 23, 42, 0.12);
  background: rgba(15, 23, 42, 0.04);
  color: #0f172a;
  border-radius: 12px;
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}

.history-view-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  background: rgba(15, 23, 42, 0.08);
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
}

.history-view-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.sidebar-history-empty {
  margin: 0;
  font-size: 13px;
  color: #64748b;
}

/* 全屏状态下的结果面板 */
.layout-fullscreen .panel-result {
  flex: 1;
  height: 100vh;
  border-radius: 0;
  border: none;
  overflow-y: auto;
  max-width: none;
}

@media (max-width: 1024px) {
  .sidebar {
    width: 320px;
    min-width: 320px;
  }
}

@media (max-width: 768px) {
  .layout-fullscreen {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
    min-width: 100%;
    height: auto;
    max-height: 40vh;
  }

  .layout-fullscreen .panel-result {
    height: 60vh;
  }
}
</style>
