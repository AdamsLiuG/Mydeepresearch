<template>
  <div
    v-if="reportMarkdown"
    class="report-block"
    :class="{ 'block-highlight': highlight }"
  >
    <div class="report-block-head">
      <h3>最终报告</h3>
      <button
        v-if="requestId && canDrilldown"
        type="button"
        class="drilldown-toggle-btn"
        @click="showDrilldown = !showDrilldown"
      >
        {{ showDrilldown ? "收起章节追问" : "章节深入追问 (HITL)" }}
      </button>
    </div>

    <SkillGovernanceBar
      v-if="workbench"
      :governance="workbench.skill_governance"
      :catalog="catalog"
      :verifier-badges="workbench.verifier_badges"
      :claim-summary="workbench.claim_verification_summary"
      :conflict-count="workbench.evidence_conflicts.length"
      @inspect-conflicts="$emit('inspect-conflicts')"
    />

    <div
      ref="reportBody"
      class="markdown-block"
      v-html="reportHtml"
      @click="handleMarkdownClick"
    ></div>

    <template v-for="(chart, index) in chartViews" :key="index">
      <figure v-if="chart" class="report-chart">
        <figcaption><strong>{{ chart.title }}</strong><p v-if="chart.caption">{{ chart.caption }}</p></figcaption>
        <svg viewBox="0 0 640 220" role="img" :aria-label="chart.title">
          <line x1="24" x2="616" :y1="chart.zero" :y2="chart.zero" stroke="#94a3b8" />
          <g v-for="(series, seriesIndex) in chart.series" :key="seriesIndex" :style="{ color: chartColors[seriesIndex % chartColors.length] }">
            <polyline v-if="series.type === 'line'" :points="series.coordinates" fill="none" stroke="currentColor" stroke-width="2" />
            <template v-for="(point, pointIndex) in series.points" :key="pointIndex">
              <rect v-if="series.type === 'bar'" :x="point.x - point.width / 2" :y="Math.min(point.y, chart.zero)"
                :width="point.width" :height="Math.abs(chart.zero - point.y) || 1" fill="currentColor">
                <title>{{ series.name }} · {{ point.label }}: {{ point.value }}</title>
              </rect>
              <circle v-else :cx="point.x" :cy="point.y" r="3" fill="currentColor">
                <title>{{ series.name }} · {{ point.label }}: {{ point.value }}</title>
              </circle>
            </template>
          </g>
        </svg>
        <details>
          <summary>图表数据</summary>
          <table>
            <thead><tr><th scope="col">系列</th><th scope="col">项目</th><th scope="col">数值</th></tr></thead>
            <tbody v-for="(series, seriesIndex) in chart.series" :key="seriesIndex">
              <tr v-for="(point, pointIndex) in series.points" :key="pointIndex"><td>{{ series.name }}</td><td>{{ point.label }}</td><td>{{ point.value }}</td></tr>
            </tbody>
          </table>
        </details>
        <button v-for="sourceId in chart.source_ids" :key="sourceId" type="button" class="citation-chip" @click="emit('select-citation', sourceId)">
          [{{ sourceId }}]
        </button>
      </figure>
      <p v-else class="chart-fallback">此图表无法显示，可查看报告中的数据表。</p>
    </template>

    <div v-if="showDrilldown && requestId" class="drilldown-panel">
      <div class="drilldown-controls">
        <select
          v-model="selectedSection"
          class="drilldown-select"
          aria-label="选择报告章节"
        >
          <option
            v-for="heading in sectionHeadings"
            :key="heading"
            :value="heading"
          >
            {{ heading }}
          </option>
          <option v-if="!sectionHeadings.length" value="核心结论">
            核心结论
          </option>
        </select>
        <input
          v-model="questionText"
          type="text"
          class="drilldown-input"
          placeholder="围绕所选章节继续追问细节或补充反证..."
          @keyup.enter="submitDrilldown"
        />
        <button
          type="button"
          class="drilldown-submit-btn"
          :disabled="drilldownLoading || !questionText.trim()"
          @click="submitDrilldown"
        >
          {{ drilldownLoading ? "追问中..." : "发起追问" }}
        </button>
      </div>
      <p v-if="drilldownError" class="drilldown-error">{{ drilldownError }}</p>
      <div
        v-if="drilldownAnswerHtml"
        class="markdown-block drilldown-answer"
        v-html="drilldownAnswerHtml"
      ></div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { computed, ref, watch, watchPostEffect } from "vue";
import type { ReportWorkbenchPayload, SkillsCatalogResponse } from "../types/research";
import { enhanceReportCitations, enhanceReportTables, getChartSeries, sanitizeChartSpec } from "../utils/workbench";
import SkillGovernanceBar from "./SkillGovernanceBar.vue";

const props = withDefaults(
  defineProps<{
    reportMarkdown: string;
    reportHtml: string;
    highlight: boolean;
    requestId: string | null;
    drilldownLoading?: boolean;
    drilldownAnswerHtml?: string;
    drilldownError?: string;
    workbench?: ReportWorkbenchPayload | null;
    catalog?: SkillsCatalogResponse | null;
  }>(),
  {
    drilldownLoading: false,
    drilldownAnswerHtml: "",
    drilldownError: "",
    workbench: null
  }
);

const emit = defineEmits<{
  drilldown: [payload: { sectionTitle: string; question: string }];
  "select-citation": [sourceId: string];
  "inspect-conflicts": [];
}>();

const reportBody = ref<HTMLElement | null>(null);
watchPostEffect(() => {
  if (props.reportHtml && reportBody.value) {
    enhanceReportCitations(reportBody.value);
    enhanceReportTables(reportBody.value);
  }
});

const chartColors = ["#1d4ed8", "#047857", "#9a3412", "#7c3aed", "#0369a1"];
const chartViews = computed(() => (props.workbench?.charts_spec || []).map((raw) => {
  const chart = sanitizeChartSpec(raw);
  if (!chart) return null;
  const series = getChartSeries(chart);
  if (!series.length) return null;
  const values = series.flatMap((item) => item.points.map((point) => point.value));
  const magnitude = Math.max(1, ...values.map(Math.abs));
  const min = Math.min(0, ...values) / magnitude;
  const max = Math.max(0, ...values) / magnitude;
  const y = (value: number) => 202 - ((value / magnitude - min) / (max - min || 1)) * 184;
  const labels = [...new Set(series.flatMap((item) => item.points.map((point) => point.label)))];
  const columns = new Map(labels.map((label, index) => [label, index]));
  const step = 592 / labels.length;
  const width = Math.min(32, step * 0.8 / series.length);
  return { ...chart, zero: y(0), series: series.map((item, seriesIndex) => {
    const points = item.points.map((point) => ({
      ...point,
      x: 24 + ((columns.get(point.label) ?? 0) + 0.5) * step +
        (item.type === "bar" ? (seriesIndex - (series.length - 1) / 2) * width : 0),
      y: y(point.value), width
    }));
    return { ...item, points, coordinates: points.map((point) => `${point.x},${point.y}`).join(" ") };
  }) };
}));

function handleMarkdownClick(event: MouseEvent) {
  const target = event.target as HTMLElement | null;
  const chip = target?.closest<HTMLElement>("button.citation-chip[data-citation-id]");
  const citationId = chip?.dataset?.citationId?.trim();
  if (citationId) {
    emit("select-citation", citationId);
  }
}

const showDrilldown = ref(false);
const questionText = ref("");

const sectionHeadings = computed(() => {
  const matches: string[] = [];
  const pattern = /^#{2,3}\s+(.+)$/gm;
  let match: RegExpExecArray | null = pattern.exec(props.reportMarkdown || "");
  while (match !== null) {
    const title = match[1]?.trim();
    if (title && !matches.includes(title)) {
      matches.push(title);
    }
    match = pattern.exec(props.reportMarkdown || "");
  }
  return matches;
});

const canDrilldown = computed(
  () =>
    Boolean(props.reportMarkdown.trim()) &&
    props.reportMarkdown.trim() !== "暂无生成的报告"
);

const selectedSection = ref(sectionHeadings.value[0] || "核心结论");

watch(sectionHeadings, (nextHeadings) => {
  if (nextHeadings.length && !nextHeadings.includes(selectedSection.value)) {
    selectedSection.value = nextHeadings[0];
  }
});

function submitDrilldown() {
  const question = questionText.value.trim();
  if (!question || props.drilldownLoading) {
    return;
  }
  emit("drilldown", {
    sectionTitle: selectedSection.value || "核心结论",
    question
  });
}
</script>

<style scoped>
.report-chart {
  margin: 16px 0;
  border-top: 1px solid #cbd5e1;
  padding-top: 12px;
}
.report-chart svg {
  display: block;
  width: 100%;
  max-height: 280px;
}
.report-chart details { margin: 8px 0; }
.report-chart td, .report-chart th { padding: 2px 8px; text-align: left; }
.chart-fallback { color: #475569; }
.markdown-block :deep(.table-export) { display: flex; flex-wrap: wrap; gap: 6px; }
.markdown-block :deep(.table-sort),
.markdown-block :deep(.table-export button) { cursor: pointer; }
.markdown-block :deep([aria-sort="ascending"] .table-sort)::after { content: " ↑"; }
.markdown-block :deep([aria-sort="descending"] .table-sort)::after { content: " ↓"; }
.report-block {
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.26);
  border-radius: 18px;
  padding: 22px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.report-block-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.report-block h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
}

.drilldown-toggle-btn {
  border: 1px solid rgba(59, 130, 246, 0.28);
  background: rgba(59, 130, 246, 0.08);
  color: #1d4ed8;
  border-radius: 10px;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.drilldown-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  border-radius: 12px;
  background: rgba(241, 245, 249, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.24);
}

.drilldown-controls {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.drilldown-select {
  min-width: 160px;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: #ffffff;
  font-size: 13px;
}

.drilldown-input {
  flex: 1;
  min-width: 220px;
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.45);
  font-size: 13px;
}

.drilldown-submit-btn {
  border: none;
  border-radius: 10px;
  padding: 8px 14px;
  background: #2563eb;
  color: #ffffff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.drilldown-submit-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.drilldown-error {
  margin: 0;
  font-size: 13px;
  color: #b91c1c;
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

.markdown-block :deep(.citation-chip) {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(37, 99, 235, 0.32);
  background: rgba(239, 246, 255, 0.92);
  color: #1d4ed8;
  border-radius: 6px;
  padding: 0 5px;
  margin: 0 2px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.markdown-block :deep(.citation-chip:hover) {
  background: rgba(219, 234, 254, 1);
}
</style>
