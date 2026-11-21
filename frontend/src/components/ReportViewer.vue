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

    <div class="markdown-block" v-html="reportHtml"></div>

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
import { computed, ref, watch } from "vue";

const props = withDefaults(
  defineProps<{
    reportMarkdown: string;
    reportHtml: string;
    highlight: boolean;
    requestId: string | null;
    drilldownLoading?: boolean;
    drilldownAnswerHtml?: string;
    drilldownError?: string;
  }>(),
  {
    drilldownLoading: false,
    drilldownAnswerHtml: "",
    drilldownError: ""
  }
);

const emit = defineEmits<{
  drilldown: [payload: { sectionTitle: string; question: string }];
}>();

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
</style>
