<template>
  <section v-if="plan || clarification" class="hitl-panel">
    <header class="hitl-head">
      <div>
        <h3>研究计划人工确认 (HITL)</h3>
        <p v-if="plan" class="hitl-subtitle">
          预计耗时：{{ Math.round(plan.estimated_duration_seconds) }} 秒 · 共 {{ editableTasks.length }} 个子任务
        </p>
      </div>
      <button
        type="button"
        class="hitl-dismiss-btn"
        :disabled="busy"
        @click="emit('dismiss')"
      >
        关闭
      </button>
    </header>

    <div v-if="clarification" class="hitl-clarify">
      <div v-if="clarification.detected_domains?.length" class="hitl-chips">
        <span
          v-for="domain in clarification.detected_domains"
          :key="domain"
          class="hitl-chip"
        >
          领域：{{ domain }}
        </span>
      </div>
      <div v-if="clarification.questions?.length" class="hitl-clarify-block">
        <strong>建议先明确的边界问题：</strong>
        <ul>
          <li v-for="q in clarification.questions" :key="q">{{ q }}</li>
        </ul>
      </div>
      <div v-if="clarification.suggested_angles?.length" class="hitl-clarify-block">
        <strong>推荐切入视角：</strong>
        <p class="hitl-angles">{{ clarification.suggested_angles.join(" · ") }}</p>
      </div>
    </div>

    <div v-if="plan" class="hitl-tasks">
      <div
        v-for="(task, index) in editableTasks"
        :key="index"
        class="hitl-task-row"
      >
        <div class="hitl-task-top">
          <span class="hitl-task-badge">#{{ index + 1 }}</span>
          <input
            v-model="task.title"
            type="text"
            class="hitl-input"
            placeholder="任务标题"
          />
          <input
            v-model="task.dependsOnInput"
            type="text"
            class="hitl-input hitl-deps"
            placeholder="前置任务ID(如 1,2)"
          />
          <button
            type="button"
            class="hitl-remove-btn"
            :disabled="busy || editableTasks.length <= 1"
            @click="removeTask(index)"
          >
            删除
          </button>
        </div>
        <input
          v-model="task.intent"
          type="text"
          class="hitl-input"
          placeholder="研究意图"
        />
        <input
          v-model="task.query"
          type="text"
          class="hitl-input"
          placeholder="检索关键词"
        />
      </div>

      <p v-if="error" class="hitl-error">{{ error }}</p>

      <div class="hitl-actions">
        <button
          type="button"
          class="hitl-secondary-btn"
          :disabled="busy"
          @click="addTask"
        >
          + 添加子任务
        </button>
        <button
          type="button"
          class="hitl-secondary-btn"
          :disabled="busy"
          @click="handleSave"
        >
          {{ busy ? "处理中..." : "保存计划修改" }}
        </button>
        <button
          type="button"
          class="hitl-primary-btn"
          :disabled="busy"
          @click="emit('approve-plan')"
        >
          {{ busy ? "启动中..." : "批准并开始研究" }}
        </button>
      </div>
    </div>
  </section>
</template>

<script lang="ts" setup>
import { ref, watch } from "vue";
import type {
  ClarifyResponse,
  EditablePlanResponse,
  ResearchTaskPlanItem
} from "../types/research";

interface EditableTaskDraft extends ResearchTaskPlanItem {
  dependsOnInput: string;
}

const props = defineProps<{
  plan: EditablePlanResponse | null;
  clarification: ClarifyResponse | null;
  busy: boolean;
  error: string;
}>();

const emit = defineEmits<{
  "save-plan": [tasks: ResearchTaskPlanItem[]];
  "approve-plan": [];
  dismiss: [];
}>();

const editableTasks = ref<EditableTaskDraft[]>([]);

watch(
  () => props.plan,
  (nextPlan) => {
    if (!nextPlan?.todo_items?.length) {
      editableTasks.value = [];
      return;
    }
    editableTasks.value = nextPlan.todo_items.map((item, idx) => ({
      id: item.id ?? idx + 1,
      title: item.title || "",
      intent: item.intent || "",
      query: item.query || "",
      depends_on: Array.isArray(item.depends_on) ? [...item.depends_on] : [],
      dependsOnInput: Array.isArray(item.depends_on)
        ? item.depends_on.join(",")
        : ""
    }));
  },
  { immediate: true }
);

function parseDependsOn(raw: string): number[] {
  return raw
    .split(/[\s,，]+/)
    .map((token) => Number.parseInt(token.trim(), 10))
    .filter((id) => Number.isFinite(id) && id > 0);
}

function addTask() {
  const nextId = editableTasks.value.length + 1;
  editableTasks.value.push({
    id: nextId,
    title: `补充子任务 ${nextId}`,
    intent: "补充核心证据与对比分析",
    query: props.plan?.topic || "",
    depends_on: [],
    dependsOnInput: ""
  });
}

function removeTask(index: number) {
  if (editableTasks.value.length <= 1) {
    return;
  }
  editableTasks.value.splice(index, 1);
}

function handleSave() {
  const normalized: ResearchTaskPlanItem[] = editableTasks.value.map(
    (item, idx) => ({
      id: item.id ?? idx + 1,
      title: item.title.trim(),
      intent: item.intent.trim(),
      query: item.query.trim(),
      depends_on: parseDependsOn(item.dependsOnInput)
    })
  );
  emit("save-plan", normalized);
}
</script>

<style scoped>
.hitl-panel {
  margin-top: 16px;
  padding: 16px;
  border-radius: 16px;
  background: rgba(248, 250, 252, 0.96);
  border: 1px solid rgba(59, 130, 246, 0.25);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.hitl-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.hitl-head h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
  color: #0f172a;
}

.hitl-subtitle {
  margin: 4px 0 0;
  font-size: 12px;
  color: #64748b;
}

.hitl-dismiss-btn {
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: #ffffff;
  border-radius: 10px;
  padding: 4px 10px;
  font-size: 12px;
  cursor: pointer;
}

.hitl-clarify {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 13px;
  color: #334155;
}

.hitl-chips {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.hitl-chip {
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.12);
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 600;
}

.hitl-clarify-block ul {
  margin: 4px 0 0 18px;
  padding: 0;
}

.hitl-angles {
  margin: 4px 0 0;
  color: #475569;
}

.hitl-tasks {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.hitl-task-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.28);
}

.hitl-task-top {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}

.hitl-task-badge {
  font-size: 12px;
  font-weight: 700;
  color: #2563eb;
}

.hitl-input {
  flex: 1;
  min-width: 140px;
  padding: 6px 10px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, 0.45);
  font-size: 13px;
}

.hitl-deps {
  max-width: 140px;
}

.hitl-remove-btn {
  border: 1px solid rgba(239, 68, 68, 0.3);
  background: rgba(254, 242, 242, 0.9);
  color: #b91c1c;
  border-radius: 8px;
  padding: 5px 10px;
  font-size: 12px;
  cursor: pointer;
}

.hitl-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.hitl-secondary-btn {
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: #ffffff;
  color: #0f172a;
  border-radius: 10px;
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.hitl-primary-btn {
  border: none;
  background: #2563eb;
  color: #ffffff;
  border-radius: 10px;
  padding: 8px 14px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.hitl-error {
  margin: 0;
  font-size: 13px;
  color: #b91c1c;
}
</style>
