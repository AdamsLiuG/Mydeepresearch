<template>
  <section class="sidebar-history">
    <div class="sidebar-history-head">
      <h3>最近研究历史</h3>
      <span class="history-count">{{ items.length }} 条</span>
    </div>
    <ul v-if="items.length" class="sidebar-history-list">
      <li
        v-for="item in items"
        :key="item.requestId"
        :class="[
          'sidebar-history-item',
          { selected: selectedRequestId === item.requestId }
        ]"
      >
        <div class="sidebar-history-main">
          <p class="sidebar-history-topic" :title="item.topic">
            {{ item.topic }}
          </p>
          <div class="sidebar-history-meta">
            <span class="history-status-chip" :class="item.status">
              {{ formatStatus(item.status) }}
            </span>
            <span>{{ formatElapsed(item.elapsedMs) }}</span>
          </div>
          <p
            v-if="item.error"
            class="sidebar-history-error"
            :title="item.error"
          >
            {{ item.error }}
          </p>
        </div>
        <div class="sidebar-history-actions">
          <button
            type="button"
            class="history-view-btn"
            @click="emit('view', item)"
            :disabled="
              loading ||
              loadingDetailRequestId === item.requestId ||
              !item.canViewContent
            "
          >
            {{
              loadingDetailRequestId === item.requestId
                ? "加载中..."
                : "查看内容"
            }}
          </button>
          <button
            type="button"
            class="history-repeat-btn"
            @click="emit('repeat', item)"
            :disabled="loading"
          >
            再次研究
          </button>
          <button
            v-if="item.canResume"
            type="button"
            class="history-repeat-btn"
            @click="emit('resume', item)"
            :disabled="loading"
          >
            恢复执行
          </button>
        </div>
      </li>
    </ul>
    <p v-else class="sidebar-history-empty">暂无历史研究记录</p>
  </section>
</template>

<script lang="ts" setup>
import type { RecentResearchView } from "../types/research";

defineProps<{
  items: RecentResearchView[];
  selectedRequestId: string | null;
  loading: boolean;
  loadingDetailRequestId: string | null;
  formatStatus: (status: string) => string;
  formatElapsed: (value: number) => string;
}>();

const emit = defineEmits<{
  view: [item: RecentResearchView];
  repeat: [item: RecentResearchView];
  resume: [item: RecentResearchView];
}>();
</script>

<style scoped>
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

.sidebar-history-error {
  margin: 0;
  color: #b91c1c;
  font-size: 12px;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
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

.sidebar-history-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
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
</style>
