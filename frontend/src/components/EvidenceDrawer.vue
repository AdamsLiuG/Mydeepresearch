<template>
  <aside
    v-if="visible"
    class="evidence-drawer"
    role="complementary"
    aria-label="引用溯源与多源分歧抽屉"
  >
    <div class="drawer-header">
      <div>
        <h4>证据溯源与口径对照</h4>
        <p v-if="selectedCitationId" class="drawer-sub">
          当前选定引用：<code>[{{ selectedCitationId }}]</code>
        </p>
        <p v-else class="drawer-sub">
          展示全部跨源分歧 ({{ allConflicts.length }} 处)
        </p>
      </div>
      <button type="button" class="drawer-close-btn" @click="$emit('close')">
        关闭
      </button>
    </div>

    <div v-if="citationEntry" class="drawer-card">
      <div class="card-meta">
        <span class="badge">{{ citationEntry.quality_label || "medium" }}</span>
        <span v-if="citationEntry.modality" class="badge badge-muted">
          {{ citationEntry.modality }}
        </span>
        <span v-if="citationEntry.domain" class="domain-txt">
          {{ citationEntry.domain }}
        </span>
      </div>
      <h5>{{ citationEntry.title }}</h5>
      <a
        v-if="citationEntry.url"
        :href="citationEntry.url"
        target="_blank"
        rel="noopener noreferrer"
        class="source-link"
      >
        {{ citationEntry.url }}
      </a>
      <p class="excerpt-text">
        {{ citationEntry.snippet || "暂无详细摘要片段。" }}
      </p>
    </div>

    <div v-else-if="selectedCitationId" class="drawer-card muted-card">
      <p>未在证据索引中找到 <code>[{{ selectedCitationId }}]</code> 的详细记录。</p>
    </div>

    <div v-if="relevantConflicts.length" class="conflicts-section">
      <h5>相关跨源冲突 ({{ relevantConflicts.length }})</h5>
      <div
        v-for="cluster in relevantConflicts"
        :key="cluster.conflict_id"
        class="conflict-card"
      >
        <div class="conflict-head">
          <strong>{{ cluster.entity }} · {{ cluster.metric }}</strong>
          <span class="severity-badge" :class="`sev-${cluster.severity}`">
            {{ cluster.adjudication_status }}
          </span>
        </div>
        <p class="rationale">{{ cluster.adjudication_rationale }}</p>
        <div class="sides-grid">
          <div
            v-for="side in cluster.sides"
            :key="side.source_id"
            class="side-box"
            :class="{ recommended: cluster.recommended_source_id === side.source_id }"
          >
            <div class="side-top">
              <code>[{{ side.source_id }}]</code>
              <span>{{ side.title }}</span>
            </div>
            <p class="side-claim">{{ side.claim_text }}</p>
            <p v-if="side.published_at" class="side-meta">
              发布时间：{{ side.published_at }} · 权威分 {{ (side.authority_rank || 0).toFixed(2) }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </aside>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import type { ReportWorkbenchPayload } from "../types/research";
import { conflictsForCitation } from "../utils/workbench";

const props = defineProps<{
  visible: boolean;
  selectedCitationId: string | null;
  workbench: ReportWorkbenchPayload | null;
}>();

defineEmits<{
  close: [];
}>();

const citationEntry = computed(() => {
  if (!props.selectedCitationId || !props.workbench) {
    return null;
  }
  return props.workbench.citation_index?.[props.selectedCitationId] || null;
});

const allConflicts = computed(() => props.workbench?.evidence_conflicts || []);

const relevantConflicts = computed(() => {
  if (!props.workbench) {
    return [];
  }
  if (props.selectedCitationId) {
    const matched = conflictsForCitation(props.workbench, props.selectedCitationId);
    return matched.length ? matched : [];
  }
  return allConflicts.value;
});
</script>

<style scoped>
.evidence-drawer {
  margin-top: 14px;
  padding: 16px 18px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(59, 130, 246, 0.3);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.drawer-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.drawer-header h4 {
  margin: 0;
  font-size: 16px;
  color: #0f172a;
}

.drawer-sub {
  margin: 4px 0 0;
  font-size: 12px;
  color: #475569;
}

.drawer-close-btn {
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: #f8fafc;
  color: #334155;
  border-radius: 8px;
  padding: 4px 10px;
  font-size: 12px;
  cursor: pointer;
}

.drawer-card {
  padding: 12px 14px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid rgba(148, 163, 184, 0.28);
}

.drawer-card h5 {
  margin: 6px 0;
  font-size: 14px;
  color: #0f172a;
}

.card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  font-size: 11px;
}

.badge {
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.12);
  color: #1d4ed8;
  font-weight: 600;
}

.badge-muted {
  background: rgba(100, 116, 139, 0.12);
  color: #475569;
}

.source-link {
  display: block;
  font-size: 12px;
  color: #2563eb;
  word-break: break-all;
  margin-bottom: 6px;
}

.excerpt-text {
  margin: 0;
  font-size: 13px;
  line-height: 1.6;
  color: #334155;
}

.conflicts-section h5 {
  margin: 0 0 8px;
  font-size: 14px;
  color: #92400e;
}

.conflict-card {
  padding: 12px;
  border-radius: 12px;
  background: rgba(255, 251, 235, 0.85);
  border: 1px solid rgba(245, 158, 11, 0.35);
  margin-bottom: 8px;
}

.conflict-head {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 13px;
  color: #78350f;
}

.severity-badge {
  font-size: 11px;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(245, 158, 11, 0.2);
}

.rationale {
  margin: 6px 0;
  font-size: 12px;
  color: #451a03;
}

.sides-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px;
}

.side-box {
  padding: 8px 10px;
  border-radius: 10px;
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.3);
}

.side-box.recommended {
  border-color: #10b981;
  background: rgba(236, 253, 245, 0.65);
}

.side-top {
  display: flex;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: #0f172a;
}

.side-claim {
  margin: 4px 0;
  font-size: 12px;
  color: #334155;
}

.side-meta {
  margin: 0;
  font-size: 11px;
  color: #64748b;
}
</style>
