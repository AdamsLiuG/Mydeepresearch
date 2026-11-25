<template>
  <div v-if="hasMetadata" class="skill-gov-bar">
    <span v-if="catalog" class="gov-pill">Skill System {{ catalog.enabled ? "已启用" : "已关闭" }}</span>
    <span v-if="governance" class="gov-pill gov-pack">
      策略包：<strong>{{ governance.pack_id || "_neutral" }}</strong>
      <template v-if="governance.version">v{{ governance.version }}</template>
      <template v-if="governance.maturity">({{ governance.maturity }})</template>
    </span>
    <span v-if="governance?.secondary_pack_id" class="gov-pill">辅助包 {{ governance.secondary_pack_id }}</span>
    <span v-if="governance?.format_pack_id" class="gov-pill">格式 {{ activeFormat?.title || governance.format_pack_id }}</span>
    <span v-if="governance?.routing_method" class="gov-pill">
      {{ governance.routing_method }}<template v-if="routingScore"> · {{ routingScore }}</template>
    </span>
    <span v-if="governance?.shadow || activeDomain?.status === 'shadow'" class="gov-pill">Shadow</span>
    <span v-if="governance?.code_sandbox_enabled || activeDomain?.code_sandbox" class="gov-pill">计算沙箱</span>
    <span v-if="governance?.mcp_tool_scopes && Object.keys(governance.mcp_tool_scopes).length" class="gov-pill">MCP</span>

    <span
      v-for="verifier in governance?.active_verifiers || []"
      :key="verifier"
      class="gov-pill gov-verifier"
    >
      ✓ {{ verifier }}
    </span>

    <span v-if="governance?.nli_verifier_enabled" class="gov-pill gov-nli">
      NLI 核验开启
    </span>

    <span v-if="faithfulnessText" class="gov-pill gov-score">
      原子一致率：{{ faithfulnessText }}
    </span>
    <span v-for="badge in verifierBadges" :key="badge.badge_id" class="gov-pill">
      {{ badge.label || badge.badge_id }} · {{ badge.status }}
    </span>
    <details v-if="catalog?.registered_verifiers.length">
      <summary>已注册核验器 ({{ catalog.registered_verifiers.length }})</summary>
      {{ catalog.registered_verifiers.join("、") }}
    </details>

    <button
      v-if="conflictCount > 0"
      type="button"
      class="gov-pill gov-conflict-btn"
      @click="$emit('inspect-conflicts')"
    >
      跨源分歧：{{ conflictCount }} 处
    </button>
  </div>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import type { ClaimVerificationSummary, SkillGovernanceInfo, SkillsCatalogResponse, VerifierBadge } from "../types/research";

const props = withDefaults(
  defineProps<{
    governance?: SkillGovernanceInfo | null;
    claimSummary?: ClaimVerificationSummary | null;
    conflictCount?: number;
    catalog?: SkillsCatalogResponse | null;
    verifierBadges?: VerifierBadge[];
  }>(),
  {
    governance: null,
    claimSummary: null,
    conflictCount: 0
  }
);

defineEmits<{
  "inspect-conflicts": [];
}>();

const faithfulnessText = computed(() => {
  const rate = props.claimSummary?.atomic_faithfulness_rate;
  if (typeof rate !== "number" || !Number.isFinite(rate)) {
    return "";
  }
  return `${(rate * 100).toFixed(1)}%`;
});

const hasMetadata = computed(() => {
  if (props.catalog || props.verifierBadges?.length) return true;
  const gov = props.governance;
  if (!gov) {
    return props.conflictCount > 0 || Boolean(faithfulnessText.value);
  }
  return Boolean(
    (gov.pack_id && gov.pack_id !== "_neutral") ||
      (gov.active_verifiers && gov.active_verifiers.length > 0) ||
      gov.nli_verifier_enabled ||
      faithfulnessText.value ||
      props.conflictCount > 0
  );
});

const activeDomain = computed(() => props.catalog?.domain_packs.find((pack) => pack.pack_id === props.governance?.pack_id));
const activeFormat = computed(() => props.catalog?.format_packs.find((pack) => pack.pack_id === props.governance?.format_pack_id));
const routingScore = computed(() => {
  const score = props.governance?.match_score;
  return typeof score === "number" && Number.isFinite(score) ? score.toFixed(2) : "";
});
</script>

<style scoped>
.skill-gov-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  padding: 8px 12px;
  border-radius: 12px;
  background: rgba(241, 245, 249, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.28);
  font-size: 12px;
}

.gov-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 9px;
  border-radius: 999px;
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.35);
  color: #334155;
}

.gov-verifier {
  color: #047857;
  border-color: rgba(16, 185, 129, 0.35);
  background: rgba(236, 253, 245, 0.9);
}

.gov-nli,
.gov-score {
  color: #1d4ed8;
  border-color: rgba(59, 130, 246, 0.3);
  background: rgba(239, 246, 255, 0.9);
}

.gov-conflict-btn {
  cursor: pointer;
  color: #b45309;
  border-color: rgba(245, 158, 11, 0.45);
  background: rgba(254, 243, 199, 0.9);
  font-weight: 600;
}
</style>
