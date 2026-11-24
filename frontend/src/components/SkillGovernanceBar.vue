<template>
  <div v-if="hasMetadata" class="skill-gov-bar">
    <span v-if="governance" class="gov-pill gov-pack">
      策略包：<strong>{{ governance.pack_id || "_neutral" }}</strong>
      <template v-if="governance.version">v{{ governance.version }}</template>
      <template v-if="governance.maturity">({{ governance.maturity }})</template>
    </span>

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
import type { ClaimVerificationSummary, SkillGovernanceInfo } from "../types/research";

const props = withDefaults(
  defineProps<{
    governance?: SkillGovernanceInfo | null;
    claimSummary?: ClaimVerificationSummary | null;
    conflictCount?: number;
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
