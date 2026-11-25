import { shallowRef } from "vue";
import type { BudgetSnapshot, ResearchStreamEvent, V4ResearchProjection } from "../types/research";

function readVersion(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function readProjection(value: unknown): V4ResearchProjection | undefined {
  if (value && typeof value === "object" && "schema_version" in value && value.schema_version === 1) {
    return value as V4ResearchProjection;
  }
}

export function createResearchV4State() {
  const budget = shallowRef<BudgetSnapshot>();
  const prunedTaskIds = shallowRef(new Set<number>());
  const seenEventKeys = new Set<string>();
  let requestId = "";
  let snapshotVersion: number | undefined;
  let budgetVersion: number | undefined;
  let dagRevision: number | undefined;

  function reset(nextRequestId = "") {
    requestId = nextRequestId;
    budget.value = undefined;
    prunedTaskIds.value = new Set();
    seenEventKeys.clear();
    snapshotVersion = undefined;
    budgetVersion = undefined;
    dagRevision = undefined;
  }

  function mergeBudget(value: unknown, version?: number) {
    if (!value || typeof value !== "object" || (version !== undefined && version < (budgetVersion ?? 0))) return;
    budget.value = value as BudgetSnapshot;
    budgetVersion = version ?? budgetVersion;
  }

  function mergeReplan(revision: number, ids: unknown) {
    dagRevision = revision;
    if (Array.isArray(ids)) {
      prunedTaskIds.value = new Set(ids.filter((id) => (readVersion(id) ?? 0) > 0));
    }
  }

  function hydrate(targetRequestId: string, snapshot: Record<string, unknown>) {
    if (targetRequestId !== requestId) reset(targetRequestId);
    const projection = readProjection(snapshot.v4);
    if (!projection) return;

    const version = readVersion(snapshot.state_version);
    if (version !== undefined && version < (snapshotVersion ?? 0)) return;
    snapshotVersion = version ?? snapshotVersion;
    mergeBudget(projection.budget, version);
    const revision = readVersion(projection.replan?.dag_revision);
    if (revision !== undefined && revision >= (dagRevision ?? 0)) {
      mergeReplan(revision, projection.replan?.pruned_task_ids);
    }
  }

  function accept(targetRequestId: string | null, event: ResearchStreamEvent): boolean {
    const budgetUpdate = event.type === "budget_updated";
    const replanUpdate = event.type === "research_replanned";
    const projection = readProjection(event.v4);
    const revision = readVersion(event.dag_revision) ?? readVersion(projection?.replan?.dag_revision);
    if (!budgetUpdate && !replanUpdate && !projection && revision === undefined) return true;
    if (typeof event.request_id === "string" && targetRequestId && event.request_id !== targetRequestId) return false;
    const eventRequestId = targetRequestId || (typeof event.request_id === "string" ? event.request_id : "");
    if (eventRequestId !== requestId) reset(eventRequestId);

    const version = readVersion(event.state_version);
    if (version !== undefined && version < (snapshotVersion ?? 0)) return false;
    if (revision !== undefined && revision < (dagRevision ?? 0)) return false;

    const key = typeof event.event_key === "string" ? event.event_key : "";
    if ((budgetUpdate || replanUpdate) && (event.schema_version !== 1 || !eventRequestId || !key)) return false;
    if (replanUpdate && (revision === undefined || revision === dagRevision)) return false;
    if (key) {
      if (seenEventKeys.has(key) || (version !== undefined && version === snapshotVersion)) return false;
      seenEventKeys.add(key);
    }

    mergeBudget(budgetUpdate ? event.budget : projection?.budget, version);
    if (revision !== undefined) {
      mergeReplan(revision, replanUpdate ? event.pruned_task_ids : projection?.replan?.pruned_task_ids);
    }
    return true;
  }

  return { budget, prunedTaskIds, reset, hydrate, accept };
}
