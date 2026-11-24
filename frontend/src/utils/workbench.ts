import type {
  ClaimVerificationSummary,
  EvidenceConflictCluster,
  ReportWorkbenchPayload,
  SkillGovernanceInfo,
  TodoTaskView,
  WorkbenchCitationEntry,
  WorkbenchParagraphBinding
} from "../types/research";

export const CITATION_TOKEN_REGEX = /\[(T\d+-(?:S|M|C)\d+)\]/g;

export function extractCitationIds(text: string): string[] {
  const input = typeof text === "string" ? text : "";
  if (!input) {
    return [];
  }
  const seen = new Set<string>();
  const ordered: string[] = [];
  const pattern = new RegExp(CITATION_TOKEN_REGEX.source, "g");
  let match: RegExpExecArray | null = pattern.exec(input);
  while (match !== null) {
    const sid = match[1];
    if (sid && !seen.has(sid)) {
      seen.add(sid);
      ordered.push(sid);
    }
    match = pattern.exec(input);
  }
  return ordered;
}

export function decorateReportHtmlWithCitations(sanitizedHtml: string): string {
  if (typeof sanitizedHtml !== "string" || !sanitizedHtml) {
    return "";
  }
  // Split into HTML tags vs text segments so href/attributes are never mutated.
  const segments = sanitizedHtml.split(/(<[^>]+>)/g);
  return segments
    .map((segment) => {
      if (!segment || segment.startsWith("<")) {
        return segment;
      }
      return segment.replace(
        new RegExp(CITATION_TOKEN_REGEX.source, "g"),
        (_full, sid: string) =>
          `<button type="button" class="citation-chip" data-citation-id="${sid}">[${sid}]</button>`
      );
    })
    .join("");
}

export function conflictsForCitation(
  workbench: ReportWorkbenchPayload | null | undefined,
  sourceId: string | null | undefined
): EvidenceConflictCluster[] {
  if (!workbench || !sourceId) {
    return [];
  }
  const target = sourceId.trim();
  return (workbench.evidence_conflicts || []).filter((cluster) =>
    (cluster.sides || []).some((side) => side.source_id === target)
  );
}

export function buildFallbackWorkbenchPayload(
  todoItems: TodoTaskView[],
  reportMarkdown: string,
  evidenceConflicts: EvidenceConflictCluster[] = [],
  claimSummary: ClaimVerificationSummary = {},
  skillGovernance: SkillGovernanceInfo = { pack_id: "_neutral" }
): ReportWorkbenchPayload {
  const citation_index: Record<string, WorkbenchCitationEntry> = {};
  for (const task of todoItems || []) {
    for (const src of task.sourceItems || []) {
      const sid = (src.sourceId || "").trim();
      if (!sid) {
        continue;
      }
      citation_index[sid] = {
        source_id: sid,
        task_id: task.id,
        title: src.title || sid,
        url: src.url || "",
        domain: src.domain || "",
        snippet: src.snippet || src.raw || "",
        quality_label: src.qualityLabel || "medium",
        source_type: src.sourceType || "web_general",
        published_at: src.publishedAt || null,
        modality: sid.includes("-C") ? "computed" : sid.includes("-M") ? "multimodal" : "text",
        anchors: []
      };
    }
  }

  const conflictIdsBySource = new Map<string, string[]>();
  for (const cluster of evidenceConflicts || []) {
    for (const side of cluster.sides || []) {
      const list = conflictIdsBySource.get(side.source_id) || [];
      if (!list.includes(cluster.conflict_id)) {
        list.push(cluster.conflict_id);
      }
      conflictIdsBySource.set(side.source_id, list);
    }
  }

  const paragraph_bindings: WorkbenchParagraphBinding[] = [];
  const paragraphs = (reportMarkdown || "")
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);
  paragraphs.forEach((para, idx) => {
    const cids = extractCitationIds(para);
    if (!cids.length) {
      return;
    }
    const matchedConflicts: string[] = [];
    for (const cid of cids) {
      for (const confId of conflictIdsBySource.get(cid) || []) {
        if (!matchedConflicts.includes(confId)) {
          matchedConflicts.push(confId);
        }
      }
    }
    paragraph_bindings.push({
      paragraph_index: idx,
      excerpt: para.slice(0, 180),
      citation_ids: cids,
      conflict_ids: matchedConflicts
    });
  });

  return {
    citation_index,
    paragraph_bindings,
    evidence_conflicts: evidenceConflicts,
    claim_verification_summary: claimSummary,
    skill_governance: skillGovernance
  };
}
