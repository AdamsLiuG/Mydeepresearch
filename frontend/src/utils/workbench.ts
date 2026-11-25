import type {
  ChartSpecArtifact,
  ClaimVerificationSummary,
  EvidenceConflictCluster,
  ReportWorkbenchPayload,
  SkillGovernanceInfo,
  TodoTaskView,
  VerifierBadge,
  WorkbenchCitationEntry,
  WorkbenchParagraphBinding
} from "../types/research";

export const CITATION_TOKEN_REGEX = /\[(T\d+-(?:S|M|C)\d+)\]/g;
const citationId = /^T\d+-(?:S|M|C)\d+$/;

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

export function enhanceReportCitations(container: HTMLElement): void {
  const doc = container.ownerDocument;
  const walker = doc.createTreeWalker(container, 4); // NodeFilter.SHOW_TEXT
  const nodes: Text[] = [];
  let node: Node | null;
  while ((node = walker.nextNode())) {
    if (!node.parentElement?.closest("a, code, pre, button, svg")) {
      nodes.push(node as Text);
    }
  }
  for (const text of nodes) {
    const matches = [...text.data.matchAll(CITATION_TOKEN_REGEX)];
    if (!matches.length) {
      continue;
    }
    const fragment = doc.createDocumentFragment();
    let end = 0;
    for (const match of matches) {
      fragment.append(text.data.slice(end, match.index));
      const button = doc.createElement("button");
      button.type = "button";
      button.className = "citation-chip";
      button.dataset.citationId = match[1];
      button.textContent = match[0];
      fragment.append(button);
      end = match.index + match[0].length;
    }
    fragment.append(text.data.slice(end));
    text.replaceWith(fragment);
  }
}

export function safeEvidenceUrl(raw: unknown): string {
  if (typeof raw !== "string" || !/^https?:\/\//i.test(raw) || /[\u0000-\u0020\\\u007f]/.test(raw)) {
    return "";
  }
  try {
    const url = new URL(raw);
    return url.username || url.password ? "" : url.href;
  } catch {
    return "";
  }
}

export function exportTableToCsv(rows: string[][]): string {
  return "\ufeff" + rows.map((row) => row.map((cell) => {
    const safe = /^(?:[\t\r\n]|\s*[=+@-])/u.test(cell) ? "'" + cell : cell;
    return `"${safe.replaceAll('"', '""')}"`;
  }).join(",")).join("\r\n");
}

export function exportTableToMarkdown(rows: string[][]): string {
  if (!rows.length) {
    return "";
  }
  const escape = (cell: string) => cell.replace(/\r?\n|\r/g, " ").replace(/[&<>\\|`*_[\]!]/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char] || `\\${char}`));
  const [header, ...body] = rows;
  return [header, header.map(() => "---"), ...body].map((row, index) =>
    `| ${row.map(index === 1 ? (cell) => cell : escape).join(" | ")} |`
  ).join("\n");
}

export function sanitizeChartSpec(raw: unknown): ChartSpecArtifact | null {
  const chart = record(raw);
  const { library } = chart;
  if (library !== "echarts" && library !== "vega-lite") {
    return null;
  }
  const spec = record(chart.spec);
  let points = 0;
  const safe = (value: unknown, depth: number): boolean => {
    if (depth > 24) return false;
    if (value === null || typeof value === "boolean") return true;
    if (typeof value === "number") return Number.isFinite(value);
    if (typeof value === "string") {
      return !/[<>]|=>|\bfunction\s*\(|(?:javascript|vbscript|https?|file|ftp|data|image):|^\s*\/\//i.test(value);
    }
    if (Array.isArray(value)) {
      points += value.length;
      return points <= 2000 && value.every((entry) => safe(entry, depth + 1));
    }
    if (!value || typeof value !== "object") return false;
    return Object.entries(value).every(([key, child]) =>
      !/^(?:url|href|__proto__|prototype|constructor|expr|calculate)$/i.test(key) &&
      !(/^(?:type|mark)$/i.test(key) && typeof child === "string" && /^(?:image|html)$/i.test(child)) &&
      safe(child, depth + 1));
  };
  if (!Object.keys(spec).length || !safe(spec, 0) ||
      new TextEncoder().encode(JSON.stringify(spec)).byteLength > 64 * 1024) {
    return null;
  }
  return {
    chart_id: typeof chart.chart_id === "string" ? chart.chart_id : "",
    title: typeof chart.title === "string" ? chart.title : "数据图表",
    library, spec,
    source_ids: Array.isArray(chart.source_ids) ? chart.source_ids.filter((id): id is string =>
      typeof id === "string" && citationId.test(id)) : [],
    caption: typeof chart.caption === "string" ? chart.caption : undefined
  };
}

interface ChartSeries {
  name: string;
  type: "line" | "bar" | "point";
  points: Array<{ label: string; value: number }>;
}

export function getChartSeries(chart: ChartSpecArtifact): ChartSeries[] {
  const text = (value: unknown, fallback: string): string =>
    typeof value === "string" || typeof value === "number" ? String(value) : fallback;
  const point = (label: unknown, value: unknown, index: number): ChartSeries["points"] =>
    typeof value === "number" ? [{ label: text(label, String(index + 1)), value }] : [];

  if (chart.library === "vega-lite") {
    const mark = typeof chart.spec.mark === "string" ? chart.spec.mark : record(chart.spec.mark).type;
    const data = record(chart.spec.data).values;
    const encoding = record(chart.spec.encoding);
    const x = record(encoding.x).field;
    const y = record(encoding.y).field;
    if (!Array.isArray(data) || typeof x !== "string" || typeof y !== "string" ||
        (mark !== "bar" && mark !== "line" && mark !== "point" && mark !== "circle")) return [];
    const points = data.flatMap((row, index) => point(record(row)[x], record(row)[y], index));
    return points.length ? [{ name: chart.title, type: mark === "circle" ? "point" : mark, points }] : [];
  }

  const xAxis = record(Array.isArray(chart.spec.xAxis) ? chart.spec.xAxis[0] : chart.spec.xAxis);
  const labels = Array.isArray(xAxis.data) ? xAxis.data : [];
  return (Array.isArray(chart.spec.series) ? chart.spec.series : []).flatMap((raw) => {
    const series = record(raw);
    const type = series.type === "scatter" ? "point" : series.type;
    if ((type !== "bar" && type !== "line" && type !== "point") || !Array.isArray(series.data)) return [];
    const points = series.data.flatMap((entry, index) => {
      const value = record(entry).value ?? entry;
      return Array.isArray(value) ? point(value[0], value[1], index) :
        point(record(entry).name ?? labels[index], value, index);
    });
    return points.length ? [{ name: text(series.name, chart.title), type, points }] : [];
  });
}

export function enhanceReportTables(container: HTMLElement): void {
  const doc = container.ownerDocument;
  for (const [index, table] of Array.from(container.querySelectorAll("table")).entries()) {
    const header = table.tHead?.rows[0];
    if (!header || header.querySelector("button")) continue;
    const controls = doc.createElement("div");
    controls.className = "table-export";
    const exportButton = (label: string, extension: string, encode: (rows: string[][]) => string) => {
      const button = doc.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.addEventListener("click", () => {
        const rows = Array.from(table.rows, (row) => Array.from(row.cells, (cell) => cell.textContent || ""));
        const url = URL.createObjectURL(new Blob([encode(rows)], { type: "text/plain;charset=utf-8" }));
        const link = doc.createElement("a");
        link.href = url;
        link.download = `table-${index + 1}.${extension}`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 0);
      });
      controls.append(button);
    };
    exportButton("导出 CSV", "csv", exportTableToCsv);
    exportButton("导出 Markdown", "md", exportTableToMarkdown);
    table.before(controls);
    for (const [column, cell] of Array.from(header.cells).entries()) {
      if (cell.querySelector("a, button")) continue;
      const button = doc.createElement("button");
      button.type = "button";
      button.className = "table-sort";
      button.append(...Array.from(cell.childNodes));
      cell.append(button);
      button.addEventListener("click", () => {
        const ascending = cell.getAttribute("aria-sort") !== "ascending";
        for (const other of Array.from(header.cells)) other.removeAttribute("aria-sort");
        cell.setAttribute("aria-sort", ascending ? "ascending" : "descending");
        for (const body of Array.from(table.tBodies)) {
          Array.from(body.rows).sort((a, b) => {
            const left = a.cells[column]?.textContent?.trim() || "";
            const right = b.cells[column]?.textContent?.trim() || "";
            const numeric = left && right && Number.isFinite(Number(left)) && Number.isFinite(Number(right));
            const order = numeric ? Number(left) - Number(right) : left.localeCompare(right, undefined, { numeric: true });
            return ascending ? order : -order;
          }).forEach((row) => body.append(row));
        }
      });
    }
  }
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function entriesFor(value: unknown): Record<string, WorkbenchCitationEntry> {
  const entries: Record<string, WorkbenchCitationEntry> = {};
  for (const [sourceId, raw] of Object.entries(record(value))) {
    if (!citationId.test(sourceId)) {
      continue;
    }
    const fields = record(raw);
    entries[sourceId] = {
      title: sourceId, url: "", domain: "", snippet: "", quality_label: "",
      ...fields, source_id: sourceId
    };
  }
  return entries;
}

export function normalizeReportWorkbench(payload: Record<string, unknown>): ReportWorkbenchPayload {
  const workbench = record(payload.workbench);
  const charts = payload.charts_spec ?? workbench.charts_spec ?? payload.charts ?? workbench.charts ?? payload.report_charts;
  const badges = payload.verifier_badges ?? workbench.verifier_badges;
  const conflicts = payload.evidence_conflicts ?? workbench.evidence_conflicts;
  const governance = record(workbench.skill_governance);
  return {
    citation_index: {
      ...entriesFor(workbench.citation_index),
      ...entriesFor(workbench.citation_drawer_map),
      ...entriesFor(payload.citation_drawer_map)
    },
    paragraph_bindings: Array.isArray(workbench.paragraph_bindings) ? workbench.paragraph_bindings : [],
    evidence_conflicts: Array.isArray(conflicts) ? conflicts : [],
    claim_verification_summary: record(payload.claim_verification_summary ?? workbench.claim_verification_summary),
    skill_governance: { pack_id: "_neutral", ...record(payload.skill_pack ?? payload.skill_pack_metadata), ...governance },
    verifier_badges: Array.isArray(badges) ? badges.filter((badge): badge is VerifierBadge =>
      typeof record(badge).badge_id === "string" && typeof record(badge).status === "string") : [],
    charts_spec: Array.isArray(charts) ? charts : []
  };
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
