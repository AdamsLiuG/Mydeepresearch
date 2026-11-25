import { describe, expect, it } from "vitest";
import type { EvidenceConflictCluster, TodoTaskView } from "../types/research";
import {
  buildFallbackWorkbenchPayload,
  computeLocalLineDiff,
  conflictsForCitation,
  enhanceReportCitations,
  enhanceReportTables,
  exportTableToCsv,
  extractCitationIds,
  getChartSeries,
  normalizeReportWorkbench,
  safeEvidenceUrl,
  sanitizeChartSpec,
  summarizeRevisionLabel
} from "../utils/workbench";

const runtime = globalThis as typeof globalThis & { process?: { features?: { require_module?: boolean } } };
const jsdom = runtime.process?.features?.require_module ? await import("jsdom") : null;

describe("interactive workbench utilities", () => {
  it("extracts ordered unique citation IDs across text/multimodal/computed modalities", () => {
    const ids = extractCitationIds(
      "年报显示营收增长 [T1-S1]，图表测算见 [T2-M1]，验证见 [T1-S1] 与 [T3-C2]。"
    );
    expect(ids).toEqual(["T1-S1", "T2-M1", "T3-C2"]);
  });

  it.skipIf(!jsdom)("decorates sanitized HTML citation tokens without mutating tag attributes", () => {
    const { window } = new jsdom!.JSDOM(
      '<main><a href="https://example.com/?x=>[T1-M1]">[T1-M1]</a> <code>[T2-S1]</code> ' +
      '<pre><em>[T3-S1]</em></pre><p>报告 [T1-S1] 与 [T2-C1]。</p></main>'
    );
    const container = window.document.querySelector("main")!;
    enhanceReportCitations(container);
    expect(container.querySelector("a")?.getAttribute("href")).toBe("https://example.com/?x=>[T1-M1]");
    expect(Array.from(container.querySelectorAll("button"), (node) => node.dataset.citationId))
      .toEqual(["T1-S1", "T2-C1"]);
    window.close();
  });

  it("accepts only absolute HTTP(S) evidence links", () => {
    expect(safeEvidenceUrl("https://example.test/source.pdf#page=2"))
      .toBe("https://example.test/source.pdf#page=2");
    for (const value of ["javascript:void(0)", "data:image/svg+xml,chart", "//example.test/a", "#local", "https://user@example.test"]) {
      expect(safeEvidenceUrl(value)).toBe("");
    }
  });

  it("exports quoted CSV cells and protects spreadsheet-looking text", () => {
    expect(exportTableToCsv([["Label", "Value"], ['a,"b"', "  =SUM(1)"], ["x", "\r123"], ["y", "-42"]]))
      .toBe('\ufeff"Label","Value"\r\n"a,""b""","\'  =SUM(1)"\r\n"x","\'\r123"\r\n"y","\'-42"');
  });

  it.skipIf(!jsdom)("sorts report table rows numerically in both directions", () => {
    const { window } = new jsdom!.JSDOM(
      '<main><table><thead><tr><th>Count</th></tr></thead><tbody><tr><td>15</td></tr><tr><td>2</td></tr></tbody></table></main>'
    );
    const container = window.document.querySelector("main")!;
    enhanceReportTables(container);
    const cell = container.querySelector("th")!;
    const values = () => Array.from(container.querySelectorAll("tbody td"), (node) => node.textContent);
    cell.querySelector("button")!.click();
    expect(values()).toEqual(["2", "15"]);
    expect(cell.getAttribute("aria-sort")).toBe("ascending");
    cell.querySelector("button")!.click();
    expect(values()).toEqual(["15", "2"]);
    expect(cell.getAttribute("aria-sort")).toBe("descending");
    expect(container.querySelectorAll(".table-export button")).toHaveLength(2);
    window.close();
  });

  it("consumes canonical and previous report workbench shapes", () => {
    const workbench = normalizeReportWorkbench({
      citation_drawer_map: { "T1-C1": { title: "computed", page_no: 7, excerpt: "row", derived_from_source_ids: ["T1-S1"] } },
      verifier_badges: [{ badge_id: "numeric", status: "passed" }],
      workbench: { citation_index: { "T1-S1": { title: "original" } }, skill_governance: { pack_id: "finance" } },
      charts: [{ chart_id: "old" }]
    });
    expect(workbench.citation_index["T1-C1"]).toMatchObject({ source_id: "T1-C1", title: "computed", page_no: 7, excerpt: "row", derived_from_source_ids: ["T1-S1"] });
    expect(workbench.citation_index["T1-S1"].title).toBe("original");
    expect(workbench.verifier_badges).toEqual([{ badge_id: "numeric", status: "passed" }]);
    expect(workbench.charts_spec).toEqual([{ chart_id: "old" }]);
    expect(workbench.skill_governance.pack_id).toBe("finance");
  });

  it("extracts safe inline chart series and rejects remote or executable settings", () => {
    const chart = { chart_id: "sales", title: "Sales", library: "vega-lite", spec: {
      data: { values: [{ label: "Q1", value: 12 }] }, mark: "bar",
      encoding: { x: { field: "label" }, y: { field: "value" } }
    } };
    expect(getChartSeries(sanitizeChartSpec(chart)!)).toEqual([
      { name: "Sales", type: "bar", points: [{ label: "Q1", value: 12 }] }
    ]);
    const echarts = sanitizeChartSpec({ ...chart, library: "echarts", spec: {
      xAxis: { data: ["Q1", "Q2"] }, series: [{ type: "line", data: [1, -2] }]
    } });
    expect(getChartSeries(echarts!)[0].points).toEqual([{ label: "Q1", value: 1 }, { label: "Q2", value: -2 }]);
    for (const spec of [
      { data: { url: "/private" } }, { tooltip: { formatter: "(x) => x" } }, { mark: "image", url: "/a" },
      { text: "<svg/>" }, { data: { values: Array(2001).fill(1) } }, { text: "思".repeat(22000) }
    ]) {
      expect(sanitizeChartSpec({ ...chart, spec })).toBeNull();
    }
  });

  it("builds fallback workbench bindings and filters conflicts per citation", () => {
    const tasks: TodoTaskView[] = [
      {
        id: 1,
        title: "财务分析",
        intent: "营收核查",
        query: "revenue",
        status: "completed",
        summary: "营收 100 亿 [T1-S1]",
        sourcesSummary: "",
        sourceItems: [
          {
            sourceId: "T1-S1",
            title: "官方年报",
            url: "https://official.example.com",
            snippet: "2025年营收100亿元",
            raw: "",
            domain: "official.example.com",
            sourceType: "official_documentation",
            qualityLabel: "high",
            freshnessLabel: "recent",
            publishedAt: "2025-03-01",
            providerCount: 1
          },
          {
            sourceId: "T1-C1",
            title: "沙盒计算",
            url: "",
            snippet: "CAGR 25%",
            raw: "",
            domain: "sandbox",
            sourceType: "computed_sandbox",
            qualityLabel: "high",
            freshnessLabel: "recent",
            publishedAt: "",
            providerCount: 1
          }
        ],
        notices: [],
        claims: [],
        reviewIssues: [],
        reviewStatus: "passed",
        noteId: null,
        notePath: null,
        toolCalls: [],
        origin: "planned",
        round: 1
      }
    ];
    const conflicts: EvidenceConflictCluster[] = [
      {
        conflict_id: "CONF-1",
        entity: "vendor",
        metric: "revenue",
        period: "2025",
        conflict_type: "numeric_divergence",
        severity: "low",
        contradiction_score: 0.8,
        sides: [
          {
            source_id: "T1-S1",
            task_id: 1,
            title: "官方年报",
            url: "https://official.example.com",
            domain: "official.example.com",
            source_type: "official_documentation",
            quality_label: "high",
            claim_text: "100亿元"
          },
          {
            source_id: "T2-S1",
            task_id: 2,
            title: "估算来源",
            url: "https://est.example.com",
            domain: "est.example.com",
            source_type: "web_general",
            quality_label: "medium",
            claim_text: "130亿元"
          }
        ],
        adjudication_status: "resolved_by_authority",
        recommended_source_id: "T1-S1",
        adjudication_rationale: "采信官方口径"
      }
    ];

    const workbench = buildFallbackWorkbenchPayload(
      tasks,
      "## 结论\n\n年报 100 亿元 [T1-S1]，计算验证 [T1-C1]。",
      conflicts,
      { atomic_faithfulness_rate: 0.95 },
      { pack_id: "finance-research", version: 2 }
    );

    expect(Object.keys(workbench.citation_index)).toEqual(["T1-S1", "T1-C1"]);
    expect(workbench.citation_index["T1-C1"].modality).toBe("computed");
    expect(workbench.paragraph_bindings).toHaveLength(1);
    expect(workbench.paragraph_bindings[0].citation_ids).toEqual(["T1-S1", "T1-C1"]);
    expect(workbench.paragraph_bindings[0].conflict_ids).toEqual(["CONF-1"]);
    expect(conflictsForCitation(workbench, "T1-S1")).toHaveLength(1);
    expect(conflictsForCitation(workbench, "T1-C1")).toHaveLength(0);
  });
});

it("keeps stable report bindings and deduplicates replayed chunks and statuses", async () => {
  const { createReportStreamState } = await import("../utils/workbench");
  const stream = createReportStreamState();
  const outline = { type: "report_outline", outline_version: 1, sections: ["Evidence"], section_ids: ["sec-a"],
    event_key: "report-outline-1" };
  expect(stream.accept(outline, "req")).toBe(true);
  stream.outline(outline);
  const completed = { type: "report_section_status", outline_version: 1, section_id: "sec-a",
    event_key: "report-section-1-completed", status: "completed" };
  expect(stream.accept(completed, "req")).toBe(true);
  expect(stream.sectionStatus(completed)).toEqual({ title: "Evidence", status: "completed" });
  expect(stream.accept(completed, "req")).toBe(false);
  expect(stream.sectionStatus({ ...completed, status: "started" })).toBeUndefined();
  const second = { type: "report_chunk", event_key: "report-final-1-chunk-2", report_version: 1, index: 2, content: "B" };
  expect(stream.accept(second, "req")).toBe(true);
  expect(stream.chunk(second)).toBe("");
  const first = { ...second, event_key: "report-final-1-chunk-1", index: 1, content: "A" };
  expect(stream.accept(first, "req")).toBe(true);
  expect(stream.chunk(first)).toBe("AB");
  expect(stream.accept(outline, "req")).toBe(false);
  const binding = { section_id: "sec-a", title: "Evidence", paragraph_index: 2 };
  expect(normalizeReportWorkbench({ workbench: { section_bindings: [binding, binding] } }).section_bindings)
    .toEqual([binding]);
  expect(normalizeReportWorkbench({})).not.toHaveProperty("section_bindings");

  const diff = computeLocalLineDiff("## 结论\n旧内容", "## 结论\n新内容\n> 定向复核");
  expect(diff.changed).toBe(true);
  expect(diff.removed_lines).toEqual(["旧内容"]);
  expect(diff.added_lines).toEqual(["新内容", "> 定向复核"]);
  expect(
    summarizeRevisionLabel({
      revision_id: 2,
      reason: "targeted_reverify",
      created_at: "2026-11-25T00:00:00Z"
    })
  ).toBe("v2 · 定向复核");

  const fsSpecifier = "node:fs";
  const fsMod = (await import(/* @vite-ignore */ fsSpecifier)) as {
    readFileSync: (path: URL, encoding: string) => string;
  };
  const viewerUrl = new URL("../components/ReportViewer.vue", import.meta.url);
  const viewerSource = fsMod.readFileSync(viewerUrl, "utf-8");
  const panelStart = viewerSource.indexOf('<div v-if="showAnnotations');
  expect(panelStart).toBeGreaterThan(-1);
  const panelSnippet = viewerSource.slice(panelStart, viewerSource.lastIndexOf("</template>"));
  expect(panelSnippet).not.toMatch(/v-html|innerHTML/);
  expect(panelSnippet).toContain("{{ item.comment }}");
  expect(panelSnippet).toContain("{{ line }}");
});
