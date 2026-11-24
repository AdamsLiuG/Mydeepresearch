import { describe, expect, it } from "vitest";
import type { EvidenceConflictCluster, TodoTaskView } from "../types/research";
import {
  buildFallbackWorkbenchPayload,
  conflictsForCitation,
  decorateReportHtmlWithCitations,
  extractCitationIds
} from "../utils/workbench";

describe("interactive workbench utilities", () => {
  it("extracts ordered unique citation IDs across text/multimodal/computed modalities", () => {
    const ids = extractCitationIds(
      "年报显示营收增长 [T1-S1]，图表测算见 [T2-M1]，验证见 [T1-S1] 与 [T3-C2]。"
    );
    expect(ids).toEqual(["T1-S1", "T2-M1", "T3-C2"]);
  });

  it("decorates sanitized HTML citation tokens without mutating tag attributes", () => {
    const html =
      '<p>核心结论 <a href="https://example.com/[T1-S1]">原文</a> 数据见 [T1-S1] 与 [T2-C1]。</p>';
    const decorated = decorateReportHtmlWithCitations(html);
    expect(decorated).toContain('href="https://example.com/[T1-S1]"');
    expect(decorated).toContain(
      '<button type="button" class="citation-chip" data-citation-id="T1-S1">[T1-S1]</button>'
    );
    expect(decorated).toContain(
      '<button type="button" class="citation-chip" data-citation-id="T2-C1">[T2-C1]</button>'
    );
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
