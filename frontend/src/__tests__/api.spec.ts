import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clarifyResearch,
  createReportAnnotation,
  createResearchPlan,
  drilldownReportSection,
  fetchReportAnnotations,
  fetchReportRevisions,
  fetchRevisionDiff,
  fetchSkillsCatalog,
  replayRequestEvents,
  runResearchStream,
  StreamHttpError,
  triggerTargetedReverify,
  updateResearchPlan
} from "../services/api";

describe("frontend API client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("forwards optional depth on creation and the reviewed plan revision on edits", async () => {
    const requests: unknown[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({ request_id: "draft", plan_revision: 8 }));
    }));

    const tasks = [{ id: 1, title: "Edited", intent: "", query: "" }];
    await createResearchPlan({ topic: "RAG", research_depth: "quick" });
    await updateResearchPlan("draft", tasks, { planRevision: 7 });
    await updateResearchPlan("draft", tasks);
    expect(requests).toEqual([
      { topic: "RAG", research_depth: "quick" },
      { tasks, revision: 7 },
      { tasks }
    ]);
  });

  it("calls HITL clarify, plan, update, drilldown, and replay endpoints with tenant headers", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/research/clarify")) {
        return new Response(
          JSON.stringify({
            topic: "RAG",
            questions: ["关注时间范围?"],
            suggested_angles: ["架构对比"]
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/research/plan")) {
        return new Response(
          JSON.stringify({
            request_id: "req-hitl-1",
            topic: "RAG",
            status: "awaiting_approval",
            phase: "awaiting_approval",
            todo_items: [{ id: 1, title: "Task 1", intent: "I1", query: "RAG" }],
            estimated_duration_seconds: 15
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/requests/req-hitl-1/plan")) {
        return new Response(
          JSON.stringify({
            request_id: "req-hitl-1",
            topic: "RAG",
            status: "awaiting_approval",
            phase: "awaiting_approval",
            todo_items: [{ id: 1, title: "Updated", intent: "I1", query: "RAG" }],
            estimated_duration_seconds: 15
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/requests/req-hitl-1/drilldown")) {
        return new Response(
          JSON.stringify({
            request_id: "req-hitl-1",
            section_title: "评估对比",
            question: "Latency如何?",
            answer_markdown: "P95 < 1.2s"
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/skills")) {
        return new Response(
          JSON.stringify({
            enabled: true,
            domain_packs: [{ pack_id: "tech-selection" }],
            format_packs: [{ pack_id: "exec-brief" }],
            registered_verifiers: ["numeric-audit"]
          }),
          { status: 200 }
        );
      }
      if (url.includes("/requests/req-hitl-1/events")) {
        expect((init?.headers as Record<string, string>)["Last-Event-ID"]).toBe("2");
        expect((init?.headers as Record<string, string>)["X-API-Key"]).toBe("secret-k");
        return new Response(
          JSON.stringify({
            events: [{ id: "3", data: { type: "done" } }]
          }),
          { status: 200 }
        );
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const clarify = await clarifyResearch("RAG", undefined, { apiKey: "secret-k" });
    expect(clarify.questions).toEqual(["关注时间范围?"]);

    const plan = await createResearchPlan({ topic: "RAG" }, { apiKey: "secret-k" });
    expect(plan.request_id).toBe("req-hitl-1");

    const updated = await updateResearchPlan(
      "req-hitl-1",
      [{ id: 1, title: "Updated", intent: "I1", query: "RAG" }],
      { apiKey: "secret-k" }
    );
    expect(updated.todo_items[0].title).toBe("Updated");

    const drill = await drilldownReportSection(
      "req-hitl-1",
      "评估对比",
      "Latency如何?",
      undefined,
      { apiKey: "secret-k" }
    );
    expect(drill.answer_markdown).toBe("P95 < 1.2s");

    const replayed = await replayRequestEvents("req-hitl-1", "2", {
      apiKey: "secret-k"
    });
    expect(replayed).toEqual([{ id: "3", data: { type: "done" } }]);

    const catalog = await fetchSkillsCatalog({ apiKey: "secret-k" });
    expect(catalog.enabled).toBe(true);
    expect(catalog.format_packs[0].pack_id).toBe("exec-brief");
  });

  it("tracks SSE event ids via onEventId callback during streaming", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('id: 1\ndata: {"type":"status","message":"hi"}\n\n'));
        controller.enqueue(encoder.encode('id: 2\ndata: {"type":"done"}\n\n'));
        controller.close();
      }
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        expect(JSON.parse(String(init?.body || "{}"))).toEqual({
          topic: "Agent Eval",
          output_format: "engineering-rfc"
        });
        return new Response(stream, { status: 200 });
      })
    );

    const events: string[] = [];
    const ids: string[] = [];
    await runResearchStream(
      { topic: "Agent Eval", output_format: "engineering-rfc" },
      (ev) => events.push(ev.type),
      { onEventId: (id) => ids.push(id) }
    );

    expect(events).toEqual(["status", "done"]);
    expect(ids).toEqual(["1", "2"]);
  });

  it("handles report annotations, targeted reverify, revision diffs, and 409 conflicts", async () => {
    const mockFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || "GET";
      if (url.endsWith("/requests/req-ann/annotations") && method === "GET") {
        return new Response(
          JSON.stringify({
            request_id: "req-ann",
            state_version: 3,
            current_revision_id: 1,
            annotations: []
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/requests/req-ann/annotations") && method === "POST") {
        const body = JSON.parse(String(init?.body || "{}"));
        if (body.expected_state_version === 1) {
          return new Response("Snapshot state_version conflict", { status: 409 });
        }
        return new Response(
          JSON.stringify({
            request_id: "req-ann",
            state_version: 4,
            annotation: {
              annotation_id: "ann-1",
              request_id: "req-ann",
              tenant_id: "default",
              revision_id: 1,
              section_id: body.section_id,
              claim_id: "",
              target_quote: "",
              comment: body.comment,
              severity: "concern",
              action_requested: "comment",
              author: "reviewer",
              created_at: "2026-11-25T00:00:00Z",
              status: "open"
            }
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/requests/req-ann/reverify")) {
        return new Response(
          JSON.stringify({
            request_id: "req-ann",
            state_version: 5,
            revision: {
              revision_id: 2,
              parent_revision_id: 1,
              request_id: "req-ann",
              report_markdown: "## 结论\n更新",
              reason: "targeted_reverify",
              evidence_refs: ["T1-S1"],
              reverification_summary: { status: "supported" },
              created_at: "2026-11-25T00:01:00Z",
              content_sha256: "abc"
            },
            summary: { status: "supported" },
            report_markdown: "## 结论\n更新"
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/requests/req-ann/revisions")) {
        return new Response(
          JSON.stringify({
            request_id: "req-ann",
            state_version: 5,
            revisions: [{ revision_id: 1, reason: "initial_report", created_at: "2026-11-25T00:00:00Z" }]
          }),
          { status: 200 }
        );
      }
      if (url.includes("/requests/req-ann/revisions/diff?base=1&target=2")) {
        return new Response(
          JSON.stringify({
            request_id: "req-ann",
            base_revision_id: 1,
            target_revision_id: 2,
            added_lines: ["更新"],
            removed_lines: [],
            changed_sections: ["结论"],
            unified_diff: "+更新",
            added_count: 1,
            removed_count: 0
          }),
          { status: 200 }
        );
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal("fetch", mockFetch);

    const annList = await fetchReportAnnotations("req-ann");
    expect(annList.state_version).toBe(3);

    await expect(
      createReportAnnotation("req-ann", { section_id: "结论", comment: "stale", expected_state_version: 1 })
    ).rejects.toMatchObject({ status: 409 });

    const created = await createReportAnnotation("req-ann", {
      section_id: "结论",
      comment: "核查来源",
      expected_state_version: 3
    });
    expect(created.annotation.annotation_id).toBe("ann-1");

    const rev = await triggerTargetedReverify("req-ann", {
      annotation_id: "ann-1",
      expected_state_version: 4
    });
    expect(rev.revision.revision_id).toBe(2);

    const history = await fetchReportRevisions("req-ann");
    expect(history.revisions).toHaveLength(1);

    const diff = await fetchRevisionDiff("req-ann", 1, 2);
    expect(diff.added_count).toBe(1);
    expect(diff.changed_sections).toEqual(["结论"]);
    expect(StreamHttpError).toBeDefined();
  });
});
