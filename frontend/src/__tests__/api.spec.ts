import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clarifyResearch,
  createResearchPlan,
  drilldownReportSection,
  fetchSkillsCatalog,
  replayRequestEvents,
  runResearchStream,
  updateResearchPlan
} from "../services/api";

describe("frontend API client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
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
});
