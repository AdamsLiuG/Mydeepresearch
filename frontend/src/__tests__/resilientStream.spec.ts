import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ACTIVE_STREAM_STORAGE_KEY,
  clearActiveStreamSession,
  computeReconnectDelayMs,
  loadActiveStreamSession,
  saveActiveStreamSession,
  useResilientStream,
  type StorageLike
} from "../composables/useResilientStream";
import { sendWorkflowSignal, streamRequestEvents, submitAsyncResearch } from "../services/api";
import { createResearchV4State } from "../utils/researchV4";

function makeMemoryStorage(): StorageLike {
  const store = new Map<string, string>();
  return {
    getItem: (k) => store.get(k) ?? null,
    setItem: (k, v) => {
      store.set(k, v);
    },
    removeItem: (k) => {
      store.delete(k);
    }
  };
}

describe("resilient durable SSE stream & session recovery", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("merges v4 telemetry by request, replay key, and committed plan revision", () => {
    const telemetry = createResearchV4State();
    telemetry.hydrate("req-42", {
      state_version: 7,
      v4: {
        schema_version: 1,
        budget: { pressure: "high", remaining: { tasks: 2 } },
        replan: { dag_revision: 2, pruned_task_ids: [3] }
      }
    });

    const event = {
      type: "research_replanned", schema_version: 1, request_id: "req-42",
      event_key: "decision-d3", state_version: 8, dag_revision: 3, pruned_task_ids: [3, 5]
    };
    expect(telemetry.accept("req-42", { ...event, request_id: "foreign" })).toBe(false);
    expect(telemetry.accept("req-42", { ...event, state_version: 7 })).toBe(false);
    expect(telemetry.accept("req-42", event)).toBe(true);
    expect(telemetry.accept("req-42", event)).toBe(false);
    expect(telemetry.accept("req-42", { type: "todo_list", dag_revision: 2 })).toBe(false);
    const revisedTasks = { ...event, type: "todo_list", event_key: "todo-d3" };
    expect(telemetry.accept("req-42", revisedTasks)).toBe(true);
    expect(telemetry.accept("req-42", revisedTasks)).toBe(false);
    expect([...telemetry.prunedTaskIds.value]).toEqual([3, 5]);

    const budget = { type: "budget_updated", schema_version: 1, request_id: "req-42" };
    expect(telemetry.accept("req-42", {
      ...budget, event_key: "old-budget", state_version: 6, budget: { pressure: "normal" }
    })).toBe(false);
    expect(telemetry.budget.value?.pressure).toBe("high");
    expect(telemetry.accept("req-42", {
      ...budget, event_key: "new-budget", state_version: 9, budget: { pressure: "exhausted" }
    })).toBe(true);
    expect(telemetry.budget.value?.pressure).toBe("exhausted");

    telemetry.hydrate("req-43", {});
    expect(telemetry.budget.value).toBeUndefined();
    expect([...telemetry.prunedTaskIds.value]).toEqual([]);
    expect(telemetry.accept("req-43", { type: "todo_list", tasks: [] })).toBe(true);
  });

  it("reconnects after clean EOF from the last applied frame", async () => {
    const storage = makeMemoryStorage();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('id: 10\ndata: {"type":"status"}\n\nid: 11\ndata: {"type":"status"}'))
      .mockResolvedValueOnce(new Response('id: 10\ndata: {"type":"status"}\n\nid: 11\ndata: {"type":"done"}\n\n'));
    vi.stubGlobal("fetch", fetchMock);
    const received: string[] = [];

    await useResilientStream().followWithResume("req-42", "Topic", (event) => {
      received.push(event.type);
    }, { storage, sleepFn: async () => {} });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1][1].headers["Last-Event-ID"]).toBe("10");
    expect(received).toEqual(["status", "done"]);
    expect(loadActiveStreamSession(storage)).toBeNull();
  });

  it("applies frames and hydrates a replay gap before advancing the cursor", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      'data: {"type":"replay_gap"}\n\nid: 2\ndata: {"type":"status"}\n\ndata: {"type":"stream_closed","status":"completed"}\n\n'
    )));
    const order: string[] = [];
    await streamRequestEvents("req-42", async (event) => {
      await Promise.resolve();
      order.push(event.type);
    }, { onEventId: (id) => order.push(id) });
    expect(order).toEqual(["replay_gap", "status", "2", "stream_closed"]);
  });

  it("retains the starting cursor and stops immediately when backoff is aborted", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const storage = makeMemoryStorage();
    const streamFn = vi.fn().mockRejectedValue(new Error("offline"));
    const { followWithResume, reconnecting } = useResilientStream();
    const result = followWithResume("req-42", "Topic", () => {}, {
      signal: controller.signal,
      storage,
      lastEventId: "42-2",
      runner: "local_durable",
      streamFn
    });
    const rejected = expect(result).rejects.toMatchObject({ name: "AbortError" });
    await Promise.resolve();
    expect(reconnecting.value).toBe(true);
    controller.abort();
    await vi.advanceTimersByTimeAsync(0);
    expect(reconnecting.value).toBe(false);
    await rejected;
    expect(streamFn).toHaveBeenCalledTimes(1);
    expect(loadActiveStreamSession(storage)).toMatchObject({ lastEventId: "42-2", runner: "local_durable" });
  });

  it("persists active session state and computes capped exponential backoff", () => {
    expect(computeReconnectDelayMs(0, 800, 10000)).toBe(800);
    expect(computeReconnectDelayMs(1, 800, 10000)).toBe(1600);
    expect(computeReconnectDelayMs(10, 800, 10000)).toBe(10000);

    const storage = makeMemoryStorage();
    saveActiveStreamSession(
      { requestId: "req-42", lastEventId: "12", topic: "Durable AI" },
      storage
    );
    expect(storage.getItem(ACTIVE_STREAM_STORAGE_KEY)).toBeTruthy();

    const restored = loadActiveStreamSession(storage);
    expect(restored?.requestId).toBe("req-42");
    expect(restored?.lastEventId).toBe("12");
    expect(restored?.topic).toBe("Durable AI");

    clearActiveStreamSession(storage);
    expect(loadActiveStreamSession(storage)).toBeNull();
  });

  it("retries dropped SSE connection with Last-Event-ID and clears storage on done", async () => {
    const storage = makeMemoryStorage();
    const { reconnecting, reconnectAttempt, followWithResume } = useResilientStream();
    const receivedTypes: string[] = [];
    const seenLastIds: Array<string | undefined> = [];

    let callNum = 0;
    const fakeStreamFn = vi.fn(async (_reqId, onEvent, opts) => {
      callNum += 1;
      seenLastIds.push(opts?.lastEventId);
      if (callNum === 1) {
        opts?.onEventId?.("5");
        onEvent({ type: "status", message: "step 1" });
        throw new Error("network blip");
      }
      opts?.onEventId?.("6");
      onEvent({ type: "done" });
    });

    await followWithResume(
      "req-res-1",
      "Resilient Topic",
      (evt) => receivedTypes.push(evt.type),
      {
        lastEventId: "1",
        maxRetries: 2,
        storage,
        sleepFn: async () => {},
        streamFn: fakeStreamFn
      }
    );

    expect(seenLastIds).toEqual(["1", "5"]);
    expect(receivedTypes).toEqual(["status", "done"]);
    expect(reconnectAttempt.value).toBe(1);
    expect(reconnecting.value).toBe(false);
    expect(loadActiveStreamSession(storage)).toBeNull();
  });

  it("calls POST /research/async and POST /requests/{id}/signals/{signal_name}", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/research/async")) {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            request_id: "req-async-1",
            workflow_run_id: "local-abc123",
            runner_backend: "local_durable",
            status: "queued",
            tenant_id: "default",
            events_url: "/requests/req-async-1/events?follow=1"
          }),
          { status: 202 }
        );
      }
      if (url.endsWith("/requests/req-async-1/signals/cancel")) {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            request_id: "req-async-1",
            tenant_id: "default",
            runner_backend: "local_durable",
            signal_name: "cancel",
            accepted: true,
            status: "cancelling"
          }),
          { status: 200 }
        );
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const submitted = await submitAsyncResearch({ topic: "Async Topic" });
    expect(submitted.request_id).toBe("req-async-1");
    expect(submitted.events_url).toContain("follow=1");

    const signaled = await sendWorkflowSignal("req-async-1", "cancel", { reason: "user" });
    expect(signaled.accepted).toBe(true);
    expect(signaled.signal_name).toBe("cancel");
  });
});
