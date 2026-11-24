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
import { sendWorkflowSignal, submitAsyncResearch } from "../services/api";

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
