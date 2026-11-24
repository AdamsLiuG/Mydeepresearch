import { ref } from "vue";
import { streamRequestEvents } from "../services/api";
import type { ResearchStreamEvent, StreamOptions } from "../types/research";

export const ACTIVE_STREAM_STORAGE_KEY = "deepresearch:active_stream";

export interface PersistedStreamSession {
  requestId: string;
  lastEventId: string;
  topic: string;
  savedAt: string;
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function defaultStorage(): StorageLike | null {
  try {
    if (typeof window !== "undefined" && window.sessionStorage) {
      return window.sessionStorage;
    }
  } catch {
    return null;
  }
  return null;
}

export function computeReconnectDelayMs(
  attempt: number,
  baseMs = 800,
  maxMs = 10000
): number {
  const safeAttempt = Math.max(0, Math.floor(attempt));
  const raw = baseMs * Math.pow(2, safeAttempt);
  return Math.min(maxMs, Math.max(baseMs, raw));
}

export function saveActiveStreamSession(
  session: Omit<PersistedStreamSession, "savedAt">,
  storage: StorageLike | null = defaultStorage()
): void {
  if (!storage || !session.requestId.trim()) {
    return;
  }
  try {
    const payload: PersistedStreamSession = {
      requestId: session.requestId.trim(),
      lastEventId: session.lastEventId || "",
      topic: session.topic || "",
      savedAt: new Date().toISOString()
    };
    storage.setItem(ACTIVE_STREAM_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore sessionStorage quota or private mode failures
  }
}

export function loadActiveStreamSession(
  storage: StorageLike | null = defaultStorage()
): PersistedStreamSession | null {
  if (!storage) {
    return null;
  }
  try {
    const raw = storage.getItem(ACTIVE_STREAM_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as Partial<PersistedStreamSession>;
    if (!parsed || typeof parsed.requestId !== "string" || !parsed.requestId.trim()) {
      return null;
    }
    return {
      requestId: parsed.requestId.trim(),
      lastEventId: typeof parsed.lastEventId === "string" ? parsed.lastEventId : "",
      topic: typeof parsed.topic === "string" ? parsed.topic : "",
      savedAt: typeof parsed.savedAt === "string" ? parsed.savedAt : ""
    };
  } catch {
    return null;
  }
}

export function clearActiveStreamSession(
  storage: StorageLike | null = defaultStorage()
): void {
  if (!storage) {
    return;
  }
  try {
    storage.removeItem(ACTIVE_STREAM_STORAGE_KEY);
  } catch {
    // Ignore storage removal failures
  }
}

export function useResilientStream() {
  const reconnecting = ref(false);
  const reconnectAttempt = ref(0);

  async function followWithResume(
    requestId: string,
    topic: string,
    onEvent: (event: ResearchStreamEvent) => void,
    options: StreamOptions & {
      maxRetries?: number;
      storage?: StorageLike | null;
      sleepFn?: (ms: number) => Promise<void>;
      streamFn?: typeof streamRequestEvents;
    } = {}
  ): Promise<void> {
    const maxRetries = Math.max(0, options.maxRetries ?? 3);
    const storage = options.storage ?? defaultStorage();
    const sleepFn =
      options.sleepFn ?? ((ms: number) => new Promise((resolve) => setTimeout(resolve, ms)));
    const streamFn = options.streamFn ?? streamRequestEvents;

    let cursor = options.lastEventId || "";
    let finished = false;
    reconnectAttempt.value = 0;

    while (!finished) {
      try {
        await streamFn(
          requestId,
          (event) => {
            if (
              event.type === "done" ||
              event.type === "error" ||
              event.type === "cancelled"
            ) {
              finished = true;
              clearActiveStreamSession(storage);
            }
            onEvent(event);
          },
          {
            ...options,
            lastEventId: cursor || undefined,
            onEventId: (eventId) => {
              cursor = eventId;
              saveActiveStreamSession({ requestId, lastEventId: eventId, topic }, storage);
              options.onEventId?.(eventId);
            }
          }
        );
        finished = true;
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          reconnecting.value = false;
          throw err;
        }
        if (reconnectAttempt.value >= maxRetries) {
          reconnecting.value = false;
          throw err;
        }
        reconnecting.value = true;
        const delay = computeReconnectDelayMs(reconnectAttempt.value);
        reconnectAttempt.value += 1;
        await sleepFn(delay);
      }
    }
    reconnecting.value = false;
  }

  return {
    reconnecting,
    reconnectAttempt,
    followWithResume
  };
}
