import { createSseFrameParser, parseSseJson } from "./sseParser";
import type {
  AsyncResearchSubmitResponse,
  ClarifyResponse,
  EditablePlanResponse,
  MetricsSnapshot,
  PersistedRequestSummary,
  ReplayEventItem,
  ResearchEventHandler,
  ResearchRequest,
  ResearchStreamEvent,
  ResearchTaskPlanItem,
  ResumeRequest,
  SectionDrilldownResponse,
  SkillsCatalogResponse,
  StreamOptions,
  WorkflowSignalResponse
} from "../types/research";

export type {
  AsyncResearchSubmitResponse,
  ClarifyResponse,
  EditablePlanResponse,
  MetricsSnapshot,
  PersistedRequestSummary,
  ReplayEventItem,
  ResearchRequest,
  ResearchStreamEvent,
  ResearchTaskPlanItem,
  ResumeRequest,
  SectionDrilldownResponse,
  SkillsCatalogResponse,
  StreamOptions,
  WorkflowSignalResponse
} from "../types/research";

function resolveBaseURL(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) {
    return configured.replace(/\/+$/, "");
  }

  if (typeof window !== "undefined" && window.location?.hostname) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  return "http://localhost:8000";
}

const baseURL = resolveBaseURL();

function buildHeaders(
  base: Record<string, string>,
  options: StreamOptions = {}
): Record<string, string> {
  const headers: Record<string, string> = { ...base };
  const apiKey = options.apiKey?.trim();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const lastEventId = options.lastEventId?.trim();
  if (lastEventId) {
    headers["Last-Event-ID"] = lastEventId;
  }
  return headers;
}

export async function runResearchStream(
  payload: ResearchRequest,
  onEvent: ResearchEventHandler,
  options: StreamOptions = {}
): Promise<void> {
  return streamFromEndpoint("/research/stream", payload, onEvent, options);
}

export async function resumeResearchStream(
  requestId: string,
  payload: ResumeRequest,
  onEvent: ResearchEventHandler,
  options: StreamOptions = {}
): Promise<void> {
  return streamFromEndpoint(
    `/requests/${encodeURIComponent(requestId)}/resume/stream`,
    payload,
    onEvent,
    options
  );
}

export function isStreamEnd(event: ResearchStreamEvent): boolean {
  return ["done", "error", "cancelled", "stream_closed"].includes(event.type);
}

export class StreamHttpError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function consumeSseBody(
  body: ReadableStream<Uint8Array>,
  onEvent: ResearchEventHandler,
  options: StreamOptions = {}
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = createSseFrameParser();
  let cursor = options.lastEventId;
  let reachedEof = false;

  try {
    while (true) {
      options.signal?.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) {
        reachedEof = true;
        return;
      }
      for (const frame of parser.feed(decoder.decode(value, { stream: true }))) {
        options.signal?.throwIfAborted();
        if (frame.data.trim() === "[DONE]") continue;
        const event = parseSseJson<ResearchStreamEvent>(frame);
        if (!event || typeof event.type !== "string") {
          throw new Error("研究事件流包含无效的数据帧");
        }
        if (frame.id && cursor && isEventIdAtOrBefore(frame.id, cursor)) {
          continue;
        }
        await onEvent(event);
        if (event.type === "replay_gap") {
          cursor = undefined;
        }
        if (frame.id !== undefined) {
          cursor = frame.id;
          options.onEventId?.(frame.id);
        }
        if (isStreamEnd(event)) {
          return;
        }
      }
    }
  } finally {
    if (!reachedEof) {
      void reader.cancel().catch(() => {});
    }
    reader.releaseLock();
  }
}

function notifyStreamStart(response: Response, options: StreamOptions): void {
  options.onStreamStart?.({
    requestId: response.headers.get("X-Request-ID") || undefined,
    runner: response.headers.get("X-Workflow-Runner") || undefined
  });
}

function isEventIdAtOrBefore(eventId: string, cursor: string): boolean {
  if (eventId === cursor) {
    return true;
  }
  if (!/^\d+(?:-\d+)?$/.test(eventId) || !/^\d+(?:-\d+)?$/.test(cursor) ||
      eventId.includes("-") !== cursor.includes("-")) {
    return false;
  }
  const [major, minor = 0n] = eventId.split("-").map(BigInt);
  const [cursorMajor, cursorMinor = 0n] = cursor.split("-").map(BigInt);
  return major < cursorMajor || (major === cursorMajor && minor <= cursorMinor);
}

async function streamFromEndpoint(
  endpoint: string,
  payload: unknown,
  onEvent: ResearchEventHandler,
  options: StreamOptions = {}
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${baseURL}${endpoint}`, {
      method: "POST",
      headers: buildHeaders(
        {
          "Content-Type": "application/json",
          Accept: "text/event-stream"
        },
        options
      ),
      body: JSON.stringify(payload),
      signal: options.signal
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new Error(
      `无法连接研究后端（${baseURL}）。请检查前端访问地址、VITE_API_BASE_URL 和后端 CORS 配置，尤其确认 localhost 与 127.0.0.1 是否一致。`
    );
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      errorText || `研究请求失败，状态码：${response.status}`
    );
  }

  const body = response.body;
  if (!body) {
    throw new Error("浏览器不支持流式响应，无法获取研究进度");
  }

  notifyStreamStart(response, options);
  await consumeSseBody(body, onEvent, options);
}

export async function submitAsyncResearch(
  payload: ResearchRequest,
  options: StreamOptions = {}
): Promise<AsyncResearchSubmitResponse> {
  const response = await fetch(`${baseURL}/research/async`, {
    method: "POST",
    headers: buildHeaders(
      {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      options
    ),
    body: JSON.stringify(payload),
    signal: options.signal
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `提交异步研究失败，状态码：${response.status}`);
  }
  return (await response.json()) as AsyncResearchSubmitResponse;
}

export async function sendWorkflowSignal(
  requestId: string,
  signalName: string,
  payload: Record<string, unknown> = {},
  options: StreamOptions = {}
): Promise<WorkflowSignalResponse> {
  const response = await fetch(
    `${baseURL}/requests/${encodeURIComponent(requestId)}/signals/${encodeURIComponent(signalName)}`,
    {
      method: "POST",
      headers: buildHeaders(
        {
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        options
      ),
      body: JSON.stringify({ payload }),
      signal: options.signal
    }
  );
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `发送工作流信号失败，状态码：${response.status}`);
  }
  return (await response.json()) as WorkflowSignalResponse;
}

export async function streamRequestEvents(
  requestId: string,
  onEvent: ResearchEventHandler,
  options: StreamOptions = {}
): Promise<void> {
  const query = options.lastEventId
    ? `?follow=1&last_event_id=${encodeURIComponent(options.lastEventId)}`
    : "?follow=1";
  const response = await fetch(
    `${baseURL}/requests/${encodeURIComponent(requestId)}/events${query}`,
    {
      method: "GET",
      headers: buildHeaders({ Accept: "text/event-stream" }, options),
      signal: options.signal
    }
  );
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new StreamHttpError(errorText || `续传事件流失败，状态码：${response.status}`, response.status);
  }
  if (!response.body) {
    throw new Error("浏览器不支持事件流读取");
  }
  notifyStreamStart(response, options);
  await consumeSseBody(response.body, onEvent, options);
}

export async function fetchMetricsSnapshot(
  options: StreamOptions = {}
): Promise<MetricsSnapshot> {
  const response = await fetch(`${baseURL}/metrics/json`, {
    method: "GET",
    headers: buildHeaders({ Accept: "application/json" }, options),
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      errorText || `获取全局指标失败，状态码：${response.status}`
    );
  }

  const payload = (await response.json().catch(() => ({}))) as MetricsSnapshot;
  return payload && typeof payload === "object" ? payload : {};
}

export async function fetchPersistedRequests(
  limit?: number,
  options: StreamOptions = {}
): Promise<PersistedRequestSummary[]> {
  const search = typeof limit === "number" && Number.isFinite(limit) ? `?limit=${limit}` : "";
  const response = await fetch(`${baseURL}/requests${search}`, {
    method: "GET",
    headers: buildHeaders({ Accept: "application/json" }, options),
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `获取持久化请求失败，状态码：${response.status}`);
  }

  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  return Array.isArray(payload.items) ? (payload.items as PersistedRequestSummary[]) : [];
}

export async function fetchPersistedRequest(
  requestId: string,
  options: StreamOptions = {}
): Promise<Record<string, unknown>> {
  const response = await fetch(`${baseURL}/requests/${encodeURIComponent(requestId)}`, {
    method: "GET",
    headers: buildHeaders({ Accept: "application/json" }, options),
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `获取历史请求详情失败，状态码：${response.status}`);
  }

  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  return payload && typeof payload === "object" ? payload : {};
}

export async function clarifyResearch(
  topic: string,
  searchApi?: string,
  options: StreamOptions = {}
): Promise<ClarifyResponse> {
  const response = await fetch(`${baseURL}/research/clarify`, {
    method: "POST",
    headers: buildHeaders(
      {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      options
    ),
    body: JSON.stringify({
      topic,
      ...(searchApi ? { search_api: searchApi } : {})
    }),
    signal: options.signal
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `主题澄清请求失败，状态码：${response.status}`);
  }
  return (await response.json()) as ClarifyResponse;
}

export async function createResearchPlan(
  payload: ResearchRequest,
  options: StreamOptions = {}
): Promise<EditablePlanResponse> {
  const response = await fetch(`${baseURL}/research/plan`, {
    method: "POST",
    headers: buildHeaders(
      {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      options
    ),
    body: JSON.stringify(payload),
    signal: options.signal
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `生成研究计划失败，状态码：${response.status}`);
  }
  return (await response.json()) as EditablePlanResponse;
}

export async function updateResearchPlan(
  requestId: string,
  tasks: ResearchTaskPlanItem[],
  options: StreamOptions = {}
): Promise<EditablePlanResponse> {
  const response = await fetch(
    `${baseURL}/requests/${encodeURIComponent(requestId)}/plan`,
    {
      method: "PUT",
      headers: buildHeaders(
        {
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        options
      ),
      body: JSON.stringify({ tasks }),
      signal: options.signal
    }
  );
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `更新研究计划失败，状态码：${response.status}`);
  }
  return (await response.json()) as EditablePlanResponse;
}

export async function drilldownReportSection(
  requestId: string,
  sectionTitle: string,
  question: string,
  searchApi?: string,
  options: StreamOptions = {}
): Promise<SectionDrilldownResponse> {
  const response = await fetch(
    `${baseURL}/requests/${encodeURIComponent(requestId)}/drilldown`,
    {
      method: "POST",
      headers: buildHeaders(
        {
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        options
      ),
      body: JSON.stringify({
        section_title: sectionTitle,
        question,
        ...(searchApi ? { search_api: searchApi } : {})
      }),
      signal: options.signal
    }
  );
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `章节追问失败，状态码：${response.status}`);
  }
  return (await response.json()) as SectionDrilldownResponse;
}

export async function replayRequestEvents(
  requestId: string,
  lastEventId?: string,
  options: StreamOptions = {}
): Promise<ReplayEventItem[]> {
  const query = lastEventId
    ? `?last_event_id=${encodeURIComponent(lastEventId)}`
    : "";
  const response = await fetch(
    `${baseURL}/requests/${encodeURIComponent(requestId)}/events${query}`,
    {
      method: "GET",
      headers: buildHeaders({ Accept: "application/json" }, { ...options, lastEventId }),
      signal: options.signal
    }
  );
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `事件回放失败，状态码：${response.status}`);
  }
  const payload = (await response.json().catch(() => ({}))) as {
    events?: ReplayEventItem[];
  };
  return Array.isArray(payload.events) ? payload.events : [];
}

export async function fetchSkillsCatalog(
  options: StreamOptions = {}
): Promise<SkillsCatalogResponse> {
  const response = await fetch(`${baseURL}/skills`, {
    method: "GET",
    headers: buildHeaders({ Accept: "application/json" }, options),
    signal: options.signal
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `获取 Skill 目录失败，状态码：${response.status}`);
  }
  return (await response.json()) as SkillsCatalogResponse;
}
