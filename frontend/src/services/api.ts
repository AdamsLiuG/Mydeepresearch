import { createSseFrameParser, parseSseJson } from "./sseParser";
import type {
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
  StreamOptions
} from "../types/research";

export type {
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
  StreamOptions
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
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  return streamFromEndpoint("/research/stream", payload, onEvent, options);
}

export async function resumeResearchStream(
  requestId: string,
  payload: ResumeRequest,
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  return streamFromEndpoint(
    `/requests/${encodeURIComponent(requestId)}/resume/stream`,
    payload,
    onEvent,
    options
  );
}

async function streamFromEndpoint(
  endpoint: string,
  payload: unknown,
  onEvent: (event: ResearchStreamEvent) => void,
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

  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = createSseFrameParser(options.lastEventId);

  const dispatchFrames = (frames: ReturnType<typeof parser.feed>): boolean => {
    for (const frame of frames) {
      if (frame.id) {
        options.onEventId?.(frame.id);
      }
      const event = parseSseJson<ResearchStreamEvent>(frame);
      if (event) {
        onEvent(event);
        if (event.type === "error" || event.type === "done") {
          return true;
        }
      }
    }
    return false;
  };

  while (true) {
    const { value, done } = await reader.read();
    const chunk = decoder.decode(value || new Uint8Array(), { stream: !done });
    if (dispatchFrames(parser.feed(chunk))) {
      return;
    }
    if (done) {
      dispatchFrames(parser.flush());
      break;
    }
  }
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
