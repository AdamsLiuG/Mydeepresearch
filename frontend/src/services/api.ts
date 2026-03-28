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

export interface ResearchRequest {
  topic: string;
  search_api?: string;
}

export interface ResumeRequest {
  search_api?: string;
}

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}

export interface StreamOptions {
  signal?: AbortSignal;
}

export interface MetricsSnapshot {
  generated_at?: string;
  cache_hit_total?: number;
  cache_miss_total?: number;
  counters?: Record<string, number>;
  recent_requests?: Record<string, unknown>[];
  [key: string]: unknown;
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
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream"
      },
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
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawEvent.startsWith("data:")) {
        const dataPayload = rawEvent.slice(5).trim();
        if (dataPayload) {
          try {
            const event = JSON.parse(dataPayload) as ResearchStreamEvent;
            onEvent(event);

            if (event.type === "error" || event.type === "done") {
              return;
            }
          } catch (error) {
            console.error("解析流式事件失败：", error, dataPayload);
          }
        }
      }

      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      // 处理可能的尾巴事件
      if (buffer.trim()) {
        const rawEvent = buffer.trim();
        if (rawEvent.startsWith("data:")) {
          const dataPayload = rawEvent.slice(5).trim();
          if (dataPayload) {
            try {
              const event = JSON.parse(dataPayload) as ResearchStreamEvent;
              onEvent(event);
            } catch (error) {
              console.error("解析流式事件失败：", error, dataPayload);
            }
          }
        }
      }
      break;
    }
  }
}

export async function fetchMetricsSnapshot(
  options: StreamOptions = {}
): Promise<MetricsSnapshot> {
  const response = await fetch(`${baseURL}/metrics/json`, {
    method: "GET",
    headers: {
      Accept: "application/json"
    },
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
): Promise<Record<string, unknown>[]> {
  const search = typeof limit === "number" && Number.isFinite(limit) ? `?limit=${limit}` : "";
  const response = await fetch(`${baseURL}/requests${search}`, {
    method: "GET",
    headers: {
      Accept: "application/json"
    },
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `获取持久化请求失败，状态码：${response.status}`);
  }

  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  return Array.isArray(payload.items) ? (payload.items as Record<string, unknown>[]) : [];
}
