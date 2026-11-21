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
  cache_approximate_hit_total?: number;
  cache_approximate_dense_hit_total?: number;
  cache_approximate_sparse_hit_total?: number;
  cache_approximate_hybrid_hit_total?: number;
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

  const parseSseFrame = (rawFrame: string): ResearchStreamEvent | null => {
    const dataLines: string[] = [];
    for (const line of rawFrame.split("\n")) {
      // Ignore empty lines and SSE heartbeat/comment lines (e.g. ": keepalive").
      if (!line || line.startsWith(":")) {
        continue;
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    const dataPayload = dataLines.join("\n").trim();
    if (!dataPayload) {
      return null;
    }
    try {
      return JSON.parse(dataPayload) as ResearchStreamEvent;
    } catch (error) {
      console.error("解析流式事件失败：", error, dataPayload);
      return null;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    const chunk = decoder
      .decode(value || new Uint8Array(), { stream: !done })
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n");
    buffer += chunk;

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawFrame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const event = parseSseFrame(rawFrame);
      if (event) {
        onEvent(event);
        if (event.type === "error" || event.type === "done") {
          return;
        }
      }

      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      // Flush trailing frame when stream closes without double newline.
      const event = parseSseFrame(buffer);
      if (event) {
        onEvent(event);
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

export async function fetchPersistedRequest(
  requestId: string,
  options: StreamOptions = {}
): Promise<Record<string, unknown>> {
  const response = await fetch(`${baseURL}/requests/${encodeURIComponent(requestId)}`, {
    method: "GET",
    headers: {
      Accept: "application/json"
    },
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `获取历史请求详情失败，状态码：${response.status}`);
  }

  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  return payload && typeof payload === "object" ? payload : {};
}
