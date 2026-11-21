export interface SseFrame {
  id?: string;
  event?: string;
  data: string;
}

export interface SseFrameParser {
  feed: (chunk: string) => SseFrame[];
  flush: () => SseFrame[];
  getLastEventId: () => string | undefined;
}

/**
 * Stateful SSE frame parser supporting arbitrary UTF-8 chunk boundaries,
 * CRLF normalization, multi-line `data:` concatenation, and `id:` tracking.
 */
export function createSseFrameParser(initialLastEventId?: string): SseFrameParser {
  let buffer = "";
  let currentId: string | undefined = initialLastEventId;
  let lastEventId: string | undefined = initialLastEventId;
  let currentEvent: string | undefined;
  let dataLines: string[] = [];

  const finalizeFrame = (): SseFrame | null => {
    if (dataLines.length === 0 && currentEvent === undefined && currentId === undefined) {
      return null;
    }
    const data = dataLines.join("\n");
    const frame: SseFrame = {
      ...(currentId !== undefined ? { id: currentId } : {}),
      ...(currentEvent !== undefined ? { event: currentEvent } : {}),
      data
    };
    if (currentId !== undefined) {
      lastEventId = currentId;
    }
    currentEvent = undefined;
    dataLines = [];
    currentId = undefined;
    if (!data && !frame.event) {
      return null;
    }
    return frame;
  };

  const processLine = (rawLine: string, out: SseFrame[]): void => {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") {
      const frame = finalizeFrame();
      if (frame) {
        out.push(frame);
      }
      return;
    }
    if (line.startsWith(":")) {
      return;
    }
    const colonIdx = line.indexOf(":");
    const field = colonIdx === -1 ? line : line.slice(0, colonIdx);
    let value = colonIdx === -1 ? "" : line.slice(colonIdx + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }

    if (field === "data") {
      dataLines.push(value);
    } else if (field === "event") {
      currentEvent = value;
    } else if (field === "id") {
      if (!value.includes("\0")) {
        currentId = value;
        lastEventId = value;
      }
    }
  };

  return {
    feed(chunk: string): SseFrame[] {
      const frames: SseFrame[] = [];
      buffer += chunk;
      let newlineIdx = buffer.indexOf("\n");
      while (newlineIdx !== -1) {
        const line = buffer.slice(0, newlineIdx);
        buffer = buffer.slice(newlineIdx + 1);
        processLine(line, frames);
        newlineIdx = buffer.indexOf("\n");
      }
      return frames;
    },
    flush(): SseFrame[] {
      const frames: SseFrame[] = [];
      if (buffer.length > 0) {
        processLine(buffer, frames);
        buffer = "";
      }
      const trailing = finalizeFrame();
      if (trailing) {
        frames.push(trailing);
      }
      return frames;
    },
    getLastEventId(): string | undefined {
      return lastEventId;
    }
  };
}

export function parseSseJson<T = Record<string, unknown>>(frame: SseFrame): T | null {
  const payload = frame.data.trim();
  if (!payload || payload === "[DONE]") {
    return null;
  }
  try {
    return JSON.parse(payload) as T;
  } catch {
    return null;
  }
}
