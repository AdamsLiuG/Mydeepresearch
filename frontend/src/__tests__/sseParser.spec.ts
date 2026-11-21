import { describe, expect, it } from "vitest";
import { createSseFrameParser, parseSseJson } from "../services/sseParser";

describe("createSseFrameParser", () => {
  it("parses events split across arbitrary chunks and ignores heartbeats", () => {
    const parser = createSseFrameParser();

    const first = parser.feed(": keepalive\n\nid: 1\ndata: {\"type\":\"stat");
    expect(first).toEqual([]);
    expect(parser.getLastEventId()).toBe("1");

    const second = parser.feed("us\",\"message\":\"ok\"}\n\n");
    expect(second).toHaveLength(1);
    expect(second[0].id).toBe("1");
    expect(parseSseJson(second[0])).toEqual({ type: "status", message: "ok" });
  });

  it("handles CRLF line endings and multi-line data concatenation", () => {
    const parser = createSseFrameParser();
    const frames = parser.feed(
      "id: 7\r\nevent: update\r\ndata: {\r\ndata: \"type\":\"done\"\r\ndata: }\r\n\r\n"
    );

    expect(frames).toHaveLength(1);
    expect(frames[0].id).toBe("7");
    expect(frames[0].event).toBe("update");
    expect(parseSseJson(frames[0])).toEqual({ type: "done" });
    expect(parser.getLastEventId()).toBe("7");
  });

  it("flushes a trailing frame on EOF even without double newline", () => {
    const parser = createSseFrameParser("5");
    expect(parser.feed("id: 6\ndata: {\"type\":\"final_report\"}")).toEqual([]);

    const flushed = parser.flush();
    expect(flushed).toHaveLength(1);
    expect(flushed[0].id).toBe("6");
    expect(parseSseJson(flushed[0])).toEqual({ type: "final_report" });
    expect(parser.getLastEventId()).toBe("6");
  });
});
