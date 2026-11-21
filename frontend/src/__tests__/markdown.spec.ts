import { describe, expect, it } from "vitest";
import { escapeHtml, renderMarkdownBlock } from "../utils/markdown";

describe("markdown rendering and XSS sanitization", () => {
  it("renders valid Markdown structures cleanly", () => {
    const html = renderMarkdownBlock("## 结论\n\n- 要点一\n- [来源](https://example.com)");
    expect(html).toContain("<h2>结论</h2>");
    expect(html).toContain("<li>要点一</li>");
    expect(html).toContain('href="https://example.com"');
  });

  it("neutralizes script injection and javascript URI payloads", () => {
    const payload =
      '<script>alert(1)</script>\n<img src=x onerror="alert(2)">\n[bad](javascript:alert(3))\n<iframe src="https://evil.test"></iframe>';
    const html = renderMarkdownBlock(payload);
    const container = document.createElement("div");
    container.innerHTML = html;

    expect(container.querySelector("script, iframe, img, object, embed")).toBeNull();
    expect(container.querySelector("[onerror], [onload], [onclick]")).toBeNull();
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<iframe");
    expect(html).not.toContain("javascript:");
  });

  it("escapes raw HTML tags before parsing", () => {
    expect(escapeHtml("a < b & c > d")).toBe("a &lt; b &amp; c &gt; d");
  });
});
