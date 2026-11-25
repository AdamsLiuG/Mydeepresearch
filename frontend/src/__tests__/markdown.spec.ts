import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
const runtime = globalThis as typeof globalThis & { process?: { features?: { require_module?: boolean } } };
const jsdom = runtime.process?.features?.require_module ? await import("jsdom") : null;

let markdown: typeof import("../utils/markdown");
const window = jsdom ? new jsdom.JSDOM().window : undefined;

beforeAll(async () => {
  vi.stubGlobal("window", window);
  markdown = await import("../utils/markdown");
});

afterAll(() => {
  window?.close();
  vi.unstubAllGlobals();
});

describe("markdown rendering and XSS sanitization", () => {
  it.skipIf(!window)("renders valid Markdown structures cleanly", () => {
    const html = markdown.renderMarkdownBlock("## 结论\n\n- 要点一\n- [来源](https://example.com)");
    expect(html).toContain("<h2>结论</h2>");
    expect(html).toContain("<li>要点一</li>");
    expect(html).toContain('href="https://example.com"');
  });

  it.skipIf(!window)("neutralizes script injection and javascript URI payloads", () => {
    const payload =
      '<script>alert(1)</script>\n<img src=x onerror="alert(2)">\n[bad](javascript:alert(3))\n<iframe src="https://evil.test"></iframe>';
    const html = markdown.renderMarkdownBlock(payload);
    const container = window!.document.createElement("div");
    container.innerHTML = html;

    expect(container.querySelector("script, iframe, img, object, embed")).toBeNull();
    expect(container.querySelector("[onerror], [onload], [onclick]")).toBeNull();
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<iframe");
    expect(html).not.toContain("javascript:");
  });

  it.skipIf(!window)("keeps report links and strips SVG images and cross-origin relative resources", () => {
    const html = markdown.renderMarkdownBlock(
      "[anchor](#section) [reference](/reference) [outside](//example.test) ![chart](https://example.test/chart.svg)"
    );
    const container = window!.document.createElement("div");
    container.innerHTML = html;
    expect(Array.from(container.querySelectorAll("[href]"), (el) => el.getAttribute("href")))
      .toEqual(["#section", "/reference"]);
    expect(container.querySelector("img, svg")).toBeNull();
    expect(container.textContent).toContain("chart");
  });

  it("escapes raw HTML tags before parsing", () => {
    expect(markdown.escapeHtml("a < b & c > d")).toBe("a &lt; b &amp; c &gt; d");
  });
});
