import DOMPurify from "dompurify";
import { marked } from "marked";

export function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

export function renderMarkdownBlock(value: string): string {
  const text = typeof value === "string" ? value.trim() : "";
  if (!text) {
    return "<p>暂无可用信息</p>";
  }
  const renderer = new marked.Renderer();
  renderer.image = ({ text: alt }) => escapeHtml(alt);
  const rawHtml = marked.parse(escapeHtml(text), {
    breaks: true,
    gfm: true,
    renderer
  }) as string;
  return DOMPurify.sanitize(rawHtml, {
    USE_PROFILES: { html: true },
    ALLOWED_URI_REGEXP: /^(?:https?:\/\/|mailto:|#|\/(?![/\\]))/i,
    FORBID_TAGS: ["script", "style", "iframe", "object", "embed", "img"],
    FORBID_ATTR: ["style", "onerror", "onload", "onclick", "onmouseover"],
    ALLOW_DATA_ATTR: false
  });
}
