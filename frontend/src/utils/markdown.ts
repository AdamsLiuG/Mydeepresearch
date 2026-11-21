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
  const rawHtml = marked.parse(escapeHtml(text), {
    breaks: true,
    gfm: true
  }) as string;
  return DOMPurify.sanitize(rawHtml, {
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|#|\/)/i,
    FORBID_TAGS: ["script", "style", "iframe", "object", "embed"],
    FORBID_ATTR: ["onerror", "onload", "onclick", "onmouseover"]
  });
}
