"""Utility helpers for normalizing agent generated text."""

from __future__ import annotations

import json
import re
from typing import Any

CITATION_PATTERN = re.compile(r"\[(T\d+-S\d+)\]")
JSON_TEXT_KEYS = (
    "report_markdown",
    "report",
    "summary",
    "content",
    "text",
    "message",
    "output",
)
JSON_FIELD_PATTERNS = tuple(
    re.compile(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
    for key in JSON_TEXT_KEYS
)
META_REASONING_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bI need to\b",
        r"\bI should\b",
        r"\bI will\b",
        r"\bfirst[, ]",
        r"\bnext[, ]",
        r"\bthen[, ]",
        r"好的[，,]我",
        r"首先",
        r"接下来",
        r"然后",
        r"现在开始",
        r"我需要",
        r"我先",
        r"我会",
        r"根据用户",
        r"用户要求",
        r"任务要求",
        r"提示词",
        r"source[_ -]?id",
        r"工具调用",
        r"最终输出",
        r"最终总结",
        r"markdown",
        r"格式正确",
        r"开始撰写",
        r"每条发现",
        r"检查.*(?:引用|来源)",
        r"确保.*(?:引用|source[_ -]?id|格式)",
        r"调用\s*`?(?:note|evidence_lookup|fetch_page|search_web)`?",
        r"\[TOOL_CALL:",
    )
)


def strip_tool_calls(text: str) -> str:
    """移除文本中的工具调用标记。"""
    if not text:
        return text

    pattern = re.compile(r"\[TOOL_CALL:[^\]]+\]")
    return pattern.sub("", text)


def normalize_agent_markdown(text: str) -> str:
    """Best-effort cleanup for model outputs that leak JSON/string wrappers."""
    if not text:
        return text

    cleaned = strip_tool_calls(text).strip()
    for _ in range(4):
        updated = _unwrap_json_like_text(cleaned)
        updated = strip_tool_calls(updated).strip()
        if updated == cleaned:
            break
        cleaned = updated

    cleaned = _unescape_common_sequences(cleaned)
    cleaned = strip_tool_calls(cleaned).strip().strip("`").strip()
    cleaned = re.sub(r'^(?:(?:"|\')?\s*[:：]\s*(?:"|\')?\s*)+', "", cleaned).strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    return cleaned


def strip_citation_markers(text: str) -> str:
    """Remove `[Tn-Sm]` style citation markers from a text fragment."""
    if not text:
        return text
    cleaned = CITATION_PATTERN.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def looks_like_meta_reasoning(text: str) -> bool:
    """Return whether the text is likely planning chatter or tool narration."""
    normalized = normalize_agent_markdown(text or "")
    if not normalized:
        return False

    compact = normalized.replace("`", "").strip()
    if compact.startswith(("任务总结", "关键发现", "证据不足", "风险", "背景概览")):
        return False

    for pattern in META_REASONING_PATTERNS:
        if pattern.search(compact):
            return True
    return False


def _unwrap_json_like_text(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return candidate

    parsed_text = _parse_json_candidate(candidate)
    if parsed_text is not None:
        return parsed_text

    embedded_text = _extract_embedded_json_text(candidate)
    if embedded_text is not None:
        return embedded_text

    return candidate


def _parse_json_candidate(candidate: str) -> str | None:
    if candidate[0] not in {'"', "{", "["}:
        return None

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return _extract_text_from_payload(payload)


def _extract_text_from_payload(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, dict):
        for key in JSON_TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for value in payload.values():
            extracted = _extract_text_from_payload(value)
            if extracted:
                return extracted

    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_text_from_payload(item)
            if extracted:
                return extracted

    return None


def _extract_embedded_json_text(candidate: str) -> str | None:
    for pattern in JSON_FIELD_PATTERNS:
        match = pattern.search(candidate)
        if not match:
            continue

        try:
            return json.loads(f'"{match.group(1)}"').strip()
        except json.JSONDecodeError:
            continue

    return None


def _unescape_common_sequences(text: str) -> str:
    if not any(token in text for token in ("\\n", "\\r", "\\t", '\\"')):
        return text

    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
    )
