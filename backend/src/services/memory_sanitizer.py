"""Shared normalization and sanitization for everything written into or read out of memory.

Two concerns live here:

1. Text/list normalization primitives that were previously duplicated across
   ``note_memory``, ``strategy_memory`` and ``strategy_synthesizer``. The strip-only and
   whitespace-collapsing variants are kept separate on purpose because the synthesizer
   relies on strip-only semantics while both memory services collapse internal runs.
2. Injection sanitization. Memory blocks are rendered into a *new* request's prompt, so
   stale evidence markers and tool directives carried over from an older request must be
   neutralized mechanically instead of relying on a prompt-level disclaimer alone.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from typing import Any, Callable

from services.text_processing import CITATION_PATTERN as _SOURCE_ID_PATTERN

# ``[TOOL_CALL:note:{...}]`` directives are how agents invoke tools. If such a string
# survives into an injected memory block the downstream model may execute it.
_TOOL_CALL_START_PATTERN = re.compile(r"\[TOOL_CALL:", re.IGNORECASE)

# Evidence ids are scoped to the request that produced them (``T1-S1``, ``T1-M1``, ``T1-C1``).
# Injected verbatim they let a new report cite sources it never retrieved.

# Prompt wrapper boundary tags that untrusted memory text must never forge or close.
_BOUNDARY_TAG_PATTERN = re.compile(
    r"</?\s*[a-z][\w:-]*\b[^<>]*>|<\|[^<>\n]+\|>|\[/?INST\]",
    re.IGNORECASE,
)
_MEMORY_URL_PATTERN = re.compile(
    r"(?:\b(?:[a-z][a-z0-9+.-]*://|(?:file|data|javascript):|www\.)|(?<![:/])//)[^\s<>\"\]]+",
    re.IGNORECASE,
)

_SOURCE_ID_PLACEHOLDER = "[历史来源]"


def normalize_text(value: Any, *, fallback: str = "") -> str:
    """Strip surrounding whitespace, preserving internal spacing."""

    text = str(value or "").strip()
    return text or fallback


def collapse_whitespace(value: Any) -> str:
    """Strip and collapse all internal whitespace runs into single spaces."""

    return " ".join(str(value or "").strip().split())


def normalize_list(value: Any, *, max_items: int) -> list[str]:
    """Coerce to a de-duplicated, whitespace-collapsed, length-capped list of strings."""

    if isinstance(value, list):
        candidates = value
    elif value in (None, ""):
        candidates = []
    else:
        candidates = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = collapse_whitespace(item)
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


def parse_json_list(value: Any) -> list[Any]:
    """Parse a JSON array that may already be a list, returning [] on any failure."""

    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _tool_directive_spans(text: str, pattern: re.Pattern[str]):
    pos = 0
    while match := pattern.search(text, pos):
        cursor = match.start()
        depth = 0
        in_string = False
        escaped = False
        while cursor < len(text):
            ch = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    cursor += 1
                    break
            cursor += 1
        yield match.start(), cursor
        pos = cursor


def strip_tool_directives(value: Any) -> str:
    """Remove tool-call directives (including nested JSON arrays) so memory cannot trigger tools."""

    text = str(value or "")
    out = []
    pos = 0
    for start, end in _tool_directive_spans(text, _TOOL_CALL_START_PATTERN):
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def neutralize_source_ids(value: Any) -> str:
    """Replace request-scoped evidence markers with a non-citable placeholder."""

    return _SOURCE_ID_PATTERN.sub(_SOURCE_ID_PLACEHOLDER, str(value or ""))


def neutralize_boundary_tags(value: Any) -> str:
    """Neutralize XML-style prompt boundary tags so memory text cannot break out of its block."""

    return _BOUNDARY_TAG_PATTERN.sub(" ", str(value or ""))


def contains_tool_directive(value: Any) -> bool:
    """Report whether the text carries a tool-call directive."""

    return bool(_TOOL_CALL_START_PATTERN.search(str(value or "")))


def contains_source_id(value: Any) -> bool:
    """Report whether the text carries a request-scoped evidence marker."""

    return bool(_SOURCE_ID_PATTERN.search(str(value or "")))


def sanitize_injection_text(value: Any) -> str:
    """Make a memory excerpt safe to place in another request's prompt.

    Strips tool directives (with nested brackets), neutralizes boundary tags and stale
    evidence ids, and collapses whitespace so the excerpt cannot trigger tools or be
    mistaken for citable current-round evidence.
    """

    text = strip_tool_directives(value)
    text = neutralize_boundary_tags(text)
    text = neutralize_source_ids(text)
    return collapse_whitespace(text)


def sanitize_memory_text(value: Any) -> str:
    """Sanitize historical prompt text and redact direct navigation targets."""

    return _MEMORY_URL_PATTERN.sub("[历史链接]", sanitize_injection_text(value))


WEB_SANITIZER_VERSION = 1
_WEB_INVISIBLE = frozenset(
    "\u00ad\u034f\u061c\u180e\u200b\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069\ufeff"
)
_WEB_JOINERS = "\u200c\u200d"
_WEB_SOURCE_ID = re.compile(_SOURCE_ID_PATTERN.pattern, re.IGNORECASE)
_WEB_TOOL_START = re.compile(r"\[\s*tool[ _-]*call\s*:")
_WEB_ROLE_LINE = re.compile(
    r"^\s*(?:[#>*\x60_-]\s*)*(?:system|developer|assistant|user|tool)\s*(?:message\s*)?:.*$",
    re.MULTILINE,
)
_WEB_CONTROL = re.compile(
    r"\[/?(?:system|developer|assistant|user|tool|im_start|im_end|start_header_id|end_header_id)\]"
    r"|<<\s*/?sys\s*>>|<!--.*?(?:-->|\Z)",
    re.DOTALL,
)
_WEB_INCOMPLETE = re.compile(
    r"(?:<\|[^>]*|</?(?:system|developer|assistant|user|tool)\b[^>]*|"
    r"\[\s*(?:tool[ _-]*call|/?inst|/?system|/?assistant)[^\]]*)\Z"
)


def _web_shadow(text: str) -> tuple[str, list[int]]:
    shadow: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char in _WEB_JOINERS:
            continue
        normalized = unicodedata.normalize("NFKC", char).casefold()
        shadow.append(normalized)
        positions.extend([index] * len(normalized))
    return "".join(shadow), positions


def sanitize_web_text(
    value: str, *, decode_entities: bool = False, max_chars: int = 50_000,
    on_drop: Callable[[], None] | None = None,
) -> str:
    """Clean a logical visible block after joining its inline children."""
    if not isinstance(value, str) or not value or max_chars <= 0:
        return ""
    if decode_entities:
        value = html.unescape(value)
    text = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        char for char in text
        if char not in _WEB_INVISIBLE and (char in "\n\t" or unicodedata.category(char) != "Cc")
    )
    shadow, positions = _web_shadow(text)
    spans = list(_tool_directive_spans(shadow, _WEB_TOOL_START))
    for pattern in (_BOUNDARY_TAG_PATTERN, _WEB_SOURCE_ID, _WEB_ROLE_LINE, _WEB_CONTROL, _WEB_INCOMPLETE):
        spans.extend(match.span() for match in pattern.finditer(shadow) if match.start() != match.end())
    if spans:
        out = []
        cursor = 0
        for start, end in sorted(spans):
            left, right = positions[start], positions[end - 1] + 1
            if left > cursor:
                out.append(text[cursor:left])
            out.append(" ")
            cursor = max(cursor, right)
        out.append(text[cursor:])
        text = "".join(out)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
        tail = _WEB_INCOMPLETE.search(_web_shadow(cleaned)[0])
        if tail:
            cleaned = cleaned[:_web_shadow(cleaned)[1][tail.start()]].rstrip()
    if on_drop and (spans or (value.strip() and not cleaned)):
        on_drop()
    return cleaned
