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

import json
import re
from typing import Any

# ``[TOOL_CALL:note:{...}]`` directives are how agents invoke tools. If such a string
# survives into an injected memory block the downstream model may execute it.
_TOOL_CALL_START_PATTERN = re.compile(r"\[TOOL_CALL:", re.IGNORECASE)

# Evidence ids are scoped to the request that produced them (``T1-S1`` = task 1 source 1).
# Injected verbatim they let a new report cite sources it never retrieved.
_SOURCE_ID_PATTERN = re.compile(r"\[T\d+-S\d+\]")

# Prompt wrapper boundary tags that untrusted memory text must never forge or close.
_BOUNDARY_TAG_PATTERN = re.compile(
    r"</?\s*(?:RESEARCH_MEMORY|RESEARCH_SKILL|SYSTEM|ASSISTANT|USER)\b[^>]*>",
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


def strip_tool_directives(value: Any) -> str:
    """Remove tool-call directives (including nested JSON arrays) so memory cannot trigger tools."""

    text = str(value or "")
    if not text:
        return ""

    out: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        match = _TOOL_CALL_START_PATTERN.search(text, pos)
        if not match:
            out.append(text[pos:])
            break
        out.append(text[pos : match.start()])
        cursor = match.start()
        depth = 0
        in_string = False
        escaped = False
        while cursor < length:
            ch = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        cursor += 1
                        break
            cursor += 1
        pos = cursor
    return "".join(out)


def neutralize_source_ids(value: Any) -> str:
    """Replace request-scoped evidence markers with a non-citable placeholder."""

    return _SOURCE_ID_PATTERN.sub(_SOURCE_ID_PLACEHOLDER, str(value or ""))


def neutralize_boundary_tags(value: Any) -> str:
    """Neutralize XML-style prompt boundary tags so memory text cannot break out of its block."""

    return _BOUNDARY_TAG_PATTERN.sub("", str(value or ""))


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
