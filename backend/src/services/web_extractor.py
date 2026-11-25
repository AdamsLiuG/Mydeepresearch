"""Request-scoped web extraction from bounded, authorized documents."""

from __future__ import annotations

import codecs
import re
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace
from html import unescape
from html.parser import HTMLParser
from threading import Lock
from typing import Literal, Protocol

from metrics import metrics_registry
from services.memory_sanitizer import WEB_SANITIZER_VERSION, sanitize_web_text
from services.multimodal_parser import MultimodalEvidenceBlock
from services.web_fetch_gateway import (
    FetchScope,
    GatewayError,
    WebFetchGateway,
    normalize_url,
)

QUALITY_VERSION = 1
POLICY_VERSION = 1
MAX_TEXT = 50_000
MAX_TABLES = 32
MAX_TABLE_SLOTS = 16_384
MAX_DOCUMENT = 2 * 1024 * 1024
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
TEXT_TYPES = frozenset({"text/plain", "text/markdown", "application/json"})
STATIC_TYPES = HTML_TYPES | TEXT_TYPES | {"application/pdf"}
_VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_EXCLUDED = frozenset(
    "script style template noscript form input button select textarea iframe frame frameset object embed "
    "svg canvas audio video nav footer aside dialog menu head".split()
)
_REGIONS = frozenset({"main", "article", "section", "body"})
_HEADINGS = frozenset(f"h{i}" for i in range(1, 7))
_BLOCK_TAGS = _HEADINGS | frozenset(
    "address article blockquote body dd details div dl dt figcaption figure header hr li main ol p pre "
    "section summary ul".split()
)
_CLOSE_P = _BLOCK_TAGS | {"table"}
_META_CHARSET = re.compile(br"""<meta\b[^>]{0,512}\bcharset\s*=\s*["']?\s*([a-zA-Z0-9_.:-]+)""", re.I)
_CHARSET = re.compile(r"""(?:^|;)\s*charset\s*=\s*["']?([\w.:-]+)""", re.I)
_LOGIN = re.compile(
    r"(?:sign[ -]?in|log[ -]?in|authenticate|subscribe)\s+(?:to|before|for)\b|"
    r"(?:access denied|subscription required|authentication required)|(?:登录后|请登录|访问受限|订阅后)",
    re.I,
)
_JS = re.compile(r"(?:enable|requires?)\s+(?:the use of\s+)?javascript|javascript\s+is\s+(?:required|disabled)|启用\s*javascript", re.I)
_UNIT = re.compile(
    r"(?:^|\n)\s*(?:units?\s*[:：]|单位\s*[:：]|金额单位\s*[:：])\s*([^\n]{1,128})",
    re.I,
)
_MD_SPECIAL = re.compile(r"\\[\\\x60\[\]<>|#*_!+~.&-]|([\\\x60\[\]<>|])")
_ENTITY = re.compile(r"\\&|&(?=#(?:x[0-9a-f]+|[0-9]+)|[a-z]+;)", re.I)
_UNESCAPED_ENTITIES = re.compile(r"\\&|&(?:#[xX][0-9a-fA-F]+;?|#[0-9]+;?|[a-zA-Z][a-zA-Z0-9]*;?)")
_MARKUP = re.compile(r"(?<!\\)<(?:/?[a-z!]|[?])", re.I)


def _markdown_text(text: str) -> str:
    return _MD_SPECIAL.sub(lambda match: "\\" + match[1] if match[1] else match[0], text)


def _markdown_paragraph(text: str) -> str:
    text = _markdown_text(text)
    text = _ENTITY.sub(lambda match: match[0] if match[0].startswith("\\") else "\\&", text)
    text = re.sub(r"(?m)^([#*_!+~-])", r"\\\1", text)
    return re.sub(r"(?m)^(\d+)([.)])(?=\s)", r"\1\\\2", text)


@dataclass(frozen=True)
class WebExtractionResult:
    tier: Literal["L0", "L1", "L2"]
    final_url: str
    title: str
    visible_text: str
    multimodal_blocks: tuple[dict, ...]
    quality_flags: tuple[str, ...]
    sanitizer_version: int
    dropped_nodes: int
    adapter: Literal["snippet", "static", "reader_http", "playwright"]
    degraded_reason: str | None = None
    text_anchors: tuple[tuple[str, int, int], ...] = ()
    supports_facts: bool = False
    dropped_text: int = 0
    quality_version: int = QUALITY_VERSION
    policy_version: int = POLICY_VERSION
    _source_url: str = field(default="", repr=False, compare=False)
    _producer: object | None = field(default=None, repr=False, compare=False)

    def provenance(self, *, artifact_version: int = 1) -> dict:
        return {
            "tier": self.tier, "adapter": self.adapter, "sanitizer_version": self.sanitizer_version,
            "quality_version": self.quality_version, "policy_version": self.policy_version,
            "quality_flags": list(self.quality_flags), "dropped_nodes": self.dropped_nodes,
            "degraded_reason": self.degraded_reason, "artifact_version": artifact_version,
            "text_anchors": [
                {"anchor_id": anchor, "start": start, "end": end}
                for anchor, start, end in self.text_anchors
            ],
        }


_PRODUCER = object()


def is_produced_result(result: WebExtractionResult, url: str) -> bool:
    return (
        isinstance(result, WebExtractionResult)
        and result._producer is _PRODUCER
        and result._source_url == normalize_url(url)
        and result.sanitizer_version == WEB_SANITIZER_VERSION
        and result.quality_version == QUALITY_VERSION
        and result.policy_version == POLICY_VERSION
        and (result.tier == "L0" or result.supports_facts)
    )


def _result(url: str, *, tier="L1", adapter="static", **values) -> WebExtractionResult:
    return WebExtractionResult(
        tier=tier, final_url=url, adapter=adapter, sanitizer_version=WEB_SANITIZER_VERSION,
        **{
            "title": "", "visible_text": "", "multimodal_blocks": (), "quality_flags": (),
            "dropped_nodes": 0, **values,
        },
    )


@dataclass(eq=False)
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: _Node | None = None
    children: list[_Node | str] = field(default_factory=list)
    end: int = 0
    index: int = 0


class _HTML(HTMLParser):
    def __init__(self, check: Callable[[], object]) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("body", {})
        self.stack = [self.root]
        self.count = 0
        self.comments = 0
        self.check = check

    def _count(self) -> None:
        self.count += 1
        if self.count > 50_000:
            raise GatewayError("size_limit")
        if self.count % 256 == 0:
            self.check()

    def handle_starttag(self, tag, attrs) -> None:
        self._count()
        if len(attrs) > 128 or any(len(key) + len(value or "") > 8192 for key, value in attrs):
            raise GatewayError("size_limit")
        if tag in _CLOSE_P:
            self._close({"p"}, {"table", "div", "section", "article", "main"})
        if tag in {"td", "th", "tr", "thead", "tbody", "tfoot"}:
            self._close({"td", "th"}, {"tr", "table"})
            if tag not in {"td", "th"}:
                self._close({"tr"}, {"table"})
            if tag in {"thead", "tbody", "tfoot"}:
                self._close({"thead", "tbody", "tfoot"}, {"table"})
        if tag == "li":
            self._close({"li"}, {"ul", "ol"})
        parent = self.stack[-1]
        # HTML keeps the first occurrence of duplicate attributes.
        node = _Node(tag, {key: value or "" for key, value in reversed(attrs)}, parent)
        parent.children.append(node)
        if tag not in _VOID:
            if len(self.stack) > 64:
                raise GatewayError("size_limit")
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs) -> None:
        self.handle_starttag(tag, attrs)
        # In HTML a slash does not close a regular non-void element.
        if tag in {"svg", "math"}:
            self.handle_endtag(tag)

    def _close(self, tags, stop=()) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            tag = self.stack[index].tag
            if tag in tags:
                del self.stack[index:]
                return
            if tag in stop:
                return

    def handle_endtag(self, tag) -> None:
        self._close({tag})

    def handle_data(self, data) -> None:
        self._count()
        if len(data) > 262_144:
            raise GatewayError("size_limit")
        self.stack[-1].children.append(data)

    def handle_comment(self, data) -> None:
        self._count()
        self.comments += 1


def _children(node: _Node) -> Iterator[_Node]:
    return (child for child in node.children if isinstance(child, _Node))


def _walk(node: _Node) -> Iterator[_Node]:
    yield node
    for child in _children(node):
        yield from _walk(child)


def _raw_text(node: _Node, *, omit_tables=False) -> str:
    parts = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        elif child.tag == "br":
            parts.append("\n")
        elif not (omit_tables and child.tag == "table"):
            parts.append(_raw_text(child, omit_tables=omit_tables))
    return "".join(parts)


def _snippet_field(value: str, limit: int) -> tuple[str, int]:
    if not isinstance(value, str):
        return "", 0
    value = value[:32_768]
    if not _MARKUP.search(value):
        decoded = _UNESCAPED_ENTITIES.sub(
            lambda match: match[0] if match[0].startswith("\\") else unescape(match[0]), value,
        )
        cleaned = sanitize_web_text(decoded, max_chars=limit)
        return _markdown_paragraph(cleaned), 0
    try:
        dom = _HTML(lambda: None)
        dom.feed(value)
        dom.close()
        visibility = _Visibility(list(_walk(dom.root)))
        visibility.filter(dom.root)
        if visibility.uncertain:
            return "", visibility.dropped + dom.comments
        cleaned = sanitize_web_text(_raw_text(dom.root), max_chars=limit)
        return _markdown_paragraph(cleaned), visibility.dropped + dom.comments
    except GatewayError:
        return "", 0


def sanitize_web_snippet(value: str, *, max_chars: int = 1800) -> str:
    return _snippet_field(value, max_chars)[0]


def normalize_snippet(
    url: str, title: str = "", snippet: str = "", *, projected: bool = False,
) -> WebExtractionResult | None:
    try:
        url = normalize_url(url)
    except GatewayError:
        return None
    if projected:
        dropped_title = dropped_snippet = 0
    else:
        title, dropped_title = _snippet_field(title, 500)
        snippet, dropped_snippet = _snippet_field(snippet, 1800)
    return _result(
        url, tier="L0", adapter="snippet", title=title, visible_text=snippet,
        quality_flags=("snippet_only",) if snippet else ("snippet_only", "link_only"),
        dropped_nodes=dropped_title + dropped_snippet, _source_url=url, _producer=_PRODUCER,
    )


_SELECTOR_PART = re.compile(
    r"""([a-zA-Z][\w-]*|\*)|([.#])([\w-]+)|\[\s*([\w:-]+)(?:\s*=\s*["']?([\w:-]+)["']?)?\s*\]"""
)
_CSS_COMMENT = re.compile(r"/\*.*?(?:\*/|\Z)", re.S)
_CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_VISIBILITY_PROPERTIES = frozenset({
    "display", "visibility", "opacity", "content-visibility", "clip", "clip-path",
    "width", "height", "max-width", "max-height", "overflow", "overflow-x", "overflow-y",
    "position", "left", "top", "right", "bottom", "text-indent", "transform", "font-size",
    "filter", "all",
})
_CSS_ZERO = r"[-+]?(?:0+(?:\.0+)?|\.0+)(?:px|em|rem|%|vw|vh|pt)?"


@dataclass(frozen=True)
class _Selector:
    parts: tuple
    relations: tuple[str, ...]
    specificity: tuple[int, int, int]

    def matches(self, node: _Node | None, index: int = -1) -> bool:
        if node is None:
            return False
        index %= len(self.parts)
        for kind, name, value in self.parts[index]:
            if kind == "tag" and node.tag != name:
                return False
            if kind == "id" and node.attrs.get("id") != name:
                return False
            if kind == "class" and name not in node.attrs.get("class", "").split():
                return False
            if kind == "attribute" and (name not in node.attrs or value is not None and node.attrs[name] != value):
                return False
        if index == 0:
            return True
        ancestor = node.parent
        if self.relations[index - 1] == ">":
            return self.matches(ancestor, index - 1)
        while ancestor is not None:
            if self.matches(ancestor, index - 1):
                return True
            ancestor = ancestor.parent
        return False


def _selector(raw: str) -> _Selector | None:
    tokens = re.split(r"(\s*>\s*|\s+)", raw.strip())
    if not tokens[0] or len(tokens) > 3:
        return None
    segments = []
    weights = [0, 0, 0]
    for part in tokens[::2]:
        parsed = []
        cursor = 0
        for match in _SELECTOR_PART.finditer(part):
            if match.start() != cursor:
                return None
            cursor = match.end()
            tag, prefix, name, attribute, value = match.groups()
            if tag:
                if match.start() != 0:
                    return None
                if tag != "*":
                    weights[2] += 1
                    parsed.append(("tag", tag.lower(), None))
            elif prefix == "#":
                weights[0] += 1
                parsed.append(("id", name, None))
            elif prefix == ".":
                weights[1] += 1
                parsed.append(("class", name, None))
            else:
                weights[1] += 1
                parsed.append(("attribute", attribute.lower(), value))
        if not part or cursor != len(part):
            return None
        segments.append(tuple(parsed))
    return _Selector(tuple(segments), tuple(token.strip() for token in tokens[1::2]), tuple(weights))


def _declarations(css: str) -> Iterator[tuple[str, str, bool]]:
    for item in _CSS_COMMENT.sub("", css).split(";"):
        key, sep, value = item.partition(":")
        key = key.strip().lower()
        if sep and key in _VISIBILITY_PROPERTIES:
            value = value.strip().lower()
            important = re.search(r"!\s*important\s*$", value)
            yield key, value[:important.start()].strip() if important else value, bool(important)


def _style_hidden(style: dict[str, str]) -> tuple[bool, bool]:
    if (
        style.get("display") == "none" or style.get("visibility") in {"hidden", "collapse"}
        or style.get("content-visibility") == "hidden"
        or re.fullmatch(r"[+-]?(?:0+(?:\.0+)?|\.0+)%?", style.get("opacity", ""))
        or re.fullmatch(r"opacity\(\s*(?:0(?:\.0+)?%?|\.0+)\s*\)", style.get("filter", ""))
        or re.fullmatch(_CSS_ZERO, style.get("font-size", ""))
    ):
        return True, False
    clip = style.get("clip", "")
    clip_path = style.get("clip-path", "")
    coordinates = re.fullmatch(r"rect\((.*)\)", clip)
    if coordinates:
        edges = re.split(r",\s*|\s+", coordinates[1].strip())
        if len(edges) == 4 and (edges[0] == edges[2] or edges[1] == edges[3]):
            return True, False
    if re.fullmatch(r"inset\(\s*(?:[5-9]\d|100)%(?:\s+(?:[5-9]\d|100)%){0,3}\s*\)", clip_path):
        return True, False
    clipped = style.get("overflow") in {"hidden", "clip"}
    for dimension, axis in (("width", "x"), ("height", "y")):
        if (clipped or style.get(f"overflow-{axis}") in {"hidden", "clip"}) and any(
            re.fullmatch(_CSS_ZERO, style.get(key, "")) for key in (dimension, f"max-{dimension}")
        ):
            return True, False
    if style.get("position") in {"absolute", "fixed"} and any(
        re.fullmatch(r"-(?:[1-9]\d{3,})px", style.get(side, "")) for side in ("left", "top", "right", "bottom")
    ):
        return True, False
    if re.fullmatch(r"-(?:[1-9]\d{3,})px", style.get("text-indent", "")):
        return True, False
    uncertain = (
        any(re.search(r"var\s*\(|env\s*\(|\\|revert", value) for value in style.values())
        or bool(clip and clip != "auto" or clip_path and clip_path != "none")
        or style.get("transform", "none") != "none"
        or style.get("filter", "none") != "none"
        or "all" in style
        or style.get("content-visibility") == "auto"
    )
    return False, uncertain


class _Visibility:
    def __init__(self, nodes: list[_Node]) -> None:
        self.dropped = 0
        self.uncertain = False
        self.scoped_uncertainty = False
        self.login_form = False
        self.rules = []
        self._by_key = {}
        for node in nodes:
            if node.tag == "link" and "stylesheet" in node.attrs.get("rel", "").lower().split():
                self.uncertain = True
            if node.tag == "style":
                css = _CSS_COMMENT.sub("", _raw_text(node))
                if node.attrs.get("media", "").strip().lower() not in {"", "all", "screen"} or (
                    "@" in css or "\\" in css or len(css) > 65_536
                ):
                    self.uncertain = True
                    continue
                for match in _CSS_RULE.finditer(css):
                    declarations = tuple(_declarations(match[2]))
                    if not declarations:
                        continue
                    for raw in match[1].split(","):
                        selector = _selector(raw)
                        if selector is None or len(self.rules) >= 256:
                            self.uncertain = True
                            break
                        key = next(iter(selector.parts[-1]), ("any", "", None))[:2]
                        rule = (len(self.rules), selector, declarations)
                        self.rules.append(rule)
                        self._by_key.setdefault(key, []).append(rule)

    def _style(self, node: _Node) -> dict[str, str]:
        keys = [
            ("any", ""), ("tag", node.tag), ("id", node.attrs.get("id")),
            *(("class", name) for name in set(node.attrs.get("class", "").split())),
            *(("attribute", name) for name in node.attrs),
        ]
        cascade = {}
        for key in keys:
            for order, selector, declarations in self._by_key.get(key, ()):
                if selector.matches(node):
                    for name, value, important in declarations:
                        priority = (important, 0, selector.specificity, order)
                        if name not in cascade or cascade[name][0] <= priority:
                            cascade[name] = (priority, value)
        for order, (name, value, important) in enumerate(_declarations(node.attrs.get("style", ""))):
            priority = (important, 1, (0, 0, 0), order)
            if name not in cascade or cascade[name][0] <= priority:
                cascade[name] = (priority, value)
        return {name: item[1] for name, item in cascade.items()}

    def state(self, node: _Node) -> tuple[bool, bool]:
        hidden, uncertain = _style_hidden(self._style(node))
        attrs = node.attrs
        hidden |= "hidden" in attrs or "inert" in attrs or attrs.get("aria-hidden", "").strip().lower() == "true"
        return hidden, uncertain

    def filter(self, node: _Node, *, check: Callable[[], object] = lambda: None) -> None:
        check()
        visible = []
        closed_details = node.tag == "details" and "open" not in node.attrs
        for child in node.children:
            if isinstance(child, str):
                if not closed_details:
                    visible.append(child)
                continue
            attrs = child.attrs
            hidden, uncertain = self.state(child)
            if hidden:
                self.dropped += 1
                continue
            if closed_details and child.tag != "summary":
                self.dropped += 1
                continue
            if child.tag in _EXCLUDED or attrs.get("role", "").lower() in {"navigation", "complementary", "banner"}:
                if child.tag == "form":
                    self.login_form |= any(
                        desc.tag == "input" and desc.attrs.get("type", "").lower() == "password"
                        for desc in _walk(child)
                    )
                self.dropped += 1
                continue
            if uncertain:
                self.scoped_uncertainty = True
                self.dropped += 1
                continue
            self.filter(child, check=check)
            visible.append(child)
            if closed_details:
                break
        node.children = visible


@dataclass
class _Paragraph:
    text: str
    node: _Node
    linked: int

    @property
    def count(self) -> int:
        return _char_count(self.text)


def _char_count(value: str) -> int:
    return sum(not char.isspace() for char in value)


def _paragraphs(root: _Node, clean: Callable[[str], str]) -> Iterator[_Paragraph]:
    def visit(node: _Node) -> Iterator[_Paragraph]:
        buffer = []
        links = []

        def flush() -> Iterator[_Paragraph]:
            text = clean("".join(buffer))
            if text:
                yield _Paragraph(text, node, min(_char_count(clean("".join(links))), _char_count(text)))
            buffer.clear()
            links.clear()

        def inline(child: _Node | str, linked: bool = False) -> Iterator[_Paragraph]:
            if isinstance(child, str):
                buffer.append(child)
                if linked:
                    links.append(child)
            elif child.tag == "br":
                buffer.append("\n")
            elif child.tag in _BLOCK_TAGS or child.tag in {"table", "title"}:
                yield from flush()
                if child.tag not in {"table", "title"}:
                    yield from visit(child)
            else:
                for grandchild in child.children:
                    yield from inline(grandchild, linked or child.tag == "a")

        for child in node.children:
            yield from inline(child)
        yield from flush()

    yield from visit(root)


def _table_groups(table: _Node) -> Iterator[list[_Node]]:
    pending: list[_Node] = []
    for child in _children(table):
        if child.tag == "tr":
            pending.append(child)
            continue
        if pending:
            yield pending
            pending = []
        if child.tag in {"thead", "tbody", "tfoot"}:
            yield [row for row in _children(child) if row.tag == "tr"]
    if pending:
        yield pending


def _span(cell: _Node, name: str, *, remaining: int, maximum: int) -> int:
    value = cell.attrs.get(name, "1").strip()
    if not re.fullmatch(r"[0-9]{1,4}", value):
        raise ValueError
    number = int(value)
    if number == 0 and name == "rowspan":
        number = remaining
    if not 1 <= number <= maximum or (name == "rowspan" and number > remaining):
        raise ValueError
    return number


@dataclass
class _Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    role: str
    text: str
    attrs: dict[str, str]
    group: int

    @property
    def anchor(self) -> str:
        return f"r{self.row}-c{self.col}"


def _overlap(a: int, a_span: int, b: int, b_span: int) -> bool:
    return a < b + b_span and b < a + a_span


def _cell_headers(cell: _Cell, headers: list[_Cell], ids: dict[str, list[_Cell]]) -> list[str]:
    if cell.role == "th":
        return []
    if "headers" in cell.attrs:
        matches = [ids.get(key, []) for key in cell.attrs["headers"].split()]
        return list(dict.fromkeys(match[0].anchor for match in matches if len(match) == 1 and match[0].role == "th"))

    explicit = []
    same_row = []
    same_col = []
    for header in headers:
        columns_overlap = _overlap(header.col, header.colspan, cell.col, cell.colspan)
        rows_overlap = _overlap(header.row, header.rowspan, cell.row, cell.rowspan)
        scope = header.attrs.get("scope", "").lower()
        if scope in {"col", "colgroup"} and header.row < cell.row and columns_overlap:
            explicit.append(header.anchor)
        elif scope == "row" and header.col < cell.col and rows_overlap:
            explicit.append(header.anchor)
        elif scope == "rowgroup" and header.group == cell.group and header.row <= cell.row and header.col < cell.col:
            explicit.append(header.anchor)
        elif not scope:
            if columns_overlap and header.row < cell.row:
                same_col.append(header.anchor)
            if rows_overlap and header.col < cell.col:
                same_row.append(header.anchor)
    if explicit:
        return explicit
    return (same_col if len(same_col) == 1 else []) + (same_row if len(same_row) == 1 else [])


@dataclass
class _Table:
    node: _Node
    caption: str
    cells: list[_Cell]
    grid: dict[tuple[int, int], _Cell]
    height: int
    width: int
    truncated: bool

    @property
    def valid(self) -> bool:
        populated_rows = {}
        for cell in self.cells:
            if cell.text:
                populated_rows[cell.row] = populated_rows.get(cell.row, 0) + 1
        return (
            self.width >= 2 and sum(count >= 2 for count in populated_rows.values()) >= 2
            and (bool(self.caption) or any(cell.role == "th" and cell.text for cell in self.cells))
        )

    def block(
        self, anchor: str, before: str, after: str, *, row_limit: int, check: Callable[[], object],
    ) -> dict | None:
        if row_limit <= 0:
            return None
        height = min(self.height, row_limit)
        # A truncated matrix cannot retain an origin whose declared span leaves it.
        for row in range(height - 1, -1, -1):
            if any(cell.row == row and cell.row + cell.rowspan > height for cell in self.cells):
                height = row
        if not height:
            return None
        cells = [cell for cell in self.cells if cell.row < height]
        lookup = {(cell.row, cell.col): cell.text for cell in cells}
        rows = [[lookup.get((row, col), "") for col in range(self.width)] for row in range(height)]
        has_header_row = any(cell.row == 0 and cell.role == "th" for cell in cells)
        headers = rows.pop(0) if has_header_row else [""] * self.width
        header_cells = [cell for cell in cells if cell.role == "th" and cell.text]
        ids: dict[str, list[_Cell]] = {}
        for cell in cells:
            if key := cell.attrs.get("id"):
                ids.setdefault(key, []).append(cell)
        structure = []
        for cell in cells:
            if len(structure) % 64 == 0:
                check()
            structure.append({
                "row": cell.row, "col": cell.col, "rowspan": cell.rowspan, "colspan": cell.colspan,
                "role": cell.role, "text": cell.text, "header_refs": _cell_headers(cell, header_cells, ids),
            })
        origins = [
            [self.grid[row, col].anchor if (row, col) in self.grid else None for col in range(self.width)]
            for row in range(height)
        ]
        unit = _UNIT.search("\n".join((self.caption, before, after)))
        markdown = []
        if self.caption:
            markdown.append(_markdown_paragraph(self.caption))
        for row in [headers, ["---"] * self.width, *rows]:
            markdown.append("| " + " | ".join(_markdown_text(value).replace("\n", " ") for value in row) + " |")
        return MultimodalEvidenceBlock(
            anchor_id=anchor, modality="table", title=_markdown_paragraph(self.caption),
            markdown="\n".join(markdown),
            structured_payload={
                "schema_version": 1, "headers": headers, "rows": rows, "caption": self.caption,
                "unit_text": unit[1].strip() if unit else "", "context_before": before, "context_after": after,
                "cells": structure, "origins": origins, "truncated": self.truncated or height < self.height,
            },
        ).to_dict()


def _parse_table(table: _Node, clean: Callable[[str], str], check: Callable[[], object]) -> _Table | None:
    if table.attrs.get("role", "").lower() in {"presentation", "none", "layout"}:
        return None
    caption = next((clean(_raw_text(child, omit_tables=True))[:512] for child in _children(table) if child.tag == "caption"), "")
    parsed = _Table(table, caption, [], {}, 0, 0, False)
    for group_index, group in enumerate(_table_groups(table)):
        for group_row, row in enumerate(group):
            check()
            row_index = parsed.height
            additions = []
            occupied = {}
            col = 0
            try:
                if row_index >= 128:
                    raise ValueError
                for child in _children(row):
                    if child.tag not in {"th", "td"}:
                        continue
                    while (row_index, col) in parsed.grid:
                        col += 1
                    rowspan = _span(child, "rowspan", remaining=len(group) - group_row, maximum=128 - row_index)
                    colspan = _span(child, "colspan", remaining=0, maximum=64 - col)
                    value = _raw_text(child, omit_tables=True)
                    text = clean(value)
                    if len(text) > 512:
                        parsed.truncated = True
                        raise ValueError
                    entry = _Cell(row_index, col, rowspan, colspan, child.tag, text, child.attrs, group_index)
                    for r in range(row_index, row_index + rowspan):
                        for c in range(col, col + colspan):
                            if (r, c) in parsed.grid or (r, c) in occupied:
                                raise ValueError
                            occupied[r, c] = entry
                    additions.append(entry)
                    col += colspan
            except ValueError:
                parsed.truncated = True
                return parsed
            parsed.cells.extend(additions)
            parsed.grid.update(occupied)
            parsed.height += 1
            parsed.width = max(parsed.width, col)
    return parsed


@dataclass
class _Region:
    node: _Node
    paragraphs: list[_Paragraph] = field(default_factory=list)
    tables: list[_Table] = field(default_factory=list)
    count: int = 0
    links: int = 0

    @property
    def semantic(self) -> int:
        if self.node.tag == "main" or self.node.attrs.get("role", "").lower() == "main":
            return 3
        return {"article": 2, "section": 1}.get(self.node.tag, 0)

    @property
    def link_ratio(self) -> float:
        return self.links / self.count if self.count else 0

    def order(self) -> tuple[float, int, int]:
        density = min(1, self.count / (40 * max(1, len(self.paragraphs))))
        paragraphs = sum(p.count - p.linked >= 24 for p in self.paragraphs)
        score = (
            min(self.count, 2400) * (1 - self.link_ratio) * (0.5 + 0.5 * density)
            + 40 * min(paragraphs, 10) + 160 * min(len(self.tables), 3) + 80 * (self.semantic >= 2)
        )
        return score, self.semantic, -self.node.index


def _regions(
    root: _Node, paragraphs: list[_Paragraph], tables: list[_Table], check: Callable[[], object],
) -> tuple[_Region, int]:
    regions = {
        node: _Region(node) for node in _walk(root)
        if node is root or node.tag in _REGIONS - {"body"} or node.attrs.get("role", "").lower() == "main"
    }
    for paragraph in paragraphs:
        check()
        node = paragraph.node
        count = paragraph.count
        while node is not None:
            if (region := regions.get(node)) is not None:
                region.paragraphs.append(paragraph)
                region.count += count
                region.links += paragraph.linked
            node = node.parent
    for table in tables:
        node = table.node.parent
        while node is not None:
            if (region := regions.get(node)) is not None:
                region.tables.append(table)
            node = node.parent
    candidates = [item for node, item in regions.items() if node is not root and (item.count or item.tables)]
    return max(candidates, key=_Region.order, default=regions[root]), regions[root].count


def _assemble_text(paragraphs: list[_Paragraph], limit: int) -> tuple[str, tuple, list[_Paragraph]]:
    parts = []
    anchors = []
    retained = []
    position = 0
    for paragraph in paragraphs:
        text = _markdown_paragraph(paragraph.text)
        start = position + (2 if parts else 0)
        if start + len(text) > limit:
            break
        position = start + len(text)
        parts.append(text)
        retained.append(paragraph)
        anchors.append((f"html-p{len(anchors) + 1}", start, position))
    return "\n\n".join(parts), tuple(anchors), retained


def _region_paragraphs(region: _Region, paragraphs: list[_Paragraph]) -> list[_Paragraph]:
    selected = region.paragraphs
    parent = region.node.parent
    if parent is None or parent.tag not in {"main", "article"}:
        return selected
    adjacent = []
    siblings = list(_children(parent))
    index = siblings.index(region.node)
    if index:
        previous = siblings[index - 1]
        if previous.tag in _HEADINGS or previous.tag == "header":
            adjacent = [p for p in paragraphs if previous.index <= p.node.index <= previous.end]
    return adjacent + selected


def _quality_flags(region: _Region, page_count: int, *, js_shell: bool, uncertain: bool) -> list[str]:
    if region.count < 240 and not region.tables and js_shell:
        flags = ["js_shell"]
    elif (
        (region.count < 400 and region.link_ratio > 0.60)
        or (region.count < 240 and page_count and region.count / page_count < 0.25)
        or (region.count - region.links < 80 and uncertain)
    ):
        flags = ["low_snr"]
    else:
        flags = []
    if uncertain:
        flags.append("visibility_uncertain")
    return flags


def extract_html(
    html: str, final_url: str, *, max_chars: int = MAX_TEXT, check: Callable[[], object] = lambda: None,
) -> WebExtractionResult:
    """Produce a deterministic visible-text projection and origin-preserving table blocks."""
    try:
        return _extract_html(html, final_url, max_chars=min(MAX_TEXT, max(0, max_chars)), check=check)
    except GatewayError as error:
        if error.reason != "size_limit":
            raise
        return _result(final_url, degraded_reason=error.reason)


def _extract_html(html: str, final_url: str, *, max_chars: int, check: Callable[[], object]) -> WebExtractionResult:
    if len(html) > 4 * MAX_DOCUMENT:
        raise GatewayError("size_limit")
    dom = _HTML(check)
    # Chunking also bounds the cancellation delay while HTMLParser scans ordinary text.
    for start in range(0, len(html), 16_384):
        check()
        dom.feed(html[start:start + 16_384])
    dom.close()
    nodes = list(_walk(dom.root))
    scripts = any(
        node.tag == "script" and (
            node.attrs.get("type", "").lower() == "module"
            or node.attrs.get("id") in {"__NEXT_DATA__", "__NUXT_DATA__"}
            or re.search(r"(?:app|bundle|main|runtime|hydrate)[^/]*\.m?js(?:\?|$)", node.attrs.get("src", ""), re.I)
        ) for node in nodes
    )
    visibility = _Visibility(nodes)
    dropped_text = 0

    def report_drop() -> None:
        nonlocal dropped_text
        dropped_text += 1

    def clean(value: str) -> str:
        check()
        return sanitize_web_text(value, on_drop=report_drop)

    title = ""
    for node in nodes:
        if node.tag == "title" and node.parent is not None and node.parent.tag == "head":
            ancestor = node
            while ancestor is not None and not any(visibility.state(ancestor)):
                ancestor = ancestor.parent
            if ancestor is None:
                title = clean(_raw_text(node))[:500]
                break
            visibility.dropped += 1
    visibility.filter(dom.root, check=check)
    dropped_nodes = visibility.dropped + dom.comments
    if visibility.uncertain:
        return _result(
            final_url, quality_flags=("visibility_uncertain", "low_snr"), dropped_nodes=dropped_nodes,
            degraded_reason="visibility_uncertain", dropped_text=dropped_text,
        )

    nodes = list(_walk(dom.root))
    for index, node in enumerate(nodes):
        node.index = node.end = index
    for node in reversed(nodes[1:]):
        node.parent.end = max(node.parent.end, node.end)
    paragraphs = list(_paragraphs(dom.root, clean))
    mount_empty = any(
        node.attrs.get("id", "").lower() in {"app", "root", "__next", "__nuxt", "app-root"}
        and _char_count(_raw_text(node, omit_tables=True)) < 40
        for node in nodes
    )

    # Recognize the page-level access form before scoring it as content.
    login_heading = any(p.node.tag in _HEADINGS and _LOGIN.search(p.text) for p in paragraphs)
    access_heading = any(
        p.node.tag in _HEADINGS and len(p.text) <= 100
        and re.search(r"access denied|subscription required|authentication required|访问受限", p.text, re.I)
        for p in paragraphs
    )
    if access_heading or (visibility.login_form and login_heading):
        return _result(final_url, dropped_nodes=dropped_nodes, dropped_text=dropped_text, degraded_reason="access_denied")

    js_placeholder = any(_JS.search(p.text) for p in paragraphs)
    if scripts and js_placeholder:
        paragraphs = [p for p in paragraphs if not _JS.search(p.text)]
    tables = []
    invalid_structure = False
    for node in nodes:
        if node.tag != "table":
            continue
        check()
        if len(tables) >= MAX_TABLES:
            invalid_structure = True
            break
        table = _parse_table(node, clean, check)
        if table is not None:
            invalid_structure |= table.truncated
            if table.valid:
                tables.append(table)
    region, page_count = _regions(dom.root, paragraphs, tables, check)
    flags = _quality_flags(
        region, page_count, js_shell=scripts and (mount_empty or js_placeholder),
        uncertain=visibility.scoped_uncertainty,
    )
    chosen_paragraphs = _region_paragraphs(region, paragraphs)
    heading = next((p.text for p in chosen_paragraphs if p.node.tag in _HEADINGS), "")
    text, anchors, retained = _assemble_text(chosen_paragraphs, max_chars)

    blocks = []
    slots = MAX_TABLE_SLOTS
    table_chars = max_chars - len(text)
    for table in region.tables:
        check()
        before = next((p.text[:512] for p in reversed(chosen_paragraphs) if p.node.end < table.node.index), "")
        after = next((p.text[:512] for p in chosen_paragraphs if p.node.index > table.node.end), "")
        row_limit = min(slots // table.width, table.height)
        block = None
        while row_limit > 0:
            block = table.block(f"html-t{len(blocks) + 1}", before, after, row_limit=row_limit, check=check)
            if block is not None and len(block["text"]) <= table_chars:
                break
            row_limit -= 1
        if not block or row_limit <= 0:
            invalid_structure = True
            break
        payload = block["structured_payload"]
        if not table.valid or sum(
            sum(bool(value) for value in row) >= 2
            for row in [payload["headers"], *payload["rows"]]
        ) < 2:
            invalid_structure = True
            continue
        blocks.append(block)
        table_chars -= len(block["text"])
        slots -= len(payload["origins"]) * table.width
        invalid_structure |= payload["truncated"]
    if blocks:
        flags.append("has_table")
    if invalid_structure or len(retained) < len(chosen_paragraphs):
        flags.append("structure_truncated")
    unlinked = sum(p.count - p.linked for p in retained)
    located_body = any(p.node.tag not in _HEADINGS and p.count > p.linked for p in retained)
    body_ok = located_body and (
        unlinked >= 80 or (40 <= unlinked < 80 and bool(heading or title) and not flags[:1] == ["js_shell"]
                           and "low_snr" not in flags and region.link_ratio <= 0.60)
    )
    valid = bool(blocks) or body_ok
    reason = None if valid else ("visibility_uncertain" if visibility.scoped_uncertainty else "quality_insufficient")
    return _result(
        final_url, title=_markdown_paragraph(heading or title), visible_text=text, multimodal_blocks=tuple(blocks),
        quality_flags=tuple(flags), dropped_nodes=dropped_nodes, text_anchors=anchors, supports_facts=valid,
        degraded_reason=reason, dropped_text=dropped_text,
    )


def _decode(body: bytes, content_type: str, *, html: bool) -> str:
    candidates = []
    for bom, encoding in (
        (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if body.startswith(bom):
            candidates.append(encoding)
            break
    if declared := _CHARSET.search(content_type):
        candidates.append(declared[1])
    if html and (meta := _META_CHARSET.search(body[:4096])):
        candidates.append(meta[1].decode("ascii"))
    for codec in candidates:
        try:
            encoding = codecs.lookup(codec).name
            if encoding not in {"utf-7", "unicode-escape", "raw-unicode-escape", "idna"}:
                return body.decode(encoding, errors="replace")
        except (LookupError, UnicodeError, ValueError):
            continue
    return body.decode("utf-8", errors="replace")


def _task_words(value: str) -> tuple[str, ...]:
    tokens = {}
    for word in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", unicodedata.normalize("NFKC", value[:4096]).casefold()):
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.update(dict.fromkeys(word[index:index + 2] for index in range(len(word) - 1)))
        else:
            tokens[word] = None
        if len(tokens) >= 32:
            break
    return tuple(tokens)[:32]


def _table_signatures(result: WebExtractionResult) -> set[tuple]:
    return {
        (
            tuple(payload["headers"]), tuple(tuple(row) for row in payload["rows"]),
            payload["unit_text"],
        )
        for block in result.multimodal_blocks
        if (payload := block.get("structured_payload")) and payload.get("schema_version") == 1
    }


def select_candidate(
    static: WebExtractionResult, dynamic: WebExtractionResult, task_terms: str = "",
) -> WebExtractionResult:
    if not dynamic.supports_facts or _table_signatures(static) - _table_signatures(dynamic):
        return static
    words = _task_words(task_terms)

    def rank(result: WebExtractionResult) -> tuple:
        combined = result.visible_text + "\n" + "\n".join(block["text"] for block in result.multimodal_blocks)
        corpus = unicodedata.normalize("NFKC", combined).casefold()
        ascii_tokens = set(re.findall(r"[a-z0-9]+", corpus))
        coverage = sum(word in ascii_tokens if word.isascii() else word in corpus for word in words)
        return (
            result.supports_facts, bool(result.multimodal_blocks), bool(result.text_anchors),
            coverage, len(result.multimodal_blocks), _char_count(result.visible_text),
        )

    return dynamic if rank(dynamic) > rank(static) else static


@dataclass(frozen=True)
class ExtractionContext:
    tenant_id: str
    request_id: str
    task_id: int
    scope: FetchScope
    task_terms: str = ""
    needs_coverage: bool = False
    source_id: str | None = None
    allow_network: bool = True
    can_dispatch: Callable[[dict[str, int]], bool] | None = None
    static_timeout: float = 10.0
    network_allowed: Callable[[], bool] | None = None
    dynamic_attempt: Callable[[dict[str, int], tuple[str, ...]], AbstractContextManager] | None = None
    dynamic_allowed: Callable[[], bool] | None = None
    on_result: Callable[[str | None, str | None], None] | None = None


class ExtractionAdapter(Protocol):
    name: Literal["reader_http", "playwright"]

    def extract(
        self, url: str, *, scope: FetchScope, tenant_id: str, request_id: str,
        on_dispatch: Callable[[], None], authorize_redirect: Callable[[str], bool],
        **kwargs,
    ): ...

    def can_reuse(self, url: str, *, tenant_id: str) -> bool: ...


def _copy_result(result: WebExtractionResult) -> WebExtractionResult:
    return replace(result, multimodal_blocks=deepcopy(result.multimodal_blocks))


@dataclass
class _DynamicAttempt:
    dispatched: bool = False


class WebExtractor:
    """Cache only sanitized artifacts and deduplicate dynamic work per request."""

    def __init__(
        self, *, gateway: WebFetchGateway | None = None, adapter: ExtractionAdapter | None = None,
        max_cache_entries: int = 32, dynamic_timeout: float = 12.0,
        pdf_parser: Callable[[bytes, Callable[[], object]], str] | None = None, pdf_limit: int = MAX_DOCUMENT,
    ) -> None:
        self.gateway = gateway if gateway is not None else WebFetchGateway()
        self.adapter = adapter
        self.pdf_parser = pdf_parser
        self.pdf_limit = min(MAX_DOCUMENT, pdf_limit)
        self.dynamic_timeout = min(12.0, max(0.001, dynamic_timeout))
        self._limit = max(1, min(64, max_cache_entries))
        self._lock = Lock()
        self._cache: OrderedDict[tuple, WebExtractionResult] = OrderedDict()
        self._dynamic_targets: dict[tuple, _DynamicAttempt] = {}

    @staticmethod
    def _key(context: ExtractionContext, url: str, tier: str, adapter: str) -> tuple:
        return (
            context.tenant_id, context.request_id, url, tier, adapter,
            POLICY_VERSION, WEB_SANITIZER_VERSION, QUALITY_VERSION,
        )

    def clear_request(self, tenant_id: str, request_id: str) -> None:
        with self._lock:
            scope = (tenant_id, request_id)
            for key in tuple(self._cache):
                if key[:2] == scope:
                    self._cache.pop(key)
            for key in tuple(self._dynamic_targets):
                if key[:2] == scope:
                    self._dynamic_targets.pop(key)

    def _remember(self, key: tuple, result: WebExtractionResult) -> None:
        with self._lock:
            self._cache[key] = _copy_result(result)
            self._cache.move_to_end(key)
            if len(self._cache) > self._limit:
                self._cache.popitem(last=False)

    def _cached(self, key: tuple) -> WebExtractionResult | None:
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
        if result is not None:
            metrics_registry.record_web_extraction("cache_hit", tier=key[3], adapter=key[4])
            return _copy_result(result)
        return None

    def _document(self, document, check: Callable[[], object]) -> WebExtractionResult:
        check()
        content_type = document.content_type.partition(";")[0].strip().lower()
        if content_type in HTML_TYPES:
            text = _decode(document.body, document.content_type, html=True)
            return extract_html(text, document.final_url, check=check)
        if content_type == "application/pdf":
            if self.pdf_parser is None:
                return _result(document.final_url, degraded_reason="adapter_unavailable")
            if len(document.body) > self.pdf_limit:
                return _result(document.final_url, degraded_reason="size_limit")
            try:
                text = self.pdf_parser(document.body, check)
            except GatewayError:
                raise
            except (RuntimeError, ValueError, OSError):
                return _result(document.final_url, degraded_reason="response_invalid")
        elif content_type in TEXT_TYPES:
            text = _decode(document.body, document.content_type, html=False)
        else:
            return _result(document.final_url, degraded_reason="content_type_denied")
        root = _Node("body", {})
        pieces = []
        dropped_text = 0
        for paragraph in re.split(r"\n[ \t]*\n", text):
            check()
            dropped = []

            def on_drop():
                dropped.append(True)

            clean = sanitize_web_text(paragraph, on_drop=on_drop)
            dropped_text += bool(dropped)
            if clean and len(clean) < MAX_TEXT:
                pieces.append(_Paragraph(clean, root, 0))
        visible_text, anchors, retained = _assemble_text(pieces, MAX_TEXT)
        valid = sum(p.count for p in retained) >= 80
        return _result(
            document.final_url, visible_text=visible_text, supports_facts=valid, text_anchors=anchors,
            quality_flags=("structure_truncated",) if len(pieces) > len(retained) else (),
            dropped_text=dropped_text, degraded_reason=None if valid else "quality_insufficient",
        )

    def _reserve_dynamic(self, keys: tuple[tuple, ...]) -> _DynamicAttempt | None:
        with self._lock:
            if any(key in self._dynamic_targets for key in keys) or len(self._dynamic_targets) >= 1024:
                return None
            attempt = _DynamicAttempt()
            for key in keys:
                self._dynamic_targets[key] = attempt
            return attempt

    def _dynamic(
        self, url: str, static: WebExtractionResult, context: ExtractionContext,
    ) -> tuple[WebExtractionResult, str | None]:
        if not (context.needs_coverage and {"js_shell", "low_snr"}.intersection(static.quality_flags)):
            return static, static.degraded_reason
        if self.adapter is None:
            return static, "adapter_unavailable"
        if context.scope.attempt is nullcontext:
            return static, "budget_denied"
        name = self.adapter.name
        key = self._key(context, url, "L2", name)
        final_key = self._key(context, static.final_url, "L2", name)
        def check_policy() -> None:
            if context.scope.check_policy is not None:
                context.scope.check_policy()
            if context.network_allowed is not None and not context.network_allowed():
                raise GatewayError("adapter_unavailable")
            if context.dynamic_allowed is not None and not context.dynamic_allowed():
                raise GatewayError("adapter_unavailable")

        scope = replace(
            context.scope, adapter=name,
            deadline=min(context.scope.deadline, context.scope.monotonic() + self.dynamic_timeout),
            check_policy=check_policy,
        )
        scope.check()
        if result := self._cached(final_key):
            if self.adapter.can_reuse(static.final_url, tenant_id=context.tenant_id) and self.adapter.can_reuse(
                result.final_url, tenant_id=context.tenant_id,
            ):
                selected = select_candidate(static, result, context.task_terms)
                return selected, None if selected is result else "quality_insufficient"
            return static, "origin_not_allowed"
        attempt = self._reserve_dynamic((key, final_key))
        if attempt is None:
            return static, "not_eligible"

        def on_dispatch() -> None:
            attempt.dispatched = True

        def authorize_redirect(target: str) -> bool:
            alias = self._key(context, target, "L2", name)
            with self._lock:
                previous = self._dynamic_targets.get(alias)
                if previous is not None:
                    return previous is attempt
                if len(self._dynamic_targets) >= 1024:
                    return False
                self._dynamic_targets[alias] = attempt
                return True

        try:
            document = self.adapter.extract(
                static.final_url, scope=scope, tenant_id=context.tenant_id, request_id=context.request_id,
                on_dispatch=on_dispatch, authorize_redirect=authorize_redirect,
                **({"dispatch_attempt": lambda demand: context.dynamic_attempt(demand, (url, static.final_url))}
                   if context.dynamic_attempt is not None else {}),
                **({"can_dispatch": context.can_dispatch} if name == "playwright" else {}),
            )
            candidate = replace(self._document(document, scope.check), tier="L2", adapter=name)
            scope.check()
            for event, count in (
                ("sanitizer_dropped_nodes", candidate.dropped_nodes), ("sanitizer_dropped_text", candidate.dropped_text),
            ):
                if count:
                    metrics_registry.record_web_extraction(event, amount=count)
            if candidate.supports_facts:
                self._remember(final_key, candidate)
                self._remember(self._key(context, candidate.final_url, "L2", name), candidate)
                selected = select_candidate(static, candidate, context.task_terms)
                return selected, None if selected is candidate else "quality_insufficient"
            return static, candidate.degraded_reason or "quality_insufficient"
        except GatewayError as error:
            if error.reason in {"lease_lost", "cancelled"}:
                raise
            return static, error.reason
        finally:
            if not attempt.dispatched:
                with self._lock:
                    for alias in tuple(self._dynamic_targets):
                        if self._dynamic_targets[alias] is attempt:
                            self._dynamic_targets.pop(alias)

    def extract(
        self, url: str, *, context: ExtractionContext, snippet: WebExtractionResult | None = None,
    ) -> WebExtractionResult | None:
        started = context.scope.monotonic()
        chosen = None
        reason = None
        origin = ""
        try:
            context.scope.check()
            origin = normalize_url(url)
            if not context.tenant_id or not context.request_id or context.task_id <= 0:
                raise GatewayError("not_eligible")
            if snippet is not None and is_produced_result(snippet, origin) and snippet.tier == "L0":
                chosen = snippet
            key = self._key(context, origin, "L1", "static")
            static = self._cached(key)
            if static is None:
                if not context.allow_network:
                    raise GatewayError("budget_denied")

                def check_network() -> None:
                    if context.network_allowed is not None and not context.network_allowed():
                        raise GatewayError("budget_denied")

                static_scope = replace(context.scope.bounded(context.static_timeout), check_policy=check_network)
                document = self.gateway.fetch(
                    origin, scope=static_scope, allowed_types=STATIC_TYPES,
                    content_limits={"application/pdf": self.pdf_limit},
                )
                static = self._document(document, static_scope.check)
                static_scope.check()
                self._remember(key, static)
                if static.final_url != origin:
                    self._remember(self._key(context, static.final_url, "L1", "static"), static)
                for event, count in (
                    ("sanitizer_dropped_nodes", static.dropped_nodes), ("sanitizer_dropped_text", static.dropped_text),
                ):
                    if count:
                        metrics_registry.record_web_extraction(event, amount=count)
            if static.supports_facts:
                chosen = static
            if not context.allow_network:
                raise GatewayError("budget_denied")
            winner, reason = self._dynamic(origin, static, context)
            if winner.supports_facts:
                chosen = winner
            context.scope.check()
        except GatewayError as error:
            reason = error.reason
            if reason in {"lease_lost", "cancelled"}:
                chosen = None
            if reason in {"lease_lost", "cancelled"}:
                self.clear_request(context.tenant_id, context.request_id)
        adapter = chosen.adapter if chosen is not None else "static"
        outcome = (
            "cancelled" if reason in {"lease_lost", "cancelled"}
            else ("empty" if chosen is None else ("degraded" if reason else "success"))
        )
        metrics_registry.record_web_extraction(
            "result", tier=chosen.tier if chosen is not None else "none", adapter=adapter, outcome=outcome,
        )
        metrics_registry.record_web_extraction(
            "elapsed", adapter=adapter, elapsed_ms=(context.scope.monotonic() - started) * 1000,
        )
        if reason is not None:
            metrics_registry.record_web_extraction("degraded", reason=reason)
        if context.on_result is not None and reason not in {"cancelled", "lease_lost"}:
            context.on_result(chosen.tier if chosen is not None else None, reason)
        if chosen is None:
            return None
        return replace(_copy_result(chosen), degraded_reason=reason, _source_url=origin, _producer=_PRODUCER)
