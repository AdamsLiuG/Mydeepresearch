"""Evidence storage, lookup, and tool helpers for grounded research outputs."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import time
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Iterator
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests

try:
    from hello_agents.tools import Tool, ToolParameter
except Exception:  # pragma: no cover - test stubs may only expose a partial module
    class ToolParameter:  # type: ignore[override]
        def __init__(
            self,
            *,
            name: str,
            type: str,
            description: str,
            required: bool = True,
            default: Any = None,
        ) -> None:
            self.name = name
            self.type = type
            self.description = description
            self.required = required
            self.default = default

    class Tool:  # type: ignore[override]
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

from config import Configuration
from metrics import WEB_EXTRACTION_REASONS, RequestTrace
from services.evidence_index import (
    ChunkLazyLoadResult,
    EvidenceChunk,
    EvidenceRetrievalHit,
    EvidenceRetrievalService,
    EvidenceStoreListener,
    pack_lazy_chunks,
)
from services.memory_sanitizer import (
    WEB_SANITIZER_VERSION,
    sanitize_injection_text,
    sanitize_web_text,
)
from services.search import dispatch_search, projected_search_snippet
from services.text_processing import (
    CITATION_PATTERN,
    normalize_agent_markdown,
    strip_citation_markers,
    truncate_cited_text,
)
from services.web_extractor import (
    POLICY_VERSION,
    QUALITY_VERSION,
    ExtractionContext,
    WebExtractionResult,
    WebExtractor,
    is_produced_result,
    normalize_snippet,
    sanitize_web_snippet,
)
from services.web_fetch_gateway import FetchScope, GatewayError
from services.web_fetch_gateway import normalize_url as normalize_web_url
from utils import truncate_text

try:  # pragma: no cover - optional dependency
    from markdownify import markdownify
except Exception:  # pragma: no cover - optional dependency
    markdownify = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import trafilatura
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import fitz as pymupdf
except Exception:  # pragma: no cover - optional dependency
    pymupdf = None  # type: ignore

TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_SCRIPT_PATTERN = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
HTML_BOILERPLATE_PATTERN = re.compile(
    r"<(script|style|noscript|nav|header|footer|aside)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
HTML_BLOCK_BREAK_PATTERN = re.compile(
    r"</?(?:p|div|section|article|main|h[1-6]|li|tr|blockquote|pre|br)\b[^>]*>",
    re.IGNORECASE,
)
HTML_TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})[./\-年](?P<month>\d{1,2})[./\-月](?P<day>\d{1,2})日?"
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
URL_PATTERN = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"\barxiv:\s*\d{4}\.\d{4,5}\b", re.IGNORECASE)
NUMBERED_REFERENCE_PATTERN = re.compile(r"(?m)^\s*(?:\[\d+\]|\d+\.)\s+\S+")
HEADING_PATTERN = re.compile(
    r"(?m)^(?:#{1,6}\s+\S+|\d+(?:\.\d+)*\s+\S+|(?:references|reference|faq|conclusion|summary|overview)\s*:?)$",
    re.IGNORECASE,
)
TITLE_CLAUSE_SPLIT_PATTERN = re.compile(r"\s*[\-|:|]\s*")
SOCIAL_HANDLE_PATTERN = re.compile(r"^/?@?(?P<handle>[A-Za-z0-9_.-]{2,40})/?")
RESEARCH_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9.+_-]{1,}|20\d{2}|[\u4e00-\u9fff]{2,}")
LIST_PREFIX_PATTERN = re.compile(r"^(?:[-*•]\s+|\d+[.)、]\s+|[（(]?\d+[)）]\s+)")
SOURCE_LABEL_PLACEHOLDER_PATTERNS = (
    re.compile(r"质量等级", re.IGNORECASE),
    re.compile(r"时效标签", re.IGNORECASE),
    re.compile(r"看起来.{0,6}(?:相关|不相关)", re.IGNORECASE),
    re.compile(r"(?:来源|网页|搜索)目录", re.IGNORECASE),
    re.compile(r"搜索结果|结果页", re.IGNORECASE),
    re.compile(
        r"^(?:[\w\u4e00-\u9fff（）()《》·/+-]{1,32})?(?:社区|论坛|专栏|网站|网页|页面|博客|链接|文章|帖子|问答)(?:[,，:： -].*)?$",
        re.IGNORECASE,
    ),
)
GENERIC_RESEARCH_TOKENS = {
    "ai",
    "agent",
    "agents",
    "analysis",
    "latest",
    "llm",
    "llms",
    "model",
    "models",
    "overview",
    "report",
    "research",
    "summary",
    "task",
    "workflow",
    "主题",
    "任务",
    "分析",
    "原因",
    "场景",
    "如何",
    "对比",
    "总结",
    "报告",
    "研究",
    "背景",
    "问题",
    "进展",
    "趋势",
}

SOURCE_TYPE_BONUS = {
    "government": 4.0,
    "education": 3.5,
    "official_documentation": 3.5,
    "official_product": 3.0,
    "official_org": 3.0,
    "peer_reviewed_paper": 3.5,
    "preprint_paper": 2.5,
    "standards_spec": 3.5,
    "repository_official": 2.5,
    "repository_unofficial": 1.5,
    "reference_curated": 1.5,
    "news_primary": 1.5,
    "news_secondary": 1.0,
    "technical_blog": 1.0,
    "company_blog": 1.0,
    "forum_expert": 0.5,
    "forum_general": -0.5,
    "social_official": 0.5,
    "social_general": -1.0,
    "content_farm_or_aggregator": -2.0,
    "web_general": 0.0,
    "mcp_curated_filing": 3.8,
    "mcp_internal_metrics": 3.2,
    "mcp_external": 1.8,
    "mcp_filing": 3.5,
    "mcp_tool": 2.5,
    "sandbox_computation": 3.5,
    "computed_sandbox": 3.5,
}

PROVIDER_BONUS = {
    1: 0.0,
    2: 0.8,
    3: 1.5,
}

NEWS_PRIMARY_DOMAINS = {
    "apnews.com",
    "axios.com",
    "bbc.com",
    "bloomberg.com",
    "cnbc.com",
    "cnn.com",
    "ft.com",
    "nytimes.com",
    "reuters.com",
    "theverge.com",
    "techcrunch.com",
    "washingtonpost.com",
    "wsj.com",
}
NEWS_SECONDARY_DOMAINS = {
    "msn.com",
    "newsbreak.com",
    "news.google.com",
    "smartnews.com",
    "yahoo.com",
}
AGGREGATOR_DOMAINS = NEWS_SECONDARY_DOMAINS | {
    "flipboard.com",
    "feedly.com",
}
FORUM_DOMAINS = {
    "reddit.com",
    "news.ycombinator.com",
    "stackoverflow.com",
    "stackexchange.com",
    "zhihu.com",
    "quora.com",
    "v2ex.com",
}
SOCIAL_DOMAINS = {
    "x.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "threads.net",
    "weibo.com",
    "youtube.com",
    "t.me",
    "mastodon.social",
}
REFERENCE_DOMAINS = {
    "baike.baidu.com",
    "britannica.com",
    "wikipedia.org",
}
PREPRINT_DOMAINS = {
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
}
PAPER_DOMAINS = {
    "acm.org",
    "doi.org",
    "ieeexplore.ieee.org",
    "jmlr.org",
    "nature.com",
    "pubmed.ncbi.nlm.nih.gov",
    "science.org",
    "sciencedirect.com",
    "semanticscholar.org",
    "springer.com",
}
STANDARDS_DOMAINS = {
    "ecma-international.org",
    "ietf.org",
    "iso.org",
    "rfc-editor.org",
    "w3.org",
}
COMMON_SUBDOMAIN_LABELS = {
    "api",
    "app",
    "blog",
    "cdn",
    "developer",
    "developers",
    "docs",
    "help",
    "m",
    "news",
    "support",
    "www",
}
GENERIC_DOMAIN_LABELS = {
    "ac",
    "co",
    "com",
    "edu",
    "gov",
    "net",
    "org",
}
OFFICIAL_PATH_TOKENS = ("api", "developer", "developers", "docs", "help", "manual", "reference", "support")
PRODUCT_PATH_TOKENS = ("changelog", "download", "platform", "pricing", "product", "products", "release", "releases")
BLOG_PATH_TOKENS = ("blog", "engineering", "insights", "news", "posts", "stories")
NAVIGATION_TOKENS = ("all", "archive", "archives", "category", "categories", "compare", "index", "list", "page", "search", "tag", "tags")
CLICKBAIT_PATTERNS = (
    "you won't believe",
    "what happens next",
    "top 10",
    "top ten",
    "best ever",
    "shocking",
    "must read",
    "ultimate guide",
)
TEASER_PATTERNS = (
    "learn more",
    "read more",
    "coming soon",
    "stay tuned",
    "click here",
)
AUTHOR_ATTRIBUTION_PATTERNS = (
    "by ",
    "published by",
    "editor",
    "author",
    "source:",
    "writer",
    "作者",
    "编辑",
    "来源",
)
EXPERT_PATTERNS = (
    "dr.",
    "phd",
    "md",
    "prof.",
    "professor",
    "research scientist",
    "staff engineer",
    "maintainer",
    "official answer",
)
AD_PATTERNS = (
    "advertisement",
    "affiliate",
    "promo",
    "promotion",
    "shop now",
    "sponsored",
    "subscribe",
)
PRIMARY_SOURCE_HINTS = (
    ".gov",
    "/docs",
    "/paper/",
    "/pdf",
    "acm.org",
    "arxiv.org",
    "doi.org",
    "github.com",
    "ieee.org",
    "ietf.org",
    "nature.com",
    "pubmed.ncbi.nlm.nih.gov",
    "rfc-editor.org",
    "science.org",
    "springer.com",
    "w3.org",
)


def _domain_matches(domain: str, candidates: set[str]) -> bool:
    return any(domain == candidate or domain.endswith(f".{candidate}") for candidate in candidates)


def _normalize_signal_text(value: str) -> str:
    cleaned = _clean_text(value or "").lower()
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", cleaned).strip()


def _ev_get(ev: Any, key: str, default: Any = None) -> Any:
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _ev_str(ev: Any, key: str) -> str:
    return str(_ev_get(ev, key, "") or "").strip()


def _ev_list(ev: Any, key: str) -> list[Any]:
    value = _ev_get(ev, key, [])
    return list(value) if isinstance(value, list) else []


def _record_tracking_updated_at(ev: Any) -> float | None:
    value = _ev_get(ev, "updated_at")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _best_text(ev: Any) -> str:
    for key in ("full_content", "raw_content", "content", "snippet"):
        value = _ev_str(ev, key)
        if value:
            return _clean_text(value)
    return ""


def _combined_text(ev: Any) -> str:
    parts = [
        _ev_str(ev, "title"),
        _ev_str(ev, "snippet"),
        _ev_str(ev, "raw_content"),
        _ev_str(ev, "full_content"),
        _ev_str(ev, "content"),
    ]
    return _clean_text(" ".join(part for part in parts if part))


def _root_domain_token(domain: str) -> str:
    labels = [
        label
        for label in extract_domain(f"https://{domain}").split(".")
        if label and label not in GENERIC_DOMAIN_LABELS
    ]
    if not labels:
        return ""
    for label in reversed(labels):
        if label not in COMMON_SUBDOMAIN_LABELS:
            return label
    return labels[-1]


def _title_subject(title: str) -> str:
    if not title:
        return ""
    return TITLE_CLAUSE_SPLIT_PATTERN.split(title, maxsplit=1)[0].strip()


def _append_unique(target: list[str], value: str) -> None:
    cleaned = str(value or "").strip()
    if cleaned and cleaned not in target:
        target.append(cleaned)


def _normalized_body_for_similarity(text: str) -> str:
    normalized = _normalize_signal_text(text)
    return " ".join(normalized.split())


def _repetition_ratio(text: str) -> float:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text or "") if block.strip()]
    if len(blocks) < 2:
        return 0.0
    normalized_blocks = [_normalized_body_for_similarity(block) for block in blocks]
    seen: set[str] = set()
    repeated = 0
    for block in normalized_blocks:
        if block in seen:
            repeated += 1
            continue
        seen.add(block)
    return repeated / max(len(normalized_blocks), 1)


def _round_component_score(value: float) -> float:
    return round(max(0.0, min(10.0, float(value))), 1)


def _quality_components(ev: Any) -> dict[str, Any]:
    if isinstance(ev, dict) and "source_type_v2" in ev and "word_count" in ev:
        return dict(ev)
    return _analyze_evidence_quality(ev).copy()


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = (parsed.path or "/").rstrip("/") or "/"
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMS and not key.lower().startswith("utm_")
    ]
    return urlunsplit((scheme, netloc, path, urlencode(filtered_query, doseq=True), ""))


def _mcp_evidence_key(server_id: str, tenant_id: str, url: str) -> str:
    return f"mcp::{tenant_id}::{server_id}::{url}"


def extract_domain(url: str) -> str:
    raw = (url or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if not parsed.netloc and (not parsed.scheme or " " in raw):
        return ""
    netloc = (parsed.netloc or "").lower().strip()
    if not netloc:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0].strip(".")
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _domain(url: str) -> str:
    return extract_domain(url)


def _clean_text(value: str) -> str:
    if not value:
        return ""
    value = HTML_SCRIPT_PATTERN.sub(" ", value)
    value = HTML_TAG_PATTERN.sub(" ", value)
    value = value.replace("\xa0", " ")
    return " ".join(value.split()).strip()


def _clean_structured_text(value: str) -> str:
    """Normalize extracted text/markdown while preserving headings, lists, and paragraph breaks."""
    if not value:
        return ""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines: list[str] = []
    blank_run = 0
    for raw_line in normalized.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not cleaned_line:
            blank_run += 1
            if blank_run == 1 and lines:
                lines.append("")
            continue
        blank_run = 0
        lines.append(cleaned_line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def _extract_structured_html_text(html: str) -> str:
    """Extract clean Markdown/structured text from HTML without flattening structure."""
    if not html:
        return ""

    # Strip boilerplate navigation/header/footer/aside/script/style blocks first.
    sanitized_html = HTML_BOILERPLATE_PATTERN.sub(" ", html)

    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                sanitized_html,
                include_formatting=True,
                include_tables=True,
                include_links=False,
            )
            cleaned = _clean_structured_text(extracted or "")
            if cleaned:
                return cleaned
        except Exception:  # pragma: no cover - optional dependency failure
            pass

    if markdownify is not None:
        try:
            converted = markdownify(sanitized_html, heading_style="ATX")
            cleaned = _clean_structured_text(converted or "")
            if cleaned:
                return cleaned
        except Exception:  # pragma: no cover - optional dependency failure
            pass

    with_breaks = HTML_BLOCK_BREAK_PATTERN.sub("\n", sanitized_html)
    without_tags = HTML_TAG_PATTERN.sub(" ", with_breaks)
    return _clean_structured_text(without_tags)


def _extract_pdf_text(
    raw_bytes: bytes, *, max_pages: int | None = None, check: Callable[[], object] | None = None,
) -> str:
    """Extract structured text from PDF bytes when PyMuPDF is available."""
    if pymupdf is None or not raw_bytes:
        return ""
    doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    try:
        pages: list[str] = []
        for index, page in enumerate(doc):
            if max_pages is not None and index >= max_pages:
                break
            if check is not None:
                check()
            page_text = _clean_structured_text(page.get_text("text") or "")
            if page_text:
                pages.append(page_text)
        return "\n\n".join(pages)
    finally:
        doc.close()


def _extract_title(html: str) -> str:
    if not html:
        return ""
    match = HTML_TITLE_PATTERN.search(html)
    if not match:
        return ""
    return _clean_text(match.group(1))


_ALLOWED_FETCH_SCHEMES = {"http", "https"}
_ALLOWED_FETCH_CONTENT_TYPES = {
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "text/markdown",
    "application/json",
}
_MAX_FETCH_BYTES = 5 * 1024 * 1024
_MAX_FETCH_REDIRECTS = 5
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


class _ValidatedFetchURL(str):
    def __new__(cls, url: str, ips: tuple[str, ...]) -> _ValidatedFetchURL:
        instance = super().__new__(cls, url)
        instance.ips = ips
        return instance


def _is_public_ip(ip_str: str) -> bool:
    """Return True only when ip_str is a globally routable public IP address."""
    try:
        address = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            return _is_public_ip(str(address.ipv4_mapped))
        if address.scope_id or address.sixtofour or address.teredo or address in ipaddress.ip_network("64:ff9b::/96"):
            return False
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        return False
    return address.is_global


def _validate_fetch_url(url: str) -> _ValidatedFetchURL:
    """Validate URL scheme and resolved IPs to block SSRF against local/private networks."""
    cleaned = (url or "").strip()
    if any(ord(char) < 33 or ord(char) == 127 or char == "\\" for char in cleaned):
        raise ValueError("URL contains invalid characters")
    parsed = urlsplit(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_FETCH_SCHEMES:
        raise ValueError(f"unsupported URL scheme: {scheme or '<empty>'}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are forbidden")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError("URL hostname is required")
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise ValueError(f"blocked private/local hostname: {hostname}")

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not _is_public_ip(hostname):
            raise ValueError(f"blocked non-public IP address: {hostname}")
        return _ValidatedFetchURL(cleaned, (hostname,))

    hostname = hostname.encode("idna").decode("ascii")
    try:
        addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"failed to resolve hostname {hostname}: {exc}") from exc

    resolved_ips = {
        str(info[4][0])
        for info in addr_infos
        if info and len(info) >= 5 and info[4]
    }
    if not resolved_ips:
        raise ValueError(f"no IP addresses resolved for hostname: {hostname}")
    for ip_str in resolved_ips:
        if not _is_public_ip(ip_str):
            raise ValueError(f"blocked non-public resolved IP {ip_str} for {hostname}")

    return _ValidatedFetchURL(cleaned, tuple(sorted(resolved_ips, key=lambda ip: (":" in ip, ip))))


class _PinnedFetchAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, ip: str, hostname: str, port: int, scheme: str) -> None:
        super().__init__()
        self.ip = ip
        self.hostname = hostname
        self.port = port
        self.scheme = scheme

    def get_connection(self, url: str, proxies: Any = None) -> Any:
        tls = {"server_hostname": self.hostname, "assert_hostname": self.hostname}
        return self.poolmanager.connection_from_host(
            self.ip, port=self.port, scheme=self.scheme, pool_kwargs=tls if self.scheme == "https" else None,
        )

    def get_connection_with_tls_context(
        self, request: Any, verify: Any, proxies: Any = None, cert: Any = None,
    ) -> Any:
        return self.get_connection(request.url)


@contextmanager
def _open_public_fetch(
    url: str, *, timeout: float, headers: dict[str, str] | None = None,
    budget: Any = None,
) -> Iterator[requests.Response]:
    validated = url if isinstance(url, _ValidatedFetchURL) else _validate_fetch_url(url)
    prepared = requests.Request(
        "GET", validated, headers={"Accept-Encoding": "identity", **(headers or {})},
    ).prepare()
    parsed = urlsplit(prepared.url)
    prepared.headers["Host"] = parsed.netloc
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for idx, ip in enumerate(validated.ips):
        adapter = _PinnedFetchAdapter(ip, parsed.hostname, port, parsed.scheme)
        try:
            with ExitStack() as stack:
                try:
                    dispatch = budget.attempt({"web_fetches": 1}) if budget is not None else nullcontext()
                    attempt = stack.enter_context(dispatch)
                    effective_timeout = min(timeout, attempt.timeout) if attempt is not None else timeout
                    deadline = time.monotonic() + effective_timeout
                    response = adapter.send(prepared, stream=True, timeout=effective_timeout)
                except requests.exceptions.ConnectionError:
                    if idx == len(validated.ips) - 1:
                        raise
                    continue
                with response:
                    if attempt is not None:
                        response._budget_deadline = deadline
                    yield response
                if attempt is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Web fetch exceeded its attempt timeout")
            return
        finally:
            adapter.close()


def _fetch_page_artifact(
    url: str,
    timeout_seconds: float,
    *,
    max_bytes: int | None = None,
    parser: Any | None = None,
    multimodal_enabled: bool = False,
    budget: Any = None,
) -> Any:
    """Safely fetch a web page or PDF into a normalized FetchedPageArtifact."""

    from services.multimodal_parser import FetchedPageArtifact, MultimodalEvidenceParser

    current_url = _validate_fetch_url(url)
    max_allowed = _MAX_FETCH_BYTES if max_bytes is None else max(1, int(max_bytes))
    active_parser = parser
    if active_parser is None and multimodal_enabled:
        active_parser = MultimodalEvidenceParser(enabled=True)

    for redirect_count in range(_MAX_FETCH_REDIRECTS + 1):
        with _open_public_fetch(
            current_url,
            timeout=timeout_seconds,
            **({"budget": budget} if budget is not None else {}),
        ) as response:
            if response.status_code in _REDIRECT_STATUS_CODES:
                location = str(response.headers.get("Location") or "").strip()
                if not location:
                    raise ValueError("redirect response missing Location header")
                if redirect_count >= _MAX_FETCH_REDIRECTS:
                    raise ValueError("too many redirects")
                current_url = _validate_fetch_url(urljoin(current_url, location))
                continue

            response.raise_for_status()
            content_type_header = str(response.headers.get("Content-Type") or "").strip()
            media_type = content_type_header.split(";", 1)[0].strip().lower()
            allowed_types = set(_ALLOWED_FETCH_CONTENT_TYPES)
            if pymupdf is not None or multimodal_enabled or (active_parser is not None and getattr(active_parser, "enabled", False)):
                allowed_types.add("application/pdf")
            if media_type and media_type not in allowed_types:
                raise ValueError(f"unsupported content type: {media_type}")

            limit = max_allowed
            if media_type == "application/pdf" and active_parser is not None and active_parser.enabled:
                pdf_limit = getattr(active_parser, "max_pdf_bytes", max_allowed)
                limit = pdf_limit if max_bytes is None else min(limit, pdf_limit)
            content_length_raw = str(response.headers.get("Content-Length") or "").strip()
            if content_length_raw.isascii() and content_length_raw.isdecimal() and (
                len(content_length_raw) > 20 or int(content_length_raw) > limit
            ):
                raise ValueError(f"fetched page exceeds max byte limit ({limit} bytes)")

            chunks: list[bytes] = []
            total_bytes = 0
            for chunk in response.iter_content(chunk_size=65536):
                if budget is not None and time.monotonic() >= response._budget_deadline:
                    raise TimeoutError("Web fetch exceeded its attempt timeout")
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > limit:
                    raise ValueError(f"fetched page exceeds max byte limit ({limit} bytes)")
                chunks.append(chunk)

            raw_payload = b"".join(chunks)
            if media_type == "application/pdf":
                if active_parser is not None and getattr(active_parser, "enabled", False):
                    return active_parser.parse_pdf_bytes(
                        raw_payload,
                        url=current_url,
                        title="",
                        text_fallback=_extract_pdf_text,
                    )
                return FetchedPageArtifact(
                    url=current_url,
                    title="",
                    content=_extract_pdf_text(raw_payload),
                    content_type="application/pdf",
                )

            encoding = response.encoding or "utf-8"
            html = raw_payload.decode(encoding, errors="replace")
            if media_type in {"text/plain", "text/markdown", "application/json"}:
                return FetchedPageArtifact(
                    url=current_url,
                    title="",
                    content=_clean_structured_text(html),
                    content_type=media_type or "text/plain",
                )

            title = _extract_title(html)
            base_text = _extract_structured_html_text(html)
            if active_parser is not None and getattr(active_parser, "enabled", False):
                return active_parser.parse_html(
                    html,
                    base_url=current_url,
                    title=title,
                    base_text=base_text,
                    fetch_chart_images=bool(getattr(active_parser, "vlm_enabled", False)),
                    image_timeout_seconds=timeout_seconds,
                )
            return FetchedPageArtifact(
                url=current_url,
                title=title,
                content=base_text,
                content_type=media_type or "text/html",
            )

    raise ValueError("too many redirects")


def _fetch_page_text(
    url: str,
    timeout_seconds: float,
    *,
    max_bytes: int = _MAX_FETCH_BYTES,
) -> tuple[str, str]:
    """Safely fetch a web page with SSRF validation, redirect checks, and size caps."""
    artifact = _fetch_page_artifact(url, timeout_seconds=timeout_seconds, max_bytes=max_bytes)
    return artifact.title, artifact.content


def _safe_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    match = DATE_PATTERN.search(text)
    if not match:
        return None

    try:
        parsed = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    return parsed


def _extract_published_at(result: dict[str, Any]) -> str | None:
    for key in (
        "published_at",
        "published_date",
        "publicationDate",
        "publication_date",
        "date",
        "datetime",
        "time",
    ):
        parsed = _parse_datetime(result.get(key))
        if parsed is not None:
            return parsed.date().isoformat()

        raw_value = str(result.get(key) or "").strip()
        if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", raw_value):
            return raw_value

    year = result.get("year")
    try:
        if year is not None and str(year).strip():
            normalized_year = int(year)
            if 1900 <= normalized_year <= 2100:
                return f"{normalized_year:04d}-01-01"
    except (TypeError, ValueError):
        pass

    combined_text = " ".join(
        str(result.get(field) or "").strip()
        for field in ("title", "content", "raw_content")
        if str(result.get(field) or "").strip()
    )
    parsed = _parse_datetime(combined_text)
    return parsed.date().isoformat() if parsed is not None else None


def _extract_source_updated_at(result: dict[str, Any]) -> str | None:
    for key in (
        "source_updated_at",
        "updated_at",
        "updatedAt",
        "last_modified",
        "lastModified",
        "modified_at",
    ):
        parsed = _parse_datetime(result.get(key))
        if parsed is not None:
            return parsed.date().isoformat()

        raw_value = str(result.get(key) or "").strip()
        if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", raw_value):
            return raw_value
    return None


def _classify_source_type(url: str, title: str) -> str:
    domain = _domain(url)
    lowered_title = (title or "").strip().lower()

    if domain.endswith(".gov") or ".gov." in domain:
        return "government"
    if domain.endswith(".edu") or ".edu." in domain:
        return "education"
    if "arxiv.org" in domain or "semanticscholar.org" in domain:
        return "paper"
    if "github.com" in domain:
        return "repository"
    if "docs." in domain or "/docs" in url.lower():
        return "documentation"
    if any(host in domain for host in ("reddit.com", "news.ycombinator.com", "zhihu.com", "weibo.com")):
        return "forum"
    if any(host in domain for host in ("x.com", "twitter.com")):
        return "social"
    if any(host in domain for host in ("wikipedia.org", "baike.baidu.com")):
        return "reference"
    if any(token in domain for token in ("news", "cnn.com", "nytimes.com", "theverge.com", "techcrunch.com")):
        return "news"
    if lowered_title.startswith("official") or "官方" in title:
        return "official"
    return "web"


def is_official_owner_match(ev: Any) -> bool:
    domain = extract_domain(_ev_str(ev, "url"))
    if not domain:
        return False
    if domain.endswith(".gov") or ".gov." in domain or domain.endswith(".edu") or ".edu." in domain:
        return True

    parsed = urlsplit(_ev_str(ev, "url"))
    path = parsed.path.lower()
    labels = [label for label in domain.split(".") if label]
    if labels and labels[0] in COMMON_SUBDOMAIN_LABELS:
        return True
    if any(f"/{token}" in path for token in OFFICIAL_PATH_TOKENS):
        return True

    root_token = _root_domain_token(domain)
    if not root_token:
        return False

    title_subject = _normalize_signal_text(_title_subject(_ev_str(ev, "title")))
    site_tokens = _normalize_signal_text(
        " ".join(
            value
            for value in (
                _ev_str(ev, "site_name"),
                _ev_str(ev, "publisher"),
                _ev_str(ev, "organization"),
                _ev_str(ev, "org"),
            )
            if value
        )
    )
    if root_token in title_subject or root_token in site_tokens:
        return True

    combined = _normalize_signal_text(_combined_text(ev))
    if ("official" in combined or "官方" in combined or "verified" in combined) and root_token in combined:
        return True
    return False


def is_navigation_or_listing_page(ev: Any) -> bool:
    title = _normalize_signal_text(_ev_str(ev, "title"))
    text = _best_text(ev)
    word_count = estimate_word_count(text)
    try:
        parsed = urlsplit(_ev_str(ev, "url"))
    except ValueError:
        parsed = urlsplit("")
    path = parsed.path.lower()
    query = parsed.query.lower()
    if any(token in title for token in NAVIGATION_TOKENS):
        return True
    if any(f"/{token}" in path for token in NAVIGATION_TOKENS):
        return True
    if any(f"{token}=" in query for token in ("page", "q", "query", "search")):
        return True
    link_count = len(set(URL_PATTERN.findall(text)))
    line_count = len([line for line in text.splitlines() if line.strip()])
    short_lines = sum(1 for line in text.splitlines() if 0 < len(line.strip()) <= 48)
    if word_count < 220 and link_count >= 8 and short_lines >= max(6, line_count // 2):
        return True
    return False


def has_author_or_org_attribution(ev: Any) -> bool:
    for key in ("author", "authors", "byline", "publisher", "site_name", "org", "organization"):
        value = _ev_get(ev, key)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
        if str(value or "").strip():
            return True
    combined = _normalize_signal_text(_combined_text(ev))
    return any(pattern in combined for pattern in AUTHOR_ATTRIBUTION_PATTERNS)


def estimate_word_count(text: str) -> int:
    cleaned = _clean_text(text or "")
    if not cleaned:
        return 0
    english_count = len(WORD_PATTERN.findall(cleaned))
    cjk_count = len(CJK_PATTERN.findall(cleaned))
    return english_count + int(math.ceil(cjk_count / 2.0))


def estimate_reference_count(ev: Any) -> int:
    explicit_lists = []
    for key in ("references", "citations"):
        value = _ev_get(ev, key)
        if isinstance(value, list):
            explicit_lists.extend(str(item).strip() for item in value if str(item).strip())
    if explicit_lists:
        return len(dict.fromkeys(explicit_lists))

    text = _combined_text(ev)
    refs: set[str] = set()
    refs.update(match.rstrip(".,)") for match in URL_PATTERN.findall(text))
    refs.update(match.lower() for match in DOI_PATTERN.findall(text))
    refs.update(match.lower() for match in ARXIV_PATTERN.findall(text))
    refs.update(match.group(0) for match in NUMBERED_REFERENCE_PATTERN.finditer(text))
    if re.search(r"\b(?:references|reference|bibliography|参考文献)\b", text, re.IGNORECASE):
        refs.add("references_section")
    return len(refs)


def has_clear_headings_and_sections(ev: Any) -> bool:
    text = ""
    for key in ("full_content", "raw_content", "content", "snippet"):
        value = _ev_str(ev, key)
        if value:
            text = HTML_SCRIPT_PATTERN.sub(" ", value)
            text = HTML_TAG_PATTERN.sub(" ", text).replace("\xa0", " ")
            break
    if not text:
        return False
    heading_matches = HEADING_PATTERN.findall(text)
    colon_headings = re.findall(r"(?m)^[A-Z][A-Za-z0-9 /&-]{2,40}:\s*$", text)
    return (len(heading_matches) + len(colon_headings)) >= 2


def estimate_extraction_quality(ev: Any) -> str:
    text = _best_text(ev)
    word_count = estimate_word_count(text)
    if word_count == 0:
        return "poor"
    html_residue = len(re.findall(r"</?[a-z][^>]*>", text, re.IGNORECASE))
    boilerplate_hits = sum(
        text.lower().count(token)
        for token in ("cookie", "privacy policy", "terms of service", "subscribe", "sign in", "menu")
    )
    repetition = _repetition_ratio(text)
    if html_residue >= 8 or boilerplate_hits >= 4 or repetition >= 0.35:
        return "poor"
    if word_count < 120:
        if word_count >= 20 and (
            _ev_str(ev, "published_at")
            or _classify_source_type(_ev_str(ev, "url"), _ev_str(ev, "title")) in {"documentation", "government", "education", "paper"}
        ):
            return "partial"
        return "poor"
    if word_count < 500 or html_residue >= 2 or boilerplate_hits >= 2 or repetition >= 0.2:
        return "partial"
    return "good"


def is_duplicate_or_near_duplicate(ev: Any) -> bool:
    for key in ("duplicate", "is_duplicate", "near_duplicate"):
        if bool(_ev_get(ev, key)):
            return True
    full_text = _normalized_body_for_similarity(_ev_str(ev, "full_content"))
    raw_text = _normalized_body_for_similarity(_ev_str(ev, "raw_content"))
    snippet = _normalized_body_for_similarity(_ev_str(ev, "snippet"))
    if full_text and snippet and full_text == snippet:
        return True
    if raw_text and snippet and raw_text == snippet:
        return True
    if _repetition_ratio(_best_text(ev)) >= 0.45:
        return True
    return False


def is_aggregator_or_rehosted_copy(ev: Any) -> bool:
    domain = extract_domain(_ev_str(ev, "url"))
    if _domain_matches(domain, AGGREGATOR_DOMAINS):
        return True
    text = _normalize_signal_text(_combined_text(ev))
    return any(pattern in text for pattern in ("originally published", "转载", "syndicated", "via "))


def is_clickbait_or_low_information_page(ev: Any) -> bool:
    title = _normalize_signal_text(_ev_str(ev, "title"))
    text = _normalize_signal_text(_best_text(ev))
    word_count = estimate_word_count(_best_text(ev))
    if any(pattern in title for pattern in CLICKBAIT_PATTERNS):
        return True
    if word_count < 120 and any(pattern in text for pattern in TEASER_PATTERNS):
        return True
    if word_count < 80 and title.endswith("?"):
        return True
    return False


def ad_ratio_high(ev: Any) -> bool:
    text = _normalize_signal_text(_combined_text(ev))
    word_count = max(estimate_word_count(text), 1)
    ad_hits = sum(text.count(pattern) for pattern in AD_PATTERNS)
    return ad_hits >= 3 or (ad_hits / word_count) > 0.03


def has_expert_identity_signal(ev: Any) -> bool:
    author_blob = _normalize_signal_text(
        " ".join(
            [
                _ev_str(ev, "author"),
                " ".join(str(item).strip() for item in _ev_list(ev, "authors")),
                _ev_str(ev, "byline"),
                _ev_str(ev, "title"),
                _best_text(ev),
            ]
        )
    )
    return any(pattern in author_blob for pattern in EXPERT_PATTERNS)


def links_to_primary_sources(ev: Any) -> bool:
    for item in _ev_list(ev, "references") + _ev_list(ev, "citations"):
        text = str(item).strip().lower()
        if any(hint in text for hint in PRIMARY_SOURCE_HINTS):
            return True
    combined = _combined_text(ev).lower()
    return any(hint in combined for hint in PRIMARY_SOURCE_HINTS)


def is_social_official(ev: Any) -> bool:
    domain = extract_domain(_ev_str(ev, "url"))
    if not _domain_matches(domain, SOCIAL_DOMAINS):
        return False
    root_token = _root_domain_token(_ev_str(ev, "site_name") or _ev_str(ev, "organization") or extract_domain(_ev_str(ev, "url")))
    title_blob = _normalize_signal_text(
        " ".join(
            value
            for value in (
                _ev_str(ev, "title"),
                _ev_str(ev, "site_name"),
                _ev_str(ev, "publisher"),
                _ev_str(ev, "organization"),
            )
            if value
        )
    )
    path_match = SOCIAL_HANDLE_PATTERN.match(urlsplit(_ev_str(ev, "url")).path)
    handle = _normalize_signal_text(path_match.group("handle")) if path_match else ""
    if root_token and (root_token in title_blob or root_token == handle):
        return True
    return "official" in title_blob or "verified organization" in title_blob


def detect_source_type(ev: Any) -> str:
    source_origin = _ev_str(ev, "source_origin").strip().lower()
    coarse_type = _ev_str(ev, "source_type").strip()
    if source_origin == "mcp":
        if coarse_type.startswith("mcp_"):
            return coarse_type
        return "mcp_tool"
    if source_origin == "sandbox":
        if coarse_type.startswith("sandbox_") or coarse_type == "computed_sandbox":
            return coarse_type
        return "computed_sandbox"

    domain = extract_domain(_ev_str(ev, "url"))
    url = _ev_str(ev, "url").lower()
    if is_aggregator_or_rehosted_copy(ev):
        return "content_farm_or_aggregator"
    if domain.endswith(".gov") or ".gov." in domain:
        return "government"
    if domain.endswith(".edu") or ".edu." in domain:
        return "education"
    if _domain_matches(domain, STANDARDS_DOMAINS):
        return "standards_spec"
    if _domain_matches(domain, PREPRINT_DOMAINS):
        return "preprint_paper"
    if _domain_matches(domain, PAPER_DOMAINS) or "/doi/" in url or url.endswith(".pdf"):
        return "peer_reviewed_paper"
    if coarse_type == "documentation" or is_official_owner_match(ev) and any(
        token in url for token in OFFICIAL_PATH_TOKENS
    ):
        return "official_documentation"
    if is_official_owner_match(ev) and any(token in url for token in PRODUCT_PATH_TOKENS):
        return "official_product"
    if is_official_owner_match(ev) and coarse_type == "official":
        return "official_org"
    if _domain_matches(domain, {"github.com", "gitlab.com"}):
        return "repository_official" if is_official_owner_match(ev) else "repository_unofficial"
    if _domain_matches(domain, REFERENCE_DOMAINS):
        return "reference_curated"
    if _domain_matches(domain, NEWS_PRIMARY_DOMAINS):
        return "news_primary" if not is_navigation_or_listing_page(ev) else "news_secondary"
    if _domain_matches(domain, NEWS_SECONDARY_DOMAINS) or ("news" in domain and not is_official_owner_match(ev)):
        return "news_secondary"
    if is_official_owner_match(ev) and any(token in url for token in BLOG_PATH_TOKENS):
        return "company_blog"
    if "blog" in domain or any(token in url for token in BLOG_PATH_TOKENS):
        return "technical_blog"
    if _domain_matches(domain, FORUM_DOMAINS):
        return "forum_expert" if has_expert_identity_signal(ev) else "forum_general"
    if _domain_matches(domain, SOCIAL_DOMAINS):
        return "social_official" if is_social_official(ev) else "social_general"
    if is_official_owner_match(ev):
        return "official_org"
    return "web_general"


def _analyze_evidence_quality(ev: Any, now: datetime | None = None) -> dict[str, Any]:
    best_text = _best_text(ev)
    full_text = _combined_text(ev)
    url = _ev_str(ev, "url")
    domain = extract_domain(url)
    provider_count = max(1, min(int(_ev_get(ev, "provider_count", 1) or 1), 3))
    source_updated_at = _ev_str(ev, "source_updated_at")
    if not source_updated_at:
        source_updated_at = _extract_source_updated_at(ev if isinstance(ev, dict) else {})
    published_at = _ev_str(ev, "published_at")
    if not published_at and isinstance(ev, dict):
        published_at = _extract_published_at(ev) or ""
    if not source_updated_at and published_at:
        source_updated_at = ""

    extracted_word_count = estimate_word_count(best_text)
    reference_count = estimate_reference_count(ev)
    extraction_quality = estimate_extraction_quality(ev)
    source_type_v2 = detect_source_type(ev)
    official_match = is_official_owner_match(ev)
    navigation_page = is_navigation_or_listing_page(ev)
    expert_signal = has_expert_identity_signal(ev)
    social_official = is_social_official(ev)
    author_attribution = has_author_or_org_attribution(ev)
    clear_sections = has_clear_headings_and_sections(ev)
    duplicate = is_duplicate_or_near_duplicate(ev)
    aggregator = is_aggregator_or_rehosted_copy(ev)
    clickbait = is_clickbait_or_low_information_page(ev)
    ad_heavy = ad_ratio_high(ev)
    primary_links = links_to_primary_sources(ev)
    age_days, freshness_label = compute_freshness_label(
        {
            "published_at": published_at or None,
            "source_updated_at": source_updated_at or None,
        },
        now=now,
    )

    return {
        "url": url,
        "domain": domain,
        "provider_count": provider_count,
        "published_at": published_at or None,
        "source_updated_at": source_updated_at or None,
        "best_text": best_text,
        "full_text": full_text,
        "word_count": extracted_word_count,
        "reference_count": reference_count,
        "extraction_quality": extraction_quality,
        "source_type_v2": source_type_v2,
        "official_owner_match": official_match,
        "navigation_page": navigation_page,
        "author_attribution": author_attribution,
        "clear_sections": clear_sections,
        "duplicate_or_near_duplicate": duplicate,
        "aggregator_or_rehosted": aggregator,
        "clickbait_or_low_information": clickbait,
        "ad_ratio_high": ad_heavy,
        "links_to_primary_sources": primary_links,
        "has_expert_identity_signal": expert_signal,
        "is_social_official": social_official,
        "freshness_days": age_days,
        "freshness_label": freshness_label,
        "source_origin": _ev_str(ev, "source_origin") or "web",
        "has_computation_artifact": bool(_ev_get(ev, "computation_artifact", None)),
    }


def compute_source_reliability_score(ev: Any) -> float:
    components = _quality_components(ev)
    score = 3.0
    score += SOURCE_TYPE_BONUS.get(components["source_type_v2"], 0.0)
    score += PROVIDER_BONUS[components["provider_count"]]
    if components["official_owner_match"]:
        score += 1.0
    if components["navigation_page"]:
        score -= 1.5
    if components["source_type_v2"].startswith("forum_") and components["has_expert_identity_signal"]:
        score += 0.8
    if components["source_type_v2"].startswith("social_") and components["is_social_official"]:
        score += 0.8
    return _round_component_score(score)


def compute_content_quality_score(ev: Any) -> float:
    components = _quality_components(ev)
    if components.get("source_origin") == "sandbox" and bool(components.get("has_computation_artifact")):
        return 9.0
    score = 3.0
    if components["published_at"]:
        score += 0.8
    if components["source_updated_at"] and components["source_updated_at"] != components["published_at"]:
        score += 0.4
    if components["author_attribution"]:
        score += 0.8

    word_count = components["word_count"]
    if word_count >= 1200:
        score += 1.2
    elif word_count >= 500:
        score += 0.8
    elif word_count >= 200:
        score += 0.3
    else:
        score -= 1.2

    reference_count = components["reference_count"]
    if reference_count >= 5:
        score += 1.0
    elif reference_count >= 1:
        score += 0.5

    if components["clear_sections"]:
        score += 0.5

    extraction_quality = components["extraction_quality"]
    if extraction_quality == "good":
        score += 0.8
    elif extraction_quality == "partial":
        score -= 0.5
    else:
        score -= 1.5

    if components["duplicate_or_near_duplicate"]:
        score -= 1.0
    if components["aggregator_or_rehosted"]:
        score -= 1.2
    if components["clickbait_or_low_information"]:
        score -= 1.5
    if components["ad_ratio_high"]:
        score -= 0.8
    if components["links_to_primary_sources"]:
        score += 0.8
    return _round_component_score(score)


def compute_freshness_label(ev: Any, now: datetime | None = None) -> tuple[int | None, str]:
    published_at = _ev_str(ev, "published_at")
    source_updated_at = _ev_str(ev, "source_updated_at")
    candidate = published_at or source_updated_at
    parsed = _parse_datetime(candidate)
    if parsed is None:
        return None, "unknown"
    current_time = now or _safe_now()
    age_days = max(int((current_time - parsed).total_seconds() // 86400), 0)
    if age_days <= 30:
        return age_days, "fresh"
    if age_days <= 180:
        return age_days, "recent"
    return age_days, "stale"


def compute_quality_score_and_label(
    source_reliability_score: float,
    content_quality_score: float,
    ev: Any,
) -> tuple[int, str]:
    components = _quality_components(ev)
    quality_score = int(round(_round_component_score(source_reliability_score) * 5.5 + _round_component_score(content_quality_score) * 4.5))
    quality_score = max(0, min(100, quality_score))
    if quality_score >= 75:
        label = "high"
    elif quality_score >= 45:
        label = "medium"
    else:
        label = "low"
    is_valid_sandbox = (
        components.get("source_origin") == "sandbox"
        and bool(components.get("has_computation_artifact"))
    )
    if label == "high" and not is_valid_sandbox and (
        components["extraction_quality"] == "poor"
        or components["navigation_page"]
        or components["word_count"] < 120
    ):
        label = "medium"
    return quality_score, label


def score_evidence_quality(ev: Any, now: datetime | None = None) -> dict[str, Any]:
    components = _analyze_evidence_quality(ev, now=now)
    reasons: list[str] = []
    flags: list[str] = []

    source_reliability_score = compute_source_reliability_score(components)
    content_quality_score = compute_content_quality_score(components)
    quality_score, quality_label = compute_quality_score_and_label(
        source_reliability_score,
        content_quality_score,
        components,
    )

    _append_unique(reasons, f"source_type={components['source_type_v2']}")
    _append_unique(reasons, f"provider_count={components['provider_count']}")
    if components["official_owner_match"]:
        _append_unique(flags, "official_owner_match")
        _append_unique(reasons, "official owner match")
    if components["navigation_page"]:
        _append_unique(flags, "navigation_page")
        _append_unique(reasons, "navigation or listing page")
    if components["source_type_v2"].startswith("forum_") and components["has_expert_identity_signal"]:
        _append_unique(flags, "forum_expert")
        _append_unique(reasons, "forum has expert identity")
    if components["source_type_v2"].startswith("social_") and components["is_social_official"]:
        _append_unique(flags, "social_official")
        _append_unique(reasons, "social account looks official")
    if components["published_at"]:
        _append_unique(reasons, "published date available")
    if components["source_updated_at"] and components["source_updated_at"] != components["published_at"]:
        _append_unique(flags, "updated_after_publish")
        _append_unique(reasons, "updated date differs from publish date")
    if components["author_attribution"]:
        _append_unique(flags, "author_attribution")
        _append_unique(reasons, "author or organization attribution found")
    if components["word_count"] >= 1200:
        _append_unique(flags, "long_form_content")
        _append_unique(reasons, "long form content")
    elif components["word_count"] < 200:
        _append_unique(reasons, "content is short")
    if components["reference_count"] >= 1:
        _append_unique(flags, "has_references")
        _append_unique(reasons, f"references={components['reference_count']}")
    if components["clear_sections"]:
        _append_unique(flags, "clear_sections")
        _append_unique(reasons, "clear section structure")
    _append_unique(flags, f"extraction_{components['extraction_quality']}")
    _append_unique(reasons, f"extraction={components['extraction_quality']}")
    if components["duplicate_or_near_duplicate"]:
        _append_unique(flags, "duplicate_or_near_duplicate")
        _append_unique(reasons, "duplicate or repeated content")
    if components["aggregator_or_rehosted"]:
        _append_unique(flags, "aggregator_or_rehosted")
        _append_unique(reasons, "aggregator or rehosted copy")
    if components["clickbait_or_low_information"]:
        _append_unique(flags, "clickbait_or_low_information")
        _append_unique(reasons, "clickbait or low information page")
    if components["ad_ratio_high"]:
        _append_unique(flags, "ad_ratio_high")
        _append_unique(reasons, "ad ratio appears high")
    if components["links_to_primary_sources"]:
        _append_unique(flags, "links_to_primary_sources")
        _append_unique(reasons, "links to primary sources")
    if quality_label == "medium" and quality_score >= 75 and (
        components["extraction_quality"] == "poor"
        or components["navigation_page"]
        or components["word_count"] < 120
    ):
        _append_unique(flags, "high_label_downgraded")
        _append_unique(reasons, "high score downgraded by safety gate")

    return {
        "source_reliability_score": source_reliability_score,
        "content_quality_score": content_quality_score,
        "quality_score": quality_score,
        "quality_label": quality_label,
        "freshness_days": components["freshness_days"],
        "freshness_label": components["freshness_label"],
        "quality_reasons": reasons,
        "quality_flags": flags,
        "source_updated_at": components["source_updated_at"],
        "published_at": components["published_at"],
    }


def _score_quality(
    *,
    source_type: str,
    provider_count: int,
    published_at: str | None,
) -> tuple[int, str]:
    payload = {
        "source_type": source_type,
        "provider_count": provider_count,
        "published_at": published_at,
    }
    scored = score_evidence_quality(payload)
    return int(scored["quality_score"]), str(scored["quality_label"])


def _freshness_metadata(
    published_at: str | None,
    *,
    freshness_reference_days: int = 365,
) -> tuple[int | None, str]:
    del freshness_reference_days
    return compute_freshness_label({"published_at": published_at})


def extract_citation_ids(text: str) -> list[str]:
    """Return citation ids in first-seen order."""

    citations: list[str] = []
    seen: set[str] = set()
    for citation in CITATION_PATTERN.findall(text or ""):
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
    return citations


@dataclass
class EvidenceRecord:
    """Normalized source evidence stored per task."""

    source_id: str
    task_id: int
    query: str
    title: str
    url: str
    snippet: str
    raw_content: str = ""
    full_content: str = ""
    backend: str = ""
    backend_sources: list[str] | None = None
    provider_count: int = 1
    domain: str = ""
    source_type: str = "web"
    source_reliability_score: float = 0.0
    content_quality_score: float = 0.0
    quality_score: int = 0
    quality_label: str = "medium"
    published_at: str | None = None
    source_updated_at: str | None = None
    freshness_days: int | None = None
    freshness_label: str = "unknown"
    quality_reasons: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    archive_doc_key: str | None = None
    archive_version_id: str | None = None
    archive_has_full_content: bool = False
    source_origin: str = "web"
    mcp_server_id: str | None = None
    mcp_tool_name: str | None = None
    tenant_id: str = "default"
    provenance_meta: dict[str, Any] = field(default_factory=dict)
    _web_managed: bool = field(default=False, init=False, repr=False)
    _web_verified: bool = field(default=False, init=False, repr=False)
    _web_published: bool = field(default=False, init=False, repr=False)
    modality: str = "text"
    page_no: int | None = None
    bbox: list[float] | None = None
    anchor_id: str | None = None
    multimodal_blocks: list[dict[str, Any]] = field(default_factory=list)
    derived_from_source_ids: list[str] = field(default_factory=list)
    computation_artifact: dict[str, Any] | None = None
    structured_payload: dict[str, Any] | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(
        self,
        *,
        include_full_content: bool = False,
        excerpt_limit: int = 1200,
    ) -> dict[str, Any]:
        payload = {
            "source_id": self.source_id,
            "task_id": self.task_id,
            "query": self.query,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "snippet": truncate_text(self.snippet, excerpt_limit),
            "backend": self.backend,
            "backend_sources": list(self.backend_sources or []),
            "provider_count": self.provider_count,
            "source_type": self.source_type,
            "source_origin": self.source_origin,
            "modality": self.modality,
            "source_reliability_score": self.source_reliability_score,
            "content_quality_score": self.content_quality_score,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "published_at": self.published_at,
            "source_updated_at": self.source_updated_at,
            "freshness_days": self.freshness_days,
            "freshness_label": self.freshness_label,
            "quality_reasons": list(self.quality_reasons),
            "quality_flags": list(self.quality_flags),
            "archive_doc_key": self.archive_doc_key,
            "archive_version_id": self.archive_version_id,
            "archive_has_full_content": bool(self.archive_has_full_content),
            "has_full_content": bool(self.full_content),
        }
        if self.mcp_server_id:
            payload["mcp_server_id"] = self.mcp_server_id
        if self.mcp_tool_name:
            payload["mcp_tool_name"] = self.mcp_tool_name
        if self.tenant_id and self.tenant_id != "default":
            payload["tenant_id"] = self.tenant_id
        if self.provenance_meta:
            payload["provenance_meta"] = deepcopy(self.provenance_meta)
        if self.page_no is not None:
            payload["page_no"] = self.page_no
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        if self.anchor_id:
            payload["anchor_id"] = self.anchor_id
        if self.multimodal_blocks:
            payload["multimodal_blocks"] = deepcopy(self.multimodal_blocks)
        if self.derived_from_source_ids:
            payload["derived_from_source_ids"] = list(self.derived_from_source_ids)
        if self.computation_artifact:
            payload["computation_artifact"] = dict(self.computation_artifact)
        if self.structured_payload:
            payload["structured_payload"] = dict(self.structured_payload)
        if include_full_content and self.full_content:
            payload["full_content"] = (
                self.full_content if self._web_verified else truncate_text(self.full_content, excerpt_limit * 3)
            )
        return payload


def _apply_quality_metadata(record: EvidenceRecord) -> None:
    scored = score_evidence_quality(record)
    record.domain = record.domain or extract_domain(record.url)
    record.source_reliability_score = float(scored["source_reliability_score"])
    record.content_quality_score = float(scored["content_quality_score"])
    record.quality_score = int(scored["quality_score"])
    record.quality_label = str(scored["quality_label"])
    record.published_at = scored["published_at"]
    record.source_updated_at = scored["source_updated_at"]
    record.freshness_days = scored["freshness_days"]
    record.freshness_label = str(scored["freshness_label"])
    record.quality_reasons = list(scored["quality_reasons"])
    record.quality_flags = list(scored["quality_flags"])


def _is_compatible_web_artifact(record: EvidenceRecord) -> bool:
    meta = record.provenance_meta.get("web_extraction")
    if not isinstance(meta, dict) or not (record.full_content.strip() or record.multimodal_blocks):
        return False
    if (meta.get("tier"), meta.get("adapter")) not in (
        ("L1", "static"), ("L2", "reader_http"), ("L2", "playwright"),
    ):
        return False
    versions = (
        ("sanitizer_version", WEB_SANITIZER_VERSION),
        ("quality_version", QUALITY_VERSION),
        ("policy_version", POLICY_VERSION),
    )
    if any(type(meta.get(key)) is not int or meta[key] != version for key, version in versions):
        return False
    artifact_version = meta.get("artifact_version")
    if type(artifact_version) is not int or artifact_version <= 0:
        return False
    try:
        normalize_web_url(record.url)
    except GatewayError:
        return False
    return True


_WEB_PUBLIC_TIERS = {"L0": {"snippet"}, "L1": {"static"}, "L2": {"reader_http", "playwright"}}
_WEB_PUBLIC_FLAGS = frozenset({
    "js_shell", "low_snr", "has_table", "structure_truncated", "visibility_uncertain",
    "snippet_only", "link_only",
})
_WEB_CREDENTIAL_PARAMETER = re.compile(
    r"(?:^|[-_])(?:token|key|secret|password|auth|authorization|credential|signature|session)(?:$|[-_])"
    r"|^(?:sig|code|accesskey|apikey|accesstoken|sessionid|x-amz-.+|x-goog-.+)$", re.I,
)


def public_web_url(url: str) -> str:
    try:
        safe = normalize_web_url(url)
    except GatewayError:
        return ""
    parts = urlsplit(safe)
    # Keep query segment encoding and ordering for URLs without credential fields.
    segments = []
    for segment in parts.query.split("&"):
        pairs = parse_qsl(segment, keep_blank_values=True)
        if all(not _WEB_CREDENTIAL_PARAMETER.search(key) for key, _ in pairs):
            segments.append(segment)
    query = "&".join(segments)
    return safe if query == parts.query else parts._replace(query=query).geturl()


def public_web_evidence_projection(value: Any, *, web_enabled: bool = False) -> Any:
    """Project nested evidence the same way in synchronous, saved, and streamed responses."""
    if isinstance(value, list):
        return [public_web_evidence_projection(item, web_enabled=web_enabled) for item in value]
    if not isinstance(value, dict):
        return value
    provenance = value.get("provenance_meta")
    web = provenance.get("web_extraction") if isinstance(provenance, dict) else None
    v4 = value.get("v4")
    mode = web_enabled or (isinstance(v4, dict) and isinstance(v4.get("web_extraction"), dict))
    if web is None and not (
        mode and "source_id" in value and value.get("source_origin", "web") == "web"
    ):
        return {key: public_web_evidence_projection(item, web_enabled=mode) for key, item in value.items()}
    projected = {
        key: public_web_evidence_projection(item, web_enabled=mode)
        for key, item in value.items()
        if key not in {
            "provenance_meta", "raw_content", "raw_html", "final_url", "redirect_chain",
            "resolved_ip", "connected_ip", "credentials", "headers",
        }
    }
    if "url" in projected:
        projected["url"] = public_web_url(str(projected["url"] or ""))
    if not isinstance(web, dict):
        projected.pop("full_content", None)
        projected.pop("multimodal_blocks", None)
        if "has_full_content" in projected:
            projected["has_full_content"] = False
        for field_name in ("title", "snippet"):
            if field_name in projected:
                projected[field_name] = sanitize_web_snippet(str(projected[field_name] or ""))
        return projected
    bounded: dict[str, Any] = {}
    tier, adapter = web.get("tier"), web.get("adapter")
    if isinstance(tier, str) and isinstance(adapter, str) and adapter in _WEB_PUBLIC_TIERS.get(tier, ()):
        bounded.update(tier=tier, adapter=adapter)
    for key in ("sanitizer_version", "quality_version", "policy_version", "artifact_version", "dropped_nodes"):
        count = web.get(key)
        if type(count) is int and 0 <= count <= 2**31 - 1:
            bounded[key] = count
    flags = web.get("quality_flags")
    if isinstance(flags, list):
        bounded["quality_flags"] = list(dict.fromkeys(
            flag for flag in flags[:32] if isinstance(flag, str) and flag in _WEB_PUBLIC_FLAGS
        ))
    reason = web.get("degraded_reason")
    if "degraded_reason" in web and (reason is None or isinstance(reason, str) and reason in WEB_EXTRACTION_REASONS):
        bounded["degraded_reason"] = reason
    anchors = web.get("text_anchors")
    if isinstance(anchors, list):
        bounded["text_anchors"] = [
            {key: anchor[key] for key in ("anchor_id", "start", "end")}
            for anchor in anchors[:4096]
            if isinstance(anchor, dict)
            and isinstance(anchor.get("anchor_id"), str)
            and re.fullmatch(r"(?:html-p|text-p|pdf-p)[0-9]{1,6}", anchor["anchor_id"])
            and type(anchor.get("start")) is int and type(anchor.get("end")) is int
            and 0 <= anchor["start"] < anchor["end"] <= 50_000
        ]
    projected["provenance_meta"] = {"web_extraction": bounded}
    return projected


def _next_source_sequence(source_ids: list[str], prefix: str) -> int:
    return 1 + max(
        (int(suffix) for sid in source_ids if sid.startswith(prefix)
         and (suffix := sid[len(prefix):]).isascii() and suffix.isdecimal() and len(suffix) <= 12),
        default=0,
    )


class EvidenceStore:
    """Thread-safe in-memory evidence store scoped to a single request."""

    def __init__(
        self,
        *,
        freshness_reference_days: int = 365,
        listeners: list[EvidenceStoreListener] | None = None,
        request_id_getter: Callable[[], str | None] | None = None,
        web_extraction_enabled: bool = False,
    ) -> None:
        self._lock = Lock()
        self._freshness_reference_days = max(1, int(freshness_reference_days or 365))
        self._records_by_id: dict[str, EvidenceRecord] = {}
        self._task_source_ids: dict[int, list[str]] = {}
        self._task_url_index: dict[int, dict[str, str]] = {}
        self._listeners = list(listeners or [])
        self._request_id_getter = request_id_getter
        self.web_extraction_enabled = web_extraction_enabled

    def is_eligible_for_facts(self, source_id: str) -> bool:
        with self._lock:
            record = self._records_by_id.get(source_id)
            return record is not None and (
                not self.web_extraction_enabled or record.source_origin != "web" or record._web_verified
            )

    def web_snippet_for_fetch(self, source_id: str, url: str) -> WebExtractionResult | None:
        with self._lock:
            record = self._records_by_id.get(source_id)
            if record is None or record.source_origin != "web" or _normalize_url(record.url) != _normalize_url(url):
                return None
            return normalize_snippet(url, record.title, record.snippet, projected=True)

    def publish_web_artifact(self, source_id: str) -> None:
        """Freeze the current artifact after publication."""
        with self._lock:
            if (record := self._records_by_id.get(source_id)) is not None and record._web_verified:
                record._web_published = True

    def _notify_listeners(
        self,
        *,
        task_id: int,
        records: list[EvidenceRecord],
        reason: str,
    ) -> None:
        if not records or not self._listeners:
            return
        request_id = self._request_id_getter() if self._request_id_getter else None
        for listener in self._listeners:
            try:
                listener.on_records_changed(
                    task_id=task_id,
                    records=deepcopy(records) if self.web_extraction_enabled else list(records),
                    reason=reason,
                    request_id=request_id,
                )
            except Exception:
                continue

    def record_search_results(
        self,
        *,
        task_id: int,
        query: str,
        search_payload: dict[str, Any] | None,
        backend: str,
    ) -> list[dict[str, Any]]:
        """Upsert normalized search results and return task-scoped evidence view."""

        if not search_payload:
            return self.list_task_evidence(task_id)

        results = list(search_payload.get("results") or [])
        changed_records: list[EvidenceRecord] = []
        with self._lock:
            task_ids = self._task_source_ids.setdefault(task_id, [])
            url_index = self._task_url_index.setdefault(task_id, {})
            next_index = _next_source_sequence(task_ids, f"T{task_id}-S")

            for result in results:
                if not isinstance(result, dict):
                    continue

                url = str(result.get("url") or "").strip()
                title = str(result.get("title") or url or f"Source {next_index}").strip()
                snippet = str(result.get("content") or "").strip()
                raw_content = str(result.get("raw_content") or "").strip()
                projection = None
                if self.web_extraction_enabled:
                    projection = projected_search_snippet(result)
                    if projection is None:
                        projection = normalize_snippet(url, str(result.get("title") or ""), snippet)
                    if projection is None:
                        continue
                    title = projection.title
                    snippet = projection.visible_text
                    raw_content = ""
                published_at = _extract_published_at(result)
                source_updated_at = _extract_source_updated_at(result)
                source_type = _classify_source_type(url, title)
                dedup_key = _normalize_url(url) or f"title::{title.casefold()}"
                existing_id = url_index.get(dedup_key)
                if existing_id:
                    record = self._records_by_id[existing_id]
                    if (not record.snippet if projection else len(snippet) > len(record.snippet)):
                        record.snippet = snippet
                    if len(raw_content) > len(record.raw_content):
                        record.raw_content = raw_content
                    record.backend = backend or record.backend
                    record.backend_sources = list(result.get("backend_sources") or record.backend_sources or [])
                    record.provider_count = int(result.get("provider_count") or record.provider_count or 1)
                    record.source_type = source_type or record.source_type
                    record.published_at = published_at or record.published_at
                    record.source_updated_at = source_updated_at or record.source_updated_at
                    _apply_quality_metadata(record)
                    record.updated_at = time.time()
                    changed_records.append(deepcopy(record) if projection else record)
                    continue

                source_id = f"T{task_id}-S{next_index}"
                next_index += 1
                record = EvidenceRecord(
                    source_id=source_id,
                    task_id=task_id,
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    raw_content=raw_content,
                    backend=backend,
                    backend_sources=list(result.get("backend_sources") or []),
                    provider_count=int(result.get("provider_count") or 1),
                    domain=_domain(url),
                    source_type=source_type,
                    published_at=published_at,
                    source_updated_at=source_updated_at,
                    created_at=time.time(),
                    updated_at=time.time(),
                )
                if projection is not None:
                    record.provenance_meta = {"web_extraction": projection.provenance()}
                    record._web_managed = True
                _apply_quality_metadata(record)
                self._records_by_id[source_id] = record
                task_ids.append(source_id)
                url_index[dedup_key] = source_id
                changed_records.append(deepcopy(record) if projection else record)

        self._notify_listeners(task_id=task_id, records=changed_records, reason="search")
        return self.list_task_evidence(task_id)

    def update_full_content(
        self,
        *,
        task_id: int,
        source_id: str | None,
        url: str,
        title: str,
        full_content: str,
        multimodal_blocks: list[dict[str, Any]] | None = None,
        web_result: WebExtractionResult | None = None,
        check_owner: Callable[[], object] | None = None,
    ) -> dict[str, Any]:
        """Attach fetched page content to an existing or new evidence record."""

        if self.web_extraction_enabled:
            if web_result is None:
                return {}
            try:
                if not is_produced_result(web_result, url):
                    return {}
            except GatewayError:
                return {}
            title = web_result.title
            full_content = web_result.visible_text if web_result.supports_facts else ""
            multimodal_blocks = list(web_result.multimodal_blocks)
        elif web_result is not None:
            return {}
        normalized_url = _normalize_url(url)
        changed_record: EvidenceRecord | None = None
        with self._lock:
            if check_owner is not None:
                check_owner()
            task_ids = self._task_source_ids.setdefault(task_id, [])
            url_index = self._task_url_index.setdefault(task_id, {})

            record: EvidenceRecord | None = None
            if source_id:
                record = self._records_by_id.get(source_id)
                if (
                    record is None or record.task_id != task_id or record.source_origin != "web"
                    or _normalize_url(record.url) != normalized_url
                ):
                    raise ValueError("source_id does not match the task's web evidence URL")
            if record is None and normalized_url:
                existing_id = url_index.get(normalized_url)
                if existing_id:
                    record = self._records_by_id.get(existing_id)

            if record is not None and record.source_origin != "web" and web_result is not None:
                return {}
            if record is not None and web_result is not None:
                previous_meta = record.provenance_meta.get("web_extraction", {})
                if record._web_verified and (
                    record._web_published or web_result.tier == "L0"
                    or (previous_meta.get("tier") == "L2" and web_result.tier == "L1")
                ):
                    if web_result.tier != "L0" or record.snippet or not web_result.visible_text:
                        return record.to_dict(include_full_content=True)
                    record.snippet = web_result.visible_text
                    _apply_quality_metadata(record)
                    record.updated_at = time.time()
                    changed_record = deepcopy(record)

            if record is None:
                prefix = f"T{task_id}-S"
                source_id = f"{prefix}{_next_source_sequence(task_ids, prefix)}"
                record = EvidenceRecord(
                    source_id=source_id,
                    task_id=task_id,
                    query="",
                    title=title if web_result is not None else title or url,
                    url=url,
                    snippet=(
                        web_result.visible_text if web_result is not None and web_result.tier == "L0"
                        else ("" if web_result is not None else truncate_text(full_content, 600))
                    ),
                    domain=_domain(url),
                    source_type=_classify_source_type(url, title or url),
                    created_at=time.time(),
                    updated_at=time.time(),
                )
                _apply_quality_metadata(record)
                self._records_by_id[source_id] = record
                task_ids.append(source_id)
                if normalized_url:
                    url_index[normalized_url] = source_id

            if web_result is not None and changed_record is None:
                previous = record.provenance_meta.get("web_extraction", {})
                version = int(previous.get("artifact_version") or 0)
                metadata = web_result.provenance(artifact_version=max(1, version))
                if (
                    previous == metadata and record.full_content == full_content
                    and record.multimodal_blocks == multimodal_blocks
                    and (not title or record.title == title)
                ):
                    return record.to_dict(include_full_content=True)
                if record._web_verified and (
                    record.full_content != full_content or record.multimodal_blocks != multimodal_blocks
                    or record.title != title or previous.get("tier") != web_result.tier
                ):
                    metadata["artifact_version"] += 1
                record.title = title or record.title
                if web_result.tier == "L0" and not record.snippet:
                    record.snippet = web_result.visible_text
                record.full_content = full_content
                record.raw_content = ""
                record.multimodal_blocks = deepcopy(multimodal_blocks)
                record.provenance_meta["web_extraction"] = metadata
                record._web_managed = True
                record._web_verified = web_result.supports_facts
                _apply_quality_metadata(record)
                record.updated_at = time.time()
                changed_record = deepcopy(record)
            elif web_result is None:
                if title and len(title) > len(record.title):
                    record.title = title
                if full_content:
                    record.full_content = full_content
                    if not record.snippet:
                        record.snippet = truncate_text(full_content, 600)
                if multimodal_blocks is not None:
                    record.multimodal_blocks = [
                        dict(block) for block in multimodal_blocks if isinstance(block, dict)
                    ]
                if not record.source_type or record.source_type == "web":
                    record.source_type = _classify_source_type(url, record.title)
                _apply_quality_metadata(record)
                record.updated_at = time.time()
                changed_record = record

        if changed_record is not None:
            self._notify_listeners(task_id=task_id, records=[changed_record], reason="fetch")
        return changed_record.to_dict(include_full_content=True) if changed_record is not None else {}

    def record_mcp_results(
        self,
        *,
        task_id: int,
        server_id: str,
        tool_name: str,
        items: list[dict[str, Any]],
        query: str = "",
        tenant_id: str = "default",
        trust_tier: str = "curated_private",
    ) -> list[dict[str, Any]]:
        """Normalize MCP tool outputs into first-class [Tn-Mm] evidence records."""

        if not items or task_id <= 0:
            return []

        changed_records: list[EvidenceRecord] = []
        normalized_tenant = str(tenant_id or "default").strip() or "default"
        prefix = f"T{task_id}-M"
        with self._lock:
            task_ids = self._task_source_ids.setdefault(task_id, [])
            url_index = self._task_url_index.setdefault(task_id, {})
            next_m = 1 + max(
                (int(sid[len(prefix):]) for sid in task_ids
                 if sid.startswith(prefix) and sid[len(prefix):].isascii() and sid[len(prefix):].isdigit()),
                default=0,
            )

            for item in items:
                if not isinstance(item, dict):
                    continue
                snippet = str(item.get("snippet") or item.get("content") or item.get("text") or "").strip()
                raw_content = str(item.get("raw_content") or item.get("full_content") or snippet).strip()
                if not snippet and not raw_content:
                    continue
                if not snippet:
                    snippet = truncate_text(raw_content, 600)
                title = str(item.get("title") or f"{server_id}:{tool_name} #{next_m}").strip()
                url = str(item.get("citation_uri") or item.get("url") or f"mcp://{server_id}/{tool_name}/{next_m}").strip()
                dedup_key = _mcp_evidence_key(server_id, normalized_tenant, url)
                existing_id = url_index.get(dedup_key)
                if existing_id and existing_id in self._records_by_id:
                    rec = self._records_by_id[existing_id]
                    if len(raw_content) > len(rec.full_content):
                        rec.full_content = raw_content
                    if len(snippet) > len(rec.snippet):
                        rec.snippet = snippet
                    rec.updated_at = time.time()
                    _apply_quality_metadata(rec)
                    changed_records.append(rec)
                    continue

                source_id = f"{prefix}{next_m}"
                next_m += 1
                raw_type = str(item.get("source_type") or "").strip()
                tier_fallback = {
                    "curated_private": "mcp_curated_filing",
                    "internal_metrics": "mcp_internal_metrics",
                    "external_mcp": "mcp_external",
                }.get(str(trust_tier or "").strip(), "mcp_tool")
                source_type = raw_type if raw_type.startswith("mcp_") else tier_fallback
                page_no_raw = item.get("page_no")
                page_no = int(page_no_raw) if isinstance(page_no_raw, int) and not isinstance(page_no_raw, bool) else None
                rec = EvidenceRecord(
                    source_id=source_id,
                    task_id=task_id,
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    raw_content=raw_content,
                    full_content=str(item.get("full_content") or "").strip(),
                    backend=f"mcp:{server_id}",
                    backend_sources=[f"mcp:{server_id}"],
                    provider_count=1,
                    domain=f"{server_id}.mcp.internal",
                    source_type=source_type,
                    source_origin="mcp",
                    mcp_server_id=server_id,
                    mcp_tool_name=tool_name,
                    tenant_id=normalized_tenant,
                    provenance_meta=dict(item["provenance_meta"]) if isinstance(item.get("provenance_meta"), dict) else {},
                    modality=str(item.get("modality") or "text").strip() or "text",
                    page_no=page_no,
                    published_at=str(item.get("published_at") or "").strip() or None,
                    structured_payload=(
                        dict(item.get("structured_payload"))
                        if isinstance(item.get("structured_payload"), dict)
                        else None
                    ),
                    created_at=time.time(),
                    updated_at=time.time(),
                )
                _apply_quality_metadata(rec)
                self._records_by_id[source_id] = rec
                task_ids.append(source_id)
                url_index[dedup_key] = source_id
                changed_records.append(rec)

        if changed_records:
            self._notify_listeners(task_id=task_id, records=changed_records, reason="mcp")
        return [rec.to_dict(include_full_content=True) for rec in changed_records]

    def record_computation_evidence(
        self,
        *,
        task_id: int,
        title: str,
        summary_text: str,
        derived_from_source_ids: list[str],
        computation_artifact: dict[str, Any],
        structured_payload: dict[str, Any] | None = None,
        tenant_id: str = "default",
        backend: str = "sandbox",
    ) -> dict[str, Any]:
        """Record a deterministic sandbox computation result as a [Tn-Cm] citation."""

        if task_id <= 0:
            raise ValueError("task_id must be positive")
        prefix = f"T{task_id}-C"
        normalized_tenant = str(tenant_id or "default").strip() or "default"
        clean_derived = list(dict.fromkeys(str(sid).strip() for sid in derived_from_source_ids if str(sid).strip()))
        raw_backend = str(backend or "sandbox").strip() or "sandbox"
        backend_label = raw_backend if raw_backend.startswith("sandbox") else f"sandbox:{raw_backend}"

        with self._lock:
            task_ids = self._task_source_ids.setdefault(task_id, [])
            next_c = _next_source_sequence(task_ids, prefix)
            source_id = f"{prefix}{next_c}"
            request_id = self._request_id_getter() if self._request_id_getter else None
            url = f"sandbox://{request_id or 'local'}/{source_id}"
            structured_payload = dict(structured_payload) if isinstance(structured_payload, dict) else None
            if structured_payload is not None:
                charts = structured_payload.get("charts")
                if isinstance(charts, list):
                    structured_payload["charts"] = [
                        {**chart, "source_ids": list(dict.fromkeys([source_id, *(chart.get("source_ids") or [])]))}
                        for chart in charts if isinstance(chart, dict) and isinstance(chart.get("source_ids", []), list)
                    ]
            rec = EvidenceRecord(
                source_id=source_id,
                task_id=task_id,
                query=title,
                title=title.strip() or f"Sandbox Computation {next_c}",
                url=url,
                snippet=truncate_text(summary_text.strip(), 1000),
                raw_content=summary_text.strip(),
                full_content=summary_text.strip(),
                backend=backend_label,
                backend_sources=[backend_label],
                provider_count=1,
                domain="sandbox.local",
                source_type="computed_sandbox",
                source_origin="sandbox",
                tenant_id=normalized_tenant,
                modality="computation",
                derived_from_source_ids=clean_derived,
                computation_artifact=dict(computation_artifact or {}),
                structured_payload=structured_payload,
                created_at=time.time(),
                updated_at=time.time(),
            )
            _apply_quality_metadata(rec)
            self._records_by_id[source_id] = rec
            task_ids.append(source_id)

        self._notify_listeners(task_id=task_id, records=[rec], reason="compute")
        return rec.to_dict(include_full_content=True)

    def _select_ids_locked(self, task_id: int | None, source_ids: Sequence[str] | None) -> list[str]:
        if source_ids is None:
            return (
                list(self._task_source_ids.get(task_id, [])) if task_id is not None
                else [sid for tid in sorted(self._task_source_ids) for sid in self._task_source_ids[tid]]
            )
        return [
            sid for sid in dict.fromkeys(source_ids)
            if sid in self._records_by_id and (
                task_id is None or self._records_by_id[sid].task_id == task_id
            )
        ]

    def list_index_cards(
        self,
        *,
        task_id: int | None = None,
        source_ids: Sequence[str] | None = None,
        max_gist_chars: int = 120,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return scoped, lightweight pointer cards for progressive disclosure."""

        with self._lock:
            ids = self._select_ids_locked(task_id, source_ids)
            cards: list[dict[str, Any]] = []
            for sid in ids if limit is None else ids[:max(0, limit)]:
                rec = self._records_by_id[sid]
                gist = sanitize_injection_text(rec.snippet or rec.full_content or rec.raw_content or rec.title)
                cards.append(
                    {
                        "source_id": rec.source_id,
                        "task_id": rec.task_id,
                        "source_origin": rec.source_origin,
                        "modality": rec.modality,
                        "title": sanitize_injection_text(rec.title)[:100],
                        "domain": sanitize_injection_text(rec.domain)[:80],
                        "source_type": rec.source_type,
                        "quality_label": rec.quality_label,
                        "freshness_label": rec.freshness_label,
                        "published_at": rec.published_at,
                        "has_full_content": bool(rec.full_content or rec.archive_has_full_content),
                        "page_no": rec.page_no,
                        "char_len": len(rec.full_content or rec.raw_content or rec.snippet),
                        "one_line_gist": gist[:max(0, max_gist_chars)],
                    }
                )
            return cards

    def get_lazy_source_chunks(
        self,
        source_ids: Sequence[str],
        *,
        chunk_indices: Sequence[int] | None,
        max_chunks_per_source: int,
        char_budget: int,
    ) -> list[ChunkLazyLoadResult]:
        """Expose a bounded record excerpt when no indexed chunk is available."""

        with self._lock:
            by_source: dict[str, list[EvidenceChunk]] = {}
            for sid in source_ids:
                rec = self._records_by_id.get(sid)
                if rec is None:
                    continue
                body = rec.full_content or rec.raw_content or rec.snippet
                if body:
                    by_source[sid] = [
                        EvidenceChunk(
                            f"{sid}::chunk::1", rec.task_id, sid, body, 0, len(body),
                            bool(rec.full_content), "current", page_no=rec.page_no,
                            modality=rec.modality, anchor_id=rec.anchor_id,
                        ),
                    ]
        return pack_lazy_chunks(
            source_ids,
            by_source,
            chunk_indices=chunk_indices,
            max_chunks_per_source=max_chunks_per_source,
            char_budget=char_budget,
        )

    def list_task_evidence(
        self,
        task_id: int,
        *,
        include_full_content: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            source_ids = list(self._task_source_ids.get(task_id, []))
            if limit is not None:
                source_ids = source_ids[: max(limit, 0)]
            return [
                self._records_by_id[source_id].to_dict(
                    include_full_content=include_full_content,
                )
                for source_id in source_ids
                if source_id in self._records_by_id
            ]

    def get_evidence(
        self,
        source_id: str,
        *,
        include_full_content: bool = False,
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self._records_by_id.get(source_id)
            if record is None:
                return None
            return record.to_dict(include_full_content=include_full_content)

    def lookup(
        self,
        *,
        task_id: int | None = None,
        source_ids: list[str] | None = None,
        include_full_content: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            ids = self._select_ids_locked(task_id, source_ids)
            if source_ids is None and task_id is None:
                ids.sort()
            if limit is not None:
                ids = ids[:max(0, limit)]
            return [
                self._records_by_id[sid].to_dict(include_full_content=include_full_content)
                for sid in ids
            ]

    def build_reference_map(self, source_ids: list[str]) -> list[dict[str, str]]:
        """Resolve source ids to compact reference records."""

        references: list[dict[str, str]] = []
        seen: set[str] = set()
        for source_id in source_ids:
            if source_id in seen:
                continue
            seen.add(source_id)
            item = self.get_evidence(source_id)
            if not item:
                continue
            title = (
                str(item.get("title") or "").strip()
                or str(item.get("url") or "").strip()
                or source_id
            )
            url = str(item.get("url") or "").strip()
            ref_entry: dict[str, Any] = {
                "source_id": source_id,
                "title": title,
                "url": url,
                "domain": str(item.get("domain") or ""),
                "published_at": str(item.get("published_at") or ""),
            }
            if str(item.get("source_origin") or "") == "sandbox" or url.startswith("sandbox://"):
                ref_entry["source_origin"] = str(item.get("source_origin") or "sandbox")
                if item.get("backend"):
                    ref_entry["backend"] = str(item.get("backend"))
                if isinstance(item.get("computation_artifact"), dict):
                    ref_entry["computation_artifact"] = dict(item.get("computation_artifact"))
                if isinstance(item.get("derived_from_source_ids"), list):
                    ref_entry["derived_from_source_ids"] = list(item.get("derived_from_source_ids"))
            references.append(ref_entry)
        return references

    def register_historical_evidence(
        self,
        *,
        task_id: int,
        historical: Any,
        url: str = "",
        title: str = "",
        origin_request_id: str | None = None,
        origin_source_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Re-register verified historical GraphEvidence into a current task with a fresh Tn-Sm ID."""
        from services.knowledge_graph import GraphEvidence, graph_evidence_from_payload
        from services.memory_scope import valid_internal_id

        ev = historical if isinstance(historical, GraphEvidence) else graph_evidence_from_payload(historical)
        if ev is None:
            return None
        if (
            ev.origin != "web" or ev.level not in {"L1", "L2"} or ev.isolated
            or ev.trusted is not True or ev.visible is not True or ev.facts_eligible is not True
            or ev.sanitizer_version != WEB_SANITIZER_VERSION or not ev.fragments
        ):
            return None
        origin_req = str(origin_request_id or ev.request_id or "").strip()
        origin_sid = str(origin_source_id or ev.source_id or "").strip()
        if not valid_internal_id(origin_req) or not valid_internal_id(origin_sid):
            return None

        cleaned_url = _normalize_url(url) if url else ""
        if url and not cleaned_url:
            return None
        if not cleaned_url:
            cleaned_url = f"https://historical.invalid/{origin_req}/{origin_sid}"
        safe_title = sanitize_web_text(title or origin_sid, max_chars=240) or origin_sid

        fragments_parts: list[str] = []
        anchors: list[dict[str, Any]] = []
        cursor = 0
        for index, frag in enumerate(ev.fragments, 1):
            clean_frag = sanitize_web_text(frag.text, max_chars=50_000)
            if not clean_frag:
                continue
            anchor_id = frag.chunk_id if re.fullmatch(r"(?:html-p|text-p|pdf-p)[0-9]{1,6}", frag.chunk_id) else f"text-p{index}"
            if cursor:
                fragments_parts.append("\n\n")
                cursor += 2
            start = cursor
            fragments_parts.append(clean_frag)
            cursor += len(clean_frag)
            anchors.append({"anchor_id": anchor_id, "start": start, "end": cursor})
        full_text = "".join(fragments_parts)
        if not full_text or not anchors:
            return None

        artifact_ver = int(ev.artifact_version) if str(ev.artifact_version).isascii() and str(ev.artifact_version).isdecimal() else 1
        adapter = "static" if ev.level == "L1" else "reader_http"
        changed_record: EvidenceRecord | None = None
        with self._lock:
            task_ids = self._task_source_ids.setdefault(task_id, [])
            url_index = self._task_url_index.setdefault(task_id, {})
            for existing_id in task_ids:
                existing = self._records_by_id.get(existing_id)
                hist_meta = (existing.provenance_meta or {}).get("historical_origin") if existing else None
                if isinstance(hist_meta, dict) and (
                    hist_meta.get("origin_request_id"),
                    hist_meta.get("origin_source_id"),
                    str(hist_meta.get("artifact_version") or ""),
                ) == (origin_req, origin_sid, ev.artifact_version):
                    return existing.to_dict()
            dedup_key = _normalize_url(cleaned_url) or f"title::{safe_title.casefold()}"
            existing_by_url = url_index.get(dedup_key)
            if existing_by_url and existing_by_url in self._records_by_id:
                return self._records_by_id[existing_by_url].to_dict()

            next_index = _next_source_sequence(task_ids, f"T{task_id}-S")
            source_id = f"T{task_id}-S{next_index}"
            record = EvidenceRecord(
                source_id=source_id,
                task_id=task_id,
                query="",
                title=safe_title,
                url=cleaned_url,
                snippet=sanitize_web_snippet(full_text[:320]),
                raw_content="",
                full_content=full_text,
                archive_has_full_content=True,
                backend="graph_historical",
                domain=_domain(cleaned_url),
                source_type=_classify_source_type(cleaned_url, safe_title),
                tenant_id=ev.tenant_id,
                source_origin="web",
                provenance_meta={
                    "web_extraction": {
                        "tier": ev.level,
                        "adapter": adapter,
                        "sanitizer_version": WEB_SANITIZER_VERSION,
                        "quality_version": QUALITY_VERSION,
                        "policy_version": POLICY_VERSION,
                        "artifact_version": max(1, artifact_ver),
                        "fact_eligible": True,
                        "text_anchors": anchors,
                    },
                    "historical_origin": {
                        "origin_request_id": origin_req,
                        "origin_source_id": origin_sid,
                        "artifact_version": ev.artifact_version,
                    },
                },
                created_at=time.time(),
                updated_at=time.time(),
            )
            if self.web_extraction_enabled:
                record._web_managed = True
                record._web_verified = True
                record._web_published = True
            _apply_quality_metadata(record)
            self._records_by_id[source_id] = record
            task_ids.append(source_id)
            url_index[dedup_key] = source_id
            changed_record = deepcopy(record) if self.web_extraction_enabled else record

        if changed_record is not None:
            self._notify_listeners(task_id=task_id, records=[changed_record], reason="hydrate")
            return self.get_evidence(changed_record.source_id)
        return None

    def hydrate_from_tasks(self, tasks: list[Any], *, trusted_web_snapshot: bool = False) -> None:
        """Rebuild evidence indices. Opt in to web artifact reuse for published internal snapshots."""

        hydrated_by_task: dict[int, list[EvidenceRecord]] = {}
        for listener in self._listeners:
            reset = getattr(listener, "reset", None)
            if callable(reset):
                reset()
        with self._lock:
            self._records_by_id = {}
            self._task_source_ids = {}
            self._task_url_index = {}

            for task in tasks or []:
                task_id_value = getattr(task, "id", None)
                if task_id_value is None and isinstance(task, dict):
                    task_id_value = task.get("id")
                try:
                    task_id = int(task_id_value or 0)
                except (TypeError, ValueError):
                    task_id = 0
                if task_id <= 0:
                    continue

                evidence_items = getattr(task, "evidence_items", None)
                if evidence_items is None and isinstance(task, dict):
                    evidence_items = task.get("evidence_items")
                if not isinstance(evidence_items, list):
                    continue

                task_ids = self._task_source_ids.setdefault(task_id, [])
                url_index = self._task_url_index.setdefault(task_id, {})

                for item in evidence_items:
                    if not isinstance(item, dict):
                        continue

                    source_id = str(item.get("source_id") or "").strip()
                    if (
                        not source_id.startswith(f"T{task_id}-")
                        or not CITATION_PATTERN.fullmatch(f"[{source_id}]")
                        or source_id in self._records_by_id
                    ):
                        continue

                    title = str(item.get("title") or item.get("url") or source_id).strip()
                    url = str(item.get("url") or "").strip()
                    snippet = str(item.get("snippet") or "").strip()
                    query = str(item.get("query") or "").strip()
                    backend = str(item.get("backend") or "").strip()
                    backend_sources = [
                        str(source).strip()
                        for source in item.get("backend_sources") or []
                        if str(source).strip()
                    ]
                    provider_count = max(1, int(item.get("provider_count") or 1))
                    domain = str(item.get("domain") or _domain(url)).strip()
                    source_type = str(item.get("source_type") or _classify_source_type(url, title)).strip() or "web"
                    published_at = str(item.get("published_at") or "").strip() or None
                    source_updated_at = str(item.get("source_updated_at") or "").strip() or None
                    freshness_days = item.get("freshness_days")
                    freshness_label = str(item.get("freshness_label") or "").strip() or "unknown"
                    quality_score = int(item.get("quality_score") or 0)
                    quality_label = str(item.get("quality_label") or "").strip()

                    record = EvidenceRecord(
                        source_id=source_id,
                        task_id=task_id,
                        query=query,
                        title=title,
                        url=url,
                        snippet=snippet,
                        raw_content=str(item.get("raw_content") or ""),
                        full_content=str(item.get("full_content") or ""),
                        backend=backend,
                        backend_sources=backend_sources,
                        provider_count=provider_count,
                        domain=domain,
                        source_type=source_type,
                        source_reliability_score=float(item.get("source_reliability_score") or 0.0),
                        content_quality_score=float(item.get("content_quality_score") or 0.0),
                        quality_score=quality_score,
                        quality_label=quality_label or "medium",
                        published_at=published_at,
                        source_updated_at=source_updated_at,
                        freshness_days=(
                            int(freshness_days)
                            if freshness_days is not None and str(freshness_days).strip()
                            else None
                        ),
                        freshness_label=freshness_label,
                        quality_reasons=[
                            str(reason).strip()
                            for reason in item.get("quality_reasons") or []
                            if str(reason).strip()
                        ],
                        quality_flags=[
                            str(flag).strip()
                            for flag in item.get("quality_flags") or []
                            if str(flag).strip()
                        ],
                        archive_doc_key=str(item.get("archive_doc_key") or "").strip() or None,
                        archive_version_id=str(item.get("archive_version_id") or "").strip() or None,
                        archive_has_full_content=bool(item.get("archive_has_full_content")),
                        source_origin=str(item.get("source_origin") or "web").strip() or "web",
                        mcp_server_id=str(item.get("mcp_server_id") or "").strip() or None,
                        mcp_tool_name=str(item.get("mcp_tool_name") or "").strip() or None,
                        tenant_id=str(item.get("tenant_id") or "default").strip() or "default",
                        provenance_meta=(
                            dict(item.get("provenance_meta"))
                            if isinstance(item.get("provenance_meta"), dict)
                            else {}
                        ),
                        modality=str(item.get("modality") or "text").strip() or "text",
                        page_no=(
                            int(item.get("page_no"))
                            if isinstance(item.get("page_no"), int) and not isinstance(item.get("page_no"), bool)
                            else None
                        ),
                        bbox=list(item.get("bbox")) if isinstance(item.get("bbox"), list) else None,
                        anchor_id=str(item.get("anchor_id") or "").strip() or None,
                        multimodal_blocks=(
                            list(item.get("multimodal_blocks"))
                            if isinstance(item.get("multimodal_blocks"), list)
                            else []
                        ),
                        derived_from_source_ids=[
                            str(sid).strip()
                            for sid in (item.get("derived_from_source_ids") or [])
                            if str(sid).strip()
                        ],
                        computation_artifact=(
                            dict(item.get("computation_artifact"))
                            if isinstance(item.get("computation_artifact"), dict)
                            else None
                        ),
                        structured_payload=(
                            dict(item.get("structured_payload"))
                            if isinstance(item.get("structured_payload"), dict)
                            else None
                        ),
                        created_at=time.time(),
                        updated_at=time.time(),
                    )
                    if self.web_extraction_enabled and record.source_origin == "web":
                        record._web_managed = True
                        record.raw_content = ""
                        snapshot_title = str(item.get("title") or "")
                        if trusted_web_snapshot is True and _is_compatible_web_artifact(record):
                            record.title = snapshot_title
                            record.provenance_meta = deepcopy(record.provenance_meta)
                            record.multimodal_blocks = deepcopy(record.multimodal_blocks)
                            record._web_verified = True
                            record._web_published = True
                        else:
                            projection = normalize_snippet(url, snapshot_title, record.snippet)
                            if projection is None:
                                continue
                            record.title = projection.title
                            record.snippet = projection.visible_text
                            record.full_content = ""
                            record.multimodal_blocks = []
                            record.archive_doc_key = record.archive_version_id = record.anchor_id = None
                            record.archive_has_full_content = False
                            record.modality = "text"
                            record.page_no = record.bbox = None
                            record.structured_payload = record.computation_artifact = None
                            record.derived_from_source_ids = []
                            record.provenance_meta = {"web_extraction": projection.provenance()}
                    _apply_quality_metadata(record)
                    self._records_by_id[source_id] = record
                    task_ids.append(source_id)
                    dedup_key = (
                        _mcp_evidence_key(record.mcp_server_id or "", record.tenant_id, url)
                        if record.source_origin == "mcp" else _normalize_url(url) or f"title::{title.casefold()}"
                    )
                    url_index[dedup_key] = source_id
                    hydrated_by_task.setdefault(task_id, []).append(record)

        for hydrated_task_id, records in hydrated_by_task.items():
            self._notify_listeners(task_id=hydrated_task_id, records=records, reason="hydrate")


def format_evidence_sources(evidence_items: list[dict[str, Any]]) -> str:
    """Render task evidence as a compact source summary with source ids."""

    lines = []
    for item in evidence_items:
        source_id = item.get("source_id")
        if not source_id:
            continue
        metadata_bits = [
            str(item.get("domain") or "").strip(),
            str(item.get("source_type") or "").strip(),
            str(item.get("quality_label") or "").strip(),
            str(item.get("freshness_label") or "").strip(),
            str(item.get("published_at") or "").strip(),
        ]
        metadata = " | ".join(bit for bit in metadata_bits if bit)
        suffix = f" ({metadata})" if metadata else ""
        lines.append(
            f"* [{source_id}] {item.get('title') or item.get('url') or '未知来源'}{suffix} : {item.get('url', '')}"
        )
    return "\n".join(lines)


def build_task_context(
    evidence_items: list[dict[str, Any]],
    *,
    answer_text: str | None,
    config: Configuration,
) -> str:
    """Build summarization context that keeps source ids visible to the model."""

    blocks: list[str] = []
    if answer_text:
        blocks.append(
            "AI直接答案：\n"
            + truncate_text(answer_text, config.resolved_direct_answer_char_limit())
        )

    for item in evidence_items:
        source_id = str(item.get("source_id") or "")
        title = str(item.get("title") or item.get("url") or "未知来源")
        url = str(item.get("url") or "")
        snippet = str(item.get("snippet") or "")
        full_content = str(item.get("full_content") or "")
        domain = str(item.get("domain") or "")
        source_type = str(item.get("source_type") or "")
        quality_label = str(item.get("quality_label") or "")
        freshness_label = str(item.get("freshness_label") or "")
        published_at = str(item.get("published_at") or "")
        body = full_content or snippet
        body = truncate_cited_text(body, config.resolved_max_tokens_per_source() * 4)
        blocks.append(
            f"[{source_id}] {title}\n"
            f"URL: {url}\n"
            f"域名: {domain}\n"
            f"来源类型: {source_type}\n"
            f"质量等级: {quality_label}\n"
            f"发布时间: {published_at or 'unknown'}\n"
            f"时效标签: {freshness_label}\n"
            f"摘要: {snippet}\n"
            f"正文摘录: {body}"
        )

    context = "\n\n".join(blocks).strip()
    return truncate_cited_text(context, config.resolved_task_context_char_limit())


def build_task_context_from_hits(
    hits: list[EvidenceRetrievalHit | dict[str, Any]],
    *,
    answer_text: str | None,
    config: Configuration,
) -> str:
    """Build summarization context from filtered evidence chunks."""

    blocks: list[str] = []
    if answer_text:
        blocks.append(
            "AI直接答案：\n"
            + truncate_text(answer_text, config.resolved_direct_answer_char_limit())
        )

    for hit in hits:
        if isinstance(hit, dict):
            source_id = str(hit.get("source_id") or "")
            title = str(hit.get("title") or hit.get("url") or "未知来源")
            url = str(hit.get("url") or "")
            domain = str(hit.get("domain") or "")
            quality_label = str(hit.get("quality_label") or "")
            freshness_label = str(hit.get("freshness_label") or "")
            body = str(hit.get("text") or "")
            chunk_id = str(hit.get("chunk_id") or "")
            citation_eligible = hit.get("citation_eligible", hit.get("origin") != "history")
        else:
            source_id = str(hit.source_id or "")
            title = str(hit.title or hit.url or "未知来源")
            url = str(hit.url or "")
            domain = str(hit.domain or "")
            quality_label = str(hit.quality_label or "")
            freshness_label = str(hit.freshness_label or "")
            body = str(hit.text or "")
            chunk_id = str(hit.chunk_id or "")
            citation_eligible = hit.citation_eligible

        if not citation_eligible:
            continue
        excerpt = truncate_cited_text(body, config.resolved_max_tokens_per_source() * 4)
        blocks.append(
            f"[{source_id}] {title}\n"
            f"URL: {url}\n"
            f"域名: {domain}\n"
            f"质量等级: {quality_label}\n"
            f"时效标签: {freshness_label}\n"
            f"Chunk: {chunk_id}\n"
            f"正文摘录: {excerpt}"
        )

    context = "\n\n".join(blocks).strip()
    return truncate_cited_text(context, config.resolved_task_context_char_limit())


def render_references(reference_items: list[dict[str, Any]]) -> str:
    """Render a standard reference section."""

    if not reference_items:
        return "- 暂无可用来源"
    rendered_lines: list[str] = []
    for item in reference_items:
        source_id = str(item.get("source_id") or "").strip()
        title = str(item.get("title") or "").strip() or source_id or "来源"
        url = str(item.get("url") or "").strip()
        if str(item.get("source_origin") or "").strip() == "sandbox" or url.startswith("sandbox://"):
            artifact = item.get("computation_artifact") or {}
            exec_hash = str(artifact.get("execution_hash") or "")[:8]
            derived = ", ".join(str(s) for s in (item.get("derived_from_source_ids") or ()) if s)
            backend = str(item.get("backend") or "sandbox").strip()
            meta = (
                f"({backend}"
                + (f" | sha256:{exec_hash}" if exec_hash else "")
                + (f" | 基于 {derived}" if derived else "")
                + ")"
            )
            rendered_lines.append(
                f"- [{source_id}] 计算验证：{title} {meta}" if source_id else f"- 计算验证：{title} {meta}"
            )
            continue
        line = f"- [{source_id}] {title}" if source_id else f"- {title}"
        if url:
            line = f"{line} - {url}"
        rendered_lines.append(line)
    return "\n".join(rendered_lines)


def build_research_keywords(*texts: Any) -> list[str]:
    """Extract stable topic keywords used to reject irrelevant evidence snippets."""

    keywords: list[str] = []
    seen: set[str] = set()
    fallback_keywords: list[str] = []

    for value in texts:
        text = normalize_agent_markdown(str(value or "").strip())
        if not text:
            continue
        for token in RESEARCH_TOKEN_PATTERN.findall(text):
            normalized = token.strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            if lowered in GENERIC_RESEARCH_TOKENS:
                fallback_keywords.append(normalized)
                continue
            keywords.append(normalized)

    if keywords:
        return keywords[:12]
    return fallback_keywords[:8]


def looks_like_source_label_text(
    value: Any,
    *,
    source_title: str = "",
) -> bool:
    """Return whether text is mostly a source label instead of a user-facing finding."""

    text = strip_citation_markers(normalize_agent_markdown(str(value or "").strip()))
    if not text:
        return True

    compact = re.sub(r"\s+", " ", text).strip(" -:：;；,.，。")
    if not compact:
        return True

    if source_title:
        title = strip_citation_markers(normalize_agent_markdown(str(source_title or "").strip()))
        if title:
            title_lower = title.casefold()
            compact_lower = compact.casefold()
            if compact_lower == title_lower or compact_lower in title_lower or title_lower in compact_lower:
                return True

    if any(pattern.search(compact) for pattern in SOURCE_LABEL_PLACEHOLDER_PATTERNS):
        return True

    if compact.endswith(("文章", "专栏", "社区", "论坛", "网站", "网页", "页面", "博客", "链接", "帖子")):
        return True

    return False


def derive_grounded_findings_from_evidence(
    evidence_items: list[dict[str, Any]],
    *,
    query_text: str,
    allowed_source_ids: list[str] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Convert evidence snippets into grounded fallback findings."""

    if not evidence_items or limit <= 0:
        return []

    keywords = build_research_keywords(query_text)
    allowed = {str(source_id).strip() for source_id in allowed_source_ids or [] if str(source_id).strip()}

    findings: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for item in evidence_items:
        source_id = str(item.get("source_id") or "").strip()
        if not source_id:
            continue
        if allowed and source_id not in allowed:
            continue

        text = _best_grounded_claim_text(item, keywords=keywords)
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        findings.append({"text": text, "source_ids": [source_id]})
        if len(findings) >= limit:
            break

    return findings


def _best_grounded_claim_text(item: dict[str, Any], *, keywords: list[str]) -> str:
    title = normalize_agent_markdown(str(item.get("title") or "").strip())
    title_compact = strip_citation_markers(title)
    candidate_blocks = [
        str(item.get("full_content") or "").strip(),
        str(item.get("raw_content") or "").strip(),
        str(item.get("snippet") or "").strip(),
    ]
    combined_text = " ".join(part for part in (title_compact, *candidate_blocks) if part).casefold()
    if keywords and not any(keyword.casefold() in combined_text for keyword in keywords):
        return ""

    best_text = ""
    best_score = -1
    quality_bonus = {
        "high": 2,
        "medium": 1,
        "low": 0,
    }.get(str(item.get("quality_label") or "").strip().lower(), 0)

    for block in candidate_blocks:
        if not block:
            continue
        for sentence in _split_grounding_sentences(block):
            candidate = _clean_grounding_sentence(sentence)
            if not candidate or looks_like_source_label_text(candidate, source_title=title_compact):
                continue
            if title_compact and candidate.casefold() in title_compact.casefold():
                continue

            match_count = _keyword_match_count(candidate, keywords)
            if keywords and match_count <= 0:
                continue

            score = match_count * 5 + quality_bonus
            if 28 <= len(candidate) <= 220:
                score += 2
            elif len(candidate) >= 18:
                score += 1
            if "..." in candidate or "…" in candidate:
                score -= 1

            if score > best_score:
                best_score = score
                best_text = candidate

    return best_text


def _split_grounding_sentences(text: str) -> list[str]:
    normalized = normalize_agent_markdown(str(text or "").strip())
    if not normalized:
        return []

    parts = re.split(r"(?<=[。！？!?；;])\s+|\n+", normalized)
    if len(parts) == 1:
        return [normalized]
    return [part for part in parts if part.strip()]


def _clean_grounding_sentence(text: str) -> str:
    cleaned = normalize_agent_markdown(str(text or "").strip())
    if not cleaned:
        return ""

    cleaned = LIST_PREFIX_PATTERN.sub("", cleaned).strip()
    cleaned = strip_citation_markers(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:：;；,.，。")
    if len(cleaned) < 18:
        return ""
    if URL_PATTERN.search(cleaned):
        return ""
    return cleaned


def _keyword_match_count(text: str, keywords: list[str]) -> int:
    if not keywords:
        return 1 if len(str(text or "").strip()) >= 18 else 0

    lowered = str(text or "").casefold()
    matches = 0
    for keyword in keywords:
        if keyword.casefold() in lowered:
            matches += 1
    return matches


class SearchWebTool(Tool):
    """Tool wrapper around the repo's structured web-search dispatcher."""

    budget_provider: Callable[[], Any] | None = None

    def __init__(
        self,
        *,
        config: Configuration,
        evidence_store: EvidenceStore,
        observer_getter: Callable[[], RequestTrace | None],
        scope_provider: Callable[[int], tuple[str, str] | None] | None = None,
    ) -> None:
        super().__init__(
            name="search_web",
            description=(
                "执行结构化网页搜索并写入证据库。参数建议使用 JSON，至少包含 "
                "task_id、query、research_topic、task_title、task_intent。"
            ),
        )
        self._config = config
        self._evidence_store = evidence_store
        self._observer_getter = observer_getter
        self._scope_provider = scope_provider

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task_id", type="integer", description="任务 ID"),
            ToolParameter(name="query", type="string", description="搜索查询"),
            ToolParameter(name="research_topic", type="string", description="研究主题"),
            ToolParameter(name="task_title", type="string", description="任务标题"),
            ToolParameter(name="task_intent", type="string", description="任务目标"),
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        task_id = int(parameters.get("task_id") or 0)
        query = str(parameters.get("query") or parameters.get("input") or "").strip()
        if task_id <= 0:
            raise ValueError("task_id is required")
        if not query:
            raise ValueError("query is required")

        overrides: dict[str, Any] = {}
        backend = str(parameters.get("backend") or "").strip()
        if backend:
            overrides["search_api"] = backend
        if "fetch_full_page" in parameters:
            overrides["fetch_full_page"] = bool(parameters.get("fetch_full_page"))
        budget = self.budget_provider() if self.budget_provider else None
        web_enabled = self._evidence_store.web_extraction_enabled
        config = self._config.model_copy(update=overrides) if overrides and budget is None and not web_enabled else self._config
        dispatch_options = {"budget": budget} if budget is not None else {}
        scope = self._scope_provider(task_id) if self._scope_provider else None
        if web_enabled:
            if self._scope_provider is not None and scope is None:
                return json.dumps({
                    "task_id": task_id, "query": query, "backend": "", "cache_hit": False,
                    "cache_strategy": "miss", "answer": None, "notices": [],
                    "evidence": self._evidence_store.list_task_evidence(task_id),
                }, ensure_ascii=False)
            dispatch_options["web_extraction_enabled"] = True
        elif self._scope_provider is not None and self._config.web_extraction_enabled_for(
            scope[0] if scope else self._config.default_tenant_id
        ):
            dispatch_options["web_extraction_enabled"] = False

        payload, notices, answer_text, backend_label, cache_hit, cache_strategy = dispatch_search(
            query,
            config,
            0,
            observer=self._observer_getter(),
            cache_context={
                "research_topic": parameters.get("research_topic"),
                "task_title": parameters.get("task_title"),
                "task_intent": parameters.get("task_intent"),
                **({"tenant_id": scope[0], "request_id": scope[1], "task_id": str(task_id)} if scope else {}),
                **({"tenant_id": budget.governor.tenant_id} if budget is not None else {}),
            },
            max_results=int(parameters.get("max_results") or 5),
            **dispatch_options,
        )
        evidence_items = self._evidence_store.record_search_results(
            task_id=task_id,
            query=query,
            search_payload=payload,
            backend=backend_label,
        )

        return json.dumps(
            {
                "task_id": task_id,
                "query": query,
                "backend": backend_label,
                "cache_hit": cache_hit,
                "cache_strategy": cache_strategy,
                "answer": answer_text,
                "notices": notices,
                "evidence": public_web_evidence_projection(evidence_items) if web_enabled else evidence_items,
            },
            ensure_ascii=False,
        )


class FetchPageTool(Tool):
    """Fetch and normalize a specific page to enrich existing evidence."""

    budget_provider: Callable[[], Any] | None = None

    def __init__(
        self,
        *,
        evidence_store: EvidenceStore,
        timeout_seconds: float = 10.0,
        multimodal_parser: Any | None = None,
        web_extractor: WebExtractor | None = None,
        extraction_context_provider: Callable[..., ExtractionContext | None] | None = None,
        commit_lock: Any = None,
        persist_web_evidence: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(
            name="fetch_page",
            description=(
                "抓取单个网页正文并回填到证据库。参数建议使用 JSON，包含 task_id、source_id、url。"
            ),
        )
        self._evidence_store = evidence_store
        self._timeout_seconds = timeout_seconds
        self._multimodal_parser = multimodal_parser
        self._extraction_context_provider = extraction_context_provider
        self._web_extractor = web_extractor
        self._commit_lock = commit_lock
        self._persist_web_evidence = persist_web_evidence
        if self._web_extractor is None and evidence_store.web_extraction_enabled:
            self._web_extractor = self.build_web_extractor()

    def build_web_extractor(self) -> WebExtractor:
        parser = self._multimodal_parser
        return WebExtractor(
            pdf_parser=lambda data, check: _extract_pdf_text(
                data, max_pages=min(20, getattr(parser, "max_pdf_pages", 20)), check=check,
            ),
            pdf_limit=getattr(parser, "max_pdf_bytes", 2 * 1024 * 1024),
        )

    def set_web_extractor(self, extractor: WebExtractor) -> None:
        self._web_extractor = extractor

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task_id", type="integer", description="任务 ID"),
            ToolParameter(name="source_id", type="string", description="来源 ID", required=False),
            ToolParameter(name="url", type="string", description="网页链接"),
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        task_id = int(parameters.get("task_id") or 0)
        source_id = str(parameters.get("source_id") or "").strip() or None
        url = str(parameters.get("url") or "").strip()
        if task_id <= 0:
            raise ValueError("task_id is required")
        if not url:
            raise ValueError("url is required")
        if source_id:
            record = self._evidence_store.get_evidence(source_id)
            if (
                not record or record["task_id"] != task_id or record["source_origin"] != "web"
                or _normalize_url(record["url"]) != _normalize_url(url)
            ):
                raise ValueError("source_id does not match the task's web evidence URL")

        parser = getattr(self, "_multimodal_parser", None)
        budget = self.budget_provider() if self.budget_provider else None
        if self._evidence_store.web_extraction_enabled:
            return self.fetch_tiered(task_id, source_id, url, budget=budget)[0]

        if budget is not None:
            artifact = _fetch_page_artifact(url, timeout_seconds=self._timeout_seconds, budget=budget)
            title, content, multimodal_blocks = artifact.title, artifact.content, None
        elif parser is not None and getattr(parser, "enabled", False):
            artifact = _fetch_page_artifact(
                url,
                timeout_seconds=self._timeout_seconds,
                parser=parser,
                multimodal_enabled=True,
            )
            title = artifact.title
            content = artifact.content
            multimodal_blocks = artifact.multimodal_blocks
        else:
            title, content = _fetch_page_text(url, timeout_seconds=self._timeout_seconds)
            multimodal_blocks = None

        item = self._evidence_store.update_full_content(
            task_id=task_id,
            source_id=source_id,
            url=url,
            title=title,
            full_content=content,
            multimodal_blocks=multimodal_blocks,
        )
        return json.dumps(item, ensure_ascii=False)


    def fetch_tiered(
        self, task_id: int, source_id: str | None, url: str, *, budget: Any = None,
        context: ExtractionContext | None = None,
    ) -> tuple[str, str | None]:
        extractor = self._web_extractor
        if extractor is None:
            return "{}", "adapter_unavailable"
        if context is None and self._extraction_context_provider:
            context = self._extraction_context_provider(task_id)
            if context is None:
                return "{}", "not_eligible"
        if context is None:
            if budget is None:
                return "{}", "budget_denied"
            remaining = budget.governor.remaining_timeout(self._timeout_seconds, stage=budget.stage)
            context = ExtractionContext(
                budget.governor.tenant_id, budget.governor.request_id, task_id,
                FetchScope(
                    time.monotonic() + remaining,
                    check_owner=lambda: not budget.blocked,
                    cancelled=budget.agent._is_cancelled,
                    attempt=lambda demand: budget.attempt(demand, required=False),
                ),
                can_dispatch=budget.can_dispatch,
            )
        if context.task_id != task_id or (context.source_id is not None and context.source_id != source_id):
            return "{}", "not_eligible"
        context = replace(context, static_timeout=min(context.static_timeout, self._timeout_seconds))
        snippet = self._evidence_store.web_snippet_for_fetch(source_id, url) if source_id else None
        observed_reason = None
        record_result = getattr(context, "on_result", None)

        def on_result(tier: str | None, reason: str | None) -> None:
            nonlocal observed_reason
            observed_reason = reason
            if record_result is not None:
                record_result(tier, reason)

        if hasattr(context, "on_result"):
            context = replace(context, on_result=on_result)
        try:
            result = extractor.extract(url, context=context, snippet=snippet)
            if result is None:
                return "{}", observed_reason or "quality_insufficient"
            with self._commit_lock or nullcontext():
                context.scope.check()
                payload = self._evidence_store.update_full_content(
                    task_id=task_id, source_id=source_id, url=url, title="", full_content="",
                    web_result=result, check_owner=context.scope.check,
                )
                if payload and self._persist_web_evidence is not None:
                    self._persist_web_evidence(task_id)
            return json.dumps(public_web_evidence_projection(payload), ensure_ascii=False), result.degraded_reason
        except GatewayError as error:
            return "{}", error.reason


class EvidenceLookupTool(Tool):
    """Expose request-local evidence records to downstream agents."""

    def __init__(
        self,
        *,
        evidence_store: EvidenceStore,
        retrieval_service: EvidenceRetrievalService | None = None,
        request_id_getter: Callable[[], str | None] | None = None,
        config: Configuration | None = None,
    ) -> None:
        super().__init__(
            name="evidence_lookup",
            description=(
                "查询当前请求的证据库或按 query 检索证据 chunk。"
            ),
        )
        self._evidence_store = evidence_store
        self._retrieval_service = retrieval_service
        self._request_id_getter = request_id_getter
        self._config = config or getattr(retrieval_service, "config", None)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task_id", type="integer", description="任务 ID", required=False),
            ToolParameter(name="source_id", type="string", description="单个来源 ID", required=False),
            ToolParameter(name="source_ids", type="string", description="多个来源 ID", required=False),
            ToolParameter(name="query", type="string", description="检索查询", required=False),
            ToolParameter(name="top_k", type="integer", description="返回的 chunk 命中数量", required=False),
            ToolParameter(name="chunk_indices", type="string", description="可选 chunk 序号列表（1-based），如 1,2", required=False),
            ToolParameter(name="max_chunks_per_source", type="integer", description="每个来源最多返回 chunk 数", required=False),
            ToolParameter(name="char_budget", type="integer", description="lazy_chunks 模式总字符上限", required=False),
            ToolParameter(name="scope", type="string", description="current | history | hybrid", required=False),
            ToolParameter(name="mode", type="string", description="grounding | lead | repair | index_cards | lazy_chunks", required=False),
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        source_ids = parameters.get("source_ids")
        normalized_source_ids: list[str] = []
        if isinstance(source_ids, list):
            normalized_source_ids = [str(item).strip() for item in source_ids if str(item).strip()]
        elif isinstance(source_ids, str) and source_ids.strip():
            normalized_source_ids = [
                item.strip()
                for item in re.split(r"[\s,]+", source_ids)
                if item.strip()
            ]

        source_id = str(parameters.get("source_id") or "").strip()
        if source_id:
            normalized_source_ids = [source_id]

        task_id_raw = parameters.get("task_id")
        task_id = int(task_id_raw) if task_id_raw not in (None, "") else None
        source_filter = normalized_source_ids if source_ids is not None or source_id else None
        include_full_content = bool(parameters.get("include_full_content", False))
        limit = int(parameters.get("limit") or 0) or None
        query_text = str(parameters.get("query") or "").strip()
        mode = str(parameters.get("mode") or "grounding").strip().lower() or "grounding"

        if mode in {"index_cards", "lazy_chunks"}:
            payload_key = "index_cards" if mode == "index_cards" else "results"
            if not getattr(self._config, "evidence_pointer_disclosure_enabled", True):
                return json.dumps({"mode": mode, payload_key: [], "notice": "evidence pointer disclosure disabled"})
            card_limit = 24 if mode == "index_cards" else (6 if source_filter is None else None)
            if limit is not None:
                card_limit = limit if card_limit is None else min(card_limit, limit)
            cards = self._evidence_store.list_index_cards(
                task_id=task_id, source_ids=source_filter, limit=card_limit,
            )
            if mode == "index_cards":
                count_chunks = getattr(self._retrieval_service, "chunk_counts_by_source", None)
                counts = count_chunks() if callable(count_chunks) else {}
                for card in cards:
                    card["chunk_count"] = counts.get(card["source_id"], 0)
                return json.dumps({"mode": mode, payload_key: cards}, ensure_ascii=False)
            return self._load_lazy_chunks(parameters, [card["source_id"] for card in cards])

        evidence = self._evidence_store.lookup(
            task_id=task_id,
            source_ids=source_filter,
            include_full_content=include_full_content,
            limit=limit,
        )
        if query_text and self._retrieval_service is not None:
            result = self._retrieval_service.query(
                query_text,
                task_id=task_id,
                scope=str(parameters.get("scope") or "current").strip() or "current",
                mode=mode,
                top_k=int(parameters.get("top_k") or 0) or None,
                request_id=self._request_id_getter() if self._request_id_getter else None,
                allowed_source_ids=(
                    [item["source_id"] for item in evidence] if source_filter is not None else None
                ),
            )
            return json.dumps(
                {
                    "evidence": evidence,
                    **result.to_dict(),
                },
                ensure_ascii=False,
            )

        return json.dumps({"evidence": evidence}, ensure_ascii=False)

    def _load_lazy_chunks(self, parameters: dict[str, Any], target_ids: list[str]) -> str:
        raw_indices = parameters.get("chunk_indices")
        indices = re.split(r"[\s,]+", raw_indices) if isinstance(raw_indices, str) else raw_indices
        parsed_indices = None
        if raw_indices is not None:
            parsed_indices = [
                int(value) for item in (indices if isinstance(indices, list) else [])
                if (value := str(item).strip()).isascii() and value.isdecimal() and len(value) <= 12
            ]
        chunk_cap = getattr(self._config, "evidence_lazy_load_max_chunks_per_source", 2)
        char_cap = getattr(self._config, "evidence_lazy_load_char_budget", 1800)
        requested_chunks = parameters.get("max_chunks_per_source")
        requested_chars = parameters.get("char_budget")
        max_chunks = chunk_cap if requested_chunks in (None, "") else min(chunk_cap, max(0, int(requested_chunks)))
        char_budget = char_cap if requested_chars in (None, "") else min(char_cap, max(0, int(requested_chars)))
        load_chunks = getattr(self._retrieval_service, "get_source_chunks", None)
        results = load_chunks(
            source_ids=target_ids,
            chunk_indices=parsed_indices,
            max_chunks_per_source=max_chunks,
            char_budget=char_budget,
        ) if callable(load_chunks) and target_ids else []
        if not any(result.total_available_chunks for result in results):
            results = self._evidence_store.get_lazy_source_chunks(
                target_ids, chunk_indices=parsed_indices,
                max_chunks_per_source=max_chunks, char_budget=char_budget,
            )
        return json.dumps(
            {"mode": "lazy_chunks", "results": [result.to_dict() for result in results]}, ensure_ascii=False,
        )


def build_graph_evidence(
    record_or_dict: Any,
    *,
    tenant_id: str,
    request_id: str,
    acl_reference: str = "",
) -> Any | None:
    """Build a fail-closed GraphEvidence object from a strict L1/L2 web evidence record."""
    from services.knowledge_graph import EvidenceFragment, GraphEvidence
    from services.memory_scope import valid_internal_id

    data = (
        record_or_dict.to_dict(include_full_content=True)
        if hasattr(record_or_dict, "to_dict")
        else dict(record_or_dict) if isinstance(record_or_dict, dict) else {}
    )
    if not data:
        return None
    source_id = str(data.get("source_id") or "").strip()
    if not valid_internal_id(source_id) or not valid_internal_id(request_id) or not valid_internal_id(tenant_id):
        return None
    record_tenant = str(data.get("tenant_id") or tenant_id).strip() or tenant_id
    if record_tenant != tenant_id or str(data.get("source_origin") or "web") != "web":
        return None

    meta = (data.get("provenance_meta") or {}).get("web_extraction")
    if not isinstance(meta, dict):
        return None
    tier = meta.get("tier")
    adapter = meta.get("adapter")
    if (tier, adapter) not in (("L1", "static"), ("L2", "reader_http"), ("L2", "playwright")):
        return None
    if (
        meta.get("sanitizer_version") != WEB_SANITIZER_VERSION
        or meta.get("quality_version") != QUALITY_VERSION
        or meta.get("policy_version") != POLICY_VERSION
    ):
        return None
    artifact_version = meta.get("artifact_version")
    if type(artifact_version) is not int or artifact_version <= 0:
        return None
    if bool(data.get("isolated")) or bool(meta.get("isolated")) or meta.get("fact_eligible", True) is not True:
        return None
    url = str(data.get("url") or "").strip()
    if url:
        try:
            normalize_web_url(url)
        except GatewayError:
            return None

    full_text = str(data.get("full_content") or "")
    multimodal_blocks = data.get("multimodal_blocks") or []
    flags = set(meta.get("quality_flags") or ())
    if flags & {"snippet_only", "link_only", "visibility_uncertain"}:
        return None

    fragments: list[EvidenceFragment] = []
    seen_chunks: set[str] = set()
    for anchor in meta.get("text_anchors") or ():
        if not isinstance(anchor, dict):
            continue
        anchor_id = str(anchor.get("anchor_id") or "").strip()
        start, end = anchor.get("start"), anchor.get("end")
        if (
            not valid_internal_id(anchor_id) or anchor_id in seen_chunks
            or type(start) is not int or type(end) is not int or not (0 <= start < end <= len(full_text))
        ):
            continue
        cleaned = sanitize_web_text(full_text[start:end], max_chars=50_000)
        if cleaned:
            seen_chunks.add(anchor_id)
            fragments.append(EvidenceFragment(chunk_id=anchor_id, anchor=anchor_id, text=cleaned))
            if len(fragments) >= 256:
                break

    for block in multimodal_blocks:
        if len(fragments) >= 256 or not isinstance(block, dict) or block.get("modality") != "table":
            continue
        anchor_id = str(block.get("anchor_id") or "").strip()
        if not valid_internal_id(anchor_id) or anchor_id in seen_chunks:
            continue
        payload = block.get("structured_payload") or block.get("text") or ""
        raw_table = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if isinstance(payload, dict) else str(payload)
        cleaned = sanitize_web_text(raw_table, max_chars=50_000)
        if cleaned:
            seen_chunks.add(anchor_id)
            fragments.append(EvidenceFragment(chunk_id=anchor_id, anchor=anchor_id, text=cleaned))

    if not fragments and full_text.strip():
        cleaned = sanitize_web_text(full_text, max_chars=50_000)
        if cleaned:
            fragments.append(EvidenceFragment(chunk_id="text-p1", anchor="text-p1", text=cleaned))

    if not fragments:
        return None
    return GraphEvidence(
        tenant_id=tenant_id,
        request_id=request_id,
        source_id=source_id,
        artifact_version=str(artifact_version),
        fragments=tuple(fragments),
        level=str(tier),
        trusted=True,
        visible=True,
        facts_eligible=True,
        sanitizer_version=WEB_SANITIZER_VERSION,
        isolated=False,
        origin="web",
        acl_reference=acl_reference,
    )

