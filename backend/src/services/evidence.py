"""Evidence storage, lookup, and tool helpers for grounded research outputs."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
from metrics import RequestTrace
from services.search import dispatch_search
from utils import truncate_text

try:  # pragma: no cover - optional dependency
    from markdownify import markdownify
except Exception:  # pragma: no cover - optional dependency
    markdownify = None  # type: ignore

TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}
CITATION_PATTERN = re.compile(r"\[(T\d+-S\d+)\]")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_SCRIPT_PATTERN = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
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


def _extract_title(html: str) -> str:
    if not html:
        return ""
    match = HTML_TITLE_PATTERN.search(html)
    if not match:
        return ""
    return _clean_text(match.group(1))


def _fetch_page_text(url: str, timeout_seconds: float) -> tuple[str, str]:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    html = response.text or ""
    title = _extract_title(html)

    if markdownify is not None:
        try:
            return title, _clean_text(markdownify(html))
        except Exception:  # pragma: no cover - optional dependency failure
            pass

    return title, _clean_text(html)


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
    domain = extract_domain(_ev_str(ev, "url"))
    url = _ev_str(ev, "url").lower()
    coarse_type = _ev_str(ev, "source_type")
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
    }


def compute_source_reliability_score(ev: Any) -> float:
    components = _quality_components(ev)
    score = 3.0
    score += SOURCE_TYPE_BONUS[components["source_type_v2"]]
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
    if label == "high" and (
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
            "has_full_content": bool(self.full_content),
        }
        if include_full_content and self.full_content:
            payload["full_content"] = truncate_text(self.full_content, excerpt_limit * 3)
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


class EvidenceStore:
    """Thread-safe in-memory evidence store scoped to a single request."""

    def __init__(self, *, freshness_reference_days: int = 365) -> None:
        self._lock = Lock()
        self._freshness_reference_days = max(1, int(freshness_reference_days or 365))
        self._records_by_id: dict[str, EvidenceRecord] = {}
        self._task_source_ids: dict[int, list[str]] = {}
        self._task_url_index: dict[int, dict[str, str]] = {}

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
        with self._lock:
            task_ids = self._task_source_ids.setdefault(task_id, [])
            url_index = self._task_url_index.setdefault(task_id, {})
            next_index = len(task_ids) + 1

            for result in results:
                if not isinstance(result, dict):
                    continue

                url = str(result.get("url") or "").strip()
                title = str(result.get("title") or url or f"Source {next_index}").strip()
                snippet = str(result.get("content") or "").strip()
                raw_content = str(result.get("raw_content") or "").strip()
                published_at = _extract_published_at(result)
                source_updated_at = _extract_source_updated_at(result)
                source_type = _classify_source_type(url, title)
                dedup_key = _normalize_url(url) or f"title::{title.casefold()}"
                existing_id = url_index.get(dedup_key)
                if existing_id:
                    record = self._records_by_id[existing_id]
                    if len(snippet) > len(record.snippet):
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
                _apply_quality_metadata(record)
                self._records_by_id[source_id] = record
                task_ids.append(source_id)
                url_index[dedup_key] = source_id

        return self.list_task_evidence(task_id)

    def update_full_content(
        self,
        *,
        task_id: int,
        source_id: str | None,
        url: str,
        title: str,
        full_content: str,
    ) -> dict[str, Any]:
        """Attach fetched page content to an existing or new evidence record."""

        normalized_url = _normalize_url(url)
        with self._lock:
            task_ids = self._task_source_ids.setdefault(task_id, [])
            url_index = self._task_url_index.setdefault(task_id, {})

            record: EvidenceRecord | None = None
            if source_id:
                record = self._records_by_id.get(source_id)
            if record is None and normalized_url:
                existing_id = url_index.get(normalized_url)
                if existing_id:
                    record = self._records_by_id.get(existing_id)

            if record is None:
                source_id = source_id or f"T{task_id}-S{len(task_ids) + 1}"
                record = EvidenceRecord(
                    source_id=source_id,
                    task_id=task_id,
                    query="",
                    title=title or url,
                    url=url,
                    snippet=truncate_text(full_content, 600),
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

            if title and len(title) > len(record.title):
                record.title = title
            if full_content:
                record.full_content = full_content
                if not record.snippet:
                    record.snippet = truncate_text(full_content, 600)
            if not record.source_type or record.source_type == "web":
                record.source_type = _classify_source_type(url, record.title)
            _apply_quality_metadata(record)
            record.updated_at = time.time()

            return record.to_dict(include_full_content=True)

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
        if source_ids:
            items = [
                self.get_evidence(source_id, include_full_content=include_full_content)
                for source_id in source_ids
            ]
            return [item for item in items if item is not None]

        if task_id is not None:
            return self.list_task_evidence(
                task_id,
                include_full_content=include_full_content,
                limit=limit,
            )

        with self._lock:
            source_ids = sorted(self._records_by_id.keys())
        if limit is not None:
            source_ids = source_ids[: max(limit, 0)]
        items: list[dict[str, Any]] = []
        for source_id in source_ids:
            item = self.get_evidence(source_id, include_full_content=include_full_content)
            if item is not None:
                items.append(item)
        return items

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
            references.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "url": url,
                    "domain": str(item.get("domain") or ""),
                    "published_at": str(item.get("published_at") or ""),
                }
            )
        return references

    def hydrate_from_tasks(self, tasks: list[Any]) -> None:
        """Rebuild the in-memory indices from persisted task evidence payloads."""

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
                    if not source_id or source_id in self._records_by_id:
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
                        raw_content="",
                        full_content="",
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
                        created_at=time.time(),
                        updated_at=time.time(),
                    )
                    _apply_quality_metadata(record)
                    self._records_by_id[source_id] = record
                    task_ids.append(source_id)
                    dedup_key = _normalize_url(url) or f"title::{title.casefold()}"
                    url_index[dedup_key] = source_id


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
        body = truncate_text(body, config.resolved_max_tokens_per_source() * 4)
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
    return truncate_text(context, config.resolved_task_context_char_limit())


def render_references(reference_items: list[dict[str, str]]) -> str:
    """Render a standard reference section."""

    if not reference_items:
        return "- 暂无可用来源"
    rendered_lines: list[str] = []
    for item in reference_items:
        source_id = str(item.get("source_id") or "").strip()
        title = str(item.get("title") or "").strip() or source_id or "来源"
        url = str(item.get("url") or "").strip()
        line = f"- [{source_id}] {title}" if source_id else f"- {title}"
        if url:
            line = f"{line} - {url}"
        rendered_lines.append(line)
    return "\n".join(rendered_lines)


class SearchWebTool(Tool):
    """Tool wrapper around the repo's structured web-search dispatcher."""

    def __init__(
        self,
        *,
        config: Configuration,
        evidence_store: EvidenceStore,
        observer_getter: Callable[[], RequestTrace | None],
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
        config = self._config.model_copy(update=overrides) if overrides else self._config

        payload, notices, answer_text, backend_label, cache_hit, cache_strategy = dispatch_search(
            query,
            config,
            0,
            observer=self._observer_getter(),
            cache_context={
                "research_topic": parameters.get("research_topic"),
                "task_title": parameters.get("task_title"),
                "task_intent": parameters.get("task_intent"),
            },
            max_results=int(parameters.get("max_results") or 5),
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
                "evidence": evidence_items,
            },
            ensure_ascii=False,
        )


class FetchPageTool(Tool):
    """Fetch and normalize a specific page to enrich existing evidence."""

    def __init__(self, *, evidence_store: EvidenceStore, timeout_seconds: float = 10.0) -> None:
        super().__init__(
            name="fetch_page",
            description=(
                "抓取单个网页正文并回填到证据库。参数建议使用 JSON，包含 task_id、source_id、url。"
            ),
        )
        self._evidence_store = evidence_store
        self._timeout_seconds = timeout_seconds

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

        title, content = _fetch_page_text(url, timeout_seconds=self._timeout_seconds)
        item = self._evidence_store.update_full_content(
            task_id=task_id,
            source_id=source_id,
            url=url,
            title=title,
            full_content=content,
        )
        return json.dumps(item, ensure_ascii=False)


class EvidenceLookupTool(Tool):
    """Expose request-local evidence records to downstream agents."""

    def __init__(self, *, evidence_store: EvidenceStore) -> None:
        super().__init__(
            name="evidence_lookup",
            description=(
                "查询当前请求的证据库。参数建议使用 JSON，可按 task_id 或 source_id/source_ids 查询。"
            ),
        )
        self._evidence_store = evidence_store

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task_id", type="integer", description="任务 ID", required=False),
            ToolParameter(name="source_id", type="string", description="单个来源 ID", required=False),
            ToolParameter(name="source_ids", type="string", description="多个来源 ID", required=False),
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
        include_full_content = bool(parameters.get("include_full_content", False))
        limit = int(parameters.get("limit") or 0) or None

        evidence = self._evidence_store.lookup(
            task_id=task_id,
            source_ids=normalized_source_ids or None,
            include_full_content=include_full_content,
            limit=limit,
        )
        return json.dumps({"evidence": evidence}, ensure_ascii=False)
