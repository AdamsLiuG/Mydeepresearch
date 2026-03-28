"""Evidence storage, lookup, and tool helpers for grounded research outputs."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
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


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).netloc or "").lower()
    except ValueError:
        return ""


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
        "date",
        "datetime",
        "time",
        "updated_at",
    ):
        parsed = _parse_datetime(result.get(key))
        if parsed is not None:
            return parsed.date().isoformat()

    combined_text = " ".join(
        str(result.get(field) or "").strip()
        for field in ("title", "content", "raw_content")
        if str(result.get(field) or "").strip()
    )
    parsed = _parse_datetime(combined_text)
    return parsed.date().isoformat() if parsed is not None else None


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


def _score_quality(
    *,
    source_type: str,
    provider_count: int,
    published_at: str | None,
) -> tuple[int, str]:
    score = 3
    if source_type in {"government", "education", "paper", "official", "documentation"}:
        score += 4
    elif source_type in {"repository", "reference", "news"}:
        score += 2
    elif source_type in {"forum", "social"}:
        score -= 1

    score += min(max(int(provider_count or 1), 1), 3) - 1
    if published_at:
        score += 1

    if score >= 7:
        return score, "high"
    if score >= 4:
        return score, "medium"
    return score, "low"


def _freshness_metadata(
    published_at: str | None,
    *,
    freshness_reference_days: int = 365,
) -> tuple[int | None, str]:
    parsed = _parse_datetime(published_at)
    if parsed is None:
        return None, "unknown"

    age_days = max(int((_safe_now() - parsed).total_seconds() // 86400), 0)
    if age_days <= 30:
        label = "fresh"
    elif age_days <= max(90, freshness_reference_days // 4):
        label = "recent"
    elif age_days <= freshness_reference_days:
        label = "current"
    else:
        label = "stale"
    return age_days, label


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
    quality_score: int = 0
    quality_label: str = "medium"
    published_at: str | None = None
    freshness_days: int | None = None
    freshness_label: str = "unknown"
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
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "published_at": self.published_at,
            "freshness_days": self.freshness_days,
            "freshness_label": self.freshness_label,
            "has_full_content": bool(self.full_content),
        }
        if include_full_content and self.full_content:
            payload["full_content"] = truncate_text(self.full_content, excerpt_limit * 3)
        return payload


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
                source_type = _classify_source_type(url, title)
                quality_score, quality_label = _score_quality(
                    source_type=source_type,
                    provider_count=int(result.get("provider_count") or 1),
                    published_at=published_at,
                )
                freshness_days, freshness_label = _freshness_metadata(
                    published_at,
                    freshness_reference_days=self._freshness_reference_days,
                )
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
                    record.quality_score = max(quality_score, record.quality_score)
                    if quality_label == "high" or record.quality_label != "high":
                        record.quality_label = quality_label
                    record.published_at = published_at or record.published_at
                    record.freshness_days = (
                        freshness_days
                        if freshness_days is not None
                        else record.freshness_days
                    )
                    if freshness_label != "unknown" or record.freshness_label == "unknown":
                        record.freshness_label = freshness_label
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
                    quality_score=quality_score,
                    quality_label=quality_label,
                    published_at=published_at,
                    freshness_days=freshness_days,
                    freshness_label=freshness_label,
                    created_at=time.time(),
                    updated_at=time.time(),
                )
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
                record.quality_score, record.quality_label = _score_quality(
                    source_type=record.source_type,
                    provider_count=record.provider_count,
                    published_at=record.published_at,
                )
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
            record.quality_score, record.quality_label = _score_quality(
                source_type=record.source_type,
                provider_count=record.provider_count,
                published_at=record.published_at,
            )
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
                    freshness_days = item.get("freshness_days")
                    freshness_label = str(item.get("freshness_label") or "").strip() or "unknown"
                    quality_score = int(item.get("quality_score") or 0)
                    quality_label = str(item.get("quality_label") or "").strip()
                    if not quality_label:
                        quality_score, quality_label = _score_quality(
                            source_type=source_type,
                            provider_count=provider_count,
                            published_at=published_at,
                        )

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
                        quality_score=quality_score,
                        quality_label=quality_label,
                        published_at=published_at,
                        freshness_days=(
                            int(freshness_days)
                            if freshness_days is not None and str(freshness_days).strip()
                            else None
                        ),
                        freshness_label=freshness_label,
                        created_at=time.time(),
                        updated_at=time.time(),
                    )
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
