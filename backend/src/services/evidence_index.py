"""Runtime and persistent evidence retrieval helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from config import Configuration
from metrics import RequestTrace, metrics_registry
from services.embeddings import embeddings_available, encode_text, encode_texts

try:  # pragma: no cover - optional dependency
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_COLLECTION_NAME = "deep_research_evidence_v1"
_DEFAULT_QUERY_MULTIPLIER = 4
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


class EvidenceStoreListener(Protocol):
    """Observer interface used by EvidenceStore write paths."""

    def on_records_changed(
        self,
        *,
        task_id: int,
        records: list[Any],
        reason: str,
        request_id: str | None = None,
    ) -> None:
        """Observe normalized evidence records after store mutation."""


@dataclass
class EvidenceChunk:
    """Normalized evidence chunk used by runtime and archive retrieval."""

    chunk_id: str
    task_id: int
    source_id: str
    text: str
    start_char: int
    end_char: int
    has_full_content: bool
    origin: str
    title: str = ""
    url: str = ""
    domain: str = ""
    snippet: str = ""
    quality_label: str = "medium"
    freshness_label: str = "unknown"
    published_at: str | None = None
    archive_doc_key: str | None = None
    archive_version_id: str | None = None


@dataclass
class EvidenceRetrievalHit:
    """Retrieval hit returned to callers and tools."""

    origin: str
    citation_eligible: bool
    task_id: int
    source_id: str
    archive_doc_key: str | None
    archive_version_id: str | None
    chunk_id: str
    text: str
    url: str
    title: str
    domain: str
    quality_label: str
    freshness_label: str
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "citation_eligible": self.citation_eligible,
            "task_id": self.task_id,
            "source_id": self.source_id,
            "archive_doc_key": self.archive_doc_key,
            "archive_version_id": self.archive_version_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "quality_label": self.quality_label,
            "freshness_label": self.freshness_label,
            "score": round(float(self.score), 6),
            "score_breakdown": {
                key: round(float(value), 6)
                for key, value in (self.score_breakdown or {}).items()
            },
        }


@dataclass
class EvidenceQueryResult:
    """Unified retrieval result returned by EvidenceRetrievalService."""

    hits: list[EvidenceRetrievalHit]
    notices: list[str] = field(default_factory=list)
    mode: str = "grounding"
    scope: str = "current"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "mode": self.mode,
            "notices": list(self.notices),
            "hits": [hit.to_dict() for hit in self.hits],
        }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split()).strip()


def _normalized_signal_text(value: Any) -> str:
    cleaned = _clean_text(value).lower()
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", cleaned).strip()


def _tokenize(value: Any) -> set[str]:
    cleaned = _normalized_signal_text(value)
    ascii_tokens = set(_WORD_PATTERN.findall(cleaned))
    cjk_tokens = {char for char in cleaned if _CJK_PATTERN.fullmatch(char)}
    return ascii_tokens | cjk_tokens


def _lexical_similarity(left: Any, right: Any) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap <= 0:
        return 0.0
    return overlap / math.sqrt(len(left_tokens) * len(right_tokens))


def _dot_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None or len(left) != len(right):
        return 0.0
    return max(0.0, min(1.0, sum(float(a) * float(b) for a, b in zip(left, right))))


def _quality_prior(label: str) -> float:
    normalized = str(label or "").strip().lower()
    if normalized == "high":
        return 1.0
    if normalized == "medium":
        return 0.6
    if normalized == "low":
        return 0.3
    return 0.5


def _freshness_prior(label: str) -> float:
    normalized = str(label or "").strip().lower()
    if normalized in {"fresh", "current"}:
        return 1.0
    if normalized == "recent":
        return 0.75
    if normalized == "stale":
        return 0.35
    return 0.5


def _normalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "/").rstrip("/") or "/"
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_PARAMS and not key.lower().startswith("utm_")
    ]
    return urlunsplit((scheme, netloc, path, urlencode(filtered_query, doseq=True), ""))


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _chunk_text(
    text: str,
    *,
    max_chars: int,
    overlap: int,
    max_chunks: int = 12,
) -> list[tuple[int, int, str]]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) < 400:
        return [(0, len(cleaned), cleaned)]

    normalized_max = max(200, int(max_chars or 900))
    normalized_overlap = max(0, min(int(overlap or 0), normalized_max // 2))
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(cleaned) and len(chunks) < max_chunks:
        end = min(len(cleaned), start + normalized_max)
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start + int(normalized_max * 0.65), end)
            if boundary > start + int(normalized_max * 0.5):
                end = boundary
        segment = cleaned[start:end].strip()
        if segment:
            chunks.append((start, end, segment))
        if end >= len(cleaned):
            break
        start = max(start + 1, end - normalized_overlap)
    return chunks or [(0, len(cleaned), cleaned)]


def _record_text(record: Any) -> str:
    for key in ("full_content", "raw_content", "snippet"):
        value = _clean_text(getattr(record, key, ""))
        if value:
            return value
    return ""


def _record_has_full_content(record: Any) -> bool:
    return bool(_clean_text(getattr(record, "full_content", "")))


def _record_archive_doc_key(record: Any) -> str | None:
    value = _normalize_text(getattr(record, "archive_doc_key", ""))
    return value or None


def _record_archive_version_id(record: Any) -> str | None:
    value = _normalize_text(getattr(record, "archive_version_id", ""))
    return value or None


class EvidenceRuntimeIndex(EvidenceStoreListener):
    """Request-scoped evidence chunk index used for current grounding."""

    def __init__(
        self,
        config: Configuration,
        *,
        archive_loader: Any | None = None,
    ) -> None:
        self._config = config
        self._archive_loader = archive_loader
        self._lock = Lock()
        self._chunks_by_task: dict[int, list[EvidenceChunk]] = {}
        self._chunk_embeddings: dict[str, list[float] | None] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._config.evidence_runtime_enabled)

    def on_records_changed(
        self,
        *,
        task_id: int,
        records: list[Any],
        reason: str,
        request_id: str | None = None,
    ) -> None:
        del request_id
        if not self.enabled or reason not in {"search", "fetch", "hydrate"}:
            return

        source_chunks: dict[str, list[EvidenceChunk]] = {}
        texts: list[str] = []
        chunk_ids: list[str] = []
        for record in records:
            chunks = self._chunks_for_record(task_id, record, reason=reason)
            if not chunks:
                continue
            source_chunks[str(getattr(record, "source_id", ""))] = chunks
            for chunk in chunks:
                texts.append(chunk.text)
                chunk_ids.append(chunk.chunk_id)

        embeddings: list[list[float] | None] = [None for _ in texts]
        if texts:
            try:
                embeddings = encode_texts(
                    texts,
                    model_name=self._config.resolved_evidence_runtime_embedding_model(),
                )
            except Exception:
                embeddings = [None for _ in texts]

        embedding_map = {
            chunk_id: embedding
            for chunk_id, embedding in zip(chunk_ids, embeddings)
        }

        with self._lock:
            existing = list(self._chunks_by_task.get(task_id, []))
            by_source: dict[str, list[EvidenceChunk]] = {}
            for chunk in existing:
                by_source.setdefault(chunk.source_id, []).append(chunk)

            if reason == "hydrate":
                by_source = {}

            for source_id, chunks in source_chunks.items():
                stale = by_source.pop(source_id, [])
                for chunk in stale:
                    self._chunk_embeddings.pop(chunk.chunk_id, None)
                by_source[source_id] = chunks

            flattened: list[EvidenceChunk] = []
            for chunks in by_source.values():
                flattened.extend(chunks)
                for chunk in chunks:
                    self._chunk_embeddings[chunk.chunk_id] = embedding_map.get(chunk.chunk_id)
            self._chunks_by_task[task_id] = flattened

    def _chunks_for_record(
        self,
        task_id: int,
        record: Any,
        *,
        reason: str,
    ) -> list[EvidenceChunk]:
        archive_version_id = _record_archive_version_id(record)
        if reason == "hydrate" and archive_version_id and self._archive_loader is not None:
            restored = self._archive_loader.load_runtime_chunks(
                archive_version_id,
                task_id=task_id,
                source_id=str(getattr(record, "source_id", "") or ""),
            )
            if restored:
                return restored

        source_id = str(getattr(record, "source_id", "") or "").strip()
        text = _record_text(record)
        if not source_id or not text:
            return []

        has_full_content = _record_has_full_content(record)
        windows = _chunk_text(
            text,
            max_chars=self._config.evidence_chunk_chars,
            overlap=self._config.evidence_chunk_overlap,
        )
        chunks: list[EvidenceChunk] = []
        for index, (start, end, window) in enumerate(windows, start=1):
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{source_id}::chunk::{index}",
                    task_id=task_id,
                    source_id=source_id,
                    text=window,
                    start_char=start,
                    end_char=end,
                    has_full_content=has_full_content,
                    origin="current",
                    title=_normalize_text(getattr(record, "title", "")),
                    url=_normalize_text(getattr(record, "url", "")),
                    domain=_normalize_text(getattr(record, "domain", "")),
                    snippet=_clean_text(getattr(record, "snippet", "")),
                    quality_label=_normalize_text(getattr(record, "quality_label", "")) or "medium",
                    freshness_label=_normalize_text(getattr(record, "freshness_label", "")) or "unknown",
                    published_at=_normalize_text(getattr(record, "published_at", "")) or None,
                    archive_doc_key=_record_archive_doc_key(record),
                    archive_version_id=archive_version_id,
                )
            )
        return chunks

    def query(
        self,
        *,
        text: str,
        task_id: int,
        top_k: int,
        per_source_limit: int = 2,
        max_sources: int = 4,
    ) -> list[EvidenceRetrievalHit]:
        if not self.enabled:
            return []

        normalized_query = _clean_text(text)
        if not normalized_query or task_id <= 0:
            return []

        with self._lock:
            chunks = list(self._chunks_by_task.get(task_id, []))
            embeddings = {
                chunk.chunk_id: self._chunk_embeddings.get(chunk.chunk_id)
                for chunk in chunks
            }
        if not chunks:
            return []

        query_embedding: list[float] | None = None
        try:
            query_embedding = encode_text(
                normalized_query,
                model_name=self._config.resolved_evidence_runtime_embedding_model(),
            )
        except Exception:
            query_embedding = None

        scored = self._score_chunks(
            chunks,
            embeddings=embeddings,
            query_text=normalized_query,
            query_embedding=query_embedding,
            citation_eligible=True,
        )
        return self._select_hits(
            scored,
            top_k=top_k,
            per_source_limit=per_source_limit,
            max_sources=max_sources,
        )

    def _score_chunks(
        self,
        chunks: list[EvidenceChunk],
        *,
        embeddings: dict[str, list[float] | None],
        query_text: str,
        query_embedding: list[float] | None,
        citation_eligible: bool,
    ) -> list[tuple[EvidenceChunk, float, dict[str, float]]]:
        scored: list[tuple[EvidenceChunk, float, dict[str, float]]] = []
        for chunk in chunks:
            dense = _dot_similarity(query_embedding, embeddings.get(chunk.chunk_id))
            lexical = _lexical_similarity(query_text, chunk.text)
            quality = _quality_prior(chunk.quality_label)
            freshness = _freshness_prior(chunk.freshness_label)
            score = (0.60 * dense) + (0.25 * lexical) + (0.10 * quality) + (0.05 * freshness)
            if score <= 0:
                continue
            scored.append(
                (
                    chunk,
                    score,
                    {
                        "dense": dense,
                        "lexical": lexical,
                        "quality": quality,
                        "freshness": freshness,
                        "citation_eligible": 1.0 if citation_eligible else 0.0,
                    },
                )
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _select_hits(
        self,
        scored: list[tuple[EvidenceChunk, float, dict[str, float]]],
        *,
        top_k: int,
        per_source_limit: int,
        max_sources: int,
        citation_eligible: bool = True,
    ) -> list[EvidenceRetrievalHit]:
        remaining = list(scored)
        selected: list[EvidenceRetrievalHit] = []
        source_counts: dict[str, int] = {}
        source_domains: dict[str, str] = {}
        while remaining and len(selected) < max(1, int(top_k or 1)):
            best_index = -1
            best_score = -1.0
            best_payload: tuple[EvidenceChunk, float, dict[str, float]] | None = None
            for index, payload in enumerate(remaining):
                chunk, base_score, breakdown = payload
                if source_counts.get(chunk.source_id, 0) >= max(1, int(per_source_limit or 1)):
                    continue
                if (
                    chunk.source_id not in source_counts
                    and len(source_counts) >= max(1, int(max_sources or 1))
                ):
                    continue

                score = base_score
                breakdown = dict(breakdown)
                source_penalty = 0.05 if chunk.source_id in source_counts else 0.0
                domain_penalty = 0.08 if chunk.domain and chunk.domain in source_domains.values() else 0.0
                score -= source_penalty + domain_penalty
                breakdown["source_penalty"] = -source_penalty
                breakdown["domain_penalty"] = -domain_penalty
                if score > best_score:
                    best_index = index
                    best_score = score
                    best_payload = (chunk, score, breakdown)

            if best_payload is None or best_index < 0:
                break

            chunk, score, breakdown = best_payload
            selected.append(
                EvidenceRetrievalHit(
                    origin=chunk.origin,
                    citation_eligible=citation_eligible,
                    task_id=chunk.task_id,
                    source_id=chunk.source_id,
                    archive_doc_key=chunk.archive_doc_key,
                    archive_version_id=chunk.archive_version_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    url=chunk.url,
                    title=chunk.title,
                    domain=chunk.domain,
                    quality_label=chunk.quality_label,
                    freshness_label=chunk.freshness_label,
                    score=score,
                    score_breakdown=breakdown,
                )
            )
            source_counts[chunk.source_id] = source_counts.get(chunk.source_id, 0) + 1
            if chunk.domain:
                source_domains[chunk.source_id] = chunk.domain
            remaining.pop(best_index)
        return selected


class EvidenceMemoryService(EvidenceStoreListener):
    """Persistent cross-request archive for evidence chunks."""

    def __init__(
        self,
        config: Configuration,
        *,
        client: Any | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._collection: Any | None = None
        self._lock = Lock()
        self._memory_dir = Path(self._config.evidence_memory_dir)
        self._manifest_path = self._memory_dir / "manifest.json"
        self._docs_dir = self._memory_dir / "docs"
        self._collection_dir = self._memory_dir / "chromadb"

    @property
    def enabled(self) -> bool:
        return bool(self._config.evidence_memory_enabled)

    def on_records_changed(
        self,
        *,
        task_id: int,
        records: list[Any],
        reason: str,
        request_id: str | None = None,
    ) -> None:
        del task_id
        if not self.enabled or reason not in {"search", "fetch"}:
            return

        with self._lock:
            manifest = self._load_manifest()
            collection = self._get_collection()
            changed = False
            for record in records:
                if self._ingest_record_locked(
                    record,
                    manifest=manifest,
                    collection=collection,
                    request_id=request_id,
                ):
                    changed = True
            if changed:
                self._write_manifest(manifest)

    def _ingest_record_locked(
        self,
        record: Any,
        *,
        manifest: dict[str, Any],
        collection: Any | None,
        request_id: str | None,
    ) -> bool:
        normalized_url = _normalize_url(getattr(record, "url", ""))
        text = _record_text(record)
        if not normalized_url or not text:
            return False

        canonical_doc_key = _sha256_text(normalized_url)
        content_hash = _sha256_text(text)
        doc_version_id = _sha256_text(f"{normalized_url}\n{content_hash}")
        has_full_content = _record_has_full_content(record)
        observed_at = time()

        doc_entry = manifest.setdefault("docs", {}).setdefault(
            canonical_doc_key,
            {
                "canonical_doc_key": canonical_doc_key,
                "normalized_url": normalized_url,
                "url": normalized_url,
                "latest_version_id": "",
                "has_full_content": False,
                "version_ids": [],
            },
        )
        versions = manifest.setdefault("versions", {})
        version_entry = versions.get(doc_version_id)

        provenance = {
            "request_id": _normalize_text(request_id),
            "task_id": int(getattr(record, "task_id", 0) or 0),
            "source_id": _normalize_text(getattr(record, "source_id", "")),
            "query": _normalize_text(getattr(record, "query", "")),
            "backend": _normalize_text(getattr(record, "backend", "")),
            "observed_at": observed_at,
        }

        setattr(record, "archive_doc_key", canonical_doc_key)
        setattr(record, "archive_version_id", doc_version_id)
        setattr(record, "archive_has_full_content", has_full_content)

        if version_entry is not None:
            metrics_registry.increment("evidence_archive_dedup_total")
            payload = self._load_doc_payload(doc_version_id)
            payload.setdefault("provenance", [])
            self._append_provenance(payload["provenance"], provenance)
            self._write_doc_payload(doc_version_id, payload)
            version_entry["updated_at"] = observed_at
            if has_full_content and not version_entry.get("has_full_content"):
                version_entry["has_full_content"] = True
            self._update_doc_entry(doc_entry, doc_version_id=doc_version_id, has_full_content=has_full_content)
            return True

        metrics_registry.increment("evidence_archive_ingest_total")
        windows = _chunk_text(
            text,
            max_chars=self._config.evidence_chunk_chars,
            overlap=self._config.evidence_chunk_overlap,
        )
        archive_chunks = [
            EvidenceChunk(
                chunk_id=f"{doc_version_id}::chunk::{index}",
                task_id=int(getattr(record, "task_id", 0) or 0),
                source_id=_normalize_text(getattr(record, "source_id", "")),
                text=window,
                start_char=start,
                end_char=end,
                has_full_content=has_full_content,
                origin="history",
                title=_normalize_text(getattr(record, "title", "")),
                url=normalized_url,
                domain=_normalize_text(getattr(record, "domain", "")),
                snippet=_clean_text(getattr(record, "snippet", "")),
                quality_label=_normalize_text(getattr(record, "quality_label", "")) or "medium",
                freshness_label=_normalize_text(getattr(record, "freshness_label", "")) or "unknown",
                published_at=_normalize_text(getattr(record, "published_at", "")) or None,
                archive_doc_key=canonical_doc_key,
                archive_version_id=doc_version_id,
            )
            for index, (start, end, window) in enumerate(windows, start=1)
        ]
        if collection is not None and archive_chunks:
            try:
                embeddings = encode_texts(
                    [chunk.text for chunk in archive_chunks],
                    model_name=self._config.resolved_evidence_memory_embedding_model(),
                )
                valid = [
                    (chunk, embedding)
                    for chunk, embedding in zip(archive_chunks, embeddings)
                    if embedding is not None
                ]
                if valid:
                    collection.upsert(
                        ids=[chunk.chunk_id for chunk, _ in valid],
                        documents=[chunk.text for chunk, _ in valid],
                        metadatas=[
                            {
                                "chunk_id": chunk.chunk_id,
                                "canonical_doc_key": canonical_doc_key,
                                "doc_version_id": doc_version_id,
                                "url": normalized_url,
                                "title": chunk.title,
                                "domain": chunk.domain,
                                "quality_label": chunk.quality_label,
                                "freshness_label": chunk.freshness_label,
                                "has_full_content": has_full_content,
                                "request_id": _normalize_text(request_id),
                            }
                            for chunk, _ in valid
                        ],
                        embeddings=[embedding for _, embedding in valid],
                    )
            except Exception:
                logger.debug("evidence archive vector upsert failed doc_version_id=%s", doc_version_id)

        payload = {
            "doc": {
                "canonical_doc_key": canonical_doc_key,
                "doc_version_id": doc_version_id,
                "normalized_url": normalized_url,
                "title": _normalize_text(getattr(record, "title", "")) or normalized_url,
                "url": normalized_url,
                "domain": _normalize_text(getattr(record, "domain", "")),
                "quality_label": _normalize_text(getattr(record, "quality_label", "")) or "medium",
                "freshness_label": _normalize_text(getattr(record, "freshness_label", "")) or "unknown",
                "published_at": _normalize_text(getattr(record, "published_at", "")) or None,
                "source_updated_at": _normalize_text(getattr(record, "source_updated_at", "")) or None,
                "content_hash": content_hash,
                "has_full_content": has_full_content,
                "created_at": observed_at,
                "updated_at": observed_at,
            },
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "has_full_content": chunk.has_full_content,
                }
                for chunk in archive_chunks
            ],
            "provenance": [provenance],
        }
        self._write_doc_payload(doc_version_id, payload)
        versions[doc_version_id] = {
            "canonical_doc_key": canonical_doc_key,
            "doc_version_id": doc_version_id,
            "content_hash": content_hash,
            "url": normalized_url,
            "title": payload["doc"]["title"],
            "domain": payload["doc"]["domain"],
            "quality_label": payload["doc"]["quality_label"],
            "freshness_label": payload["doc"]["freshness_label"],
            "published_at": payload["doc"]["published_at"],
            "source_updated_at": payload["doc"]["source_updated_at"],
            "has_full_content": has_full_content,
            "chunk_ids": [chunk.chunk_id for chunk in archive_chunks],
            "created_at": observed_at,
            "updated_at": observed_at,
        }
        self._update_doc_entry(doc_entry, doc_version_id=doc_version_id, has_full_content=has_full_content)
        return True

    def _update_doc_entry(
        self,
        doc_entry: dict[str, Any],
        *,
        doc_version_id: str,
        has_full_content: bool,
    ) -> None:
        version_ids = list(doc_entry.get("version_ids") or [])
        if doc_version_id not in version_ids:
            version_ids.append(doc_version_id)
        doc_entry["version_ids"] = version_ids
        if has_full_content or not str(doc_entry.get("latest_version_id") or "").strip():
            doc_entry["latest_version_id"] = doc_version_id
        doc_entry["has_full_content"] = bool(doc_entry.get("has_full_content")) or has_full_content

    def _append_provenance(self, items: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
        candidate = (
            provenance.get("request_id"),
            provenance.get("task_id"),
            provenance.get("source_id"),
            provenance.get("query"),
            provenance.get("backend"),
        )
        for item in items:
            current = (
                item.get("request_id"),
                item.get("task_id"),
                item.get("source_id"),
                item.get("query"),
                item.get("backend"),
            )
            if current == candidate:
                return
        items.append(provenance)

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "embedding_model": self._config.resolved_evidence_memory_embedding_model(),
            "docs": {},
            "versions": {},
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return self._empty_manifest()
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_manifest()
        if not isinstance(payload, dict):
            return self._empty_manifest()
        payload.setdefault("docs", {})
        payload.setdefault("versions", {})
        return payload

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_doc_payload(self, doc_version_id: str) -> dict[str, Any]:
        path = self._docs_dir / f"{doc_version_id}.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_doc_payload(self, doc_version_id: str, payload: dict[str, Any]) -> None:
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        (self._docs_dir / f"{doc_version_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_collection(self) -> Any | None:
        if self._collection is not None:
            return self._collection
        if chromadb is None or not embeddings_available():
            return None
        try:
            self._collection_dir.mkdir(parents=True, exist_ok=True)
            client = self._client or chromadb.PersistentClient(path=str(self._collection_dir))
            self._collection = client.get_or_create_collection(_COLLECTION_NAME)
        except Exception:
            logger.debug("evidence archive backend unavailable")
            self._collection = None
        return self._collection

    def load_runtime_chunks(
        self,
        archive_version_id: str,
        *,
        task_id: int,
        source_id: str,
    ) -> list[EvidenceChunk]:
        if not self.enabled or not archive_version_id:
            return []
        payload = self._load_doc_payload(archive_version_id)
        if not payload:
            metrics_registry.increment("evidence_archive_restore_miss_total")
            return []
        metrics_registry.increment("evidence_archive_restore_total")
        doc = payload.get("doc") if isinstance(payload.get("doc"), dict) else {}
        chunks = payload.get("chunks") if isinstance(payload.get("chunks"), list) else []
        restored: list[EvidenceChunk] = []
        for item in chunks:
            if not isinstance(item, dict):
                continue
            restored.append(
                EvidenceChunk(
                    chunk_id=f"{source_id}::{str(item.get('chunk_id') or '').split('::')[-1]}",
                    task_id=task_id,
                    source_id=source_id,
                    text=_clean_text(item.get("text")),
                    start_char=int(item.get("start_char") or 0),
                    end_char=int(item.get("end_char") or 0),
                    has_full_content=bool(item.get("has_full_content")),
                    origin="current",
                    title=_normalize_text(doc.get("title")),
                    url=_normalize_text(doc.get("url")),
                    domain=_normalize_text(doc.get("domain")),
                    quality_label=_normalize_text(doc.get("quality_label")) or "medium",
                    freshness_label=_normalize_text(doc.get("freshness_label")) or "unknown",
                    published_at=_normalize_text(doc.get("published_at")) or None,
                    archive_doc_key=_normalize_text(doc.get("canonical_doc_key")) or None,
                    archive_version_id=_normalize_text(doc.get("doc_version_id")) or archive_version_id,
                )
            )
        if not restored:
            metrics_registry.increment("evidence_archive_restore_miss_total")
        return restored

    def query(
        self,
        *,
        text: str,
        top_k: int,
        request_id_exclude: str | None = None,
        mode: str = "lead",
    ) -> list[EvidenceRetrievalHit]:
        if not self.enabled:
            return []
        normalized_query = _clean_text(text)
        if not normalized_query:
            return []

        metrics_registry.increment("evidence_archive_query_total")
        query_embedding: list[float] | None = None
        try:
            query_embedding = encode_text(
                normalized_query,
                model_name=self._config.resolved_evidence_memory_embedding_model(),
            )
        except Exception:
            query_embedding = None

        candidates = self._query_candidates(
            query_text=normalized_query,
            query_embedding=query_embedding,
            top_k=top_k,
            request_id_exclude=request_id_exclude,
            mode=mode,
        )
        scored: list[tuple[EvidenceChunk, float, dict[str, float]]] = []
        for chunk, dense_hint in candidates:
            dense = max(0.0, min(1.0, float(dense_hint or 0.0)))
            lexical = _lexical_similarity(normalized_query, chunk.text)
            quality = _quality_prior(chunk.quality_label)
            freshness = _freshness_prior(chunk.freshness_label)
            if mode == "repair":
                score = (0.45 * dense) + (0.20 * lexical) + (0.25 * quality) + (0.10 * freshness)
            else:
                score = (0.55 * dense) + (0.25 * lexical) + (0.15 * quality) + (0.05 * freshness)
            if score <= 0:
                continue
            scored.append(
                (
                    chunk,
                    score,
                    {
                        "dense": dense,
                        "lexical": lexical,
                        "quality": quality,
                        "freshness": freshness,
                        "citation_eligible": 0.0,
                    },
                )
            )
        runtime_index = EvidenceRuntimeIndex(self._config)
        return runtime_index._select_hits(
            scored,
            top_k=top_k,
            per_source_limit=2,
            max_sources=max(1, int(top_k or 1)),
            citation_eligible=False,
        )

    def _query_candidates(
        self,
        *,
        query_text: str,
        query_embedding: list[float] | None,
        top_k: int,
        request_id_exclude: str | None,
        mode: str,
    ) -> list[tuple[EvidenceChunk, float | None]]:
        collection = self._get_collection()
        candidates: list[tuple[EvidenceChunk, float | None]] = []
        manifest = self._load_manifest()
        full_version_ids = self._full_version_ids(manifest)
        if collection is not None and query_embedding is not None:
            try:
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(int(top_k or 1) * _DEFAULT_QUERY_MULTIPLIER, int(top_k or 1)),
                    include=["documents", "distances", "metadatas"],
                )
            except Exception:
                result = {}
            documents = (result.get("documents") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            for index, metadata in enumerate(metadatas):
                if not isinstance(metadata, dict):
                    continue
                doc_version_id = _normalize_text(metadata.get("doc_version_id"))
                if not doc_version_id:
                    continue
                if not self._version_allowed(manifest, doc_version_id, full_version_ids=full_version_ids):
                    continue
                payload = self._load_doc_payload(doc_version_id)
                chunk = self._chunk_from_payload(
                    payload,
                    chunk_id_hint=_normalize_text(metadata.get("chunk_id")) or "",
                    query_text=query_text,
                    request_id_exclude=request_id_exclude,
                    mode=mode,
                )
                if chunk is None:
                    continue
                distance = float((distances[index] if index < len(distances) else 1.0) or 1.0)
                dense_hint = max(0.0, 1.0 - distance)
                document = documents[index] if index < len(documents) else chunk.text
                candidates.append((chunk, dense_hint if document else None))
        if candidates:
            return candidates

        versions = manifest.get("versions") if isinstance(manifest.get("versions"), dict) else {}
        for doc_version_id, version in versions.items():
            if not isinstance(version, dict):
                continue
            if not self._version_allowed(manifest, str(doc_version_id), full_version_ids=full_version_ids):
                continue
            payload = self._load_doc_payload(str(doc_version_id))
            chunk = self._chunk_from_payload(
                payload,
                chunk_id_hint="",
                query_text=query_text,
                request_id_exclude=request_id_exclude,
                mode=mode,
            )
            if chunk is None:
                continue
            candidates.append((chunk, None))
        return candidates

    def _full_version_ids(self, manifest: dict[str, Any]) -> set[str]:
        docs = manifest.get("docs") if isinstance(manifest.get("docs"), dict) else {}
        versions = manifest.get("versions") if isinstance(manifest.get("versions"), dict) else {}
        full_versions: set[str] = set()
        for doc in docs.values():
            if not isinstance(doc, dict):
                continue
            version_ids = [str(item).strip() for item in doc.get("version_ids") or [] if str(item).strip()]
            doc_has_full = any(
                bool((versions.get(version_id) or {}).get("has_full_content"))
                for version_id in version_ids
            )
            if doc_has_full:
                full_versions.update(version_ids)
        return full_versions

    def _version_allowed(
        self,
        manifest: dict[str, Any],
        doc_version_id: str,
        *,
        full_version_ids: set[str],
    ) -> bool:
        versions = manifest.get("versions") if isinstance(manifest.get("versions"), dict) else {}
        version = versions.get(doc_version_id) if isinstance(versions, dict) else None
        if not isinstance(version, dict):
            return False
        if bool(version.get("has_full_content")):
            return True
        if doc_version_id in full_version_ids:
            return False
        return True

    def _chunk_from_payload(
        self,
        payload: dict[str, Any],
        *,
        chunk_id_hint: str,
        query_text: str,
        request_id_exclude: str | None,
        mode: str,
    ) -> EvidenceChunk | None:
        doc = payload.get("doc") if isinstance(payload.get("doc"), dict) else {}
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), list) else []
        if request_id_exclude and any(
            _normalize_text(item.get("request_id")) == _normalize_text(request_id_exclude)
            for item in provenance
            if isinstance(item, dict)
        ):
            return None

        chunks = payload.get("chunks") if isinstance(payload.get("chunks"), list) else []
        chosen = None
        if chunk_id_hint:
            for item in chunks:
                if isinstance(item, dict) and _normalize_text(item.get("chunk_id")) == chunk_id_hint:
                    chosen = item
                    break
        if chosen is None:
            best_score = -1.0
            for item in chunks:
                if not isinstance(item, dict):
                    continue
                score = _lexical_similarity(query_text, item.get("text"))
                if score > best_score:
                    best_score = score
                    chosen = item
        if not isinstance(chosen, dict):
            return None

        return EvidenceChunk(
            chunk_id=_normalize_text(chosen.get("chunk_id")) or _normalize_text(doc.get("doc_version_id")),
            task_id=int(
                next(
                    (
                        item.get("task_id")
                        for item in provenance
                        if isinstance(item, dict) and item.get("task_id") not in (None, "")
                    ),
                    0,
                )
                or 0
            ),
            source_id=_normalize_text(
                next(
                    (
                        item.get("source_id")
                        for item in provenance
                        if isinstance(item, dict) and _normalize_text(item.get("source_id"))
                    ),
                    "",
                )
            )
            or "",
            text=_clean_text(chosen.get("text")),
            start_char=int(chosen.get("start_char") or 0),
            end_char=int(chosen.get("end_char") or 0),
            has_full_content=bool(chosen.get("has_full_content")),
            origin="history",
            title=_normalize_text(doc.get("title")),
            url=_normalize_text(doc.get("url")),
            domain=_normalize_text(doc.get("domain")),
            quality_label=_normalize_text(doc.get("quality_label")) or "medium",
            freshness_label=_normalize_text(doc.get("freshness_label")) or "unknown",
            published_at=_normalize_text(doc.get("published_at")) or None,
            archive_doc_key=_normalize_text(doc.get("canonical_doc_key")) or None,
            archive_version_id=_normalize_text(doc.get("doc_version_id")) or None,
        )

class EvidenceRetrievalService:
    """Unified retrieval facade over runtime and persistent evidence stores."""

    def __init__(
        self,
        *,
        runtime_index: EvidenceRuntimeIndex | None,
        memory_service: EvidenceMemoryService | None,
        config: Configuration,
    ) -> None:
        self._runtime_index = runtime_index
        self._memory_service = memory_service
        self._config = config

    def query(
        self,
        text: str,
        *,
        task_id: int | None = None,
        scope: str = "current",
        mode: str = "grounding",
        top_k: int | None = None,
        request_id: str | None = None,
        observer: RequestTrace | None = None,
    ) -> EvidenceQueryResult:
        normalized_scope = str(scope or "current").strip().lower() or "current"
        if normalized_scope not in {"current", "history", "hybrid"}:
            normalized_scope = "current"
        normalized_mode = str(mode or "grounding").strip().lower() or "grounding"
        if normalized_mode not in {"grounding", "lead", "repair"}:
            normalized_mode = "grounding"

        notices: list[str] = []
        if normalized_mode == "grounding" and normalized_scope != "current":
            normalized_scope = "current"
            notices.append("grounding mode only returns current-request citeable chunks")

        requested_top_k = int(top_k or 0) or (
            self._config.evidence_runtime_top_k
            if normalized_mode == "grounding"
            else self._config.evidence_memory_top_k
        )
        hits: list[EvidenceRetrievalHit] = []

        if normalized_scope in {"current", "hybrid"} and task_id is not None and self._runtime_index is not None:
            current_hits = self._runtime_index.query(
                text=text,
                task_id=task_id,
                top_k=requested_top_k,
            )
            hits.extend(current_hits)

        if normalized_scope in {"history", "hybrid"} and normalized_mode in {"lead", "repair"} and self._memory_service is not None:
            history_hits = self._memory_service.query(
                text=text,
                top_k=requested_top_k,
                request_id_exclude=request_id,
                mode=normalized_mode,
            )
            hits.extend(history_hits)

        if normalized_mode == "repair":
            hits = self._rerank_repair_hits(hits)
        elif normalized_mode == "lead":
            hits.sort(key=lambda item: item.score, reverse=True)

        if observer and not hits:
            observer.record_degraded(f"evidence_retrieval_miss:{normalized_scope}:{normalized_mode}")
        return EvidenceQueryResult(
            hits=hits[:requested_top_k],
            notices=notices,
            mode=normalized_mode,
            scope=normalized_scope,
        )

    def _rerank_repair_hits(self, hits: list[EvidenceRetrievalHit]) -> list[EvidenceRetrievalHit]:
        reranked = list(hits)
        reranked.sort(
            key=lambda item: (
                0 if item.citation_eligible else 1,
                0 if item.quality_label == "high" else 1,
                0 if item.freshness_label in {"fresh", "current"} else 1,
                -item.score,
            )
        )
        return reranked
