"""Persistent note retrieval memory backed by a local vector database."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from config import Configuration
from metrics import RequestTrace, metrics_registry
from services.embeddings import embeddings_available, encode_text, encode_texts
from utils import truncate_text

try:  # pragma: no cover - exercised through runtime fallback
    import chromadb
except Exception:  # pragma: no cover - exercised through runtime fallback
    chromadb = None

logger = logging.getLogger(__name__)

_ALLOWED_NOTE_TYPES = {"task_state", "conclusion"}
_STATUS_PRIORITY = {"success": 3, "partial_success": 2, "unknown": 1, "failed": 0, "in_progress": -1}
_SCHEMA_VERSION = 1
_CHUNKING_VERSION = 1
_DEFAULT_COLLECTION_NAME = "deep_research_notes_v1"
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120
_DEFAULT_QUERY_MULTIPLIER = 6
_FRONTMATTER_DELIMITER = "---"
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_status(value: Any) -> str:
    normalized = str(value or "").strip() or "unknown"
    return normalized if normalized in _STATUS_PRIORITY else "unknown"


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checksum_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _iter_chunk_windows(text: str, *, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]

    windows: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + size].strip()
        if chunk:
            windows.append(chunk)
        if start + size >= len(cleaned):
            break
    return windows


@dataclass
class NoteDocument:
    note_id: str
    title: str
    note_type: str
    tags: list[str]
    created_at: str | None
    updated_at: str | None
    note_path: str
    body: str
    checksum: str


@dataclass
class NoteChunk:
    chunk_id: str
    note_id: str
    note_type: str
    title: str
    tags: list[str]
    heading_path: str
    content: str
    note_path: str
    created_at: str | None
    updated_at: str | None
    resolved_topic: str | None
    resolved_request_id: str | None
    resolved_request_status: str
    resolved_task_id: int | None
    section_kind: str
    chunk_index: int

    def to_document(self) -> str:
        sections = [self.title, self.heading_path, self.content]
        return "\n".join(part for part in sections if str(part or "").strip())

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "note_id": self.note_id,
            "note_type": self.note_type,
            "title": self.title,
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "heading_path": self.heading_path,
            "content": self.content,
            "note_path": self.note_path,
            "created_at": self.created_at or "",
            "updated_at": self.updated_at or "",
            "resolved_topic": self.resolved_topic or "",
            "resolved_request_id": self.resolved_request_id or "",
            "resolved_request_status": self.resolved_request_status,
            "resolved_task_id": self.resolved_task_id if self.resolved_task_id is not None else -1,
            "section_kind": self.section_kind,
            "chunk_index": self.chunk_index,
        }


@dataclass
class MemoryHit:
    note_id: str
    note_type: str
    title: str
    heading_path: str
    content: str
    note_path: str
    resolved_topic: str | None
    resolved_request_id: str | None
    resolved_request_status: str
    resolved_task_id: int | None
    section_kind: str
    similarity: float
    score: float
    match_reason: str


class NoteProvenanceResolver:
    """Resolve note provenance from persisted request snapshots."""

    def __init__(self, request_state_dir: str) -> None:
        self._directory = Path(request_state_dir)

    def resolve(self) -> dict[str, dict[str, Any]]:
        if not self._directory.exists():
            return {}

        resolved: dict[str, dict[str, Any]] = {}
        for path in sorted(self._directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue

            request_id = str(payload.get("request_id") or "").strip() or None
            topic = str(payload.get("topic") or "").strip() or None
            status = _safe_status(payload.get("status"))
            updated_at = str(payload.get("updated_at") or "").strip() or None

            for item in payload.get("todo_items") or []:
                if not isinstance(item, dict):
                    continue
                note_id = str(item.get("note_id") or "").strip()
                if not note_id:
                    continue
                self._store_candidate(
                    resolved,
                    note_id,
                    {
                        "request_id": request_id,
                        "topic": topic,
                        "request_status": status,
                        "task_id": _safe_int(item.get("id")),
                        "updated_at": updated_at,
                    },
                )

            report_note_id = str(payload.get("report_note_id") or "").strip()
            if report_note_id:
                self._store_candidate(
                    resolved,
                    report_note_id,
                    {
                        "request_id": request_id,
                        "topic": topic,
                        "request_status": status,
                        "task_id": None,
                        "updated_at": updated_at,
                    },
                )

        return resolved

    @staticmethod
    def _store_candidate(
        resolved: dict[str, dict[str, Any]],
        note_id: str,
        candidate: dict[str, Any],
    ) -> None:
        existing = resolved.get(note_id)
        if existing is None:
            resolved[note_id] = candidate
            return

        existing_score = (
            _STATUS_PRIORITY.get(_safe_status(existing.get("request_status")), -1),
            str(existing.get("updated_at") or ""),
        )
        candidate_score = (
            _STATUS_PRIORITY.get(_safe_status(candidate.get("request_status")), -1),
            str(candidate.get("updated_at") or ""),
        )
        if candidate_score >= existing_score:
            resolved[note_id] = candidate


class NoteMemoryService:
    """Manage local note retrieval memory using chunked semantic search."""

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
        self._memory_dir = Path(self._config.note_memory_dir)
        self._manifest_path = self._memory_dir / "manifest.json"
        self._collection_dir = self._memory_dir / "chromadb"
        self._reconciled = False
        self._backend_unavailable_logged = False

    @property
    def enabled(self) -> bool:
        return bool(
            self._config.note_memory_enabled
            and self._config.enable_notes
            and str(self._config.notes_workspace or "").strip()
        )

    def ensure_reconciled(self, *, observer: RequestTrace | None = None) -> None:
        if not self.enabled:
            return

        with self._lock:
            if self._reconciled:
                return
            manifest = self._load_manifest()
            if self._can_reuse_existing_index_locked(manifest, observer=observer):
                self._reconciled = True
                return
            try:
                self._reconcile_locked(observer=observer)
            except Exception:
                metrics_registry.increment("note_memory_refresh_failed_total")
                raise
            self._reconciled = True

    def search_for_planning(
        self,
        research_topic: str,
        *,
        current_request_id: str | None = None,
        exclude_note_ids: set[str] | None = None,
        observer: RequestTrace | None = None,
    ) -> str:
        return self._search_and_render(
            query_text=research_topic,
            stage="planning",
            top_k=self._config.note_memory_planning_top_k,
            current_request_id=current_request_id,
            exclude_note_ids=exclude_note_ids,
            observer=observer,
        )

    def search_for_task(
        self,
        research_topic: str,
        task_title: str,
        task_intent: str,
        *,
        current_request_id: str | None = None,
        exclude_note_ids: set[str] | None = None,
        observer: RequestTrace | None = None,
    ) -> str:
        query_text = " ".join(
            part.strip()
            for part in [research_topic, task_title, task_intent]
            if str(part or "").strip()
        )
        return self._search_and_render(
            query_text=query_text,
            stage="execution",
            top_k=self._config.note_memory_execution_top_k,
            current_request_id=current_request_id,
            exclude_note_ids=exclude_note_ids,
            observer=observer,
        )

    def refresh_notes(
        self,
        note_ids: list[str] | set[str],
        *,
        observer: RequestTrace | None = None,
    ) -> None:
        if not self.enabled:
            return

        normalized_ids = sorted(
            {
                str(note_id or "").strip()
                for note_id in note_ids or []
                if str(note_id or "").strip()
            }
        )
        if not normalized_ids:
            return

        with self._lock:
            self._refresh_locked(normalized_ids, observer=observer)

    def _search_and_render(
        self,
        *,
        query_text: str,
        stage: str,
        top_k: int,
        current_request_id: str | None,
        exclude_note_ids: set[str] | None,
        observer: RequestTrace | None,
    ) -> str:
        if not self.enabled or not _normalize_text(query_text):
            return ""

        try:
            self.ensure_reconciled(observer=observer)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.warning("note memory reconcile failed stage=%s error=%s", stage, exc)
            if observer:
                observer.record_degraded("note_memory_reconcile_failed")
            return ""

        if self._collection is None:
            return ""

        query_embedding = self._encode_query(query_text, observer=observer)
        if query_embedding is None:
            return ""

        hits = self._query_hits(
            query_embedding=query_embedding,
            top_k=max(1, int(top_k or 1)),
            stage=stage,
            current_request_id=current_request_id,
            exclude_note_ids=exclude_note_ids or set(),
        )
        if observer:
            observer.record_note_memory_query(
                hit_count=len(hits),
                match_types=[hit.note_type for hit in hits],
            )
        if not hits:
            return ""

        rendered = self._render_prompt_context(hits, stage=stage)
        if rendered and observer:
            observer.record_note_memory_prompt_injection(
                match_types=[hit.note_type for hit in hits],
            )
        return rendered

    def _reconcile_locked(self, *, observer: RequestTrace | None = None) -> None:
        metrics_registry.increment("note_memory_refresh_total")
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        collection = self._get_collection(observer=observer)
        if collection is None:
            return

        manifest = self._load_manifest()
        if self._manifest_requires_reset(manifest):
            self._recreate_collection_locked()
            collection = self._get_collection(observer=observer)
            manifest = self._empty_manifest()

        provenance = self._load_provenance()
        notes_path = Path(self._config.notes_workspace)
        notes_on_disk: dict[str, NoteDocument] = {}
        if notes_path.exists():
            for path in sorted(notes_path.glob("*.md")):
                document = self._parse_note_file(path)
                if document is None:
                    continue
                notes_on_disk[document.note_id] = document

        known_manifest_ids = set((manifest.get("notes") or {}).keys())
        disk_note_ids = set(notes_on_disk.keys())

        for removed_note_id in sorted(known_manifest_ids - disk_note_ids):
            self._delete_note_from_collection(collection, manifest, removed_note_id)

        for note_id, document in notes_on_disk.items():
            record = (manifest.get("notes") or {}).get(note_id) or {}
            if (
                str(record.get("checksum") or "") == document.checksum
                and str(record.get("updated_at") or "") == str(document.updated_at or "")
            ):
                continue
            self._upsert_document(collection, manifest, document, provenance.get(note_id) or {})

        self._write_manifest(manifest)

    def _can_reuse_existing_index_locked(
        self,
        manifest: dict[str, Any],
        *,
        observer: RequestTrace | None = None,
    ) -> bool:
        """Reuse an existing persisted index without re-scanning every note file.

        A fresh agent instance is created for every HTTP request, so per-instance state
        alone is not enough to know whether note memory has already been reconciled in
        this process. When a valid manifest and persisted Chroma files already exist,
        we can reopen that collection immediately and avoid blocking the request on a
        full historical re-embedding pass before planner execution starts.
        """

        if self._manifest_requires_reset(manifest):
            return False

        indexed_notes = manifest.get("notes") or {}
        if not indexed_notes:
            return False

        if not self._collection_dir.exists():
            return False

        try:
            has_collection_files = any(self._collection_dir.rglob("*"))
        except OSError:
            return False
        if not has_collection_files:
            return False

        collection = self._get_collection(observer=observer)
        if collection is None:
            return False

        logger.info(
            "note memory reused persisted index indexed_notes=%s memory_dir=%s",
            len(indexed_notes),
            self._memory_dir,
        )
        return True

    def _refresh_locked(
        self,
        note_ids: list[str],
        *,
        observer: RequestTrace | None = None,
    ) -> None:
        metrics_registry.increment("note_memory_refresh_total")
        try:
            collection = self._get_collection(observer=observer)
            if collection is None:
                return
            manifest = self._load_manifest()
            if self._manifest_requires_reset(manifest):
                self._recreate_collection_locked()
                collection = self._get_collection(observer=observer)
                manifest = self._empty_manifest()

            provenance = self._load_provenance()
            for note_id in note_ids:
                path = Path(self._config.notes_workspace) / f"{note_id}.md"
                if not path.exists():
                    self._delete_note_from_collection(collection, manifest, note_id)
                    continue
                document = self._parse_note_file(path)
                if document is None:
                    self._delete_note_from_collection(collection, manifest, note_id)
                    continue
                self._upsert_document(collection, manifest, document, provenance.get(note_id) or {})
            self._write_manifest(manifest)
        except Exception as exc:
            metrics_registry.increment("note_memory_refresh_failed_total")
            logger.warning("note memory refresh failed note_ids=%s error=%s", note_ids, exc)
            if observer:
                observer.record_degraded("note_memory_refresh_failed")

    def _get_collection(self, *, observer: RequestTrace | None = None) -> Any | None:
        if self._collection is not None:
            return self._collection

        client = self._client
        if client is None:
            try:
                client = self._create_default_client()
            except Exception as exc:  # pragma: no cover - depends on runtime packages
                if not self._backend_unavailable_logged:
                    logger.warning("note memory backend unavailable error=%s", exc)
                    self._backend_unavailable_logged = True
                if observer:
                    observer.record_degraded("note_memory_backend_unavailable")
                return None
            self._client = client

        self._collection = client.get_or_create_collection(
            name=_DEFAULT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _create_default_client(self) -> Any:
        if chromadb is None:
            raise RuntimeError("chromadb is not installed")
        self._collection_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._collection_dir))

    def _recreate_collection_locked(self) -> None:
        client = self._client
        if client is None:
            client = self._create_default_client()
            self._client = client

        try:
            client.delete_collection(_DEFAULT_COLLECTION_NAME)
        except Exception:
            pass
        self._collection = client.get_or_create_collection(
            name=_DEFAULT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _manifest_requires_reset(self, manifest: dict[str, Any]) -> bool:
        return (
            int(manifest.get("schema_version") or 0) != _SCHEMA_VERSION
            or int(manifest.get("chunking_version") or 0) != _CHUNKING_VERSION
            or str(manifest.get("embedding_model") or "")
            != self._config.resolved_note_memory_embedding_model()
        )

    def _load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return self._empty_manifest()
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_manifest()
        if not isinstance(payload, dict):
            return self._empty_manifest()
        payload.setdefault("notes", {})
        return payload

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "embedding_model": self._config.resolved_note_memory_embedding_model(),
            "chunking_version": _CHUNKING_VERSION,
            "notes": {},
        }

    def _load_provenance(self) -> dict[str, dict[str, Any]]:
        if not self._config.request_state_enabled:
            return {}
        resolver = NoteProvenanceResolver(self._config.request_state_dir)
        return resolver.resolve()

    def _parse_note_file(self, path: Path) -> NoteDocument | None:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError:
            return None

        frontmatter, body = self._split_frontmatter(raw_text)
        note_id = str(frontmatter.get("id") or "").strip()
        note_type = str(frontmatter.get("type") or "").strip()
        title = str(frontmatter.get("title") or "").strip()
        if not note_id or not title or note_type not in _ALLOWED_NOTE_TYPES:
            return None

        tags = self._parse_tags(frontmatter.get("tags"))
        return NoteDocument(
            note_id=note_id,
            title=title,
            note_type=note_type,
            tags=tags,
            created_at=str(frontmatter.get("created_at") or "").strip() or None,
            updated_at=str(frontmatter.get("updated_at") or "").strip() or None,
            note_path=str(path),
            body=body.strip(),
            checksum=_checksum_text(raw_text),
        )

    def _split_frontmatter(self, raw_text: str) -> tuple[dict[str, Any], str]:
        if not raw_text.startswith(f"{_FRONTMATTER_DELIMITER}\n"):
            return {}, raw_text

        lines = raw_text.splitlines()
        end_index = None
        for index in range(1, len(lines)):
            if lines[index].strip() == _FRONTMATTER_DELIMITER:
                end_index = index
                break
        if end_index is None:
            return {}, raw_text

        frontmatter: dict[str, Any] = {}
        for line in lines[1:end_index]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

        body = "\n".join(lines[end_index + 1 :]).strip()
        return frontmatter, body

    def _parse_tags(self, raw_tags: Any) -> list[str]:
        if raw_tags in (None, ""):
            return []
        text = str(raw_tags).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = []
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        return [part.strip() for part in text.split(",") if part.strip()]

    def _upsert_document(
        self,
        collection: Any,
        manifest: dict[str, Any],
        document: NoteDocument,
        provenance: dict[str, Any],
    ) -> None:
        note_chunks = self._chunk_document(document, provenance=provenance)
        if not note_chunks:
            self._delete_note_from_collection(collection, manifest, document.note_id)
            return

        embeddings = encode_texts(
            [chunk.to_document() for chunk in note_chunks],
            model_name=self._config.resolved_note_memory_embedding_model(),
        )
        valid_triplets = [
            (chunk, embedding)
            for chunk, embedding in zip(note_chunks, embeddings)
            if embedding is not None
        ]
        if not valid_triplets:
            raise RuntimeError("note memory failed to embed note chunks")

        existing_record = (manifest.get("notes") or {}).get(document.note_id) or {}
        existing_chunk_ids = list(existing_record.get("chunk_ids") or [])
        if existing_chunk_ids:
            try:
                collection.delete(ids=existing_chunk_ids)
            except Exception:
                logger.debug("failed to delete stale note-memory chunks note_id=%s", document.note_id)

        collection.upsert(
            ids=[chunk.chunk_id for chunk, _ in valid_triplets],
            documents=[chunk.to_document() for chunk, _ in valid_triplets],
            metadatas=[chunk.to_metadata() for chunk, _ in valid_triplets],
            embeddings=[embedding for _, embedding in valid_triplets],
        )

        manifest.setdefault("notes", {})[document.note_id] = {
            "checksum": document.checksum,
            "updated_at": document.updated_at or "",
            "chunk_ids": [chunk.chunk_id for chunk, _ in valid_triplets],
            "last_indexed_at": _utc_now(),
        }

    def _delete_note_from_collection(
        self,
        collection: Any,
        manifest: dict[str, Any],
        note_id: str,
    ) -> None:
        record = (manifest.get("notes") or {}).pop(note_id, None)
        chunk_ids = list((record or {}).get("chunk_ids") or [])
        if not chunk_ids:
            return
        try:
            collection.delete(ids=chunk_ids)
        except Exception:
            logger.debug("failed to delete missing note-memory chunks note_id=%s", note_id)

    def _chunk_document(
        self,
        document: NoteDocument,
        *,
        provenance: dict[str, Any],
    ) -> list[NoteChunk]:
        sections = self._split_sections(document.body, default_heading=document.title)
        chunks: list[NoteChunk] = []
        chunk_index = 0
        for heading_path, content in sections:
            windows = _iter_chunk_windows(content)
            section_kind = self._section_kind(document.note_type, heading_path)
            for window in windows:
                chunk_index += 1
                chunks.append(
                    NoteChunk(
                        chunk_id=f"{document.note_id}::chunk::{chunk_index}",
                        note_id=document.note_id,
                        note_type=document.note_type,
                        title=document.title,
                        tags=list(document.tags),
                        heading_path=heading_path,
                        content=window,
                        note_path=document.note_path,
                        created_at=document.created_at,
                        updated_at=document.updated_at,
                        resolved_topic=str(provenance.get("topic") or "").strip() or None,
                        resolved_request_id=str(provenance.get("request_id") or "").strip() or None,
                        resolved_request_status=_safe_status(provenance.get("request_status")),
                        resolved_task_id=_safe_int(provenance.get("task_id")),
                        section_kind=section_kind,
                        chunk_index=chunk_index,
                    )
                )
        return chunks

    def _split_sections(self, body: str, *, default_heading: str) -> list[tuple[str, str]]:
        lines = body.splitlines()
        heading_stack: list[str] = []
        current_heading = default_heading
        current_lines: list[str] = []
        sections: list[tuple[str, str]] = []

        def flush() -> None:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_heading, content))

        for line in lines:
            match = _HEADING_PATTERN.match(line.strip())
            if not match:
                current_lines.append(line)
                continue

            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(heading)
            current_heading = " / ".join(item for item in heading_stack if item)
            current_lines = []

        flush()
        if sections:
            return sections
        return [(default_heading, body.strip())] if body.strip() else []

    def _section_kind(self, note_type: str, heading_path: str) -> str:
        lowered = str(heading_path or "").lower()
        if note_type == "conclusion" and any(token in lowered for token in ("参考来源", "references", "sources")):
            return "references"
        return "body"

    def _encode_query(
        self,
        query_text: str,
        *,
        observer: RequestTrace | None = None,
    ) -> list[float] | None:
        if not embeddings_available():
            if observer:
                observer.record_degraded("note_memory_embedding_unavailable")
            return None

        try:
            return encode_text(
                query_text,
                model_name=self._config.resolved_note_memory_embedding_model(),
            )
        except Exception as exc:  # pragma: no cover - depends on runtime embedding stack
            logger.warning("note memory query embedding failed error=%s", exc)
            if observer:
                observer.record_degraded("note_memory_embedding_failed")
            return None

    def _query_hits(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        stage: str,
        current_request_id: str | None,
        exclude_note_ids: set[str],
    ) -> list[MemoryHit]:
        if self._collection is None:
            return []

        request_n = max(top_k * _DEFAULT_QUERY_MULTIPLIER, top_k)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=request_n,
            include=["documents", "distances", "metadatas"],
        )

        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not metadatas:
            return []

        grouped: dict[str, MemoryHit] = {}
        for index, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            note_id = str(metadata.get("note_id") or "").strip()
            if not note_id or note_id in exclude_note_ids:
                continue

            resolved_request_id = str(metadata.get("resolved_request_id") or "").strip() or None
            if current_request_id and resolved_request_id == current_request_id:
                continue

            resolved_status = _safe_status(metadata.get("resolved_request_status"))
            if stage in {"planning", "execution"} and resolved_status == "failed":
                continue

            similarity = max(0.0, 1.0 - float((distances[index] if index < len(distances) else 1.0) or 1.0))
            hit = MemoryHit(
                note_id=note_id,
                note_type=str(metadata.get("note_type") or "").strip() or "unknown",
                title=str(metadata.get("title") or "").strip() or note_id,
                heading_path=str(metadata.get("heading_path") or "").strip(),
                content=str(metadata.get("content") or documents[index] or "").strip(),
                note_path=str(metadata.get("note_path") or "").strip(),
                resolved_topic=str(metadata.get("resolved_topic") or "").strip() or None,
                resolved_request_id=resolved_request_id,
                resolved_request_status=resolved_status,
                resolved_task_id=_safe_int(metadata.get("resolved_task_id")),
                section_kind=str(metadata.get("section_kind") or "body").strip() or "body",
                similarity=similarity,
                score=self._score_hit(
                    similarity=similarity,
                    note_type=str(metadata.get("note_type") or "").strip(),
                    resolved_status=resolved_status,
                    section_kind=str(metadata.get("section_kind") or "body").strip() or "body",
                    stage=stage,
                ),
                match_reason="",
            )
            hit.match_reason = self._build_match_reason(hit, stage=stage)

            existing = grouped.get(note_id)
            if existing is None or hit.score > existing.score:
                grouped[note_id] = hit

        ranked = sorted(grouped.values(), key=lambda item: (item.score, item.similarity), reverse=True)
        return ranked[:top_k]

    def _score_hit(
        self,
        *,
        similarity: float,
        note_type: str,
        resolved_status: str,
        section_kind: str,
        stage: str,
    ) -> float:
        score = similarity
        if stage == "planning":
            score += 0.08 if note_type == "conclusion" else 0.03
        elif stage == "execution":
            score += 0.08 if note_type == "task_state" else 0.03

        score += {"success": 0.08, "partial_success": 0.04, "unknown": 0.0, "failed": -0.12}.get(
            resolved_status,
            0.0,
        )
        if section_kind == "references":
            score -= 0.04
        return score

    def _build_match_reason(self, hit: MemoryHit, *, stage: str) -> str:
        parts: list[str] = []
        if stage == "planning":
            parts.append("历史研究主题相似")
        else:
            parts.append("历史任务处理方式相似")
        if hit.note_type == "conclusion":
            parts.append("结论笔记")
        elif hit.note_type == "task_state":
            parts.append("任务笔记")
        if hit.resolved_request_status == "success":
            parts.append("来自成功请求")
        elif hit.resolved_request_status == "partial_success":
            parts.append("来自部分成功请求")
        return "；".join(parts)

    def _render_prompt_context(self, hits: list[MemoryHit], *, stage: str) -> str:
        if not hits:
            return ""

        header = (
            "历史研究记忆（仅用于启发任务拆解，不代表本轮已验证事实）"
            if stage == "planning"
            else "历史研究记忆（仅用于启发总结组织与补充思路，不是本轮可引用证据）"
        )
        lines = [header]
        char_limit = self._config.note_memory_prompt_char_limit
        remaining = max(240, int(char_limit or 240))
        remaining -= len(header)

        for hit in hits:
            excerpt = truncate_text(hit.content, max(80, min(remaining, 260)))
            topic = hit.resolved_topic or "未知主题"
            item_lines = [
                f"- 标题：{hit.title}",
                f"  主题：{topic}",
                f"  note_type：{hit.note_type}；匹配原因：{hit.match_reason}",
                f"  摘录：{excerpt}",
            ]
            block = "\n".join(item_lines)
            if len(block) > remaining and len(lines) > 1:
                break
            if len(block) > remaining:
                excerpt = truncate_text(hit.content, max(40, remaining - 60))
                block = "\n".join(
                    [
                        f"- 标题：{hit.title}",
                        f"  主题：{topic}",
                        f"  note_type：{hit.note_type}；匹配原因：{hit.match_reason}",
                        f"  摘录：{excerpt}",
                    ]
                )
            lines.append(block)
            remaining -= len(block)
            if remaining <= 80:
                break

        if len(lines) == 1:
            return ""

        lines.append(
            "请把这些历史 notes 仅当作方法提示或背景启发，不能把其中内容直接当作本轮 source_id 证据。"
        )
        return "\n".join(lines)
