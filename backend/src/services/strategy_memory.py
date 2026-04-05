"""Persistent cross-request strategy memory backed by a local vector database."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from config import Configuration
from metrics import RequestTrace, metrics_registry
from services.embeddings import embeddings_available, encode_text, encode_texts
from services.strategy_synthesizer import StrategyCard, StrategySourceRequest, StrategySynthesizer
from utils import truncate_text

try:  # pragma: no cover - exercised through runtime fallback
    import chromadb
except Exception:  # pragma: no cover - exercised through runtime fallback
    chromadb = None

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"success", "partial_success", "failed"}
_STATUS_PRIORITY = {"success": 3, "partial_success": 2, "failed": 1, "unknown": 0}
_SCHEMA_VERSION = 1
_DEFAULT_COLLECTION_NAME = "deep_research_strategies_v1"
_DEFAULT_QUERY_MULTIPLIER = 4
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


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


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_list(value: Any, *, max_items: int = 6) -> list[str]:
    if isinstance(value, list):
        candidates = value
    elif value in (None, ""):
        candidates = []
    else:
        candidates = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = _normalize_text(item)
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


def _parse_json_list(value: Any) -> list[Any]:
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


def _checksum_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass
class StrategyHit:
    strategy_id: str
    strategy_kind: str
    stage_scope: str
    title: str
    applicable_when: str
    match_signals: list[str]
    recommended_actions: list[str]
    query_templates: list[str]
    preferred_sources: list[str]
    pitfalls_to_avoid: list[str]
    origin_request_id: str
    origin_status: str
    origin_review_status: str
    similarity: float
    score: float
    match_reason: str


class StrategyMemoryService:
    """Manage local strategy retrieval memory using semantic search."""

    def __init__(
        self,
        config: Configuration,
        *,
        synthesizer: StrategySynthesizer | None = None,
        client: Any | None = None,
    ) -> None:
        self._config = config
        self._synthesizer = synthesizer
        self._client = client
        self._collection: Any | None = None
        self._lock = Lock()
        self._memory_dir = Path(self._config.strategy_memory_dir)
        self._cards_dir = self._memory_dir / "cards"
        self._manifest_path = self._memory_dir / "manifest.json"
        self._collection_dir = self._memory_dir / "chromadb"
        self._reconciled = False
        self._backend_unavailable_logged = False

    @property
    def enabled(self) -> bool:
        return bool(
            self._config.strategy_memory_enabled
            and self._config.request_state_enabled
            and str(self._config.request_state_dir or "").strip()
        )

    def ensure_reconciled(self, *, observer: RequestTrace | None = None) -> None:
        if not self.enabled:
            return

        with self._lock:
            if self._reconciled:
                return
            try:
                self._reconcile_locked(observer=observer)
            except Exception:
                metrics_registry.increment("strategy_memory_refresh_failed_total")
                raise
            self._reconciled = True

    def refresh_request(
        self,
        request_id: str,
        *,
        observer: RequestTrace | None = None,
    ) -> None:
        if not self.enabled:
            return

        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            return

        with self._lock:
            metrics_registry.increment("strategy_memory_refresh_total")
            try:
                collection = self._get_collection(observer=observer)
                if collection is None:
                    return
                manifest = self._load_manifest()
                if self._manifest_requires_reset(manifest):
                    self._recreate_collection_locked()
                    collection = self._get_collection(observer=observer)
                    manifest = self._empty_manifest()

                snapshot_path = Path(self._config.request_state_dir) / f"{normalized_request_id}.json"
                if not snapshot_path.exists():
                    self._delete_request_from_collection(
                        collection,
                        manifest,
                        normalized_request_id,
                    )
                    self._write_manifest(manifest)
                    return

                snapshot = self._load_snapshot(snapshot_path)
                if snapshot is None or not self._is_terminal_snapshot(snapshot):
                    self._delete_request_from_collection(
                        collection,
                        manifest,
                        normalized_request_id,
                    )
                    self._write_manifest(manifest)
                    return

                self._sync_snapshot_locked(
                    collection,
                    manifest,
                    snapshot,
                    observer=observer,
                )
                self._write_manifest(manifest)
            except Exception as exc:
                metrics_registry.increment("strategy_memory_refresh_failed_total")
                logger.warning(
                    "strategy memory refresh failed request_id=%s error=%s",
                    normalized_request_id,
                    exc,
                )
                if observer:
                    observer.record_degraded("strategy_memory_refresh_failed")

    def search_for_planning(
        self,
        research_topic: str,
        *,
        current_request_id: str | None = None,
        observer: RequestTrace | None = None,
    ) -> str:
        return self._search_and_render(
            query_text=research_topic,
            stage="planning",
            top_k=self._config.strategy_memory_planning_top_k,
            current_request_id=current_request_id,
            observer=observer,
        )

    def search_for_reflection(
        self,
        research_topic: str,
        *,
        gap_signals: list[str] | None = None,
        task_titles: list[str] | None = None,
        current_request_id: str | None = None,
        observer: RequestTrace | None = None,
    ) -> str:
        query_text = " ".join(
            part
            for part in [
                research_topic,
                " ".join(_normalize_list(gap_signals or [], max_items=6)),
                " ".join(_normalize_list(task_titles or [], max_items=6)),
            ]
            if _normalize_text(part)
        )
        return self._search_and_render(
            query_text=query_text,
            stage="reflection",
            top_k=self._config.strategy_memory_reflection_top_k,
            current_request_id=current_request_id,
            observer=observer,
        )

    def _search_and_render(
        self,
        *,
        query_text: str,
        stage: str,
        top_k: int,
        current_request_id: str | None,
        observer: RequestTrace | None,
    ) -> str:
        if not self.enabled or not _normalize_text(query_text):
            return ""

        try:
            self.ensure_reconciled(observer=observer)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.warning("strategy memory reconcile failed stage=%s error=%s", stage, exc)
            if observer:
                observer.record_degraded("strategy_memory_reconcile_failed")
            return ""

        if self._collection is None:
            return ""

        query_embedding = self._encode_query(query_text, observer=observer)
        if query_embedding is None:
            return ""

        hits = self._query_hits(
            query_embedding=query_embedding,
            stage=stage,
            top_k=max(1, int(top_k or 1)),
            current_request_id=current_request_id,
        )
        if observer:
            observer.record_strategy_memory_query(
                hit_count=len(hits),
                match_kinds=[hit.strategy_kind for hit in hits],
            )
        if not hits:
            return ""

        rendered = self._render_prompt_context(hits, stage=stage)
        if rendered and observer:
            observer.record_strategy_memory_prompt_injection(
                match_kinds=[hit.strategy_kind for hit in hits],
            )
        return rendered

    def _reconcile_locked(self, *, observer: RequestTrace | None = None) -> None:
        metrics_registry.increment("strategy_memory_refresh_total")
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._cards_dir.mkdir(parents=True, exist_ok=True)
        collection = self._get_collection(observer=observer)
        if collection is None:
            return

        manifest = self._load_manifest()
        if self._manifest_requires_reset(manifest):
            self._recreate_collection_locked()
            collection = self._get_collection(observer=observer)
            manifest = self._empty_manifest()

        snapshots_on_disk: dict[str, dict[str, Any]] = {}
        request_state_path = Path(self._config.request_state_dir)
        if request_state_path.exists():
            for path in sorted(request_state_path.glob("*.json")):
                snapshot = self._load_snapshot(path)
                if snapshot is None or not self._is_terminal_snapshot(snapshot):
                    continue
                request_id = _normalize_text(snapshot.get("request_id"))
                if request_id:
                    snapshots_on_disk[request_id] = snapshot

        known_request_ids = set((manifest.get("requests") or {}).keys())
        disk_request_ids = set(snapshots_on_disk.keys())
        for removed_request_id in sorted(known_request_ids - disk_request_ids):
            self._delete_request_from_collection(collection, manifest, removed_request_id)

        for request_id, snapshot in snapshots_on_disk.items():
            source_request, checksum = self._build_source_request(snapshot)
            record = (manifest.get("requests") or {}).get(request_id) or {}
            if source_request is None:
                self._delete_request_from_collection(collection, manifest, request_id)
                continue
            if str(record.get("checksum") or "") == checksum:
                continue
            try:
                self._sync_snapshot_locked(
                    collection,
                    manifest,
                    snapshot,
                    observer=observer,
                    source_request=source_request,
                    checksum=checksum,
                )
            except Exception as exc:
                metrics_registry.increment("strategy_memory_refresh_failed_total")
                logger.warning(
                    "strategy memory reconcile skipped request_id=%s error=%s",
                    request_id,
                    exc,
                )
                if observer:
                    observer.record_degraded("strategy_memory_reconcile_failed")

        self._write_manifest(manifest)

    def _sync_snapshot_locked(
        self,
        collection: Any,
        manifest: dict[str, Any],
        snapshot: dict[str, Any],
        *,
        observer: RequestTrace | None = None,
        source_request: StrategySourceRequest | None = None,
        checksum: str | None = None,
    ) -> None:
        request_id = _normalize_text(snapshot.get("request_id"))
        if not request_id:
            return

        source_request = source_request or self._build_source_request(snapshot)[0]
        if source_request is None:
            self._delete_request_from_collection(collection, manifest, request_id)
            return

        checksum = checksum or self._build_source_request(snapshot)[1]
        if not self._synthesizer:
            logger.warning("strategy memory synthesizer unavailable request_id=%s", request_id)
            if observer:
                observer.record_degraded("strategy_memory_synthesizer_unavailable")
            return

        try:
            cards = self._synthesizer.synthesize(source_request)
        except Exception as exc:
            logger.warning("strategy memory synthesis failed request_id=%s error=%s", request_id, exc)
            if observer:
                observer.record_degraded("strategy_memory_synthesis_failed")
            return

        if not cards:
            self._delete_request_from_collection(collection, manifest, request_id)
            return

        try:
            embeddings = encode_texts(
                [card.to_document() for card in cards],
                model_name=self._config.resolved_strategy_memory_embedding_model(),
            )
        except Exception as exc:
            logger.warning("strategy memory card embedding failed request_id=%s error=%s", request_id, exc)
            if observer:
                observer.record_degraded("strategy_memory_embedding_failed")
            return
        valid_pairs = [
            (card, embedding)
            for card, embedding in zip(cards, embeddings)
            if embedding is not None
        ]
        if not valid_pairs:
            if observer:
                observer.record_degraded("strategy_memory_embedding_failed")
            return

        self._delete_request_from_collection(collection, manifest, request_id)
        collection.upsert(
            ids=[card.strategy_id for card, _ in valid_pairs],
            documents=[card.to_document() for card, _ in valid_pairs],
            metadatas=[card.to_metadata() for card, _ in valid_pairs],
            embeddings=[embedding for _, embedding in valid_pairs],
        )
        for card, _ in valid_pairs:
            self._write_card_file(card)

        metrics_registry.increment("strategy_memory_synthesized_card_total", len(valid_pairs))
        manifest.setdefault("requests", {})[request_id] = {
            "checksum": checksum,
            "status": source_request.status,
            "card_ids": [card.strategy_id for card, _ in valid_pairs],
            "last_indexed_at": source_request.request_metrics.get("updated_at") or "",
        }

    def _build_source_request(
        self,
        snapshot: dict[str, Any],
    ) -> tuple[StrategySourceRequest | None, str | None]:
        request_id = _normalize_text(snapshot.get("request_id"))
        topic = _normalize_text(snapshot.get("topic"))
        status = _safe_status(snapshot.get("status"))
        if not request_id or not topic or status not in _TERMINAL_STATUSES:
            return None, None

        todo_items = snapshot.get("todo_items") or []
        tasks: list[dict[str, Any]] = []
        completed_task_count = 0
        failed_task_count = 0
        non_empty_result = False
        for item in todo_items:
            if not isinstance(item, dict):
                continue
            task_status = _normalize_text(item.get("status")) or "unknown"
            if task_status == "completed":
                completed_task_count += 1
            if task_status in {"failed", "skipped"}:
                failed_task_count += 1
            summary = truncate_text(_normalize_text(item.get("summary")), 220)
            if summary and summary != "暂无可用信息":
                non_empty_result = True
            tasks.append(
                {
                    "id": _safe_int(item.get("id")),
                    "title": truncate_text(_normalize_text(item.get("title")), 80),
                    "intent": truncate_text(_normalize_text(item.get("intent")), 140),
                    "query": truncate_text(_normalize_text(item.get("query")), 120),
                    "status": task_status,
                    "summary_excerpt": summary,
                    "notices": _normalize_list(item.get("notices"), max_items=4),
                    "review_issues": _normalize_list(item.get("review_issues"), max_items=4),
                    "react_gap_signals": _normalize_list(item.get("react_gap_signals"), max_items=4),
                    "react_last_action": _normalize_text(item.get("react_last_action")),
                    "react_stop_reason": _normalize_text(item.get("react_stop_reason")),
                }
            )

        report_markdown = _normalize_text(snapshot.get("report_markdown"))
        review_summary = snapshot.get("review_summary") or {}
        review_status = _normalize_text(review_summary.get("overall_status")) or "unknown"
        request_metrics = snapshot.get("request_metrics") or {}
        reflection_gap_signals = _normalize_list(
            request_metrics.get("reflection_gap_signals"),
            max_items=6,
        )
        degraded_reasons = _normalize_list(
            request_metrics.get("degraded_reasons"),
            max_items=6,
        )
        repair_cycles = max(
            int(
                snapshot.get("report_repair_cycles")
                or request_metrics.get("report_repair_cycles")
                or 0
            ),
            0,
        )

        positive_eligible = (
            status in {"success", "partial_success"}
            and bool(report_markdown)
            and completed_task_count > 0
            and (bool(non_empty_result) or bool(report_markdown))
        )
        anti_only = (
            status == "failed"
            or review_status == "blocked"
            or repair_cycles > 0
            or failed_task_count > 0
        )
        if anti_only:
            requested_kinds = ["anti_pattern"]
        elif positive_eligible:
            requested_kinds = ["planning_pattern", "reflection_pattern"]
        else:
            requested_kinds = []
        if not requested_kinds:
            return None, None

        condensed_metrics = {
            "fallback_reasons": _normalize_list(request_metrics.get("fallback_reasons"), max_items=4),
            "degraded_reasons": degraded_reasons,
            "reflection_reason": _normalize_text(request_metrics.get("reflection_reason")),
            "reflection_gap_signals": reflection_gap_signals,
            "reflection_added_tasks": int(request_metrics.get("reflection_added_tasks") or 0),
            "report_repair_triggered": bool(request_metrics.get("report_repair_triggered")),
            "report_repair_added_tasks": int(request_metrics.get("report_repair_added_tasks") or 0),
            "report_repair_cycles": int(request_metrics.get("report_repair_cycles") or repair_cycles or 0),
            "task_react_stop_reasons": request_metrics.get("task_react_stop_reasons") or {},
            "updated_at": _normalize_text(snapshot.get("updated_at")),
        }
        checksum = _checksum_payload(
            {
                "topic": topic,
                "status": status,
                "requested_kinds": requested_kinds,
                "tasks": tasks,
                "review_summary": {
                    "overall_status": review_status,
                    "reason": _normalize_text(review_summary.get("reason")),
                    "issue_count": int(review_summary.get("issue_count") or 0),
                },
                "reflection_reason": condensed_metrics.get("reflection_reason"),
                "reflection_gap_signals": reflection_gap_signals,
                "report_repair": {
                    "triggered": condensed_metrics.get("report_repair_triggered"),
                    "added_tasks": condensed_metrics.get("report_repair_added_tasks"),
                    "cycles": condensed_metrics.get("report_repair_cycles"),
                },
                "request_metrics": condensed_metrics,
            }
        )
        return (
            StrategySourceRequest(
                request_id=request_id,
                topic=topic,
                status=status,
                review_status=review_status,
                report_available=bool(report_markdown),
                completed_task_count=completed_task_count,
                failed_task_count=failed_task_count,
                repair_cycles=repair_cycles,
                reflection_gap_signals=reflection_gap_signals,
                degraded_reasons=degraded_reasons,
                tasks=tasks,
                request_metrics=condensed_metrics,
                requested_kinds=requested_kinds,
            ),
            checksum,
        )

    def _load_snapshot(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _is_terminal_snapshot(self, snapshot: dict[str, Any]) -> bool:
        return _safe_status(snapshot.get("status")) in _TERMINAL_STATUSES

    def _get_collection(self, *, observer: RequestTrace | None = None) -> Any | None:
        if self._collection is not None:
            return self._collection

        client = self._client
        if client is None:
            try:
                client = self._create_default_client()
            except Exception as exc:  # pragma: no cover - depends on runtime packages
                if not self._backend_unavailable_logged:
                    logger.warning("strategy memory backend unavailable error=%s", exc)
                    self._backend_unavailable_logged = True
                if observer:
                    observer.record_degraded("strategy_memory_backend_unavailable")
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
            or str(manifest.get("embedding_model") or "")
            != self._config.resolved_strategy_memory_embedding_model()
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
        payload.setdefault("requests", {})
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
            "embedding_model": self._config.resolved_strategy_memory_embedding_model(),
            "requests": {},
        }

    def _delete_request_from_collection(
        self,
        collection: Any,
        manifest: dict[str, Any],
        request_id: str,
    ) -> None:
        record = (manifest.get("requests") or {}).pop(request_id, None)
        card_ids = list((record or {}).get("card_ids") or [])
        if card_ids:
            try:
                collection.delete(ids=card_ids)
            except Exception:
                logger.debug("failed to delete strategy-memory cards request_id=%s", request_id)
        for card_id in card_ids:
            self._delete_card_file(card_id)

    def _write_card_file(self, card: StrategyCard) -> None:
        self._cards_dir.mkdir(parents=True, exist_ok=True)
        self._card_path(card.strategy_id).write_text(
            json.dumps(card.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _delete_card_file(self, strategy_id: str) -> None:
        self._card_path(strategy_id).unlink(missing_ok=True)

    def _card_path(self, strategy_id: str) -> Path:
        safe_name = _SAFE_FILENAME_PATTERN.sub("_", strategy_id)
        return self._cards_dir / f"{safe_name}.json"

    def _encode_query(
        self,
        query_text: str,
        *,
        observer: RequestTrace | None = None,
    ) -> list[float] | None:
        if not embeddings_available():
            if observer:
                observer.record_degraded("strategy_memory_embedding_unavailable")
            return None

        try:
            return encode_text(
                query_text,
                model_name=self._config.resolved_strategy_memory_embedding_model(),
            )
        except Exception as exc:  # pragma: no cover - depends on runtime embedding stack
            logger.warning("strategy memory query embedding failed error=%s", exc)
            if observer:
                observer.record_degraded("strategy_memory_embedding_failed")
            return None

    def _query_hits(
        self,
        *,
        query_embedding: list[float],
        stage: str,
        top_k: int,
        current_request_id: str | None,
    ) -> list[StrategyHit]:
        if self._collection is None:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * _DEFAULT_QUERY_MULTIPLIER, top_k),
            include=["documents", "distances", "metadatas"],
        )
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not metadatas:
            return []

        hits: list[StrategyHit] = []
        for index, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            origin_request_id = _normalize_text(metadata.get("origin_request_id"))
            if current_request_id and origin_request_id == current_request_id:
                continue

            strategy_kind = _normalize_text(metadata.get("strategy_kind"))
            if stage == "planning" and strategy_kind not in {"planning_pattern", "anti_pattern"}:
                continue
            if stage == "reflection" and strategy_kind not in {"reflection_pattern", "anti_pattern"}:
                continue

            similarity = max(0.0, 1.0 - float((distances[index] if index < len(distances) else 1.0) or 1.0))
            match_signals = _normalize_list(_parse_json_list(metadata.get("match_signals")), max_items=4)
            recommended_actions = _normalize_list(
                _parse_json_list(metadata.get("recommended_actions")),
                max_items=4,
            )
            query_templates = _normalize_list(
                _parse_json_list(metadata.get("query_templates")),
                max_items=3,
            )
            preferred_sources = _normalize_list(
                _parse_json_list(metadata.get("preferred_sources")),
                max_items=3,
            )
            pitfalls_to_avoid = _normalize_list(
                _parse_json_list(metadata.get("pitfalls_to_avoid")),
                max_items=3,
            )
            origin_status = _safe_status(metadata.get("origin_status"))
            hit = StrategyHit(
                strategy_id=_normalize_text(metadata.get("strategy_id")),
                strategy_kind=strategy_kind,
                stage_scope=_normalize_text(metadata.get("stage_scope")) or "planning",
                title=_normalize_text(metadata.get("title")) or _normalize_text(documents[index]),
                applicable_when=_normalize_text(metadata.get("applicable_when")),
                match_signals=match_signals,
                recommended_actions=recommended_actions,
                query_templates=query_templates,
                preferred_sources=preferred_sources,
                pitfalls_to_avoid=pitfalls_to_avoid,
                origin_request_id=origin_request_id,
                origin_status=origin_status,
                origin_review_status=_normalize_text(metadata.get("origin_review_status")) or "unknown",
                similarity=similarity,
                score=self._score_hit(
                    similarity=similarity,
                    stage=stage,
                    strategy_kind=strategy_kind,
                    origin_status=origin_status,
                ),
                match_reason="",
            )
            hit.match_reason = self._build_match_reason(hit, stage=stage)
            hits.append(hit)

        ranked = sorted(hits, key=lambda item: (item.score, item.similarity), reverse=True)
        unique_hits: list[StrategyHit] = []
        seen_ids: set[str] = set()
        for hit in ranked:
            if hit.strategy_id in seen_ids:
                continue
            seen_ids.add(hit.strategy_id)
            unique_hits.append(hit)
            if len(unique_hits) >= top_k:
                break
        return unique_hits

    def _score_hit(
        self,
        *,
        similarity: float,
        stage: str,
        strategy_kind: str,
        origin_status: str,
    ) -> float:
        score = similarity
        if stage == "planning":
            if strategy_kind == "planning_pattern":
                score += 0.08
            elif strategy_kind == "anti_pattern":
                score += 0.03
        elif stage == "reflection":
            if strategy_kind == "reflection_pattern":
                score += 0.08
            elif strategy_kind == "anti_pattern":
                score += 0.08

        score += {"success": 0.08, "partial_success": 0.04, "failed": 0.01, "unknown": 0.0}.get(
            origin_status,
            0.0,
        )
        return score

    def _build_match_reason(self, hit: StrategyHit, *, stage: str) -> str:
        parts: list[str] = []
        if stage == "planning":
            parts.append("历史任务拆解模式相似")
        else:
            parts.append("历史覆盖修补经验相似")

        if hit.strategy_kind == "planning_pattern":
            parts.append("规划策略")
        elif hit.strategy_kind == "reflection_pattern":
            parts.append("反思策略")
        else:
            parts.append("失败反模式")

        if hit.origin_status == "success":
            parts.append("来自成功请求")
        elif hit.origin_status == "partial_success":
            parts.append("来自部分成功请求")
        elif hit.origin_status == "failed":
            parts.append("来自失败请求")
        return "；".join(parts)

    def _render_prompt_context(self, hits: list[StrategyHit], *, stage: str) -> str:
        if not hits:
            return ""

        header = (
            "历史策略记忆（仅用于启发任务拆解，不代表当前主题事实或本轮证据）"
            if stage == "planning"
            else "历史策略记忆（仅用于启发覆盖缺口判断与修补思路，不代表当前主题事实或本轮证据）"
        )
        lines = [header]
        remaining = max(240, int(self._config.strategy_memory_prompt_char_limit or 240)) - len(header)

        for hit in hits:
            action_line = "；".join(hit.recommended_actions[:3]) or "暂无明确推荐动作"
            query_line = "；".join(hit.query_templates[:2]) or "暂无固定检索模板"
            pitfall_line = "；".join(hit.pitfalls_to_avoid[:2]) or "暂无显式反模式"
            signal_line = "；".join(hit.match_signals[:3]) or truncate_text(hit.applicable_when, 100)
            block = "\n".join(
                [
                    f"- 标题：{hit.title}",
                    f"  类型：{hit.strategy_kind}；匹配原因：{hit.match_reason}",
                    f"  适用场景：{truncate_text(hit.applicable_when, 160)}",
                    f"  触发信号：{truncate_text(signal_line, 120)}",
                    f"  推荐动作：{truncate_text(action_line, 160)}",
                    f"  检索模板：{truncate_text(query_line, 140)}",
                    f"  避免事项：{truncate_text(pitfall_line, 140)}",
                ]
            )
            if len(block) > remaining and len(lines) > 1:
                break
            if len(block) > remaining:
                block = truncate_text(block, max(80, remaining))
            lines.append(block)
            remaining -= len(block)
            if remaining <= 80:
                break

        if len(lines) == 1:
            return ""

        lines.append(
            "这些内容只是历史方法经验，不能当作当前主题事实、不能替代本轮搜索，也不能直接作为当前证据引用。"
        )
        return "\n".join(lines)
