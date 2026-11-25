"""Persistent cross-request strategy memory backed by a local vector database."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from config import Configuration
from metrics import RequestTrace, metrics_registry
from models import MemoryPurpose
from services.embeddings import embeddings_available, encode_text, encode_texts
from services.memory_sanitizer import (
    collapse_whitespace,
    normalize_list,
    parse_json_list,
    sanitize_memory_text,
)
from services.memory_scope import (
    MemoryPolicy,
    MemoryScope,
    authorize,
    checked_path,
    open_partition_directory,
    partition_lock,
    partition_root,
    read_partition_text,
    remove_partition_file,
    scope_is_current,
    strict_memory_text,
    strict_partition_required,
    valid_internal_id,
    write_partition_text,
)
from services.strategy_synthesizer import (
    DOMAIN_GENERAL,
    MATURITY_ARCHIVED,
    MATURITY_DRAFT,
    MATURITY_ORDER,
    MATURITY_RETRIEVAL_WEIGHTS,
    StrategyCard,
    StrategySourceRequest,
    StrategySynthesizer,
    clean_strategy_fields,
    maturity_meets_gate,
    normalize_maturity,
    normalize_topic_domain,
)
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
_PARTITION_SAMPLE_LIMIT = 16
_PARTITION_REQUEST_LIMIT = 256
_SANITIZATION_VERSION = 1
_PARTITION_MANIFEST = "strategies.json"
_STRATEGY_STAGES = {
    "planning_pattern": "planning",
    "reflection_pattern": "reflection",
    "anti_pattern": "reflection",
}


class StrategySnapshotReader(Protocol):
    def load(self, scope: MemoryScope, request_id: str) -> dict[str, Any] | None: ...

    def iter_terminal(
        self, scope: MemoryScope, *, after: str = "", limit: int = 64,
    ) -> list[dict[str, Any]]: ...


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
    return collapse_whitespace(value)


def _normalize_list(value: Any, *, max_items: int = 6) -> list[str]:
    return normalize_list(value, max_items=max_items)


def _parse_json_list(value: Any) -> list[Any]:
    return parse_json_list(value)


def _checksum_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp, normalizing naive values to UTC so comparisons are safe."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


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
    maturity: str = MATURITY_DRAFT
    topic_domain: str = DOMAIN_GENERAL


class StrategyMemoryService:
    """Manage local strategy retrieval memory using semantic search."""

    def __init__(
        self,
        config: Configuration,
        *,
        synthesizer: StrategySynthesizer | None = None,
        client: Any | None = None,
        scope: MemoryScope | None = None,
        policy: MemoryPolicy | None = None,
        snapshot_reader: StrategySnapshotReader | None = None,
        frozen_strict: bool = False,
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
        self._scope = scope
        self._policy = policy
        self._snapshot_reader = snapshot_reader
        self._frozen_strict = frozen_strict
        self._partition_collection: Any | None = None

    @property
    def enabled(self) -> bool:
        if self._strict:
            return bool(
                self._config.strategy_memory_enabled and self._config.request_state_enabled
                and self._snapshot_reader is not None and scope_is_current(self._policy, self._scope)
            )
        return bool(
            self._config.strategy_memory_enabled
            and self._config.request_state_enabled
            and str(self._config.request_state_dir or "").strip()
        )

    @property
    def _strict(self) -> bool:
        return strict_partition_required(
            self._config, self._memory_dir, self._scope, frozen_strict=self._frozen_strict,
        )

    def ensure_reconciled(self, *, observer: RequestTrace | None = None) -> None:
        if self._strict:
            self.rebuild(dry_run=False)
            return
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
        if self._strict:
            self._refresh_partition_request(request_id, observer=observer)
            return
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
        topic_domains: set[str] | None = None,
    ) -> str:
        return self._search_and_render(
            query_text=research_topic,
            stage="planning",
            top_k=self._config.strategy_memory_planning_top_k,
            current_request_id=current_request_id,
            observer=observer,
            topic_domains=topic_domains,
        )

    def search_for_reflection(
        self,
        research_topic: str,
        *,
        gap_signals: list[str] | None = None,
        task_titles: list[str] | None = None,
        current_request_id: str | None = None,
        observer: RequestTrace | None = None,
        topic_domains: set[str] | None = None,
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
            topic_domains=topic_domains,
        )

    def _search_and_render(
        self,
        *,
        query_text: str,
        stage: str,
        top_k: int,
        current_request_id: str | None,
        observer: RequestTrace | None,
        topic_domains: set[str] | None = None,
    ) -> str:
        if self._strict:
            return self._search_partition(
                query_text, stage=stage, top_k=top_k, current_request_id=current_request_id,
                observer=observer, topic_domains=topic_domains,
            )
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
            topic_domains=topic_domains,
        )
        if observer:
            observer.record_strategy_memory_query(
                hit_count=len(hits),
                match_kinds=[hit.strategy_kind for hit in hits],
                strategy_ids=[hit.strategy_id for hit in hits],
            )
        if not hits:
            return ""

        rendered = self._render_prompt_context(hits, stage=stage)
        if rendered and observer:
            observer.record_strategy_memory_prompt_injection(
                match_kinds=[hit.strategy_kind for hit in hits],
            )
        return rendered

    def _partition_path(self, *, create: bool = False) -> Path:
        if not self.enabled or self._scope is None:
            raise PermissionError("strategy scope unavailable")
        tenant_dir = partition_root(self._memory_dir, self._scope, create=create)
        policy_dir = f"policy-{self._scope.policy_version}"
        if create:
            with open_partition_directory(tenant_dir, policy_dir, create=True):
                pass
        return checked_path(tenant_dir, policy_dir)

    def _partition_manifest(self, root: Path) -> dict[str, Any]:
        binding = {
            "schema_version": _SCHEMA_VERSION,
            "tenant_id": self._scope.tenant_id,
            "policy_version": self._scope.policy_version,
            "sanitization_version": _SANITIZATION_VERSION,
            "embedding_model": self._config.resolved_strategy_memory_embedding_model(),
        }
        try:
            manifest = json.loads(read_partition_text(root, _PARTITION_MANIFEST))
        except FileNotFoundError:
            return {**binding, "requests": {}, "revoked": {}}
        if (
            not isinstance(manifest, dict)
            or any(manifest.get(key) != value for key, value in binding.items())
            or not isinstance(manifest.get("requests"), dict)
            or not isinstance(manifest.get("revoked"), dict)
            or len(manifest["requests"]) > _PARTITION_REQUEST_LIMIT
        ):
            raise ValueError("invalid strategy manifest")
        return manifest

    def _save_partition_manifest(self, root: Path, manifest: dict[str, Any]) -> None:
        write_partition_text(
            root, _PARTITION_MANIFEST, text=json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        )

    def _get_partition_collection(self, root: Path) -> Any:
        if self._partition_collection is not None:
            return self._partition_collection
        client = self._client
        if client is None:
            if chromadb is None:
                raise RuntimeError("strategy vector service unavailable")
            with open_partition_directory(root, "chromadb", create=True):
                collection_dir = checked_path(root, "chromadb")
                for leaf in ("chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"):
                    checked_path(collection_dir, leaf)
                client = chromadb.PersistentClient(path=str(collection_dir))
        collection_name = f"sc_{self._scope.namespace[2:]}_{self._scope.policy_version}"
        self._partition_collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"},
        )
        return self._partition_collection

    def _record_partition(self, operation: str, outcome: str) -> None:
        metrics_registry.record_v4_memory_partition(
            "strategy", operation, outcome, tenant_id=self._scope.tenant_id if self._scope else None,
        )

    def _strict_snapshot(self, request_id: str, purpose: MemoryPurpose) -> dict[str, Any] | None:
        if not authorize(self._policy, self._scope, request_id, purpose):
            return None
        snapshot = self._snapshot_reader.load(self._scope, request_id)
        if (
            not isinstance(snapshot, dict) or snapshot.get("tenant_id") != self._scope.tenant_id
            or snapshot.get("request_id") != request_id or not self._is_terminal_snapshot(snapshot)
            or type(snapshot.get("state_version")) is not int or snapshot["state_version"] < 1
        ):
            return None
        return snapshot

    def _strict_source(
        self, snapshot: dict[str, Any],
    ) -> tuple[StrategySourceRequest | None, str | None]:
        topic = strict_memory_text(snapshot.get("topic")) or ""
        statuses = {"completed", "pending", "in_progress", "running", "failed", "skipped"}
        items = snapshot.get("todo_items")
        if not isinstance(items, list):
            return None, None
        tasks = [
            {
                "id": item.get("id") if type(item.get("id")) is int else None,
                "title": (strict_memory_text(item.get("title")) or "")[:80],
                "intent": (strict_memory_text(item.get("intent")) or "")[:140],
                "status": item.get("status") if isinstance(item.get("status"), str)
                and item["status"] in statuses else "unknown",
            }
            for item in items[:16] if isinstance(item, dict)
        ]
        review = snapshot.get("review_summary")
        review_status = review.get("overall_status") if isinstance(review, dict) else None
        repair_cycles = snapshot.get("report_repair_cycles")
        metrics = snapshot.get("request_metrics")
        if type(repair_cycles) is not int and isinstance(metrics, dict):
            repair_cycles = metrics.get("report_repair_cycles")
        gaps = metrics.get("reflection_gap_signals") if isinstance(metrics, dict) else None
        gaps = [
            clean[:120] for item in gaps[:6] if (clean := strict_memory_text(item))
        ] if isinstance(gaps, list) else []
        source, checksum = self._build_source_request({
            "request_id": snapshot["request_id"],
            "topic": topic[:200] or "已完成研究",
            "status": snapshot["status"],
            "todo_items": tasks,
            "report_markdown": "published" if snapshot.get("report_markdown") else "",
            "review_summary": {
                "overall_status": review_status if review_status in ("passed", "blocked", "warning") else "unknown",
            },
            "report_repair_cycles": repair_cycles if type(repair_cycles) is int else 0,
            "request_metrics": {"reflection_gap_signals": gaps},
        })
        if source is not None:
            source._verified_scope = self._scope
        return source, checksum

    def _partition_card(
        self, root: Path, card_id: str, record_id: str, request_id: str, source_version: int,
    ) -> dict[str, Any] | None:
        if not valid_internal_id(card_id) or not valid_internal_id(record_id):
            return None
        try:
            card = json.loads(read_partition_text(root, "cards", f"{record_id}.json", max_bytes=32_768))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(card, dict) or card.get("tenant_id") != self._scope.tenant_id
            or card.get("policy_version") != self._scope.policy_version
            or card.get("sanitization_version") != _SANITIZATION_VERSION
            or card.get("source_version") != source_version or card.get("origin_request_id") != request_id
            or card.get("strategy_id") != card_id or card.get("record_id") != record_id
            or clean_strategy_fields(card) is None
        ):
            return None
        identity = {
            key: value for key, value in card.items()
            if key not in {"record_id", "hit_count", "last_hit_at", "maturity", "created_at"}
        }
        return card if "sr_" + _checksum_payload(identity) == record_id else None

    def _stage_strategy_cards(
        self, root: Path, manifest: dict[str, Any], snapshot: dict[str, Any], *,
        observer: RequestTrace | None = None,
    ) -> bool:
        request_id = snapshot["request_id"]
        source_version = snapshot["state_version"]
        if request_id in manifest["revoked"]:
            return False
        old = manifest["requests"].get(request_id, {})
        if (
            old.get("source_version") == source_version
            and old.get("cards")
            and all(
                self._partition_card(root, card_id, record_id, request_id, source_version)
                for card_id, record_id in old["cards"].items()
            )
        ):
            return False
        if not old and len(manifest["requests"]) >= _PARTITION_REQUEST_LIMIT:
            return False
        source, checksum = self._strict_source(snapshot)
        if source is None or self._synthesizer is None:
            return False

        cards: dict[str, StrategyCard] = {}
        for candidate in self._synthesizer.synthesize(source):
            kind = candidate.strategy_kind
            if kind not in source.requested_kinds or kind in cards:
                continue
            clean = clean_strategy_fields(candidate.to_json_dict())
            if clean is None:
                continue
            strategy_id = "sc_" + _checksum_payload({
                "tenant_id": self._scope.tenant_id, "policy_version": self._scope.policy_version,
                "request_id": request_id, "kind": kind, "candidate": kind,
            })
            cards[kind] = StrategyCard(
                strategy_id=strategy_id, strategy_kind=kind, stage_scope=_STRATEGY_STAGES[kind],
                **clean, origin_request_id=request_id, origin_status=source.status,
                origin_review_status=source.review_status, created_at=_utc_now(),
                origin_task_ids=[task["id"] for task in source.tasks if type(task["id"]) is int],
                topic_domain=normalize_topic_domain(candidate.topic_domain),
            )
        if not cards:
            return False
        embeddings = encode_texts(
            [card.to_document() for card in cards.values()],
            model_name=self._config.resolved_strategy_memory_embedding_model(),
        )
        if not embeddings or len(embeddings) != len(cards) or not all(embeddings):
            if observer:
                observer.record_degraded("strategy_memory_embedding_failed")
            return False

        current = self._strict_snapshot(request_id, "strategy_synthesis")
        if current is None or current["state_version"] != source_version or self._strict_source(current)[1] != checksum:
            return False
        collection = self._get_partition_collection(root)
        record_ids: dict[str, str] = {}
        metadata = []
        for card in cards.values():
            card_id = card.strategy_id
            old_record_id = old.get("cards", {}).get(card_id)
            previous = self._partition_card(root, card_id, old_record_id, request_id, old.get("source_version"))
            if previous:
                card.hit_count = max(0, _safe_int(previous.get("hit_count")) or 0)
                card.last_hit_at = str(previous.get("last_hit_at") or "")
                # Prior grades reviewed a different source version.
                if old.get("source_version") == source_version:
                    card.maturity = normalize_maturity(previous.get("maturity"))
            binding = {
                "tenant_id": self._scope.tenant_id, "policy_version": self._scope.policy_version,
                "source_version": source_version, "sanitization_version": _SANITIZATION_VERSION,
            }
            payload = {**card.to_json_dict(), **binding}
            record_id = "sr_" + _checksum_payload({
                key: value for key, value in payload.items()
                if key not in {"hit_count", "last_hit_at", "maturity", "created_at"}
            })
            payload["record_id"] = record_id
            write_partition_text(root, "cards", f"{record_id}.json", text=json.dumps(payload, ensure_ascii=False))
            record_ids[card_id] = record_id
            metadata.append({
                **binding, "record_id": record_id, "strategy_id": card_id, "origin_request_id": request_id,
            })
        collection.upsert(
            ids=list(record_ids.values()), documents=[card.to_document() for card in cards.values()],
            metadatas=metadata, embeddings=embeddings,
        )
        manifest["requests"][request_id] = {"source_version": source_version, "cards": record_ids}
        return True

    def _refresh_partition_request(self, request_id: str, *, observer: RequestTrace | None) -> None:
        if not self.enabled or not valid_internal_id(request_id):
            self._record_partition("write", "denied")
            return
        try:
            snapshot = self._strict_snapshot(request_id, "strategy_synthesis")
            if snapshot is None:
                self._record_partition("write", "denied")
                return
            root = self._partition_path(create=True)
            with self._lock, partition_lock(root):
                manifest = self._partition_manifest(root)
                previous = manifest["requests"].get(request_id, {}).get("cards", {})
                changed = self._stage_strategy_cards(root, manifest, snapshot, observer=observer)
                if changed:
                    self._save_partition_manifest(root, manifest)
                    self._delete_partition_records(
                        root, set(previous.values()) - set(manifest["requests"][request_id]["cards"].values()),
                    )
                self._record_partition("write", "ok" if changed else "empty")
        except Exception:
            self._record_partition("write", "unavailable")
            if observer:
                observer.record_degraded("strategy_memory_refresh_failed")

    def rebuild(self, *, dry_run: bool = True, limit: int = 16, after: str = "") -> dict[str, Any]:
        """Rebuild a bounded page from the authoritative terminal snapshot reader."""
        report = {"scanned": 0, "eligible": 0, "rebuilt": 0, "after": after, "unavailable": False}
        if (
            not self._strict or not self.enabled or type(limit) is not int or limit < 1
            or (after and not valid_internal_id(after))
        ):
            return report
        try:
            count = min(limit, _PARTITION_SAMPLE_LIMIT)
            rows = self._snapshot_reader.iter_terminal(self._scope, after=after, limit=count)
            ids = sorted({
                row["request_id"] for row in rows[:count]
                if isinstance(row, dict) and row.get("tenant_id") == self._scope.tenant_id
                and valid_internal_id(row.get("request_id")) and row["request_id"] > after
            })
            report["scanned"] = len(ids)
            report["after"] = ids[-1] if ids else after
            snapshots = []
            for request_id in ids:
                if not authorize(self._policy, self._scope, request_id, "maintenance"):
                    continue
                snapshot = self._strict_snapshot(request_id, "strategy_synthesis")
                if snapshot and self._strict_source(snapshot)[0]:
                    snapshots.append(snapshot)
            report["eligible"] = len(snapshots)
            if dry_run or not snapshots:
                return report
            root = self._partition_path(create=True)
            with self._lock, partition_lock(root):
                manifest = self._partition_manifest(root)
                obsolete: set[str] = set()
                for snapshot in snapshots:
                    request_id = snapshot["request_id"]
                    previous = manifest["requests"].get(request_id, {}).get("cards", {})
                    if self._stage_strategy_cards(root, manifest, snapshot):
                        report["rebuilt"] += 1
                        obsolete.update(
                            set(previous.values()) - set(manifest["requests"][request_id]["cards"].values()),
                        )
                if report["rebuilt"]:
                    self._save_partition_manifest(root, manifest)
                    self._delete_partition_records(root, obsolete)
            self._record_partition("rebuild", "ok" if report["rebuilt"] else "empty")
        except Exception:
            report["rebuilt"] = 0
            report["unavailable"] = True
            if not dry_run:
                self._record_partition("rebuild", "unavailable")
        return report

    def _delete_partition_records(self, root: Path, record_ids: set[str]) -> None:
        ids = sorted(record_id for record_id in record_ids if valid_internal_id(record_id))
        if not ids:
            return
        try:
            self._get_partition_collection(root).delete(ids=ids)
        except Exception:
            self._record_partition("delete", "unavailable")
            return
        for record_id in ids:
            try:
                remove_partition_file(root, "cards", f"{record_id}.json")
            except OSError:
                continue

    def retract_request(self, request_id: str, *, dry_run: bool = True) -> int:
        """Tombstone an authorized request before purging its staged records."""
        if not self._strict or not self.enabled or not authorize(
            self._policy, self._scope, request_id, "maintenance",
        ):
            return 0
        try:
            root = self._partition_path(create=not dry_run)
            if dry_run:
                return len(self._partition_manifest(root)["requests"].get(request_id, {}).get("cards", {}))
            with self._lock, partition_lock(root):
                if not authorize(self._policy, self._scope, request_id, "maintenance"):
                    return 0
                manifest = self._partition_manifest(root)
                cards = manifest["requests"].pop(request_id, {}).get("cards", {})
                pending = set(manifest["revoked"].get(request_id, ())) | set(cards.values())
                manifest["revoked"][request_id] = sorted(pending)
                self._save_partition_manifest(root, manifest)
                self._delete_partition_records(root, pending)
            self._record_partition("delete", "ok")
            return len(cards)
        except Exception:
            if not dry_run:
                self._record_partition("delete", "unavailable")
            return 0

    def _update_partition_cards(
        self, *, strategy_ids: list[str] | None = None, maturity: str | None = None,
        cutoff: datetime | None = None,
    ) -> list[str]:
        if not self.enabled:
            return []
        selected = {
            strategy_id for strategy_id in (strategy_ids or [])[:64] if valid_internal_id(strategy_id)
        }
        if not selected and cutoff is None:
            return []
        updated = []
        try:
            root = self._partition_path()
            if not root.is_dir():
                return []
            with self._lock, partition_lock(root):
                manifest = self._partition_manifest(root)
                for request_id, entry in manifest["requests"].items():
                    cards = entry["cards"]
                    if cutoff is None and not selected.intersection(cards):
                        continue
                    purpose: MemoryPurpose = "maintenance" if (cutoff is not None or maturity is not None) else "plan"
                    current = self._strict_snapshot(request_id, purpose)
                    if current is None and purpose == "plan":
                        current = self._strict_snapshot(request_id, "task")
                    if current is None or current["state_version"] != entry["source_version"]:
                        continue
                    for card_id, record_id in cards.items():
                        if cutoff is None and card_id not in selected:
                            continue
                        card = self._partition_card(root, card_id, record_id, request_id, entry["source_version"])
                        if card is None:
                            continue
                        if cutoff is not None:
                            last_use = _parse_timestamp(card["last_hit_at"]) or _parse_timestamp(card["created_at"])
                            if card["maturity"] != MATURITY_DRAFT or last_use is None or last_use >= cutoff:
                                continue
                            card["maturity"] = MATURITY_ARCHIVED
                        elif maturity is not None:
                            card["maturity"] = normalize_maturity(maturity)
                        else:
                            card["hit_count"] = max(0, _safe_int(card["hit_count"]) or 0) + 1
                            card["last_hit_at"] = _utc_now()
                        write_partition_text(
                            root, "cards", f"{record_id}.json", text=json.dumps(card, ensure_ascii=False),
                        )
                        updated.append(card_id)
            self._record_partition("write", "ok" if updated else "empty")
        except (OSError, ValueError, TypeError, KeyError):
            self._record_partition("write", "unavailable")
        return updated

    def _search_partition(
        self, query_text: str, *, stage: str, top_k: int, current_request_id: str | None,
        observer: RequestTrace | None, topic_domains: set[str] | None,
    ) -> str:
        if not self.enabled or current_request_id not in (None, self._scope.request_id):
            self._record_partition("read", "denied")
            return ""
        if not _normalize_text(query_text):
            return ""
        if observer is not None and (
            observer.tenant_id != self._scope.tenant_id or observer.request_id != self._scope.request_id
        ):
            observer = None
        purpose: MemoryPurpose = "plan" if stage == "planning" else "task"
        try:
            root = self._partition_path()
            manifest = self._partition_manifest(root)
            candidates = {}
            for request_id, entry in manifest["requests"].items():
                if request_id == self._scope.request_id or request_id in manifest["revoked"]:
                    continue
                current = self._strict_snapshot(request_id, purpose)
                if current is None or current["state_version"] != entry["source_version"]:
                    continue
                for card_id, record_id in entry["cards"].items():
                    card = self._partition_card(root, card_id, record_id, request_id, entry["source_version"])
                    if card and self._eligible_partition_card(card, stage, topic_domains):
                        candidates[record_id] = card
            hits = self._rank_partition_cards(root, query_text, candidates, stage, top_k, observer)
            visible = []
            for hit, source_version in hits:
                latest = self._strict_snapshot(hit.origin_request_id, purpose)
                if latest is not None and latest["state_version"] == source_version:
                    visible.append(hit)
            # Recheck manifest after ranking so a concurrent local tombstone is honored.
            if manifest != self._partition_manifest(root) or not scope_is_current(self._policy, self._scope):
                return ""
            if observer:
                observer.record_strategy_memory_query(
                    hit_count=len(visible), match_kinds=[hit.strategy_kind for hit in visible],
                    strategy_ids=[hit.strategy_id for hit in visible], internal_ids=True,
                )
            rendered = self._render_prompt_context(visible, stage=stage)
            if rendered and observer:
                observer.record_strategy_memory_prompt_injection(
                    match_kinds=[hit.strategy_kind for hit in visible],
                )
            self._record_partition("read", "ok" if rendered else "empty")
            return rendered
        except Exception:
            self._record_partition("read", "unavailable")
            return ""

    def _eligible_partition_card(
        self, card: dict[str, Any], stage: str, topic_domains: set[str] | None,
    ) -> bool:
        kind = card.get("strategy_kind")
        return (
            kind in _STRATEGY_STAGES
            and (kind == "anti_pattern" or _STRATEGY_STAGES[kind] == stage)
            and maturity_meets_gate(card.get("maturity"), self._config.memory_maturity_gate)
            and (not topic_domains or card.get("topic_domain") in topic_domains)
        )

    def _rank_partition_cards(
        self, root: Path, query_text: str, candidates: dict[str, dict[str, Any]],
        stage: str, top_k: int, observer: RequestTrace | None,
    ) -> list[tuple[StrategyHit, int]]:
        if not candidates:
            return []
        query_embedding = self._encode_query(query_text, observer=observer)
        if query_embedding is None:
            return []
        top_k = max(1, int(top_k or 1))
        result = self._get_partition_collection(root).query(
            query_embeddings=[query_embedding], n_results=min(len(candidates), top_k * _DEFAULT_QUERY_MULTIPLIER),
            include=["distances", "metadatas"], where={"record_id": {"$in": sorted(candidates)}},
        )
        matches = []
        seen = set()
        binding = ("tenant_id", "policy_version", "source_version", "sanitization_version", "strategy_id",
                   "origin_request_id", "record_id")
        for record_id, metadata, distance in zip(
            (result.get("ids") or [[]])[0], (result.get("metadatas") or [[]])[0],
            (result.get("distances") or [[]])[0],
        ):
            card = candidates.get(record_id)
            if (
                card is None or record_id in seen or not isinstance(metadata, dict)
                or any(metadata.get(key) != card[key] for key in binding)
                or type(distance) not in (int, float) or not math.isfinite(distance)
            ):
                continue
            seen.add(record_id)
            similarity = min(1.0, max(0.0, 1.0 - distance))
            maturity = normalize_maturity(card["maturity"])
            origin_status = _safe_status(card["origin_status"])
            kind = card["strategy_kind"]
            hit = StrategyHit(
                strategy_id=card["strategy_id"], strategy_kind=kind, stage_scope=card["stage_scope"],
                **clean_strategy_fields(card), origin_request_id=card["origin_request_id"],
                origin_status=origin_status, origin_review_status=card["origin_review_status"],
                similarity=similarity, maturity=maturity, topic_domain=card["topic_domain"], match_reason="",
                score=self._score_hit(
                    similarity=similarity, stage=stage, strategy_kind=kind, origin_status=origin_status,
                    maturity=maturity,
                ),
            )
            hit.match_reason = self._build_match_reason(hit, stage=stage)
            matches.append((hit, card["source_version"]))
        matches.sort(key=lambda pair: (-pair[0].score, -pair[0].similarity, pair[0].strategy_id))
        return matches[:top_k]

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

        # Carry the existing governance grade and usage counters over before the old card files
        # are removed, otherwise re-synthesising a request would demote a promoted card back to
        # draft and wipe the evidence of it ever having been useful.
        for card, _ in valid_pairs:
            card.maturity = self._read_card_maturity(card.strategy_id)
            card.hit_count, card.last_hit_at = self._read_card_usage(card.strategy_id)

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

    def _read_card_maturity(self, strategy_id: str) -> str:
        """Return the maturity already recorded on disk for this card id.

        Re-synthesis rewrites a card from scratch, which would silently reset a promotion.
        The on-disk card file is therefore the durable record of governance grade and is
        read back before the card is re-indexed.
        """

        return normalize_maturity(self._read_card_payload(strategy_id).get("maturity"))

    def _read_card_payload(self, strategy_id: str) -> dict[str, Any]:
        """Load a stored card file, returning an empty mapping when unreadable."""

        try:
            payload = json.loads(self._card_path(strategy_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _read_card_usage(self, strategy_id: str) -> tuple[int, str]:
        """Return the persisted ``(hit_count, last_hit_at)`` for this card id."""

        payload = self._read_card_payload(strategy_id)
        try:
            hit_count = max(0, int(payload.get("hit_count") or 0))
        except (TypeError, ValueError):
            hit_count = 0
        return hit_count, str(payload.get("last_hit_at") or "")

    def record_card_hits(self, strategy_ids: list[str]) -> int:
        """Flush per-request usage onto the stored cards in a single batch.

        Called once when a request finishes rather than on every retrieval, so the hot query
        path stays free of writes. Usage is what later distinguishes a card that is genuinely
        useful from one that merely exists.
        """

        if self._strict:
            return len(self._update_partition_cards(strategy_ids=strategy_ids))
        updated = 0
        timestamp = _utc_now()
        for strategy_id in dict.fromkeys(_normalize_list(strategy_ids or [], max_items=64)):
            payload = self._read_card_payload(strategy_id)
            if not payload:
                continue
            try:
                current = max(0, int(payload.get("hit_count") or 0))
            except (TypeError, ValueError):
                current = 0
            payload["hit_count"] = current + 1
            payload["last_hit_at"] = timestamp
            try:
                self._card_path(strategy_id).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:  # pragma: no cover - defensive filesystem guard
                continue
            updated += 1
        return updated

    def archive_stale_cards(self, *, now: datetime | None = None) -> list[str]:
        """Archive draft cards that have gone unused past the configured idle window.

        Only drafts decay. A verified or promoted card represents an explicit decision and is
        never archived automatically. Cards never hit fall back to their creation time.
        """

        idle_days = max(0, int(self._config.memory_decay_idle_days or 0))
        if idle_days <= 0:
            return []

        reference = now or datetime.now(timezone.utc)
        cutoff = reference - timedelta(days=idle_days)
        if self._strict:
            return self._update_partition_cards(cutoff=cutoff)
        archived: list[str] = []
        if not self._cards_dir.is_dir():
            return []

        for path in sorted(self._cards_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if normalize_maturity(payload.get("maturity")) != MATURITY_DRAFT:
                continue

            last_used = _parse_timestamp(payload.get("last_hit_at")) or _parse_timestamp(
                payload.get("created_at")
            )
            if last_used is None or last_used >= cutoff:
                continue

            strategy_id = _normalize_text(payload.get("strategy_id"))
            if strategy_id and self.set_card_maturity(strategy_id, MATURITY_ARCHIVED):
                archived.append(strategy_id)
        return archived

    def run_maintenance(
        self,
        strategy_ids: list[str] | None = None,
        *,
        observer: RequestTrace | None = None,
    ) -> dict[str, Any]:
        """Single post-request pass: persist usage, then decay unused drafts."""

        if not self.enabled:
            return {"recorded_hits": 0, "archived": []}

        recorded = self.record_card_hits(strategy_ids or [])
        archived = self.archive_stale_cards()
        if archived:
            metrics_registry.increment("strategy_memory_archived_card_total", len(archived))
            if observer:
                observer.record_degraded(f"strategy_memory_cards_archived:{len(archived)}")
        return {"recorded_hits": recorded, "archived": archived}

    def set_card_maturity(self, strategy_id: str, maturity: str) -> bool:
        """Promote or demote a stored card, updating both the card file and the index.

        Returns ``False`` when the card is unknown so callers can distinguish a no-op from a
        successful grade change.
        """

        if self._strict:
            return bool(self._update_partition_cards(strategy_ids=[strategy_id], maturity=maturity))
        normalized_id = _normalize_text(strategy_id)
        if not normalized_id:
            return False
        normalized_maturity = normalize_maturity(maturity)
        if normalized_maturity not in MATURITY_ORDER:  # pragma: no cover - defensive
            return False

        path = self._card_path(normalized_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False

        payload["maturity"] = normalized_maturity
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        collection = self._get_collection()
        if collection is not None and hasattr(collection, "update"):
            try:
                collection.update(ids=[normalized_id], metadatas=[{"maturity": normalized_maturity}])
            except Exception:  # pragma: no cover - depends on runtime vector backend
                logger.debug("failed to update strategy card maturity id=%s", normalized_id)
        return True

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
        topic_domains: set[str] | None = None,
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
            maturity = normalize_maturity(metadata.get("maturity"))
            if not maturity_meets_gate(maturity, self._config.memory_maturity_gate):
                continue
            topic_domain = normalize_topic_domain(metadata.get("topic_domain"))
            if topic_domains and topic_domain not in topic_domains:
                continue
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
                    maturity=maturity,
                ),
                match_reason="",
                maturity=maturity,
                topic_domain=topic_domain,
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
        maturity: str = MATURITY_DRAFT,
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
        score += MATURITY_RETRIEVAL_WEIGHTS.get(normalize_maturity(maturity), 0.0)
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

        if hit.maturity == MATURITY_DRAFT:
            parts.append("未经验证草稿")
        elif hit.maturity == "promoted":
            parts.append("已验证并采纳")
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
            action_line = sanitize_memory_text("；".join(hit.recommended_actions[:3])) or "暂无明确推荐动作"
            query_line = sanitize_memory_text("；".join(hit.query_templates[:2])) or "暂无固定检索模板"
            pitfall_line = sanitize_memory_text("；".join(hit.pitfalls_to_avoid[:2])) or "暂无显式反模式"
            applicable_when = sanitize_memory_text(hit.applicable_when)
            signal_line = sanitize_memory_text("；".join(hit.match_signals[:3])) or truncate_text(
                applicable_when,
                100,
            )
            block = "\n".join(
                [
                    f"- 标题：{sanitize_memory_text(hit.title)}",
                    f"  类型：{sanitize_memory_text(hit.strategy_kind)}；匹配原因：{sanitize_memory_text(hit.match_reason)}",
                    f"  适用场景：{truncate_text(applicable_when, 160)}",
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
