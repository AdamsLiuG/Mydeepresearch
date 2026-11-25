"""Offline/bounded tenant memory rebuild, graph batch reconciliation, and origin purge utilities."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Literal

from config import Configuration
from metrics import metrics_registry
from services.evidence import build_graph_evidence
from services.knowledge_graph import KnowledgeGraphService, build_graph_store
from services.memory_scope import MemoryPolicy, MemoryScope, valid_internal_id
from services.storage_backends import (
    JsonStrictSnapshotReader,
    PostgresRequestStateStore,
    StrictSnapshotReader,
    build_request_state_store,
    build_strict_snapshot_reader,
)
from state_manager import (
    RequestStateGraphAuthority,
    TenantSnapshotMemoryPolicy,
    record_v4_graph_ack,
)

_MAX_PAGE_LIMIT = 64


def _pending_age_bucket(recorded_at: Any, now: datetime) -> str:
    if isinstance(recorded_at, str) and recorded_at.strip():
        try:
            dt = datetime.fromisoformat(recorded_at.strip())
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta_seconds = max(0.0, (now - dt).total_seconds())
            if delta_seconds < 60:
                return "<1m"
            if delta_seconds < 600:
                return "<10m"
            if delta_seconds < 3600:
                return "<1h"
            return "≥1h"
        except ValueError:
            pass
    return "<1m"


def _oldest_bucket(buckets: list[str]) -> str:
    order = ("<1m", "<10m", "<1h", "≥1h")
    best = 0
    for item in buckets:
        if item in order:
            best = max(best, order.index(item))
    return order[best]


def _resolve_reader(
    config: Configuration,
    *,
    snapshot_reader: StrictSnapshotReader | None = None,
    state_store: Any | None = None,
) -> tuple[StrictSnapshotReader | None, Any | None]:
    owned_store = None
    reader = snapshot_reader
    store = state_store
    if reader is None:
        if store is None:
            store = build_request_state_store(config)
            owned_store = store
        if isinstance(store, PostgresRequestStateStore):
            reader = build_strict_snapshot_reader(config, postgres_store=store)
        elif config.request_state_enabled:
            reader = JsonStrictSnapshotReader(
                getattr(store, "_directory", None) or getattr(config, "request_state_dir", "./.state/requests")
            )
    return reader, owned_store


def _maintenance_scope(tenant_id: str, request_id: str = "maintenance") -> MemoryScope:
    return MemoryScope.trusted(tenant_id=tenant_id, request_id=request_id, owner_epoch=None, policy_version=1)


def reconcile_pending_batches(
    config: Configuration,
    *,
    tenant_id: str,
    limit: int = 16,
    after: str = "",
    dry_run: bool = True,
    now: datetime | None = None,
    deadline_seconds: float | None = None,
    store: Any | None = None,
    snapshot_reader: StrictSnapshotReader | None = None,
    state_store: Any | None = None,
    policy: MemoryPolicy | None = None,
) -> dict[str, Any]:
    """Scan terminal tenant snapshots and apply pending knowledge-graph batches within bounds."""
    result: dict[str, Any] = {
        "tenant_id": tenant_id,
        "dry_run": bool(dry_run),
        "scanned": 0,
        "pending": 0,
        "applied": 0,
        "already_applied": 0,
        "errors": 0,
        "next_after": after,
        "outcomes": [],
    }
    if not valid_internal_id(tenant_id) or not config.knowledge_graph_enabled_for(tenant_id):
        return result
    if type(limit) is not int or limit < 1:
        return result

    bounded_limit = min(limit, _MAX_PAGE_LIMIT)
    reader, owned_store = _resolve_reader(config, snapshot_reader=snapshot_reader, state_store=state_store)
    effective_store = state_store or owned_store
    graph_store = store
    owned_graph_store = None
    if graph_store is None and not dry_run:
        graph_store = build_graph_store(config)
        owned_graph_store = graph_store

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    try:
        if reader is None:
            return result
        maintenance_scope = _maintenance_scope(tenant_id, "maintenance-reconcile")
        active_policy = policy or TenantSnapshotMemoryPolicy(tenant_id=tenant_id, snapshot_reader=reader)
        rows = reader.iter_terminal(maintenance_scope, after=after, limit=bounded_limit)
        result["scanned"] = len(rows)
        if rows:
            last_id = rows[-1].get("request_id")
            if isinstance(last_id, str):
                result["next_after"] = last_id

        pending_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        pending_buckets: list[str] = []
        deferred_buckets: list[str] = []
        pending_count = 0
        deferred_count = 0

        for row in rows:
            if not isinstance(row, dict) or row.get("tenant_id") != tenant_id:
                continue
            kg = (row.get("v4") or {}).get("knowledge_graph")
            if not isinstance(kg, dict):
                continue
            raw_snap = kg.get("terminal_snapshot") if isinstance(kg.get("terminal_snapshot"), dict) else kg
            batch = raw_snap.get("batch") if isinstance(raw_snap, dict) else None
            batch_id = kg.get("batch_id") or (batch.get("batch_id") if isinstance(batch, dict) else None)
            digest = kg.get("digest") or (batch.get("digest") if isinstance(batch, dict) else None)
            if not batch_id or not digest or kg.get("ack_digest") == digest:
                continue
            pending_entries.append((row, kg))
            bucket = _pending_age_bucket(kg.get("recorded_at"), current_time)
            if kg.get("ack_status") in {"error", "timeout", "stale"}:
                deferred_count += 1
                deferred_buckets.append(bucket)
            else:
                pending_count += 1
                pending_buckets.append(bucket)

        result["pending"] = len(pending_entries)
        metrics_registry.record_v4_graph_pending(
            "pending", _oldest_bucket(pending_buckets), min(16, pending_count), tenant_id=tenant_id,
        )
        metrics_registry.record_v4_graph_pending(
            "retry_deferred", _oldest_bucket(deferred_buckets), min(16, deferred_count), tenant_id=tenant_id,
        )
        if dry_run or not pending_entries or graph_store is None:
            return result

        budget_sec = float(deadline_seconds or (config.v4_graph_request_timeout_ms / 1000.0) or 0.3)
        deadline = time.monotonic() + max(0.1, budget_sec)
        authority = RequestStateGraphAuthority(
            snapshot_reader=reader, policy=active_policy, config=config,
        )
        service = KnowledgeGraphService(
            graph_store,
            authority,
            min_entailment=config.v4_graph_min_entailment,
            nli_entailment_threshold=max(config.nli_entailment_threshold, config.v4_graph_min_entailment),
            max_candidates=config.v4_graph_max_candidates,
            max_projection_tokens=config.v4_graph_max_projection_tokens,
            query_timeout_ms=config.v4_graph_query_timeout_ms,
            tenants=tuple(config.v4_knowledge_graph_tenants) or None,
            metrics=metrics_registry,
        )

        remaining_pending = pending_count
        for row, kg in pending_entries:
            if time.monotonic() >= deadline:
                result["errors"] += 1
                result["outcomes"].append({"request_id": row.get("request_id"), "outcome": "timeout"})
                break
            req_id = str(row.get("request_id") or "")
            raw_snap = kg.get("terminal_snapshot") if isinstance(kg.get("terminal_snapshot"), dict) else kg
            batch = raw_snap.get("batch") or {}
            batch_id = str(kg.get("batch_id") or batch.get("batch_id") or "")
            digest = str(kg.get("digest") or batch.get("digest") or "")
            owner_epoch = raw_snap.get("owner_epoch")
            req_scope = MemoryScope.trusted(
                tenant_id=tenant_id,
                request_id=req_id,
                owner_epoch=int(owner_epoch) if owner_epoch is not None else None,
                policy_version=int(raw_snap.get("policy_version") or 1),
            )
            apply_res = service.apply_terminal_batch(req_scope, batch_id, deadline=deadline)
            result["outcomes"].append({
                "request_id": req_id,
                "batch_id": batch_id,
                "outcome": apply_res.outcome,
                "inserted": apply_res.inserted,
            })
            if apply_res.outcome in {"applied", "duplicate", "volatile_applied"}:
                if apply_res.outcome == "duplicate":
                    result["already_applied"] += 1
                else:
                    result["applied"] += 1
                remaining_pending = max(0, remaining_pending - 1)
                if effective_store is not None and apply_res.watermark is not None:
                    record_v4_graph_ack(
                        effective_store,
                        tenant_id=tenant_id,
                        request_id=req_id,
                        batch_id=batch_id,
                        digest=digest,
                        revision=apply_res.watermark.graph_revision,
                        status=apply_res.outcome,
                    )
            else:
                result["errors"] += 1
        metrics_registry.record_v4_graph_pending(
            "pending", _oldest_bucket(pending_buckets), min(16, remaining_pending), tenant_id=tenant_id,
        )
        return result
    finally:
        for res in (owned_graph_store, owned_store):
            close = getattr(res, "close", None)
            if callable(close):
                close()


def rebuild_tenant_memory(
    config: Configuration,
    *,
    tenant_id: str,
    target: Literal["all", "notes", "strategy", "evidence"] = "all",
    limit: int = 16,
    after: str = "",
    dry_run: bool = True,
    snapshot_reader: StrictSnapshotReader | None = None,
    policy: MemoryPolicy | None = None,
    evidence_sources: list[Any] | None = None,
) -> dict[str, Any]:
    """Rebuild partitioned Note, Strategy, and Evidence indices from authoritative tenant sources."""
    report: dict[str, Any] = {"tenant_id": tenant_id, "target": target, "dry_run": bool(dry_run)}
    if not valid_internal_id(tenant_id) or not config.v4_tenant_memory_partition_enabled:
        return report
    if type(limit) is not int or limit < 1:
        return report

    from services.evidence_index import EvidenceArchiveSource, EvidenceMemoryService
    from services.note_memory import NoteMemoryService
    from services.strategy_memory import StrategyMemoryService

    reader, owned_store = _resolve_reader(config, snapshot_reader=snapshot_reader)
    active_policy = policy or TenantSnapshotMemoryPolicy(tenant_id=tenant_id, snapshot_reader=reader)
    scope = _maintenance_scope(tenant_id, "maintenance-rebuild")
    bounded = min(limit, _MAX_PAGE_LIMIT)

    try:
        if target in {"all", "notes"}:
            note_svc = NoteMemoryService(config, scope=scope, policy=active_policy, frozen_strict=True)
            report["notes"] = note_svc.rebuild(dry_run=dry_run, limit=bounded, after=after)
        if target in {"all", "strategy"}:
            strat_svc = StrategyMemoryService(
                config, scope=scope, policy=active_policy, snapshot_reader=reader, frozen_strict=True,
            )
            report["strategy"] = strat_svc.rebuild(dry_run=dry_run, limit=bounded, after=after)
        if target in {"all", "evidence"}:
            ev_svc = EvidenceMemoryService(config, scope=scope, policy=active_policy, frozen_strict=True)
            sources = list(evidence_sources or ())
            if not sources and reader is not None:
                for row in reader.iter_terminal(scope, after=after, limit=bounded):
                    req_id = str(row.get("request_id") or "")
                    for task in row.get("todo_items") or ():
                        if not isinstance(task, dict):
                            continue
                        task_id = int(task.get("id") or 0)
                        for item in task.get("evidence_items") or ():
                            if not isinstance(item, dict):
                                continue
                            if build_graph_evidence(item, tenant_id=tenant_id, request_id=req_id) is None:
                                continue
                            sources.append(EvidenceArchiveSource(request_id=req_id, task_id=task_id, record=item))
            report["evidence"] = ev_svc.rebuild(sources, dry_run=dry_run, limit=bounded)
        return report
    finally:
        if owned_store is not None and hasattr(owned_store, "close"):
            owned_store.close()


def purge_origin_data(
    config: Configuration,
    *,
    tenant_id: str,
    origin_request_id: str,
    limit: int = 64,
    dry_run: bool = True,
    targets: tuple[str, ...] = ("graph", "notes", "strategy", "evidence"),
    store: Any | None = None,
    snapshot_reader: StrictSnapshotReader | None = None,
    policy: MemoryPolicy | None = None,
    deleted_origins: set[str] | None = None,
) -> dict[str, Any]:
    """Boundedly purge or tombstone one deleted origin request across graph and memory partitions."""
    report: dict[str, Any] = {
        "tenant_id": tenant_id,
        "origin_request_id": origin_request_id,
        "dry_run": bool(dry_run),
        "graph_removed": 0,
        "notes_removed": 0,
        "strategy_removed": 0,
        "evidence_removed": 0,
    }
    if not valid_internal_id(tenant_id) or not valid_internal_id(origin_request_id):
        return report

    from services.evidence_index import EvidenceMemoryService
    from services.note_memory import NoteMemoryService
    from services.strategy_memory import StrategyMemoryService

    reader, owned_store = _resolve_reader(config, snapshot_reader=snapshot_reader)
    active_policy = policy or TenantSnapshotMemoryPolicy(
        tenant_id=tenant_id,
        snapshot_reader=reader,
        deleted_origins=frozenset(deleted_origins) if deleted_origins is not None else frozenset(),
    )
    scope = _maintenance_scope(tenant_id, "maintenance-purge")
    graph_store = store
    owned_graph_store = None
    if graph_store is None and "graph" in targets and config.knowledge_graph_enabled_for(tenant_id):
        graph_store = build_graph_store(config)
        owned_graph_store = graph_store

    try:
        if "graph" in targets and graph_store is not None and config.knowledge_graph_enabled_for(tenant_id):
            authority = RequestStateGraphAuthority(
                snapshot_reader=reader, policy=active_policy, config=config,
                deleted_origins=frozenset(deleted_origins) if deleted_origins is not None else frozenset(),
            )
            service = KnowledgeGraphService(
                graph_store,
                authority,
                min_entailment=config.v4_graph_min_entailment,
                nli_entailment_threshold=max(config.nli_entailment_threshold, config.v4_graph_min_entailment),
                tenants=tuple(config.v4_knowledge_graph_tenants) or None,
                metrics=metrics_registry,
            )
            report["graph_removed"] = service.purge_origin(
                scope,
                origin_request_id,
                deadline=time.monotonic() + max(0.1, config.v4_graph_request_timeout_ms / 1000.0),
                dry_run=dry_run,
                limit=min(limit, 256),
            )
        if config.v4_tenant_memory_partition_enabled:
            if "notes" in targets:
                note_svc = NoteMemoryService(config, scope=scope, policy=active_policy, frozen_strict=True)
                report["notes_removed"] = note_svc.retract_request(origin_request_id, dry_run=dry_run, limit=min(limit, 64))
            if "strategy" in targets:
                strat_svc = StrategyMemoryService(
                    config, scope=scope, policy=active_policy, snapshot_reader=reader, frozen_strict=True,
                )
                report["strategy_removed"] = strat_svc.retract_request(origin_request_id, dry_run=dry_run)
            if "evidence" in targets:
                ev_svc = EvidenceMemoryService(config, scope=scope, policy=active_policy, frozen_strict=True)
                ev_res = ev_svc.retract(origin_request_id, dry_run=dry_run, limit=min(limit, 64))
                report["evidence_removed"] = int(ev_res.get("removed") or 0)
        return report
    finally:
        for res in (owned_graph_store, owned_store):
            close = getattr(res, "close", None)
            if callable(close):
                close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepResearch tenant graph and memory maintenance CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("reconcile", help="Reconcile pending knowledge-graph batches from terminal snapshots")
    rec.add_argument("--tenant", required=True, help="Tenant identifier")
    rec.add_argument("--after", default="", help="Pagination cursor request_id")
    rec.add_argument("--limit", type=int, default=16, help="Maximum batches per page (<=64)")
    rec.add_argument("--apply", action="store_true", help="Apply mutations (default is dry-run)")

    reb = sub.add_parser("rebuild-memory", help="Rebuild partitioned tenant memory indices")
    reb.add_argument("--tenant", required=True, help="Tenant identifier")
    reb.add_argument("--target", choices=("all", "notes", "strategy", "evidence"), default="all")
    reb.add_argument("--after", default="", help="Pagination cursor request_id")
    reb.add_argument("--limit", type=int, default=16)
    reb.add_argument("--apply", action="store_true", help="Apply mutations (default is dry-run)")

    prg = sub.add_parser("purge", help="Purge deleted origin request records from graph and partitions")
    prg.add_argument("--tenant", required=True, help="Tenant identifier")
    prg.add_argument("--origin-request-id", required=True, help="Deleted request identifier")
    prg.add_argument("--limit", type=int, default=64)
    prg.add_argument("--apply", action="store_true", help="Apply deletions (default is dry-run)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Configuration.from_env()
    dry_run = not bool(args.apply)
    if args.command == "reconcile":
        output = reconcile_pending_batches(
            config, tenant_id=args.tenant, limit=args.limit, after=args.after, dry_run=dry_run,
        )
    elif args.command == "rebuild-memory":
        output = rebuild_tenant_memory(
            config, tenant_id=args.tenant, target=args.target, limit=args.limit, after=args.after, dry_run=dry_run,
        )
    elif args.command == "purge":
        output = purge_origin_data(
            config, tenant_id=args.tenant, origin_request_id=args.origin_request_id, limit=args.limit, dry_run=dry_run,
        )
    else:  # pragma: no cover
        return 2
    sys.stdout.write(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
