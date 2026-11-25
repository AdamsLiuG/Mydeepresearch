"""Pluggable persistent request state backends (JSON file store and optional PostgreSQL)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

from services.request_state import RequestStateStore, SnapshotConflictError

try:  # pragma: no cover - optional dependency
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from psycopg_pool import ConnectionPool as _PsycopgConnectionPool
except Exception:  # pragma: no cover - optional dependency
    _PsycopgConnectionPool = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PostgresRequestStateStore:
    """Optional PostgreSQL snapshot store using strictly parameterized SQL queries."""

    _CREATE_TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS deepresearch_request_states ("
        "request_id TEXT PRIMARY KEY, "
        "tenant_id TEXT NOT NULL DEFAULT 'default', "
        "topic TEXT NOT NULL DEFAULT '', "
        "phase TEXT NOT NULL DEFAULT 'unknown', "
        "status TEXT NOT NULL DEFAULT 'unknown', "
        "state_version BIGINT NOT NULL DEFAULT 1, "
        "runner_backend TEXT NOT NULL DEFAULT 'inline', "
        "workflow_run_id TEXT, "
        "updated_at TEXT NOT NULL, "
        "payload_json TEXT NOT NULL"
        ")"
    )
    _MIGRATE_COLUMNS_SQL = (
        "ALTER TABLE deepresearch_request_states "
        "ADD COLUMN IF NOT EXISTS state_version BIGINT NOT NULL DEFAULT 1",
        "ALTER TABLE deepresearch_request_states "
        "ADD COLUMN IF NOT EXISTS runner_backend TEXT NOT NULL DEFAULT 'inline'",
        "ALTER TABLE deepresearch_request_states ADD COLUMN IF NOT EXISTS workflow_run_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_drs_tenant_updated "
        "ON deepresearch_request_states (tenant_id, updated_at DESC)",
    )
    _UPSERT_SQL = (
        "INSERT INTO deepresearch_request_states "
        "(request_id, tenant_id, topic, phase, status, runner_backend, workflow_run_id, updated_at, payload_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (request_id) DO UPDATE SET "
        "topic = EXCLUDED.topic, "
        "phase = EXCLUDED.phase, "
        "status = EXCLUDED.status, "
        "state_version = deepresearch_request_states.state_version + 1, "
        "runner_backend = EXCLUDED.runner_backend, "
        "workflow_run_id = EXCLUDED.workflow_run_id, "
        "updated_at = EXCLUDED.updated_at, "
        "payload_json = EXCLUDED.payload_json "
        "WHERE deepresearch_request_states.tenant_id = EXCLUDED.tenant_id "
        "RETURNING state_version"
    )
    _INSERT_CAS_SQL = (
        "INSERT INTO deepresearch_request_states "
        "(request_id, tenant_id, topic, phase, status, state_version, runner_backend, workflow_run_id, updated_at, payload_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (request_id) DO NOTHING"
    )
    _UPDATE_CAS_SQL = (
        "UPDATE deepresearch_request_states SET "
        "tenant_id = %s, topic = %s, phase = %s, status = %s, "
        "state_version = %s, runner_backend = %s, workflow_run_id = %s, "
        "updated_at = %s, payload_json = %s "
        "WHERE tenant_id = %s AND request_id = %s AND state_version = %s"
    )
    _SELECT_BY_ID_SQL = (
        "SELECT tenant_id, payload_json, state_version FROM deepresearch_request_states WHERE request_id = %s"
    )
    _SELECT_BY_ID_TENANT_SQL = (
        "SELECT tenant_id, payload_json, state_version FROM deepresearch_request_states "
        "WHERE request_id = %s AND tenant_id = %s"
    )
    _LIST_ALL_SQL = (
        "SELECT payload_json FROM deepresearch_request_states ORDER BY updated_at DESC LIMIT %s"
    )
    _LIST_TENANT_SQL = (
        "SELECT payload_json FROM deepresearch_request_states "
        "WHERE tenant_id = %s ORDER BY updated_at DESC LIMIT %s"
    )

    def __init__(
        self,
        *,
        dsn: str,
        recent_limit: int = 50,
        connection_factory: Callable[[], Any] | None = None,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        self._dsn = str(dsn or "").strip()
        self._recent_limit = max(1, int(recent_limit or 1))
        self._connection_factory = connection_factory
        self._schema_ready = False
        self._schema_lock = Lock()
        self._pool: Any = None
        if (
            self._connection_factory is None
            and self._dsn
            and _PsycopgConnectionPool is not None
        ):
            try:
                self._pool = _PsycopgConnectionPool(
                    self._dsn,
                    min_size=max(1, int(pool_min_size or 2)),
                    max_size=max(1, int(pool_max_size or 10)),
                    open=False,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Failed to prepare psycopg_pool; using direct connections (%s)", type(exc).__name__)
                self._pool = None

    def close(self) -> None:
        if self._pool is not None and hasattr(self._pool, "close"):
            try:
                self._pool.close()
            except Exception:
                pass

    def _connect(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory()
        if self._pool is not None:
            if self._pool.closed:
                self._pool.open(wait=False)
            return self._pool.connection()
        if psycopg is None:
            raise RuntimeError("psycopg package is required for PostgresRequestStateStore")
        return psycopg.connect(self._dsn)

    def _ensure_schema(self, conn: Any) -> None:
        with self._schema_lock:
            if self._schema_ready:
                return
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(hashtext('deepresearch_request_states_schema'))")
                cur.execute(self._CREATE_TABLE_SQL)
                for migration_sql in self._MIGRATE_COLUMNS_SQL:
                    cur.execute(migration_sql)
            conn.commit()
            self._schema_ready = True

    def save(
        self,
        request_id: str,
        payload: dict[str, Any],
        *,
        tenant_id: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        stored = dict(payload)
        stored["request_id"] = request_id
        resolved_tenant = str(tenant_id or stored.get("tenant_id") or "default").strip() or "default"
        runner_backend = str(stored.get("runner_backend") or "inline").strip() or "inline"
        workflow_run_id = stored.get("workflow_run_id")
        stored["tenant_id"] = resolved_tenant
        stored["runner_backend"] = runner_backend
        stored["updated_at"] = _utc_now()
        next_version = int(expected_version) + 1 if expected_version is not None else 1
        # The row's version is authoritative, including for legacy payloads.
        stored.pop("state_version", None)
        serialized = json.dumps(stored, ensure_ascii=False)

        with self._connect() as conn:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                if expected_version is None:
                    cur.execute(
                        self._UPSERT_SQL,
                        (
                            request_id,
                            resolved_tenant,
                            str(stored.get("topic") or ""),
                            str(stored.get("phase") or "unknown"),
                            str(stored.get("status") or "unknown"),
                            runner_backend,
                            workflow_run_id,
                            stored["updated_at"],
                            serialized,
                        ),
                    )
                    version_row = cur.fetchone()
                    if version_row is None:
                        conn.rollback()
                        raise SnapshotConflictError(f"postgres snapshot owner conflict for request_id={request_id}")
                    next_version = version_row[0]
                elif int(expected_version) == 0:
                    cur.execute(
                        self._INSERT_CAS_SQL,
                        (
                            request_id,
                            resolved_tenant,
                            str(stored.get("topic") or ""),
                            str(stored.get("phase") or "unknown"),
                            str(stored.get("status") or "unknown"),
                            next_version,
                            runner_backend,
                            workflow_run_id,
                            stored["updated_at"],
                            serialized,
                        ),
                    )
                    if getattr(cur, "rowcount", 1) == 0:
                        conn.rollback()
                        raise SnapshotConflictError(
                            f"postgres snapshot conflict for request_id={request_id} expected_version=0"
                        )
                else:
                    cur.execute(
                        self._UPDATE_CAS_SQL,
                        (
                            resolved_tenant,
                            str(stored.get("topic") or ""),
                            str(stored.get("phase") or "unknown"),
                            str(stored.get("status") or "unknown"),
                            next_version,
                            runner_backend,
                            workflow_run_id,
                            stored["updated_at"],
                            serialized,
                            resolved_tenant,
                            request_id,
                            int(expected_version),
                        ),
                    )
                    if getattr(cur, "rowcount", 1) == 0:
                        conn.rollback()
                        raise SnapshotConflictError(
                            f"postgres snapshot conflict for request_id={request_id} expected_version={expected_version}"
                        )
            conn.commit()
        stored["state_version"] = next_version
        return stored

    def load(self, request_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                if tenant_id is not None:
                    cur.execute(self._SELECT_BY_ID_TENANT_SQL, (request_id, tenant_id))
                else:
                    cur.execute(self._SELECT_BY_ID_SQL, (request_id,))
                row = cur.fetchone()
        if not row:
            return None
        raw_json = row[1] if len(row) > 1 else row[0]
        try:
            payload = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload["tenant_id"] = row[0]
        payload["state_version"] = row[2]
        resolved_error = RequestStateStore._resolved_error(payload)
        if resolved_error:
            payload.setdefault("error", resolved_error)
        return payload

    def list_recent(
        self,
        *,
        limit: int | None = None,
        include_content: bool = False,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        max_items = max(1, int(limit or self._recent_limit))
        with self._connect() as conn:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                if tenant_id is not None:
                    cur.execute(self._LIST_TENANT_SQL, (tenant_id, max_items))
                else:
                    cur.execute(self._LIST_ALL_SQL, (max_items,))
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows or ():
            raw_json = row[0]
            try:
                payload = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                results.append(RequestStateStore._summarize(payload, include_content=include_content))
        return results


def build_request_state_store(config: Any) -> Any | None:
    """Return PostgresRequestStateStore when DATABASE_URL + psycopg are available, else RequestStateStore."""

    if not getattr(config, "request_state_enabled", True):
        return None

    database_url = str(getattr(config, "database_url", None) or "").strip()
    recent_limit = int(getattr(config, "request_state_recent_limit", 50) or 50)
    pool_min = int(getattr(config, "postgres_pool_min_size", 2) or 2)
    pool_max = int(getattr(config, "postgres_pool_max_size", 10) or 10)
    if database_url and psycopg is not None:
        try:
            return PostgresRequestStateStore(
                dsn=database_url,
                recent_limit=recent_limit,
                pool_min_size=pool_min,
                pool_max_size=pool_max,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize PostgresRequestStateStore; falling back to JSON files (%s)",
                type(exc).__name__,
            )

    return RequestStateStore(
        str(getattr(config, "request_state_dir", "./.state/requests")),
        recent_limit=recent_limit,
    )
