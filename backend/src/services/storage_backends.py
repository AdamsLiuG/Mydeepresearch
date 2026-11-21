"""Pluggable persistent request state backends (JSON file store and optional PostgreSQL)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from services.request_state import RequestStateStore

try:  # pragma: no cover - optional dependency
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]

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
        "updated_at TEXT NOT NULL, "
        "payload_json TEXT NOT NULL"
        ")"
    )
    _UPSERT_SQL = (
        "INSERT INTO deepresearch_request_states "
        "(request_id, tenant_id, topic, phase, status, updated_at, payload_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (request_id) DO UPDATE SET "
        "tenant_id = EXCLUDED.tenant_id, "
        "topic = EXCLUDED.topic, "
        "phase = EXCLUDED.phase, "
        "status = EXCLUDED.status, "
        "updated_at = EXCLUDED.updated_at, "
        "payload_json = EXCLUDED.payload_json"
    )
    _SELECT_BY_ID_SQL = (
        "SELECT tenant_id, payload_json FROM deepresearch_request_states WHERE request_id = %s"
    )
    _SELECT_BY_ID_TENANT_SQL = (
        "SELECT tenant_id, payload_json FROM deepresearch_request_states "
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
    ) -> None:
        self._dsn = str(dsn or "").strip()
        self._recent_limit = max(1, int(recent_limit or 1))
        self._connection_factory = connection_factory
        self._schema_ready = False

    def _connect(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory()
        if psycopg is None:
            raise RuntimeError("psycopg package is required for PostgresRequestStateStore")
        return psycopg.connect(self._dsn)

    def _ensure_schema(self, conn: Any) -> None:
        if self._schema_ready:
            return
        with conn.cursor() as cur:
            cur.execute(self._CREATE_TABLE_SQL)
        conn.commit()
        self._schema_ready = True

    def save(
        self,
        request_id: str,
        payload: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        stored = dict(payload)
        stored.setdefault("request_id", request_id)
        resolved_tenant = str(tenant_id or stored.get("tenant_id") or "default").strip() or "default"
        stored["tenant_id"] = resolved_tenant
        stored["updated_at"] = _utc_now()
        serialized = json.dumps(stored, ensure_ascii=False)

        with self._connect() as conn:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    self._UPSERT_SQL,
                    (
                        request_id,
                        resolved_tenant,
                        str(stored.get("topic") or ""),
                        str(stored.get("phase") or "unknown"),
                        str(stored.get("status") or "unknown"),
                        stored["updated_at"],
                        serialized,
                    ),
                )
            conn.commit()
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
        return payload if isinstance(payload, dict) else None

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
    if database_url and psycopg is not None:
        try:
            return PostgresRequestStateStore(dsn=database_url, recent_limit=recent_limit)
        except Exception as exc:
            logger.warning("Failed to initialize PostgresRequestStateStore; falling back to JSON files: %s", exc)

    return RequestStateStore(
        str(getattr(config, "request_state_dir", "./.state/requests")),
        recent_limit=recent_limit,
    )
