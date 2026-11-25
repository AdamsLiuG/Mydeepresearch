"""Pluggable persistent request state backends (JSON file store and optional PostgreSQL)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol

from services.memory_scope import (
    MemoryScope,
    open_partition_directory,
    partition_lock,
    partition_root,
    read_partition_text,
    valid_internal_id,
    valid_scope,
    write_partition_text,
)
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

    def load_scoped(self, scope: MemoryScope, request_id: str) -> dict[str, Any] | None:
        if not valid_scope(scope) or not valid_internal_id(request_id):
            return None
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(self._SELECT_BY_ID_TENANT_SQL, (request_id, scope.tenant_id))
                row = cur.fetchone()
            if not row or row[0] != scope.tenant_id:
                return None
            return _strict_payload(row[1], scope, request_id, row_version=row[2])
        except Exception:
            return None

    def iter_terminal(
        self, scope: MemoryScope, *, after: str = "", limit: int = 64,
    ) -> list[dict[str, Any]]:
        if not valid_scope(scope) or (after and not valid_internal_id(after)):
            return []
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT request_id, tenant_id, payload_json, state_version FROM deepresearch_request_states "
                    'WHERE tenant_id = %s AND request_id COLLATE "C" > %s AND status IN (%s, %s, %s, %s, %s) '
                    'ORDER BY request_id COLLATE "C" LIMIT %s',
                    (scope.tenant_id, after, *_TERMINAL_STATUSES, _page_limit(limit)),
                )
                rows = cur.fetchall()
            return [payload for request_id, tenant_id, raw, version in rows
                    if tenant_id == scope.tenant_id and valid_internal_id(request_id)
                    and (payload := _strict_payload(raw, scope, request_id, row_version=version))
                    and payload.get("status") in _TERMINAL_STATUSES]
        except Exception:
            return []

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


_TERMINAL_STATUSES = ("success", "partial_success", "failed", "canceled", "cancelled")


def _page_limit(limit: int) -> int:
    return max(0, min(64, limit)) if type(limit) is int else 0


def _strict_payload(
    raw: str, scope: MemoryScope, request_id: str, *, row_version: int | None = None,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(payload, dict) or payload.get("request_id") != request_id
        or payload.get("tenant_id") != scope.tenant_id
    ):
        return None
    version = payload.get("state_version") if row_version is None else row_version
    if type(version) is not int or version < 1:
        return None
    if row_version is not None and payload.get("state_version", row_version) != row_version:
        return None
    payload["state_version"] = version
    return payload


class StrictSnapshotReader(Protocol):
    def load(self, scope: MemoryScope, request_id: str) -> dict[str, Any] | None: ...

    def iter_terminal(
        self, scope: MemoryScope, *, after: str = "", limit: int = 64,
    ) -> list[dict[str, Any]]: ...


class JsonStrictSnapshotReader:
    """Per-tenant owner pointers for the existing authoritative JSON snapshots."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @staticmethod
    def _owner_pointer(scope: MemoryScope, request_id: str) -> str:
        return json.dumps({"tenant_id": scope.tenant_id, "request_id": request_id, "schema_version": 1})

    def register_owner(self, scope: MemoryScope) -> None:
        """Register a trusted owner before first save or from a verified offline mapping."""
        if not valid_scope(scope):
            raise PermissionError("invalid snapshot scope")
        # The legacy raw files are keyed globally, so a request ID can have one owner.
        registry = self._root / "v4" / "owners"
        pointer = scope.request_id + ".json"
        expected = self._owner_pointer(scope, scope.request_id)
        with partition_lock(registry):
            try:
                owner = read_partition_text(registry, pointer, max_bytes=16384)
            except FileNotFoundError:
                try:
                    existing = read_partition_text(self._root, pointer)
                except FileNotFoundError:
                    existing = None
                if existing is not None and _strict_payload(existing, scope, scope.request_id) is None:
                    raise PermissionError("snapshot owner unavailable")
                write_partition_text(registry, pointer, text=expected)
            else:
                if owner != expected:
                    raise PermissionError("snapshot owner mismatch")
            tenant_dir = partition_root(self._root, scope, create=True)
            with partition_lock(tenant_dir):
                write_partition_text(tenant_dir, "owners", pointer, text=expected)

    def load(self, scope: MemoryScope, request_id: str) -> dict[str, Any] | None:
        if not valid_scope(scope) or not valid_internal_id(request_id):
            return None
        try:
            tenant_dir = partition_root(self._root, scope)
            pointer = request_id + ".json"
            expected = self._owner_pointer(scope, request_id)
            if read_partition_text(tenant_dir, "owners", pointer, max_bytes=16384) != expected:
                return None
            if read_partition_text(self._root, "v4", "owners", pointer, max_bytes=16384) != expected:
                return None
            return _strict_payload(read_partition_text(self._root, pointer), scope, request_id)
        except (OSError, ValueError):
            return None

    def iter_terminal(
        self, scope: MemoryScope, *, after: str = "", limit: int = 64,
    ) -> list[dict[str, Any]]:
        if not valid_scope(scope) or (after and not valid_internal_id(after)):
            return []
        try:
            tenant_dir = partition_root(self._root, scope)
            with open_partition_directory(tenant_dir, "owners") as directory:
                ids = sorted(
                    name[:-5] for name in os.listdir(directory)
                    if name.endswith(".json") and name[:-5] > after and valid_internal_id(name[:-5])
                )
            rows = []
            for request_id in ids:
                if len(rows) >= _page_limit(limit):
                    break
                payload = self.load(scope, request_id)
                if payload is not None and payload.get("status") in _TERMINAL_STATUSES:
                    rows.append(payload)
            return rows
        except (OSError, ValueError):
            return []


class PostgresStrictSnapshotReader:
    def __init__(self, store: PostgresRequestStateStore) -> None:
        self._store = store

    def load(self, scope: MemoryScope, request_id: str) -> dict[str, Any] | None:
        return self._store.load_scoped(scope, request_id)

    def iter_terminal(
        self, scope: MemoryScope, *, after: str = "", limit: int = 64,
    ) -> list[dict[str, Any]]:
        return self._store.iter_terminal(scope, after=after, limit=limit)


def build_strict_snapshot_reader(config: Any, *, postgres_store: PostgresRequestStateStore | None = None):
    if not config.request_state_enabled:
        return None
    if config.database_url:
        if not isinstance(postgres_store, PostgresRequestStateStore):
            return None
        return PostgresStrictSnapshotReader(postgres_store)
    return JsonStrictSnapshotReader(config.request_state_dir)


_GRAPH_SCHEMA = "deepresearch_graph"
_GRAPH_POLICY = "graph_tenant_scope"
_GRAPH_TENANT_PREDICATE = "tenant_id = NULLIF(current_setting('deepresearch.tenant_id', true), '')"


def migrate_postgres_graph(connection: Any, *, runtime_role: str) -> None:
    """Run once per deployment as the graph schema owner or migration role."""
    from services.knowledge_graph import GRAPH_SCHEMA_SQL, GRAPH_TABLES

    if not isinstance(runtime_role, str) or not runtime_role or len(runtime_role.encode()) > 63 or not runtime_role.isprintable():
        raise ValueError("invalid Postgres graph runtime role")
    role = '"' + runtime_role.replace('"', '""') + '"'
    with connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('deepresearch_graph_schema'))")
        if connection.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s AND NOT rolsuper AND NOT rolbypassrls "
            "AND rolname <> current_user", (runtime_role,),
        ).fetchone() is None:
            raise PermissionError("invalid graph runtime role")
        connection.execute(f"CREATE SCHEMA IF NOT EXISTS {_GRAPH_SCHEMA}")
        connection.execute(f"SET LOCAL search_path TO {_GRAPH_SCHEMA}, pg_catalog")
        connection.execute(f"REVOKE ALL ON SCHEMA {_GRAPH_SCHEMA} FROM PUBLIC")
        connection.execute(f"GRANT USAGE ON SCHEMA {_GRAPH_SCHEMA} TO {role}")
        for statement in GRAPH_SCHEMA_SQL:
            connection.execute(statement)
        for table in GRAPH_TABLES:
            connection.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            connection.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            connection.execute(f"DROP POLICY IF EXISTS {_GRAPH_POLICY} ON {table}")
            connection.execute(
                f"CREATE POLICY {_GRAPH_POLICY} ON {table} FOR ALL "
                f"USING ({_GRAPH_TENANT_PREDICATE}) WITH CHECK ({_GRAPH_TENANT_PREDICATE})",
            )
            connection.execute(f"REVOKE ALL ON TABLE {table} FROM PUBLIC")
            connection.execute(f"REVOKE ALL ON TABLE {table} FROM {role}")
            connection.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {table} TO {role}")


class PostgresGraphStore:
    backend = "postgres"

    def __init__(self, *, dsn: str = "", connection_factory: Callable[[], Any] | None = None) -> None:
        self._dsn = dsn
        self._connection_factory = connection_factory
        if connection_factory is None and (not dsn or psycopg is None):
            raise RuntimeError("Postgres graph connection unavailable")

    @contextmanager
    def _connect(self, deadline: float) -> Iterator[Any]:
        from services.knowledge_graph import _remaining

        seconds = _remaining(deadline)
        connection = (
            self._connection_factory() if self._connection_factory is not None
            else psycopg.connect(self._dsn, connect_timeout=max(2, min(3, int(seconds) + 1)))
        )
        with connection as conn:
            _remaining(deadline)
            yield conn

    @staticmethod
    def _validate_role(connection: Any) -> None:
        from services.knowledge_graph import GRAPH_TABLES

        safe = connection.execute(
            "SELECT NOT r.rolsuper AND NOT r.rolbypassrls AND NOT "
            "has_schema_privilege(current_user, %s, 'CREATE') "
            "FROM pg_roles AS r WHERE r.rolname = current_user", (_GRAPH_SCHEMA,),
        ).fetchone()
        if safe is None or safe[0] is not True:
            raise PermissionError("invalid graph runtime role")
        rows = connection.execute(
            "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, "
            "pg_has_role(current_user, c.relowner, 'MEMBER'), p.policyname, p.cmd, p.roles, "
            "p.qual, p.with_check, has_table_privilege(current_user, c.oid, 'TRUNCATE'), "
            "has_table_privilege(current_user, c.oid, 'SELECT') "
            "FROM pg_class AS c JOIN pg_namespace AS n ON n.oid = c.relnamespace "
            "LEFT JOIN pg_policies AS p ON p.schemaname = n.nspname AND p.tablename = c.relname "
            "WHERE n.nspname = %s AND c.relname = ANY(%s) AND c.relkind = 'r'",
            (_GRAPH_SCHEMA, list(GRAPH_TABLES)),
        ).fetchall()
        expected = "(tenant_id = NULLIF(current_setting('deepresearch.tenant_id'::text, true), ''::text))"
        if len(rows) != len(GRAPH_TABLES) or {row[0] for row in rows} != set(GRAPH_TABLES):
            raise PermissionError("incomplete graph RLS schema")
        for _, enabled, forced, owner, policy, command, roles, using, check, truncate, select in rows:
            if (
                not enabled or not forced or owner or policy != _GRAPH_POLICY or command != "ALL"
                or list(roles) != ["public"] or using != expected or check != expected or truncate or not select
            ):
                raise PermissionError("invalid graph RLS policy")

    @contextmanager
    def _transaction(self, tenant_id: str, *, write: bool, deadline: float) -> Iterator[Any]:
        from services.knowledge_graph import _remaining

        with self._connect(deadline) as conn, conn.transaction():
            if not write:
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            conn.execute("SELECT set_config('statement_timeout', %s, true), set_config('lock_timeout', %s, true)",
                         (str(max(1, int(_remaining(deadline) * 1000))),) * 2)
            conn.execute(
                "SELECT set_config('deepresearch.tenant_id', %s, true), set_config('search_path', %s, true), "
                "set_config('row_security', 'on', true)", (tenant_id, f"pg_catalog, {_GRAPH_SCHEMA}, pg_temp"),
            )
            self._validate_role(conn)
            if write:
                conn.execute("SELECT pg_advisory_xact_lock(hashtext('deepresearch_graph'), hashtext(%s))", (tenant_id,))
            _remaining(deadline)
            yield conn
            _remaining(deadline)

    def probe(self, *, deadline: float) -> None:
        with self._transaction("", write=False, deadline=deadline):
            pass

    @contextmanager
    def transaction(
        self, scope: MemoryScope, *, write: bool, deadline: float,
    ) -> Iterator[Any]:
        from services.knowledge_graph import SqlGraphTransaction

        if not valid_scope(scope):
            raise PermissionError("invalid graph scope")
        with self._transaction(scope.tenant_id, write=write, deadline=deadline) as conn:
            yield SqlGraphTransaction(conn, scope.tenant_id, placeholder="%s")
