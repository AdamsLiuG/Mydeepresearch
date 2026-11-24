"""Persistent request snapshot helpers used for history and resume."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

try:
    import fcntl
except ImportError:  # Windows uses the shared in-process lock.
    fcntl = None

_SAVE_LOCK = Lock()


@contextmanager
def _lock_snapshot_directory(directory: Path) -> Iterator[None]:
    if fcntl is None:
        with _SAVE_LOCK:
            yield
        return
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class SnapshotConflictError(RuntimeError):
    """Raised on optimistic CAS mismatch when persisting request snapshots."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestStateStore:
    """Persist request snapshots as JSON files."""

    def __init__(self, directory: str, *, recent_limit: int = 50) -> None:
        self._directory = Path(directory)
        self._recent_limit = max(1, int(recent_limit or 1))

    @property
    def directory(self) -> Path:
        return self._directory

    def path_for(self, request_id: str) -> Path:
        safe_request_id = "".join(char for char in (request_id or "").strip() if char.isalnum() or char in {"-", "_"})
        if not safe_request_id:
            raise ValueError("request_id is required")
        return self._directory / f"{safe_request_id}.json"

    def save(
        self,
        request_id: str,
        payload: dict[str, Any],
        *,
        tenant_id: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(request_id)
        with _lock_snapshot_directory(self._directory):
            current_version = 0
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict):
                        current_version = max(0, int(existing.get("state_version") or 0))
                except (OSError, json.JSONDecodeError, ValueError):
                    current_version = 0

            if expected_version is not None and current_version != int(expected_version):
                raise SnapshotConflictError(
                    f"snapshot version conflict for request_id={request_id}: "
                    f"expected={expected_version} current={current_version}"
                )

            next_version = current_version + 1
            stored_payload = dict(payload)
            stored_payload.setdefault("request_id", request_id)
            resolved_tenant = str(tenant_id or stored_payload.get("tenant_id") or "default").strip() or "default"
            stored_payload["tenant_id"] = resolved_tenant
            stored_payload["state_version"] = next_version
            stored_payload["updated_at"] = _utc_now()
            # Use a unique temp file per save so concurrent task snapshots for the
            # same request_id cannot clobber each other's atomic replace step.
            temp_path = path.with_name(f"{path.stem}.{uuid4().hex}.json.tmp")
            try:
                temp_path.write_text(
                    json.dumps(stored_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temp_path.replace(path)
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
            return stored_payload

    def load(self, request_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        path = self.path_for(request_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        if tenant_id is not None:
            record_tenant = str(payload.get("tenant_id") or "default").strip() or "default"
            if record_tenant != str(tenant_id).strip():
                return None

        resolved_error = self._resolved_error(payload)
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
        if not self._directory.exists():
            return []

        max_items = max(1, int(limit or self._recent_limit))
        paths = sorted(
            self._directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        snapshots: list[dict[str, Any]] = []
        for path in paths:
            if len(snapshots) >= max_items:
                break
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if tenant_id is not None:
                record_tenant = str(payload.get("tenant_id") or "default").strip() or "default"
                if record_tenant != str(tenant_id).strip():
                    continue
            snapshots.append(self._summarize(payload, include_content=include_content))
        return snapshots

    @staticmethod
    def _summarize(
        payload: dict[str, Any],
        *,
        include_content: bool = False,
    ) -> dict[str, Any]:
        report_markdown = str(payload.get("report_markdown") or "").strip()
        status = str(payload.get("status") or "unknown").strip() or "unknown"
        phase = str(payload.get("phase") or "unknown").strip() or "unknown"
        error = RequestStateStore._resolved_error(payload)
        todo_items = payload.get("todo_items") or []
        summary: dict[str, Any] = {
            "request_id": str(payload.get("request_id") or "").strip(),
            "topic": str(payload.get("topic") or "").strip(),
            "status": status,
            "phase": phase,
            "error": error,
            "updated_at": payload.get("updated_at"),
            "search_api": payload.get("search_api"),
            "elapsed_ms": payload.get("elapsed_ms"),
            "cache_diagnostics": payload.get("cache_diagnostics") or {},
            "review_summary": payload.get("review_summary") or {},
            "task_count": len(todo_items),
            "has_report": bool(report_markdown),
            "can_resume": status not in {"success"} and phase != "completed",
            "can_view_content": bool(report_markdown) or bool(todo_items) or bool(error),
        }
        if include_content:
            summary["report_markdown"] = report_markdown
            summary["todo_items"] = todo_items
        return summary

    @staticmethod
    def _resolved_error(payload: dict[str, Any]) -> str | None:
        error = str(payload.get("error") or "").strip()
        if error:
            return error

        request_metrics = payload.get("request_metrics")
        if not isinstance(request_metrics, dict):
            return None

        nested_error = str(request_metrics.get("error") or "").strip()
        return nested_error or None
