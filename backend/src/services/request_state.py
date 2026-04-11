"""Persistent request snapshot helpers used for history and resume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


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

    def save(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(request_id)
        stored_payload = dict(payload)
        stored_payload.setdefault("request_id", request_id)
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

    def load(self, request_id: str) -> dict[str, Any] | None:
        path = self.path_for(request_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        resolved_error = self._resolved_error(payload)
        if resolved_error:
            payload.setdefault("error", resolved_error)
        return payload

    def list_recent(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self._directory.exists():
            return []

        max_items = max(1, int(limit or self._recent_limit))
        paths = sorted(
            self._directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        snapshots: list[dict[str, Any]] = []
        for path in paths[:max_items]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            snapshots.append(self._summarize(payload))
        return snapshots

    @staticmethod
    def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
        report_markdown = str(payload.get("report_markdown") or "").strip()
        status = str(payload.get("status") or "unknown").strip() or "unknown"
        phase = str(payload.get("phase") or "unknown").strip() or "unknown"
        error = RequestStateStore._resolved_error(payload)
        todo_items = payload.get("todo_items") or []
        return {
            "request_id": str(payload.get("request_id") or "").strip(),
            "topic": str(payload.get("topic") or "").strip(),
            "status": status,
            "phase": phase,
            "error": error,
            "updated_at": payload.get("updated_at"),
            "search_api": payload.get("search_api"),
            "elapsed_ms": payload.get("elapsed_ms"),
            "cache_diagnostics": payload.get("cache_diagnostics") or {},
            "report_markdown": report_markdown,
            "todo_items": todo_items,
            "review_summary": payload.get("review_summary") or {},
            "can_resume": status not in {"success"} and phase != "completed",
            "can_view_content": bool(report_markdown) or bool(todo_items) or bool(error),
        }

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
