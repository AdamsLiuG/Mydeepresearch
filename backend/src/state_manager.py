"""Request snapshot and resume helpers for the deep research workflow."""

from __future__ import annotations

from typing import Any

from metrics import RequestTrace
from models import SummaryState, TodoItem
from services.dag_scheduler import normalize_depends_on
from services.evidence import format_evidence_sources


class StateManagerMixin:
    """Persist, restore, and serialize request workflow state."""

    def _restore_task_counters(self, observer: RequestTrace, state: SummaryState) -> None:
        """Replay persisted task counts into a fresh request trace."""

        observer.set_task_totals(total_tasks=len(state.todo_items))
        observer.update_task_status_counts(
            completed=sum(1 for task in state.todo_items if task.status == "completed"),
            skipped=sum(1 for task in state.todo_items if task.status == "skipped"),
            failed=sum(1 for task in state.todo_items if task.status == "failed"),
        )

    def _persist_request_state(
        self,
        state: SummaryState,
        *,
        phase: str,
        status: str = "in_progress",
        report_markdown: str | None = None,
    ) -> None:
        """Persist the latest request snapshot for history and resume."""

        if self._request_state_store is None:
            return

        observer_snapshot = self._request_trace.snapshot() if self._request_trace else {}
        cache_diagnostics = {
            "cache_hits": observer_snapshot.get("cache_hits", 0),
            "cache_exact_hits": observer_snapshot.get("cache_exact_hits", 0),
            "cache_semantic_hits": observer_snapshot.get("cache_semantic_hits", 0),
            "cache_misses": observer_snapshot.get("cache_misses", 0),
            "last_search_cache_details": dict(observer_snapshot.get("last_search_cache_details") or {}),
        }
        payload = {
            "snapshot_version": 2,
            "request_id": self.request_id,
            "topic": state.research_topic,
            "phase": phase,
            "status": status,
            "error": observer_snapshot.get("error"),
            "search_api": observer_snapshot.get("search_api"),
            "elapsed_ms": observer_snapshot.get("elapsed_ms"),
            "report_markdown": report_markdown or state.structured_report or state.running_summary or "",
            "report_charts": list(getattr(state, "report_charts", None) or []),
            "report_note_id": state.report_note_id,
            "report_note_path": state.report_note_path,
            "todo_items": [self._serialize_task(task) for task in state.todo_items],
            "review_summary": dict(state.review_summary or {}),
            "reflection_completed": bool(state.reflection_completed),
            "review_completed": bool(state.review_completed),
            "report_repair_completed": bool(state.report_repair_completed),
            "report_repair_cycles": int(state.report_repair_cycles or 0),
            "topic_cache_warmup_completed": bool(state.topic_cache_warmup_completed),
            "skill_pack": dict(state.skill_pack_metadata or {}),
            "fact_ledger": dict(getattr(state, "fact_ledger", None) or {}),
            "evidence_index_cards": list(getattr(state, "evidence_index_cards", None) or []),
            "frozen_prefix_version": int(getattr(state, "frozen_prefix_version", 0) or 0),
            "pending_tool_approval": (
                dict(state.pending_tool_approval)
                if isinstance(getattr(state, "pending_tool_approval", None), dict)
                else None
            ),
            "approved_tool_calls": dict(getattr(state, "approved_tool_calls", None) or {}),
            "evidence_conflicts": list(getattr(state, "evidence_conflicts", None) or []),
            "claim_verification_summary": dict(
                getattr(state, "claim_verification_summary", None) or {}
            ),
            "tenant_id": str(getattr(self, "tenant_id", "default") or "default").strip()
            or "default",
            "cache_diagnostics": cache_diagnostics,
            "request_metrics": observer_snapshot,
        }
        tenant_id = str(getattr(self, "tenant_id", "default") or "default").strip() or "default"
        if self._store_supports_tenant_id("save"):
            self._request_state_store.save(self.request_id, payload, tenant_id=tenant_id)
        else:
            self._request_state_store.save(self.request_id, payload)
        self._refresh_note_memory_for_state(state, observer=self._request_trace)

    def _store_supports_tenant_id(self, method_name: str) -> bool:
        import inspect

        method = getattr(self._request_state_store, method_name, None)
        if method is None:
            return False
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return False
        return "tenant_id" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )

    def _load_resume_snapshot(self, request_id: str) -> dict[str, Any]:
        if self._request_state_store is None:
            raise ValueError("request_state store is disabled")
        tenant_id = str(getattr(self, "tenant_id", "default") or "default").strip() or "default"
        if self._store_supports_tenant_id("load"):
            payload = self._request_state_store.load(request_id, tenant_id=tenant_id)
        else:
            payload = self._request_state_store.load(request_id)
        if not payload:
            raise ValueError(f"resume snapshot not found: {request_id}")
        return payload

    @staticmethod
    def _task_from_payload(payload: dict[str, Any]) -> TodoItem:
        """Deserialize a persisted task payload into TodoItem."""

        return TodoItem(
            id=int(payload.get("id") or 0),
            title=str(payload.get("title") or "").strip() or "任务",
            intent=str(payload.get("intent") or "").strip() or "恢复执行任务",
            query=str(payload.get("query") or "").strip(),
            status=str(payload.get("status") or "pending").strip() or "pending",
            summary=str(payload.get("summary") or "").strip() or None,
            summary_payload=(
                dict(payload.get("summary_payload") or {})
                if isinstance(payload.get("summary_payload"), dict)
                else None
            ),
            sources_summary=str(payload.get("sources_summary") or "").strip() or None,
            notices=[
                str(item).strip()
                for item in payload.get("notices") or []
                if str(item).strip()
            ],
            evidence_items=list(payload.get("evidence_items") or []),
            claims=list(payload.get("claims") or []),
            review_issues=list(payload.get("review_issues") or []),
            review_status=str(payload.get("review_status") or "pending").strip() or "pending",
            note_id=str(payload.get("note_id") or "").strip() or None,
            note_path=str(payload.get("note_path") or "").strip() or None,
            stream_token=str(payload.get("stream_token") or "").strip() or None,
            origin=str(payload.get("origin") or "planned").strip() or "planned",
            round=max(1, int(payload.get("round") or 1)),
            react_rounds=max(0, int(payload.get("react_rounds") or 0)),
            react_fetch_count=max(0, int(payload.get("react_fetch_count") or 0)),
            react_additional_search_count=max(
                0,
                int(payload.get("react_additional_search_count") or 0),
            ),
            react_gap_signals=[
                str(item).strip()
                for item in payload.get("react_gap_signals") or []
                if str(item).strip()
            ],
            react_last_action=str(payload.get("react_last_action") or "").strip() or None,
            react_stop_reason=str(payload.get("react_stop_reason") or "").strip() or None,
            react_observation=(
                dict(payload.get("react_observation") or {})
                if isinstance(payload.get("react_observation"), dict)
                else {}
            ),
            depends_on=normalize_depends_on(payload.get("depends_on")),
            upstream_context=str(payload.get("upstream_context") or "").strip(),
            fact_claims=list(payload.get("fact_claims") or []),
            atomic_claims=list(payload.get("atomic_claims") or []),
        )

    def _state_from_snapshot(self, payload: dict[str, Any]) -> tuple[SummaryState, str]:
        """Restore SummaryState and phase from a persisted snapshot."""

        todo_items = []
        for item in payload.get("todo_items") or []:
            if isinstance(item, dict):
                todo_items.append(self._task_from_payload(item))

        report_markdown = str(payload.get("report_markdown") or "").strip() or None
        state = SummaryState(
            research_topic=str(payload.get("topic") or "").strip() or None,
            running_summary=report_markdown,
            todo_items=todo_items,
            structured_report=report_markdown,
            report_note_id=str(payload.get("report_note_id") or "").strip() or None,
            report_note_path=str(payload.get("report_note_path") or "").strip() or None,
            review_summary=dict(payload.get("review_summary") or {}),
            reflection_completed=bool(payload.get("reflection_completed")),
            review_completed=bool(payload.get("review_completed")),
            report_repair_completed=bool(payload.get("report_repair_completed")),
            report_repair_cycles=max(0, int(payload.get("report_repair_cycles") or 0)),
            topic_cache_warmup_completed=bool(payload.get("topic_cache_warmup_completed")),
            skill_pack_metadata=(
                dict(payload.get("skill_pack") or {})
                if isinstance(payload.get("skill_pack"), dict)
                else {}
            ),
            fact_ledger=(
                dict(payload.get("fact_ledger") or {})
                if isinstance(payload.get("fact_ledger"), dict)
                else {}
            ),
            evidence_index_cards=list(payload.get("evidence_index_cards") or []),
            frozen_prefix_version=int(payload.get("frozen_prefix_version") or 0),
            pending_tool_approval=(
                dict(payload.get("pending_tool_approval"))
                if isinstance(payload.get("pending_tool_approval"), dict)
                else None
            ),
            approved_tool_calls=(
                dict(payload.get("approved_tool_calls") or {})
                if isinstance(payload.get("approved_tool_calls"), dict)
                else {}
            ),
            report_charts=(
                list(payload.get("report_charts") or [])
                if isinstance(payload.get("report_charts"), list)
                else []
            ),
            evidence_conflicts=(
                list(payload.get("evidence_conflicts") or [])
                if isinstance(payload.get("evidence_conflicts"), list)
                else []
            ),
            claim_verification_summary=(
                dict(payload.get("claim_verification_summary") or {})
                if isinstance(payload.get("claim_verification_summary"), dict)
                else {}
            ),
        )
        self._evidence_store.hydrate_from_tasks(state.todo_items)
        for task in state.todo_items:
            if not task.sources_summary and task.id > 0:
                evidence_items = self._evidence_store.list_task_evidence(task.id)
                if evidence_items:
                    task.evidence_items = evidence_items
                    task.sources_summary = format_evidence_sources(evidence_items)
        return state, str(payload.get("phase") or "planning").strip() or "planning"
