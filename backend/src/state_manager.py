"""Request snapshot and resume helpers for the deep research workflow."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from metrics import RequestTrace
from models import SummaryState, TodoItem
from services.dag_scheduler import normalize_depends_on
from services.evidence import EvidenceStore, format_evidence_sources
from services.mcp_bridge import tool_approval_expired
from services.request_state import SnapshotConflictError


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
        observer.restore_review_result(state.review_summary, state.claim_verification_summary)

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
            **({"charts_spec": list(state.report_charts)} if state.report_charts else {}),
            **({
                "workbench": state.report_workbench,
                "citation_drawer_map": state.report_workbench.get("citation_index", {}),
                "verifier_badges": state.report_workbench.get("verifier_badges", []),
            } if getattr(state, "report_workbench", None) else {}),
            "report_note_id": state.report_note_id,
            "report_note_path": state.report_note_path,
            "review_summary": dict(state.review_summary or {}),
            "reflection_completed": bool(state.reflection_completed),
            "review_completed": bool(state.review_completed),
            "report_repair_completed": bool(state.report_repair_completed),
            "report_repair_cycles": int(state.report_repair_cycles or 0),
            "topic_cache_warmup_completed": bool(state.topic_cache_warmup_completed),
            "skill_pack": dict(state.skill_pack_metadata or {}),
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
            "state_version": int(getattr(state, "state_version", 0) or 0),
            "runner_backend": str(getattr(state, "runner_backend", "inline") or "inline"),
            "workflow_run_id": getattr(state, "workflow_run_id", None),
            "tenant_id": str(getattr(self, "tenant_id", "default") or "default").strip()
            or "default",
            "cache_diagnostics": cache_diagnostics,
            "request_metrics": observer_snapshot,
        }
        tenant_id = str(getattr(self, "tenant_id", "default") or "default").strip() or "default"
        with getattr(self, "_request_snapshot_lock", nullcontext()):
            serialized_tasks = [self._serialize_task(task) for task in state.todo_items]
            if isinstance(self._evidence_store, EvidenceStore) and (
                getattr(self.config, "evidence_pointer_disclosure_enabled", True)
                or getattr(self.config, "resolved_sandbox_effective_enabled", lambda: False)()
            ):
                for task in serialized_tasks:
                    evidence = self._evidence_store.list_task_evidence(task["id"], include_full_content=True)
                    if evidence:
                        task["evidence_items"] = evidence
            payload.update(
                todo_items=serialized_tasks,
                fact_ledger=deepcopy(state.fact_ledger),
                evidence_index_cards=deepcopy(state.evidence_index_cards),
                frozen_prefix_version=state.frozen_prefix_version,
                context_checkpoint_blocks=deepcopy(state.context_checkpoint_blocks),
            )
            payload["approved_tool_calls"] = dict(getattr(state, "approved_tool_calls", None) or {})
            payload["state_version"] = int(getattr(state, "state_version", 0) or 0)
            checkpoint = getattr(self, "_workflow_checkpoint", None)
            if checkpoint is not None:
                payload["runner_backend"] = checkpoint.backend
                payload["workflow_run_id"] = checkpoint.run_id
                state.runner_backend = checkpoint.backend
                state.workflow_run_id = checkpoint.run_id
                saved = checkpoint.persist(payload)
            elif self._store_supports_tenant_id("save"):
                kwargs: dict[str, Any] = {"tenant_id": tenant_id}
                if payload["pending_tool_approval"] or payload["approved_tool_calls"]:
                    kwargs["expected_version"] = payload["state_version"]
                saved = self._request_state_store.save(self.request_id, payload, **kwargs)
            else:
                saved = self._request_state_store.save(self.request_id, payload)
            if isinstance(saved, dict) and "state_version" in saved:
                state.state_version = int(saved.get("state_version") or state.state_version or 0)
        self._refresh_note_memory_for_state(state, observer=self._request_trace)

    def _consume_mcp_approval(self, key: str, entry: dict[str, Any]) -> bool:
        state = self._active_state
        if state is None or self._request_state_store is None:
            return False
        with getattr(self, "_request_snapshot_lock", nullcontext()):
            checkpoint = getattr(self, "_workflow_checkpoint", None)
            if checkpoint is not None:
                saved = checkpoint.consume_tool_approval(key, entry)
                if saved is None:
                    return False
                state.state_version = saved["state_version"]
                state.approved_tool_calls[key] = saved["approved_tool_calls"][key]
                return True
            snapshot = self._load_resume_snapshot(self.request_id)
            registry = dict(snapshot.get("approved_tool_calls") or {})
            current = registry.get(key)
            if (
                snapshot.get("state_version") != state.state_version
                or current != entry or not isinstance(current, dict) or current.get("status") != "approved_once"
                or tool_approval_expired(current)
            ):
                return False
            consumed = {**current, "status": "consumed"}
            registry[key] = consumed
            snapshot["approved_tool_calls"] = registry
            try:
                saved = self._request_state_store.save(
                    self.request_id, snapshot, tenant_id=self.tenant_id, expected_version=state.state_version,
                )
            except SnapshotConflictError:
                return False
            state.state_version = saved["state_version"]
            state.approved_tool_calls[key] = consumed
        return True

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
        if (
            not payload
            or (payload.get("tenant_id") or "default") != tenant_id
            or (payload.get("request_id") or request_id) != request_id
        ):
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
            atomic_claims=[
                deepcopy(claim)
                for claim in payload["atomic_claims"]
                if isinstance(claim, dict)
            ] if isinstance(payload.get("atomic_claims"), list) else [],
        )

    def _state_from_snapshot(self, payload: dict[str, Any]) -> tuple[SummaryState, str]:
        """Restore SummaryState and phase from a persisted snapshot."""

        if payload.get("phase") == "awaiting_tool_approval":
            pending = payload.get("pending_tool_approval")
            if (
                not isinstance(pending, dict)
                or pending.get("status") != "approved_once"
                or tool_approval_expired(pending)
            ):
                raise ValueError("A valid tool approval is required before resuming")

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
                deepcopy(payload["fact_ledger"])
                if isinstance(payload.get("fact_ledger"), dict)
                else {}
            ),
            evidence_index_cards=deepcopy(payload.get("evidence_index_cards") or []),
            frozen_prefix_version=int(payload.get("frozen_prefix_version") or 0),
            context_checkpoint_blocks=deepcopy(payload.get("context_checkpoint_blocks") or []),
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
                else list(payload["charts_spec"]) if isinstance(payload.get("charts_spec"), list) else []
            ),
            report_workbench=(
                deepcopy(payload["workbench"]) if isinstance(payload.get("workbench"), dict) else {}
            ),
            evidence_conflicts=(
                [deepcopy(conflict) for conflict in payload["evidence_conflicts"] if isinstance(conflict, dict)]
                if isinstance(payload.get("evidence_conflicts"), list)
                else []
            ),
            claim_verification_summary=(
                deepcopy(payload["claim_verification_summary"])
                if isinstance(payload.get("claim_verification_summary"), dict)
                else {}
            ),
            state_version=max(0, int(payload.get("state_version") or 0)),
            runner_backend=str(payload.get("runner_backend") or "inline").strip() or "inline",
            workflow_run_id=str(payload.get("workflow_run_id") or "").strip() or None,
        )
        self._evidence_store.hydrate_from_tasks(state.todo_items)
        sandbox = getattr(self, "_sandbox_service", None)
        if sandbox is not None:
            sandbox.restore_task_counts(state.todo_items)
        for task in state.todo_items:
            if not task.sources_summary and task.id > 0:
                evidence_items = self._evidence_store.list_task_evidence(task.id)
                if evidence_items:
                    task.evidence_items = evidence_items
                    task.sources_summary = format_evidence_sources(evidence_items)
        phase = str(payload.get("phase") or "planning").strip() or "planning"
        return state, "task_execution" if phase == "awaiting_tool_approval" else phase
