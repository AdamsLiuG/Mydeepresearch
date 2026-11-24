"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, Iterator, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI, backend_root
from metrics import metrics_registry
from services.event_bus import build_stream_event_bus
from services.hitl import HitlService
from services.mcp_bridge import (
    MCPBridgeManager,
    compute_arguments_digest,
    tool_approval_expired,
)
from services.request_state import SnapshotConflictError
from services.skill_pack import SkillPackService
from services.storage_backends import build_request_state_store
from services.tenant import (
    TenantContext,
    TenantRateLimiter,
    authenticate_tenant,
    extract_api_key,
)
from services.workflow_runner import build_workflow_runner

try:  # pragma: no cover - optional runtime dependency
    import redis as _redis_mod
except Exception:  # pragma: no cover - optional runtime dependency
    _redis_mod = None  # type: ignore[assignment]

DeepResearchAgent: Any | None = None

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "request_id=%(request_id)s | %(message)s"
)

logger = logging.getLogger("deepresearch.api")


class RequestContextFilter(logging.Filter):
    """Ensure logs always have a request_id attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging(level: str) -> None:
    """Configure the root logger for API and service observability."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        force=True,
    )

    request_filter = RequestContextFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(request_filter)


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )
    output_format: str | None = Field(
        default=None,
        description="Optional output format skill id (e.g., exec-brief, engineering-rfc, deep-whitepaper)",
    )


class ResumeRequest(BaseModel):
    """Optional payload used when resuming a persisted request."""

    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )
    charts_spec: list[dict[str, Any]] | None = Field(
        default=None,
        description="Sanitized charts backed by calculation evidence",
    )
    evidence_conflicts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detected cross-source conflicts and adjudication status",
    )
    claim_verification_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Atomic claim NLI verification aggregate metrics",
    )


class ClarifyRequest(BaseModel):
    """Payload for HITL topic ambiguity check."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Optional search backend override",
    )


class EditableTaskPayload(BaseModel):
    """Editable task entry supplied by a user during HITL plan review."""

    id: int | None = None
    title: str
    intent: str = ""
    query: str = ""
    depends_on: list[int] = Field(default_factory=list)
    origin: str = "planned"
    round: int = 1


class PlanUpdateRequest(BaseModel):
    """Payload for updating a persisted awaiting_approval task plan."""

    tasks: list[EditableTaskPayload] = Field(..., min_length=1)


class DrilldownRequest(BaseModel):
    """Payload for targeted incremental section drill-down on an existing request."""

    section_title: str = Field(..., min_length=1, description="Report section heading to expand")
    question: str = Field(..., min_length=1, description="Focused follow-up question for drill-down")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Optional search backend override",
    )


class ToolApprovalDecisionRequest(BaseModel):
    """Payload for approving or rejecting a paused non-readonly MCP tool call."""

    decision: Literal["approve", "reject"] = Field(..., description="Approval decision")
    reason: str = Field(default="", max_length=2000, description="Optional operator reason")
    arguments_digest: str | None = None


class ResearchToolApprovalRequest(ToolApprovalDecisionRequest):
    request_id: str
    approval_id: str


class AsyncResearchRequest(BaseModel):
    """Payload for submitting an asynchronous durable research run."""

    topic: str = Field(..., min_length=1, description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )
    output_format: str | None = Field(
        default=None,
        description="Optional output format skill id",
    )


class WorkflowSignalRequest(BaseModel):
    """Payload delivered via POST /requests/{request_id}/signals/{signal_name}."""

    payload: dict[str, Any] = Field(default_factory=dict, description="Optional signal metadata")


def _mask_secret(value: str | None, visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest, *, base_config: Configuration) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api
    if getattr(payload, "output_format", None) is not None and str(payload.output_format).strip():
        overrides["default_output_format"] = str(payload.output_format).strip().lower()

    return Configuration.model_validate({**base_config.model_dump(), **overrides})


def _warmup_semantic_cache_model(config: Configuration) -> None:
    """Warm up the approximate-cache embedding model without blocking startup on failure."""

    if not config.resolved_approximate_cache_enabled():
        logger.info(
            "Approximate cache warmup skipped: approximate cache disabled",
            extra={"request_id": "startup"},
        )
        return

    if not config.resolved_approximate_cache_dense_enabled():
        logger.info(
            "Approximate cache warmup skipped: dense signal disabled",
            extra={"request_id": "startup"},
        )
        return

    if not config.semantic_cache_warmup_enabled:
        logger.info(
            "Approximate cache warmup skipped: startup warmup disabled",
            extra={"request_id": "startup"},
        )
        return

    model_name = str(config.semantic_cache_embedding_model or "").strip()
    if not model_name:
        logger.info(
            "Approximate cache warmup skipped: embedding model unset",
            extra={"request_id": "startup"},
        )
        return

    from services.embeddings import embeddings_available, load_sentence_transformer

    if not embeddings_available():
        logger.info(
            "Approximate cache warmup skipped: sentence-transformers unavailable model=%s",
            model_name,
            extra={"request_id": "startup"},
        )
        return

    started_at = time.perf_counter()
    try:
        model = load_sentence_transformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on local runtime state
        logger.warning(
            "Approximate cache warmup failed model=%s error=%s",
            model_name,
            exc,
            extra={"request_id": "startup"},
        )
        return

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if model is None:
        logger.info(
            "Approximate cache warmup skipped: model unavailable model=%s elapsed_ms=%.2f",
            model_name,
            elapsed_ms,
            extra={"request_id": "startup"},
        )
        return

    logger.info(
        "Approximate cache warmup completed model=%s elapsed_ms=%.2f",
        model_name,
        elapsed_ms,
        extra={"request_id": "startup"},
    )


def _warmup_nli_model(config: Configuration) -> None:
    """Best-effort NLI verifier warmup during startup when explicitly enabled."""

    if not getattr(config, "nli_verifier_enabled", False) or not getattr(
        config, "nli_warmup_enabled", False
    ):
        return

    model_name = str(getattr(config, "nli_model_name", "") or "").strip()
    if not model_name:
        return

    started_at = time.perf_counter()
    try:
        from services.claim_verifier import load_nli_model

        model = load_nli_model(
            model_name,
            backend=str(getattr(config, "nli_model_backend", "auto") or "auto"),
            device=str(getattr(config, "nli_device", "auto") or "auto"),
            max_length=int(getattr(config, "nli_max_length", 512) or 512),
            allow_download=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local runtime state
        logger.warning(
            "NLI verifier warmup failed model=%s error=%s",
            model_name,
            exc,
            extra={"request_id": "startup"},
        )
        return

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if model is None:
        logger.info(
            "NLI verifier warmup skipped: local model unavailable model=%s elapsed_ms=%.2f",
            model_name,
            elapsed_ms,
            extra={"request_id": "startup"},
        )
        return

    logger.info(
        "NLI verifier warmup completed model=%s backend=%s elapsed_ms=%.2f",
        model_name,
        model.backend,
        elapsed_ms,
        extra={"request_id": "startup"},
    )


def _request_id(request: Request) -> str:
    """Fetch the request identifier assigned by middleware."""
    return getattr(request.state, "request_id", "-")


def _build_dev_reload_kwargs(config: Configuration) -> dict[str, Any]:
    """Restrict dev reload watchers to source files to avoid runtime-output loops.

    The backend writes notes, request snapshots, caches, and vector-memory artifacts
    underneath the backend project root while serving a request. Watching the whole
    backend directory would make those runtime writes look like source edits and
    trigger a hot-reload loop before planning/execution can finish.
    """

    source_dir = (backend_root() / "src").resolve(strict=False)
    return {
        "reload": True,
        "reload_dirs": [str(source_dir)],
    }


def _resolve_agent_factory(
    base_config: Configuration,
    agent_factory: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    if agent_factory is not None:
        return agent_factory

    if base_config.benchmark_stub_enabled:
        from perf.stub_agent import BenchmarkStubAgent

        return BenchmarkStubAgent

    global DeepResearchAgent
    if DeepResearchAgent is None:
        from agent import DeepResearchAgent as ImportedDeepResearchAgent

        DeepResearchAgent = ImportedDeepResearchAgent

    return DeepResearchAgent


def create_app(
    *,
    base_config: Configuration | None = None,
    agent_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    base_config = base_config or Configuration.from_env()
    resolved_agent_factory = _resolve_agent_factory(base_config, agent_factory)
    configure_logging(base_config.log_level)
    metrics_registry.configure(
        recent_request_limit=base_config.metrics_recent_requests_limit,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = app.state.base_config

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider=%s model=%s base_url=%s search_api=%s "
            "max_loops=%s fetch_full_page=%s tool_calling=%s strip_thinking=%s "
            "request_reflection_enabled=%s reflection_max_additional_tasks=%s "
            "review_stage_enabled=%s review_agent_enabled=%s "
            "task_react_enabled=%s task_react_max_rounds=%s "
            "report_repair_enabled=%s report_repair_max_tasks=%s request_state_enabled=%s api_key=%s "
            "notes_workspace=%s cors_origins=%s search_cache_enabled=%s search_cache_ttl_seconds=%s "
            "approximate_cache_enabled=%s semantic_cache_enabled=%s semantic_cache_warmup_enabled=%s semantic_cache_embedding_model=%s "
            "benchmark_stub_enabled=%s benchmark_profile=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            config.request_reflection_enabled,
            config.reflection_max_additional_tasks,
            config.review_stage_enabled,
            config.review_agent_enabled,
            config.task_react_enabled,
            config.task_react_max_rounds,
            config.report_repair_enabled,
            config.report_repair_max_tasks,
            config.request_state_enabled,
            _mask_secret(config.llm_api_key),
            config.notes_workspace,
            ",".join(config.cors_origins),
            config.search_cache_enabled,
            config.search_cache_ttl_seconds,
            config.resolved_approximate_cache_enabled(),
            config.semantic_cache_enabled,
            config.semantic_cache_warmup_enabled,
            config.semantic_cache_embedding_model,
            config.benchmark_stub_enabled,
            config.benchmark_profile,
            extra={"request_id": "startup"},
        )
        _warmup_semantic_cache_model(config)
        _warmup_nli_model(config)
        if getattr(app.state, "mcp_bridge", None) is None and getattr(config, "mcp_enabled", False):
            app.state.mcp_bridge = MCPBridgeManager(config)
        try:
            yield
        finally:
            bridge = getattr(app.state, "mcp_bridge", None)
            if bridge is not None and hasattr(bridge, "close"):
                bridge.close()
                app.state.mcp_bridge = None
            state_store = getattr(app.state, "request_state_store", None)
            if state_store is not None and hasattr(state_store, "close"):
                state_store.close()

    app = FastAPI(title="HelloAgents Deep Researcher", lifespan=lifespan)
    app.state.base_config = base_config
    app.state.agent_factory = resolved_agent_factory
    app.state.request_state_store = build_request_state_store(base_config)
    app.state.event_bus = build_stream_event_bus(base_config)
    app.state.mcp_bridge = MCPBridgeManager(base_config) if getattr(base_config, "mcp_enabled", False) else None
    rate_limit_redis = None
    if getattr(base_config, "redis_url", None) and _redis_mod is not None:
        try:
            rate_limit_redis = _redis_mod.Redis.from_url(
                str(base_config.redis_url),
                decode_responses=True,
            )
        except Exception:
            rate_limit_redis = None
    app.state.rate_limiter = TenantRateLimiter(
        rpm_limit=base_config.tenant_rate_limit_rpm,
        max_concurrent=base_config.tenant_max_concurrent_requests,
        redis_client=rate_limit_redis,
    )
    app.state.workflow_runner = build_workflow_runner(
        base_config,
        agent_factory=resolved_agent_factory,
        event_bus=app.state.event_bus,
        state_store=app.state.request_state_store,
        on_release=app.state.rate_limiter.release,
    )

    allow_wildcard_cors = "*" in base_config.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=base_config.cors_origins,
        allow_credentials=not allow_wildcard_cors,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-Request-ID",
            "Authorization",
            "X-API-Key",
            "Last-Event-ID",
            "Accept",
        ],
    )

    @app.middleware("http")
    async def add_request_context(
        request: Request,
        call_next,
    ):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex[:12]
        request.state.request_id = request_id
        start_time = time.perf_counter()

        logger.info(
            "Request started method=%s path=%s client=%s",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
            extra={"request_id": request_id},
        )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Request crashed method=%s path=%s",
                request.method,
                request.url.path,
                extra={"request_id": request_id},
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        logger.info(
            "Request completed method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={"request_id": request_id},
        )
        return response

    def _accepts_kwarg(fn: Callable[..., Any] | None, arg_name: str) -> bool:
        if fn is None:
            return False
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == arg_name:
                return True
        return False

    def _resolve_tenant(request: Request) -> TenantContext:
        api_key = extract_api_key(request.headers)
        try:
            return authenticate_tenant(api_key, app.state.base_config)
        except PermissionError as exc:
            raise HTTPException(
                status_code=401,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    def _acquire_rate_limit(tenant: TenantContext) -> None:
        cfg = app.state.base_config
        allowed, retry_after, reason = app.state.rate_limiter.try_acquire(
            tenant.tenant_id,
            rpm_limit=getattr(cfg, "tenant_rate_limit_rpm", 0),
            max_concurrent=getattr(cfg, "tenant_max_concurrent_requests", 0),
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Tenant rate limit exceeded ({reason})",
                headers={"Retry-After": str(retry_after)},
            )

    def _release_rate_limit(tenant: TenantContext) -> None:
        app.state.rate_limiter.release(tenant.tenant_id)

    def _instantiate_agent(
        *,
        config: Configuration,
        request_id: str,
        tenant_id: str,
    ) -> Any:
        factory = app.state.agent_factory
        kwargs: dict[str, Any] = {"config": config, "request_id": request_id}
        if _accepts_kwarg(factory, "tenant_id"):
            kwargs["tenant_id"] = tenant_id
        if _accepts_kwarg(factory, "mcp_bridge") and getattr(app.state, "mcp_bridge", None) is not None:
            kwargs["mcp_bridge"] = app.state.mcp_bridge
        agent = factory(**kwargs)
        if not hasattr(agent, "tenant_id"):
            try:
                setattr(agent, "tenant_id", tenant_id)
            except Exception:
                pass
        return agent

    def _store_supports_tenant_id(store: Any, method_name: str) -> bool:
        return _accepts_kwarg(getattr(store, method_name, None), "tenant_id")

    def _store_load(
        store: Any,
        request_id: str,
        *,
        tenant_id: str | None,
    ) -> dict[str, Any] | None:
        if _store_supports_tenant_id(store, "load"):
            snapshot = store.load(request_id, tenant_id=tenant_id)
        else:
            snapshot = store.load(request_id)
        if not isinstance(snapshot, dict) or (snapshot.get("request_id") or request_id) != request_id:
            return None
        if tenant_id is not None and (snapshot.get("tenant_id") or "default") != tenant_id:
            return None
        return snapshot

    def _store_save(
        store: Any,
        request_id: str,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
    ) -> dict[str, Any]:
        if _store_supports_tenant_id(store, "save"):
            return store.save(request_id, payload, tenant_id=tenant_id)
        return store.save(request_id, payload)

    def _store_list_recent(
        store: Any,
        *,
        limit: int | None,
        include_content: bool,
        tenant_id: str | None,
    ) -> list[dict[str, Any]]:
        if _store_supports_tenant_id(store, "list_recent"):
            return store.list_recent(
                limit=limit,
                include_content=include_content,
                tenant_id=tenant_id,
            )
        return [
            item for item in store.list_recent(limit=limit, include_content=include_content)
            if tenant_id is None or (item.get("tenant_id") or "default") == tenant_id
        ]

    def _ensure_request_ownership(request_id: str, tenant: TenantContext) -> None:
        store = app.state.request_state_store
        if store is None:
            return
        unfiltered = _store_load(store, request_id, tenant_id=None)
        if unfiltered is None:
            return
        owned = _store_load(store, request_id, tenant_id=tenant.tenant_id)
        if owned is None:
            raise HTTPException(
                status_code=404,
                detail=f"request snapshot not found: {request_id}",
            )

    def _save_tool_approval_snapshot(store: Any, snapshot: dict[str, Any], tenant: TenantContext) -> None:
        if not _accepts_kwarg(store.save, "expected_version") or "state_version" not in snapshot:
            raise HTTPException(status_code=409, detail="Tool approvals require a versioned request snapshot")
        store.save(
            snapshot["request_id"], snapshot, tenant_id=tenant.tenant_id,
            expected_version=snapshot["state_version"],
        )

    def _expire_tool_approval(snapshot: dict[str, Any], pending: dict[str, Any]) -> None:
        pending["status"] = "expired"
        snapshot["status"] = "expired"
        snapshot["approved_tool_calls"] = {}

    def _check_tool_approval_before_resume(request_id: str, tenant: TenantContext) -> None:
        store = app.state.request_state_store
        snapshot = _store_load(store, request_id, tenant_id=tenant.tenant_id) if store is not None else None
        if not snapshot or snapshot.get("phase") != "awaiting_tool_approval":
            return
        pending = snapshot.get("pending_tool_approval")
        if not isinstance(pending, dict) or pending.get("status") == "rejected":
            raise HTTPException(status_code=409, detail="Tool approval is closed")
        if tool_approval_expired(pending):
            if pending.get("status") != "expired":
                _expire_tool_approval(snapshot, pending)
                try:
                    _save_tool_approval_snapshot(store, snapshot, tenant)
                except SnapshotConflictError as exc:
                    raise HTTPException(status_code=409, detail="Tool approval state changed") from exc
            raise HTTPException(status_code=410, detail="Tool approval expired")
        if pending.get("status") != "approved_once":
            raise HTTPException(status_code=409, detail="Tool approval is required before resuming")

    @app.get("/healthz")
    async def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics/json")
    async def metrics_json(request: Request) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        return metrics_registry.snapshot(tenant_id=tenant.tenant_id)

    @app.get("/skills")
    async def list_skills_catalog(request: Request) -> Dict[str, Any]:
        _resolve_tenant(request)
        return SkillPackService(app.state.base_config).list_catalog()

    @app.post(
        "/research",
        response_model=ResearchResponse,
        response_model_exclude_none=True,
        response_model_exclude_unset=True,
    )
    def run_research(payload: ResearchRequest, request: Request) -> ResearchResponse:
        request_id = _request_id(request)
        tenant = _resolve_tenant(request)
        _ensure_request_ownership(request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(payload, base_config=app.state.base_config)
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
            result = agent.run(payload.topic)
        except HTTPException:
            raise
        except SnapshotConflictError as exc:
            raise HTTPException(status_code=409, detail="Request snapshot changed") from exc
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception(
                "Research failed topic=%s",
                payload.topic,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Research failed (request_id={request_id})",
            ) from exc
        finally:
            _release_rate_limit(tenant)

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "notices": getattr(item, "notices", []),
                "evidence_items": getattr(item, "evidence_items", []),
                "claims": getattr(item, "claims", []),
                **(
                    {"atomic_claims": list(getattr(item, "atomic_claims", []) or [])}
                    if getattr(item, "atomic_claims", None)
                    else {}
                ),
                "review_issues": getattr(item, "review_issues", []),
                "review_status": getattr(item, "review_status", "pending"),
                "note_id": item.note_id,
                "note_path": item.note_path,
                "origin": getattr(item, "origin", "planned"),
                "round": getattr(item, "round", 1),
                **(
                    {"depends_on": list(getattr(item, "depends_on", []) or [])}
                    if getattr(item, "depends_on", None)
                    else {}
                ),
            }
            for item in result.todo_items
        ]

        raw_conflicts = list(getattr(result, "evidence_conflicts", []) or [])
        raw_summary = dict(getattr(result, "claim_verification_summary", {}) or {})
        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
            charts_spec=list(getattr(result, "report_charts", []) or []) or None,
            **({"evidence_conflicts": raw_conflicts} if raw_conflicts else {}),
            **({"claim_verification_summary": raw_summary} if raw_summary else {}),
        )

    @app.post("/research/clarify")
    async def clarify_research(payload: ClarifyRequest, request: Request) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(topic=payload.topic, search_api=payload.search_api),
                base_config=app.state.base_config,
            )
            skill_pack = SkillPackService(config)
            return HitlService.clarify_topic(
                payload.topic,
                config=config,
                skill_pack_service=skill_pack,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            _release_rate_limit(tenant)

    @app.post("/research/plan")
    def create_editable_plan(payload: ResearchRequest, request: Request) -> Dict[str, Any]:
        request_id = _request_id(request)
        tenant = _resolve_tenant(request)
        _ensure_request_ownership(request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(payload, base_config=app.state.base_config)
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
            return agent.plan_only(payload.topic)
        except HTTPException:
            raise
        except SnapshotConflictError as exc:
            raise HTTPException(status_code=409, detail="Request snapshot changed") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "HITL plan_only failed topic=%s",
                payload.topic,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Planning failed (request_id={request_id})",
            ) from exc
        finally:
            _release_rate_limit(tenant)

    @app.put("/requests/{request_id}/plan")
    async def update_editable_plan(
        request_id: str,
        payload: PlanUpdateRequest,
        request: Request,
    ) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        store = app.state.request_state_store
        if store is None:
            raise HTTPException(status_code=404, detail="request state store is disabled")

        snapshot = await asyncio.to_thread(
            _store_load,
            store,
            request_id,
            tenant_id=tenant.tenant_id,
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"request snapshot not found: {request_id}")

        if snapshot.get("pending_tool_approval") or snapshot.get("approved_tool_calls"):
            raise HTTPException(status_code=409, detail="Request has a tool approval record")

        config = app.state.base_config
        try:
            normalized_tasks = HitlService.validate_editable_tasks(
                [item.model_dump() for item in payload.tasks],
                max_tasks=config.resolved_max_total_tasks(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        updated_snapshot = dict(snapshot)
        updated_snapshot["tenant_id"] = tenant.tenant_id
        updated_snapshot["todo_items"] = normalized_tasks
        updated_snapshot["phase"] = "awaiting_approval"
        updated_snapshot["status"] = "awaiting_approval"
        updated_snapshot["review_completed"] = False
        updated_snapshot["report_repair_completed"] = False
        updated_snapshot["review_summary"] = {}
        updated_snapshot["evidence_conflicts"] = []
        updated_snapshot["claim_verification_summary"] = {}
        if isinstance(updated_snapshot.get("request_metrics"), dict):
            updated_snapshot["request_metrics"] = {
                key: value for key, value in updated_snapshot["request_metrics"].items()
                if key not in {"review_summary", "claim_verification_summary", "todo_items"}
            }
        saved = await asyncio.to_thread(
            _store_save,
            store,
            request_id,
            updated_snapshot,
            tenant_id=tenant.tenant_id,
        )
        return {
            "request_id": request_id,
            "topic": saved.get("topic", ""),
            "status": saved.get("status", "awaiting_approval"),
            "phase": saved.get("phase", "awaiting_approval"),
            "todo_items": saved.get("todo_items", []),
            "estimated_duration_seconds": HitlService.estimate_duration_seconds(
                len(normalized_tasks), config
            ),
        }

    @app.post("/requests/{request_id}/drilldown")
    def drilldown_report_section(
        request_id: str,
        payload: DrilldownRequest,
        request: Request,
    ) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        _ensure_request_ownership(request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(topic="drilldown", search_api=payload.search_api),
                base_config=app.state.base_config,
            )
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
            return agent.drilldown_section(
                request_id,
                section_title=payload.section_title,
                question=payload.question,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "Drilldown failed request_id=%s section=%s",
                request_id,
                payload.section_title,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Drilldown failed (request_id={request_id})",
            ) from exc
        finally:
            _release_rate_limit(tenant)

    @app.get("/requests")
    async def list_persisted_requests(
        request: Request,
        limit: int | None = None,
        include_content: bool = False,
    ) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        store = app.state.request_state_store
        if store is None:
            return {"items": []}
        items = await asyncio.to_thread(
            _store_list_recent,
            store,
            limit=limit,
            include_content=include_content,
            tenant_id=tenant.tenant_id,
        )
        return {"items": items}

    @app.get("/requests/{request_id}")
    async def get_persisted_request(request_id: str, request: Request) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        store = app.state.request_state_store
        if store is None:
            raise HTTPException(status_code=404, detail="request state store is disabled")

        payload = await asyncio.to_thread(
            _store_load,
            store,
            request_id,
            tenant_id=tenant.tenant_id,
        )
        if payload is None:
            raise HTTPException(status_code=404, detail=f"request snapshot not found: {request_id}")
        return payload

    @app.get("/requests/{request_id}/events")
    async def replay_request_events(
        request_id: str,
        request: Request,
        last_event_id: str | None = None,
        follow: bool = False,
    ) -> Any:
        tenant = _resolve_tenant(request)
        await asyncio.to_thread(_ensure_request_ownership, request_id, tenant)
        resolved_last_id = (
            last_event_id
            if last_event_id is not None
            else request.headers.get("last-event-id")
        )
        if follow:
            runner = getattr(app.state, "workflow_runner", None)
            store = getattr(app.state, "request_state_store", None)

            def _is_terminal() -> bool:
                if runner is not None:
                    rstat = runner.status(request_id, tenant_id=tenant.tenant_id)
                    if isinstance(rstat, dict) and rstat.get("status") in {
                        "completed",
                        "failed",
                        "cancelled",
                    }:
                        return True
                if store is not None:
                    snap = _store_load(store, request_id, tenant_id=tenant.tenant_id)
                    if isinstance(snap, dict) and snap.get("status") in {
                        "success",
                        "partial_success",
                        "failed",
                        "cancelled",
                    }:
                        return True
                return False

            async def _follow_iter() -> AsyncIterator[str]:
                tail_q: Queue[tuple[str, Any]] = Queue()

                def _pump() -> None:
                    try:
                        tail_fn = getattr(app.state.event_bus, "tail_since", None)
                        if callable(tail_fn):
                            for env in tail_fn(
                                tenant.tenant_id,
                                request_id,
                                last_event_id=resolved_last_id,
                                is_terminal_fn=_is_terminal,
                                block_ms=int(
                                    max(
                                        200,
                                        float(
                                            getattr(
                                                app.state.base_config,
                                                "sse_heartbeat_interval_seconds",
                                                15.0,
                                            )
                                            or 15.0
                                        )
                                        * 1000,
                                    )
                                ),
                            ):
                                tail_q.put(("env", env))
                        else:
                            for env in app.state.event_bus.replay_since(
                                tenant.tenant_id,
                                request_id,
                                last_event_id=resolved_last_id,
                            ):
                                tail_q.put(("env", env))
                    except Exception as exc:
                        tail_q.put(("error", exc))
                    finally:
                        tail_q.put(("end", None))

                pump_thread = Thread(
                    target=_pump,
                    name=f"sse-tail-{request_id[:8]}",
                    daemon=True,
                )
                pump_thread.start()
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        kind, item = await asyncio.to_thread(tail_q.get, True, 0.25)
                    except Empty:
                        continue
                    if kind == "env":
                        payload = item.payload
                        if payload.get("type") == "__keepalive__":
                            yield ": keepalive\n\n"
                        else:
                            yield (
                                f"id: {item.event_id}\n"
                                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                            )
                    elif kind == "error":
                        yield (
                            f"data: {json.dumps({'type': 'error', 'detail': str(item)}, ensure_ascii=False)}\n\n"
                        )
                        break
                    else:
                        break

            return StreamingResponse(
                _follow_iter(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": request_id,
                },
            )

        envelopes = app.state.event_bus.replay_since(
            tenant.tenant_id,
            request_id,
            last_event_id=resolved_last_id,
        )
        accept = str(request.headers.get("accept") or "").lower()
        if "text/event-stream" in accept:
            async def _replay_iter() -> AsyncIterator[str]:
                for envelope in envelopes:
                    yield (
                        f"id: {envelope.event_id}\n"
                        f"data: {json.dumps(envelope.payload, ensure_ascii=False)}\n\n"
                    )

            return StreamingResponse(
                _replay_iter(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return {
            "request_id": request_id,
            "tenant_id": tenant.tenant_id,
            "events": [
                {"id": envelope.event_id, "data": envelope.payload}
                for envelope in envelopes
            ],
        }

    @app.post("/research/async", status_code=202)
    async def submit_async_research(
        payload: AsyncResearchRequest,
        request: Request,
    ) -> Dict[str, Any]:
        request_id = _request_id(request)
        tenant = _resolve_tenant(request)
        await asyncio.to_thread(_ensure_request_ownership, request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(
                    topic=payload.topic,
                    search_api=payload.search_api,
                    output_format=payload.output_format,
                ),
                base_config=app.state.base_config,
            )
            runner = app.state.workflow_runner
            handle = await asyncio.to_thread(
                runner.submit,
                payload.topic,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
                config=config,
                output_format=payload.output_format,
            )
            if runner.backend_name == "inline":
                _release_rate_limit(tenant)
            return {
                **handle.to_dict(),
                "events_url": f"/requests/{request_id}/events?follow=1",
            }
        except HTTPException:
            _release_rate_limit(tenant)
            raise
        except ValueError as exc:
            _release_rate_limit(tenant)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            _release_rate_limit(tenant)
            logger.exception("Async submit failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail=f"Async submit failed: {exc}") from exc

    @app.post("/requests/{request_id}/signals/{signal_name}")
    async def send_workflow_signal(
        request_id: str,
        signal_name: str,
        request: Request,
        payload: WorkflowSignalRequest | None = None,
    ) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        await asyncio.to_thread(_ensure_request_ownership, request_id, tenant)
        runner = app.state.workflow_runner
        signal_body = payload.payload if payload is not None else {}
        return await asyncio.to_thread(
            runner.signal,
            request_id,
            signal_name,
            signal_body,
            tenant_id=tenant.tenant_id,
        )

    @app.post(
        "/requests/{request_id}/resume",
        response_model=ResearchResponse,
        response_model_exclude_none=True,
        response_model_exclude_unset=True,
    )
    def resume_research(
        request_id: str,
        request: Request,
        payload: ResumeRequest | None = None,
    ) -> ResearchResponse:
        tenant = _resolve_tenant(request)
        _ensure_request_ownership(request_id, tenant)
        _check_tool_approval_before_resume(request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(
                    topic="resume",
                    search_api=payload.search_api if payload else None,
                ),
                base_config=app.state.base_config,
            )
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
            result = agent.run_resume(request_id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception(
                "Resume research failed request_id=%s",
                request_id,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Resume failed (request_id={request_id})",
            ) from exc
        finally:
            _release_rate_limit(tenant)

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "notices": getattr(item, "notices", []),
                "evidence_items": getattr(item, "evidence_items", []),
                "claims": getattr(item, "claims", []),
                **(
                    {"atomic_claims": list(getattr(item, "atomic_claims", []) or [])}
                    if getattr(item, "atomic_claims", None)
                    else {}
                ),
                "review_issues": getattr(item, "review_issues", []),
                "review_status": getattr(item, "review_status", "pending"),
                "note_id": item.note_id,
                "note_path": item.note_path,
                "origin": getattr(item, "origin", "planned"),
                "round": getattr(item, "round", 1),
                **(
                    {"depends_on": list(getattr(item, "depends_on", []) or [])}
                    if getattr(item, "depends_on", None)
                    else {}
                ),
            }
            for item in result.todo_items
        ]
        raw_conflicts = list(getattr(result, "evidence_conflicts", []) or [])
        raw_summary = dict(getattr(result, "claim_verification_summary", {}) or {})
        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
            charts_spec=list(getattr(result, "report_charts", []) or []) or None,
            **({"evidence_conflicts": raw_conflicts} if raw_conflicts else {}),
            **({"claim_verification_summary": raw_summary} if raw_summary else {}),
        )

    def _build_sse_stream(
        stream_starter: Callable[[Event], Iterator[dict[str, Any]]],
        *,
        request: Request | None,
        request_id: str,
        tenant_id: str = "default",
        heartbeat_seconds: float,
        error_log_label: str,
        cancel_on_disconnect: bool = True,
        on_complete: Callable[[], None] | None = None,
    ) -> StreamingResponse:
        cancel_event = Event()
        event_queue: Queue[tuple[str, Any]] = Queue()
        heartbeat_interval = max(0.05, float(heartbeat_seconds or 15.0))
        client_detached = Event()

        def _publish_and_format(event_payload: dict[str, Any]) -> str:
            event_id = app.state.event_bus.publish(tenant_id, request_id, event_payload)
            return (
                f"id: {event_id}\n"
                f"data: {json.dumps(event_payload, ensure_ascii=False)}\n\n"
            )

        def producer() -> None:
            try:
                for event in stream_starter(cancel_event):
                    if cancel_on_disconnect and cancel_event.is_set():
                        break
                    if not cancel_on_disconnect and client_detached.is_set():
                        if isinstance(event, dict):
                            app.state.event_bus.publish(tenant_id, request_id, event)
                        continue
                    event_queue.put(("event", event))
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception(
                    "%s",
                    error_log_label,
                    extra={"request_id": request_id},
                )
                if not cancel_on_disconnect and client_detached.is_set():
                    app.state.event_bus.publish(
                        tenant_id,
                        request_id,
                        {"type": "error", "detail": f"{exc} (request_id={request_id})"},
                    )
                else:
                    event_queue.put(("error", exc))
            finally:
                if not cancel_on_disconnect and client_detached.is_set():
                    if on_complete is not None:
                        on_complete()
                event_queue.put(("done", None))

        worker = Thread(
            target=producer,
            name=f"sse-producer-{request_id[:8]}",
            daemon=True,
        )
        worker.start()

        async def event_iterator() -> AsyncIterator[str]:
            last_emitted = time.monotonic()
            poll_slice = min(0.1, heartbeat_interval)
            detached = False
            try:
                while True:
                    if request is not None and await request.is_disconnected():
                        detached = True
                        client_detached.set()
                        if cancel_on_disconnect:
                            cancel_event.set()
                        break

                    try:
                        kind, payload = await asyncio.to_thread(event_queue.get, True, poll_slice)
                    except Empty:
                        now = time.monotonic()
                        if now - last_emitted >= heartbeat_interval:
                            yield ": keepalive\n\n"
                            last_emitted = now
                        continue

                    if kind == "event":
                        yield _publish_and_format(payload)
                        last_emitted = time.monotonic()
                    elif kind == "error":
                        error_payload = {
                            "type": "error",
                            "detail": f"{payload} (request_id={request_id})",
                        }
                        yield _publish_and_format(error_payload)
                        break
                    else:
                        break
            finally:
                if cancel_on_disconnect or not detached:
                    cancel_event.set()
                    await asyncio.to_thread(worker.join, 1.0)
                    if on_complete is not None:
                        on_complete()

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    def _supports_cancel_event(fn: Callable[..., Any]) -> bool:
        return _accepts_kwarg(fn, "cancel_event")

    @app.post("/research/stream")
    async def stream_research(payload: ResearchRequest, request: Request) -> StreamingResponse:
        request_id = _request_id(request)
        tenant = _resolve_tenant(request)
        await asyncio.to_thread(_ensure_request_ownership, request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(payload, base_config=app.state.base_config)
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
        except HTTPException:
            _release_rate_limit(tenant)
            raise
        except ValueError as exc:
            _release_rate_limit(tenant)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            _release_rate_limit(tenant)
            logger.exception(
                "Failed to initialise streaming research topic=%s",
                payload.topic,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Research failed to start (request_id={request_id})",
            ) from exc

        def _start(cancel_event: Event) -> Iterator[dict[str, Any]]:
            if _supports_cancel_event(agent.run_stream):
                return agent.run_stream(payload.topic, cancel_event=cancel_event)
            return agent.run_stream(payload.topic)

        return _build_sse_stream(
            _start,
            request=request,
            request_id=request_id,
            tenant_id=tenant.tenant_id,
            heartbeat_seconds=config.sse_heartbeat_interval_seconds,
            error_log_label=f"Streaming research failed topic={payload.topic}",
            cancel_on_disconnect=(getattr(config, "workflow_runner", "inline") == "inline"),
            on_complete=lambda: _release_rate_limit(tenant),
        )

    @app.post("/requests/{request_id}/resume/stream")
    async def stream_resume_research(
        request_id: str,
        request: Request,
        payload: ResumeRequest | None = None,
    ) -> StreamingResponse:
        tenant = _resolve_tenant(request)
        await asyncio.to_thread(_ensure_request_ownership, request_id, tenant)
        await asyncio.to_thread(_check_tool_approval_before_resume, request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(
                    topic="resume",
                    search_api=payload.search_api if payload else None,
                ),
                base_config=app.state.base_config,
            )
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
        except HTTPException:
            _release_rate_limit(tenant)
            raise
        except ValueError as exc:
            _release_rate_limit(tenant)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            _release_rate_limit(tenant)
            logger.exception(
                "Failed to initialise streaming resume request_id=%s",
                request_id,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Resume failed to start (request_id={request_id})",
            ) from exc

        def _start(cancel_event: Event) -> Iterator[dict[str, Any]]:
            if _supports_cancel_event(agent.run_stream_resume):
                return agent.run_stream_resume(request_id, cancel_event=cancel_event)
            return agent.run_stream_resume(request_id)

        return _build_sse_stream(
            _start,
            request=request,
            request_id=request_id,
            tenant_id=tenant.tenant_id,
            heartbeat_seconds=config.sse_heartbeat_interval_seconds,
            error_log_label=f"Streaming resume failed request_id={request_id}",
            cancel_on_disconnect=(getattr(config, "workflow_runner", "inline") == "inline"),
            on_complete=lambda: _release_rate_limit(tenant),
        )

    @app.post("/requests/{request_id}/tool-approvals/{approval_id}")
    def resolve_mcp_tool_approval(
        request_id: str,
        approval_id: str,
        payload: ToolApprovalDecisionRequest,
        request: Request,
    ) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        _ensure_request_ownership(request_id, tenant)
        store = app.state.request_state_store
        if store is None:
            raise HTTPException(status_code=400, detail="request_state store is disabled")
        snapshot = _store_load(store, request_id, tenant_id=tenant.tenant_id)
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("request_id") != request_id
            or snapshot.get("tenant_id") != tenant.tenant_id
        ):
            raise HTTPException(status_code=404, detail=f"request snapshot not found: {request_id}")
        pending = snapshot.get("pending_tool_approval")
        if not isinstance(pending, dict) or str(pending.get("approval_id") or "") != approval_id:
            raise HTTPException(status_code=404, detail=f"pending tool approval not found: {approval_id}")
        digest = str(pending.get("arguments_digest") or "").strip()
        server_id = str(pending.get("server_id") or "").strip()
        tool_name = str(pending.get("tool_name") or "").strip()
        if (
            pending.get("request_id") != request_id
            or pending.get("tenant_id") != tenant.tenant_id
            or not server_id or not tool_name
            or not isinstance(pending.get("arguments_preview"), dict)
            or digest != compute_arguments_digest(pending["arguments_preview"])
            or (payload.arguments_digest is not None and digest != payload.arguments_digest)
        ):
            raise HTTPException(status_code=409, detail="pending tool approval metadata is incomplete")

        desired = "approved_once" if payload.decision == "approve" else "rejected"
        response = {
            "request_id": request_id,
            "approval_id": approval_id,
            "decision": payload.decision,
            "status": desired,
        }
        status = pending.get("status")
        if status == "rejected" and status == desired:
            return response
        if status == "expired" or (status in {"awaiting_approval", "approved_once"} and tool_approval_expired(pending)):
            if status != "expired":
                _expire_tool_approval(snapshot, pending)
                try:
                    _save_tool_approval_snapshot(store, snapshot, tenant)
                except SnapshotConflictError as exc:
                    raise HTTPException(status_code=409, detail="Tool approval state changed") from exc
            raise HTTPException(status_code=410, detail="Tool approval expired")
        if status == "approved_once" and status == desired:
            return response
        if status != "awaiting_approval" or snapshot.get("phase") != "awaiting_tool_approval":
            raise HTTPException(status_code=409, detail="Tool approval is already resolved")

        approved_map = dict(snapshot.get("approved_tool_calls") or {})
        approval_key = f"{server_id}:{tool_name}:{digest}"
        if payload.decision == "approve":
            approved_map[approval_key] = {
                "approval_id": approval_id,
                "server_id": server_id,
                "tool_name": tool_name,
                "arguments_digest": digest,
                "tenant_id": tenant.tenant_id,
                "request_id": request_id,
                "task_id": pending.get("task_id"),
                "expires_at": pending["expires_at"],
                "status": "approved_once",
                "reason": payload.reason,
            }
            pending["status"] = "approved_once"
            snapshot["status"] = "approved_ready"
        else:
            approved_map.pop(approval_key, None)
            pending["status"] = "rejected"
            pending["reason"] = payload.reason
            snapshot["status"] = "rejected"
        snapshot["pending_tool_approval"] = pending
        snapshot["approved_tool_calls"] = approved_map
        try:
            _save_tool_approval_snapshot(store, snapshot, tenant)
        except SnapshotConflictError as exc:
            latest = _store_load(store, request_id, tenant_id=tenant.tenant_id) or {}
            resolved = latest.get("pending_tool_approval") or {}
            if resolved.get("approval_id") == approval_id and resolved.get("status") == desired:
                return response
            raise HTTPException(status_code=409, detail="Tool approval state changed") from exc
        return response

    @app.post("/api/research/tool-approval")
    def resolve_research_tool_approval(payload: ResearchToolApprovalRequest, request: Request) -> Dict[str, Any]:
        return resolve_mcp_tool_approval(payload.request_id, payload.approval_id, payload, request)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime_config = Configuration.from_env()
    uvicorn.run(
        "main:app",
        host=runtime_config.host,
        port=runtime_config.port,
        log_level=runtime_config.log_level.lower(),
        **_build_dev_reload_kwargs(runtime_config),
    )
