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
from typing import Any, Callable, Dict, Iterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI, backend_root
from metrics import metrics_registry
from services.event_bus import build_stream_event_bus
from services.hitl import HitlService
from services.skill_pack import SkillPackService
from services.storage_backends import build_request_state_store
from services.tenant import (
    TenantContext,
    TenantRateLimiter,
    authenticate_tenant,
    extract_api_key,
)

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


def _mask_secret(value: str | None, visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    return Configuration.from_env(overrides=overrides)


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
        yield

    app = FastAPI(title="HelloAgents Deep Researcher", lifespan=lifespan)
    app.state.base_config = base_config
    app.state.agent_factory = resolved_agent_factory
    app.state.request_state_store = build_request_state_store(base_config)
    app.state.event_bus = build_stream_event_bus(base_config)
    app.state.rate_limiter = TenantRateLimiter(
        rpm_limit=base_config.tenant_rate_limit_rpm,
        max_concurrent=base_config.tenant_max_concurrent_requests,
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
            return store.load(request_id, tenant_id=tenant_id)
        return store.load(request_id)

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
        return store.list_recent(limit=limit, include_content=include_content)

    def _ensure_request_ownership(request_id: str, tenant: TenantContext) -> None:
        store = app.state.request_state_store
        if store is None or not _store_supports_tenant_id(store, "load"):
            return
        unfiltered = store.load(request_id, tenant_id=None)
        if unfiltered is None:
            return
        owned = store.load(request_id, tenant_id=tenant.tenant_id)
        if owned is None:
            raise HTTPException(
                status_code=404,
                detail=f"request snapshot not found: {request_id}",
            )

    @app.get("/healthz")
    async def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics/json")
    async def metrics_json(request: Request) -> Dict[str, Any]:
        _resolve_tenant(request)
        return metrics_registry.snapshot()

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest, request: Request) -> ResearchResponse:
        request_id = _request_id(request)
        tenant = _resolve_tenant(request)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(payload)
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
            result = agent.run(payload.topic)
        except HTTPException:
            raise
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

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/clarify")
    async def clarify_research(payload: ClarifyRequest, request: Request) -> Dict[str, Any]:
        tenant = _resolve_tenant(request)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(topic=payload.topic, search_api=payload.search_api)
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
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(payload)
            agent = _instantiate_agent(
                config=config,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )
            return agent.plan_only(payload.topic)
        except HTTPException:
            raise
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

        config = Configuration.from_env()
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
                ResearchRequest(topic="drilldown", search_api=payload.search_api)
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
    ) -> Any:
        tenant = _resolve_tenant(request)
        await asyncio.to_thread(_ensure_request_ownership, request_id, tenant)
        resolved_last_id = (
            last_event_id
            if last_event_id is not None
            else request.headers.get("last-event-id")
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

    @app.post("/requests/{request_id}/resume", response_model=ResearchResponse)
    def resume_research(
        request_id: str,
        request: Request,
        payload: ResumeRequest | None = None,
    ) -> ResearchResponse:
        tenant = _resolve_tenant(request)
        _ensure_request_ownership(request_id, tenant)
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(
                    topic="resume",
                    search_api=payload.search_api if payload else None,
                )
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
        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    def _build_sse_stream(
        stream_starter: Callable[[Event], Iterator[dict[str, Any]]],
        *,
        request: Request | None,
        request_id: str,
        tenant_id: str = "default",
        heartbeat_seconds: float,
        error_log_label: str,
        on_complete: Callable[[], None] | None = None,
    ) -> StreamingResponse:
        cancel_event = Event()
        event_queue: Queue[tuple[str, Any]] = Queue()
        heartbeat_interval = max(0.05, float(heartbeat_seconds or 15.0))

        def _publish_and_format(event_payload: dict[str, Any]) -> str:
            event_id = app.state.event_bus.publish(tenant_id, request_id, event_payload)
            return (
                f"id: {event_id}\n"
                f"data: {json.dumps(event_payload, ensure_ascii=False)}\n\n"
            )

        def producer() -> None:
            try:
                for event in stream_starter(cancel_event):
                    if cancel_event.is_set():
                        break
                    event_queue.put(("event", event))
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception(
                    "%s",
                    error_log_label,
                    extra={"request_id": request_id},
                )
                event_queue.put(("error", exc))
            finally:
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
            try:
                while True:
                    if request is not None and await request.is_disconnected():
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
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(payload)
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
        _acquire_rate_limit(tenant)
        try:
            config = _build_config(
                ResearchRequest(
                    topic="resume",
                    search_api=payload.search_api if payload else None,
                )
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
            on_complete=lambda: _release_rate_limit(tenant),
        )

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
