"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Iterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI, backend_root
from metrics import metrics_registry
from services.request_state import RequestStateStore

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
    app.state.request_state_store = (
        RequestStateStore(
            base_config.request_state_dir,
            recent_limit=base_config.request_state_recent_limit,
        )
        if base_config.request_state_enabled
        else None
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=base_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
        logger.info(
            "Request completed method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={"request_id": request_id},
        )
        return response

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics/json")
    def metrics_json() -> Dict[str, Any]:
        return metrics_registry.snapshot()

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest, request: Request) -> ResearchResponse:
        request_id = _request_id(request)
        try:
            config = _build_config(payload)
            agent = app.state.agent_factory(config=config, request_id=request_id)
            result = agent.run(payload.topic)
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
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.get("/requests")
    def list_persisted_requests(limit: int | None = None) -> Dict[str, Any]:
        store = app.state.request_state_store
        if store is None:
            return {"items": []}
        return {"items": store.list_recent(limit=limit)}

    @app.get("/requests/{request_id}")
    def get_persisted_request(request_id: str) -> Dict[str, Any]:
        store = app.state.request_state_store
        if store is None:
            raise HTTPException(status_code=404, detail="request state store is disabled")

        payload = store.load(request_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"request snapshot not found: {request_id}")
        return payload

    @app.post("/requests/{request_id}/resume", response_model=ResearchResponse)
    def resume_research(
        request_id: str,
        payload: ResumeRequest | None = None,
    ) -> ResearchResponse:
        try:
            config = _build_config(
                ResearchRequest(
                    topic="resume",
                    search_api=payload.search_api if payload else None,
                )
            )
            agent = app.state.agent_factory(config=config, request_id=request_id)
            result = agent.run_resume(request_id)
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
            }
            for item in result.todo_items
        ]
        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest, request: Request) -> StreamingResponse:
        request_id = _request_id(request)
        try:
            config = _build_config(payload)
            agent = app.state.agent_factory(config=config, request_id=request_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception(
                "Failed to initialise streaming research topic=%s",
                payload.topic,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Research failed to start (request_id={request_id})",
            ) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(payload.topic):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception(
                    "Streaming research failed topic=%s",
                    payload.topic,
                    extra={"request_id": request_id},
                )
                error_payload = {
                    "type": "error",
                    "detail": f"{exc} (request_id={request_id})",
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    @app.post("/requests/{request_id}/resume/stream")
    def stream_resume_research(
        request_id: str,
        payload: ResumeRequest | None = None,
    ) -> StreamingResponse:
        try:
            config = _build_config(
                ResearchRequest(
                    topic="resume",
                    search_api=payload.search_api if payload else None,
                )
            )
            agent = app.state.agent_factory(config=config, request_id=request_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception(
                "Failed to initialise streaming resume request_id=%s",
                request_id,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Resume failed to start (request_id={request_id})",
            ) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream_resume(request_id):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception(
                    "Streaming resume failed request_id=%s",
                    request_id,
                    extra={"request_id": request_id},
                )
                error_payload = {
                    "type": "error",
                    "detail": f"{exc} (request_id={request_id})",
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
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
