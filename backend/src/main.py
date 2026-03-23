"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import DeepResearchAgent
from config import Configuration, SearchAPI
from metrics import metrics_registry

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


def _request_id(request: Request) -> str:
    """Fetch the request identifier assigned by middleware."""
    return getattr(request.state, "request_id", "-")


def create_app() -> FastAPI:
    base_config = Configuration.from_env()
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
            "max_loops=%s fetch_full_page=%s tool_calling=%s strip_thinking=%s api_key=%s "
            "notes_workspace=%s cors_origins=%s search_cache_enabled=%s search_cache_ttl_seconds=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            _mask_secret(config.llm_api_key),
            config.notes_workspace,
            ",".join(config.cors_origins),
            config.search_cache_enabled,
            config.search_cache_ttl_seconds,
            extra={"request_id": "startup"},
        )
        yield

    app = FastAPI(title="HelloAgents Deep Researcher", lifespan=lifespan)
    app.state.base_config = base_config

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
            agent = DeepResearchAgent(config=config, request_id=request_id)
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
                "note_id": item.note_id,
                "note_path": item.note_path,
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
            agent = DeepResearchAgent(config=config, request_id=request_id)
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

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime_config = Configuration.from_env()
    uvicorn.run(
        "main:app",
        host=runtime_config.host,
        port=runtime_config.port,
        reload=True,
        reload_dirs=["./"],
        log_level=runtime_config.log_level.lower(),
    )
