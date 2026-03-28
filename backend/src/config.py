import json
import os
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _BACKEND_ROOT / ".env"
_DOTENV_LOADED = False


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _expand_loopback_origin(origin: str) -> list[str]:
    cleaned = (origin or "").strip()
    if not cleaned or cleaned == "*":
        return [cleaned] if cleaned else []

    parsed = urlsplit(cleaned)
    hostname = (parsed.hostname or "").strip().lower()
    if parsed.scheme not in {"http", "https"} or hostname not in {"localhost", "127.0.0.1"}:
        return [cleaned]

    port = f":{parsed.port}" if parsed.port is not None else ""
    paired_host = "127.0.0.1" if hostname == "localhost" else "localhost"
    paired_origin = f"{parsed.scheme}://{paired_host}{port}"
    if paired_origin == cleaned:
        return [cleaned]

    return [cleaned, paired_origin]


def _normalize_origin_values(value: Any) -> list[str]:
    if value is None:
        return ["*"]

    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value if str(item).strip()]
    else:
        return value

    normalized: list[str] = []
    for item in raw_items or ["*"]:
        for expanded in _expand_loopback_origin(item):
            _append_unique(normalized, expanded)

    return normalized or ["*"]


def backend_root() -> Path:
    """Return the backend project root directory."""
    return _BACKEND_ROOT


def _load_backend_env() -> None:
    """Load the backend .env file once when available."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    load_dotenv(_DOTENV_PATH, override=False)
    _DOTENV_LOADED = True


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"

    @classmethod
    def fusion_backends(cls) -> set[str]:
        """Return the concrete search backends allowed in fusion mode."""
        return {member.value for member in cls if member is not cls.ADVANCED}


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    advanced_search_backends: list[str] = Field(
        default_factory=lambda: ["searxng", "tavily", "serpapi", "duckduckgo"],
        title="Advanced Search Backends",
        description="Ordered backends to query and fuse when search_api=advanced",
    )
    enable_notes: bool = Field(
        default=True,
        title="Enable Notes",
        description="Whether to store task progress in NoteTool",
    )
    notes_workspace: str = Field(
        default="./notes",
        title="Notes Workspace",
        description="Directory for NoteTool to persist task notes",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=True,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    llm_api_key: str | None = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: str | None = Field(
        default=None,
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_model_id: str | None = Field(
        default=None,
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )
    llm_context_window: int = Field(
        default=32768,
        title="LLM Context Window",
        description="Approximate maximum prompt context supported by the active model",
    )
    host: str = Field(
        default="0.0.0.0",
        title="Server Host",
        description="Host interface used when starting the development server",
    )
    port: int = Field(
        default=8000,
        title="Server Port",
        description="Port used when starting the development server",
    )
    log_level: str = Field(
        default="INFO",
        title="Log Level",
        description="Application log verbosity",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        title="CORS Origins",
        description="Allowed CORS origins for the HTTP API",
    )
    search_cache_enabled: bool = Field(
        default=True,
        title="Search Cache Enabled",
        description="Whether to cache search results in-process",
    )
    search_cache_ttl_seconds: int = Field(
        default=900,
        title="Search Cache TTL",
        description="How long to keep in-process search cache entries",
    )
    search_cache_dir: str = Field(
        default="./.cache/search",
        title="Search Cache Directory",
        description="Directory used by the persistent search cache store",
    )
    semantic_cache_enabled: bool = Field(
        default=True,
        title="Semantic Cache Enabled",
        description="Whether to use embedding similarity to reuse semantically similar search results",
    )
    semantic_cache_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        title="Semantic Cache Embedding Model",
        description="Embedding model used to vectorize search queries for semantic cache matching",
    )
    semantic_cache_similarity_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        title="Semantic Cache Similarity Threshold",
        description="Cosine similarity threshold for semantic search cache hits",
    )
    semantic_cache_lexical_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        title="Semantic Cache Lexical Threshold",
        description="Lexical similarity threshold used to reuse semantically related cache entries when embeddings are weak",
    )
    max_tokens_per_source: int | None = Field(
        default=None,
        title="Max Tokens Per Source",
        description="Optional override for source token budget used when expanding search results into prompts",
    )
    task_context_char_limit: int | None = Field(
        default=None,
        title="Task Context Character Limit",
        description="Optional override for task summarizer context character budget",
    )
    task_summary_max_concurrency: int = Field(
        default=1,
        ge=1,
        title="Task Summary Max Concurrency",
        description="Maximum number of task summarization LLM calls allowed to run at once",
    )
    max_agent_tasks: int = Field(
        default=5,
        ge=1,
        title="Max Agent Tasks",
        description="Maximum number of planned tasks the agent will execute for a single request",
    )
    request_reflection_enabled: bool = Field(
        default=True,
        title="Request Reflection Enabled",
        description="Whether to run a single request-level reflection step before report generation",
    )
    reflection_max_additional_tasks: int = Field(
        default=2,
        ge=1,
        title="Reflection Max Additional Tasks",
        description="Maximum number of supplemental tasks that request-level reflection may add",
    )
    review_stage_enabled: bool = Field(
        default=True,
        title="Review Stage Enabled",
        description="Whether to run a request-level review stage before report generation",
    )
    review_agent_enabled: bool = Field(
        default=True,
        title="Review Agent Enabled",
        description="Whether to allow the reviewer LLM agent to augment deterministic checks",
    )
    review_min_sources_per_task: int = Field(
        default=2,
        ge=1,
        title="Review Min Sources Per Task",
        description="Minimum evidence item count expected for a completed task",
    )
    review_min_domains_per_task: int = Field(
        default=2,
        ge=1,
        title="Review Min Domains Per Task",
        description="Minimum unique domain count expected for a completed task",
    )
    freshness_reference_days: int = Field(
        default=365,
        ge=1,
        title="Freshness Reference Days",
        description="Reference window used to classify evidence freshness",
    )
    task_query_rewrite_enabled: bool = Field(
        default=True,
        title="Task Query Rewrite Enabled",
        description="Whether to add a rewritten search query candidate for generic or underspecified task queries",
    )
    search_tool_timeout_seconds: float | None = Field(
        default=8.0,
        gt=0.0,
        title="Search Tool Timeout Seconds",
        description="Timeout guard applied to a single search tool invocation",
    )
    search_tool_retry_attempts: int = Field(
        default=1,
        ge=0,
        title="Search Tool Retry Attempts",
        description="How many retries to allow for a failed or timed-out search tool invocation",
    )
    search_tool_retry_backoff_seconds: float = Field(
        default=0.0,
        ge=0.0,
        title="Search Tool Retry Backoff Seconds",
        description="Fixed backoff between search tool retries",
    )
    direct_answer_char_limit: int | None = Field(
        default=None,
        title="Direct Answer Character Limit",
        description="Optional override for search backend direct answer character budget",
    )
    report_summary_char_limit: int | None = Field(
        default=None,
        title="Report Summary Character Limit",
        description="Optional override for task summary budget in the final report prompt",
    )
    report_sources_char_limit: int | None = Field(
        default=None,
        title="Report Sources Character Limit",
        description="Optional override for task source budget in the final report prompt",
    )
    metrics_recent_requests_limit: int = Field(
        default=25,
        title="Recent Request Limit",
        description="How many completed request traces to keep in memory",
    )
    request_state_enabled: bool = Field(
        default=True,
        title="Request State Enabled",
        description="Persist request snapshots to disk for history inspection and resume",
    )
    request_state_dir: str = Field(
        default="./.state/requests",
        title="Request State Directory",
        description="Directory used to persist request snapshots",
    )
    request_state_recent_limit: int = Field(
        default=50,
        ge=1,
        title="Request State Recent Limit",
        description="How many persisted request snapshots to surface via the API",
    )
    benchmark_stub_enabled: bool = Field(
        default=False,
        title="Benchmark Stub Enabled",
        description="Use the deterministic benchmark stub agent for perf runs",
    )
    benchmark_profile: str = Field(
        default="real_local",
        title="Benchmark Profile",
        description="Named benchmark profile used by perf tooling",
    )
    perf_sample_interval_seconds: float = Field(
        default=0.5,
        gt=0.0,
        title="Perf Sample Interval Seconds",
        description="Sampling interval used by performance profiling helpers",
    )
    perf_thresholds_path: str | None = Field(
        default=None,
        title="Perf Thresholds Path",
        description="Optional baseline or threshold file consumed by perf tooling",
    )
    llm_pricing_json: dict[str, Any] = Field(
        default_factory=dict,
        title="LLM Pricing Catalog",
        description="Static pricing map keyed by provider and model",
    )

    @field_validator("search_api", mode="before")
    @classmethod
    def _normalize_search_api(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("advanced_search_backends", mode="before")
    @classmethod
    def _normalize_advanced_search_backends(cls, value: Any) -> list[str]:
        default_backends = ["searxng", "tavily", "serpapi", "duckduckgo"]
        if value in (None, "", []):
            return default_backends

        if isinstance(value, str):
            raw_items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raise ValueError("advanced_search_backends must be a string or list of backends")

        supported = SearchAPI.fusion_backends()
        normalized: list[str] = []
        seen: set[str] = set()
        invalid: list[str] = []

        for item in raw_items:
            backend = str(item or "").strip().lower()
            if not backend or backend == SearchAPI.ADVANCED.value:
                continue
            if backend not in supported:
                invalid.append(backend)
                continue
            if backend in seen:
                continue
            seen.add(backend)
            normalized.append(backend)

        if invalid:
            supported_list = ", ".join(sorted(supported))
            invalid_list = ", ".join(sorted(set(invalid)))
            raise ValueError(
                f"Unsupported advanced_search_backends: {invalid_list}. "
                f"Supported backends: {supported_list}"
            )

        return normalized or default_backends

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper() or "INFO"
        return value

    @field_validator("benchmark_profile", mode="before")
    @classmethod
    def _normalize_benchmark_profile(cls, value: Any) -> Any:
        if value is None:
            return "real_local"

        if isinstance(value, str):
            normalized = value.strip().lower() or "real_local"
            if normalized not in {"stub", "real_local"}:
                raise ValueError("benchmark_profile must be either 'stub' or 'real_local'")
            return normalized

        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _normalize_cors_origins(cls, value: Any) -> Any:
        return _normalize_origin_values(value)

    @field_validator("llm_pricing_json", mode="before")
    @classmethod
    def _normalize_pricing_catalog(cls, value: Any) -> Any:
        if value in (None, "", {}):
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return value if isinstance(value, dict) else {}

    @model_validator(mode="after")
    def _normalize_paths(self) -> "Configuration":
        workspace = (self.notes_workspace or "").strip()
        if workspace:
            workspace_path = Path(workspace).expanduser()
            if not workspace_path.is_absolute():
                workspace_path = backend_root() / workspace_path
            self.notes_workspace = str(workspace_path.resolve(strict=False))

        cache_dir = (self.search_cache_dir or "").strip()
        if cache_dir:
            cache_path = Path(cache_dir).expanduser()
            if not cache_path.is_absolute():
                cache_path = backend_root() / cache_path
            self.search_cache_dir = str(cache_path.resolve(strict=False))

        thresholds_path = (self.perf_thresholds_path or "").strip()
        if thresholds_path:
            perf_path = Path(thresholds_path).expanduser()
            if not perf_path.is_absolute():
                perf_path = backend_root() / perf_path
            self.perf_thresholds_path = str(perf_path.resolve(strict=False))

        request_state_dir = (self.request_state_dir or "").strip()
        if request_state_dir:
            state_path = Path(request_state_dir).expanduser()
            if not state_path.is_absolute():
                state_path = backend_root() / state_path
            self.request_state_dir = str(state_path.resolve(strict=False))
        return self

    @classmethod
    def from_env(
        cls,
        overrides: dict[str, Any] | None = None,
        *,
        load_env_file: bool = True,
    ) -> "Configuration":
        """Create a configuration object using environment variables and overrides."""
        if load_env_file:
            _load_backend_env()

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        # Additional mappings for explicit env names
        env_aliases = {
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_context_window": os.getenv("LLM_CONTEXT_WINDOW"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "use_tool_calling": os.getenv("USE_TOOL_CALLING"),
            "search_api": os.getenv("SEARCH_API"),
            "advanced_search_backends": os.getenv("ADVANCED_SEARCH_BACKENDS"),
            "enable_notes": os.getenv("ENABLE_NOTES"),
            "notes_workspace": os.getenv("NOTES_WORKSPACE"),
            "host": os.getenv("HOST"),
            "port": os.getenv("PORT"),
            "log_level": os.getenv("LOG_LEVEL"),
            "cors_origins": os.getenv("CORS_ORIGINS"),
            "search_cache_enabled": os.getenv("SEARCH_CACHE_ENABLED"),
            "search_cache_ttl_seconds": os.getenv("SEARCH_CACHE_TTL_SECONDS"),
            "search_cache_dir": os.getenv("SEARCH_CACHE_DIR"),
            "semantic_cache_enabled": os.getenv("SEMANTIC_CACHE_ENABLED"),
            "semantic_cache_embedding_model": os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL"),
            "semantic_cache_similarity_threshold": os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD"),
            "semantic_cache_lexical_threshold": os.getenv("SEMANTIC_CACHE_LEXICAL_THRESHOLD"),
            "max_tokens_per_source": os.getenv("MAX_TOKENS_PER_SOURCE"),
            "task_context_char_limit": os.getenv("TASK_CONTEXT_CHAR_LIMIT"),
            "task_summary_max_concurrency": os.getenv("TASK_SUMMARY_MAX_CONCURRENCY"),
            "max_agent_tasks": os.getenv("MAX_AGENT_TASKS"),
            "request_reflection_enabled": os.getenv("REQUEST_REFLECTION_ENABLED"),
            "reflection_max_additional_tasks": os.getenv("REFLECTION_MAX_ADDITIONAL_TASKS"),
            "review_stage_enabled": os.getenv("REVIEW_STAGE_ENABLED"),
            "review_agent_enabled": os.getenv("REVIEW_AGENT_ENABLED"),
            "review_min_sources_per_task": os.getenv("REVIEW_MIN_SOURCES_PER_TASK"),
            "review_min_domains_per_task": os.getenv("REVIEW_MIN_DOMAINS_PER_TASK"),
            "freshness_reference_days": os.getenv("FRESHNESS_REFERENCE_DAYS"),
            "task_query_rewrite_enabled": os.getenv("TASK_QUERY_REWRITE_ENABLED"),
            "search_tool_timeout_seconds": os.getenv("SEARCH_TOOL_TIMEOUT_SECONDS"),
            "search_tool_retry_attempts": os.getenv("SEARCH_TOOL_RETRY_ATTEMPTS"),
            "search_tool_retry_backoff_seconds": os.getenv("SEARCH_TOOL_RETRY_BACKOFF_SECONDS"),
            "direct_answer_char_limit": os.getenv("DIRECT_ANSWER_CHAR_LIMIT"),
            "report_summary_char_limit": os.getenv("REPORT_SUMMARY_CHAR_LIMIT"),
            "report_sources_char_limit": os.getenv("REPORT_SOURCES_CHAR_LIMIT"),
            "metrics_recent_requests_limit": os.getenv("METRICS_RECENT_REQUESTS_LIMIT"),
            "request_state_enabled": os.getenv("REQUEST_STATE_ENABLED"),
            "request_state_dir": os.getenv("REQUEST_STATE_DIR"),
            "request_state_recent_limit": os.getenv("REQUEST_STATE_RECENT_LIMIT"),
            "benchmark_stub_enabled": os.getenv("BENCHMARK_STUB_ENABLED"),
            "benchmark_profile": os.getenv("BENCHMARK_PROFILE"),
            "perf_sample_interval_seconds": os.getenv("PERF_SAMPLE_INTERVAL_SECONDS"),
            "perf_thresholds_path": os.getenv("PERF_THRESHOLDS_PATH"),
            "llm_pricing_json": os.getenv("LLM_PRICING_JSON"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """Ensure Ollama base URL includes the /v1 suffix required by OpenAI clients."""
        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> str | None:
        """Best-effort resolution of the model identifier to use."""
        return self.llm_model_id or self.local_llm

    def resolved_context_window(self) -> int:
        """Return the configured model context window with a safe lower bound."""
        return max(2048, int(self.llm_context_window or 2048))

    def resolved_search_cache_dir(self) -> str:
        """Return the persistent cache directory."""
        return self.search_cache_dir

    def resolved_advanced_search_backends(self) -> list[str]:
        """Return the ordered concrete backends used for fused search."""
        return list(self.advanced_search_backends or ["searxng", "tavily", "serpapi", "duckduckgo"])

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))

    def resolved_max_tokens_per_source(self) -> int:
        """Resolve per-source token budget, scaling with model context by default."""
        if self.max_tokens_per_source is not None:
            return max(32, int(self.max_tokens_per_source))

        window = self.resolved_context_window()
        return self._clamp(window // 64, 120, 2048)

    def resolved_direct_answer_char_limit(self) -> int:
        """Resolve direct-answer character budget, scaling with model context by default."""
        if self.direct_answer_char_limit is not None:
            return max(100, int(self.direct_answer_char_limit))

        window = self.resolved_context_window()
        return self._clamp((window // 32) * 4, 300, 12000)

    def resolved_task_context_char_limit(self) -> int:
        """Resolve task summarization context budget, scaling with model context by default."""
        if self.task_context_char_limit is not None:
            return max(500, int(self.task_context_char_limit))

        source_chars = self.resolved_max_tokens_per_source() * 4 * 5
        answer_chars = self.resolved_direct_answer_char_limit()
        return source_chars + answer_chars + 2000

    def resolved_report_summary_char_limit(self) -> int:
        """Resolve per-task summary budget for the final report prompt."""
        if self.report_summary_char_limit is not None:
            return max(100, int(self.report_summary_char_limit))

        window = self.resolved_context_window()
        return self._clamp((window // 48) * 4, 120, 6000)

    def resolved_report_sources_char_limit(self) -> int:
        """Resolve per-task source budget for the final report prompt."""
        if self.report_sources_char_limit is not None:
            return max(80, int(self.report_sources_char_limit))

        window = self.resolved_context_window()
        return self._clamp((window // 96) * 4, 80, 3000)
