import json
import logging
import os
import re
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _BACKEND_ROOT / ".env"
_DOTENV_LOADED = False
_LOGGER = logging.getLogger(__name__)
_BOOLEAN_ADAPTER = TypeAdapter(bool)
_WEB_EXTRACTOR_ALIAS = "v4_web_extraction_enabled"
_WEB_L2_LIMITS: dict[str, tuple[type[int] | type[float], int]] = {
    "v4_web_l2_max_concurrency": (int, 2),
    "v4_web_l2_per_tenant": (int, 1),
    "v4_web_l2_timeout_seconds": (float, 12),
}
_WEB_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/\-]+={0,2}")
WebConfigReason = Literal[
    "invalid_web_extractor_enabled",
    "web_extractor_alias_conflict",
    "invalid_web_extractor_tenants",
    "invalid_web_l2_adapter",
    "invalid_browser_use_enabled",
    "invalid_web_l2_origin_rules",
    "invalid_web_reader_url",
    "invalid_web_reader_token",
    "invalid_web_l2_limits",
]


def _web_bool(value: Any) -> bool:
    if type(value) not in (bool, str):
        raise ValueError("invalid web extraction switch")
    return _BOOLEAN_ADAPTER.validate_python(value.strip() if isinstance(value, str) else value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate web extraction configuration key")
    return result


def _web_structured(value: Any, expected: type[list] | type[dict]) -> Any:
    if isinstance(value, str):
        value = json.loads(value, object_pairs_hook=_unique_json_object)
    if not isinstance(value, expected):
        raise ValueError("invalid web extraction configuration shape")
    return value


def _web_tenants(value: Any) -> list[str]:
    tenants = _web_structured(value, list)
    if any(
        not isinstance(tenant, str)
        or not tenant
        or tenant != tenant.strip()
        or not tenant.isprintable()
        or "*" in tenant
        for tenant in tenants
    ):
        raise ValueError("invalid web extraction tenant")
    return list(dict.fromkeys(tenants))


def _web_limit(value: Any, numeric_type: type[int] | type[float], maximum: int) -> int | float:
    allowed_types = (str, int) if numeric_type is int else (str, int, float)
    if type(value) not in allowed_types:
        raise ValueError("invalid web extraction limit type")
    normalized = numeric_type(value)
    if not 0 < normalized <= maximum:
        raise ValueError("web extraction limit is outside its bounds")
    return normalized


def validate_reader_token(value: str) -> str:
    """Validate deployment bearer credentials before they reach HTTP headers."""
    if not isinstance(value, str) or not _WEB_BEARER_TOKEN.fullmatch(value):
        raise ValueError("invalid reader bearer token")
    return value


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
    SEMANTICSCHOLAR = "semanticscholar"
    ADVANCED = "advanced"

    @classmethod
    def fusion_backends(cls) -> set[str]:
        """Return the concrete search backends allowed in fusion mode."""
        return {
            cls.PERPLEXITY.value,
            cls.TAVILY.value,
            cls.SERPAPI.value,
            cls.DUCKDUCKGO.value,
            cls.SEARXNG.value,
            cls.SEMANTICSCHOLAR.value,
        }


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
    advanced_search_max_concurrency: int = Field(
        default=4,
        ge=1,
        title="Advanced Search Max Concurrency",
        description="Maximum number of advanced search backends to execute in parallel",
    )
    advanced_search_fetch_full_page_override: bool | None = Field(
        default=None,
        title="Advanced Search Fetch Full Page Override",
        description="Optional override for fetch_full_page applied only to advanced search fan-out requests",
    )
    advanced_backend_timeout_seconds: float | None = Field(
        default=None,
        gt=0.0,
        title="Advanced Backend Timeout Seconds",
        description="Optional deadline for a single advanced search fusion round before slow backends are skipped",
    )
    advanced_rerank_enabled: bool = Field(
        default=False,
        title="Advanced Rerank Enabled",
        description="Whether to rerank fused advanced search candidates using an OpenAI-compatible LLM",
    )
    advanced_rerank_base_url: str | None = Field(
        default=None,
        title="Advanced Rerank Base URL",
        description="Optional OpenAI-compatible base URL used for advanced search reranking",
    )
    advanced_rerank_api_key: str | None = Field(
        default=None,
        title="Advanced Rerank API Key",
        description="Optional API key used for advanced search reranking requests",
    )
    advanced_rerank_model: str | None = Field(
        default=None,
        title="Advanced Rerank Model",
        description="Optional model identifier used for advanced search reranking",
    )
    advanced_rerank_candidate_pool: int = Field(
        default=20,
        ge=1,
        title="Advanced Rerank Candidate Pool",
        description="How many fused advanced results are eligible for reranking before final truncation",
    )
    advanced_rerank_timeout_seconds: float = Field(
        default=3.0,
        gt=0.0,
        title="Advanced Rerank Timeout Seconds",
        description="Timeout for a single advanced search reranking request",
    )
    advanced_rerank_max_content_chars: int = Field(
        default=1200,
        ge=100,
        title="Advanced Rerank Max Content Characters",
        description="Maximum document content characters included for each rerank candidate",
    )
    semantic_scholar_api_key: str | None = Field(
        default=None,
        title="Semantic Scholar API Key",
        description="Optional API key used for the Semantic Scholar Academic Graph API",
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
    memory_shadow_mode: bool = Field(
        default=False,
        title="Memory Shadow Mode",
        description="When enabled, memory retrieval still runs and is traced but results are never injected into prompts",
    )
    memory_maturity_gate: str = Field(
        default="draft",
        title="Memory Maturity Gate",
        description="Lowest strategy card maturity still eligible for retrieval (draft | verified | promoted)",
    )
    memory_decay_idle_days: int = Field(
        default=0,
        ge=0,
        title="Memory Decay Idle Days",
        description="Archive draft strategy cards unused for this many days; 0 disables decay",
    )
    skill_pack_enabled: bool = Field(
        default=False,
        title="Skill Pack Enabled",
        description="Whether to select a research skill pack and inject its stage fragments into prompts",
    )
    skill_pack_dir: str = Field(
        default="./skills/research",
        title="Skill Pack Directory",
        description="Directory holding research skill packs; each subdirectory is one pack with a SKILL.md",
    )
    skill_prompt_char_limit: int = Field(
        default=1200,
        ge=200,
        title="Skill Prompt Character Limit",
        description="Maximum characters allowed for a single injected skill pack stage fragment",
    )
    skill_pack_min_match_score: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        title="Skill Pack Minimum Match Score",
        description="Global floor for pack selection; below this the neutral fallback pack is used",
    )
    skill_format_dir: str = Field(
        default="./skills/formats",
        title="Skill Format Directory",
        description="Directory holding orthogonal output format skill packs",
    )
    skill_pack_composable_enabled: bool = Field(
        default=True,
        title="Skill Pack Composable Enabled",
        description="Allow composing 1 primary domain pack with 0..1 secondary constraint pack",
    )
    skill_pack_secondary_min_match_score: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        title="Skill Pack Secondary Minimum Match Score",
        description="Minimum positive match score to attach a composable secondary constraint skill",
    )
    skill_pack_semantic_routing_enabled: bool = Field(
        default=False,
        title="Skill Pack Semantic Routing Enabled",
        description="Enable embedding similarity fallback when lexical skill routing is below threshold",
    )
    skill_pack_llm_router_enabled: bool = Field(
        default=False,
        title="Skill Pack LLM Router Enabled",
        description="Enable lightweight LLM tie-breaker routing when candidate skill packs tie closely",
    )
    skill_pack_maturity_gate: str = Field(
        default="draft",
        title="Skill Pack Maturity Gate",
        description="Minimum maturity tier allowed when loading SkillPacks (draft, verified, promoted/production)",
    )
    skill_distill_min_support_cases: int = Field(
        default=3,
        ge=1,
        title="Skill Distillation Minimum Support Cases",
        description="Minimum benchmark failure/win cases supporting a distilled skill candidate",
    )
    skill_distill_min_hybrid_gain: float = Field(
        default=0.03,
        ge=0.0,
        title="Skill Distillation Minimum Hybrid Gain",
        description="Minimum normalized quality gain required to promote a distilled skill candidate",
    )
    skill_distill_max_adversarial_regression: float = Field(
        default=0.02,
        ge=0.0,
        title="Skill Distillation Max Adversarial Regression",
        description="Maximum allowed hybrid_score drop on adv_* regression cases",
    )
    skill_distill_min_top_traces: int = Field(default=10, ge=1)
    skill_distill_min_fail_traces: int = Field(default=5, ge=1)
    skill_distill_max_latency_regression_ratio: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    skill_distill_max_cost_regression_ratio: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    default_output_format: str | None = Field(
        default=None,
        title="Default Output Format Skill",
        description="Optional default output format skill id (e.g., exec-brief, engineering-rfc, deep-whitepaper)",
    )
    active_retrieval_profile: dict[str, Any] = Field(
        default_factory=dict,
        title="Active Retrieval Profile",
        description="Request-scoped retrieval overrides applied by the active skill pack",
    )
    active_skill_verifiers: list[str] = Field(
        default_factory=list,
        title="Active Skill Verifiers",
        description="Request-scoped deterministic verifier ids enabled by active skill packs",
    )
    active_output_format_id: str | None = Field(
        default=None,
        title="Active Output Format Skill ID",
        description="Request-scoped output format skill id bound during planning/reporting",
    )
    kv_cache_prefix_optimization_enabled: bool = Field(
        default=True,
        title="KV-Cache Prefix Optimization Enabled",
        description="Whether to use 3-segment stable prefix ordering across research prompts",
    )
    kv_cache_provider_mode: str = Field(
        default="auto",
        title="KV-Cache Provider Mode",
        description="Prefix caching payload mode: auto | openai | anthropic | vllm | disabled",
    )
    kv_cache_anthropic_breakpoints_enabled: bool = Field(
        default=True,
        title="KV-Cache Anthropic Breakpoints Enabled",
        description="Whether to emit cache_control ephemeral blocks for Anthropic-compatible backends",
    )
    kv_cache_min_segment_tokens: int = Field(
        default=1024,
        ge=0,
        title="KV-Cache Minimum Segment Tokens",
        description="Minimum estimated prefix tokens before emitting explicit cache_control breakpoints",
    )
    kv_cache_prompt_cache_key_enabled: bool = Field(
        default=True,
        title="KV-Cache Prompt Cache Key Enabled",
        description="Whether to emit tenant-isolated prompt_cache_key for OpenAI/vLLM requests",
    )
    kv_cache_block_alignment_tokens: int = Field(
        default=16,
        ge=1,
        title="KV-Cache Block Alignment Tokens",
        description="PagedAttention block size hint used when aligning prefix boundaries",
    )
    context_compaction_enabled: bool = Field(
        default=True,
        title="Context Compaction Enabled",
        description="Whether to compact cross-wave and downstream task prose into FactLedger",
    )
    fact_ledger_llm_compaction_enabled: bool = Field(
        default=False,
        title="Fact Ledger LLM Compaction Enabled",
        description="Whether the task summarizer may include structured ledger fields in its existing response",
    )
    fact_ledger_max_entries: int = Field(
        default=40,
        ge=4,
        title="Fact Ledger Max Entries",
        description="Maximum FactEntry items rendered into frozen downstream checkpoints",
    )
    evidence_pointer_disclosure_enabled: bool = Field(
        default=True,
        title="Evidence Pointer Disclosure Enabled",
        description="Whether to use EvidenceIndexCard pointers and lazy source_id chunk loading",
    )
    evidence_lazy_load_max_chunks_per_source: int = Field(
        default=2,
        ge=1,
        title="Evidence Lazy Load Max Chunks Per Source",
        description="Maximum chunks hydrated per source_id during lazy retrieval or pre-hydration",
    )
    evidence_lazy_load_char_budget: int = Field(
        default=1800,
        ge=200,
        title="Evidence Lazy Load Char Budget",
        description="Character ceiling for lazy chunk hydration per call",
    )
    mcp_enabled: bool = Field(
        default=False,
        title="MCP Enabled",
        description="Master flag enabling dynamic MCP server integration",
    )
    mcp_config_path: str = Field(
        default="./config/mcp_servers.yaml",
        title="MCP Servers Config Path",
        description="Path to YAML registry defining external MCP servers",
    )
    mcp_enforce_readonly: bool = Field(
        default=True,
        title="MCP Enforce Readonly",
        description="Fail-Closed gate blocking MCP tools lacking readOnlyHint=True",
    )
    mcp_allow_hitl_write_tools: bool = Field(
        default=False,
        title="MCP Allow HITL Write Tools",
        description="When True, whitelisted write tools trigger HITL approval instead of flat rejection",
    )
    mcp_allowed_private_hosts: list[str] = Field(
        default_factory=list,
        title="MCP Allowed Private Hosts",
        description="Whitelist of internal sidecar hostnames or CIDRs allowed past SSRF check",
    )
    mcp_approval_timeout_seconds: float = Field(default=86400, gt=0, allow_inf_nan=False)
    mcp_allow_insecure_http: bool = Field(
        default=False,
        title="MCP Allow Insecure HTTP",
        description="Allow http:// MCP endpoints for controlled local/sidecar deployments",
    )
    mcp_tool_timeout_seconds: float = Field(
        default=10.0,
        gt=0.1,
        title="MCP Tool Timeout Seconds",
        description="Timeout ceiling for a single MCP tool invocation",
    )
    mcp_tool_cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        title="MCP Tool Cache TTL Seconds",
        description="TTL for cached tools/list discovery metadata per MCP server",
    )
    mcp_max_tools_per_stage: int = Field(
        default=6,
        ge=0,
        title="MCP Max Tools Per Stage",
        description="Anti-Tool-Bloat hard limit on MCP tools exposed to one stage prompt",
    )
    mcp_tenant_token_secret: str | None = Field(
        default=None,
        title="MCP Tenant Token Secret",
        description="Optional HMAC-SHA256 secret used to sign scoped tenant tokens across MCP boundary",
    )
    active_mcp_tool_scopes: dict[str, Any] = Field(
        default_factory=dict,
        title="Active MCP Tool Scopes",
        description="Request-scoped MCP tool whitelist bound from active SkillPack",
    )
    multimodal_evidence_enabled: bool = Field(
        default=False,
        title="Multimodal Evidence Enabled",
        description="Whether to parse HTML tables/chart candidates and L0/L1/L2 PDF pages into structured multimodal blocks",
    )
    multimodal_vlm_enabled: bool = Field(
        default=False,
        title="Multimodal VLM Enabled",
        description="Whether to invoke a Vision-Language Model on L2 visual PDF pages and chart images",
    )
    multimodal_vlm_model: str | None = Field(
        default=None,
        title="Multimodal VLM Model",
        description="Vision-Language model identifier for chart/table OCR extraction",
    )
    multimodal_vlm_base_url: str | None = Field(
        default=None,
        title="Multimodal VLM Base URL",
        description="Optional OpenAI-compatible base URL override for the VLM endpoint",
    )
    multimodal_vlm_api_key: str | None = Field(
        default=None,
        title="Multimodal VLM API Key",
        description="Optional API key override for the VLM endpoint",
    )
    multimodal_vlm_timeout_seconds: float = Field(
        default=12.0,
        ge=1.0,
        title="Multimodal VLM Timeout Seconds",
        description="HTTP timeout in seconds per VLM chart extraction request",
    )
    multimodal_vlm_max_concurrency: int = Field(
        default=4,
        ge=1,
        title="Multimodal VLM Max Concurrency",
        description="Semaphore bound on concurrent VLM calls",
    )
    multimodal_max_pdf_pages: int = Field(
        default=20,
        ge=1,
        title="Multimodal Max PDF Pages",
        description="Maximum number of pages triaged per PDF document",
    )
    multimodal_max_pdf_bytes: int = Field(default=15 * 1024 * 1024, ge=1)
    multimodal_pdf_render_dpi: int = Field(default=144, ge=72, le=200)
    multimodal_max_vlm_pages_per_doc: int = Field(
        default=6,
        ge=0,
        title="Multimodal Max VLM Pages Per Doc",
        description="Maximum number of L2 PDF pages routed to VLM per document",
    )
    multimodal_max_images_per_page: int = Field(
        default=2,
        ge=0,
        title="Multimodal Max Images Per Page",
        description="Maximum chart image candidates fetched per HTML page",
    )
    multimodal_image_max_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1024,
        title="Multimodal Image Max Bytes",
        description="Hard byte cap per downloaded or rendered chart image",
    )
    multimodal_image_max_pixels: int = Field(
        default=16_000_000,
        ge=1024,
        title="Multimodal Image Max Pixels",
        description="Hard width*height pixel ceiling against decompression bombs",
    )
    sandbox_enabled: bool = Field(
        default=False,
        title="Code Sandbox Enabled",
        description="Whether isolated Python quantitative sandbox execution is provisioned",
    )
    sandbox_backend: str = Field(
        default="auto",
        title="Code Sandbox Backend",
        description="Sandbox backend selection: auto, docker, or pyodide",
    )
    sandbox_docker_image: str = Field(
        default="helloagents-sandbox:latest",
        title="Code Sandbox Docker Image",
        description="Hardened non-root container image for Python quantitative sandbox execution",
    )
    sandbox_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        title="Code Sandbox Timeout Seconds",
        description="Wall-clock timeout for a single sandbox execution",
    )
    sandbox_memory_mb: int = Field(
        default=512,
        ge=64,
        title="Code Sandbox Memory MB",
        description="Memory cap in MB enforced on the sandbox container or Pyodide worker",
    )
    sandbox_cpus: str = Field(
        default="1.0",
        title="Code Sandbox CPUs",
        description="CPU quota flag passed to Docker sandbox runtime",
    )
    sandbox_max_executions_per_task: int = Field(
        default=2,
        ge=1,
        title="Code Sandbox Max Executions Per Task",
        description="Maximum sandbox executions permitted per research task",
    )
    sandbox_pyodide_cache_dir: str = Field(
        default="./.cache/pyodide",
        title="Code Sandbox Pyodide Cache Dir",
        description="Local Pyodide distribution directory for offline WASM fallback execution",
    )
    sandbox_pyodide_node_bin: str = "node"
    active_code_sandbox_enabled: bool | None = Field(
        default=None,
        title="Active Code Sandbox Override",
        description="Request-scoped sandbox activation state bound from active SkillPack",
    )
    nli_verifier_enabled: bool = Field(
        default=False,
        title="NLI Claim Verifier Enabled",
        description="Whether to run atomic claim decomposition, local NLI verification, and conflict graph checks",
    )
    nli_model_backend: str = Field(
        default="auto",
        title="NLI Model Backend",
        description="NLI inference backend: auto, cross_encoder, transformers_nli, minicheck, or disabled",
    )
    nli_model_name: str = Field(
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        title="NLI Model Name",
        description="HuggingFace/local model id used by LocalNLIScorer",
    )
    nli_device: str = Field(
        default="auto",
        title="NLI Device",
        description="Execution device for NLI model inference: auto, cpu, cuda, or mps",
    )
    nli_warmup_enabled: bool = Field(
        default=False,
        title="NLI Warmup Enabled",
        description="Whether FastAPI lifespan warms up local cached NLI model on startup",
    )
    nli_decompose_with_fast_llm: bool = Field(
        default=True,
        title="NLI Decompose With Fast LLM",
        description="Whether compound claim decomposition may use fast_llm before deterministic splitter fallback",
    )
    nli_batch_size: int = Field(
        default=16,
        ge=1,
        le=64,
        title="NLI Batch Size",
        description="Pairs per NLI inference batch",
    )
    nli_max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        title="NLI Max Sequence Length",
        description="Maximum tokenized sequence length per premise-hypothesis pair",
    )
    nli_max_atomic_claims_per_task: int = Field(
        default=12,
        ge=1,
        title="NLI Max Atomic Claims Per Task",
        description="Maximum atomic claims verified per research task",
    )
    nli_top_k_chunks_per_source: int = Field(
        default=3,
        ge=1,
        le=8,
        title="NLI Top K Chunks Per Source",
        description="Top-K chunks recalled per cited source_id for atomic verification",
    )
    nli_entailment_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        title="NLI Entailment Threshold",
        description="Calibrated probability threshold to classify an atomic claim as ENTAILED",
    )
    nli_contradiction_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        title="NLI Contradiction Threshold",
        description="Calibrated probability threshold to flag an atomic claim as CONTRADICTED",
    )
    nli_stage_budget_seconds: float = Field(
        default=8.0,
        gt=0.0,
        allow_inf_nan=False,
        title="NLI Stage Budget Seconds",
        description="Wall-clock budget for NLI verification before degrading gracefully to numeric checks",
    )
    nli_conflict_graph_enabled: bool = Field(
        default=True,
        title="NLI Conflict Graph Enabled",
        description="Whether to cluster cross-source conflicting statements into state.evidence_conflicts",
    )
    nli_conflict_numeric_rel_tol: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        title="NLI Conflict Numeric Relative Tolerance",
        description="Relative numeric divergence threshold above which sources form a conflict edge",
    )
    nli_conflict_min_score: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        title="NLI Conflict Min Score",
        description="Minimum cross-NLI contradiction probability to form a semantic conflict edge",
    )
    note_memory_enabled: bool = Field(
        default=False,
        title="Note Memory Enabled",
        description="Whether to build a persistent vector note memory for historical retrieval",
    )
    note_memory_dir: str = Field(
        default="./.memory/notes",
        title="Note Memory Directory",
        description="Directory used by the local note memory vector store and manifest",
    )
    note_memory_embedding_model: str | None = Field(
        default=None,
        title="Note Memory Embedding Model",
        description="Optional embedding model override for note memory retrieval",
    )
    note_memory_planning_top_k: int = Field(
        default=3,
        ge=1,
        title="Note Memory Planning Top K",
        description="How many historical note memories to inject during planning",
    )
    note_memory_execution_top_k: int = Field(
        default=3,
        ge=1,
        title="Note Memory Execution Top K",
        description="How many historical note memories to inject during task execution",
    )
    note_memory_prompt_char_limit: int = Field(
        default=1800,
        ge=200,
        title="Note Memory Prompt Character Limit",
        description="Maximum prompt characters reserved for injected historical note memory",
    )
    strategy_memory_enabled: bool = Field(
        default=False,
        title="Strategy Memory Enabled",
        description="Whether to build a persistent cross-request strategy memory from terminal request snapshots",
    )
    strategy_memory_dir: str = Field(
        default="./.memory/strategies",
        title="Strategy Memory Directory",
        description="Directory used by the local strategy memory vector store, cards, and manifest",
    )
    strategy_memory_embedding_model: str | None = Field(
        default=None,
        title="Strategy Memory Embedding Model",
        description="Optional embedding model override for strategy memory retrieval",
    )
    strategy_memory_planning_top_k: int = Field(
        default=3,
        ge=1,
        title="Strategy Memory Planning Top K",
        description="How many historical strategy memories to inject during planning",
    )
    strategy_memory_reflection_top_k: int = Field(
        default=3,
        ge=1,
        title="Strategy Memory Reflection Top K",
        description="How many historical strategy memories to inject during reflection",
    )
    strategy_memory_prompt_char_limit: int = Field(
        default=1600,
        ge=200,
        title="Strategy Memory Prompt Character Limit",
        description="Maximum prompt characters reserved for injected historical strategy memory",
    )
    evidence_runtime_enabled: bool = Field(
        default=True,
        title="Evidence Runtime Enabled",
        description="Whether to build a request-scoped runtime chunk index for grounding",
    )
    evidence_runtime_top_k: int = Field(
        default=8,
        ge=1,
        title="Evidence Runtime Top K",
        description="How many runtime evidence chunks to surface for task grounding",
    )
    evidence_chunk_chars: int = Field(
        default=900,
        ge=200,
        title="Evidence Chunk Characters",
        description="Chunk size used by runtime and persistent evidence indexing",
    )
    evidence_chunk_overlap: int = Field(
        default=120,
        ge=0,
        title="Evidence Chunk Overlap",
        description="Chunk overlap used by runtime and persistent evidence indexing",
    )
    evidence_runtime_max_chunks: int = Field(
        default=32,
        ge=1,
        title="Evidence Runtime Max Chunks",
        description="Maximum number of chunks indexed per evidence document",
    )
    evidence_runtime_embedding_model: str | None = Field(
        default=None,
        title="Evidence Runtime Embedding Model",
        description="Optional embedding model override for runtime evidence retrieval",
    )
    evidence_memory_enabled: bool = Field(
        default=False,
        title="Evidence Memory Enabled",
        description="Whether to persist cross-request evidence chunks and metadata",
    )
    evidence_memory_dir: str = Field(
        default="./.memory/evidence",
        title="Evidence Memory Directory",
        description="Directory used by the persistent evidence archive and vector store",
    )
    evidence_memory_top_k: int = Field(
        default=6,
        ge=1,
        title="Evidence Memory Top K",
        description="How many historical evidence chunks to retrieve for lead/repair flows",
    )
    evidence_memory_embedding_model: str | None = Field(
        default=None,
        title="Evidence Memory Embedding Model",
        description="Optional embedding model override for persistent evidence retrieval",
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
    fast_llm_provider: str | None = Field(
        default=None,
        title="Fast LLM Provider",
        description="Optional lightweight/fast LLM provider for planning, summarization, and reflection",
    )
    fast_llm_model_id: str | None = Field(
        default=None,
        title="Fast LLM Model ID",
        description="Optional fast-tier model identifier; when unset, primary LLM is used for all stages",
    )
    fast_llm_base_url: str | None = Field(
        default=None,
        title="Fast LLM Base URL",
        description="Optional base URL for the fast-tier LLM endpoint",
    )
    fast_llm_api_key: str | None = Field(
        default=None,
        title="Fast LLM API Key",
        description="Optional API key for the fast-tier LLM endpoint",
    )
    fast_llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        title="Fast LLM Temperature",
        description="Sampling temperature for fast-tier LLM calls",
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
        default_factory=lambda: ["http://localhost:5174", "http://127.0.0.1:5174"],
        title="CORS Origins",
        description="Allowed CORS origins for the HTTP API",
    )
    sse_heartbeat_interval_seconds: float = Field(
        default=15.0,
        gt=0.0,
        title="SSE Heartbeat Interval Seconds",
        description="Keepalive comment interval emitted on idle SSE streams",
    )
    async_memory_refresh_enabled: bool = Field(
        default=False,
        title="Async Memory Refresh Enabled",
        description="When enabled, run post-report strategy memory synthesis in a background thread",
    )
    langfuse_enabled: bool = Field(
        default=False,
        title="Langfuse Observability Enabled",
        description="Opt-in export of request traces and stage spans to Langfuse",
    )
    langfuse_public_key: str | None = Field(
        default=None,
        title="Langfuse Public Key",
        description="Optional Langfuse public key",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        title="Langfuse Secret Key",
        description="Optional Langfuse secret key",
    )
    langfuse_host: str | None = Field(
        default=None,
        title="Langfuse Host",
        description="Optional Langfuse host URL",
    )
    auth_enabled: bool = Field(
        default=False,
        title="Multi-Tenant Auth Enabled",
        description="Opt-in API key authentication for multi-tenant HTTP access",
    )
    tenant_api_keys: dict[str, str] = Field(
        default_factory=dict,
        title="Tenant API Keys",
        description="Mapping of API key to tenant_id (JSON object via TENANT_API_KEYS)",
    )
    default_tenant_id: str = Field(
        default="default",
        title="Default Tenant ID",
        description="Fallback tenant namespace used when multi-tenant auth is disabled",
    )
    tenant_rate_limit_rpm: int = Field(
        default=0,
        ge=0,
        title="Tenant Rate Limit RPM",
        description="Optional per-tenant requests-per-minute limit (0 disables rate limiting)",
    )
    tenant_max_concurrent_requests: int = Field(
        default=0,
        ge=0,
        title="Tenant Max Concurrent Requests",
        description="Optional per-tenant in-flight request limit (0 disables concurrency cap)",
    )
    redis_url: str | None = Field(
        default=None,
        title="Redis URL",
        description="Optional Redis URL for Redis Streams SSE event replay",
    )
    database_url: str | None = Field(
        default=None,
        title="Database URL",
        description="Optional PostgreSQL DSN for persistent request snapshot storage",
    )
    event_bus_max_events_per_request: int = Field(
        default=2000,
        ge=50,
        title="Event Bus Max Events Per Request",
        description="Maximum buffered SSE events retained per request for Last-Event-ID replay",
    )
    event_bus_stream_ttl_seconds: int = Field(
        default=86400,
        ge=60,
        title="Event Bus Stream TTL Seconds",
        description="Retention TTL in seconds applied to finished Redis Streams",
    )
    workflow_runner: str = Field(
        default="inline",
        title="Workflow Runner Backend",
        description="Execution backend: inline, local_durable, arq, or temporal",
    )
    workflow_runner_fallback_inline: bool = False
    temporal_activity_heartbeat_seconds: float = Field(default=15.0, gt=1.0)
    hitl_signal_timeout_seconds: int = Field(default=86400, ge=1)
    temporal_address: str = Field(
        default="127.0.0.1:7233",
        title="Temporal Server Address",
        description="Host:port of the Temporal gRPC frontend",
    )
    temporal_namespace: str = Field(
        default="default",
        title="Temporal Namespace",
        description="Temporal namespace for workflow executions",
    )
    temporal_task_queue: str = Field(
        default="deepresearch-tasks",
        title="Temporal Task Queue",
        description="Temporal task queue subscribed by DeepResearch workers",
    )
    arq_redis_dsn: str | None = Field(
        default=None,
        title="ARQ Redis DSN",
        description="Optional Redis DSN override used by ARQ durable queue runner",
    )
    postgres_pool_min_size: int = Field(
        default=2,
        ge=1,
        title="Postgres Pool Min Size",
        description="Minimum connections kept in optional psycopg_pool connection pool",
    )
    postgres_pool_max_size: int = Field(
        default=10,
        ge=1,
        title="Postgres Pool Max Size",
        description="Maximum connections allowed in optional psycopg_pool connection pool",
    )
    policy_model_enabled: bool = Field(
        default=False,
        title="Policy Model Enabled",
        description="Enable pluggable small-model policy controller during ReAct routing",
    )
    policy_model_id: str | None = Field(
        default=None,
        title="Policy Model ID",
        description="Model identifier served by the OpenAI-compatible policy endpoint",
    )
    policy_model_base_url: str | None = Field(
        default=None,
        title="Policy Model Base URL",
        description="OpenAI-compatible base URL for policy controller inference",
    )
    policy_model_api_key: str | None = Field(
        default=None,
        title="Policy Model API Key",
        description="Optional API key for the policy controller endpoint",
    )
    policy_model_timeout_seconds: float = Field(
        default=0.8,
        ge=0.01,
        allow_inf_nan=False,
        title="Policy Model Timeout Seconds",
        description="Request timeout in seconds for external policy model calls, capped at five seconds",
    )
    policy_model_mode: str = Field(
        default="off",
        title="Policy Model Mode",
        description="Policy rollout mode: off, shadow, canary, or active",
    )
    policy_model_canary_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Policy Model Canary Ratio",
        description="Deterministic traffic fraction routed to learned policy in canary mode",
    )
    policy_model_canary_tenants: tuple[str, ...] = Field(
        default=(),
        title="Policy Model Canary Tenants",
        description="Explicit tenant IDs always enrolled in canary policy execution",
    )
    policy_model_prompt_version: str = Field(
        default="policy_prompt_v2",
        title="Policy Model Prompt Version",
        description="Prompt format template version used by both GRPO exporter and online policy controller",
    )
    policy_model_circuit_breaker_window: int = Field(
        default=50,
        ge=1,
        le=10000,
        title="Policy Model Circuit Breaker Window",
        description="Sliding window size for policy fallback-rate circuit breaker",
    )
    policy_model_circuit_breaker_fallback_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        title="Policy Model Circuit Breaker Fallback Rate",
        description="Fallback ratio threshold that opens the policy circuit breaker",
    )
    policy_model_max_p95_ms: float = Field(
        default=800.0,
        ge=1.0,
        allow_inf_nan=False,
        title="Policy Model Max P95 Ms",
        description="Latency SLO budget in milliseconds for online policy calls",
    )
    policy_model_consecutive_timeout_limit: int = Field(default=5, ge=1)
    search_cache_enabled: bool = Field(
        default=True,
        title="Search Cache Enabled",
        description="Whether to cache search results in-process",
    )
    search_cache_ttl_seconds: int = Field(
        default=43200,
        title="Search Cache TTL",
        description="How long to keep in-process search cache entries",
    )
    search_cache_dynamic_ttl_enabled: bool = Field(
        default=True,
        title="Search Cache Dynamic TTL Enabled",
        description="Whether to assign cache TTL dynamically based on query freshness signals",
    )
    search_cache_fresh_ttl_seconds: int = Field(
        default=14400,
        title="Search Cache Fresh TTL",
        description="TTL for freshness-sensitive search cache entries",
    )
    search_cache_evergreen_ttl_seconds: int = Field(
        default=172800,
        title="Search Cache Evergreen TTL",
        description="TTL for evergreen search cache entries",
    )
    search_cache_dir: str = Field(
        default="./.cache/search",
        title="Search Cache Directory",
        description="Directory used by the persistent search cache store",
    )
    search_cache_vector_dir: str = Field(
        default="./.cache/search/vector",
        title="Search Cache Vector Directory",
        description="Directory used by the dense ANN vector index for approximate cache recall",
    )
    semantic_cache_enabled: bool = Field(
        default=True,
        title="Semantic Cache Enabled",
        description="Legacy alias for approximate cache enablement when APPROXIMATE_CACHE_ENABLED is unset",
    )
    semantic_cache_warmup_enabled: bool = Field(
        default=True,
        title="Semantic Cache Warmup Enabled",
        description="Whether to warm up the approximate-cache embedding model during API startup",
    )
    semantic_cache_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        title="Semantic Cache Embedding Model",
        description="Embedding model used to vectorize search queries for dense approximate cache matching",
    )
    semantic_cache_similarity_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        title="Semantic Cache Similarity Threshold",
        description="Legacy alias for the dense approximate cache threshold",
    )
    semantic_cache_lexical_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        title="Semantic Cache Lexical Threshold",
        description="Legacy alias for the sparse approximate cache threshold",
    )
    approximate_cache_enabled: bool | None = Field(
        default=None,
        title="Approximate Cache Enabled",
        description="Whether to reuse similar search results through dense and sparse approximate signals",
    )
    approximate_cache_dense_enabled: bool | None = Field(
        default=None,
        title="Approximate Cache Dense Enabled",
        description="Whether approximate cache uses embedding/ANN dense recall",
    )
    approximate_cache_sparse_enabled: bool | None = Field(
        default=None,
        title="Approximate Cache Sparse Enabled",
        description="Whether approximate cache uses lexical sparse recall",
    )
    approximate_cache_dense_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Approximate Cache Dense Threshold",
        description="Cosine similarity threshold for dense approximate cache hits",
    )
    approximate_cache_sparse_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Approximate Cache Sparse Threshold",
        description="Lexical similarity threshold for sparse approximate cache hits",
    )
    approximate_cache_dense_top_k: int = Field(
        default=12,
        ge=1,
        title="Approximate Cache Dense Top K",
        description="How many ANN candidates to recall for approximate cache scoring",
    )
    approximate_cache_sparse_top_k: int = Field(
        default=12,
        ge=1,
        title="Approximate Cache Sparse Top K",
        description="How many lexical candidates to recall for approximate cache scoring",
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
    initial_plan_max_tasks: int | None = Field(
        default=None,
        ge=1,
        title="Initial Plan Max Tasks",
        description="Optional cap for the initial planning stage so reflection/repair reserves remain available",
    )
    reflection_reserve_tasks: int = Field(
        default=2,
        ge=0,
        title="Reflection Reserve Tasks",
        description="Reserved task headroom for request-level reflection when multi-stage budgets are active",
    )
    repair_reserve_tasks: int = Field(
        default=2,
        ge=0,
        title="Repair Reserve Tasks",
        description="Reserved task headroom for review-driven report repair when multi-stage budgets are active",
    )
    max_total_tasks: int | None = Field(
        default=None,
        ge=1,
        title="Max Total Tasks",
        description="Optional hard total task ceiling across planning, reflection, and report repair",
    )
    v4_research_budget_enabled: bool = Field(
        default=False,
        title="V4 Research Budget Enabled",
        description="Enable the v4 request budget controller",
    )
    v4_replan_enabled: bool = Field(
        default=False,
        title="V4 Replan Enabled",
        description="Enable the v4 Wave 1 research tree replanner",
    )
    v4_web_extractor_enabled: bool = False
    v4_web_extractor_tenants: list[str] = Field(default_factory=list, repr=False, exclude=True)
    v4_web_l2_adapter: Literal["off", "reader_http", "playwright"] = "off"
    v4_browser_use_enabled: bool = False
    v4_web_l2_origin_rules: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict, repr=False, exclude=True,
    )
    v4_web_reader_url: str | None = Field(default=None, repr=False, exclude=True)
    v4_web_reader_token: str | None = Field(default=None, repr=False, exclude=True)
    v4_web_l2_max_concurrency: int = Field(default=2, gt=0, le=2)
    v4_web_l2_per_tenant: int = Field(default=1, gt=0, le=1)
    v4_web_l2_timeout_seconds: float = Field(default=12, gt=0, le=12)
    v4_web_config_reason: WebConfigReason | None = None

    def web_extraction_enabled_for(self, tenant_id: str) -> bool:
        return self.v4_web_extractor_enabled and tenant_id in self.v4_web_extractor_tenants

    def request_config_values(self) -> dict[str, Any]:
        """Copy validated settings for server-side request overrides."""
        values = self.model_dump()
        for key in (
            "v4_web_extractor_tenants", "v4_web_l2_origin_rules",
            "v4_web_reader_url", "v4_web_reader_token",
        ):
            values[key] = deepcopy(getattr(self, key))
        return values
    v4_default_research_depth: str = Field(
        default="standard",
        title="V4 Default Research Depth",
        description="Default profile used when a request does not select a research depth",
    )
    v4_research_budget_limits: dict[str, int] = Field(
        default_factory=dict,
        title="V4 Research Budget Limits",
        description="Optional service-wide resource ceilings applied to v4 profiles",
    )
    v4_tenant_budget_limits: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        title="V4 Tenant Budget Limits",
        description="Optional tenant-specific resource ceilings applied to v4 profiles",
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
    task_react_enabled: bool = Field(
        default=True,
        title="Task ReAct Enabled",
        description="Whether to run a bounded task-level evidence repair loop before summarization",
    )
    task_react_max_rounds: int = Field(
        default=2,
        ge=1,
        title="Task ReAct Max Rounds",
        description="Maximum evidence-repair rounds allowed for a single task",
    )
    task_react_max_fetches_per_task: int = Field(
        default=2,
        ge=0,
        title="Task ReAct Max Fetches Per Task",
        description="Maximum number of fetch-page enrichment actions allowed per task",
    )
    task_react_max_additional_searches_per_task: int = Field(
        default=1,
        ge=0,
        title="Task ReAct Max Additional Searches Per Task",
        description="Maximum number of extra search rounds allowed per task after the initial search",
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
    report_layout_mode: str = Field(
        default="flexible",
        title="Report Layout Mode",
        description="Report layout strategy: flexible (core sections + optional custom sections) or fixed (legacy structure backup)",
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
    report_repair_enabled: bool = Field(
        default=True,
        title="Report Repair Enabled",
        description="Whether to run a bounded post-review evidence repair cycle before the final report",
    )
    report_repair_max_tasks: int = Field(
        default=2,
        ge=1,
        title="Report Repair Max Tasks",
        description="Maximum targeted repair tasks that may be added after the review stage",
    )
    report_repair_max_cycles: int = Field(
        default=1,
        ge=1,
        title="Report Repair Max Cycles",
        description="Maximum post-review repair cycles allowed per request",
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

    @field_validator("report_layout_mode", mode="before")
    @classmethod
    def _normalize_report_layout_mode(cls, value: Any) -> Any:
        if value is None:
            return "flexible"

        if isinstance(value, str):
            normalized = value.strip().lower() or "flexible"
            if normalized not in {"flexible", "fixed"}:
                raise ValueError("report_layout_mode must be either 'flexible' or 'fixed'")
            return normalized

        return value

    @field_validator("memory_maturity_gate", "skill_pack_maturity_gate", mode="before")
    @classmethod
    def _normalize_memory_maturity_gate(cls, value: Any) -> Any:
        if value is None:
            return "draft"

        if isinstance(value, str):
            normalized = value.strip().lower() or "draft"
            if normalized == "production":
                return "promoted"
            if normalized not in {"draft", "verified", "promoted"}:
                raise ValueError(
                    "maturity_gate must be one of 'draft', 'verified', 'promoted', 'production'"
                )
            return normalized

        return value

    @field_validator("policy_model_mode", mode="before")
    @classmethod
    def _normalize_policy_model_mode(cls, value: Any) -> Any:
        if value is None:
            return "off"
        if isinstance(value, str):
            normalized = value.strip().lower() or "off"
            if normalized not in {"off", "shadow", "canary", "active"}:
                raise ValueError("policy_model_mode must be one of 'off', 'shadow', 'canary', 'active'")
            return normalized
        return value

    @field_validator("nli_model_backend", mode="before")
    @classmethod
    def _normalize_nli_model_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower() or "auto"
            if value not in {"auto", "cross_encoder", "transformers_nli", "minicheck", "disabled"}:
                raise ValueError("nli_model_backend must be auto, cross_encoder, transformers_nli, minicheck, or disabled")
        return value

    @field_validator("nli_device", mode="before")
    @classmethod
    def _normalize_nli_device(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower() or "auto"
            if value not in {"auto", "cpu", "cuda", "mps"}:
                raise ValueError("nli_device must be auto, cpu, cuda, or mps")
        return value

    @field_validator("workflow_runner", mode="before")
    @classmethod
    def _normalize_workflow_runner(cls, value: Any) -> Any:
        if value is None:
            return "inline"
        if isinstance(value, str):
            normalized = value.strip().lower() or "inline"
            if normalized not in {"inline", "local_durable", "arq", "temporal"}:
                raise ValueError(
                    "workflow_runner must be one of 'inline', 'local_durable', 'arq', 'temporal'"
                )
            return normalized
        return value

    @field_validator("policy_model_canary_tenants", mode="before")
    @classmethod
    def _normalize_policy_canary_tenants(cls, value: Any) -> Any:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple, set)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        return ()

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

    @field_validator("mcp_allowed_private_hosts", mode="before")
    @classmethod
    def _normalize_mcp_private_hosts(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [host.strip() for host in value.split(",") if host.strip()]
        return [] if value is None else value

    @field_validator("llm_pricing_json", "tenant_api_keys", mode="before")
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

    @field_validator("v4_research_budget_limits", "v4_tenant_budget_limits", mode="before")
    @classmethod
    def _normalize_v4_budget_limits(cls, value: Any) -> Any:
        if value in (None, "", {}):
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("v4 budget limits must contain valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("v4 budget limits must be a JSON object")
        return value

    @model_validator(mode="before")
    @classmethod
    def _normalize_web_extraction_settings(cls, raw_values: Any) -> Any:
        if not isinstance(raw_values, dict):
            return raw_values
        values = raw_values.copy()
        reason = cls._normalize_web_extractor_gate(values)
        dynamic_reason = cls._normalize_web_dynamic_settings(values)
        reason = reason or dynamic_reason
        if reason:
            values["v4_web_l2_adapter"] = "off"
            values["v4_browser_use_enabled"] = False
            values["v4_web_config_reason"] = reason
            _LOGGER.warning("Web extraction configuration rejected (%s)", reason)
        return values

    @staticmethod
    def _normalize_web_extractor_gate(values: dict[str, Any]) -> WebConfigReason | None:
        enabled_key = "v4_web_extractor_enabled"
        reason = None
        try:
            enabled = _web_bool(values.get(enabled_key, values.get(_WEB_EXTRACTOR_ALIAS, False)))
            if enabled_key in values and _WEB_EXTRACTOR_ALIAS in values:
                if enabled is not _web_bool(values[_WEB_EXTRACTOR_ALIAS]):
                    reason = "web_extractor_alias_conflict"
        except ValueError:
            enabled = False
            reason = "invalid_web_extractor_enabled"

        tenants_key = "v4_web_extractor_tenants"
        if tenants_key in values:
            try:
                values[tenants_key] = _web_tenants(values[tenants_key])
            except (ValueError, RecursionError):
                values[tenants_key] = []
                reason = reason or "invalid_web_extractor_tenants"
        if enabled_key in values or _WEB_EXTRACTOR_ALIAS in values or reason:
            values[enabled_key] = enabled and reason is None
        values.pop(_WEB_EXTRACTOR_ALIAS, None)
        return reason

    @staticmethod
    def _normalize_web_dynamic_settings(values: dict[str, Any]) -> WebConfigReason | None:
        adapter_key = "v4_web_l2_adapter"
        adapter = values.get(adapter_key, "off")
        if isinstance(adapter, str):
            adapter = adapter.strip().lower()
        reason = None
        if adapter not in ("off", "reader_http", "playwright"):
            reason = "invalid_web_l2_adapter"
            adapter = "off"
        if adapter_key in values:
            values[adapter_key] = adapter

        browser_key = "v4_browser_use_enabled"
        if browser_key in values:
            try:
                values[browser_key] = _web_bool(values[browser_key])
            except ValueError:
                reason = reason or "invalid_browser_use_enabled"

        reader_url_key = "v4_web_reader_url"
        reader_url = values.get(reader_url_key)
        if reader_url == "":
            reader_url = None
        if reader_url is not None and not isinstance(reader_url, str):
            reason = reason or "invalid_web_reader_url"
            reader_url = None
        if reader_url_key in values:
            values[reader_url_key] = reader_url

        reader_token_key = "v4_web_reader_token"
        token = values.get(reader_token_key)
        if reader_token_key in values:
            try:
                values[reader_token_key] = None if token in (None, "") else validate_reader_token(token)
            except ValueError:
                values[reader_token_key] = None
                reason = reason or "invalid_web_reader_token"

        for key, (numeric_type, maximum) in _WEB_L2_LIMITS.items():
            if key not in values:
                continue
            try:
                values[key] = _web_limit(values[key], numeric_type, maximum)
            except (ValueError, OverflowError):
                values[key] = maximum
                reason = reason or "invalid_web_l2_limits"

        rules_key = "v4_web_l2_origin_rules"
        rules = values.get(rules_key, {})
        try:
            rules = _web_structured(rules, dict)
            if any(
                not isinstance(tenant, str)
                or not isinstance(items, list)
                or any(
                    not isinstance(rule, dict) or any(not isinstance(key, str) for key in rule)
                    for rule in items
                )
                for tenant, items in rules.items()
            ):
                raise ValueError("invalid web extraction origin rules")
        except (ValueError, RecursionError):
            reason = reason or "invalid_web_l2_origin_rules"
            rules = {}
        if rules_key in values:
            values[rules_key] = rules

        if (
            reason
            or not values.get("v4_web_extractor_enabled")
            or adapter == "off"
            or (adapter == "playwright" and not values.get(browser_key))
        ):
            return reason

        from services.web_dynamic_adapter import (
            DynamicTargetPolicy,
            validate_reader_endpoint,
        )

        if rules:
            try:
                DynamicTargetPolicy.from_config(rules)
            except ValueError:
                values[rules_key] = {}
                return "invalid_web_l2_origin_rules"
        if adapter == "reader_http":
            try:
                values[reader_url_key] = validate_reader_endpoint(reader_url)
            except ValueError:
                values[reader_url_key] = None
                return "invalid_web_reader_url"
            if token is None or token == "":
                return "invalid_web_reader_token"
        return None

    @field_validator("v4_default_research_depth", mode="before")
    @classmethod
    def _normalize_v4_research_depth(cls, value: Any) -> str:
        depth = str(value or "standard").strip().lower()
        if depth not in {"quick", "standard", "deep"}:
            raise ValueError("v4_default_research_depth must be quick, standard, or deep")
        return depth

    @field_validator("v4_research_budget_limits")
    @classmethod
    def _validate_v4_service_budget_limits(cls, value: dict[str, int]) -> dict[str, int]:
        valid_resources = {
            "tasks", "search_calls", "web_fetches", "dynamic_fetches", "sandbox_calls",
            "llm_calls", "input_tokens", "output_tokens", "cost_microusd",
        }
        if set(value).difference(valid_resources):
            raise ValueError("v4_research_budget_limits contains an unknown resource")
        if any(isinstance(amount, bool) or amount < 0 for amount in value.values()):
            raise ValueError("v4 research budget limits must be non-negative integers")
        return dict(value)

    @field_validator("v4_tenant_budget_limits")
    @classmethod
    def _validate_v4_tenant_budget_limits(
        cls,
        value: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int]]:
        valid_resources = {
            "tasks", "search_calls", "web_fetches", "dynamic_fetches", "sandbox_calls",
            "llm_calls", "input_tokens", "output_tokens", "cost_microusd",
        }
        result: dict[str, dict[str, int]] = {}
        for tenant_id, limits in value.items():
            if not str(tenant_id).strip() or not isinstance(limits, dict):
                raise ValueError("each v4 tenant budget limit must map a tenant to resource limits")
            if set(limits).difference(valid_resources):
                raise ValueError("v4 tenant budget limits contain an unknown resource")
            if any(isinstance(amount, bool) or amount < 0 for amount in limits.values()):
                raise ValueError("v4 tenant budget limits must be non-negative integers")
            result[str(tenant_id)] = dict(limits)
        return result

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

        vector_dir = (self.search_cache_vector_dir or "").strip()
        if vector_dir:
            vector_path = Path(vector_dir).expanduser()
            if not vector_path.is_absolute():
                vector_path = backend_root() / vector_path
            self.search_cache_vector_dir = str(vector_path.resolve(strict=False))

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

        note_memory_dir = (self.note_memory_dir or "").strip()
        if note_memory_dir:
            memory_path = Path(note_memory_dir).expanduser()
            if not memory_path.is_absolute():
                memory_path = backend_root() / memory_path
            self.note_memory_dir = str(memory_path.resolve(strict=False))

        strategy_memory_dir = (self.strategy_memory_dir or "").strip()
        if strategy_memory_dir:
            memory_path = Path(strategy_memory_dir).expanduser()
            if not memory_path.is_absolute():
                memory_path = backend_root() / memory_path
            self.strategy_memory_dir = str(memory_path.resolve(strict=False))

        evidence_memory_dir = (self.evidence_memory_dir or "").strip()
        if evidence_memory_dir:
            memory_path = Path(evidence_memory_dir).expanduser()
            if not memory_path.is_absolute():
                memory_path = backend_root() / memory_path
            self.evidence_memory_dir = str(memory_path.resolve(strict=False))

        skill_pack_dir = (self.skill_pack_dir or "").strip()
        if skill_pack_dir:
            pack_path = Path(skill_pack_dir).expanduser()
            if not pack_path.is_absolute():
                pack_path = backend_root() / pack_path
            self.skill_pack_dir = str(pack_path.resolve(strict=False))

        skill_format_dir = (self.skill_format_dir or "").strip()
        if skill_format_dir:
            format_path = Path(skill_format_dir).expanduser()
            if not format_path.is_absolute():
                format_path = backend_root() / format_path
            self.skill_format_dir = str(format_path.resolve(strict=False))

        mcp_cfg = (self.mcp_config_path or "").strip()
        if mcp_cfg:
            mcp_path = Path(mcp_cfg).expanduser()
            if not mcp_path.is_absolute():
                mcp_path = backend_root() / mcp_path
            self.mcp_config_path = str(mcp_path.resolve(strict=False))

        pyodide_dir = (self.sandbox_pyodide_cache_dir or "").strip()
        if pyodide_dir:
            pyodide_path = Path(pyodide_dir).expanduser()
            if not pyodide_path.is_absolute():
                pyodide_path = backend_root() / pyodide_path
            self.sandbox_pyodide_cache_dir = str(pyodide_path.resolve(strict=False))
        return self

    def resolved_max_total_tasks(self) -> int:
        """Return the hard total task ceiling across planning, reflection, and repair."""
        if self.max_total_tasks is not None:
            return max(1, int(self.max_total_tasks))

        base = max(1, int(self.max_agent_tasks or 1))
        if base < 5:
            return base

        reserve = 0
        if self.request_reflection_enabled:
            reserve += max(0, int(self.reflection_reserve_tasks or 0))
        if self.report_repair_enabled:
            reserve += max(0, int(self.repair_reserve_tasks or 0))
        if reserve <= 0:
            return base

        return min(max(base, 4 + reserve), max(base + 2, 7))

    def resolved_initial_plan_max_tasks(self) -> int:
        """Return the task budget allocated to initial planning."""
        total_cap = self.resolved_max_total_tasks()
        if self.initial_plan_max_tasks is not None:
            return min(max(1, int(self.initial_plan_max_tasks)), total_cap)

        base = max(1, int(self.max_agent_tasks or 1))
        has_post_stage_reserve = (
            (self.request_reflection_enabled and int(self.reflection_reserve_tasks or 0) > 0)
            or (self.report_repair_enabled and int(self.repair_reserve_tasks or 0) > 0)
        )
        if base >= 5 and has_post_stage_reserve:
            return min(4, total_cap)
        return min(base, total_cap)

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
            if field_name != "v4_web_config_reason" and env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]
        old_web_flag = os.getenv(_WEB_EXTRACTOR_ALIAS.upper())
        if old_web_flag is not None:
            raw_values[_WEB_EXTRACTOR_ALIAS] = old_web_flag

        # Additional mappings for explicit env names
        env_aliases = {
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_context_window": os.getenv("LLM_CONTEXT_WINDOW"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "fast_llm_provider": os.getenv("FAST_LLM_PROVIDER"),
            "fast_llm_model_id": os.getenv("FAST_LLM_MODEL_ID"),
            "fast_llm_base_url": os.getenv("FAST_LLM_BASE_URL"),
            "fast_llm_api_key": os.getenv("FAST_LLM_API_KEY"),
            "fast_llm_temperature": os.getenv("FAST_LLM_TEMPERATURE"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "use_tool_calling": os.getenv("USE_TOOL_CALLING"),
            "search_api": os.getenv("SEARCH_API"),
            "advanced_search_backends": os.getenv("ADVANCED_SEARCH_BACKENDS"),
            "advanced_search_max_concurrency": os.getenv("ADVANCED_SEARCH_MAX_CONCURRENCY"),
            "advanced_search_fetch_full_page_override": os.getenv("ADVANCED_SEARCH_FETCH_FULL_PAGE_OVERRIDE"),
            "advanced_backend_timeout_seconds": os.getenv("ADVANCED_BACKEND_TIMEOUT_SECONDS"),
            "advanced_rerank_enabled": os.getenv("ADVANCED_RERANK_ENABLED"),
            "advanced_rerank_base_url": os.getenv("ADVANCED_RERANK_BASE_URL"),
            "advanced_rerank_api_key": os.getenv("ADVANCED_RERANK_API_KEY"),
            "advanced_rerank_model": os.getenv("ADVANCED_RERANK_MODEL"),
            "advanced_rerank_candidate_pool": os.getenv("ADVANCED_RERANK_CANDIDATE_POOL"),
            "advanced_rerank_timeout_seconds": os.getenv("ADVANCED_RERANK_TIMEOUT_SECONDS"),
            "advanced_rerank_max_content_chars": os.getenv("ADVANCED_RERANK_MAX_CONTENT_CHARS"),
            "semantic_scholar_api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            "enable_notes": os.getenv("ENABLE_NOTES"),
            "notes_workspace": os.getenv("NOTES_WORKSPACE"),
            "memory_shadow_mode": os.getenv("MEMORY_SHADOW_MODE"),
            "memory_maturity_gate": os.getenv("MEMORY_MATURITY_GATE"),
            "memory_decay_idle_days": os.getenv("MEMORY_DECAY_IDLE_DAYS"),
            "skill_pack_enabled": os.getenv("SKILL_PACK_ENABLED"),
            "skill_pack_dir": os.getenv("SKILL_PACK_DIR"),
            "skill_prompt_char_limit": os.getenv("SKILL_PROMPT_CHAR_LIMIT"),
            "skill_pack_min_match_score": os.getenv("SKILL_PACK_MIN_MATCH_SCORE"),
            "skill_format_dir": os.getenv("SKILL_FORMAT_DIR"),
            "skill_pack_composable_enabled": os.getenv("SKILL_PACK_COMPOSABLE_ENABLED"),
            "skill_pack_secondary_min_match_score": os.getenv("SKILL_PACK_SECONDARY_MIN_MATCH_SCORE"),
            "skill_pack_semantic_routing_enabled": os.getenv("SKILL_PACK_SEMANTIC_ROUTING_ENABLED"),
            "skill_pack_llm_router_enabled": os.getenv("SKILL_PACK_LLM_ROUTER_ENABLED"),
            "skill_pack_maturity_gate": os.getenv("SKILL_PACK_MATURITY_GATE"),
            "skill_distill_min_support_cases": os.getenv("SKILL_DISTILL_MIN_SUPPORT_CASES"),
            "skill_distill_min_hybrid_gain": os.getenv("SKILL_DISTILL_MIN_HYBRID_GAIN"),
            "skill_distill_max_adversarial_regression": os.getenv(
                "SKILL_DISTILL_MAX_ADVERSARIAL_REGRESSION"
            ),
            "default_output_format": os.getenv("DEFAULT_OUTPUT_FORMAT"),
            "kv_cache_prefix_optimization_enabled": os.getenv("KV_CACHE_PREFIX_OPTIMIZATION_ENABLED"),
            "kv_cache_provider_mode": os.getenv("KV_CACHE_PROVIDER_MODE"),
            "kv_cache_anthropic_breakpoints_enabled": os.getenv("KV_CACHE_ANTHROPIC_BREAKPOINTS_ENABLED"),
            "kv_cache_min_segment_tokens": os.getenv("KV_CACHE_MIN_SEGMENT_TOKENS"),
            "kv_cache_prompt_cache_key_enabled": os.getenv("KV_CACHE_PROMPT_CACHE_KEY_ENABLED"),
            "kv_cache_block_alignment_tokens": os.getenv("KV_CACHE_BLOCK_ALIGNMENT_TOKENS"),
            "context_compaction_enabled": os.getenv("CONTEXT_COMPACTION_ENABLED"),
            "fact_ledger_llm_compaction_enabled": os.getenv("FACT_LEDGER_LLM_COMPACTION_ENABLED"),
            "fact_ledger_max_entries": os.getenv("FACT_LEDGER_MAX_ENTRIES"),
            "evidence_pointer_disclosure_enabled": os.getenv("EVIDENCE_POINTER_DISCLOSURE_ENABLED"),
            "evidence_lazy_load_max_chunks_per_source": os.getenv("EVIDENCE_LAZY_LOAD_MAX_CHUNKS_PER_SOURCE"),
            "evidence_lazy_load_char_budget": os.getenv("EVIDENCE_LAZY_LOAD_CHAR_BUDGET"),
            "mcp_enabled": os.getenv("MCP_ENABLED"),
            "mcp_config_path": os.getenv("MCP_CONFIG_PATH"),
            "mcp_enforce_readonly": os.getenv("MCP_ENFORCE_READONLY"),
            "mcp_allow_hitl_write_tools": os.getenv("MCP_ALLOW_HITL_WRITE_TOOLS"),
            "mcp_approval_timeout_seconds": os.getenv("MCP_APPROVAL_TIMEOUT_SECONDS"),
            "mcp_allowed_private_hosts": os.getenv("MCP_ALLOWED_PRIVATE_HOSTS"),
            "mcp_allow_insecure_http": os.getenv("MCP_ALLOW_INSECURE_HTTP"),
            "mcp_tool_timeout_seconds": os.getenv("MCP_TOOL_TIMEOUT_SECONDS"),
            "mcp_tool_cache_ttl_seconds": os.getenv("MCP_TOOL_CACHE_TTL_SECONDS"),
            "mcp_max_tools_per_stage": os.getenv("MCP_MAX_TOOLS_PER_STAGE"),
            "mcp_tenant_token_secret": os.getenv("MCP_TENANT_TOKEN_SECRET"),
            "multimodal_evidence_enabled": os.getenv("MULTIMODAL_EVIDENCE_ENABLED"),
            "multimodal_vlm_enabled": os.getenv("MULTIMODAL_VLM_ENABLED"),
            "multimodal_vlm_model": os.getenv("MULTIMODAL_VLM_MODEL_ID") or os.getenv("MULTIMODAL_VLM_MODEL"),
            "multimodal_vlm_base_url": os.getenv("MULTIMODAL_VLM_BASE_URL"),
            "multimodal_vlm_api_key": os.getenv("MULTIMODAL_VLM_API_KEY"),
            "multimodal_vlm_timeout_seconds": os.getenv("MULTIMODAL_VLM_TIMEOUT_SECONDS"),
            "multimodal_vlm_max_concurrency": os.getenv("MULTIMODAL_VLM_MAX_CONCURRENCY"),
            "multimodal_max_pdf_pages": os.getenv("MULTIMODAL_MAX_PDF_PAGES"),
            "multimodal_max_pdf_bytes": os.getenv("MULTIMODAL_MAX_PDF_BYTES"),
            "multimodal_pdf_render_dpi": os.getenv("MULTIMODAL_PDF_RENDER_DPI"),
            "multimodal_max_vlm_pages_per_doc": os.getenv("MULTIMODAL_MAX_VLM_PAGES_PER_DOC"),
            "multimodal_max_images_per_page": os.getenv("MULTIMODAL_MAX_WEB_IMAGES") or os.getenv("MULTIMODAL_MAX_IMAGES_PER_PAGE"),
            "multimodal_image_max_bytes": os.getenv("MULTIMODAL_IMAGE_MAX_BYTES"),
            "multimodal_image_max_pixels": os.getenv("MULTIMODAL_IMAGE_MAX_PIXELS"),
            "sandbox_enabled": os.getenv("SANDBOX_ENABLED"),
            "sandbox_backend": os.getenv("SANDBOX_BACKEND"),
            "sandbox_docker_image": os.getenv("SANDBOX_DOCKER_IMAGE"),
            "sandbox_timeout_seconds": os.getenv("SANDBOX_TIMEOUT_SECONDS"),
            "sandbox_memory_mb": os.getenv("SANDBOX_MEMORY_MB"),
            "sandbox_cpus": os.getenv("SANDBOX_CPUS"),
            "sandbox_max_executions_per_task": os.getenv("SANDBOX_MAX_RUNS_PER_TASK") or os.getenv("SANDBOX_MAX_EXECUTIONS_PER_TASK"),
            "sandbox_pyodide_cache_dir": os.getenv("SANDBOX_PYODIDE_CACHE_DIR"),
            "sandbox_pyodide_node_bin": os.getenv("SANDBOX_PYODIDE_NODE_BIN"),
            "nli_verifier_enabled": os.getenv("NLI_VERIFIER_ENABLED"),
            "nli_model_backend": os.getenv("NLI_MODEL_BACKEND"),
            "nli_model_name": os.getenv("NLI_MODEL_NAME"),
            "nli_device": os.getenv("NLI_DEVICE"),
            "nli_warmup_enabled": os.getenv("NLI_WARMUP_ENABLED"),
            "nli_decompose_with_fast_llm": os.getenv("NLI_DECOMPOSE_WITH_FAST_LLM"),
            "nli_batch_size": os.getenv("NLI_BATCH_SIZE"),
            "nli_max_length": os.getenv("NLI_MAX_LENGTH"),
            "nli_max_atomic_claims_per_task": os.getenv("NLI_MAX_ATOMIC_CLAIMS_PER_TASK"),
            "nli_top_k_chunks_per_source": os.getenv("NLI_TOP_K_CHUNKS_PER_SOURCE"),
            "nli_entailment_threshold": os.getenv("NLI_ENTAILMENT_THRESHOLD"),
            "nli_contradiction_threshold": os.getenv("NLI_CONTRADICTION_THRESHOLD"),
            "nli_stage_budget_seconds": os.getenv("NLI_STAGE_BUDGET_SECONDS"),
            "nli_conflict_graph_enabled": os.getenv("NLI_CONFLICT_GRAPH_ENABLED"),
            "nli_conflict_numeric_rel_tol": os.getenv("NLI_CONFLICT_NUMERIC_REL_TOL"),
            "nli_conflict_min_score": os.getenv("NLI_CONFLICT_MIN_SCORE"),
            "note_memory_enabled": os.getenv("NOTE_MEMORY_ENABLED"),
            "note_memory_dir": os.getenv("NOTE_MEMORY_DIR"),
            "note_memory_embedding_model": os.getenv("NOTE_MEMORY_EMBEDDING_MODEL"),
            "note_memory_planning_top_k": os.getenv("NOTE_MEMORY_PLANNING_TOP_K"),
            "note_memory_execution_top_k": os.getenv("NOTE_MEMORY_EXECUTION_TOP_K"),
            "note_memory_prompt_char_limit": os.getenv("NOTE_MEMORY_PROMPT_CHAR_LIMIT"),
            "strategy_memory_enabled": os.getenv("STRATEGY_MEMORY_ENABLED"),
            "strategy_memory_dir": os.getenv("STRATEGY_MEMORY_DIR"),
            "strategy_memory_embedding_model": os.getenv("STRATEGY_MEMORY_EMBEDDING_MODEL"),
            "strategy_memory_planning_top_k": os.getenv("STRATEGY_MEMORY_PLANNING_TOP_K"),
            "strategy_memory_reflection_top_k": os.getenv("STRATEGY_MEMORY_REFLECTION_TOP_K"),
            "strategy_memory_prompt_char_limit": os.getenv("STRATEGY_MEMORY_PROMPT_CHAR_LIMIT"),
            "evidence_runtime_enabled": os.getenv("EVIDENCE_RUNTIME_ENABLED"),
            "evidence_runtime_top_k": os.getenv("EVIDENCE_RUNTIME_TOP_K"),
            "evidence_chunk_chars": os.getenv("EVIDENCE_CHUNK_CHARS"),
            "evidence_chunk_overlap": os.getenv("EVIDENCE_CHUNK_OVERLAP"),
            "evidence_runtime_max_chunks": os.getenv("EVIDENCE_RUNTIME_MAX_CHUNKS"),
            "evidence_runtime_embedding_model": os.getenv("EVIDENCE_RUNTIME_EMBEDDING_MODEL"),
            "evidence_memory_enabled": os.getenv("EVIDENCE_MEMORY_ENABLED"),
            "evidence_memory_dir": os.getenv("EVIDENCE_MEMORY_DIR"),
            "evidence_memory_top_k": os.getenv("EVIDENCE_MEMORY_TOP_K"),
            "evidence_memory_embedding_model": os.getenv("EVIDENCE_MEMORY_EMBEDDING_MODEL"),
            "host": os.getenv("HOST"),
            "port": os.getenv("PORT"),
            "log_level": os.getenv("LOG_LEVEL"),
            "cors_origins": os.getenv("CORS_ORIGINS"),
            "sse_heartbeat_interval_seconds": os.getenv("SSE_HEARTBEAT_INTERVAL_SECONDS"),
            "async_memory_refresh_enabled": os.getenv("ASYNC_MEMORY_REFRESH_ENABLED"),
            "langfuse_enabled": os.getenv("LANGFUSE_ENABLED"),
            "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "langfuse_host": os.getenv("LANGFUSE_HOST"),
            "auth_enabled": os.getenv("AUTH_ENABLED"),
            "tenant_api_keys": os.getenv("TENANT_API_KEYS"),
            "default_tenant_id": os.getenv("DEFAULT_TENANT_ID"),
            "tenant_rate_limit_rpm": os.getenv("TENANT_RATE_LIMIT_RPM"),
            "tenant_max_concurrent_requests": os.getenv("TENANT_MAX_CONCURRENT_REQUESTS"),
            "redis_url": os.getenv("REDIS_URL"),
            "database_url": os.getenv("DATABASE_URL"),
            "event_bus_max_events_per_request": os.getenv("EVENT_BUS_MAX_EVENTS_PER_REQUEST"),
            "event_bus_stream_ttl_seconds": os.getenv("EVENT_BUS_STREAM_TTL_SECONDS"),
            "workflow_runner": os.getenv("WORKFLOW_RUNNER"),
            "workflow_runner_fallback_inline": os.getenv("WORKFLOW_RUNNER_FALLBACK_INLINE"),
            "temporal_activity_heartbeat_seconds": os.getenv("TEMPORAL_ACTIVITY_HEARTBEAT_SECONDS"),
            "hitl_signal_timeout_seconds": os.getenv("HITL_SIGNAL_TIMEOUT_SECONDS"),
            "temporal_address": os.getenv("TEMPORAL_HOST") or os.getenv("TEMPORAL_ADDRESS"),
            "temporal_namespace": os.getenv("TEMPORAL_NAMESPACE"),
            "temporal_task_queue": os.getenv("TEMPORAL_TASK_QUEUE"),
            "arq_redis_dsn": os.getenv("ARQ_REDIS_URL") or os.getenv("ARQ_REDIS_DSN"),
            "postgres_pool_min_size": os.getenv("PG_POOL_MIN_SIZE") or os.getenv("POSTGRES_POOL_MIN_SIZE"),
            "postgres_pool_max_size": os.getenv("PG_POOL_MAX_SIZE") or os.getenv("POSTGRES_POOL_MAX_SIZE"),
            "policy_model_enabled": os.getenv("POLICY_MODEL_ENABLED"),
            "policy_model_id": os.getenv("POLICY_MODEL_ID"),
            "policy_model_base_url": os.getenv("POLICY_MODEL_BASE_URL"),
            "policy_model_api_key": os.getenv("POLICY_MODEL_API_KEY"),
            "policy_model_timeout_seconds": os.getenv("POLICY_MODEL_TIMEOUT_SECONDS"),
            "policy_model_mode": os.getenv("POLICY_MODEL_MODE"),
            "policy_model_canary_ratio": os.getenv("POLICY_MODEL_CANARY_RATIO"),
            "policy_model_canary_tenants": os.getenv("POLICY_MODEL_CANARY_TENANTS"),
            "policy_model_prompt_version": os.getenv("POLICY_MODEL_PROMPT_VERSION"),
            "policy_model_circuit_breaker_window": os.getenv("POLICY_MODEL_CIRCUIT_BREAKER_WINDOW"),
            "policy_model_circuit_breaker_fallback_rate": os.getenv(
                "POLICY_MODEL_CIRCUIT_BREAKER_FALLBACK_RATE"
            ),
            "policy_model_max_p95_ms": os.getenv("POLICY_MODEL_MAX_P95_MS"),
            "search_cache_enabled": os.getenv("SEARCH_CACHE_ENABLED"),
            "search_cache_ttl_seconds": os.getenv("SEARCH_CACHE_TTL_SECONDS"),
            "search_cache_dynamic_ttl_enabled": os.getenv("SEARCH_CACHE_DYNAMIC_TTL_ENABLED"),
            "search_cache_fresh_ttl_seconds": os.getenv("SEARCH_CACHE_FRESH_TTL_SECONDS"),
            "search_cache_evergreen_ttl_seconds": os.getenv("SEARCH_CACHE_EVERGREEN_TTL_SECONDS"),
            "search_cache_dir": os.getenv("SEARCH_CACHE_DIR"),
            "search_cache_vector_dir": os.getenv("SEARCH_CACHE_VECTOR_DIR"),
            "semantic_cache_enabled": os.getenv("SEMANTIC_CACHE_ENABLED"),
            "semantic_cache_embedding_model": os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL"),
            "semantic_cache_similarity_threshold": os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD"),
            "semantic_cache_lexical_threshold": os.getenv("SEMANTIC_CACHE_LEXICAL_THRESHOLD"),
            "approximate_cache_enabled": os.getenv("APPROXIMATE_CACHE_ENABLED"),
            "approximate_cache_dense_enabled": os.getenv("APPROXIMATE_CACHE_DENSE_ENABLED"),
            "approximate_cache_sparse_enabled": os.getenv("APPROXIMATE_CACHE_SPARSE_ENABLED"),
            "approximate_cache_dense_threshold": os.getenv("APPROXIMATE_CACHE_DENSE_THRESHOLD"),
            "approximate_cache_sparse_threshold": os.getenv("APPROXIMATE_CACHE_SPARSE_THRESHOLD"),
            "approximate_cache_dense_top_k": os.getenv("APPROXIMATE_CACHE_DENSE_TOP_K"),
            "approximate_cache_sparse_top_k": os.getenv("APPROXIMATE_CACHE_SPARSE_TOP_K"),
            "max_tokens_per_source": os.getenv("MAX_TOKENS_PER_SOURCE"),
            "task_context_char_limit": os.getenv("TASK_CONTEXT_CHAR_LIMIT"),
            "task_summary_max_concurrency": os.getenv("TASK_SUMMARY_MAX_CONCURRENCY"),
            "max_agent_tasks": os.getenv("MAX_AGENT_TASKS"),
            "initial_plan_max_tasks": os.getenv("INITIAL_PLAN_MAX_TASKS"),
            "reflection_reserve_tasks": os.getenv("REFLECTION_RESERVE_TASKS"),
            "repair_reserve_tasks": os.getenv("REPAIR_RESERVE_TASKS"),
            "max_total_tasks": os.getenv("MAX_TOTAL_TASKS"),
            "request_reflection_enabled": os.getenv("REQUEST_REFLECTION_ENABLED"),
            "reflection_max_additional_tasks": os.getenv("REFLECTION_MAX_ADDITIONAL_TASKS"),
            "review_stage_enabled": os.getenv("REVIEW_STAGE_ENABLED"),
            "review_agent_enabled": os.getenv("REVIEW_AGENT_ENABLED"),
            "review_min_sources_per_task": os.getenv("REVIEW_MIN_SOURCES_PER_TASK"),
            "review_min_domains_per_task": os.getenv("REVIEW_MIN_DOMAINS_PER_TASK"),
            "freshness_reference_days": os.getenv("FRESHNESS_REFERENCE_DAYS"),
            "task_query_rewrite_enabled": os.getenv("TASK_QUERY_REWRITE_ENABLED"),
            "task_react_enabled": os.getenv("TASK_REACT_ENABLED"),
            "task_react_max_rounds": os.getenv("TASK_REACT_MAX_ROUNDS"),
            "task_react_max_fetches_per_task": os.getenv("TASK_REACT_MAX_FETCHES_PER_TASK"),
            "task_react_max_additional_searches_per_task": os.getenv(
                "TASK_REACT_MAX_ADDITIONAL_SEARCHES_PER_TASK"
            ),
            "search_tool_timeout_seconds": os.getenv("SEARCH_TOOL_TIMEOUT_SECONDS"),
            "search_tool_retry_attempts": os.getenv("SEARCH_TOOL_RETRY_ATTEMPTS"),
            "search_tool_retry_backoff_seconds": os.getenv("SEARCH_TOOL_RETRY_BACKOFF_SECONDS"),
            "direct_answer_char_limit": os.getenv("DIRECT_ANSWER_CHAR_LIMIT"),
            "report_summary_char_limit": os.getenv("REPORT_SUMMARY_CHAR_LIMIT"),
            "report_sources_char_limit": os.getenv("REPORT_SOURCES_CHAR_LIMIT"),
            "report_layout_mode": os.getenv("REPORT_LAYOUT_MODE"),
            "metrics_recent_requests_limit": os.getenv("METRICS_RECENT_REQUESTS_LIMIT"),
            "request_state_enabled": os.getenv("REQUEST_STATE_ENABLED"),
            "request_state_dir": os.getenv("REQUEST_STATE_DIR"),
            "request_state_recent_limit": os.getenv("REQUEST_STATE_RECENT_LIMIT"),
            "report_repair_enabled": os.getenv("REPORT_REPAIR_ENABLED"),
            "report_repair_max_tasks": os.getenv("REPORT_REPAIR_MAX_TASKS"),
            "report_repair_max_cycles": os.getenv("REPORT_REPAIR_MAX_CYCLES"),
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
                web_setting = key.startswith(("v4_web_", "v4_browser_use_"))
                if key != "v4_web_config_reason" and (value is not None or web_setting):
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

    def resolved_search_cache_vector_dir(self) -> str:
        """Return the persistent ANN vector cache directory."""
        return self.search_cache_vector_dir

    def resolved_approximate_cache_enabled(self) -> bool:
        """Return whether approximate cache reuse is enabled, honoring semantic-cache aliases."""

        if self.approximate_cache_enabled is not None:
            return bool(self.approximate_cache_enabled)
        return bool(self.semantic_cache_enabled)

    def resolved_approximate_cache_dense_enabled(self) -> bool:
        """Return whether dense embedding recall participates in approximate cache reuse."""

        if self.approximate_cache_dense_enabled is not None:
            return bool(self.approximate_cache_dense_enabled)
        return True

    def resolved_approximate_cache_sparse_enabled(self) -> bool:
        """Return whether sparse lexical recall participates in approximate cache reuse."""

        if self.approximate_cache_sparse_enabled is not None:
            return bool(self.approximate_cache_sparse_enabled)
        return True

    def resolved_approximate_cache_dense_threshold(self) -> float:
        """Return the dense approximate-cache hit threshold."""

        if self.approximate_cache_dense_threshold is not None:
            return float(self.approximate_cache_dense_threshold)
        return float(self.semantic_cache_similarity_threshold)

    def resolved_approximate_cache_sparse_threshold(self) -> float:
        """Return the sparse approximate-cache hit threshold."""

        if self.approximate_cache_sparse_threshold is not None:
            return float(self.approximate_cache_sparse_threshold)
        return float(self.semantic_cache_lexical_threshold)

    def resolved_approximate_cache_dense_top_k(self) -> int:
        """Return the bounded ANN recall size for approximate cache matching."""

        return max(int(self.approximate_cache_dense_top_k or 1), 1)

    def resolved_approximate_cache_sparse_top_k(self) -> int:
        """Return the bounded lexical recall size for approximate cache matching."""

        return max(int(self.approximate_cache_sparse_top_k or 1), 1)

    def resolved_note_memory_embedding_model(self) -> str:
        """Return the embedding model used by note memory retrieval."""

        return (
            str(self.note_memory_embedding_model or "").strip()
            or str(self.semantic_cache_embedding_model or "").strip()
            or "sentence-transformers/all-MiniLM-L6-v2"
        )

    def resolved_strategy_memory_embedding_model(self) -> str:
        """Return the embedding model used by strategy memory retrieval."""

        return (
            str(self.strategy_memory_embedding_model or "").strip()
            or self.resolved_note_memory_embedding_model()
        )

    def resolved_evidence_runtime_embedding_model(self) -> str:
        """Return the embedding model used by runtime evidence retrieval."""

        return (
            str(self.evidence_runtime_embedding_model or "").strip()
            or self.resolved_note_memory_embedding_model()
        )

    def resolved_evidence_memory_embedding_model(self) -> str:
        """Return the embedding model used by persistent evidence retrieval."""

        return (
            str(self.evidence_memory_embedding_model or "").strip()
            or self.resolved_evidence_runtime_embedding_model()
        )

    def resolved_advanced_search_backends(self) -> list[str]:
        """Return the ordered concrete backends used for fused search."""
        return list(self.advanced_search_backends or ["searxng", "tavily", "serpapi", "duckduckgo"])

    def resolved_advanced_search_max_concurrency(self) -> int:
        """Return the bounded worker count used by advanced fan-out search."""
        requested = int(self.advanced_search_max_concurrency or 1)
        return max(1, min(requested, len(self.resolved_advanced_search_backends()) or 1))

    def resolved_advanced_fetch_full_page(self) -> bool:
        """Return the fetch_full_page flag used by advanced fan-out requests."""
        if self.advanced_search_fetch_full_page_override is not None:
            return bool(self.advanced_search_fetch_full_page_override)
        # Advanced fan-out already expands multiple providers, so defaulting to
        # snippet-only results avoids slow per-result page fetches blowing past
        # the outer search tool timeout. Teams can opt back in explicitly.
        return False

    def resolved_advanced_backend_timeout_seconds(self) -> float:
        """Return the deadline used to stop waiting on slow advanced backends."""
        if self.advanced_backend_timeout_seconds is not None:
            return max(0.25, float(self.advanced_backend_timeout_seconds))

        search_timeout = max(0.25, float(self.search_tool_timeout_seconds or 10.0))
        if search_timeout <= 1.5:
            return max(0.25, search_timeout * 0.75)
        return max(0.25, search_timeout - 1.0)

    def resolved_advanced_rerank_model(self) -> str | None:
        """Return the model identifier used for advanced search reranking."""
        return (self.advanced_rerank_model or self.resolved_model() or "").strip() or None

    def resolved_advanced_rerank_base_url(self) -> str | None:
        """Return the OpenAI-compatible base URL used for advanced reranking."""
        explicit = (self.advanced_rerank_base_url or "").strip()
        if explicit:
            return explicit

        provider = (self.llm_provider or "").strip()
        if provider == "ollama":
            return self.sanitized_ollama_url()
        if provider == "lmstudio":
            return self.lmstudio_base_url
        return (self.llm_base_url or "").strip() or None

    def resolved_advanced_rerank_api_key(self) -> str | None:
        """Return the API key used for advanced reranking, when required."""
        explicit = (self.advanced_rerank_api_key or "").strip()
        if explicit:
            return explicit
        return (self.llm_api_key or "").strip() or None

    def resolved_advanced_rerank_candidate_pool(self) -> int:
        """Return the candidate pool size used before final truncation."""
        return max(1, int(self.advanced_rerank_candidate_pool or 1))

    def resolved_advanced_rerank_timeout_seconds(self) -> float:
        """Return the timeout budget for a single reranking request."""
        return max(0.1, float(self.advanced_rerank_timeout_seconds or 0.1))

    def resolved_advanced_rerank_max_content_chars(self) -> int:
        """Return the content budget included per rerank candidate."""
        return max(100, int(self.advanced_rerank_max_content_chars or 100))

    def resolved_search_cache_signature(self, search_api: str) -> dict[str, Any]:
        """Return cache-isolating configuration knobs for the requested search mode."""
        normalized_api = str(search_api or "").strip().lower()
        if normalized_api != SearchAPI.ADVANCED.value:
            return {}

        return {
            "advanced_search_backends": self.resolved_advanced_search_backends(),
            "advanced_search_fetch_full_page_override": self.advanced_search_fetch_full_page_override,
            "advanced_rerank_enabled": bool(self.advanced_rerank_enabled),
            "advanced_rerank_model": self.resolved_advanced_rerank_model(),
            "advanced_rerank_candidate_pool": self.resolved_advanced_rerank_candidate_pool(),
        }

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
        return self._clamp((window // 128) * 4, 120, 1600)

    def resolved_report_sources_char_limit(self) -> int:
        """Resolve per-task source budget for the final report prompt."""
        if self.report_sources_char_limit is not None:
            return max(80, int(self.report_sources_char_limit))

        window = self.resolved_context_window()
        return self._clamp((window // 256) * 4, 80, 600)

    def resolved_report_layout_mode(self) -> str:
        """Return the normalized report layout mode."""
        return str(self.report_layout_mode or "flexible").strip().lower() or "flexible"

    def resolved_fast_llm_enabled(self) -> bool:
        """Return True when a dedicated fast-tier LLM model is configured."""
        return bool((self.fast_llm_model_id or "").strip())

    def resolved_fast_llm_model(self) -> str:
        """Return the fast-tier model id, falling back to the primary model."""
        explicit = (self.fast_llm_model_id or "").strip()
        if explicit:
            return explicit
        return (self.llm_model_id or self.local_llm or "").strip()

    def resolved_sandbox_effective_enabled(self) -> bool:
        """Require provisioning and explicit activation from the request's SkillPack."""
        return (
            self.sandbox_enabled
            and self.active_code_sandbox_enabled is True
            and self.sandbox_backend.strip().lower() in {"auto", "docker", "pyodide"}
        )

    def resolved_multimodal_vlm_base_url(self) -> str | None:
        """Return configured VLM base URL, falling back to primary LLM base URL if set."""
        explicit = (self.multimodal_vlm_base_url or "").strip()
        if explicit:
            return explicit
        fallback = (self.llm_base_url or "").strip()
        return fallback or None

    def resolved_multimodal_vlm_api_key(self) -> str | None:
        """Return configured VLM API key, falling back to primary LLM API key if set."""
        explicit = (self.multimodal_vlm_api_key or "").strip()
        if explicit:
            return explicit
        if (self.multimodal_vlm_base_url or "").strip().rstrip("/") != (self.llm_base_url or "").strip().rstrip("/"):
            if self.multimodal_vlm_base_url:
                return None
        fallback = (self.llm_api_key or "").strip()
        return fallback or None

    def resolved_policy_model_mode(self) -> str:
        """Resolve effective policy model mode honoring explicit mode and legacy flag."""
        mode = str(getattr(self, "policy_model_mode", "off") or "off").strip().lower()
        if mode != "off" or "policy_model_mode" in self.model_fields_set:
            return mode
        if bool(getattr(self, "policy_model_enabled", False)):
            return "active"
        return "off"
