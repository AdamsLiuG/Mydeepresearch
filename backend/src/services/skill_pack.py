"""Research skill packs: externalised, domain-specific SOPs injected as prompt fragments.

A skill pack carries *method* guidance (how to approach a class of research), never facts,
never tool directives, never source ids. Exactly one pack is selected per request; when nothing
matches confidently the neutral pack is used, so behaviour degrades cleanly to the built-in
prompts (``prompts.py``) instead of guessing with the wrong SOP.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from services.skill_verifiers import REGISTERED_VERIFIER_IDS
from services.text_processing import CITATION_PATTERN as _SOURCE_ID_PATTERN

logger = logging.getLogger(__name__)

NEUTRAL_PACK_ID = "_neutral"
CURRENT_SCHEMA_VERSION = 1
MAX_ACTIVE_PACKS = 9

SKILL_KIND_DOMAIN = "domain"
SKILL_KIND_FORMAT = "format"
SKILL_KINDS = {SKILL_KIND_DOMAIN, SKILL_KIND_FORMAT}

ALLOWED_RETRIEVAL_BACKENDS = {
    "searxng",
    "tavily",
    "serpapi",
    "duckduckgo",
    "perplexity",
    "semanticscholar",
}
RETRIEVAL_PROFILE_KEYS = {
    "preferred_backends",
    "backend_weights",
    "authoritative_hosts",
    "deprioritized_hosts",
    "prior_boost",
    "prior_penalty",
}
_HOST_HINT_PATTERN = re.compile(
    r"^\.?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)*$"
)
_IP_HINT_PATTERN = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_FORBIDDEN_HOST_TOKENS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", ".local", ".internal"}

# The skill stage vocabulary is defined here on purpose, independent of the RequestTrace stage
# literals. Report-repair reuses ``stage="reflection"`` in the trace, but a skill fragment for
# the reflection stage means something different; coupling the two vocabularies would make one
# rename silently break the other.
STAGE_PLANNING = "planning"
STAGE_TASK_EXECUTION = "task_execution"
STAGE_REFLECTION = "reflection"
STAGE_REVIEW = "review"
STAGE_REPORT = "report"
SKILL_STAGES = (
    STAGE_PLANNING,
    STAGE_TASK_EXECUTION,
    STAGE_REFLECTION,
    STAGE_REVIEW,
    STAGE_REPORT,
)

STATUS_ACTIVE = "active"
STATUS_SHADOW = "shadow"
STATUS_DISABLED = "disabled"
_STATUS_VALUES = {STATUS_ACTIVE, STATUS_SHADOW, STATUS_DISABLED}

MATURITY_DRAFT = "draft"
MATURITY_VERIFIED = "verified"
MATURITY_PROMOTED = "promoted"
_MATURITY_VALUES = {MATURITY_DRAFT, MATURITY_VERIFIED, MATURITY_PROMOTED}

# kebab-case identifiers; the neutral pack id is the one allowed underscore-prefixed exception.
_PACK_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Whitelisted override knobs. Only these config fields may be overridden by a pack. Quality gates
# carry a safe direction so the validator can reject reward-hacking (a candidate loosening the
# acceptance bar to score better). Directions: ``increase`` = stricter when larger, ``decrease``
# = stricter when smaller.
QUALITY_GATE_DIRECTIONS = {
    "review_min_sources_per_task": "increase",
    "review_min_domains_per_task": "increase",
    "freshness_reference_days": "decrease",
}
BUDGET_KNOBS = {
    "max_agent_tasks",
    "reflection_max_additional_tasks",
    "task_react_max_rounds",
    "task_react_max_fetches_per_task",
    "task_react_max_additional_searches_per_task",
    "report_repair_max_tasks",
    "report_repair_max_cycles",
}
LAYOUT_KNOBS = {"report_layout_mode", "search_api"}

_FRONTMATTER_DELIMITER = "---"
_TOOL_CALL_MARKER = "[TOOL_CALL:"
_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
# A run of >=4 digits (optionally with %/.,) reads as a concrete finding rather than method text.
_HARD_NUMBER_PATTERN = re.compile(r"\d[\d.,]{3,}%?")
_SECTION_HEADER_PATTERN = re.compile(r"^##\s+(?P<name>[A-Za-z_]+)\s*$")


class SkillPackError(Exception):
    """Raised when a skill pack file cannot be parsed into a well-formed pack."""


@dataclass
class SkillPack:
    """A parsed skill pack. ``body_sections`` maps a skill stage to its method fragment."""

    pack_id: str
    schema_version: int
    version: int
    maturity: str
    status: str
    title: str
    topic_domain: str
    applicable_when: str
    match_signals: list[str]
    negative_signals: list[str]
    min_match_score: float
    stage_scope: list[str]
    quality_gates: dict[str, Any]
    budget_knobs: dict[str, Any]
    layout: dict[str, Any]
    body_sections: dict[str, str]
    supersedes: str = ""
    derived_from: str = ""
    owner: str = ""
    eval_meta: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    pinned_content_hash: str = ""
    source_path: str = ""
    skill_kind: str = SKILL_KIND_DOMAIN
    retrieval_profile: dict[str, Any] = field(default_factory=dict)
    verifiers: list[str] = field(default_factory=list)
    composable_with: list[str] = field(default_factory=list)
    outline_sections: list[str] = field(default_factory=list)
    mcp_tool_scopes: dict[str, Any] = field(default_factory=dict)
    code_sandbox: bool = False

    @property
    def is_neutral(self) -> bool:
        return self.pack_id == NEUTRAL_PACK_ID

    @property
    def is_format(self) -> bool:
        return self.skill_kind == SKILL_KIND_FORMAT

    def fragment_for(self, stage: str) -> str:
        return str(self.body_sections.get(stage, "")).strip()


@dataclass
class SkillSelection:
    """Outcome of selecting a pack bundle for one request."""

    pack: SkillPack
    match_score: float
    fallback_reason: str = ""
    shadow: bool = False
    secondary_pack: SkillPack | None = None
    secondary_match_score: float = 0.0
    format_pack: SkillPack | None = None
    routing_method: str = "lexical"
    active_verifiers: list[str] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return bool(self.fallback_reason)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _split_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Split a ``---`` delimited YAML frontmatter block from the Markdown body.

    Unlike the flat parser in note_memory, skill packs need nested maps and lists, so the block
    is handed to ``yaml.safe_load``.
    """

    if not raw_text.startswith(f"{_FRONTMATTER_DELIMITER}\n"):
        raise SkillPackError("SKILL.md must start with a '---' frontmatter block")

    lines = raw_text.splitlines()
    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONTMATTER_DELIMITER:
            end_index = index
            break
    if end_index is None:
        raise SkillPackError("frontmatter block is not terminated by a closing '---'")

    frontmatter_text = "\n".join(lines[1:end_index])
    try:
        parsed = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise SkillPackError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SkillPackError("frontmatter must be a mapping")

    body = "\n".join(lines[end_index + 1 :]).strip()
    return parsed, body


def _split_body_sections(body: str) -> dict[str, str]:
    """Parse ``## <stage>`` headers into a stage -> fragment mapping, preserving order."""

    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    def _flush() -> None:
        if current is not None:
            sections[current] = "\n".join(buffer).strip()

    for line in body.splitlines():
        header = _SECTION_HEADER_PATTERN.match(line.strip())
        if header:
            _flush()
            current = header.group("name").strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    _flush()
    return sections


def compute_content_hash(pack: SkillPack) -> str:
    """Deterministic hash over the selection-relevant fields and body, independent of governance.

    Governance/lineage/timestamps are excluded so the hash reflects *what the pack does*, which
    is what ``(pack_id, version)`` immutability must protect.
    """

    canonical = {
        "pack_id": pack.pack_id,
        "title": pack.title,
        "topic_domain": pack.topic_domain,
        "applicable_when": pack.applicable_when,
        "match_signals": sorted(pack.match_signals),
        "negative_signals": sorted(pack.negative_signals),
        "min_match_score": round(float(pack.min_match_score), 4),
        "stage_scope": sorted(pack.stage_scope),
        "quality_gates": pack.quality_gates,
        "budget_knobs": pack.budget_knobs,
        "layout": pack.layout,
        "body_sections": {key: pack.body_sections[key] for key in sorted(pack.body_sections)},
    }
    if pack.skill_kind != SKILL_KIND_DOMAIN:
        canonical["skill_kind"] = pack.skill_kind
    if pack.retrieval_profile:
        canonical["retrieval_profile"] = pack.retrieval_profile
    if pack.verifiers:
        canonical["verifiers"] = sorted(pack.verifiers)
    if pack.composable_with:
        canonical["composable_with"] = sorted(pack.composable_with)
    if pack.outline_sections:
        canonical["outline_sections"] = list(pack.outline_sections)
    if pack.mcp_tool_scopes:
        canonical["mcp_tool_scopes"] = pack.mcp_tool_scopes
    if pack.code_sandbox:
        canonical["code_sandbox"] = True
    serialised = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:16]


def load_pack(path: Path) -> SkillPack:
    """Load and structurally parse a single ``SKILL.md`` (no policy validation here)."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillPackError(f"cannot read {path}: {exc}") from exc

    frontmatter, body = _split_frontmatter(raw_text)
    body_sections = _split_body_sections(body)

    try:
        schema_version = int(frontmatter.get("schema_version") or 0)
        version = int(frontmatter.get("version") or 0)
    except (TypeError, ValueError) as exc:
        raise SkillPackError(f"schema_version/version must be integers in {path}") from exc

    try:
        min_match_score = float(frontmatter.get("min_match_score", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise SkillPackError(f"min_match_score must be a number in {path}") from exc

    pack = SkillPack(
        pack_id=str(frontmatter.get("pack_id") or "").strip(),
        schema_version=schema_version,
        version=version,
        maturity=str(frontmatter.get("maturity") or MATURITY_DRAFT).strip().lower(),
        status=str(frontmatter.get("status") or STATUS_ACTIVE).strip().lower(),
        title=str(frontmatter.get("title") or "").strip(),
        topic_domain=str(frontmatter.get("topic_domain") or "").strip().lower(),
        applicable_when=str(frontmatter.get("applicable_when") or "").strip(),
        match_signals=_as_list(frontmatter.get("match_signals")),
        negative_signals=_as_list(frontmatter.get("negative_signals")),
        min_match_score=min_match_score,
        stage_scope=_as_list(frontmatter.get("stage_scope")),
        quality_gates=_as_dict(frontmatter.get("quality_gates")),
        budget_knobs=_as_dict(frontmatter.get("budget_knobs")),
        layout=_as_dict(frontmatter.get("layout")),
        body_sections=body_sections,
        supersedes=str(frontmatter.get("supersedes") or "").strip(),
        derived_from=str(frontmatter.get("derived_from") or "").strip(),
        owner=str(frontmatter.get("owner") or "").strip(),
        eval_meta=_as_dict(frontmatter.get("eval")),
        pinned_content_hash=str(frontmatter.get("content_hash") or "").strip(),
        source_path=str(path),
        skill_kind=str(frontmatter.get("skill_kind") or SKILL_KIND_DOMAIN).strip().lower(),
        retrieval_profile=_as_dict(frontmatter.get("retrieval_profile")),
        verifiers=_as_list(frontmatter.get("verifiers")),
        composable_with=_as_list(frontmatter.get("composable_with")),
        outline_sections=_as_list(frontmatter.get("outline_sections")),
        mcp_tool_scopes=_as_dict(frontmatter.get("mcp_tool_scopes")),
        code_sandbox=bool(frontmatter.get("code_sandbox", False)),
    )
    pack.content_hash = compute_content_hash(pack)
    return pack


def validate_pack(
    pack: SkillPack,
    *,
    char_limit: int,
    directory_name: str | None = None,
) -> list[str]:
    """Return a list of validation errors for one pack (empty list means valid).

    The rules are grouped identity / structure / safety / governance and are exercised
    one-to-one by the validator unit tests.
    """

    errors: list[str] = []
    is_neutral = pack.pack_id == NEUTRAL_PACK_ID

    # --- identity ---
    if directory_name is not None and pack.pack_id != directory_name:
        errors.append(f"pack_id '{pack.pack_id}' does not match directory '{directory_name}'")
    if is_neutral:
        if pack.pack_id != NEUTRAL_PACK_ID:
            errors.append("neutral pack must use the reserved id '_neutral'")
    elif not _PACK_ID_PATTERN.match(pack.pack_id):
        errors.append(f"pack_id '{pack.pack_id}' must be kebab-case")
    if pack.schema_version != CURRENT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CURRENT_SCHEMA_VERSION}")
    if pack.version < 1:
        errors.append("version must be a positive integer")
    if pack.pinned_content_hash and pack.pinned_content_hash != pack.content_hash:
        errors.append("content_hash does not match the computed hash of the pack content")
    if pack.status not in _STATUS_VALUES:
        errors.append(f"status '{pack.status}' is not one of {sorted(_STATUS_VALUES)}")
    if pack.maturity not in _MATURITY_VALUES:
        errors.append(f"maturity '{pack.maturity}' is not one of {sorted(_MATURITY_VALUES)}")
    if pack.skill_kind not in SKILL_KINDS:
        errors.append(f"skill_kind '{pack.skill_kind}' is not one of {sorted(SKILL_KINDS)}")

    # --- structure ---
    unknown_stages = [stage for stage in pack.stage_scope if stage not in SKILL_STAGES]
    if unknown_stages:
        errors.append(f"stage_scope contains unknown stages: {unknown_stages}")
    if set(pack.body_sections) != set(pack.stage_scope):
        errors.append(
            "body sections "
            f"{sorted(pack.body_sections)} must equal stage_scope {sorted(pack.stage_scope)}"
        )
    if pack.is_format and (not pack.stage_scope or set(pack.stage_scope) != {STAGE_REPORT}):
        errors.append("format skill pack stage_scope must equal ['report']")
    for stage, fragment in pack.body_sections.items():
        if len(fragment) > char_limit:
            errors.append(f"stage '{stage}' fragment exceeds char limit {char_limit}")

    # --- safety ---
    errors.extend(_validate_quality_gates(pack))
    errors.extend(_validate_budget_and_layout(pack))
    errors.extend(_validate_retrieval_profile(pack))
    errors.extend(_validate_verifiers(pack))
    errors.extend(_validate_composable_with(pack))
    errors.extend(_validate_mcp_tool_scopes(pack))
    for stage, fragment in pack.body_sections.items():
        errors.extend(_validate_fragment_safety(stage, fragment))

    # --- governance ---
    overlap = sorted(set(s.lower() for s in pack.match_signals) & set(s.lower() for s in pack.negative_signals))
    if overlap:
        errors.append(f"match_signals and negative_signals overlap: {overlap}")
    if is_neutral:
        if (
            pack.quality_gates
            or pack.budget_knobs
            or pack.layout
            or pack.retrieval_profile
            or pack.verifiers
            or pack.composable_with
            or pack.mcp_tool_scopes
            or pack.code_sandbox
        ):
            errors.append("neutral pack must not override any knobs")
        if pack.match_signals:
            errors.append("neutral pack must not declare match_signals")
    if pack.is_format and (
        pack.quality_gates
        or pack.budget_knobs
        or pack.retrieval_profile
        or pack.verifiers
        or pack.composable_with
        or pack.mcp_tool_scopes
        or pack.code_sandbox
    ):
        errors.append("format skill pack must not declare domain quality/budget/retrieval/verifier knobs")
    if pack.maturity == MATURITY_PROMOTED and _as_dict(pack.eval_meta).get("verdict") != "pass":
        errors.append("promoted pack requires an eval with verdict=pass")

    return errors


def _validate_quality_gates(pack: SkillPack) -> list[str]:
    errors: list[str] = []
    for key in pack.quality_gates:
        if key not in QUALITY_GATE_DIRECTIONS:
            errors.append(f"quality_gates key '{key}' is not in the override whitelist")
    return errors


def _validate_budget_and_layout(pack: SkillPack) -> list[str]:
    errors: list[str] = []
    for key in pack.budget_knobs:
        if key not in BUDGET_KNOBS:
            errors.append(f"budget_knobs key '{key}' is not in the override whitelist")
    for key in pack.layout:
        if key not in LAYOUT_KNOBS:
            errors.append(f"layout key '{key}' is not in the override whitelist")
    return errors


def _validate_mcp_tool_scopes(pack: SkillPack) -> list[str]:
    scopes = pack.mcp_tool_scopes
    if not scopes:
        return []
    errors: list[str] = []
    allowed_keys = {
        "allowed_servers",
        "allowed_tools",
        "auto_query_tools",
        "stage_allowlist",
        "readonly_only",
        "max_tools_injected",
    }
    for key in scopes:
        if key not in allowed_keys:
            errors.append(f"mcp_tool_scopes key '{key}' is not in the allowed whitelist")
    for list_key in ("allowed_servers", "allowed_tools", "auto_query_tools"):
        raw = scopes.get(list_key)
        if raw is None:
            continue
        if not isinstance(raw, list):
            errors.append(f"mcp_tool_scopes.{list_key} must be a list")
            continue
        for item in raw:
            tok = str(item or "").strip()
            if "*" in tok:
                errors.append(f"mcp_tool_scopes.{list_key} must not contain wildcard '*'")
    if "max_tools_injected" in scopes:
        try:
            cap = int(scopes["max_tools_injected"])
            if cap < 0 or cap > 12:
                errors.append("mcp_tool_scopes.max_tools_injected must be in [0, 12]")
        except (TypeError, ValueError):
            errors.append("mcp_tool_scopes.max_tools_injected must be an integer")
    return errors


def _validate_host_hint_list(hosts: Any, field_name: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    normalized: list[str] = []
    if not isinstance(hosts, list):
        return [f"retrieval_profile.{field_name} must be a list"], normalized
    for raw in hosts:
        host = str(raw or "").strip().lower()
        if not host:
            continue
        if (
            "://" in host
            or "/" in host
            or ":" in host
            or _IP_HINT_PATTERN.match(host.lstrip("."))
            or any(bad in host for bad in _FORBIDDEN_HOST_TOKENS)
            or not _HOST_HINT_PATTERN.match(host)
        ):
            errors.append(f"retrieval_profile.{field_name} contains unsafe/invalid host hint {raw!r}")
            continue
        normalized.append(host)
    return errors, normalized


def _validate_retrieval_profile(pack: SkillPack) -> list[str]:
    profile = pack.retrieval_profile
    if not profile:
        return []
    errors: list[str] = []
    for key in profile:
        if key not in RETRIEVAL_PROFILE_KEYS:
            errors.append(f"retrieval_profile key '{key}' is not in the override whitelist")

    if "preferred_backends" in profile:
        pref = profile["preferred_backends"]
        if not isinstance(pref, list) or not pref:
            errors.append("retrieval_profile.preferred_backends must be a non-empty list")
        else:
            for backend in pref:
                b = str(backend or "").strip().lower()
                if b not in ALLOWED_RETRIEVAL_BACKENDS:
                    errors.append(f"retrieval_profile.preferred_backends has unsupported backend {backend!r}")

    if "backend_weights" in profile:
        weights = profile["backend_weights"]
        if not isinstance(weights, dict):
            errors.append("retrieval_profile.backend_weights must be a mapping")
        else:
            for backend, weight in weights.items():
                b = str(backend or "").strip().lower()
                if b not in ALLOWED_RETRIEVAL_BACKENDS:
                    errors.append(f"retrieval_profile.backend_weights has unsupported backend {backend!r}")
                    continue
                try:
                    w = float(weight)
                except (TypeError, ValueError):
                    errors.append(f"retrieval_profile.backend_weights[{backend!r}] must be numeric")
                    continue
                if not (0.2 <= w <= 2.0):
                    errors.append(f"retrieval_profile.backend_weights[{backend!r}] must be in [0.2, 2.0]")

    auth_hosts: list[str] = []
    dep_hosts: list[str] = []
    if "authoritative_hosts" in profile:
        host_errs, auth_hosts = _validate_host_hint_list(profile["authoritative_hosts"], "authoritative_hosts")
        errors.extend(host_errs)
    if "deprioritized_hosts" in profile:
        host_errs, dep_hosts = _validate_host_hint_list(profile["deprioritized_hosts"], "deprioritized_hosts")
        errors.extend(host_errs)
    overlap = sorted(set(auth_hosts) & set(dep_hosts))
    if overlap:
        errors.append(f"retrieval_profile authoritative/deprioritized hosts overlap: {overlap}")

    if "prior_boost" in profile:
        try:
            boost = float(profile["prior_boost"])
            if not (0.0 <= boost <= 0.08):
                errors.append("retrieval_profile.prior_boost must be in [0.0, 0.08]")
        except (TypeError, ValueError):
            errors.append("retrieval_profile.prior_boost must be numeric")
    if "prior_penalty" in profile:
        try:
            penalty = float(profile["prior_penalty"])
            if not (-0.08 <= penalty <= 0.0):
                errors.append("retrieval_profile.prior_penalty must be in [-0.08, 0.0]")
        except (TypeError, ValueError):
            errors.append("retrieval_profile.prior_penalty must be numeric")
    return errors


def _validate_verifiers(pack: SkillPack) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for verifier_id in pack.verifiers:
        vid = str(verifier_id or "").strip().lower()
        if vid not in REGISTERED_VERIFIER_IDS:
            errors.append(f"verifiers contains unknown id '{verifier_id}'")
            continue
        if vid in seen:
            errors.append(f"verifiers contains duplicate id '{vid}'")
        seen.add(vid)
    return errors


def _validate_composable_with(pack: SkillPack) -> list[str]:
    errors: list[str] = []
    for target in pack.composable_with:
        tid = str(target or "").strip()
        if tid == pack.pack_id:
            errors.append("composable_with must not reference pack itself")
        elif tid == NEUTRAL_PACK_ID:
            errors.append("composable_with must not reference '_neutral'")
        elif not _PACK_ID_PATTERN.match(tid):
            errors.append(f"composable_with entry '{tid}' must be kebab-case")
    return errors


def _validate_fragment_safety(stage: str, fragment: str) -> list[str]:
    """A skill fragment is method text; it must not smuggle facts, tools, ids or citations."""

    errors: list[str] = []
    if _TOOL_CALL_MARKER in fragment:
        errors.append(f"stage '{stage}' fragment must not contain a tool directive")
    if _SOURCE_ID_PATTERN.search(fragment):
        errors.append(f"stage '{stage}' fragment must not contain a source id")
    if _URL_PATTERN.search(fragment):
        errors.append(f"stage '{stage}' fragment must not contain a URL")
    if _HARD_NUMBER_PATTERN.search(fragment):
        errors.append(f"stage '{stage}' fragment must not contain a concrete numeric finding")
    return errors


def enforce_quality_gate_direction(field_name: str, baseline: Any, override: Any) -> bool:
    """Return True when ``override`` only tightens the gate relative to ``baseline``."""

    direction = QUALITY_GATE_DIRECTIONS.get(field_name)
    if direction is None:
        return False
    try:
        baseline_value = float(baseline)
        override_value = float(override)
    except (TypeError, ValueError):
        return False
    if direction == "increase":
        return override_value >= baseline_value
    return override_value <= baseline_value


def _merge_verifier_ids(*pack_lists: SkillPack | None) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for pack in pack_lists:
        if pack is None or getattr(pack, "status", STATUS_ACTIVE) == STATUS_SHADOW:
            continue
        for vid in getattr(pack, "verifiers", []) or ():
            normalized = str(vid or "").strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
    return merged


def apply_skill_pack_overrides(
    base_config: Any,
    pack: SkillPack,
    *,
    secondary_pack: SkillPack | None = None,
    format_pack: SkillPack | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply whitelisted pack overrides to a request-scoped config copy.

    Quality gates are strictly tightening-only (verified via ``enforce_quality_gate_direction``)
    so a pack (or secondary constraint pack) can never weaken review/freshness bars.
    """

    if base_config is None:
        return base_config, {}

    updates: dict[str, Any] = {}
    knob_updates: dict[str, Any] = {}

    if not pack.is_neutral:
        source_packs = [pack]
        if (
            secondary_pack is not None
            and not secondary_pack.is_neutral
            and secondary_pack.status != STATUS_SHADOW
        ):
            source_packs.append(secondary_pack)

        for current_pack in source_packs:
            for key, value in (current_pack.quality_gates or {}).items():
                if key not in QUALITY_GATE_DIRECTIONS or not hasattr(base_config, key):
                    continue
                current_baseline = knob_updates.get(key, getattr(base_config, key))
                if not enforce_quality_gate_direction(key, current_baseline, value):
                    if current_pack is pack:
                        logger.warning(
                            "skill pack '%s' quality_gate override rejected (loosens baseline): %s=%r (baseline=%r)",
                            current_pack.pack_id,
                            key,
                            value,
                            current_baseline,
                        )
                    continue
                knob_updates[key] = value
                updates[key] = value

        for key, value in (pack.budget_knobs or {}).items():
            if key not in BUDGET_KNOBS or not hasattr(base_config, key):
                continue
            knob_updates[key] = value
            updates[key] = value

        for key, value in (pack.layout or {}).items():
            if key not in LAYOUT_KNOBS or not hasattr(base_config, key):
                continue
            if key == "search_api" and isinstance(value, str):
                try:
                    from config import SearchAPI

                    resolved_api = SearchAPI(value.strip().lower())
                    knob_updates[key] = resolved_api
                    updates[key] = resolved_api
                except Exception:
                    logger.warning("skill pack '%s' invalid search_api layout override: %r", pack.pack_id, value)
                    continue
            else:
                knob_updates[key] = value
                updates[key] = value

        if pack.retrieval_profile and hasattr(base_config, "active_retrieval_profile"):
            updates["active_retrieval_profile"] = deepcopy(pack.retrieval_profile)

        merged_verifiers = _merge_verifier_ids(pack, secondary_pack)
        if merged_verifiers and hasattr(base_config, "active_skill_verifiers"):
            updates["active_skill_verifiers"] = merged_verifiers

        merged_scopes: dict[str, Any] = {}
        for current_pack in source_packs:
            raw_scopes = getattr(current_pack, "mcp_tool_scopes", None) or {}
            if not raw_scopes:
                continue
            if not merged_scopes:
                merged_scopes = deepcopy(raw_scopes)
            else:
                for list_key in ("allowed_servers", "allowed_tools", "auto_query_tools"):
                    existing_vals = list(merged_scopes.get(list_key) or [])
                    for item in raw_scopes.get(list_key) or ():
                        if item not in existing_vals:
                            existing_vals.append(item)
                    if existing_vals:
                        merged_scopes[list_key] = existing_vals
        if merged_scopes and hasattr(base_config, "active_mcp_tool_scopes"):
            updates["active_mcp_tool_scopes"] = merged_scopes

        sandbox_enabled = any(bool(getattr(p, "code_sandbox", False)) for p in source_packs)
        if hasattr(base_config, "active_code_sandbox_enabled"):
            updates["active_code_sandbox_enabled"] = sandbox_enabled

    if (
        format_pack is not None
        and format_pack.status != STATUS_SHADOW
        and hasattr(base_config, "active_output_format_id")
    ):
        updates["active_output_format_id"] = format_pack.pack_id

    if not updates:
        return base_config, {}

    try:
        updated_config = base_config.model_copy(update=updates)
    except Exception as exc:
        logger.warning("skill pack '%s' config override failed: %s", pack.pack_id, exc)
        return base_config, {}

    applied_summary = {
        key: (val.value if hasattr(val, "value") else val)
        for key, val in knob_updates.items()
    }
    return updated_config, applied_summary


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class SkillPackService:
    """Load, validate and select research skill packs from the configured directory."""

    def __init__(
        self,
        config: Any,
        *,
        llm_router: Callable[[str, list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    ) -> None:
        self._config = config
        self._packs_dir = Path(getattr(config, "skill_pack_dir", "./skills/research"))
        self._formats_dir = Path(getattr(config, "skill_format_dir", "./skills/formats"))
        self._packs: dict[str, SkillPack] | None = None
        self._formats: dict[str, SkillPack] | None = None
        self._llm_router = llm_router
        self._semantic_pack_vectors: dict[tuple[str, str], list[float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._config, "skill_pack_enabled", False))

    @property
    def char_limit(self) -> int:
        return int(getattr(self._config, "skill_prompt_char_limit", 1200) or 1200)

    @property
    def global_min_match_score(self) -> float:
        return float(getattr(self._config, "skill_pack_min_match_score", 0.30) or 0.0)

    @property
    def composable_enabled(self) -> bool:
        return bool(getattr(self._config, "skill_pack_composable_enabled", True))

    @property
    def secondary_min_match_score(self) -> float:
        return float(getattr(self._config, "skill_pack_secondary_min_match_score", 0.50) or 0.50)

    @property
    def semantic_routing_enabled(self) -> bool:
        return bool(getattr(self._config, "skill_pack_semantic_routing_enabled", False))

    @property
    def llm_router_enabled(self) -> bool:
        return bool(getattr(self._config, "skill_pack_llm_router_enabled", False))

    def _load_directory_packs(
        self,
        directory: Path,
        *,
        expected_kind: str | None = None,
    ) -> dict[str, SkillPack]:
        packs: dict[str, SkillPack] = {}
        if directory.is_dir():
            for entry in sorted(directory.iterdir()):
                if not entry.is_dir():
                    continue
                skill_file = entry / "SKILL.md"
                if not skill_file.is_file():
                    continue
                try:
                    pack = load_pack(skill_file)
                    if expected_kind is not None and pack.skill_kind != expected_kind:
                        logger.warning(
                            "skill pack kind mismatch dir=%s expected=%s got=%s",
                            entry.name,
                            expected_kind,
                            pack.skill_kind,
                        )
                        continue
                    errors = validate_pack(
                        pack,
                        char_limit=self.char_limit,
                        directory_name=entry.name,
                    )
                except SkillPackError as exc:
                    logger.warning("skill pack load failed dir=%s error=%s", entry.name, exc)
                    continue
                if errors:
                    logger.warning("skill pack invalid dir=%s errors=%s", entry.name, errors)
                    continue
                if pack.status == STATUS_DISABLED:
                    continue
                packs[pack.pack_id] = pack
        return packs

    def _load_all(self) -> dict[str, SkillPack]:
        if self._packs is not None:
            return self._packs
        self._packs = self._load_directory_packs(self._packs_dir)
        return self._packs

    def _load_formats(self) -> dict[str, SkillPack]:
        if self._formats is not None:
            return self._formats
        self._formats = self._load_directory_packs(
            self._formats_dir,
            expected_kind=SKILL_KIND_FORMAT,
        )
        return self._formats

    def neutral_pack(self) -> SkillPack | None:
        return self._load_all().get(NEUTRAL_PACK_ID)

    def select_format(
        self,
        topic: str,
        *,
        signals: list[str] | None = None,
        output_format: str | None = None,
    ) -> SkillPack | None:
        formats = self._load_formats()
        if not formats:
            return None

        requested = str(
            output_format
            or getattr(self._config, "default_output_format", "")
            or ""
        ).strip().lower()
        if requested:
            return formats.get(requested)

        haystack = " ".join(
            part.lower()
            for part in [topic or "", " ".join(signals or [])]
            if part and part.strip()
        )
        best: SkillPack | None = None
        best_score = 0.0
        for pack in formats.values():
            score, disqualified = self._score(pack, haystack)
            if disqualified:
                continue
            threshold = max(pack.min_match_score, self.global_min_match_score)
            if score >= threshold and score > best_score:
                best = pack
                best_score = score
        return best

    def _select_secondary(
        self,
        primary: SkillPack,
        packs: dict[str, SkillPack],
        haystack: str,
    ) -> tuple[SkillPack | None, float]:
        if (
            not self.composable_enabled
            or primary.is_neutral
            or primary.status == STATUS_SHADOW
            or not primary.composable_with
        ):
            return None, 0.0

        best_sec: SkillPack | None = None
        best_score = 0.0
        threshold = max(0.0, self.secondary_min_match_score)
        for target_id in primary.composable_with:
            candidate = packs.get(target_id)
            if (
                candidate is None
                or candidate.is_neutral
                or candidate.status == STATUS_SHADOW
                or candidate.pack_id == primary.pack_id
            ):
                continue
            score = self._positive_signal_score(candidate, haystack)
            if score >= threshold and score > best_score:
                best_sec = candidate
                best_score = score
        return best_sec, round(best_score, 4) if best_sec is not None else 0.0

    def _semantic_fallback_select(
        self,
        topic: str,
        candidates: list[SkillPack],
    ) -> tuple[SkillPack | None, float]:
        if not candidates or not str(topic or "").strip():
            return None, 0.0
        try:
            from services.embeddings import encode_text
        except Exception:
            return None, 0.0

        topic_vec = encode_text(topic)
        if not topic_vec:
            return None, 0.0

        best: SkillPack | None = None
        best_score = 0.0
        for pack in candidates:
            cache_key = (pack.pack_id, pack.content_hash)
            pack_vec = self._semantic_pack_vectors.get(cache_key)
            if pack_vec is None:
                descriptor = " ".join(
                    part for part in (pack.title, pack.applicable_when, " ".join(pack.match_signals)) if part
                ).strip()
                if not descriptor:
                    continue
                pack_vec = encode_text(descriptor) or []
                self._semantic_pack_vectors[cache_key] = pack_vec
            if not pack_vec:
                continue
            score = round(0.75 * _cosine_similarity(topic_vec, pack_vec), 4)
            threshold = max(pack.min_match_score, self.global_min_match_score)
            if score >= threshold and score > best_score:
                best = pack
                best_score = score
        return best, best_score

    def _llm_tiebreak_select(
        self,
        topic: str,
        candidates: list[SkillPack],
    ) -> tuple[SkillPack | None, float]:
        if self._llm_router is None or not candidates:
            return None, 0.0
        candidate_meta = [
            {
                "pack_id": pack.pack_id,
                "title": pack.title,
                "applicable_when": pack.applicable_when,
            }
            for pack in candidates
        ]
        try:
            decision = self._llm_router(topic, candidate_meta)
        except Exception as exc:
            logger.warning("skill pack LLM router failed: %s", exc)
            return None, 0.0
        if not isinstance(decision, dict):
            return None, 0.0
        chosen_id = str(decision.get("pack_id") or "").strip()
        by_id = {pack.pack_id: pack for pack in candidates}
        chosen = by_id.get(chosen_id)
        if chosen is None:
            return None, 0.0
        raw_conf = decision.get("confidence", chosen.min_match_score or self.global_min_match_score)
        try:
            score = round(max(0.0, min(1.0, float(raw_conf))), 4)
        except (TypeError, ValueError):
            score = round(max(chosen.min_match_score, self.global_min_match_score), 4)
        return chosen, score

    def select(
        self,
        topic: str,
        *,
        signals: list[str] | None = None,
        output_format: str | None = None,
    ) -> SkillSelection | None:
        """Choose domain pack bundle (+ optional secondary & format pack) for one request."""

        packs = self._load_all()
        neutral = packs.get(NEUTRAL_PACK_ID)
        haystack = " ".join(
            part.lower()
            for part in [topic or "", " ".join(signals or [])]
            if part and part.strip()
        )

        best: SkillPack | None = None
        best_score = 0.0
        best_hits = 0
        non_disqualified: list[tuple[SkillPack, float]] = []
        eligible_packs: list[SkillPack] = []

        for pack in packs.values():
            if pack.is_neutral or pack.is_format:
                continue
            score, disqualified = self._score(pack, haystack)
            if disqualified:
                continue
            hits = self._positive_hit_count(pack, haystack)
            eligible_packs.append(pack)
            non_disqualified.append((pack, score))
            threshold = max(pack.min_match_score, self.global_min_match_score)
            if score >= threshold and (score, hits) > (best_score, best_hits):
                best = pack
                best_score = score
                best_hits = hits

        routing_method = "lexical"
        format_pack = self.select_format(topic, signals=signals, output_format=output_format)

        if (
            best is not None
            and self.llm_router_enabled
            and self._llm_router is not None
            and len(non_disqualified) >= 2
        ):
            sorted_candidates = sorted(non_disqualified, key=lambda item: item[1], reverse=True)
            top_pack, top_score = sorted_candidates[0]
            runner_up_pack, runner_up_score = sorted_candidates[1]
            runner_threshold = max(runner_up_pack.min_match_score, self.global_min_match_score)
            if runner_up_score >= runner_threshold and (top_score - runner_up_score) < 0.05:
                chosen, chosen_score = self._llm_tiebreak_select(
                    topic,
                    [top_pack, runner_up_pack],
                )
                if chosen is not None:
                    best = chosen
                    best_score = chosen_score
                    routing_method = "llm_tiebreak"

        if best is None and self.semantic_routing_enabled:
            sem_pack, sem_score = self._semantic_fallback_select(topic, eligible_packs)
            if sem_pack is not None:
                best = sem_pack
                best_score = sem_score
                routing_method = "hybrid_semantic"

        if best is None and self.llm_router_enabled and self._llm_router is not None:
            llm_pack, llm_score = self._llm_tiebreak_select(topic, eligible_packs)
            if llm_pack is not None:
                best = llm_pack
                best_score = llm_score
                routing_method = "llm_tiebreak"

        if best is not None:
            secondary_pack, secondary_score = self._select_secondary(best, packs, haystack)
            active_verifiers = _merge_verifier_ids(best, secondary_pack) if best.status != STATUS_SHADOW else []
            return SkillSelection(
                pack=best,
                match_score=round(best_score, 4),
                shadow=best.status == STATUS_SHADOW,
                secondary_pack=secondary_pack,
                secondary_match_score=secondary_score,
                format_pack=format_pack,
                routing_method=routing_method,
                active_verifiers=active_verifiers,
            )

        if neutral is None:
            return None
        return SkillSelection(
            pack=neutral,
            match_score=0.0,
            fallback_reason="no_pack_met_min_match_score",
            shadow=neutral.status == STATUS_SHADOW,
            format_pack=format_pack,
            routing_method=routing_method,
            active_verifiers=[],
        )

    def _positive_hit_count(self, pack: SkillPack, haystack: str) -> int:
        if not pack.match_signals:
            return 0
        return sum(1 for sig in pack.match_signals if sig.lower() in haystack)

    def _positive_signal_score(self, pack: SkillPack, haystack: str) -> float:
        if not pack.match_signals:
            return 0.0
        hits = self._positive_hit_count(pack, haystack)
        saturation_denom = min(max(1, len(pack.match_signals)), 3)
        return min(1.0, hits / saturation_denom)

    def _score(self, pack: SkillPack, haystack: str) -> tuple[float, bool]:
        if not pack.match_signals:
            return 0.0, False
        if any(sig.lower() in haystack for sig in pack.negative_signals):
            return 0.0, True
        lexical_score = self._positive_signal_score(pack, haystack)
        if lexical_score > 0.0:
            return lexical_score, False
        return 0.0, False

    def render_block(self, pack: SkillPack, stage: str) -> str:
        """Render the ``<RESEARCH_SKILL>`` block for one stage, or empty when there is nothing."""

        fragment = pack.fragment_for(stage)
        if not fragment:
            return ""
        return (
            f'\n\n<RESEARCH_SKILL pack="{pack.pack_id}" version="{pack.version}">\n'
            "以下是针对当前研究类型的方法约定，只用于指导你的做法、检查点与取舍，"
            "不是当前主题事实，不包含证据，也不能作为 source_id 引用。\n"
            f"{fragment}\n"
            "</RESEARCH_SKILL>\n"
        )

    def render_secondary_block(self, pack: SkillPack | None, stage: str) -> str:
        """Render a compact secondary constraint fragment for review/report stages only."""

        if pack is None or pack.is_neutral or stage not in {STAGE_REVIEW, STAGE_REPORT}:
            return ""
        fragment = pack.fragment_for(stage)
        if not fragment:
            return ""
        max_len = max(120, self.char_limit // 2)
        if len(fragment) > max_len:
            fragment = fragment[:max_len].rstrip()
        return (
            f'\n\n<SECONDARY_RESEARCH_SKILL pack="{pack.pack_id}" version="{pack.version}">\n'
            "以下是补充的跨领域合规/约束检查要点，仅用于补充审核与呈现边界，"
            "不是当前主题事实，不能作为 source_id 引用。\n"
            f"{fragment}\n"
            "</SECONDARY_RESEARCH_SKILL>\n"
        )

    def render_format_block(self, format_pack: SkillPack | None, stage: str) -> str:
        """Render the orthogonal ``<OUTPUT_FORMAT_SKILL>`` block for the report stage only."""

        if format_pack is None or stage != STAGE_REPORT:
            return ""
        fragment = format_pack.fragment_for(STAGE_REPORT)
        if not fragment:
            return ""
        return (
            f'\n\n<OUTPUT_FORMAT_SKILL format="{format_pack.pack_id}" version="{format_pack.version}">\n'
            "以下是最终报告的交付形态与版式要求，请在严格保留全部事实引用规范的前提下遵照组织结构。\n"
            f"{fragment}\n"
            "</OUTPUT_FORMAT_SKILL>\n"
        )

    def list_catalog(self) -> dict[str, Any]:
        """Return available domain packs, output format packs, and registered verifier IDs."""

        domain_packs = [
            {
                "pack_id": pack.pack_id,
                "title": pack.title,
                "topic_domain": pack.topic_domain,
                "applicable_when": pack.applicable_when,
                "verifiers": list(pack.verifiers),
                "composable_with": list(pack.composable_with),
                "code_sandbox": bool(pack.code_sandbox),
                "version": pack.version,
                "status": pack.status,
            }
            for pack in self._load_all().values()
        ]
        format_packs = [
            {
                "pack_id": pack.pack_id,
                "title": pack.title,
                "applicable_when": pack.applicable_when,
                "outline_sections": list(pack.outline_sections),
                "version": pack.version,
                "status": pack.status,
            }
            for pack in self._load_formats().values()
        ]
        return {
            "enabled": self.enabled,
            "domain_packs": domain_packs,
            "format_packs": format_packs,
            "registered_verifiers": list(REGISTERED_VERIFIER_IDS),
        }
