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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

NEUTRAL_PACK_ID = "_neutral"
CURRENT_SCHEMA_VERSION = 1
MAX_ACTIVE_PACKS = 9

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
_SOURCE_ID_PATTERN = re.compile(r"\[T\d+-S\d+\]")
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

    @property
    def is_neutral(self) -> bool:
        return self.pack_id == NEUTRAL_PACK_ID

    def fragment_for(self, stage: str) -> str:
        return str(self.body_sections.get(stage, "")).strip()


@dataclass
class SkillSelection:
    """Outcome of selecting a pack for one request."""

    pack: SkillPack
    match_score: float
    fallback_reason: str = ""
    shadow: bool = False

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

    # --- structure ---
    unknown_stages = [stage for stage in pack.stage_scope if stage not in SKILL_STAGES]
    if unknown_stages:
        errors.append(f"stage_scope contains unknown stages: {unknown_stages}")
    if set(pack.body_sections) != set(pack.stage_scope):
        errors.append(
            "body sections "
            f"{sorted(pack.body_sections)} must equal stage_scope {sorted(pack.stage_scope)}"
        )
    for stage, fragment in pack.body_sections.items():
        if len(fragment) > char_limit:
            errors.append(f"stage '{stage}' fragment exceeds char limit {char_limit}")

    # --- safety ---
    errors.extend(_validate_quality_gates(pack))
    errors.extend(_validate_budget_and_layout(pack))
    for stage, fragment in pack.body_sections.items():
        errors.extend(_validate_fragment_safety(stage, fragment))

    # --- governance ---
    overlap = sorted(set(s.lower() for s in pack.match_signals) & set(s.lower() for s in pack.negative_signals))
    if overlap:
        errors.append(f"match_signals and negative_signals overlap: {overlap}")
    if is_neutral:
        if pack.quality_gates or pack.budget_knobs or pack.layout:
            errors.append("neutral pack must not override any knobs")
        if pack.match_signals:
            errors.append("neutral pack must not declare match_signals")
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


def apply_skill_pack_overrides(
    base_config: Any,
    pack: SkillPack,
) -> tuple[Any, dict[str, Any]]:
    """Apply whitelisted pack overrides to a request-scoped config copy.

    Quality gates are strictly tightening-only (verified via ``enforce_quality_gate_direction``)
    so a pack can never weaken review/freshness bars.
    """

    if pack.is_neutral or base_config is None:
        return base_config, {}

    updates: dict[str, Any] = {}

    for key, value in (pack.quality_gates or {}).items():
        if key not in QUALITY_GATE_DIRECTIONS or not hasattr(base_config, key):
            continue
        baseline = getattr(base_config, key)
        if not enforce_quality_gate_direction(key, baseline, value):
            logger.warning(
                "skill pack '%s' quality_gate override rejected (loosens baseline): %s=%r (baseline=%r)",
                pack.pack_id,
                key,
                value,
                baseline,
            )
            continue
        updates[key] = value

    for key, value in (pack.budget_knobs or {}).items():
        if key not in BUDGET_KNOBS or not hasattr(base_config, key):
            continue
        updates[key] = value

    for key, value in (pack.layout or {}).items():
        if key not in LAYOUT_KNOBS or not hasattr(base_config, key):
            continue
        if key == "search_api" and isinstance(value, str):
            try:
                from config import SearchAPI

                updates[key] = SearchAPI(value.strip().lower())
            except Exception:
                logger.warning("skill pack '%s' invalid search_api layout override: %r", pack.pack_id, value)
                continue
        else:
            updates[key] = value

    if not updates:
        return base_config, {}

    try:
        updated_config = base_config.model_copy(update=updates)
    except Exception as exc:
        logger.warning("skill pack '%s' config override failed: %s", pack.pack_id, exc)
        return base_config, {}

    applied_summary = {
        key: (val.value if hasattr(val, "value") else val)
        for key, val in updates.items()
    }
    return updated_config, applied_summary


class SkillPackService:
    """Load, validate and select research skill packs from the configured directory."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._packs_dir = Path(getattr(config, "skill_pack_dir", "./skills/research"))
        self._packs: dict[str, SkillPack] | None = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._config, "skill_pack_enabled", False))

    @property
    def char_limit(self) -> int:
        return int(getattr(self._config, "skill_prompt_char_limit", 1200) or 1200)

    @property
    def global_min_match_score(self) -> float:
        return float(getattr(self._config, "skill_pack_min_match_score", 0.30) or 0.0)

    def _load_all(self) -> dict[str, SkillPack]:
        if self._packs is not None:
            return self._packs

        packs: dict[str, SkillPack] = {}
        if self._packs_dir.is_dir():
            for entry in sorted(self._packs_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_file = entry / "SKILL.md"
                if not skill_file.is_file():
                    continue
                try:
                    pack = load_pack(skill_file)
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
        self._packs = packs
        return packs

    def neutral_pack(self) -> SkillPack | None:
        return self._load_all().get(NEUTRAL_PACK_ID)

    def select(self, topic: str, *, signals: list[str] | None = None) -> SkillSelection | None:
        """Choose exactly one pack for a request, always resolving to a concrete pack or None.

        Returns None only when the neutral fallback pack itself is missing/invalid, which the
        caller treats as "skill packs unavailable" and proceeds with built-in prompts.
        """

        packs = self._load_all()
        neutral = packs.get(NEUTRAL_PACK_ID)
        haystack = " ".join(
            part.lower()
            for part in [topic or "", " ".join(signals or [])]
            if part and part.strip()
        )

        best: SkillPack | None = None
        best_score = 0.0
        for pack in packs.values():
            if pack.is_neutral:
                continue
            score, disqualified = self._score(pack, haystack)
            if disqualified:
                continue
            threshold = max(pack.min_match_score, self.global_min_match_score)
            if score >= threshold and score > best_score:
                best = pack
                best_score = score

        if best is not None:
            return SkillSelection(
                pack=best,
                match_score=round(best_score, 4),
                shadow=best.status == STATUS_SHADOW,
            )

        if neutral is None:
            return None
        return SkillSelection(
            pack=neutral,
            match_score=0.0,
            fallback_reason="no_pack_met_min_match_score",
            shadow=neutral.status == STATUS_SHADOW,
        )

    def _score(self, pack: SkillPack, haystack: str) -> tuple[float, bool]:
        if not pack.match_signals:
            return 0.0, False
        if any(sig.lower() in haystack for sig in pack.negative_signals):
            return 0.0, True
        hits = sum(1 for sig in pack.match_signals if sig.lower() in haystack)
        # Saturated denominator (cap at 3 signals) prevents long signal lists from
        # penalizing clear domain queries while still keeping 0-hit topics at 0.0.
        saturation_denom = min(max(1, len(pack.match_signals)), 3)
        lexical_score = min(1.0, hits / saturation_denom)
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
