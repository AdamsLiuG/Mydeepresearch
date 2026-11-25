"""Append-only Fact Ledger compaction and lightweight Evidence Index Card builders."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from html import escape
from typing import Any

from services.memory_sanitizer import strip_tool_directives
from services.text_processing import CITATION_PATTERN, strip_citation_markers
from utils import truncate_text

_METRIC_HINT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?\s*(?:%|ms|s|GB|MB|TB|x|倍|亿|万|美元|元|B|M|K))",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(
    r"\b(?:20\d{2}(?:[-/]Q[1-4]|[-/]H[12]|[-/]\d{2})?|FY\d{2,4}|Q[1-4])\b",
    re.IGNORECASE,
)
_LEDGER_OPEN = "<FACT_LEDGER>"
_LEDGER_CLOSE = "</FACT_LEDGER>"
_LEDGER_CHAR_BUDGET = 4200
_CARD_CAP = 16


def _confidence(value: Any, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(1.0, max(0.0, score)) if math.isfinite(score) else default


def _source_ids(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    normalized = {str(s).strip() for s in value}
    return sorted(s for s in normalized if CITATION_PATTERN.fullmatch(f"[{s}]"))


def _prompt_line(value: Any, max_chars: int) -> str:
    text = strip_tool_directives(str(value or ""))
    text = " ".join(strip_citation_markers(text).split())
    return escape(text, quote=False)[:max_chars].rstrip()


@dataclass
class FactEntry:
    """Atomic fact unit flowing across waves and terminal convergence stages."""

    claim_id: str
    entity: str
    metric: str
    value: str
    period: str
    source_ids: list[str]
    confidence: float
    task_id: int
    statement: str = ""
    unit: str = ""
    stance: str = "asserted"  # asserted | disputed | gap
    conflict_group_id: str | None = None
    supporting_task_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "entity": self.entity,
            "metric": self.metric,
            "value": self.value,
            "period": self.period,
            "source_ids": list(self.source_ids),
            "confidence": round(float(self.confidence), 4),
            "task_id": self.task_id,
            "statement": self.statement,
            "unit": self.unit,
            "stance": self.stance,
            "conflict_group_id": self.conflict_group_id,
            "supporting_task_ids": list(self.supporting_task_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FactEntry:
        return cls(
            claim_id=str(payload.get("claim_id") or "").strip(),
            entity=str(payload.get("entity") or "").strip() or "general",
            metric=str(payload.get("metric") or "").strip() or "finding",
            value=str(payload.get("value") or "").strip(),
            period=str(payload.get("period") or "unspecified").strip() or "unspecified",
            source_ids=_source_ids(payload.get("source_ids")),
            confidence=_confidence(payload.get("confidence"), 0.7),
            task_id=int(payload.get("task_id") or 0),
            statement=str(payload.get("statement") or "").strip(),
            unit=str(payload.get("unit") or "").strip(),
            stance=str(payload.get("stance") or "asserted").strip() or "asserted",
            conflict_group_id=str(payload.get("conflict_group_id") or "").strip() or None,
            supporting_task_ids=list(payload.get("supporting_task_ids") or []),
        )

    def to_compact_line(self) -> str:
        cites = " ".join(f"[{sid}]" for sid in _source_ids(self.source_ids))
        conflict_tag = f" [CONFLICT:{self.conflict_group_id}]" if self.conflict_group_id else ""
        body = self.statement or f"{self.entity} | {self.metric}={self.value} {self.unit} ({self.period})"
        return f"- {self.claim_id}{conflict_tag}: {_prompt_line(body, 200)} {cites}".rstrip()


@dataclass
class FactLedger:
    """Append-only structured fact ledger keeping frozen wave prefix stability."""

    version: int = 0
    entries: list[FactEntry] = field(default_factory=list)
    conflict_groups: dict[str, list[str]] = field(default_factory=dict)
    evidence_gaps: list[str] = field(default_factory=list)
    prior_revisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version or 0),
            "entries": [entry.to_dict() for entry in self.entries],
            "conflict_groups": {k: list(v) for k, v in self.conflict_groups.items()},
            "evidence_gaps": list(self.evidence_gaps),
            "prior_revisions": deepcopy(self.prior_revisions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> FactLedger:
        if not isinstance(payload, dict) or not payload:
            return cls()
        raw_entries = payload.get("entries") or []
        entries = [FactEntry.from_dict(item) for item in raw_entries if isinstance(item, dict)]
        raw_groups = payload.get("conflict_groups") or {}
        groups = (
            {str(k): [str(cid) for cid in v] for k, v in raw_groups.items() if isinstance(v, list)}
            if isinstance(raw_groups, dict)
            else {}
        )
        gaps = [str(g).strip() for g in (payload.get("evidence_gaps") or []) if str(g).strip()]
        return cls(
            version=max(0, int(payload.get("version") or 0)),
            entries=entries,
            conflict_groups=groups,
            evidence_gaps=gaps,
            prior_revisions=deepcopy(payload.get("prior_revisions") or []),
        )

    def select_entries(self, *, max_entries: int = 40, char_budget: int = _LEDGER_CHAR_BUDGET) -> list[FactEntry]:
        """Select complete rows and prefer conflict groups under both limits."""
        remaining = char_budget - len(_LEDGER_OPEN) - len(_LEDGER_CLOSE) - 1
        groups: dict[str, list[FactEntry]] = {}
        agreed: list[FactEntry] = []
        for entry in self.entries:
            if entry.conflict_group_id:
                groups.setdefault(entry.conflict_group_id, []).append(entry)
            else:
                agreed.append(entry)
        chosen: set[str] = set()

        def include(entries: list[FactEntry]) -> None:
            nonlocal remaining
            length = sum(len(entry.to_compact_line()) + 1 for entry in entries)
            if len(chosen) + len(entries) <= max_entries and length <= remaining:
                chosen.update(entry.claim_id for entry in entries)
                remaining -= length

        for group in groups.values():
            include(group)
        for group in groups.values():
            if group[0].claim_id not in chosen:
                for entry in group:
                    include([entry])
        for entry in sorted(agreed, key=lambda row: (-len(row.source_ids), -row.confidence, row.claim_id)):
            include([entry])
        return [entry for entry in self.entries if entry.claim_id in chosen]

    def render_frozen_markdown(self, *, max_entries: int = 40, char_budget: int = _LEDGER_CHAR_BUDGET) -> str:
        rows = self.select_entries(max_entries=max_entries, char_budget=char_budget)
        return _render_fact_lines([entry.to_compact_line() for entry in rows])


def _render_fact_lines(lines: list[str]) -> str:
    return "\n".join([_LEDGER_OPEN, *lines, _LEDGER_CLOSE]) if lines else ""


def _render_committed_facts(facts: dict[str, str]) -> str:
    return _render_fact_lines([facts[cid] for cid in sorted(facts, key=lambda cid: (len(cid), cid))])


class FactLedgerCompactor:
    """Deterministic append-only extractor and deduplicator preserving cross-source contradictions."""

    def update_ledger(
        self,
        existing_ledger: FactLedger | None,
        tasks: list[Any],
        *,
        valid_source_ids: set[str] | None = None,
        evidence_store: Any | None = None,
    ) -> FactLedger:
        ledger = deepcopy(existing_ledger) if existing_ledger is not None else FactLedger()
        before = {entry.claim_id: entry.to_dict() for entry in ledger.entries}
        indexed_keys: dict[tuple[str, ...], list[FactEntry]] = {}
        next_seq = max(
            (int(entry.claim_id[2:]) for entry in ledger.entries if re.fullmatch(r"C-[0-9]+", entry.claim_id)),
            default=0,
        ) + 1
        if getattr(evidence_store, "web_extraction_enabled", False):
            if valid_source_ids is None:
                valid_source_ids = {item["source_id"] for item in evidence_store.lookup()}
            valid_source_ids = {
                sid for sid in valid_source_ids if evidence_store.is_eligible_for_facts(sid)
            }
            # Existing facts cannot keep an unsupported primary or corroborating claim.
            ledger.entries = [
                entry for entry in ledger.entries
                if entry.source_ids and set(entry.source_ids) <= valid_source_ids
            ]
            for group_id, claim_ids in list(ledger.conflict_groups.items()):
                members = [entry for entry in ledger.entries if entry.claim_id in claim_ids]
                if len(members) > 1:
                    ledger.conflict_groups[group_id] = [entry.claim_id for entry in members]
                else:
                    del ledger.conflict_groups[group_id]
                    for entry in members:
                        entry.conflict_group_id = None
                        entry.stance = "supported"
        for entry in ledger.entries:
            indexed_keys.setdefault(self._fact_key(entry.to_dict()), []).append(entry)

        for task in sorted(tasks, key=lambda item: int(item.id)):
            if str(task.status or "").lower() != "completed":
                continue
            task_id = int(task.id)
            for item in self._extract_task_claims(task, valid_source_ids=valid_source_ids):
                key = self._fact_key(item)
                peers = indexed_keys.setdefault(key, [])
                matched = next((peer for peer in peers if peer.value.casefold() == item["value"].casefold()), None)
                if matched is not None:
                    new_sources = set(item["source_ids"]) - set(matched.source_ids)
                    if new_sources:
                        matched.source_ids = _source_ids(set(matched.source_ids) | new_sources)
                        matched.confidence = max(matched.confidence, min(0.99, round(matched.confidence + 0.08, 4)))
                    if task_id != matched.task_id and task_id not in matched.supporting_task_ids:
                        matched.supporting_task_ids.append(task_id)
                        matched.supporting_task_ids.sort()
                    continue

                entry = FactEntry(
                    claim_id=f"C-{next_seq:03d}",
                    entity=item["entity"],
                    metric=item["metric"],
                    value=item["value"],
                    period=item["period"],
                    source_ids=item["source_ids"],
                    confidence=item["confidence"],
                    task_id=task_id,
                    statement=item["statement"],
                    unit=item.get("unit", ""),
                )
                if peers and self._is_comparable_metric(entry.metric):
                    group = peers[0].conflict_group_id or f"CG-{peers[0].claim_id}"
                    for peer in (*peers, entry):
                        peer.conflict_group_id = group
                        peer.stance = "disputed"
                    group_claim_ids = ledger.conflict_groups.setdefault(group, [])
                    group_claim_ids.extend(peer.claim_id for peer in (*peers, entry) if peer.claim_id not in group_claim_ids)
                peers.append(entry)
                ledger.entries.append(entry)
                next_seq += 1

        after = [entry.to_dict() for entry in ledger.entries]
        changed = [row for row in after if before.get(row["claim_id"]) != row]
        if changed or len(after) != len(before):
            ledger.prior_revisions.extend(
                before[row["claim_id"]] for row in changed if row["claim_id"] in before
            )
            ledger.version += 1
        return ledger

    @classmethod
    def _fact_key(cls, item: dict[str, Any]) -> tuple[str, ...]:
        key = tuple(" ".join(str(item.get(k) or "").casefold().split()) for k in ("entity", "metric", "period", "unit"))
        if cls._is_comparable_metric(key[1]):
            return key
        # An inferred, generic metric cannot equate separate statements that share a number.
        return (*key, str(item.get("statement") or item.get("value") or "").casefold())

    @staticmethod
    def _is_comparable_metric(metric: str) -> bool:
        return bool(metric) and metric.lower() not in {"finding", "summary", "general"}

    def _extract_task_claims(
        self,
        task: Any,
        *,
        valid_source_ids: set[str] | None,
    ) -> list[dict[str, Any]]:
        task_valid = {
            str(item.get("source_id") or "").strip()
            for item in (getattr(task, "evidence_items", None) or [])
            if isinstance(item, dict) and str(item.get("source_id") or "").strip()
        }
        allowed_ids = valid_source_ids if valid_source_ids is not None else task_valid
        candidates: list[dict[str, Any]] = []
        grounded_statements: set[str] = set()
        for claim in getattr(task, "fact_claims", None) or []:
            if not isinstance(claim, dict):
                continue
            sids = [sid for sid in _source_ids(claim.get("source_ids")) if sid in allowed_ids]
            if not sids:
                continue
            statement = strip_citation_markers(str(claim.get("statement") or claim.get("text") or ""))
            entity = str(claim.get("entity") or "").strip() or self._infer_entity(statement, task)
            metric = str(claim.get("metric") or "").strip() or self._infer_metric(statement)
            raw_value = claim.get("value")
            value = str(raw_value if raw_value is not None else "").strip() or self._infer_value(statement)
            if not value:
                continue
            period = str(claim.get("period") or "").strip() or self._infer_period(statement)
            candidates.append({
                "entity": entity,
                "metric": metric,
                "value": value,
                "period": period,
                "unit": str(claim.get("unit") or "").strip(),
                "source_ids": sids,
                "confidence": _confidence(claim.get("confidence"), 0.78),
                "statement": statement or f"{entity} {metric}: {value} ({period})",
            })
            if statement:
                grounded_statements.add(statement.casefold())

        summary_payload = getattr(task, "summary_payload", None) or {}
        findings = summary_payload.get("key_findings") if isinstance(summary_payload, dict) else None
        for finding in [*(findings or []), *(getattr(task, "claims", None) or [])]:
            if not isinstance(finding, dict):
                continue
            raw_text = str(finding.get("text") or finding.get("statement") or "")
            text = strip_citation_markers(raw_text)
            sids = [
                sid for sid in _source_ids((
                    *_source_ids(finding.get("source_ids")), *CITATION_PATTERN.findall(raw_text)
                ))
                if sid in allowed_ids
            ]
            if text and sids and text.casefold() not in grounded_statements:
                candidates.append(self._structured_from_text(text, sids, task))
                grounded_statements.add(text.casefold())
        if candidates:
            return candidates

        for line in str(getattr(task, "summary", "") or "").splitlines():
            stripped = line.strip().lstrip("-* ").strip()
            if not stripped or stripped.startswith("#"):
                continue
            cited = [sid for sid in CITATION_PATTERN.findall(stripped) if sid in allowed_ids]
            clean_stmt = strip_citation_markers(stripped)
            if cited and clean_stmt:
                candidates.append(self._structured_from_text(clean_stmt, cited, task))
        return candidates

    def _structured_from_text(self, text: str, sids: list[str], task: Any) -> dict[str, Any]:
        entity = self._infer_entity(text, task)
        metric = self._infer_metric(text)
        value = self._infer_value(text)
        period = self._infer_period(text)
        return {
            "entity": entity,
            "metric": metric,
            "value": value,
            "period": period,
            "source_ids": _source_ids(sids),
            "confidence": 0.75,
            "statement": truncate_text(text, 240),
        }

    @staticmethod
    def _infer_entity(text: str, task: Any) -> str:
        ascii_entities = re.findall(r"\b[A-Z][A-Za-z0-9.+-]{1,24}\b", text)
        for ent in ascii_entities:
            if not ent.startswith(("T", "Q", "FY")):
                return ent
        title = str(getattr(task, "title", "") or "").strip()
        return truncate_text(title, 28) if title else "general"

    @staticmethod
    def _infer_metric(text: str) -> str:
        for keyword in (
            "营收",
            "利润",
            "毛利率",
            "市占率",
            "份额",
            "延迟",
            "吞吐",
            "准确率",
            "成本",
            "增速",
            "latency",
            "throughput",
            "revenue",
            "margin",
            "accuracy",
        ):
            if keyword in text.lower():
                return keyword
        return "finding"

    @staticmethod
    def _infer_value(text: str) -> str:
        match = _METRIC_HINT_RE.search(text)
        if match:
            return match.group("value").strip()
        return text

    @staticmethod
    def _infer_period(text: str) -> str:
        match = _PERIOD_RE.search(text)
        return match.group(0).upper() if match else "unspecified"


def build_evidence_index_cards(
    evidence_store: Any,
    *,
    runtime_index: Any | None = None,
    task_id: int | None = None,
    max_cards: int | None = 24,
) -> list[dict[str, Any]]:
    """Build lightweight EvidenceIndexCard dicts enriched with runtime chunk counts."""

    if evidence_store is None or not hasattr(evidence_store, "list_index_cards"):
        return []
    cards = evidence_store.list_index_cards(task_id=task_id)
    if getattr(evidence_store, "web_extraction_enabled", False):
        cards = [card for card in cards if evidence_store.is_eligible_for_facts(card["source_id"])]
    chunk_counts = (
        runtime_index.chunk_counts_by_source()
        if runtime_index is not None and hasattr(runtime_index, "chunk_counts_by_source")
        else {}
    )
    enriched: list[dict[str, Any]] = []
    for card in cards[:max_cards]:
        item = dict(card)
        item["chunk_count"] = int(chunk_counts.get(str(item.get("source_id") or ""), 0))
        enriched.append(item)
    return enriched


def render_evidence_index_cards_block(cards: list[dict[str, Any]], *, max_cards: int = 16) -> str:
    """Render EvidenceIndexCard entries into a compact pointer directory block."""

    if not cards:
        return ""
    lines = ["<EVIDENCE_INDEX_CARDS>"]
    lines.extend(filter(None, (render_evidence_index_card(card) for card in cards[:max_cards])))
    if len(lines) == 1:
        return ""
    lines.append("</EVIDENCE_INDEX_CARDS>")
    return "\n".join(lines)


def render_evidence_index_card(card: dict[str, Any]) -> str:
    sid = card.get("source_id")
    if not isinstance(sid, str) or not CITATION_PATTERN.fullmatch(f"[{sid}]"):
        return ""
    title = _prompt_line(card.get("title"), 64)
    domain = _prompt_line(card.get("domain"), 64)
    qual = _prompt_line(card.get("quality_label") or "medium", 16)
    fresh = _prompt_line(card.get("freshness_label") or "unknown", 16)
    pub = _prompt_line(card.get("published_at") or "unknown", 32)
    chunks = _prompt_line(card.get("chunk_count", 0), 8) or "0"
    gist = _prompt_line(card.get("one_line_gist"), 100)
    return f"- [{sid}] {title} | {domain} | {qual}/{fresh}/{pub} | chunks={chunks} | {gist}"


def append_context_checkpoint(state: Any, config: Any) -> bool:
    """Commit an immutable batch of fact lines and pointer cards to the cached prefix."""
    checkpoint = state.context_checkpoint_blocks
    committed = {cid: line for batch in checkpoint for cid, line in batch["facts"].items()}
    card_ids = {sid for batch in checkpoint for sid in batch["cards"]}
    fact_slots = max(0, int(config.fact_ledger_max_entries) // 2 - len(committed))
    fact_chars = max(0, _LEDGER_CHAR_BUDGET // 2 - sum(
        len(_render_committed_facts(batch["facts"])) + 2 for batch in checkpoint if batch["facts"]
    ))
    ledger = FactLedger.from_dict(state.fact_ledger)
    ledger.entries = [entry for entry in ledger.entries if entry.claim_id not in committed]
    facts = (
        {entry.claim_id: entry.to_compact_line() for entry in ledger.select_entries(
            max_entries=fact_slots, char_budget=fact_chars,
        )}
        if config.context_compaction_enabled else {}
    )
    cards = (
        {card["source_id"]: line
         for card in [card for card in state.evidence_index_cards if card["source_id"] not in card_ids][
             :max(0, _CARD_CAP - len(card_ids))
         ] if (line := render_evidence_index_card(card))}
        if config.evidence_pointer_disclosure_enabled else {}
    )
    if facts or cards:
        state.context_checkpoint_blocks = [*checkpoint, {"facts": facts, "cards": cards}]
        return True
    return False


def render_context_checkpoint(state: Any, config: Any) -> str:
    """Render committed batches in the order received, with legacy snapshot fallback."""
    checkpoint = (getattr(state, "context_checkpoint_blocks", None) or [])
    if not checkpoint and not getattr(state, "frozen_prefix_version", 0):
        ledger = FactLedger.from_dict(getattr(state, "fact_ledger", None))
        return "\n\n".join(filter(None, (
            ledger.render_frozen_markdown(max_entries=config.fact_ledger_max_entries)
            if config.context_compaction_enabled else "",
            render_evidence_index_cards_block(getattr(state, "evidence_index_cards", None) or [])
            if config.evidence_pointer_disclosure_enabled else "",
        )))

    blocks: list[str] = []
    for batch in checkpoint:
        if config.context_compaction_enabled and batch["facts"]:
            blocks.append(_render_committed_facts(batch["facts"]))
        if config.evidence_pointer_disclosure_enabled and batch["cards"]:
            blocks.append("\n".join([
                "<EVIDENCE_INDEX_CARDS>",
                *(batch["cards"][sid] for sid in sorted(batch["cards"], key=_source_order)),
                "</EVIDENCE_INDEX_CARDS>",
            ]))
    return "\n\n".join(blocks)


def _source_order(source_id: str) -> tuple[int, str, int]:
    task, source = source_id.split("-", 1)
    return int(task[1:]), source[0], int(source[1:])


def _select_context_delta(state: Any, config: Any, ledger: FactLedger) -> list[FactEntry]:
    committed = {cid: line for batch in (getattr(state, "context_checkpoint_blocks", None) or []) for cid, line in batch["facts"].items()}
    remaining = [entry for entry in ledger.entries if committed.get(entry.claim_id) != entry.to_compact_line()]
    used_chars = sum(len(_render_committed_facts(batch["facts"])) + 2
                     for batch in (getattr(state, "context_checkpoint_blocks", None) or []) if batch["facts"])
    return FactLedger(entries=remaining).select_entries(
        max_entries=int(config.fact_ledger_max_entries) - len(committed),
        char_budget=_LEDGER_CHAR_BUDGET - used_chars,
    )


def context_fact_entries(state: Any, config: Any) -> list[FactEntry]:
    """Return the entries represented in this checkpoint and its bounded suffix."""
    if not config.context_compaction_enabled:
        return []
    ledger = FactLedger.from_dict(getattr(state, "fact_ledger", None))
    if not getattr(state, "context_checkpoint_blocks", None) and not getattr(state, "frozen_prefix_version", 0):
        return ledger.select_entries(max_entries=config.fact_ledger_max_entries)
    shown_ids = {cid for batch in (getattr(state, "context_checkpoint_blocks", None) or []) for cid in batch["facts"]}
    shown_ids.update(entry.claim_id for entry in _select_context_delta(state, config, ledger))
    return [entry for entry in ledger.entries if entry.claim_id in shown_ids]


def render_context_delta(state: Any, config: Any) -> str:
    """Render new or amended facts that are absent from the frozen prefix."""
    if not config.context_compaction_enabled or not (
        getattr(state, "context_checkpoint_blocks", None) or getattr(state, "frozen_prefix_version", 0)
    ):
        return ""
    rows = _select_context_delta(state, config, FactLedger.from_dict(getattr(state, "fact_ledger", None)))
    facts = _render_fact_lines([entry.to_compact_line() for entry in rows])
    return f"事实账本补充（相同 claim_id 以此处为准）：\n{facts}" if facts else ""


def context_covered_task_ids(state: Any, config: Any) -> set[int]:
    """Identify completed tasks whose committed facts all fit in the prompt."""
    if not config.kv_cache_prefix_optimization_enabled:
        return set()
    shown_ids = {entry.claim_id for entry in context_fact_entries(state, config)}
    if not shown_ids:
        return set()
    covered: set[int] = set()
    omitted: set[int] = set()
    for entry in FactLedger.from_dict(getattr(state, "fact_ledger", None)).entries:
        target = covered if entry.claim_id in shown_ids else omitted
        target.add(entry.task_id)
        target.update(entry.supporting_task_ids)
    completed = {task.id for task in state.todo_items if task.status == "completed"}
    return (covered - omitted) & completed
