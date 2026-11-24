"""Append-only Fact Ledger compaction and lightweight Evidence Index Card builders."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FactEntry:
        return cls(
            claim_id=str(payload.get("claim_id") or "").strip(),
            entity=str(payload.get("entity") or "").strip() or "general",
            metric=str(payload.get("metric") or "").strip() or "finding",
            value=str(payload.get("value") or "").strip(),
            period=str(payload.get("period") or "unspecified").strip() or "unspecified",
            source_ids=[str(s).strip() for s in (payload.get("source_ids") or []) if str(s).strip()],
            confidence=float(payload.get("confidence") or 0.7),
            task_id=int(payload.get("task_id") or 0),
            statement=str(payload.get("statement") or "").strip(),
            unit=str(payload.get("unit") or "").strip(),
            stance=str(payload.get("stance") or "asserted").strip() or "asserted",
            conflict_group_id=str(payload.get("conflict_group_id") or "").strip() or None,
        )

    def to_compact_line(self) -> str:
        cites = ",".join(self.source_ids)
        conflict_tag = f" [CONFLICT:{self.conflict_group_id}]" if self.conflict_group_id else ""
        body = self.statement or f"{self.entity} | {self.metric}={self.value} ({self.period})"
        return f"- {self.claim_id}{conflict_tag} (T{self.task_id}, conf={self.confidence:.2f}, refs={cites}): {truncate_text(body, 200)}"


@dataclass
class FactLedger:
    """Append-only structured fact ledger keeping frozen wave prefix stability."""

    version: int = 0
    entries: list[FactEntry] = field(default_factory=list)
    conflict_groups: dict[str, list[str]] = field(default_factory=dict)
    evidence_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version or 0),
            "entries": [entry.to_dict() for entry in self.entries],
            "conflict_groups": {k: list(v) for k, v in self.conflict_groups.items()},
            "evidence_gaps": list(self.evidence_gaps),
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
        )

    def render_frozen_markdown(self, *, max_entries: int = 40, char_budget: int = 4200) -> str:
        """Render entries in monotonic ascending claim_id order, preserving conflicts first if capped."""

        if not self.entries:
            return ""

        cap = max(1, int(max_entries or 40))
        if len(self.entries) <= cap:
            chosen = list(self.entries)
        else:
            disputed = [e for e in self.entries if e.conflict_group_id is not None]
            agreed = [e for e in self.entries if e.conflict_group_id is None]
            agreed_sorted = sorted(agreed, key=lambda e: (-e.confidence, e.claim_id))
            keep_ids = {e.claim_id for e in disputed[:cap]}
            for entry in agreed_sorted:
                if len(keep_ids) >= cap:
                    break
                keep_ids.add(entry.claim_id)
            chosen = [e for e in self.entries if e.claim_id in keep_ids]

        # Preserve strict append-only monotonic ordering by claim_id
        chosen.sort(key=lambda e: (len(e.claim_id), e.claim_id))
        lines: list[str] = ["<FACT_LEDGER>"]
        used = len(lines[0])
        budget = max(400, int(char_budget or 4200))
        for entry in chosen:
            line = entry.to_compact_line()
            if used + len(line) + 2 > budget:
                break
            lines.append(line)
            used += len(line) + 1
        lines.append("</FACT_LEDGER>")
        return "\n".join(lines)


class FactLedgerCompactor:
    """Deterministic append-only extractor and deduplicator preserving cross-source contradictions."""

    def update_ledger(
        self,
        existing_ledger: FactLedger | None,
        tasks: list[Any],
        *,
        valid_source_ids: set[str] | None = None,
    ) -> FactLedger:
        ledger = existing_ledger if existing_ledger is not None else FactLedger()

        indexed_keys: dict[tuple[str, str, str], list[FactEntry]] = {}
        existing_claim_texts: set[tuple[int, str]] = set()
        for entry in ledger.entries:
            key = (entry.entity.casefold(), entry.metric.casefold(), entry.period.casefold())
            indexed_keys.setdefault(key, []).append(entry)
            existing_claim_texts.add((entry.task_id, entry.statement.casefold()))

        next_seq = len(ledger.entries) + 1
        added_any = False

        for task in sorted(tasks or [], key=lambda t: int(getattr(t, "id", 0) or 0)):
            if str(getattr(task, "status", "") or "").lower() != "completed":
                continue
            task_id = int(getattr(task, "id", 0) or 0)
            extracted = self._extract_task_claims(task, valid_source_ids=valid_source_ids)
            for item in extracted:
                statement_key = (task_id, item["statement"].casefold())
                if statement_key in existing_claim_texts:
                    continue

                key = (item["entity"].casefold(), item["metric"].casefold(), item["period"].casefold())
                peers = indexed_keys.get(key, [])
                matched_same = next(
                    (
                        peer
                        for peer in peers
                        if peer.value.casefold() == item["value"].casefold()
                    ),
                    None,
                )
                if matched_same is not None:
                    for sid in item["source_ids"]:
                        if sid not in matched_same.source_ids:
                            matched_same.source_ids.append(sid)
                    matched_same.confidence = min(0.99, round(matched_same.confidence + 0.08, 4))
                    existing_claim_texts.add(statement_key)
                    added_any = True
                    continue

                conflict_group_id: str | None = None
                stance = "asserted"
                if peers and self._is_comparable_metric(item["metric"]):
                    conflict_group_id = peers[0].conflict_group_id or f"CG-{peers[0].claim_id}"
                    for peer in peers:
                        peer.conflict_group_id = conflict_group_id
                        peer.stance = "disputed"
                    stance = "disputed"
                    group_ids = [peer.claim_id for peer in peers]
                    new_id = f"C-{next_seq:03d}"
                    if new_id not in group_ids:
                        group_ids.append(new_id)
                    ledger.conflict_groups[conflict_group_id] = group_ids

                entry = FactEntry(
                    claim_id=f"C-{next_seq:03d}",
                    entity=item["entity"],
                    metric=item["metric"],
                    value=item["value"],
                    period=item["period"],
                    source_ids=list(item["source_ids"]),
                    confidence=float(item["confidence"]),
                    task_id=task_id,
                    statement=item["statement"],
                    stance=stance,
                    conflict_group_id=conflict_group_id,
                )
                ledger.entries.append(entry)
                indexed_keys.setdefault(key, []).append(entry)
                existing_claim_texts.add(statement_key)
                next_seq += 1
                added_any = True

        if added_any:
            ledger.version += 1
        return ledger

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
        allowed_ids = (valid_source_ids & task_valid) if valid_source_ids is not None else task_valid
        if valid_source_ids is not None and not task_valid:
            allowed_ids = set(valid_source_ids)

        candidates: list[dict[str, Any]] = []
        raw_fact_claims = getattr(task, "fact_claims", None) or []
        for fc in raw_fact_claims:
            if not isinstance(fc, dict):
                continue
            sids = [str(s).strip() for s in (fc.get("source_ids") or []) if str(s).strip() in allowed_ids]
            if not sids:
                continue
            statement = str(fc.get("statement") or fc.get("text") or "").strip()
            entity = str(fc.get("entity") or "").strip() or self._infer_entity(statement, task)
            metric = str(fc.get("metric") or "").strip() or self._infer_metric(statement)
            value = str(fc.get("value") or "").strip() or self._infer_value(statement)
            period = str(fc.get("period") or "").strip() or self._infer_period(statement)
            candidates.append(
                {
                    "entity": entity,
                    "metric": metric,
                    "value": value,
                    "period": period,
                    "source_ids": sids,
                    "confidence": float(fc.get("confidence") or 0.78),
                    "statement": statement or f"{entity} {metric}: {value} ({period})",
                }
            )
        if candidates:
            return candidates

        # Fallback from structured key_findings / task.claims / summary prose
        summary_payload = getattr(task, "summary_payload", None)
        if isinstance(summary_payload, dict) and isinstance(summary_payload.get("key_findings"), list):
            for finding in summary_payload["key_findings"]:
                if not isinstance(finding, dict):
                    continue
                text = strip_citation_markers(str(finding.get("text") or "").strip())
                sids = [
                    str(s).strip()
                    for s in (finding.get("source_ids") or [])
                    if str(s).strip() in allowed_ids
                ]
                if text and sids:
                    candidates.append(self._structured_from_text(text, sids, task))
        if candidates:
            return candidates

        for line in str(getattr(task, "summary", "") or "").splitlines():
            stripped = line.strip().lstrip("-* ").strip()
            if not stripped or stripped.startswith("#"):
                continue
            cited = [sid for sid in CITATION_PATTERN.findall(stripped) if sid in allowed_ids]
            if not cited:
                continue
            clean_stmt = strip_citation_markers(stripped)
            if clean_stmt:
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
            "source_ids": list(dict.fromkeys(sids)),
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
        return truncate_text(text, 64)

    @staticmethod
    def _infer_period(text: str) -> str:
        match = _PERIOD_RE.search(text)
        return match.group(0).upper() if match else "unspecified"


def build_evidence_index_cards(
    evidence_store: Any,
    *,
    runtime_index: Any | None = None,
    task_id: int | None = None,
    max_cards: int = 24,
) -> list[dict[str, Any]]:
    """Build lightweight EvidenceIndexCard dicts enriched with runtime chunk counts."""

    if evidence_store is None or not hasattr(evidence_store, "list_index_cards"):
        return []
    cards = evidence_store.list_index_cards(task_id=task_id)
    chunk_counts = (
        runtime_index.chunk_counts_by_source()
        if runtime_index is not None and hasattr(runtime_index, "chunk_counts_by_source")
        else {}
    )
    enriched: list[dict[str, Any]] = []
    for card in cards[: max(1, int(max_cards or 24))]:
        item = dict(card)
        item["chunk_count"] = int(chunk_counts.get(str(item.get("source_id") or ""), 0))
        enriched.append(item)
    return enriched


def render_evidence_index_cards_block(cards: list[dict[str, Any]], *, max_cards: int = 16) -> str:
    """Render EvidenceIndexCard entries into a compact pointer directory block."""

    if not cards:
        return ""
    lines = ["<EVIDENCE_INDEX_CARDS>"]
    for card in cards[: max(1, int(max_cards or 16))]:
        sid = card.get("source_id") or ""
        title = truncate_text(str(card.get("title") or ""), 64)
        domain = card.get("domain") or ""
        qual = card.get("quality_label") or "medium"
        fresh = card.get("freshness_label") or "unknown"
        pub = card.get("published_at") or "unknown"
        chunks = card.get("chunk_count", 0)
        gist = truncate_text(str(card.get("one_line_gist") or ""), 100)
        lines.append(f"- [{sid}] {title} | {domain} | {qual}/{fresh}/{pub} | chunks={chunks} | {gist}")
    lines.append("</EVIDENCE_INDEX_CARDS>")
    return "\n".join(lines)
