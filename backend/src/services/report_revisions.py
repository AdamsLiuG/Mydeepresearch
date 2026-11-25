"""Persistent interactive report annotations, targeted reverification, and immutable revision diffs."""

from __future__ import annotations

import difflib
import hashlib
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from services.memory_sanitizer import sanitize_injection_text

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+|[\u4e00-\u9fff]{1,2}")
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _tokenize(text: str) -> set[str]:
    return {tok.casefold() for tok in _TOKEN_RE.findall(str(text or "")) if tok.strip()}


class ReportAnnotationCreate(BaseModel):
    """Payload to append a persistent annotation on a report section or claim."""

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(default="", max_length=120)
    claim_id: str = Field(default="", max_length=120)
    target_quote: str = Field(default="", max_length=1000)
    comment: str = Field(..., min_length=1, max_length=2000)
    severity: Literal["info", "concern", "disputed"] = "concern"
    action_requested: Literal["comment", "reverify"] = "comment"
    author: str = Field(default="reviewer", min_length=1, max_length=120)
    expected_state_version: int | None = Field(default=None, ge=0)

    @field_validator("section_id", "claim_id", "target_quote", "comment", "author")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _require_nonempty_comment(self) -> ReportAnnotationCreate:
        if not self.comment:
            raise ValueError("comment must not be empty")
        if not self.author:
            raise ValueError("author must not be empty")
        return self


class ReportAnnotation(BaseModel):
    """Persisted interactive report annotation tied to a report revision."""

    model_config = ConfigDict(extra="forbid")

    annotation_id: str
    request_id: str
    tenant_id: str
    revision_id: int
    section_id: str = ""
    claim_id: str = ""
    target_quote: str = ""
    comment: str
    severity: Literal["info", "concern", "disputed"] = "concern"
    action_requested: Literal["comment", "reverify"] = "comment"
    author: str = "reviewer"
    created_at: str
    status: Literal["open", "reverified", "resolved"] = "open"


class ReverifyRequest(BaseModel):
    """Targeted reverification trigger against an annotated section or claim."""

    model_config = ConfigDict(extra="forbid")

    annotation_id: str | None = Field(default=None, max_length=120)
    section_id: str = Field(default="", max_length=120)
    claim_id: str = Field(default="", max_length=120)
    note: str = Field(default="", max_length=2000)
    expected_state_version: int | None = Field(default=None, ge=0)

    @field_validator("annotation_id", "section_id", "claim_id", "note")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).strip()

    @model_validator(mode="after")
    def _require_target(self) -> ReverifyRequest:
        if not any(
            (
                self.annotation_id or "",
                self.section_id or "",
                self.claim_id or "",
                self.note or "",
            )
        ):
            raise ValueError("at least one of annotation_id, section_id, claim_id, or note is required")
        return self


class ReportRevisionSnapshot(BaseModel):
    """Immutable snapshot of one report revision."""

    model_config = ConfigDict(extra="forbid")

    revision_id: int
    parent_revision_id: int | None = None
    request_id: str
    report_markdown: str
    reason: str
    source_annotation_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    reverification_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    content_sha256: str


class RevisionDiff(BaseModel):
    """Structured deterministic diff between two immutable report revisions."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    base_revision_id: int
    target_revision_id: int
    added_lines: list[str] = Field(default_factory=list)
    removed_lines: list[str] = Field(default_factory=list)
    changed_sections: list[str] = Field(default_factory=list)
    unified_diff: str = ""
    added_count: int = 0
    removed_count: int = 0


def ensure_revisions(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a deep-copied revision list, initializing revision 1 from current markdown if needed."""
    request_id = str(snapshot.get("request_id") or "").strip() or "unknown"
    raw = snapshot.get("report_revisions")
    if isinstance(raw, list) and raw:
        revisions = [deepcopy(item) for item in raw if isinstance(item, dict)]
        if revisions:
            return sorted(revisions, key=lambda item: int(item.get("revision_id") or 0))

    initial_md = str(snapshot.get("report_markdown") or "")
    created_at = str(snapshot.get("updated_at") or _utc_now())
    initial = ReportRevisionSnapshot(
        revision_id=1,
        parent_revision_id=None,
        request_id=request_id,
        report_markdown=initial_md,
        reason="initial_report",
        source_annotation_id=None,
        evidence_refs=[],
        reverification_summary={},
        created_at=created_at,
        content_sha256=_sha256_text(initial_md),
    ).model_dump()
    return [initial]


def list_annotations(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    raw = snapshot.get("report_annotations")
    if not isinstance(raw, list):
        return []
    return [deepcopy(item) for item in raw if isinstance(item, dict)]


def append_annotation(
    snapshot: dict[str, Any],
    payload: ReportAnnotationCreate,
    *,
    tenant_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append a validated interactive annotation to the snapshot copy without mutating past revisions."""
    updated = deepcopy(snapshot)
    revisions = ensure_revisions(updated)
    annotations = list_annotations(updated)
    request_id = str(updated.get("request_id") or "").strip() or "unknown"
    current_revision_id = int(revisions[-1].get("revision_id") or 1)

    record = ReportAnnotation(
        annotation_id=f"ann-{uuid4().hex[:12]}",
        request_id=request_id,
        tenant_id=str(tenant_id or "default").strip() or "default",
        revision_id=current_revision_id,
        section_id=payload.section_id,
        claim_id=payload.claim_id,
        target_quote=sanitize_injection_text(payload.target_quote)[:1000],
        comment=sanitize_injection_text(payload.comment)[:2000],
        severity=payload.severity,
        action_requested=payload.action_requested,
        author=sanitize_injection_text(payload.author)[:120] or "reviewer",
        created_at=_utc_now(),
        status="open",
    ).model_dump()

    annotations.append(record)
    updated["report_revisions"] = revisions
    updated["report_annotations"] = annotations
    return updated, record


def _collect_eligible_evidence(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect eligible verifiable evidence entries from workbench citation index and todo items."""
    collected: dict[str, dict[str, Any]] = {}
    workbench = snapshot.get("report_workbench")
    if isinstance(workbench, dict):
        citation_index = workbench.get("citation_index")
        if isinstance(citation_index, dict):
            for sid, entry in citation_index.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("eligible_for_facts") is False:
                    continue
                clean_sid = str(entry.get("source_id") or sid).strip()
                if clean_sid:
                    collected[clean_sid] = {
                        "source_id": clean_sid,
                        "title": str(entry.get("title") or clean_sid),
                        "snippet": str(entry.get("excerpt") or entry.get("snippet") or ""),
                        "url": str(entry.get("url") or ""),
                    }

    for task in snapshot.get("todo_items") or []:
        if not isinstance(task, dict):
            continue
        for item in task.get("source_items") or []:
            if not isinstance(item, dict):
                continue
            if item.get("eligible_for_facts") is False or item.get("is_eligible_for_facts") is False:
                continue
            sid = str(item.get("source_id") or item.get("sourceId") or "").strip()
            if sid and sid not in collected:
                collected[sid] = {
                    "source_id": sid,
                    "title": str(item.get("title") or sid),
                    "snippet": str(item.get("snippet") or item.get("raw") or ""),
                    "url": str(item.get("url") or ""),
                }
    return list(collected.values())


def _has_conflict_for_sources(snapshot: dict[str, Any], source_ids: set[str]) -> bool:
    if not source_ids:
        return False
    clusters: list[Any] = []
    workbench = snapshot.get("report_workbench")
    if isinstance(workbench, dict) and isinstance(workbench.get("evidence_conflicts"), list):
        clusters.extend(workbench["evidence_conflicts"])
    if isinstance(snapshot.get("evidence_conflicts"), list):
        clusters.extend(snapshot["evidence_conflicts"])
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        for side in cluster.get("sides") or []:
            if isinstance(side, dict) and str(side.get("source_id") or "").strip() in source_ids:
                return True
    return False


def _inject_reverify_note(report_markdown: str, *, section_id: str, note_block: str) -> str:
    lines = str(report_markdown or "").splitlines()
    target_norm = section_id.strip().casefold()
    if target_norm and lines:
        insert_at: int | None = None
        for idx, line in enumerate(lines):
            match = _HEADING_RE.match(line)
            if not match:
                continue
            heading_text = match.group(2).strip().casefold()
            if heading_text == target_norm or target_norm in heading_text:
                # Find end of section before next heading of same/higher level or EOF
                level = len(match.group(1))
                cursor = idx + 1
                while cursor < len(lines):
                    nxt = _HEADING_RE.match(lines[cursor])
                    if nxt and len(nxt.group(1)) <= level:
                        break
                    cursor += 1
                insert_at = cursor
                break
        if insert_at is not None:
            new_lines = lines[:insert_at] + ["", note_block, ""] + lines[insert_at:]
            return "\n".join(new_lines).strip() + "\n"

    base = str(report_markdown or "").rstrip()
    heading = "## 定向复核记录"
    if base:
        return f"{base}\n\n{heading}\n\n{note_block}\n"
    return f"{heading}\n\n{note_block}\n"


def reverify_claim_or_section(
    snapshot: dict[str, Any],
    payload: ReverifyRequest,
    *,
    tenant_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run targeted reverification over eligible evidence and append an immutable revision."""
    del tenant_id  # Ownership verified at store/route boundary.
    updated = deepcopy(snapshot)
    revisions = ensure_revisions(updated)
    annotations = list_annotations(updated)

    linked_ann: dict[str, Any] | None = None
    if payload.annotation_id:
        for ann in annotations:
            if ann.get("annotation_id") == payload.annotation_id:
                linked_ann = ann
                break
        if linked_ann is None:
            raise KeyError(f"annotation not found: {payload.annotation_id}")

    target_section = (payload.section_id or (linked_ann.get("section_id") if linked_ann else "") or "").strip()
    target_claim = (payload.claim_id or (linked_ann.get("claim_id") if linked_ann else "") or "").strip()
    target_quote = (linked_ann.get("target_quote") if linked_ann else "") or ""
    target_note = (payload.note or (linked_ann.get("comment") if linked_ann else "") or "").strip()

    query_tokens = _tokenize(" ".join(part for part in (target_section, target_claim, target_quote, target_note) if part))
    candidates = _collect_eligible_evidence(updated)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for entry in candidates:
        doc_tokens = _tokenize(f"{entry['title']} {entry['snippet']} {entry['source_id']}")
        overlap = len(query_tokens & doc_tokens)
        if overlap > 0 or (target_claim and target_claim.casefold() in entry["snippet"].casefold()):
            scored.append((overlap, entry["source_id"], entry))
    scored.sort(key=lambda row: (-row[0], row[1]))

    matched_entries = [row[2] for row in scored[:4]]
    if not matched_entries and target_section and candidates:
        # Check if section heading paragraph in report already cites known eligible source IDs
        cited_in_report = re.findall(r"\[(T\d+-(?:S|M|C)\d+)\]", str(updated.get("report_markdown") or ""))
        cited_set = set(cited_in_report)
        matched_entries = [entry for entry in candidates if entry["source_id"] in cited_set][:3]

    matched_source_ids = [entry["source_id"] for entry in matched_entries]
    has_conflict = _has_conflict_for_sources(updated, set(matched_source_ids))
    if has_conflict:
        verification_status = "disputed"
    elif matched_source_ids:
        verification_status = "supported"
    else:
        verification_status = "needs_additional_evidence"

    refs_text = " ".join(f"[{sid}]" for sid in matched_source_ids)
    safe_label = sanitize_injection_text(target_section or target_claim or "指定结论")[:120]
    safe_note = sanitize_injection_text(target_note)[:300]
    note_lines = [
        f"> **定向复核 ({verification_status})** · 目标：`{safe_label}`",
        f"> - 复核依据：{refs_text or '暂无匹配的既有合资格来源，建议补充检索外部一手证据'}",
    ]
    if safe_note:
        note_lines.append(f"> - 审阅说明：{safe_note}")
    note_block = "\n".join(note_lines)

    latest_revision = revisions[-1]
    prev_markdown = str(latest_revision.get("report_markdown") or updated.get("report_markdown") or "")
    new_markdown = _inject_reverify_note(prev_markdown, section_id=target_section, note_block=note_block)
    next_revision_id = max(int(rev.get("revision_id") or 0) for rev in revisions) + 1
    parent_revision_id = int(latest_revision.get("revision_id") or 1)
    request_id = str(updated.get("request_id") or "").strip() or "unknown"

    summary = {
        "status": verification_status,
        "section_id": target_section,
        "claim_id": target_claim,
        "annotation_id": payload.annotation_id,
        "matched_source_ids": matched_source_ids,
        "has_conflict": has_conflict,
        "reused_existing_evidence": bool(matched_source_ids),
    }

    new_revision = ReportRevisionSnapshot(
        revision_id=next_revision_id,
        parent_revision_id=parent_revision_id,
        request_id=request_id,
        report_markdown=new_markdown,
        reason="targeted_reverify",
        source_annotation_id=payload.annotation_id,
        evidence_refs=matched_source_ids,
        reverification_summary=summary,
        created_at=_utc_now(),
        content_sha256=_sha256_text(new_markdown),
    ).model_dump()

    revisions.append(new_revision)
    if linked_ann is not None:
        linked_ann["status"] = "reverified"
    updated["report_revisions"] = revisions
    updated["report_annotations"] = annotations
    updated["report_markdown"] = new_markdown
    return updated, new_revision, summary


def _changed_sections(base_markdown: str, target_markdown: str) -> list[str]:
    """Identify headings whose body content differs between base and target revisions."""
    def _split_sections(md: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current = "全文"
        for line in str(md or "").splitlines():
            match = _HEADING_RE.match(line)
            if match:
                current = match.group(2).strip()
                sections.setdefault(current, [])
            else:
                sections.setdefault(current, []).append(line)
        return sections

    base_map = _split_sections(base_markdown)
    target_map = _split_sections(target_markdown)
    ordered_headings = list(dict.fromkeys([*base_map.keys(), *target_map.keys()]))
    return [
        heading
        for heading in ordered_headings
        if base_map.get(heading) != target_map.get(heading)
    ]


def compute_revision_diff(
    request_id: str,
    base_revision: dict[str, Any],
    target_revision: dict[str, Any],
) -> RevisionDiff:
    """Compute a deterministic line + section diff between two immutable revisions."""
    base_id = int(base_revision.get("revision_id") or 1)
    target_id = int(target_revision.get("revision_id") or 1)
    base_md = str(base_revision.get("report_markdown") or "")
    target_md = str(target_revision.get("report_markdown") or "")

    diff_lines = list(
        difflib.unified_diff(
            base_md.splitlines(),
            target_md.splitlines(),
            fromfile=f"revision-{base_id}",
            tofile=f"revision-{target_id}",
            lineterm="",
        )
    )
    added_lines = [line[1:] for line in diff_lines if line.startswith("+") and not line.startswith("+++")]
    removed_lines = [line[1:] for line in diff_lines if line.startswith("-") and not line.startswith("---")]
    changed_sections = _changed_sections(base_md, target_md)
    return RevisionDiff(
        request_id=request_id,
        base_revision_id=base_id,
        target_revision_id=target_id,
        added_lines=added_lines,
        removed_lines=removed_lines,
        changed_sections=changed_sections,
        unified_diff="\n".join(diff_lines),
        added_count=len(added_lines),
        removed_count=len(removed_lines),
    )
