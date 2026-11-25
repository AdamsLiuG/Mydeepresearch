"""Independent, evidence-bound review of a report draft."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict

from models import (
    CriticIssue,
    ReportClaim,
    ReportConflict,
    ReportCriticResult,
    ReportExcerpt,
    ReportModelResponse,
    ReportRevisionPlan,
)
from services.claim_verifier import numeric_grounding_mismatch
from services.context_builder import parse_report_json, report_json
from services.reviewer import ReviewService

CRITIC_MAX_ISSUES = 8
CRITIC_MAX_CLAIMS = 24
CRITIC_MAX_EXCERPTS = 12
REVISION_MAX_TARGETS = 4
_REASONS = {
    "counter_evidence_gap": "存在未在本节说明的反向证据或来源分歧。",
    "survivorship_bias": "证据中的样本或分母与陈述中的总体不一致。",
    "temporal_skew": "陈述与证据使用的观测期间不一致。",
    "causal_overreach": "证据只支持相关性或观察结果，因果表述缺少支持。",
}
_ACTIONS = {"qualify", "surface_conflict", "mark_gap", "reverify"}
_POPULATION_ALL = re.compile(r"\b(?:all|every)\s+(?:users|customers|participants|companies|firms)\b|全体|所有用户", re.I)
_POPULATION_SUBSET = re.compile(r"\b(?:active|surviving|successful|responding|voluntary|enrolled|paying)\s+\w+|活跃用户|存续样本|自愿披露|付费用户", re.I)
_DENOMINATOR = re.compile(r"\bper\s+(?:1000\s+|100\s+)?([\w-]+)|每[千百万]?[\u4e00-\u9fff]{1,5}", re.I)
_PERIOD = re.compile(r"\b(?:FY|CY|TTM|Q[1-4]|fiscal year|calendar year)\s*[- ]?\s*(?:20\d{2})?\b|\b20\d{2}\b|财年|自然年|滚动十二月", re.I)
_CAUSAL = re.compile(r"\b(?:caused?|causes|led to|leads to|resulted in|results in|drives?|will produce)\b|导致|造成|证明.*引起", re.I)
_OBSERVATIONAL = re.compile(r"\b(?:correlat\w+|associat\w+|observational|coincided)\b|相关|观察性", re.I)
_NON_CAUSAL = re.compile(r"\b(?:not|cannot|can't|didn't|did not|may not|does not)\b|不能|并非|未证明", re.I)


def _periods(text: str) -> set[str]:
    normalized = re.sub(r"fiscal year|财年", "FY", text, flags=re.I)
    normalized = re.sub(r"calendar year|自然年", "CY", normalized, flags=re.I)
    normalized = normalized.replace("滚动十二月", "TTM")
    return {re.sub(r"[\s-]", "", match.group()).upper() for match in _PERIOD.finditer(normalized)}


def _denominators(text: str) -> set[str]:
    return {match.group().casefold() for match in _DENOMINATOR.finditer(text)}


class AdversarialRedTeamCritic:
    @staticmethod
    def sample(claims, excerpts, conflicts, coverage_gaps) -> dict:
        cited = {ref for _, claim in claims for ref in claim.support_refs}
        contrary = {ref for group in conflicts for side in group.sides for ref in side.support_refs}
        selected = []
        size = 0
        for excerpt in sorted(excerpts, key=lambda item: (item.ref not in contrary, item.ref not in cited)):
            if excerpt.ref not in cited | contrary:
                continue
            size += len(excerpt.text.encode("utf-8"))
            if size > 8192 or len(selected) == CRITIC_MAX_EXCERPTS:
                break
            selected.append(asdict(excerpt))
        return {
            "claims": [{"section_id": section, **asdict(claim)} for section, claim in claims[:CRITIC_MAX_CLAIMS]],
            "excerpts": selected,
            "conflicts": [asdict(group) for group in conflicts[:8]],
            "coverage_gaps": list(coverage_gaps)[:24],
        }

    @staticmethod
    def confirm(
        claims: tuple[tuple[str, ReportClaim], ...],
        excerpts: tuple[ReportExcerpt, ...],
        conflicts: tuple[ReportConflict, ...],
        visible_conflict_ids: tuple[str, ...],
    ) -> tuple[CriticIssue, ...]:
        """Confirm findings against exact references and frozen conflict groups."""
        by_ref = {excerpt.ref: excerpt.text for excerpt in excerpts}
        visible = set(visible_conflict_ids)
        issues = []
        for section_id, claim in claims:
            evidence = tuple(by_ref[ref] for ref in claim.support_refs if ref in by_ref)
            if not evidence:
                continue
            candidates = []
            for group in conflicts:
                if group.conflict_id not in visible and claim.claim_id in group.claim_ids:
                    side_ids = {ref.source_id for side in group.sides for ref in side.support_refs}
                    candidates.append(("counter_evidence_gap", "surface_conflict", group.conflict_id, side_ids))
            verified = ReviewService.verify_report_claim(claim.text, evidence) == "rule_passed"
            if not verified:
                claim_denoms = _denominators(claim.text)
                claim_periods = _periods(claim.text)
                checks = (
                    ("survivorship_bias", lambda text: (
                        claim_denoms and _denominators(text) and claim_denoms.isdisjoint(_denominators(text))
                    ) or (_POPULATION_ALL.search(claim.text) and _POPULATION_SUBSET.search(text))),
                    ("temporal_skew", lambda text: (
                        claim_periods and _periods(text) and claim_periods.isdisjoint(_periods(text))
                    )),
                    ("causal_overreach", lambda text: (
                        _CAUSAL.search(claim.text) and not _NON_CAUSAL.search(claim.text)
                        and _OBSERVATIONAL.search(text) and not _CAUSAL.search(text)
                    )),
                )
                for kind, mismatches in checks:
                    if all(mismatches(text) for text in evidence):
                        candidates.append((kind, "mark_gap", None, set()))
            for kind, action, conflict_id, side_ids in candidates:
                identity = report_json([section_id, claim.claim_id, kind, conflict_id])
                issue_id = "issue-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                issues.append(CriticIssue(
                    section_id, claim.claim_id, kind, "high",
                    tuple(sorted(side_ids | {ref.source_id for ref in claim.support_refs})),
                    action, _REASONS[kind], conflict_id, kind, issue_id,
                ))
        return tuple(dict.fromkeys(issues))

    @staticmethod
    def evaluate(
        response: ReportModelResponse | None, *, sample: dict, confirmed: tuple[CriticIssue, ...],
        reason: str = "disabled",
    ) -> ReportCriticResult:
        if response is None:
            status = "timeout" if reason in {"timeout", "deadline"} else "skipped"
            return ReportCriticResult(status, confirmed, reason)
        payload = parse_report_json(response.text, byte_limit=8192) if response.finish_reason in {"stop", "end_turn"} else None
        rows = payload.get("issues") if payload and set(payload) == {"issues"} else None
        if not isinstance(rows, list) or len(rows) > CRITIC_MAX_ISSUES:
            return ReportCriticResult("invalid", confirmed, "invalid_output")
        targets = {(row["section_id"], row["claim_id"]) for row in sample["claims"]}
        targets.update((section_id, None) for section_id, _ in sample["coverage_gaps"])
        source_ids = {item["ref"]["source_id"] for item in sample["excerpts"]}
        conflict_ids = {item["conflict_id"] for item in sample["conflicts"]}
        fields = {"section_id", "claim_id", "kind", "severity", "source_ids", "proposed_action", "rationale"}
        for row in rows:
            if not isinstance(row, dict) or not fields <= row.keys() <= fields | {"conflict_id"}:
                return ReportCriticResult("invalid", confirmed, "invalid_output")
            if (
                any(not isinstance(row[key], str) for key in fields - {"claim_id", "source_ids"})
                or (row["claim_id"] is not None and not isinstance(row["claim_id"], str))
                or (row["section_id"], row["claim_id"]) not in targets
                or row["kind"] not in _REASONS or row["severity"] not in ("medium", "high")
                or row["proposed_action"] not in _ACTIONS or not 0 < len(row["rationale"]) <= 240
                or not isinstance(row["source_ids"], list) or len(row["source_ids"]) > CRITIC_MAX_EXCERPTS
                or any(not isinstance(sid, str) or sid not in source_ids for sid in row["source_ids"])
                or (row.get("conflict_id") is not None and (
                    not isinstance(row["conflict_id"], str) or row["conflict_id"] not in conflict_ids
                ))
            ):
                return ReportCriticResult("invalid", confirmed, "invalid_output")
        return ReportCriticResult("ok", confirmed)

    @staticmethod
    def plan_revision(
        draft_digest: str, issues: tuple[CriticIssue, ...], *, round_number: int,
    ) -> ReportRevisionPlan | None:
        if round_number != 0 or not issues:
            return None
        ranked = sorted(issues, key=lambda issue: (
            issue.severity != "high", issue.proposed_action != "surface_conflict",
        ))
        targets = {}
        for issue in ranked:
            key = (issue.section_id, issue.claim_id or issue.conflict_id)
            targets.setdefault(key, issue)
        return ReportRevisionPlan(draft_digest, tuple(targets.values())[:REVISION_MAX_TARGETS])

    @staticmethod
    def qualifies(claim: ReportClaim, text: str, excerpts: tuple[str, ...]) -> bool:
        """Allow an exact supported sentence only when its displayed numbers match the target."""
        return (
            ReviewService.verify_report_claim(text, excerpts) == "rule_passed"
            and not numeric_grounding_mismatch(text, [claim.text])
            and not numeric_grounding_mismatch(claim.text, [text])
            and not _CAUSAL.search(text)
        )
