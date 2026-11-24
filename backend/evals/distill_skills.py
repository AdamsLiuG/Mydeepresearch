"""Offline SkillPack failure clustering, candidate distillation, and gated promotion pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _BACKEND_ROOT / "src"
for entry in (str(_SRC_ROOT), str(_BACKEND_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from evals.compare_variants import build_comparison  # noqa: E402
from services.skill_pack import (  # noqa: E402
    MATURITY_DRAFT,
    MATURITY_VERIFIED,
    STATUS_SHADOW,
    SkillPack,
    compute_content_hash,
    load_pack,
    validate_candidate_against_parent,
)

_FAILURE_GUIDANCE: dict[str, dict[str, str]] = {
    "stale_evidence": {
        "stage": "task_execution",
        "rule": "优先检索近 12 个月官方公告与一线发布日期，并在总结中显式标注年份口径。",
    },
    "low_source_diversity": {
        "stage": "task_execution",
        "rule": "对同一核心结论至少交叉检索两类不同独立域名与权威数据源，避免单一站点偏见。",
    },
    "claim_contradicted": {
        "stage": "review",
        "rule": "核查关键数据与引用原文数值/极性是否一致，发现口径差异时保留分歧来源并标注适用边界。",
    },
    "numeric_audit_mismatch": {
        "stage": "report",
        "rule": "报告正文中所有量化数值必须与来源原文单位及口径保持一致，严禁估算替代实测值。",
    },
    "weak_evidence": {
        "stage": "planning",
        "rule": "拆解子任务时显式指定一手来源与对比维度检索词，减少泛化概述。",
    },
}


def cluster_failure_trajectories(
    results: list[dict[str, Any]],
    *,
    min_support_cases: int = 3,
    hybrid_threshold: float = 0.72,
) -> dict[str, dict[str, Any]]:
    """Aggregate recurring review/verifier failure checks across low-scoring eval cases."""

    counts: Counter[str] = Counter()
    case_ids_by_check: dict[str, list[str]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") or {}
        score = metrics.get("hybrid_score")
        has_error = bool(row.get("error"))
        raw_issues = row.get("review_issues") or metrics.get("review_issues") or []
        if score is not None and float(score) >= hybrid_threshold and not has_error and not raw_issues:
            continue
        cid = str(row.get("id") or row.get("case_id") or "").strip()
        seen_in_case: set[str] = set()
        for issue in raw_issues:
            if not isinstance(issue, dict):
                continue
            check = str(issue.get("check") or "").strip().lower()
            if check and check not in seen_in_case:
                counts[check] += 1
                case_ids_by_check.setdefault(check, []).append(cid)
                seen_in_case.add(check)

    return {
        check: {
            "count": cnt,
            "case_ids": sorted(set(case_ids_by_check.get(check, []))),
        }
        for check, cnt in counts.most_common()
        if cnt >= max(1, int(min_support_cases))
    }


def synthesize_candidate_pack(
    parent: SkillPack,
    clusters: dict[str, dict[str, Any]],
    *,
    candidate_id: str | None = None,
    char_limit: int = 1200,
) -> SkillPack:
    """Create a candidate SkillPack preserving parent invariants while adding clustered guidance."""

    new_sections = dict(parent.body_sections)
    for check in clusters:
        guidance = _FAILURE_GUIDANCE.get(check)
        if not guidance:
            continue
        stage = guidance["stage"]
        rule = guidance["rule"]
        existing = str(new_sections.get(stage) or "").strip()
        if rule not in existing:
            candidate_text = f"{existing}\n- {rule}".strip() if existing else f"- {rule}"
            if len(candidate_text) <= char_limit:
                new_sections[stage] = candidate_text

    candidate = SkillPack(
        pack_id=candidate_id or f"{parent.pack_id}-evolved",
        schema_version=parent.schema_version,
        version=int(parent.version or 1) + 1,
        maturity=MATURITY_DRAFT,
        status=STATUS_SHADOW,
        title=f"{parent.title} (Candidate)",
        topic_domain=parent.topic_domain,
        applicable_when=parent.applicable_when,
        match_signals=list(parent.match_signals),
        negative_signals=list(parent.negative_signals),
        min_match_score=parent.min_match_score,
        stage_scope=list(parent.stage_scope),
        quality_gates=deepcopy(parent.quality_gates),
        budget_knobs=deepcopy(parent.budget_knobs),
        layout=deepcopy(parent.layout),
        body_sections=new_sections,
        supersedes=parent.pack_id,
        derived_from=f"{parent.pack_id}@v{parent.version}",
        owner=parent.owner or "distiller",
        eval_meta={"source": "offline_distillation", "clusters": list(clusters.keys())},
        skill_kind=parent.skill_kind,
        retrieval_profile=deepcopy(parent.retrieval_profile),
        verifiers=list(parent.verifiers),
        composable_with=list(parent.composable_with),
        outline_sections=list(parent.outline_sections),
        mcp_tool_scopes=deepcopy(parent.mcp_tool_scopes),
        code_sandbox=parent.code_sandbox,
    )
    candidate.content_hash = compute_content_hash(candidate)
    return candidate


def render_skill_pack_markdown(pack: SkillPack) -> str:
    """Serialize a SkillPack to YAML-frontmatter + ## stage SKILL.md text."""

    frontmatter: dict[str, Any] = {
        "pack_id": pack.pack_id,
        "schema_version": pack.schema_version,
        "skill_kind": pack.skill_kind,
        "version": int(pack.version or 1),
        "title": pack.title,
        "topic_domain": pack.topic_domain,
        "applicable_when": pack.applicable_when,
        "status": pack.status,
        "maturity": pack.maturity,
        "match_signals": list(pack.match_signals),
        "negative_signals": list(pack.negative_signals),
        "min_match_score": float(pack.min_match_score),
        "stage_scope": list(pack.stage_scope),
        "verifiers": list(pack.verifiers),
    }
    if pack.supersedes:
        frontmatter["supersedes"] = pack.supersedes
    if pack.derived_from:
        frontmatter["derived_from"] = pack.derived_from
    if pack.owner:
        frontmatter["owner"] = pack.owner
    if pack.quality_gates:
        frontmatter["quality_gates"] = dict(pack.quality_gates)
    if pack.budget_knobs:
        frontmatter["budget_knobs"] = dict(pack.budget_knobs)
    if pack.layout:
        frontmatter["layout"] = dict(pack.layout)
    if pack.retrieval_profile:
        frontmatter["retrieval_profile"] = dict(pack.retrieval_profile)
    if pack.composable_with:
        frontmatter["composable_with"] = list(pack.composable_with)
    if pack.outline_sections:
        frontmatter["outline_sections"] = list(pack.outline_sections)
    if pack.mcp_tool_scopes:
        frontmatter["mcp_tool_scopes"] = dict(pack.mcp_tool_scopes)
    if pack.code_sandbox:
        frontmatter["code_sandbox"] = True
    if pack.eval_meta:
        frontmatter["eval"] = dict(pack.eval_meta)

    header = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    body_blocks = [
        f"## {stage}\n\n{body.strip()}"
        for stage, body in pack.body_sections.items()
        if body.strip()
    ]
    return f"---\n{header}\n---\n\n" + "\n\n".join(body_blocks) + "\n"


def evaluate_promotion_gate(
    *,
    validation_errors: list[str],
    comparison_report: dict[str, Any] | None,
    min_hybrid_gain: float = 0.03,
    max_adversarial_regression: float = 0.02,
) -> dict[str, Any]:
    """Decide whether a candidate SkillPack earns MATURITY_VERIFIED or must be rejected."""

    reasons: list[str] = list(validation_errors)
    if comparison_report is None:
        reasons.append("missing_comparison_report")
        return {"approved": False, "maturity": MATURITY_DRAFT, "reasons": reasons}

    guardrails = list(comparison_report.get("regression_guardrails_tripped") or [])
    if guardrails:
        reasons.extend(guardrails)

    agg_deltas = comparison_report.get("aggregate_deltas") or {}
    hybrid_gain = float(
        agg_deltas.get("hybrid_score")
        if "hybrid_score" in agg_deltas
        else comparison_report.get("normalized_quality_gain") or 0.0
    )
    if hybrid_gain < float(min_hybrid_gain):
        reasons.append(f"insufficient_hybrid_gain:{hybrid_gain:.4f}<{min_hybrid_gain:.4f}")

    for case_cmp in comparison_report.get("cases") or []:
        cid = str(case_cmp.get("id") or "")
        if cid.lower().startswith("adv_"):
            h_delta = float((case_cmp.get("deltas") or {}).get("hybrid_score") or 0.0)
            if -h_delta > float(max_adversarial_regression):
                reasons.append(f"adversarial_drop:{cid}:{h_delta:.4f}")

    approved = len(reasons) == 0
    return {
        "approved": approved,
        "maturity": MATURITY_VERIFIED if approved else MATURITY_DRAFT,
        "hybrid_gain": round(hybrid_gain, 6),
        "reasons": reasons,
    }


def run_distillation(
    *,
    results_payload: dict[str, Any],
    parent_pack: SkillPack,
    output_dir: Path,
    candidate_id: str | None = None,
    baseline_payload: dict[str, Any] | None = None,
    candidate_eval_payload: dict[str, Any] | None = None,
    forbidden_case_ids: list[str] | tuple[str, ...] = (),
    min_support_cases: int = 3,
    min_hybrid_gain: float = 0.03,
    max_adversarial_regression: float = 0.02,
    char_limit: int = 1200,
) -> dict[str, Any]:
    clusters = cluster_failure_trajectories(
        list(results_payload.get("results") or []),
        min_support_cases=min_support_cases,
    )
    candidate = synthesize_candidate_pack(
        parent_pack,
        clusters,
        candidate_id=candidate_id,
        char_limit=char_limit,
    )
    validation_errors = validate_candidate_against_parent(
        candidate,
        parent_pack,
        forbidden_case_ids=forbidden_case_ids,
        char_limit=char_limit,
    )

    comparison = None
    if baseline_payload is not None and candidate_eval_payload is not None:
        comparison = build_comparison(
            baseline_payload,
            candidate_eval_payload,
            max_adversarial_regression=max_adversarial_regression,
        )

    gate = evaluate_promotion_gate(
        validation_errors=validation_errors,
        comparison_report=comparison,
        min_hybrid_gain=min_hybrid_gain,
        max_adversarial_regression=max_adversarial_regression,
    )
    candidate_dir = output_dir / candidate.pack_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    # Reflect gate maturity in emitted draft artifact
    emitted_pack = SkillPack(
        **{
            **candidate.__dict__,
            "maturity": gate["maturity"],
            "eval_meta": {
                **dict(candidate.eval_meta),
                "verdict": "pass" if gate["approved"] else "draft_or_rejected",
            },
        }
    )
    skill_path = candidate_dir / "SKILL.md"
    skill_path.write_text(render_skill_pack_markdown(emitted_pack), encoding="utf-8")

    report = {
        "candidate_id": candidate.pack_id,
        "parent_id": parent_pack.pack_id,
        "clusters": clusters,
        "validation_errors": validation_errors,
        "promotion_gate": gate,
        "skill_path": str(skill_path),
    }
    report_path = candidate_dir / "distill_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline SkillPack distillation and promotion gate.")
    parser.add_argument("--results", required=True, help="Path to eval results JSON file")
    parser.add_argument("--parent-pack", required=True, help="Path to parent SKILL.md file")
    parser.add_argument("--output-dir", default="backend/skills/candidates", help="Output directory")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--baseline-eval", default=None, help="Optional baseline compare JSON")
    parser.add_argument("--candidate-eval", default=None, help="Optional candidate compare JSON")
    parser.add_argument("--forbidden-cases", default="", help="Comma-separated forbidden case IDs")
    parser.add_argument("--min-support-cases", type=int, default=3)
    parser.add_argument("--min-hybrid-gain", type=float, default=0.03)
    parser.add_argument("--max-adversarial-regression", type=float, default=0.02)
    args = parser.parse_args(argv)

    results_payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    parent_pack = load_pack(Path(args.parent_pack))
    base_eval = json.loads(Path(args.baseline_eval).read_text(encoding="utf-8")) if args.baseline_eval else None
    cand_eval = json.loads(Path(args.candidate_eval).read_text(encoding="utf-8")) if args.candidate_eval else None
    forbidden = [s.strip() for s in args.forbidden_cases.split(",") if s.strip()]

    report = run_distillation(
        results_payload=results_payload,
        parent_pack=parent_pack,
        output_dir=Path(args.output_dir),
        candidate_id=args.candidate_id,
        baseline_payload=base_eval,
        candidate_eval_payload=cand_eval,
        forbidden_case_ids=forbidden,
        min_support_cases=args.min_support_cases,
        min_hybrid_gain=args.min_hybrid_gain,
        max_adversarial_regression=args.max_adversarial_regression,
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
