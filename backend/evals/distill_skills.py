"""Offline SkillPack failure clustering, candidate distillation, and gated promotion pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import yaml

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _BACKEND_ROOT / "src"
for entry in (str(_SRC_ROOT), str(_BACKEND_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from evals.compare_variants import build_comparison  # noqa: E402
from evals.export_rl_trajectories import (  # noqa: E402
    compute_verifiable_reward,
    load_snapshots_from_dir,
)
from evals.loader import load_benchmark_cases  # noqa: E402
from evals.schema import BenchmarkCase  # noqa: E402
from services.skill_pack import (  # noqa: E402
    DISTILLATION_SOURCE,
    MATURITY_DRAFT,
    MATURITY_VERIFIED,
    NEUTRAL_PACK_ID,
    SKILL_PROMOTION_BENCHMARK,
    STATUS_ACTIVE,
    STATUS_SHADOW,
    SkillPack,
    compute_content_hash,
    load_pack,
    validate_candidate_against_parent,
    validate_pack,
)

_GOLDEN_BENCHMARK = _BACKEND_ROOT / "evals" / "benchmarks" / "golden_multidomain_24cases.jsonl"
_BENCHMARK_LABEL = SKILL_PROMOTION_BENCHMARK
_DRAFTS_DIR = _BACKEND_ROOT / "skills" / "candidates"
_REQUIRED_GATE_METRICS = (
    "hybrid_score", "verifier_clean_rate", "citation_validity_rate", "total_latency_ms", "estimated_cost",
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
    "timeline_inconsistency": {
        "stage": "review",
        "rule": "逐项核验时间线中事件发生、公告发布和政策生效的日期口径，标注仍不明确的节点。",
    },
    "comparison_asymmetry": {
        "stage": "report",
        "rule": "对比各选项时采用相同指标、口径与适用范围，分别注明缺失或不可比的项目。",
    },
}


def cluster_failure_trajectories(
    results: list[dict[str, Any]],
    *,
    min_support_cases: int = 3,
    hybrid_threshold: float = 0.72,
) -> dict[str, dict[str, Any]]:
    """Aggregate recurring review/verifier failure checks across low-scoring eval cases."""

    case_ids_by_check: dict[str, set[str]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        metrics = _mapping(row.get("metrics"))
        trace = _mapping(row.get("trace"))
        summary = _mapping(trace.get("review_summary") or metrics.get("review_summary"))
        score = _finite(metrics.get("hybrid_score"))
        cid = row.get("id") or row.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        raw_issues = row.get("review_issues") or metrics.get("review_issues") or summary.get("issues")
        if not isinstance(raw_issues, list):
            continue
        for issue in raw_issues:
            if not isinstance(issue, dict):
                continue
            check = str(issue.get("check") or "").strip().lower()
            if check in _FAILURE_GUIDANCE and (
                score is None or score < hybrid_threshold or row.get("error") or summary.get("repair_candidates")
                or issue.get("severity") in {"high", "medium"}
            ):
                case_ids_by_check.setdefault(check, set()).add(cid.strip())

    return {
        check: {
            "count": len(case_ids),
            "case_ids": sorted(case_ids),
        }
        for check in sorted(case_ids_by_check, key=lambda key: (-len(case_ids_by_check[key]), key))
        if len(case_ids := case_ids_by_check[check]) >= min_support_cases
    }


def mine_contrastive_traces(
    snapshots: list[dict[str, Any]],
    parent: SkillPack,
    *,
    min_top_traces: int,
    min_fail_traces: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Match completed traces to the active parent and select clean and failing cohorts."""
    matched: dict[tuple[str, str], dict[str, Any]] = {}
    for snapshot in snapshots:
        request_id = snapshot.get("request_id")
        tenant = snapshot.get("tenant_id") or "default"
        pack = _mapping(snapshot.get("skill_pack"))
        if (
            not isinstance(request_id, str) or not request_id or not isinstance(tenant, str)
            or pack.get("pack_id") != parent.pack_id or str(pack.get("version")) != str(parent.version)
            or pack.get("shadow") is not False or snapshot.get("phase") != "completed"
            or (pack.get("content_hash") and pack["content_hash"] != parent.content_hash)
        ):
            continue
        matched.setdefault((tenant, request_id), snapshot)

    clean_rewards: list[float] = []
    failures: list[dict[str, Any]] = []
    for (tenant, request_id), snapshot in matched.items():
        trace = _mapping(snapshot.get("request_metrics"))
        review = _mapping(snapshot.get("review_summary") or trace.get("review_summary"))
        issue_count = review.get("verifier_issue_count")
        repaired = trace.get("report_repair_triggered") is True or (_finite(snapshot.get("report_repair_cycles")) or 0) > 0
        blocked = review.get("overall_status") == "blocked"
        if repaired or blocked or (type(issue_count) is int and issue_count > 0):
            issues = review.get("issues")
            candidates = review.get("repair_candidates")
            issues = list(issues) if isinstance(issues, list) else []
            if isinstance(candidates, list):
                issues.extend({"check": item.get("origin_check")} for item in candidates if isinstance(item, dict))
            # Tenant and request identifiers stay out of draft and audit content.
            case_digest = hashlib.sha256(json.dumps([tenant, request_id]).encode()).hexdigest()[:32]
            failures.append({"id": case_digest, "review_issues": issues})
        elif (
            snapshot.get("status") == "success" and not trace.get("degraded")
            and type(issue_count) is int and issue_count == 0
        ):
            score = compute_verifiable_reward(snapshot)
            if score["verifier_clean_rate"] == 1 and score["citation_validity_rate"] > 0:
                clean_rewards.append(score["reward"])

    top = sorted(clean_rewards, reverse=True)[: math.ceil(0.2 * len(clean_rewards))]
    top_count = sum(score >= 0.70 for score in top)
    counts = {"top_clean_traces": top_count, "failed_traces": len(failures)}
    if top_count < min_top_traces or len(failures) < min_fail_traces:
        return [], counts
    return failures, counts


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

    digest = hashlib.sha256("\n".join(sorted(clusters)).encode()).hexdigest()[:16]
    candidate = SkillPack(
        pack_id=candidate_id if candidate_id is not None else parent.pack_id,
        schema_version=parent.schema_version,
        version=int(parent.version or 1) + 1,
        maturity=MATURITY_DRAFT,
        status=STATUS_SHADOW,
        title=parent.title,
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
        supersedes=f"{parent.pack_id}@v{parent.version}",
        derived_from=f"distill-{digest}",
        owner=parent.owner or "distiller",
        eval_meta={"source": DISTILLATION_SOURCE},
        skill_kind=parent.skill_kind,
        retrieval_profile=deepcopy(parent.retrieval_profile),
        verifiers=list(parent.verifiers),
        composable_with=list(parent.composable_with),
        outline_sections=list(parent.outline_sections),
        mcp_tool_scopes=deepcopy(parent.mcp_tool_scopes),
        code_sandbox=parent.code_sandbox,
    )
    candidate.content_hash = compute_content_hash(candidate)
    candidate.pinned_content_hash = candidate.content_hash
    return candidate


def render_skill_pack_markdown(pack: SkillPack) -> str:
    """Serialize a SkillPack to YAML-frontmatter + ## stage SKILL.md text."""

    frontmatter: dict[str, Any] = {
        "pack_id": pack.pack_id,
        "schema_version": pack.schema_version,
        "skill_kind": pack.skill_kind,
        "version": int(pack.version or 1),
        "content_hash": pack.content_hash,
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
    ]
    return f"---\n{header}\n---\n\n" + "\n\n".join(body_blocks) + "\n"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _result_rows(payload: dict[str, Any] | None) -> list[Any]:
    rows = _mapping(payload).get("results")
    return rows if isinstance(rows, list) else []


def _finite(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def evaluate_promotion_gate(
    *,
    validation_errors: list[str],
    comparison_report: dict[str, Any] | None,
    target_case_ids: frozenset[str] = frozenset(),
    min_hybrid_gain: float = 0.03,
    max_adversarial_regression: float = 0.02,
) -> dict[str, Any]:
    """Require valid aggregate, target-domain and adversarial evidence before verification."""

    reasons = list(validation_errors)
    comparison = _mapping(comparison_report)
    if comparison.get("verdict") != "pass":
        reasons.append("comparison_did_not_pass")
    guardrails = comparison.get("regression_guardrails_tripped")
    if isinstance(guardrails, list):
        reasons.extend(str(item) for item in guardrails)
    else:
        reasons.append("missing_comparison_guardrails")

    deltas = _mapping(comparison.get("aggregate_deltas"))
    hybrid_gain = _finite(deltas.get("hybrid_score"))
    if hybrid_gain is None or hybrid_gain < 0.015:
        reasons.append("insufficient_hybrid_gain")
    for metric in ("verifier_clean_rate", "citation_validity_rate"):
        delta = _finite(deltas.get(metric))
        if delta is None or delta < 0:
            reasons.append(f"{metric}_regression_or_missing")

    target_deltas: dict[str, float] = {}
    adversarial_deltas: list[float] = []
    cases = comparison.get("cases")
    for case in cases if isinstance(cases, list) else ():
        row = _mapping(case)
        cid = row.get("id")
        if not isinstance(cid, str):
            continue
        delta = _finite(_mapping(row.get("deltas")).get("hybrid_score"))
        if delta is None:
            reasons.append(f"invalid_hybrid_delta:{cid}")
            continue
        if cid in target_case_ids:
            target_deltas[cid] = delta
        if cid.startswith("adv_"):
            adversarial_deltas.append(delta)

    target_gain = _mean(list(target_deltas.values())) if target_deltas else None
    target_threshold = _finite(min_hybrid_gain)
    if (
        target_deltas.keys() != target_case_ids or target_gain is None or target_threshold is None
        or target_threshold < 0.03 or target_gain < target_threshold - 1e-12
    ):
        reasons.append("insufficient_target_domain_hybrid_gain")
    adversarial_min_delta = min(adversarial_deltas, default=None)
    adversarial_threshold = _finite(max_adversarial_regression)
    if (
        len(adversarial_deltas) != 4 or adversarial_threshold is None or not 0 < adversarial_threshold <= 0.02
        or adversarial_min_delta is None or -adversarial_min_delta >= adversarial_threshold - 1e-12
    ):
        reasons.append("adversarial_coverage_or_regression")
    improved = comparison.get("improved_cases")
    regressed = comparison.get("regressed_cases")
    if type(improved) is not int or type(regressed) is not int or not 0 <= regressed < improved <= 24:
        reasons.append("insufficient_improved_cases")
    if comparison.get("compared_cases") != 24 or not isinstance(cases, list) or len(cases) != 24:
        reasons.append("incomplete_golden_comparison")
    for metric in ("latency_delta_ratio", "cost_delta_ratio"):
        ratio = _finite(comparison.get(metric))
        if ratio is None or ratio > 0.25:
            reasons.append(f"{metric}_regression_or_missing")

    return {
        "approved": not reasons,
        "maturity": MATURITY_DRAFT if reasons else MATURITY_VERIFIED,
        "hybrid_gain": hybrid_gain,
        "target_domain_hybrid_delta": target_gain,
        "adversarial_min_delta": adversarial_min_delta,
        "reasons": reasons,
    }


def _evaluation_errors(
    payload: dict[str, Any],
    cases: list[BenchmarkCase],
    pack: SkillPack,
    *,
    parent_id: str,
    baseline: dict[str, Any] | None = None,
) -> list[str]:
    metadata = _mapping(payload.get("skill_evaluation"))
    if metadata != {
        "benchmark": _BENCHMARK_LABEL,
        "judge": "hybrid",
        "pack_id": pack.pack_id,
        "version": pack.version,
        "content_hash": compute_content_hash(pack),
    }:
        return ["unbound_golden_evaluation"]

    expected = {case.id: case for case in cases}
    errors = []
    results = payload.get("results")
    baseline_routes = {
        row["id"]: _mapping(row.get("trace")).get("skill_pack_id")
        for row in _result_rows(baseline)
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    target_executions = 0
    for item in results if isinstance(results, list) else ():
        row = _mapping(item)
        cid = row.get("id")
        if not isinstance(cid, str) or cid not in expected:
            continue
        case = expected[cid]
        metrics = _mapping(row.get("metrics"))
        if row.get("topic") != case.topic or metrics.get("judge_type") != "hybrid" or metrics.get("llm_judge_status") != "ok":
            errors.append(f"invalid_hybrid_evaluation:{cid}")
        trace = _mapping(row.get("trace"))
        _gate_metrics(row, pack)
        route = trace.get("skill_pack_id")
        expected_pack = case.metadata.get("skill_pack")
        if not isinstance(route, str) or not route:
            errors.append(f"missing_skill_route:{cid}")
        if expected_pack == NEUTRAL_PACK_ID and route == pack.pack_id:
            errors.append(f"neutral_route_hijacked:{cid}")
        if baseline is not None and (
            (baseline_routes.get(cid) == parent_id and route != pack.pack_id)
            or (baseline_routes.get(cid) != parent_id and route == pack.pack_id)
        ):
            errors.append(f"skill_route_changed:{cid}")
        if route == pack.pack_id:
            if str(trace.get("skill_pack_version")) != str(pack.version) or trace.get("skill_pack_shadow") is not False:
                errors.append(f"target_pack_not_executed:{cid}")
            if expected_pack == parent_id:
                target_executions += 1
    if not target_executions:
        errors.append("no_target_domain_execution")
    return errors


def _write_artifact(path: Path, content: str) -> None:
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _draft_directory(output_dir: Path, candidate: SkillPack, parent: SkillPack, config: Any) -> Path:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", candidate.pack_id):
        raise ValueError("candidate_id must be a kebab-case SkillPack ID")
    draft_root = output_dir.expanduser().resolve()
    destination = draft_root / candidate.pack_id
    runtime_roots = [
        _BACKEND_ROOT / "skills" / "research",
        _BACKEND_ROOT / "skills" / "formats",
        Path(config.skill_pack_dir),
        Path(config.skill_format_dir),
    ]
    if parent.source_path:
        runtime_roots.append(Path(parent.source_path).resolve().parents[1])
    if destination.is_symlink() or any(
        destination.is_relative_to(root.expanduser().resolve())
        or root.expanduser().resolve().is_relative_to(destination)
        for root in runtime_roots
    ):
        raise ValueError("candidate output must be isolated from runtime and parent SkillPack directories")
    return destination


def _gate_metrics(row: dict[str, Any], pack: SkillPack) -> None:
    """Derive verification scores from the persisted review and claim data."""
    metrics = _mapping(row.get("metrics"))
    metrics.pop("citation_validity_rate", None)
    metrics.pop("verifier_clean_rate", None)
    trace = _mapping(row.get("trace"))
    review = _mapping(trace.get("review_summary"))
    severities = _mapping(review.get("severity_counts"))
    verifier_issues = review.get("verifier_issue_count")
    executed = review.get("verifiers_executed")
    if not isinstance(review.get("severity_counts"), dict) or not isinstance(executed, list):
        return
    if trace.get("skill_pack_id") == pack.pack_id and not set(pack.verifiers).issubset(
        item for item in executed if isinstance(item, str)
    ):
        return
    if any(type(count) is not int or count < 0 for count in (verifier_issues, *severities.values())):
        return
    task_rows = row.get("todo_items")
    if not isinstance(task_rows, list):
        return
    quality = compute_verifiable_reward({"todo_items": task_rows, "request_metrics": trace})
    metrics["citation_validity_rate"] = quality["citation_validity_rate"]
    metrics["verifier_clean_rate"] = quality["verifier_clean_rate"]


def run_golden_ab_gate(
    parent: SkillPack,
    candidate: SkillPack,
    *,
    config: Any,
    judge: Any = None,
    agent_factory: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run each Golden variant in a temporary, independently configured environment."""
    from evals.judges.hybrid import HybridJudge
    from evals.judges.llm import LLMJudge
    from evals.runner import run_benchmark_suite

    cases = load_benchmark_cases(_GOLDEN_BENCHMARK)
    judge = judge if judge is not None else HybridJudge(llm_judge=LLMJudge(config=config, load_env_file=False))
    if not isinstance(judge, HybridJudge):
        raise ValueError("Golden evaluation requires HybridJudge")
    with TemporaryDirectory(prefix="deepresearch-skill-gate-") as directory:
        root = Path(directory)
        payloads = []
        for label, pack in (("baseline", parent), ("candidate", candidate)):
            variant = root / label
            packs_dir = variant / "skills"
            source = Path(config.skill_pack_dir).expanduser().resolve()
            if source.is_dir():
                shutil.copytree(source, packs_dir, symlinks=True)
            for target in (packs_dir / parent.pack_id, packs_dir / pack.pack_id):
                if target.is_symlink():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
            pack_dir = packs_dir / pack.pack_id
            pack_dir.mkdir(parents=True)
            evaluation_pack = replace(pack, status=STATUS_ACTIVE, maturity=MATURITY_DRAFT, supersedes="",
                                      derived_from="", eval_meta={})
            (pack_dir / "SKILL.md").write_text(render_skill_pack_markdown(evaluation_pack), encoding="utf-8")
            evaluation_config = config.model_copy(update={
                "skill_pack_enabled": True, "skill_pack_dir": str(packs_dir),
                "skill_pack_maturity_gate": MATURITY_DRAFT, "request_state_dir": str(variant / "state"),
                "notes_workspace": str(variant / "notes"),
                "note_memory_dir": str(variant / "note-memory"), "note_memory_enabled": False,
                "strategy_memory_enabled": False, "evidence_memory_enabled": False,
                "search_cache_enabled": False, "policy_model_enabled": False, "policy_model_mode": "off",
                "database_url": None, "redis_url": None, "workflow_runner": "inline",
                "sandbox_pyodide_cache_dir": str(variant / "sandbox-cache"),
                "langfuse_enabled": False, "mcp_enforce_readonly": True, "mcp_allow_hitl_write_tools": False,
            })
            payload = run_benchmark_suite(
                cases, config=evaluation_config, judge=judge, agent_factory=agent_factory,
                benchmark_path=_GOLDEN_BENCHMARK, variant=label, request_id_prefix=f"skill-gate-{label}",
            )
            for row in payload["results"]:
                _gate_metrics(row, pack)
            payload["skill_evaluation"] = {
                "benchmark": _BENCHMARK_LABEL, "judge": "hybrid", "pack_id": pack.pack_id,
                "version": pack.version, "content_hash": compute_content_hash(pack),
            }
            payloads.append(payload)
        return payloads[0], payloads[1]


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
    config: Any = None,
    run_golden: bool = False,
    judge: Any = None,
    agent_factory: Any = None,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    from config import Configuration

    config = config if config is not None else Configuration.from_env(load_env_file=False)
    if type(min_support_cases) is not int or min_support_cases < 1:
        raise ValueError("min_support_cases must be a positive integer")
    parent_errors = validate_pack(parent_pack, char_limit=char_limit)
    if parent_pack.is_neutral or parent_pack.status != STATUS_ACTIVE or parent_errors:
        raise ValueError(f"invalid active distillation parent: {parent_errors}")
    cases = load_benchmark_cases(_GOLDEN_BENCHMARK)
    expected_ids = frozenset(case.id for case in cases)
    target_ids = frozenset(case.id for case in cases if case.metadata.get("skill_pack") == parent_pack.pack_id)
    raw_results = _result_rows(results_payload)
    mining_counts = None
    if snapshot_dir is not None:
        raw_results, mining_counts = mine_contrastive_traces(
            load_snapshots_from_dir(snapshot_dir), parent_pack,
            min_top_traces=config.skill_distill_min_top_traces,
            min_fail_traces=config.skill_distill_min_fail_traces,
        )
    clusters = cluster_failure_trajectories(
        raw_results,
        min_support_cases=min_support_cases,
    )
    candidate = synthesize_candidate_pack(
        parent_pack,
        clusters,
        candidate_id=candidate_id,
        char_limit=char_limit,
    )
    candidate_dir = _draft_directory(output_dir, candidate, parent_pack, config)
    validation_errors = validate_candidate_against_parent(
        candidate,
        parent_pack,
        forbidden_case_ids=tuple(sorted(expected_ids)) + tuple(forbidden_case_ids),
        char_limit=char_limit,
    )
    if candidate.body_sections == parent_pack.body_sections:
        validation_errors.append("no_eligible_methodology_change")
    synthesis_errors = list(validation_errors)
    if not target_ids:
        validation_errors.append("golden_benchmark_has_no_target_domain_cases")

    comparison = None
    if run_golden and (baseline_payload is not None or candidate_eval_payload is not None):
        raise ValueError("select one source of Golden evaluation results")
    if run_golden and not validation_errors:
        baseline_payload, candidate_eval_payload = run_golden_ab_gate(
            parent_pack, candidate, config=config, judge=judge, agent_factory=agent_factory,
        )
    if not validation_errors and baseline_payload is not None and candidate_eval_payload is not None:
        baseline_payload = deepcopy(baseline_payload)
        candidate_eval_payload = deepcopy(candidate_eval_payload)
        validation_errors.extend(_evaluation_errors(
            baseline_payload, cases, parent_pack, parent_id=parent_pack.pack_id,
        ))
        validation_errors.extend(_evaluation_errors(
            candidate_eval_payload, cases, candidate, parent_id=parent_pack.pack_id, baseline=baseline_payload,
        ))
        comparison = build_comparison(
            baseline_payload,
            candidate_eval_payload,
            max_adversarial_regression=max_adversarial_regression,
            max_latency_regression_ratio=min(0.25, getattr(config, "skill_distill_max_latency_regression_ratio", 0.25)),
            max_cost_regression_ratio=min(0.25, getattr(config, "skill_distill_max_cost_regression_ratio", 0.25)),
            required_metrics=_REQUIRED_GATE_METRICS,
            expected_case_ids=expected_ids,
        )

    gate = evaluate_promotion_gate(
        validation_errors=validation_errors,
        comparison_report=comparison,
        target_case_ids=target_ids,
        min_hybrid_gain=min_hybrid_gain,
        max_adversarial_regression=max_adversarial_regression,
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = candidate_dir / "comparison.json"
    if comparison is not None:
        _write_artifact(comparison_path, json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False))
    audit = {"source": DISTILLATION_SOURCE}
    if gate["approved"] and comparison is not None:
        assert baseline_payload is not None and candidate_eval_payload is not None
        audit.update({
            "verdict": "pass", "benchmark": _BENCHMARK_LABEL, "judge": "hybrid",
            "baseline_version": parent_pack.version, "baseline_content_hash": compute_content_hash(parent_pack),
            "candidate_content_hash": candidate.content_hash,
            "compared_cases": comparison["compared_cases"], "improved_cases": comparison["improved_cases"],
            "regressed_cases": comparison["regressed_cases"], "hybrid_score_delta": gate["hybrid_gain"],
            "target_domain_hybrid_delta": gate["target_domain_hybrid_delta"],
            "adversarial_min_delta": gate["adversarial_min_delta"],
            "citation_validity_rate_delta": comparison["aggregate_deltas"]["citation_validity_rate"],
            "verifier_clean_rate_baseline": _mean([
                row["metrics"]["verifier_clean_rate"] for row in baseline_payload["results"]
            ]),
            "verifier_clean_rate_candidate": _mean([
                row["metrics"]["verifier_clean_rate"] for row in candidate_eval_payload["results"]
            ]),
            "latency_delta_ratio": comparison["latency_delta_ratio"],
            "cost_delta_ratio": comparison["cost_delta_ratio"], "comparison_artifact": str(comparison_path),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })
    skill_path = None
    if not synthesis_errors:
        emitted_pack = replace(candidate, maturity=gate["maturity"], eval_meta=audit)
        skill_path = candidate_dir / "SKILL.md"
        _write_artifact(skill_path, render_skill_pack_markdown(emitted_pack))

    report = {
        "candidate_id": candidate.pack_id,
        "parent_id": parent_pack.pack_id,
        "clusters": clusters,
        "snapshot_mining": mining_counts,
        "validation_errors": validation_errors,
        "promotion_gate": gate,
        "skill_path": str(skill_path) if skill_path else None,
        "comparison_path": str(comparison_path) if comparison is not None else None,
    }
    report_path = candidate_dir / "distill_report.json"
    _write_artifact(report_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


def main(argv: list[str] | None = None) -> int:
    from config import Configuration

    parser = argparse.ArgumentParser(description="Offline SkillPack distillation and promotion gate.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results", help="Path to eval results JSON file")
    source.add_argument("--snapshots", type=Path, help="Request snapshot directory (default: configured state directory)")
    parser.add_argument("--parent-pack", required=True, help="Path to parent SKILL.md file")
    parser.add_argument("--output-dir", type=Path, default=_DRAFTS_DIR, help="Isolated draft output directory")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--baseline-eval", default=None, help="Optional baseline compare JSON")
    parser.add_argument("--candidate-eval", default=None, help="Optional candidate compare JSON")
    parser.add_argument("--run-golden", action="store_true", help="Run both isolated Golden HybridJudge variants")
    parser.add_argument("--forbidden-cases", default="", help="Comma-separated forbidden case IDs")
    parser.add_argument("--min-support-cases", type=int, default=None)
    parser.add_argument("--min-hybrid-gain", type=float, default=None)
    parser.add_argument("--max-adversarial-regression", type=float, default=None)
    args = parser.parse_args(argv)
    if bool(args.baseline_eval) != bool(args.candidate_eval):
        parser.error("--baseline-eval and --candidate-eval must be provided together")
    if args.run_golden and args.baseline_eval:
        parser.error("--run-golden cannot be combined with precomputed evaluation files")

    results_payload = json.loads(Path(args.results).read_text(encoding="utf-8")) if args.results else {}
    parent_pack = load_pack(Path(args.parent_pack))
    base_eval = json.loads(Path(args.baseline_eval).read_text(encoding="utf-8")) if args.baseline_eval else None
    cand_eval = json.loads(Path(args.candidate_eval).read_text(encoding="utf-8")) if args.candidate_eval else None
    forbidden = [s.strip() for s in args.forbidden_cases.split(",") if s.strip()]
    config = Configuration.from_env(load_env_file=args.run_golden)

    report = run_distillation(
        results_payload=results_payload,
        parent_pack=parent_pack,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        baseline_payload=base_eval,
        candidate_eval_payload=cand_eval,
        forbidden_case_ids=forbidden,
        min_support_cases=(
            args.min_support_cases if args.min_support_cases is not None else config.skill_distill_min_support_cases
        ),
        min_hybrid_gain=(
            args.min_hybrid_gain if args.min_hybrid_gain is not None else config.skill_distill_min_hybrid_gain
        ),
        max_adversarial_regression=(
            args.max_adversarial_regression if args.max_adversarial_regression is not None
            else config.skill_distill_max_adversarial_regression
        ),
        config=config,
        run_golden=args.run_golden,
        snapshot_dir=None if args.results else args.snapshots or Path(config.request_state_dir),
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
    evaluated = args.run_golden or args.baseline_eval is not None
    return int(not report["skill_path"] or (evaluated and not report["promotion_gate"]["approved"]))


if __name__ == "__main__":
    raise SystemExit(main())
