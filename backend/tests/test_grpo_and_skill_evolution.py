from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BACKEND_ROOT / "src"
for entry in (str(SRC_ROOT), str(BACKEND_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from config import Configuration  # noqa: E402
from evals.compare_variants import build_comparison  # noqa: E402
from evals.distill_skills import run_distillation  # noqa: E402
from evals.export_rl_trajectories import (  # noqa: E402
    compute_trajectory_reward,
    export_grpo_records,
    load_snapshots_from_dir,
)
from evals.export_rl_trajectories import main as exporter_main  # noqa: E402
from evals.rl_training.reward_functions import (  # noqa: E402
    budget_legality_reward_fn,
    compute_score,
    json_format_reward_fn,
    verifiable_policy_reward_fn,
)
from evals.rl_training.train_grpo_trl import main as grpo_train_main  # noqa: E402
from services.policy_controller import (  # noqa: E402
    ModelPolicyController,
    PolicyStateFeatures,
    ShadowCanaryPolicyController,
    build_policy_controller,
    format_policy_messages,
)
from services.skill_pack import (  # noqa: E402
    MATURITY_DRAFT,
    MATURITY_PROMOTED,
    MATURITY_VERIFIED,
    SkillPack,
    SkillPackService,
    load_pack,
    validate_candidate_against_parent,
)


def _make_features(**overrides: Any) -> PolicyStateFeatures:
    base: dict[str, Any] = {
        "task_id": 1,
        "round_index": 1,
        "max_rounds": 3,
        "evidence_count": 2,
        "distinct_domains": 1,
        "high_quality_sources": 0,
        "has_gap_signals": True,
        "searches_left": 2,
        "fetches_left": 1,
        "remaining_budget_ratio": 1.0,
        "task_title": "Benchmark Task",
        "task_intent": "Gather evidence",
        "gap_signals": ("low_source_diversity",),
        "can_fetch": True,
        "skill_pack_id": "tech-research",
        "topic_domain": "ai",
        "freshness_ok": True,
        "evidence_sufficiency": False,
        "archive_repair_hits": 1,
        "recent_actions": ("search",),
    }
    base.update(overrides)
    return PolicyStateFeatures(**base)


def test_reward_severity_counts_and_recursive_snapshot_discovery(tmp_path: Path) -> None:
    snap = {
        "request_id": "r1",
        "topic": "Semiconductors",
        "status": "success",
        "report_markdown": "Detailed research synthesis " * 12,
        "todo_items": [{"id": 1, "status": "completed"}],
        "request_metrics": {
            "total_tasks": 1,
            "completed_tasks": 1,
            "review_summary": {
                "severity_counts": {"high": 2, "medium": 1, "low": 0},
                "verifier_issue_count": 1,
            },
        },
    }
    reward = compute_trajectory_reward(snap)
    # Two high severity issues (-0.6), one medium (-0.05), one verifier (-0.08) => review_score = 0.27
    assert abs(reward["review_score"] - 0.27) < 1e-4

    nested = tmp_path / "tenant-a" / "sub"
    nested.mkdir(parents=True)
    (nested / "snap1.json").write_text(json.dumps(snap), encoding="utf-8")
    (nested / ".hidden.json").write_text(json.dumps(snap), encoding="utf-8")
    loaded = load_snapshots_from_dir(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["request_id"] == "r1"


def test_grpo_export_filtering_cli_and_dry_run_trainer(tmp_path: Path) -> None:
    feat = _make_features().to_dict()
    snapshots: list[dict[str, Any]] = []

    # Group 1 ("grpo topic", task 1, round 1): 4 diverse outcomes -> retained
    actions_and_ok = [
        ("search", 2, 2, False),
        ("fetch", 2, 2, False),
        ("stop", 1, 0, True),  # degraded / low reward
        ("reflect", 2, 1, False),
    ]
    for idx, (act, comp, searches, is_deg) in enumerate(actions_and_ok):
        snapshots.append(
            {
                "request_id": f"req-ok-{idx}",
                "topic": "GRPO Topic",
                "status": "success" if not is_deg else "failed",
                "report_markdown": ("Report prose " * 20) if not is_deg else "",
                "todo_items": [{"id": 1, "status": "completed" if not is_deg else "pending"}],
                "request_metrics": {
                    "request_id": f"req-ok-{idx}",
                    "topic": "GRPO Topic",
                    "status": "success" if not is_deg else "failed",
                    "total_tasks": 2,
                    "completed_tasks": comp,
                    "degraded": is_deg,
                    "policy_decisions": [
                        {
                            "task_id": 1,
                            "round": 1,
                            "features": feat,
                            "decision": {
                                "action": act,
                                "confidence": 0.85,
                                "reason": "policy choice",
                            },
                        }
                    ],
                },
            }
        )

    # Group 2 ("zero variance topic"): 4 identical choices -> dropped due to std < min_reward_std
    for idx in range(4):
        snapshots.append(
            {
                "request_id": f"req-same-{idx}",
                "topic": "Flat Topic",
                "status": "success",
                "report_markdown": "Report prose " * 20,
                "todo_items": [{"id": 1, "status": "completed"}],
                "request_metrics": {
                    "request_id": f"req-same-{idx}",
                    "topic": "Flat Topic",
                    "total_tasks": 1,
                    "completed_tasks": 1,
                    "policy_decisions": [
                        {
                            "task_id": 1,
                            "round": 1,
                            "features": feat,
                            "decision": {"action": "search", "confidence": 0.9, "reason": "same"},
                        }
                    ],
                },
            }
        )

    records = export_grpo_records(snapshots, min_group_size=4, min_reward_std=0.03)
    assert len(records) == 1
    assert records[0]["metadata"]["topic_key"] == "grpo topic"
    assert len(records[0]["candidate_responses"]) == 4
    expected_msgs = format_policy_messages(_make_features())
    assert records[0]["messages"] == expected_msgs

    in_dir = tmp_path / "states"
    out_dir = tmp_path / "exports"
    in_dir.mkdir()
    for i, s in enumerate(snapshots):
        (in_dir / f"s_{i}.json").write_text(json.dumps(s), encoding="utf-8")

    rc = exporter_main(
        [
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--mode",
            "grpo-group",
            "--min-group-size",
            "4",
            "--min-reward-std",
            "0.03",
        ]
    )
    assert rc == 0
    grpo_file = out_dir / "grpo_groups.jsonl"
    assert grpo_file.is_file()
    lines = grpo_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    # Trainer --dry-run runs with zero GPU/TRL dependencies
    train_rc = grpo_train_main(["--dataset-path", str(grpo_file), "--dry-run"])
    assert train_rc == 0


def test_rlvr_reward_functions_penalize_malformed_and_illegal_budgets() -> None:
    zero_search_feat = _make_features(searches_left=0, fetches_left=0, has_gap_signals=True).to_dict()
    valid_stop = json.dumps({"action": "stop", "confidence": 0.9, "reason": "budget exhausted"})
    illegal_search = json.dumps({"action": "search", "confidence": 0.9, "reason": "want search"})
    malformed = "not a json string"

    fmt_scores = json_format_reward_fn([valid_stop, illegal_search, malformed])
    assert fmt_scores[0] == 1.0
    assert fmt_scores[2] == -1.0

    bud_scores = budget_legality_reward_fn(
        [valid_stop, illegal_search, malformed],
        state_features=[zero_search_feat, zero_search_feat, zero_search_feat],
    )
    assert bud_scores[0] > 0
    assert bud_scores[1] < 0
    assert bud_scores[2] < 0

    total = verifiable_policy_reward_fn(
        [valid_stop, illegal_search],
        state_features=[zero_search_feat, zero_search_feat],
    )
    assert total[0] > total[1]
    single = compute_score("deepresearch", valid_stop, extra_info={"state_features": zero_search_feat})
    assert abs(single - total[0]) < 1e-6


def test_shadow_canary_and_circuit_breaker_controllers() -> None:
    calls = 0

    def _model_search(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"action": "fetch", "confidence": 0.91, "reason": "need primary doc"}
                        )
                    }
                }
            ]
        }

    model_ctrl = ModelPolicyController(
        model_id="policy-v3",
        base_url="http://127.0.0.1:8001/v1",
        transport=_model_search,
    )

    # 1. Shadow mode evaluates model & emits agreement telemetry, NEVER overrides rule output
    shadow = ShadowCanaryPolicyController(mode="shadow", model_controller=model_ctrl)
    feat = _make_features(gap_signals=("thin_content",), fetches_left=1, can_fetch=True)
    decision, telemetry = shadow.decide_with_context(feat, request_id="r-shadow-1")
    assert decision.source == "rule"
    assert calls == 1
    assert telemetry["mode"] == "shadow"
    assert telemetry["executed"] == "rule"
    assert telemetry["shadow_agreement"] is True  # both rule & model chose fetch

    # 2. Canary tenant allowlist + deterministic hash stability + circuit breaker after consecutive failures
    canary = ShadowCanaryPolicyController(
        mode="canary",
        model_controller=model_ctrl,
        canary_ratio=0.0,
        canary_tenants=("vip-tenant",),
    )
    c_dec, c_tel = canary.decide_with_context(feat, request_id="req-1", tenant_id="vip-tenant")
    assert c_dec.source == "model"
    assert c_dec.action == "fetch"
    assert c_tel["canary_selected"] is True

    other_dec, other_tel = canary.decide_with_context(feat, request_id="req-1", tenant_id="free-tenant")
    assert other_dec.source == "rule"
    assert other_tel["canary_selected"] is False

    # Circuit breaker opens on consecutive timeouts / failures and stops calling failing endpoint
    fail_calls = 0

    def _failing_transport(*args: Any) -> dict[str, Any]:
        nonlocal fail_calls
        fail_calls += 1
        raise TimeoutError("vllm timeout")

    failing_model = ModelPolicyController(
        model_id="policy-bad",
        base_url="http://127.0.0.1:8001/v1",
        transport=_failing_transport,
    )
    breaker_ctrl = ShadowCanaryPolicyController(
        mode="canary",
        model_controller=failing_model,
        canary_ratio=1.0,
        consecutive_failure_limit=2,
    )
    d1, _ = breaker_ctrl.decide_with_context(feat, request_id="a")
    d2, t2 = breaker_ctrl.decide_with_context(feat, request_id="b")
    assert d1.source == "rule_fallback"
    assert d2.source == "rule_fallback"
    assert t2["circuit_open"] is True
    assert fail_calls == 2

    # Subsequent calls short-circuit without touching _failing_transport
    d3, t3 = breaker_ctrl.decide_with_context(feat, request_id="c")
    assert d3.source == "rule"
    assert t3["circuit_open"] is True
    assert fail_calls == 2

    # Config factory builds ShadowCanaryPolicyController for shadow/canary and RuleBased for off
    shadow_cfg = Configuration(
        policy_model_mode="shadow",
        policy_model_id="policy-v3",
        policy_model_base_url="http://127.0.0.1:8001/v1",
    )
    assert isinstance(build_policy_controller(shadow_cfg), ShadowCanaryPolicyController)


def test_skill_pack_maturity_gate_anti_overfitting_and_distillation(tmp_path: Path) -> None:
    packs_dir = tmp_path / "packs"
    formats_dir = tmp_path / "formats"
    formats_dir.mkdir(parents=True)

    def _write_pack(pid: str, maturity: str, status: str = "active") -> None:
        pdir = packs_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        is_neutral = pid == "_neutral"
        extra_eval = "\neval:\n  verdict: pass" if maturity == MATURITY_PROMOTED else ""
        signals_block = "match_signals: []\nverifiers: []" if is_neutral else "match_signals:\n  - bench\nverifiers:\n  - numeric-audit"
        (pdir / "SKILL.md").write_text(
            f"---\npack_id: {pid}\nschema_version: 1\nskill_kind: domain\nversion: 1\n"
            f"title: {pid}\ntopic_domain: general\napplicable_when: Benchmark evaluation scenarios.\n"
            f"status: {status}\nmaturity: {maturity}\nmin_match_score: 0.3\n"
            f"stage_scope:\n  - planning\n{signals_block}{extra_eval}\n---\n\n"
            "## planning\n\n优先采用一手数据。\n",
            encoding="utf-8",
        )

    _write_pack("_neutral", MATURITY_PROMOTED)
    _write_pack("draft-pack", MATURITY_DRAFT)
    _write_pack("verified-pack", MATURITY_VERIFIED)
    _write_pack("promoted-pack", MATURITY_PROMOTED)

    svc_verified = SkillPackService(
        Configuration(
            skill_pack_enabled=True,
            skill_pack_dir=str(packs_dir),
            skill_format_dir=str(formats_dir),
            skill_pack_maturity_gate="verified",
        )
    )
    loaded_verified = svc_verified._load_all()
    assert "_neutral" in loaded_verified
    assert "draft-pack" not in loaded_verified
    assert "verified-pack" in loaded_verified
    assert "promoted-pack" in loaded_verified

    svc_prod = SkillPackService(
        Configuration(
            skill_pack_enabled=True,
            skill_pack_dir=str(packs_dir),
            skill_format_dir=str(formats_dir),
            skill_pack_maturity_gate="production",
        )
    )
    loaded_prod = svc_prod._load_all()
    assert set(loaded_prod) == {"_neutral", "promoted-pack"}

    # Anti-overfitting & parent invariant validator
    parent = SkillPack(
        pack_id="parent-pack",
        schema_version=1,
        version=1,
        maturity=MATURITY_VERIFIED,
        status="active",
        title="Parent",
        topic_domain="finance",
        applicable_when="Baseline domain methodology",
        match_signals=["finance"],
        negative_signals=[],
        min_match_score=0.3,
        stage_scope=["planning", "task_execution"],
        quality_gates={"review_min_sources_per_task": 3},
        budget_knobs={},
        layout={},
        body_sections={"planning": "检查核心财务指标。"},
        verifiers=["numeric-audit"],
    )
    bad_overfit = SkillPack(
        pack_id="parent-pack-cand",
        schema_version=1,
        version=2,
        maturity=MATURITY_DRAFT,
        status="shadow",
        title="Cand",
        topic_domain="finance",
        applicable_when="Bad candidate referencing case_007",
        match_signals=["finance"],
        negative_signals=[],
        min_match_score=0.3,
        stage_scope=["planning"],
        quality_gates={"review_min_sources_per_task": 2},
        budget_knobs={},
        layout={},
        body_sections={"planning": "针对 case_007 做特化。"},
        verifiers=[],
    )
    errs = validate_candidate_against_parent(
        bad_overfit,
        parent,
        forbidden_case_ids=["case_007"],
    )
    assert any("overfitting_forbidden_case_id:case_007" in e for e in errs)
    assert any("dropped required parent verifiers" in e for e in errs)
    assert any("loosened quality_gate" in e for e in errs)

    # compare_variants adversarial regression guardrail + distill_skills gate
    baseline_eval = {
        "variant": "base",
        "results": [
            {"id": "std_1", "topic": "T1", "metrics": {"hybrid_score": 0.70, "section_completeness": 0.8}},
            {"id": "adv_1", "topic": "T2", "metrics": {"hybrid_score": 0.85, "section_completeness": 0.9}},
        ],
    }
    good_cand_eval = {
        "variant": "cand",
        "results": [
            {"id": "std_1", "topic": "T1", "metrics": {"hybrid_score": 0.79, "section_completeness": 0.9}},
            {"id": "adv_1", "topic": "T2", "metrics": {"hybrid_score": 0.84, "section_completeness": 0.9}},
        ],
    }
    bad_adv_eval = {
        "variant": "cand_bad_adv",
        "results": [
            {"id": "std_1", "topic": "T1", "metrics": {"hybrid_score": 0.82, "section_completeness": 0.9}},
            {"id": "adv_1", "topic": "T2", "metrics": {"hybrid_score": 0.78, "section_completeness": 0.9}},
        ],
    }
    bad_cmp = build_comparison(baseline_eval, bad_adv_eval)
    assert "adversarial_regression_exceeded" in bad_cmp["regression_guardrails_tripped"]
    assert bad_cmp["verdict"] == "fail"

    failure_results = {
        "results": [
            {"id": f"case_{i}", "review_issues": [{"check": "stale_evidence"}]}
            for i in range(4)
        ]
    }
    out_dir = tmp_path / "distilled"
    promoted_report = run_distillation(
        results_payload=failure_results,
        parent_pack=parent,
        output_dir=out_dir / "pass",
        baseline_payload=baseline_eval,
        candidate_eval_payload=good_cand_eval,
        min_support_cases=3,
        min_hybrid_gain=0.03,
    )
    assert promoted_report["promotion_gate"]["approved"] is True
    assert promoted_report["promotion_gate"]["maturity"] == MATURITY_VERIFIED
    reloaded = load_pack(Path(promoted_report["skill_path"]))
    assert reloaded.pack_id == "parent-pack-evolved"
    assert reloaded.maturity == MATURITY_VERIFIED
    assert validate_candidate_against_parent(reloaded, parent) == []

    rejected_report = run_distillation(
        results_payload=failure_results,
        parent_pack=parent,
        output_dir=out_dir / "reject",
        baseline_payload=baseline_eval,
        candidate_eval_payload=bad_adv_eval,
        min_support_cases=3,
        min_hybrid_gain=0.03,
    )
    assert rejected_report["promotion_gate"]["approved"] is False
    assert rejected_report["promotion_gate"]["maturity"] == MATURITY_DRAFT
