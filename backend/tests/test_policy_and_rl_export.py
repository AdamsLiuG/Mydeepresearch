"""Tests for P2-WP5 PolicyController routing and RL SFT/DPO trajectory exporter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "backend" / "src"
EVALS_DIR = ROOT_DIR / "backend"
for path_entry in (str(SRC_DIR), str(EVALS_DIR)):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from config import Configuration  # noqa: E402
from evals.export_rl_trajectories import (  # noqa: E402
    compute_trajectory_reward,
    export_dpo_pairs,
    export_sft_records,
)
from evals.export_rl_trajectories import (
    main as exporter_main,
)
from metrics import RequestTrace  # noqa: E402
from services.policy_controller import (  # noqa: E402
    ModelPolicyController,
    PolicyStateFeatures,
    RuleBasedPolicyController,
    build_policy_controller,
)


def _sample_features(**overrides) -> PolicyStateFeatures:
    base = {
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
        "task_title": "Test Task",
        "task_intent": "Compare frameworks",
        "gap_signals": ("low_source_diversity",),
        "can_fetch": False,
    }
    base.update(overrides)
    return PolicyStateFeatures(**base)


class PolicyControllerTests(unittest.TestCase):
    def test_rule_based_controller_decisions(self) -> None:
        ctrl = RuleBasedPolicyController()

        # Max round budget stops.
        stop_budget = ctrl.decide(_sample_features(round_index=3, max_rounds=3))
        self.assertEqual(stop_budget.action, "stop")
        self.assertEqual(stop_budget.source, "rule")

        # Thin content triggers fetch when fetch budget remains.
        fetch_decision = ctrl.decide(
            _sample_features(gap_signals=("thin_content",), can_fetch=True, fetches_left=1)
        )
        self.assertEqual(fetch_decision.action, "fetch")

        # Missing evidence triggers search.
        search_decision = ctrl.decide(_sample_features(evidence_count=0, searches_left=1))
        self.assertEqual(search_decision.action, "search")

        # Sufficient evidence stops.
        stop_ok = ctrl.decide(_sample_features(has_gap_signals=False, evidence_count=3))
        self.assertEqual(stop_ok.action, "stop")

    def test_model_controller_parses_valid_json_and_falls_back_on_error(self) -> None:
        def _ok_transport(url, payload, headers, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"action": "search", "confidence": 0.92, "reason": "need authoritative benchmark", "query_override": "RAG benchmark 2025"}\n```'
                        }
                    }
                ]
            }

        model_ctrl = ModelPolicyController(
            model_id="qwen2.5-7b-policy",
            base_url="http://localhost:8001/v1",
            transport=_ok_transport,
        )
        ok_decision = model_ctrl.decide(_sample_features())
        self.assertEqual(ok_decision.action, "search")
        self.assertEqual(ok_decision.source, "model")
        self.assertAlmostEqual(ok_decision.confidence, 0.92)
        self.assertEqual(ok_decision.query_override, "RAG benchmark 2025")

        # Malformed output falls back cleanly.
        bad_ctrl = ModelPolicyController(
            model_id="qwen2.5-7b-policy",
            base_url="http://localhost:8001/v1",
            transport=lambda *_: {"choices": [{"message": {"content": "not json"}}]},
        )
        fb = bad_ctrl.decide(_sample_features())
        self.assertEqual(fb.source, "rule_fallback")
        self.assertEqual(fb.action, "search")

        # Transport exception falls back cleanly.
        timeout_ctrl = ModelPolicyController(
            model_id="qwen2.5-7b-policy",
            base_url="http://localhost:8001/v1",
            transport=lambda *_: (_ for _ in ()).throw(TimeoutError("timed out")),
        )
        fb_timeout = timeout_ctrl.decide(_sample_features(has_gap_signals=False))
        self.assertEqual(fb_timeout.source, "rule_fallback")
        self.assertEqual(fb_timeout.action, "stop")

    def test_build_policy_controller_respects_config(self) -> None:
        disabled = Configuration(policy_model_enabled=False, llm_api_key="k")
        self.assertIsInstance(build_policy_controller(disabled), RuleBasedPolicyController)

        enabled = Configuration(
            policy_model_enabled=True,
            policy_model_id="policy-7b",
            policy_model_base_url="http://127.0.0.1:8001/v1",
            llm_api_key="k",
        )
        self.assertIsInstance(build_policy_controller(enabled), ModelPolicyController)

    def test_request_trace_records_policy_decisions(self) -> None:
        trace = RequestTrace(
            request_id="req-policy-1",
            topic="Policy Trace",
            search_api="tavily",
            provider="openai",
            model="gpt-4o-mini",
            pricing_catalog=None,
        )
        trace.record_policy_decision({"task_id": 1, "action": "search", "source": "rule"})
        snap = trace.snapshot()
        self.assertEqual(len(snap["policy_decisions"]), 1)
        self.assertEqual(snap["policy_decisions"][0]["action"], "search")


class RlExporterTests(unittest.TestCase):
    def test_reward_and_sft_dpo_exports_and_cli(self) -> None:
        high_snap = {
            "request_id": "req-high",
            "topic": "Agent Memory Benchmark",
            "status": "success",
            "report_markdown": "# Agent Memory Benchmark\n\n" + ("Detailed grounded analysis. " * 12),
            "todo_items": [{"id": 1, "status": "completed"}],
            "request_metrics": {
                "request_id": "req-high",
                "topic": "Agent Memory Benchmark",
                "status": "success",
                "total_tasks": 2,
                "completed_tasks": 2,
                "total_tokens": 3000,
                "elapsed_ms": 12000.0,
                "degraded": False,
                "fallback_reasons": [],
                "review_summary": {"critical_issues": 0, "warning_issues": 0},
                "policy_decisions": [{"task_id": 1, "action": "stop", "source": "rule"}],
            },
        }
        low_snap = {
            "request_id": "req-low",
            "topic": "Agent Memory Benchmark",
            "status": "failed",
            "report_markdown": "",
            "todo_items": [{"id": 1, "status": "pending"}],
            "request_metrics": {
                "request_id": "req-low",
                "topic": "Agent Memory Benchmark",
                "status": "failed",
                "total_tasks": 2,
                "completed_tasks": 0,
                "total_tokens": 25000,
                "elapsed_ms": 150000.0,
                "degraded": True,
                "fallback_reasons": ["provider_timeout"],
                "review_summary": {"critical_issues": 2, "warning_issues": 1},
            },
        }

        high_reward = compute_trajectory_reward(high_snap)
        low_reward = compute_trajectory_reward(low_snap)
        self.assertGreater(high_reward["reward"], 0.7)
        self.assertLess(low_reward["reward"], 0.2)

        sft = export_sft_records([high_snap, low_snap], min_reward=0.5)
        self.assertEqual(len(sft), 1)
        self.assertEqual(sft[0]["metadata"]["request_id"], "req-high")
        self.assertEqual(sft[0]["messages"][0]["role"], "system")

        dpo = export_dpo_pairs([high_snap, low_snap], min_reward_margin=0.15)
        self.assertEqual(len(dpo), 1)
        self.assertEqual(dpo[0]["chosen_request_id"], "req-high")
        self.assertEqual(dpo[0]["rejected_request_id"], "req-low")
        self.assertGreaterEqual(dpo[0]["reward_margin"], 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "in"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir()
            (in_dir / "req-high.json").write_text(json.dumps(high_snap), encoding="utf-8")
            (in_dir / "req-low.json").write_text(json.dumps(low_snap), encoding="utf-8")

            rc = exporter_main(
                [
                    "--input-dir",
                    str(in_dir),
                    "--output-dir",
                    str(out_dir),
                    "--min-reward",
                    "0.5",
                    "--min-margin",
                    "0.15",
                ]
            )
            self.assertEqual(rc, 0)
            sft_lines = (out_dir / "sft_trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
            dpo_lines = (out_dir / "dpo_pairs.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(sft_lines), 1)
            self.assertEqual(len(dpo_lines), 1)
