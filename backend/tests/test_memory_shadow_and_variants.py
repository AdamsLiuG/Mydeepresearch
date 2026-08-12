import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from agent import DeepResearchAgent
from config import Configuration
from evals.compare_variants import build_comparison, compare_cases, render_markdown
from evals.judges.heuristic import _memory_metrics
from evals.run_benchmark import load_variant
from metrics import RequestTrace


def _make_trace() -> RequestTrace:
    return RequestTrace(
        request_id="req-shadow",
        topic="shadow topic",
        search_api="duckduckgo",
        provider="openai",
        model="gpt-test",
        pricing_catalog=None,
    )


class MemoryShadowGateTests(unittest.TestCase):
    """The shadow gate must stay observable while changing nothing downstream."""

    def test_gate_passes_content_through_when_shadow_disabled(self):
        agent = SimpleNamespace(config=SimpleNamespace(memory_shadow_mode=False))
        trace = _make_trace()

        rendered = DeepResearchAgent._shadow_gate(
            agent,
            "memory block",
            stage="planning_note",
            observer=trace,
        )

        self.assertEqual(rendered, "memory block")
        self.assertEqual(trace.memory_shadow_suppressions, 0)

    def test_gate_withholds_content_but_records_suppression(self):
        agent = SimpleNamespace(config=SimpleNamespace(memory_shadow_mode=True))
        trace = _make_trace()

        rendered = DeepResearchAgent._shadow_gate(
            agent,
            "memory block",
            stage="planning_note",
            observer=trace,
        )

        self.assertEqual(rendered, "")
        self.assertEqual(trace.memory_shadow_suppressions, 1)
        self.assertEqual(trace.memory_shadow_suppressed_stages, {"planning_note": 1})

    def test_gate_ignores_empty_render(self):
        agent = SimpleNamespace(config=SimpleNamespace(memory_shadow_mode=True))
        trace = _make_trace()

        rendered = DeepResearchAgent._shadow_gate(
            agent,
            "",
            stage="reflection_strategy",
            observer=trace,
        )

        self.assertEqual(rendered, "")
        self.assertEqual(trace.memory_shadow_suppressions, 0)

    def test_snapshot_exposes_shadow_fields(self):
        trace = _make_trace()
        trace.memory_shadow_mode = True
        trace.record_memory_shadow_suppression("task_execution_note")

        snapshot = trace.snapshot()

        self.assertTrue(snapshot["memory_shadow_mode"])
        self.assertEqual(snapshot["memory_shadow_suppressions"], 1)
        self.assertEqual(
            snapshot["memory_shadow_suppressed_stages"],
            {"task_execution_note": 1},
        )

    def test_shadow_mode_defaults_to_disabled(self):
        config = Configuration.from_env(load_env_file=False)
        self.assertFalse(config.memory_shadow_mode)


class MemoryJudgeMetricTests(unittest.TestCase):
    def test_effective_injection_is_zero_under_shadow_mode(self):
        metrics = _memory_metrics(
            {
                "note_memory_queries": 2,
                "note_memory_hits": 3,
                "note_memory_prompt_injections": 1,
                "strategy_memory_queries": 1,
                "strategy_memory_hits": 2,
                "strategy_memory_prompt_injections": 1,
                "memory_shadow_mode": True,
                "memory_shadow_suppressions": 2,
                "note_memory_last_match_types": ["conclusion"],
                "strategy_memory_last_match_kinds": ["planning_pattern"],
            }
        )

        self.assertEqual(metrics["memory_query_count"], 3)
        self.assertEqual(metrics["memory_hit_count"], 5)
        self.assertEqual(metrics["memory_rendered_injection_count"], 2)
        self.assertEqual(metrics["memory_effective_injection_count"], 0)
        self.assertEqual(
            metrics["memory_match_kinds"],
            ["conclusion", "planning_pattern"],
        )

    def test_effective_injection_counted_when_not_shadowed(self):
        metrics = _memory_metrics(
            {
                "note_memory_prompt_injections": 1,
                "strategy_memory_prompt_injections": 2,
                "memory_shadow_mode": False,
            }
        )

        self.assertEqual(metrics["memory_effective_injection_count"], 3)

    def test_missing_and_malformed_trace_fields_are_tolerated(self):
        metrics = _memory_metrics({"note_memory_hits": "not-a-number"})

        self.assertEqual(metrics["memory_hit_count"], 0)
        self.assertFalse(metrics["memory_shadow_mode"])


def _payload(variant: str, cases: list[dict]) -> dict:
    return {"variant": variant, "results": cases}


def _case(case_id: str, *, titles: list[str], **metrics) -> dict:
    return {
        "id": case_id,
        "topic": f"topic-{case_id}",
        "todo_items": [{"title": title} for title in titles],
        "metrics": metrics,
    }


class CompareVariantsTests(unittest.TestCase):
    def test_only_shared_cases_are_compared(self):
        baseline = _payload("memory-off", [_case("a", titles=["t1"]), _case("b", titles=["t1"])])
        candidate = _payload("memory-both", [_case("a", titles=["t1"]), _case("c", titles=["t1"])])

        comparison = build_comparison(baseline, candidate)

        self.assertEqual(comparison["compared_cases"], 1)
        self.assertEqual(comparison["baseline_only_cases"], ["b"])
        self.assertEqual(comparison["candidate_only_cases"], ["c"])

    def test_plan_change_detected_via_task_title_jaccard(self):
        baseline = _payload("memory-off", [_case("a", titles=["alpha", "beta"])])
        candidate = _payload("memory-both", [_case("a", titles=["alpha", "gamma"])])

        cases = compare_cases(baseline, candidate)

        self.assertTrue(cases[0]["plan_changed"])
        self.assertAlmostEqual(cases[0]["task_title_jaccard"], 1 / 3, places=4)

    def test_identical_plans_score_full_jaccard(self):
        baseline = _payload("memory-off", [_case("a", titles=["alpha"])])
        candidate = _payload("memory-both", [_case("a", titles=["alpha"])])

        cases = compare_cases(baseline, candidate)

        self.assertFalse(cases[0]["plan_changed"])
        self.assertEqual(cases[0]["task_title_jaccard"], 1.0)

    def test_quality_gain_yields_pass_verdict(self):
        baseline = _payload(
            "memory-off",
            [_case("a", titles=["t"], reference_match_rate=0.5, grounded_bullet_ratio=0.5)],
        )
        candidate = _payload(
            "memory-both",
            [_case("a", titles=["t"], reference_match_rate=0.8, grounded_bullet_ratio=0.7)],
        )

        comparison = build_comparison(baseline, candidate)

        self.assertEqual(comparison["verdict"], "pass")
        self.assertAlmostEqual(
            comparison["aggregate_deltas"]["reference_match_rate"],
            0.3,
            places=4,
        )
        self.assertEqual(comparison["improved_cases"], 1)

    def test_quality_loss_yields_fail_verdict(self):
        baseline = _payload("memory-off", [_case("a", titles=["t"], reference_match_rate=0.9)])
        candidate = _payload("memory-both", [_case("a", titles=["t"], reference_match_rate=0.4)])

        comparison = build_comparison(baseline, candidate)

        self.assertEqual(comparison["verdict"], "fail")
        self.assertEqual(comparison["regressed_cases"], 1)

    def test_no_shared_cases_is_inconclusive(self):
        comparison = build_comparison(
            _payload("memory-off", [_case("a", titles=["t"])]),
            _payload("memory-both", [_case("b", titles=["t"])]),
        )

        self.assertEqual(comparison["verdict"], "inconclusive")

    def test_markdown_report_includes_labels_and_verdict(self):
        comparison = build_comparison(
            _payload("memory-off", [_case("a", titles=["t"], reference_match_rate=0.5)]),
            _payload("memory-both", [_case("a", titles=["t"], reference_match_rate=0.9)]),
        )

        rendered = render_markdown(comparison)

        self.assertIn("memory-off", rendered)
        self.assertIn("memory-both", rendered)
        self.assertIn("Verdict: **pass**", rendered)
        self.assertIn("reference_match_rate", rendered)


class VariantLoaderTests(unittest.TestCase):
    def test_shipped_variants_are_loadable(self):
        for name in ("memory-off", "note-only", "strategy-only", "memory-both", "memory-shadow"):
            variant = load_variant(name)
            self.assertEqual(variant["name"], name)
            self.assertIn("config_overrides", variant)

    def test_variant_overrides_are_valid_config_fields(self):
        valid_fields = set(Configuration.model_fields)
        for name in ("memory-off", "memory-both", "memory-shadow"):
            for field in load_variant(name)["config_overrides"]:
                self.assertIn(field, valid_fields, f"{name} references unknown field {field}")

    def test_baseline_variant_disables_all_memory(self):
        overrides = load_variant("memory-off")["config_overrides"]

        self.assertFalse(overrides["note_memory_enabled"])
        self.assertFalse(overrides["strategy_memory_enabled"])

    def test_shadow_variant_enables_memory_without_injection(self):
        overrides = load_variant("memory-shadow")["config_overrides"]

        self.assertTrue(overrides["note_memory_enabled"])
        self.assertTrue(overrides["strategy_memory_enabled"])
        self.assertTrue(overrides["memory_shadow_mode"])

    def test_unknown_variant_raises_system_exit(self):
        with self.assertRaises(SystemExit):
            load_variant("no-such-variant")

    def test_variant_missing_config_overrides_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            broken = Path(tmp_dir) / "broken.json"
            broken.write_text(json.dumps({"name": "broken"}), encoding="utf-8")

            from evals import run_benchmark

            original_dir = run_benchmark.VARIANT_DIR
            run_benchmark.VARIANT_DIR = Path(tmp_dir)
            try:
                with self.assertRaises(SystemExit):
                    run_benchmark.load_variant("broken")
            finally:
                run_benchmark.VARIANT_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
