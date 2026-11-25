import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from metrics import metrics_registry
from models import BudgetDenial, BudgetLease, DagAction, DagPatch, TodoItem
from services.budget import (
    RESOURCE_NAMES,
    BudgetGovernor,
    build_request_budget,
    build_request_budget_for_config,
    estimate_llm_cost_microusd,
    profile_for_depth,
)
from services.dag_scheduler import (
    ResearchTreeRePlanner,
    apply_dag_patch,
    partition_into_waves,
    score_wave1_branch,
    select_wave_one,
)
from state_manager import public_v4_projection


class FakeClock:
    def __init__(self):
        self.monotonic = 100.0
        self.utc = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def tick(self, seconds):
        self.monotonic += seconds
        self.utc += timedelta(seconds=seconds)


class BudgetRecoveryTests(unittest.TestCase):
    def test_recovers_dispatched_at_upper_bound_and_releases_known_unsent(self):
        clock = FakeClock()
        budget = build_request_budget("quick", now=clock.utc)
        scope = dict(tenant_id="tenant-a", request_id="req", monotonic_clock=lambda: clock.monotonic, utc_clock=lambda: clock.utc)
        original = BudgetGovernor(budget, **scope)
        sent = original.reserve("sent", {"search_calls": 2})
        original.reserve("unsent", {"search_calls": 3})
        self.assertTrue(original.dispatch(sent))
        ledger = original.export_ledger()
        clock.tick(10)
        restored = BudgetGovernor(budget, **scope)
        restored.restore_ledger(ledger)
        second_restart = BudgetGovernor(budget, **scope)
        second_restart.restore_ledger(restored.export_ledger())
        self.assertEqual(second_restart.snapshot().spent["search_calls"], 2)
        self.assertEqual(second_restart.snapshot().reserved["search_calls"], 0)
        self.assertEqual(second_restart.snapshot().limit.deadline_at, budget.deadline_at)
        self.assertFalse(second_restart.dispatch(sent))
        foreign = BudgetGovernor(budget, **{**scope, "tenant_id": "tenant-b"})
        with self.assertRaises(ValueError):
            foreign.restore_ledger(ledger)

    def test_projection_only_exposes_public_budget_and_revision(self):
        v4 = {
            "schema_version": 1,
            "budget": {"snapshot": {"spent": {"search_calls": 1}, "private": "secret"}, "ledger": {"owner": "secret"}},
            "replan": {"dag_revision": 2, "pruned_task_ids": [3], "decision_id": "wave1", "wave_1_committed": True},
            "pending_events": [{"private": "secret"}],
        }
        self.assertEqual(public_v4_projection(v4), {
            "schema_version": 1,
            "budget": {"spent": {"search_calls": 1}},
            "replan": {"dag_revision": 2, "pruned_task_ids": [3], "decision_id": "wave1"},
        })


class BudgetProfileAndProtocolTests(unittest.TestCase):
    def test_profiles_and_safe_disabled_configuration(self):
        expected = {
            "quick": (90, 20, 4, 8, 6, 1, 1, 12, 32000, 6000, 150000),
            "standard": (240, 45, 8, 22, 16, 3, 3, 28, 90000, 16000, 600000),
            "deep": (600, 90, 14, 48, 36, 6, 6, 56, 200000, 36000, 2000000),
        }
        for depth, values in expected.items():
            profile = profile_for_depth(depth)
            actual = (
                profile.duration_seconds,
                profile.finalization_reserve_ms / 1000,
                profile.limits["tasks"],
                profile.limits["search_calls"],
                profile.limits["web_fetches"],
                profile.limits["dynamic_fetches"],
                profile.limits["sandbox_calls"],
                profile.limits["llm_calls"],
                profile.limits["input_tokens"],
                profile.limits["output_tokens"],
                profile.limits["cost_microusd"],
            )
            self.assertEqual(actual, values)
            self.assertEqual(
                profile.finalization_reserve,
                {
                    "llm_calls": {"quick": 3, "standard": 5, "deep": 8}[depth],
                    "input_tokens": {"quick": 8000, "standard": 18000, "deep": 32000}[depth],
                    "output_tokens": {"quick": 2000, "standard": 4000, "deep": 8000}[depth],
                    "cost_microusd": {"quick": 40000, "standard": 150000, "deep": 400000}[depth],
                },
            )
            self.assertEqual(profile.wave_one_limit, {"quick": 1, "standard": 2, "deep": 3}[depth])
            self.assertEqual(profile.deepen_limit, {"quick": 0, "standard": 2, "deep": 4}[depth])

        config = Configuration.from_env(load_env_file=False)
        self.assertFalse(config.v4_research_budget_enabled)
        self.assertFalse(config.v4_replan_enabled)
        self.assertEqual(config.v4_default_research_depth, "standard")

    def test_tenant_limits_clamp_profiles_and_input_maps_are_copied(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        tenant_limits = {"tasks": 2, "llm_calls": 1, "cost_microusd": 100000}
        budget = build_request_budget("standard", now=now, tenant_limits=tenant_limits)
        tenant_limits["tasks"] = 99
        self.assertEqual(budget.limits["tasks"], 2)
        self.assertEqual(budget.limits["llm_calls"], 1)
        self.assertEqual(budget.finalization_reserve["llm_calls"], 1)
        exported = budget.to_dict()
        exported["limits"]["tasks"] = 88
        self.assertEqual(budget.limits["tasks"], 2)
        self.assertEqual(set(budget.limits), set(RESOURCE_NAMES))

    def test_config_parses_v4_overrides_and_rejects_unknown_depth(self):
        config = Configuration.from_env(
            overrides={
                "v4_research_budget_enabled": True,
                "v4_replan_enabled": True,
                "v4_default_research_depth": "deep",
                "v4_tenant_budget_limits": {"tenant-a": {"tasks": 2}},
            },
            load_env_file=False,
        )
        self.assertTrue(config.v4_research_budget_enabled)
        self.assertTrue(config.v4_replan_enabled)
        self.assertEqual(config.v4_default_research_depth, "deep")
        self.assertEqual(config.v4_tenant_budget_limits["tenant-a"]["tasks"], 2)
        budget = build_request_budget_for_config(
            config,
            "tenant-a",
            now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(budget.limits["tasks"], 2)
        self.assertEqual(budget.limits["search_calls"], 48)
        with self.assertRaises(ValueError):
            Configuration(v4_default_research_depth="max")
        with self.assertRaises(ValueError):
            Configuration(v4_research_budget_limits={"unknown": 1})

    def test_invalid_resource_names_negative_and_boolean_demand_are_rejected(self):
        governor = BudgetGovernor(build_request_budget("quick"))
        for demand in ({"alien": 1}, {"tasks": -1}, {"tasks": True}):
            with self.subTest(demand=demand), self.assertRaises(ValueError):
                governor.reserve("bad", demand)


class AtomicBudgetLeaseTests(unittest.TestCase):
    def setUp(self):
        metrics_registry.reset()
        self.clock = FakeClock()
        limits = {resource: 10 for resource in RESOURCE_NAMES}
        reserves = {resource: 0 for resource in RESOURCE_NAMES}
        reserves.update({"llm_calls": 2, "input_tokens": 2, "output_tokens": 2, "cost_microusd": 2})
        self.budget = build_request_budget(
            "quick",
            limits=limits,
            finalization_reserve=reserves,
            now=self.clock.utc,
            duration_seconds=100,
            finalization_reserve_ms=10,
        )
        self.governor = BudgetGovernor(
            self.budget,
            tenant_id="tenant-a",
            request_id="request-a",
            monotonic_clock=lambda: self.clock.monotonic,
            utc_clock=lambda: self.clock.utc,
        )

    def test_atomic_multiresource_denial_leaves_every_balance_unchanged(self):
        result = self.governor.reserve("first", {"search_calls": 11, "tasks": 1})
        self.assertIsInstance(result, BudgetDenial)
        self.assertEqual(result.reason, "limit_exceeded")
        snapshot = self.governor.snapshot()
        self.assertEqual(snapshot.reserved["tasks"], 0)
        self.assertEqual(snapshot.reserved["search_calls"], 0)

    def test_each_resource_enforces_zero_exact_and_over_limit_atomically(self):
        for resource in RESOURCE_NAMES:
            with self.subTest(resource=resource):
                clock = FakeClock()
                limits = {name: 3 for name in RESOURCE_NAMES}
                reserves = {name: 0 for name in RESOURCE_NAMES}
                budget = build_request_budget(
                    "quick",
                    now=clock.utc,
                    limits=limits,
                    finalization_reserve=reserves,
                )
                governor = BudgetGovernor(
                    budget,
                    monotonic_clock=lambda: clock.monotonic,
                    utc_clock=lambda: clock.utc,
                )
                zero = governor.reserve("zero", {resource: 0})
                exact = governor.reserve("exact", {resource: 3})
                self.assertIsInstance(zero, BudgetLease)
                self.assertIsInstance(exact, BudgetLease)
                overflow = governor.reserve("overflow", {resource: 1})
                self.assertEqual(overflow.reason, "limit_exceeded")

    def test_same_key_is_idempotent_and_changed_parameters_conflict(self):
        lease = self.governor.reserve("op-1", {"search_calls": 1})
        self.assertIsInstance(lease, BudgetLease)
        self.assertEqual(self.governor.reserve("op-1", {"search_calls": 1}), lease)
        conflict = self.governor.reserve("op-1", {"search_calls": 2})
        self.assertEqual(conflict.reason, "operation_conflict")
        self.assertTrue(self.governor.dispatch(lease))
        self.assertFalse(self.governor.dispatch(lease))

    def test_two_threads_cannot_reserve_the_last_unit_twice(self):
        gate = threading.Barrier(3)
        results = []

        def reserve(operation_id):
            gate.wait()
            results.append(self.governor.reserve(operation_id, {"search_calls": 10}))

        workers = [threading.Thread(target=reserve, args=(f"op-{index}",)) for index in range(2)]
        for worker in workers:
            worker.start()
        gate.wait()
        for worker in workers:
            worker.join()
        self.assertEqual(sum(isinstance(value, BudgetLease) for value in results), 1)
        self.assertEqual(sum(isinstance(value, BudgetDenial) for value in results), 1)

    def test_settle_is_idempotent_and_unknown_usage_charges_reserved_upper_bound(self):
        lease = self.governor.reserve("unknown", {"search_calls": 1, "input_tokens": 4})
        self.governor.dispatch(lease)
        settled = self.governor.settle(lease, usage=None, usage_source="unknown")
        repeated = self.governor.settle(lease, usage={"search_calls": 0}, usage_source="reported")
        self.assertEqual(settled.usage, {"search_calls": 1, "input_tokens": 4})
        self.assertEqual(repeated, settled)
        self.assertEqual(self.governor.snapshot().spent["input_tokens"], 4)

    def test_only_explicitly_undispatched_lease_can_be_released(self):
        lease = self.governor.reserve("release", {"web_fetches": 2})
        self.assertTrue(self.governor.release(lease))
        self.assertFalse(self.governor.dispatch(lease))
        self.assertEqual(self.governor.snapshot().remaining["web_fetches"], 10)

    def test_release_and_dispatch_race_has_only_one_winner(self):
        lease = self.governor.reserve("race", {"search_calls": 1})
        gate = threading.Barrier(3)
        outcomes = {}

        def release():
            gate.wait()
            outcomes["release"] = self.governor.release(lease)

        def dispatch():
            gate.wait()
            outcomes["dispatch"] = self.governor.dispatch(lease)

        workers = [threading.Thread(target=release), threading.Thread(target=dispatch)]
        for worker in workers:
            worker.start()
        gate.wait()
        for worker in workers:
            worker.join()
        self.assertNotEqual(outcomes["release"], outcomes["dispatch"])
        self.assertEqual(self.governor.external_attempt_count, int(outcomes["dispatch"]))

    def test_tenants_and_requests_have_isolated_ledger_keys(self):
        second = BudgetGovernor(
            self.budget,
            tenant_id="tenant-b",
            request_id="request-a",
            monotonic_clock=lambda: self.clock.monotonic,
            utc_clock=lambda: self.clock.utc,
        )
        first_lease = self.governor.reserve("same-op", {"tasks": 1})
        second_lease = second.reserve("same-op", {"tasks": 1})
        self.assertIsInstance(first_lease, BudgetLease)
        self.assertIsInstance(second_lease, BudgetLease)
        self.assertNotEqual(self.governor.tenant_id, second.tenant_id)


class StageReserveClockAndPressureTests(unittest.TestCase):
    def make_governor(self, *, limit=100, held=0, deadline_seconds=100):
        clock = FakeClock()
        limits = {resource: 1000 for resource in RESOURCE_NAMES}
        reserves = {resource: 0 for resource in RESOURCE_NAMES}
        limits["search_calls"] = limit
        reserves["search_calls"] = held
        budget = build_request_budget(
            "quick",
            limits=limits,
            finalization_reserve=reserves,
            now=clock.utc,
            duration_seconds=deadline_seconds,
            finalization_reserve_ms=10000,
        )
        governor = BudgetGovernor(
            budget,
            monotonic_clock=lambda: clock.monotonic,
            utc_clock=lambda: clock.utc,
        )
        return clock, governor

    def test_pressure_is_normal_at_79_percent_and_high_at_80_percent(self):
        _, governor = self.make_governor()
        first = governor.reserve("79", {"search_calls": 79})
        self.assertIsInstance(first, BudgetLease)
        self.assertEqual(governor.snapshot(minimum_demand={"search_calls": 1}).pressure, "normal")
        _, governor = self.make_governor()
        first = governor.reserve("80", {"search_calls": 80})
        self.assertIsInstance(first, BudgetLease)
        self.assertEqual(governor.snapshot(minimum_demand={"search_calls": 1}).pressure, "high")

    def test_research_cannot_consume_finalization_reserve_but_report_can(self):
        _, governor = self.make_governor(limit=10, held=4)
        research = governor.reserve("research", {"search_calls": 7}, work_class="required")
        self.assertEqual(research.reason, "finalization_reserved")
        report = governor.reserve("report", {"search_calls": 7}, work_class="required", stage="report")
        self.assertIsInstance(report, BudgetLease)

    def test_finalization_pressure_and_deadline_use_utc_and_monotonic_clocks(self):
        clock, governor = self.make_governor(deadline_seconds=20)
        clock.tick(10)
        self.assertEqual(governor.remaining_timeout(30, stage="research"), 0)
        clock.utc -= timedelta(seconds=5)
        self.assertEqual(governor.remaining_timeout(30, stage="research"), 0)
        clock.tick(11)
        self.assertEqual(governor.remaining_timeout(30, stage="research"), 0)
        result = governor.reserve("late", {"search_calls": 1})
        self.assertEqual(result.reason, "deadline_exceeded")

    def test_deadline_reconstruction_keeps_absolute_utc_expiry(self):
        clock, governor = self.make_governor(deadline_seconds=20)
        clock.tick(5)
        restored = BudgetGovernor(
            governor.budget,
            monotonic_clock=lambda: clock.monotonic,
            utc_clock=lambda: clock.utc,
        )
        self.assertAlmostEqual(restored.remaining_timeout(100, stage="report", work_class="required"), 15)


class PricingAndExternalAttemptTests(unittest.TestCase):
    def test_cost_estimate_rounds_up_and_uses_worst_input_rate(self):
        price = estimate_llm_cost_microusd(
            input_tokens=1,
            output_tokens=1,
            pricing={
                "prompt_per_1k_tokens": 0.001,
                "cached_prompt_per_1k_tokens": 0.0001,
                "cache_creation_per_1k_tokens": 0.002,
                "completion_per_1k_tokens": 0.003,
            },
        )
        self.assertEqual(price, 5)
        self.assertIsNone(estimate_llm_cost_microusd(input_tokens=1, output_tokens=1, pricing=None))

    def test_llm_attempt_with_unknown_price_is_denied_before_dispatch(self):
        budget = build_request_budget("quick")
        governor = BudgetGovernor(budget)
        denied = governor.reserve_llm_attempt(
            "provider-call",
            input_tokens=100,
            output_tokens=10,
            pricing=None,
        )
        self.assertEqual(denied.reason, "pricing_unavailable")
        self.assertEqual(governor.external_attempt_count, 0)

    def test_external_attempt_is_counted_once_after_dispatch_and_settle(self):
        budget = build_request_budget("quick")
        governor = BudgetGovernor(budget)
        lease = governor.reserve("search-1", {"search_calls": 1})
        self.assertTrue(governor.dispatch(lease))
        governor.settle(lease, usage={"search_calls": 1})
        governor.settle(lease, usage={"search_calls": 1})
        self.assertEqual(governor.external_attempt_count, 1)
        retry = governor.reserve("search-retry-2", {"search_calls": 1})
        self.assertTrue(governor.dispatch(retry))
        self.assertEqual(governor.external_attempt_count, 2)

    def test_dispatched_attempt_is_charged_even_when_usage_is_missing(self):
        governor = BudgetGovernor(build_request_budget("quick"))
        lease = governor.reserve("json-fallback:1", {"llm_calls": 1, "input_tokens": 40})
        self.assertTrue(governor.dispatch(lease))
        settlement = governor.settle(lease, {"llm_calls": 0})
        self.assertEqual(settlement.usage, {"llm_calls": 1, "input_tokens": 40})

    def test_estimated_usage_uses_bounds_and_billing_overrun_blocks_new_paid_work(self):
        governor = BudgetGovernor(build_request_budget("quick"))
        lease = governor.reserve("model:1", {"llm_calls": 1, "cost_microusd": 10})
        self.assertTrue(governor.dispatch(lease))
        self.assertEqual(governor.settle(lease, {"cost_microusd": 2}, usage_source="estimated").charged_cost_microusd, 10)
        lease = governor.reserve("model:2", {"llm_calls": 1, "cost_microusd": 10})
        self.assertTrue(governor.dispatch(lease))
        governor.settle(lease, {"cost_microusd": 12})
        self.assertEqual(governor.snapshot().overrun["cost_microusd"], 2)
        self.assertEqual(governor.reserve("model:3", {"llm_calls": 1, "cost_microusd": 10}).reason, "dispatch_unbounded")

    def test_lease_cannot_be_used_in_a_second_governor_with_identical_parameters(self):
        budget = build_request_budget("quick")
        first = BudgetGovernor(budget, tenant_id="first")
        second = BudgetGovernor(budget, tenant_id="second")
        lease = first.reserve("search:1", {"search_calls": 1})
        other_lease = second.reserve("search:1", {"search_calls": 1})
        self.assertFalse(second.dispatch(lease))
        self.assertTrue(second.dispatch(other_lease))


class PlannerWaveAndScoringTests(unittest.TestCase):
    def make_task(self, task_id, *, deps=(), coverage=(), status="pending", query=None):
        task = TodoItem(
            id=task_id,
            title=f"Task {task_id}",
            intent="Research scope",
            query=query or f"query {task_id}",
            status=status,
            depends_on=list(deps),
        )
        task.coverage_keys = list(coverage)
        return task

    def test_wave_one_respects_depth_cap_coverage_and_stable_tie_break(self):
        tasks = [
            self.make_task(4, coverage=["a"]),
            self.make_task(2, coverage=["b"]),
            self.make_task(1, coverage=["a", "b"]),
            self.make_task(3, deps=[1], coverage=["c"]),
        ]
        self.assertEqual([task.id for task in select_wave_one(tasks, "quick")], [1])
        self.assertEqual([task.id for task in select_wave_one(tasks, "standard")], [1, 2])
        self.assertEqual([task.id for task in select_wave_one(tasks, "deep")], [1, 2, 4])

    def test_wave_one_and_rewaving_cover_plain_two_and_three_wave_graphs(self):
        tasks = [self.make_task(1), self.make_task(2), self.make_task(3, deps=[1]), self.make_task(4, deps=[3])]
        self.assertEqual([[task.id for task in wave] for wave in partition_into_waves(tasks)], [[1, 2], [3], [4]])
        self.assertEqual([task.id for task in select_wave_one(tasks, "deep")], [1, 2])

    def test_ig_cd_deduplicates_facts_and_source_clusters(self):
        score = score_wave1_branch(
            task_id=1,
            required_coverage=["A", "B"],
            baseline_coverage=[],
            wave_coverage=["A"],
            baseline_facts=[{"entity": "x", "metric": "m", "value": 1, "source_id": "old"}],
            wave_facts=[
                {"entity": "x", "metric": "m", "value": 1, "source_id": "new-a"},
                {"entity": "y", "metric": "n", "value": 2, "source_id": "new-b"},
                {"entity": "y", "metric": "n", "value": 2, "source_id": "new-b"},
                {"entity": "z", "metric": "p", "value": 3, "source_id": "new-c"},
            ],
            wave_sources=[
                {"source_id": "s1", "source_cluster": "origin-1"},
                {"source_id": "s2", "source_cluster": "origin-2"},
                {"source_id": "mirror", "source_cluster": "origin-2"},
            ],
            baseline_sources=[{"source_id": "old-source", "source_cluster": "origin-1"}],
            comparable_pairs=[
                {"dimension": ["x", "m", "", "", "unit:"], "left_source": "old", "right_source": "new-a", "conflict": True},
                {"dimension": ["x", "m", "", "", "unit:"], "left_source": "old", "right_source": "new-a", "conflict": True},
            ],
        )
        self.assertEqual(score.ig_numerators, (1, 2, 1))
        self.assertEqual(score.ig_denominators, (2, 3, 2))
        self.assertAlmostEqual(score.information_gain, 0.55)
        self.assertEqual(score.comparable_dimensions, 1)
        self.assertEqual(score.conflicting_dimensions, 1)
        self.assertEqual(score.conflict_density, 1.0)

    def test_numeric_comparison_converts_units_and_respects_rounding_intervals(self):
        score = score_wave1_branch(
            task_id=1,
            required_coverage=[],
            baseline_coverage=[],
            wave_coverage=[],
            wave_facts=[
                {"entity": "x", "metric": "length", "scope": "total", "period": "2026", "unit": "km", "value": "1.0", "source_id": "a"},
                {"entity": "x", "metric": "length", "scope": "total", "period": "2026", "unit": "m", "value": "1000", "source_id": "b"},
            ],
            wave_sources=[{"source_id": "a"}, {"source_id": "b"}],
        )
        self.assertEqual(score.comparable_dimensions, 1)
        self.assertEqual(score.conflicting_dimensions, 0)

    def test_low_quality_or_untrusted_facts_do_not_contribute_to_scores(self):
        score = score_wave1_branch(
            task_id=1,
            required_coverage=["source dimension"],
            baseline_coverage=[],
            wave_coverage=["source dimension"],
            wave_facts=[
                {"entity": "x", "metric": "m", "value": 1, "source_id": "valid"},
                {"entity": "l0", "metric": "m", "value": 2, "source_id": "l0", "tier": "L0"},
                {"entity": "untrusted", "metric": "m", "value": 3, "source_id": "untrusted", "authorized": False},
                {"entity": "no-anchor", "metric": "m", "value": 4, "source_id": "no-anchor", "anchor_valid": False},
            ],
            wave_sources=[
                {"source_id": "valid"},
                {"source_id": "l0", "tier": "L0"},
                {"source_id": "untrusted", "authorized": False},
                {"source_id": "no-anchor", "anchor_valid": False},
            ],
        )
        self.assertEqual(score.effective_fact_count, 1)
        self.assertEqual(score.independent_source_clusters, 1)
        self.assertTrue(score.insufficient_samples)


class RePlannerAndStrictDagTests(unittest.TestCase):
    def task(self, task_id, *, deps=(), status="pending", coverage=(), query=None):
        task = TodoItem(
            id=task_id,
            title=f"Task {task_id}",
            intent="Intent",
            query=query or f"query {task_id}",
            status=status,
            depends_on=list(deps),
        )
        task.coverage_keys = list(coverage)
        return task

    def test_pivot_precedes_deepen_and_replanner_is_deterministic(self):
        tasks = [self.task(1, status="completed"), self.task(2, deps=[1])]
        score = score_wave1_branch(
            task_id=1,
            required_coverage=["gap"],
            baseline_coverage=[],
            wave_coverage=["gap"],
            baseline_facts=[{"entity": "x", "metric": "m", "value": 1, "source_id": "a"}],
            wave_facts=[{"entity": "x", "metric": "m", "value": 2, "source_id": "b"}],
            wave_sources=[{"source_id": "a"}, {"source_id": "b"}],
            comparable_pairs=[{"dimension": "x-m", "left_source": "a", "right_source": "b", "conflict": True}],
        )
        replanner = ResearchTreeRePlanner()
        first = replanner.plan(
            tasks,
            [score],
            depth="standard",
            next_task_id=3,
            available_task_slots=2,
            pivot_queries={2: "query 2 from independent source"},
            missing_coverage_by_branch={1: ["gap"]},
        )
        second = replanner.plan(
            tasks,
            [score],
            depth="standard",
            next_task_id=3,
            available_task_slots=2,
            pivot_queries={2: "query 2 from independent source"},
            missing_coverage_by_branch={1: ["gap"]},
        )
        self.assertEqual(first, second)
        self.assertEqual(first.patch.actions[0].action, "PIVOT")
        self.assertEqual(first.patch.decision_id, "wave1-r1")

    def test_deepen_caps_by_profile_and_preserves_monotonic_ids(self):
        tasks = [self.task(1, status="completed", coverage=["known"])]
        score = score_wave1_branch(
            task_id=1,
            required_coverage=["missing"],
            baseline_coverage=[],
            wave_coverage=["missing"],
            wave_facts=[{"entity": "x", "metric": f"m{i}", "value": i, "source_id": f"s{i}"} for i in range(1, 4)],
            wave_sources=[{"source_id": f"s{i}"} for i in range(1, 4)],
        )
        result = ResearchTreeRePlanner().plan(
            tasks,
            [score],
            depth="standard",
            next_task_id=8,
            available_task_slots=4,
            missing_coverage_by_branch={1: ["missing"]},
        )
        self.assertEqual(result.patch.actions[0].action, "DEEPEN")
        self.assertEqual(result.patch.actions[0].task_id, 8)
        self.assertEqual(result.patch.actions[0].parent_task_ids, [1])

    def test_no_patch_returns_finite_reason(self):
        result = ResearchTreeRePlanner().plan([], [], depth="quick", next_task_id=1, available_task_slots=4)
        self.assertIsNone(result.patch)
        self.assertEqual(result.reason_code, "no_eligible_signal")

    def test_strict_patch_rejects_cycles_duplicate_and_running_targets_transactionally(self):
        tasks = [self.task(1, status="completed"), self.task(2, deps=[1])]
        original = [(task.query, list(task.depends_on)) for task in tasks]
        invalid = DagPatch(
            base_revision=0,
            decision_id="wave1-r1",
            actions=[DagAction(action="DEEPEN", task_id=3, parent_task_ids=[2], reason_code="high_ig", after={"title": "Task 3", "intent": "Intent", "query": "query 3", "depends_on": [3]})],
        )
        result = apply_dag_patch(
            tasks,
            invalid,
            current_revision=0,
            next_task_id=3,
            available_task_slots=1,
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.reason_code, "invalid_patch")
        self.assertEqual([(task.query, task.depends_on) for task in tasks], original)

        running = DagPatch(
            base_revision=0,
            decision_id="wave1-r1",
            actions=[DagAction(action="PIVOT", task_id=2, parent_task_ids=[1], reason_code="conflict", after={"query": "new query"})],
        )
        result = apply_dag_patch(tasks, running, current_revision=0, dispatched_ids={2})
        self.assertFalse(result.applied)

    def test_valid_deepen_is_rewaved_and_stale_revision_is_rejected(self):
        tasks = [self.task(1, status="completed", coverage=["covered"]), self.task(2, deps=[1], coverage=["required"])]
        patch = DagPatch(
            base_revision=2,
            decision_id="wave1-r3",
            actions=[DagAction(action="DEEPEN", task_id=3, parent_task_ids=[1], reason_code="high_ig", after={"title": "New", "intent": "Intent", "query": "new query", "depends_on": [1], "coverage_keys": ["required"]})],
        )
        stale = apply_dag_patch(tasks, patch, current_revision=1, next_task_id=3)
        self.assertFalse(stale.applied)
        applied = apply_dag_patch(
            tasks,
            patch,
            current_revision=2,
            next_task_id=3,
            required_coverage=["required"],
            available_task_slots=2,
        )
        self.assertTrue(applied.applied)
        self.assertEqual(applied.revision, 3)
        self.assertEqual(applied.added_task_ids, [3])
        self.assertEqual([[task.id for task in wave] for wave in applied.waves], [[2, 3]])

    def test_prune_keeps_history_and_does_not_orphan_required_coverage(self):
        tasks = [self.task(1, status="completed", coverage=["required"]), self.task(2, deps=[1], coverage=["optional"])]
        patch = DagPatch(
            base_revision=0,
            decision_id="wave1-r1",
            actions=[DagAction(action="PRUNE", task_id=2, parent_task_ids=[1], reason_code="redundant")],
        )
        result = apply_dag_patch(tasks, patch, current_revision=0, required_coverage=["required"])
        self.assertTrue(result.applied)
        self.assertEqual(result.pruned_task_ids, [2])
        self.assertEqual([task.id for task in result.tasks], [1, 2])
        self.assertEqual(result.waves, [])


if __name__ == "__main__":
    unittest.main()


def test_report_optional_can_use_finalization_pool_with_guarded_priority():
    budget = build_request_budget("quick", limits={"llm_calls": 5}, finalization_reserve={"llm_calls": 3})
    governor = BudgetGovernor(budget, tenant_id="default", request_id="reserved-report")
    research = governor.reserve("research", {"llm_calls": 2}, "required")
    assert governor.dispatch(research)
    governor.settle(research, {"llm_calls": 2})
    denied = governor.reserve("legacy", {"llm_calls": 1}, "optional", stage="report")
    assert denied.reason == "finalization_reserved"
    denied = governor.reserve(
        "outline", {"llm_calls": 1}, "optional", stage="report",
        allow_optional_finalization=True, keep_available={"llm_calls": 3},
    )
    assert isinstance(denied, BudgetDenial)
    critic = governor.reserve(
        "critic", {"llm_calls": 1}, "optional", stage="report", allow_optional_finalization=True,
    )
    assert isinstance(critic, BudgetLease)
    assert critic.work_class == "optional"


def test_unpublished_report_projection_filters_drafts_and_claims():
    v4 = {
        "schema_version": 1,
        "report": {
            "outline": {"outline_version": 1, "sections": [{"section_id": "sec-ab", "title": "A", "order": 1}]},
            "sections_by_id": {"sec-ab": {"status": "started", "draft": "secret", "claim_ids": ["C-1"]}},
            "operations_by_key": {"internal": {"lease": "secret"}},
        },
    }
    assert public_v4_projection(v4)["report"] == {
        "outline_version": 1,
        "sections": [{"section_id": "sec-ab", "title": "A", "order": 1, "status": "started"}],
        "critic_status": "skipped",
    }
