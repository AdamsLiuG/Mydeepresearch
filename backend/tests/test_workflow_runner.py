from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import Configuration  # noqa: E402
from main import create_app  # noqa: E402
from models import SummaryState, TodoItem  # noqa: E402
from services.event_bus import (  # noqa: E402
    InMemoryStreamEventBus,
    RedisStreamEventBus,
)
from services.reporter import build_report_workbench_payload  # noqa: E402
from services.request_state import (  # noqa: E402
    RequestStateStore,
    SnapshotConflictError,
)
from services.storage_backends import PostgresRequestStateStore  # noqa: E402
from services.tenant import TenantRateLimiter  # noqa: E402
from services.workflow_runner import (  # noqa: E402
    LocalDurableWorkflowRunner,
    build_workflow_runner,
)
from worker import WORKFLOW_STAGES, DeepResearchWorkflowSpec  # noqa: E402


def test_inmemory_event_bus_tail_since_wakeup_and_terminal() -> None:
    bus = InMemoryStreamEventBus(max_events_per_request=100)
    id1 = bus.publish("t1", "req-tail", {"type": "status", "message": "boot"})
    assert id1 == "1"

    collected: list[dict[str, Any]] = []

    def _consumer() -> None:
        for env in bus.tail_since("t1", "req-tail", last_event_id=id1, block_ms=1000):
            collected.append({"id": env.event_id, **env.payload})

    thread = threading.Thread(target=_consumer, daemon=True)
    thread.start()
    time.sleep(0.05)
    bus.publish("t1", "req-tail", {"type": "task_status", "task_id": 1, "status": "completed"})
    bus.publish("t1", "req-tail", {"type": "done"})
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert [c["type"] for c in collected] == ["task_status", "done"]


def test_redis_event_bus_tail_since_and_terminal_ttl() -> None:
    class _FakeRedisStream:
        def __init__(self) -> None:
            self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
            self.ttls: dict[str, int] = {}
            self.seq = 0

        def xadd(self, key: str, fields: dict[str, str], **kwargs: Any) -> str:
            del kwargs
            self.seq += 1
            eid = f"{self.seq}-0"
            self.streams.setdefault(key, []).append((eid, fields))
            return eid

        def expire(self, key: str, seconds: int) -> None:
            self.ttls[key] = seconds

        def xrange(self, key: str, min: str = "-", max: str = "+") -> list[tuple[str, dict[str, str]]]:  # noqa: A002
            del max
            entries = self.streams.get(key, [])
            if min.startswith("("):
                after = min[1:]
                idx = next((i for i, (eid, _) in enumerate(entries) if eid == after), -1)
                return entries[idx + 1 :] if idx >= 0 else entries
            return entries

        def xread(self, streams: dict[str, str], block: int = 5000) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
            del block
            out = []
            for key, cursor in streams.items():
                entries = self.xrange(key, min=f"({cursor}") if cursor != "0-0" else self.streams.get(key, [])
                if entries:
                    out.append((key, entries))
            return out

    fake = _FakeRedisStream()
    bus = RedisStreamEventBus(client=fake, stream_ttl_seconds=3600)
    eid1 = bus.publish("tenant-x", "req-r1", {"type": "status", "message": "init"})
    fake.xadd(bus._stream_key("tenant-x", "req-r1"), {"payload": json.dumps({"type": "done"})})
    fake.expire(bus._stream_key("tenant-x", "req-r1"), 3600)

    tail = list(bus.tail_since("tenant-x", "req-r1", last_event_id=eid1, block_ms=100))
    assert len(tail) == 1
    assert tail[0].payload["type"] == "done"
    assert fake.ttls[bus._stream_key("tenant-x", "req-r1")] == 3600


def test_snapshot_cas_conflict_on_file_and_mock_postgres(tmp_path: Path) -> None:
    store = RequestStateStore(str(tmp_path / "states"))
    saved1 = store.save(
        "req-cas",
        {"topic": "CAS", "phase": "planning", "runner_backend": "local_durable", "workflow_run_id": "run-1"},
        expected_version=0,
    )
    assert saved1["state_version"] == 1
    assert saved1["runner_backend"] == "local_durable"

    saved2 = store.save("req-cas", {"topic": "CAS", "phase": "execution"}, expected_version=1)
    assert saved2["state_version"] == 2

    with pytest.raises(SnapshotConflictError):
        store.save("req-cas", {"topic": "CAS", "phase": "stale"}, expected_version=1)

    # Mock psycopg connection for PostgresRequestStateStore CAS check
    pg_rows: dict[str, dict[str, Any]] = {}

    class _FakeCursor:
        def __init__(self) -> None:
            self.rowcount = 1
            self._one = None

        def __enter__(self) -> "_FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
            if sql.startswith("INSERT") and "ON CONFLICT (request_id) DO NOTHING" in sql:
                req_id = params[0]
                if req_id in pg_rows:
                    self.rowcount = 0
                else:
                    pg_rows[req_id] = {"state_version": params[5], "payload": params[9]}
                    self.rowcount = 1
            elif sql.startswith("UPDATE deepresearch_request_states"):
                req_id, exp_ver = params[-2], params[-1]
                existing = pg_rows.get(req_id)
                if not existing or existing["state_version"] != exp_ver:
                    self.rowcount = 0
                else:
                    existing["state_version"] = params[4]
                    existing["payload"] = params[8]
                    self.rowcount = 1

        def fetchone(self) -> Any:
            return self._one

    class _FakeConn:
        def __enter__(self) -> "_FakeConn":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def cursor(self) -> _FakeCursor:
            return _FakeCursor()

        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            pass

    pg = PostgresRequestStateStore(dsn="postgresql://fake", connection_factory=_FakeConn)
    p1 = pg.save("req-pg", {"topic": "PG CAS"}, expected_version=0)
    assert p1["state_version"] == 1
    p2 = pg.save("req-pg", {"topic": "PG CAS", "phase": "done"}, expected_version=1)
    assert p2["state_version"] == 2
    with pytest.raises(SnapshotConflictError):
        pg.save("req-pg", {"topic": "stale"}, expected_version=1)


def test_tenant_rate_limiter_redis_zset_and_fallback() -> None:
    class _FakeRedisRL:
        def __init__(self) -> None:
            self.fail = False
            self.rpm: dict[str, set[str]] = {}
            self.concurrent: dict[str, set[str]] = {}
            self.atomic_calls = 0

        def eval(self, script, num_keys, rpm_key, conc_key, rpm, concurrency, lease, ttl):
            self.atomic_calls += 1
            if self.fail:
                raise ConnectionError("redis down")
            assert num_keys == 2 and ttl > 0
            assert "redis.call('TIME')" in script
            active = self.concurrent.setdefault(conc_key, set())
            requests = self.rpm.setdefault(rpm_key, set())
            if concurrency and len(active) >= concurrency:
                return [0, 5]
            if rpm and len(requests) >= rpm:
                return [2, 60]
            if concurrency:
                active.add(lease)
            if rpm:
                requests.add(lease)
            return [1, 0]

        def zrem(self, key, lease):
            self.concurrent.setdefault(key, set()).discard(lease)

    redis = _FakeRedisRL()
    limiter = TenantRateLimiter(rpm_limit=2, max_concurrent=5, redis_client=redis)
    replica = TenantRateLimiter(rpm_limit=2, max_concurrent=5, redis_client=redis)
    ok1, _, _ = limiter.try_acquire("t1")
    limiter.release("t1")
    ok2, _, _ = limiter.try_acquire("t1")
    limiter.release("t1")
    ok3, retry_after, reason = replica.try_acquire("t1")
    assert ok1 is True and ok2 is True
    assert ok3 is False and reason == "rate_limit_rpm" and retry_after >= 1
    assert redis.atomic_calls == 3

    limiter = TenantRateLimiter(redis_client=redis)
    assert limiter.try_acquire("t2", max_concurrent=2, lease_id="distributed")[0]
    redis.fail = True
    assert limiter.try_acquire("t2", max_concurrent=2, lease_id="local")[0]
    assert limiter.try_acquire("t2", max_concurrent=2, lease_id="blocked")[2] == "max_concurrent_requests"

    redis.fail = False
    limiter.release("t2", lease_id="local")
    assert replica.try_acquire("t2", max_concurrent=1, lease_id="other-replica")[2] == "max_concurrent_requests"
    limiter.release("t2", lease_id="distributed")
    assert replica.try_acquire("t2", max_concurrent=1, lease_id="other-replica")[0]


def test_async_research_endpoints_follow_stream_and_signal(tmp_path: Path) -> None:
    class _DurableStubAgent:
        def __init__(self, config: Configuration, request_id: str | None = None, tenant_id: str | None = None) -> None:
            self.config = config
            self.request_id = request_id or "req-async"
            self.tenant_id = tenant_id or "default"

        def run_stream(self, topic: str, *, cancel_event: threading.Event | None = None):
            yield {"type": "status", "message": f"running {topic}"}
            time.sleep(0.05)
            if cancel_event is not None and cancel_event.is_set():
                yield {"type": "cancelled", "request_id": self.request_id}
                return
            yield {"type": "final_report", "report": f"# {topic}\n\n完成"}
            yield {"type": "done"}

    cfg = Configuration(
        workflow_runner="local_durable",
        request_state_dir=str(tmp_path / "states"),
        notes_workspace=str(tmp_path / "notes"),
        sse_heartbeat_interval_seconds=0.2,
    )
    app = create_app(base_config=cfg, agent_factory=_DurableStubAgent)
    assert isinstance(app.state.workflow_runner, LocalDurableWorkflowRunner)

    with TestClient(app) as client:
        submit_resp = client.post(
            "/research/async",
            json={"topic": "Durable Research"},
            headers={"X-Request-ID": "req-dur-1"},
        )
        assert submit_resp.status_code == 202
        submit_data = submit_resp.json()
        assert submit_data["request_id"] == "req-dur-1"
        assert submit_data["runner_backend"] == "local_durable"
        assert submit_data["events_url"] == "/requests/req-dur-1/events?follow=1"

        events_lines: list[str] = []
        with client.stream(
            "GET",
            "/requests/req-dur-1/events?follow=1",
            headers={"Accept": "text/event-stream"},
        ) as stream_resp:
            assert stream_resp.status_code == 200
            for line in stream_resp.iter_lines():
                if line:
                    events_lines.append(line)

        joined = "\n".join(events_lines)
        assert "Durable Research" in joined
        assert '"type": "done"' in joined

        # Verify cancel signal endpoint works
        submit2 = client.post(
            "/research/async",
            json={"topic": "Cancellable Job"},
            headers={"X-Request-ID": "req-dur-cancel"},
        )
        assert submit2.status_code == 202
        sig_resp = client.post(
            "/requests/req-dur-cancel/signals/cancel",
            json={"payload": {"reason": "user_stop"}},
        )
        assert sig_resp.status_code == 200
        assert sig_resp.json()["accepted"] is True
        assert sig_resp.json()["signal_name"] == "cancel"

    # Factory default builds inline runner cleanly
    inline_runner = build_workflow_runner(
        Configuration(workflow_runner="inline"),
        agent_factory=_DurableStubAgent,
        event_bus=InMemoryStreamEventBus(),
    )
    assert inline_runner.backend_name == "inline"


def test_build_report_workbench_payload_and_worker_imports() -> None:
    state = SummaryState(
        research_topic="Workbench Test",
        todo_items=[
            TodoItem(
                id=1,
                title="指标检索",
                intent="检索财报与图表",
                query="q1",
                status="completed",
                evidence_items=[
                    {
                        "source_id": "T1-S1",
                        "title": "官方年报",
                        "url": "https://official.example.com",
                        "domain": "official.example.com",
                        "snippet": "2025年营收为100亿元。",
                        "quality_label": "high",
                        "source_type": "official_documentation",
                    },
                    {
                        "source_id": "T1-C1",
                        "title": "沙盒计算产物",
                        "url": "",
                        "domain": "sandbox",
                        "snippet": "CAGR = 22.5%",
                        "computation_artifact": {"code": "print(22.5)"},
                    },
                ],
            )
        ],
        evidence_conflicts=[
            {
                "conflict_id": "CONF-WB1",
                "entity": "vendor",
                "metric": "revenue",
                "period": "2025",
                "sides": [{"source_id": "T1-S1"}, {"source_id": "T2-S1"}],
                "adjudication_status": "resolved_by_authority",
            }
        ],
        claim_verification_summary={"total_atomic_claims": 2, "entailed_count": 2, "atomic_faithfulness_rate": 1.0},
        skill_pack_metadata={"pack_id": "tech-research", "version": 2, "maturity": "verified"},
    )
    report_md = "## 核心结论\n年报显示营收 100 亿元 [T1-S1]，经计算复合增速达 22.5% [T1-C1]。"
    wb = build_report_workbench_payload(
        state,
        report_md,
        config=Configuration(nli_verifier_enabled=True, active_skill_verifiers=["numeric-audit"]),
    )
    assert set(wb["citation_index"]) == {"T1-S1", "T1-C1"}
    assert wb["citation_index"]["T1-C1"]["modality"] == "computed"
    assert len(wb["paragraph_bindings"]) == 1
    assert wb["paragraph_bindings"][0]["citation_ids"] == ["T1-S1", "T1-C1"]
    assert wb["paragraph_bindings"][0]["conflict_ids"] == ["CONF-WB1"]
    assert wb["skill_governance"]["pack_id"] == "tech-research"
    assert wb["skill_governance"]["active_verifiers"] == ["numeric-audit"]

    # Worker package lazy imports & docker-compose profiles parse cleanly
    spec = DeepResearchWorkflowSpec(topic="Test", request_id="r1")
    assert spec.request_id == "r1"
    assert WORKFLOW_STAGES == ("planning", "execution", "reflection", "review", "reporting")

    compose_data = yaml.safe_load((REPO_ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8"))
    assert "temporal-worker" in compose_data["services"]
    assert "arq-worker" in compose_data["services"]
    assert compose_data["services"]["temporal-worker"]["profiles"] == ["temporal"]
    assert compose_data["services"]["arq-worker"]["profiles"] == ["arq"]


def test_local_submit_reuses_run_and_rejects_late_signals(tmp_path: Path) -> None:
    running = threading.Event()
    finish = threading.Event()
    bus = InMemoryStreamEventBus()
    releases: list[str | None] = []

    class Agent:
        def __init__(self, config: Configuration) -> None:
            pass

        def run_stream(self, topic: str):
            running.set()
            assert finish.wait(2)
            yield {"type": "done"}

    runner = LocalDurableWorkflowRunner(
        agent_factory=Agent, event_bus=bus,
        state_store=RequestStateStore(str(tmp_path)),
        on_release=lambda tenant, *, lease_id: releases.append(lease_id),
    )
    args = {"request_id": "duplicate", "tenant_id": "t1", "config": Configuration()}
    original = runner.submit("test", **args, lease_id="lease1")
    assert running.wait(2)
    duplicate = runner.submit("test", **args, lease_id="lease2")
    finish.set()
    assert list(bus.tail_since("t1", "duplicate", block_ms=1000))[-1].payload["type"] == "done"
    runner.shutdown()
    assert duplicate.workflow_run_id == original.workflow_run_id
    assert duplicate.metadata["duplicate"] is True
    assert releases == ["lease1"]
    assert runner.signal("duplicate", "cancel", tenant_id="t1")["accepted"] is False
    assert runner.signal("duplicate", "cancel", tenant_id="t2")["accepted"] is False


def test_worker_reclaims_expired_lease_and_reuses_completed_tasks(tmp_path: Path) -> None:
    from services.workflow_runner import WorkflowCheckpoint, run_durable_attempt

    store = RequestStateStore(str(tmp_path))
    bus = InMemoryStreamEventBus()
    checkpoint = WorkflowCheckpoint(store, "tenant", "retry", "temporal", "run-1")
    checkpoint.reserve("topic")
    payload = store.load("retry", tenant_id="tenant")
    store.save(
        "retry",
        {**payload, "phase": "task_execution", "workflow_lease": {"owner": "crashed", "expires_at": 1},
         "todo_items": [{"id": 1, "status": "completed"}, {"id": 2, "status": "completed"},
                        {"id": 3, "status": "pending"}]},
        tenant_id="tenant", expected_version=payload["state_version"],
    )
    executed: list[int] = []

    class Agent:
        def run_stream_resume(self, request_id: str, *, cancel_event: threading.Event):
            snapshot = store.load(request_id, tenant_id="tenant")
            executed.extend(item["id"] for item in snapshot["todo_items"] if item["status"] != "completed")
            yield {"type": "final_report", "report": "# Done"}
            yield {"type": "done"}

    assert run_durable_attempt(checkpoint, lambda: Agent(), bus, threading.Event())["status"] == "completed"
    assert run_durable_attempt(checkpoint, lambda: Agent(), bus, threading.Event())["status"] == "completed"
    assert executed == [3]


def test_web_only_durable_reservation_keeps_frozen_policy_and_quick_deadline(tmp_path: Path) -> None:
    from config import Configuration
    from services.workflow_runner import WorkflowCheckpoint
    from state_manager import public_v4_projection

    store = RequestStateStore(str(tmp_path))
    checkpoint = WorkflowCheckpoint(store, "default", "web-only", "local", "web-run")
    config = Configuration(
        v4_web_extractor_enabled=True, v4_web_extractor_tenants=["default"],
        v4_default_research_depth="deep", v4_tenant_budget_limits={"default": {"web_fetches": 2}},
    )
    first, _ = checkpoint.reserve("example", config=config, plan_first=True)
    assert first["v4"]["web_downstream_only"]
    limit = first["v4"]["budget"]["limit"]
    assert limit["depth"] == "quick"
    assert limit["limits"]["web_fetches"] == 2
    assert public_v4_projection(first["v4"])["web_extraction"] == {
        "schema_version": 1, "tier_counts": {}, "degraded_counts": {},
    }
    second, _ = checkpoint.reserve("example", config=Configuration(), plan_first=True)
    assert second["v4"] == first["v4"]


def test_v4_durable_reservation_freezes_budget_and_options(tmp_path: Path) -> None:
    from services.workflow_runner import WorkflowCheckpoint, run_durable_attempt

    checkpoint = WorkflowCheckpoint(RequestStateStore(str(tmp_path)), "tenant", "v4-plan", "arq", "run")
    config = Configuration(v4_research_budget_enabled=True, v4_default_research_depth="quick")
    initial, created = checkpoint.reserve("topic", config=config, plan_first=True, output_format="finance")
    assert created
    assert initial["v4"]["schema_version"] == 1
    assert initial["v4"]["budget"]["limit"]["depth"] == "quick"
    assert initial["workflow_request_options"]["default_output_format"] == "finance"
    repeated, created = checkpoint.reserve("topic", config=config, plan_first=True, output_format="finance")
    assert not created
    assert repeated["v4"]["budget"]["limit"] == initial["v4"]["budget"]["limit"]
    with pytest.raises(SnapshotConflictError):
        checkpoint.reserve("topic", config=config, plan_first=True, output_format="general")
    with pytest.raises(SnapshotConflictError):
        checkpoint.reserve("topic", config=config, plan_first=True, output_format="finance", research_depth="deep")

    class Agent:
        def plan_only(self, topic: str) -> dict[str, Any]:
            assert topic == "topic"
            return {"todo_items": [{"id": 1}]}

    def agent_factory(*, workflow_request_options: dict[str, Any]) -> Agent:
        assert workflow_request_options == initial["workflow_request_options"]
        return Agent()

    bus = InMemoryStreamEventBus()
    assert run_durable_attempt(checkpoint, agent_factory, bus, threading.Event())["status"] == "awaiting_approval"
    plan_event = bus.replay_since("tenant", "v4-plan")[-1].payload
    assert checkpoint.signal("approve_plan", {"revision": plan_event["revision"]})["accepted"]
    assert checkpoint.load()["v4"]["replan"]["plan_locked"] is True


def test_v4_durable_checkpoint_preserves_ack_and_rejects_stale_graph_commit(tmp_path: Path) -> None:
    from services.workflow_runner import WorkflowCheckpoint, WorkflowLeaseLost

    store = RequestStateStore(str(tmp_path))
    checkpoint = WorkflowCheckpoint(store, "tenant", "v4-cas", "arq", "run")
    config = Configuration(v4_research_budget_enabled=True)
    checkpoint.reserve("topic", config=config)
    assert checkpoint.claim()[1]
    budget = checkpoint.load()["v4"]["budget"]
    checkpoint.persist({"v4": {"report": {"report_version": 2}}})
    saved = checkpoint.persist(
        {"v4": {"replan": {"dag_revision": 1, "wave_1_committed": True}}},
        expected_dag_revision=0,
        expected_wave_1_committed=False,
        pending_events=({"type": "research_replanned", "event_key": "wave1-r1", "decision_id": "wave1-r1"},),
    )
    assert saved["v4"]["pending_events"][0]["state_version"] == saved["state_version"]
    checkpoint.acknowledge_event("wave1-r1")
    latest = checkpoint.persist({"v4": saved["v4"], "status": "in_progress"})
    assert latest["v4"].get("pending_events", []) == []
    with pytest.raises(SnapshotConflictError):
        checkpoint.persist(
            {"v4": {"replan": {"dag_revision": 2, "wave_1_committed": False}}},
            expected_dag_revision=0,
            expected_wave_1_committed=False,
        )
    latest = checkpoint.persist_budget({**budget, "ledger": {"revision": 1}})
    assert latest["v4"]["budget"]["ledger"] == {"revision": 1}
    assert latest["v4"]["replan"]["dag_revision"] == 1
    assert latest["v4"]["report"] == {"report_version": 2}
    with pytest.raises(SnapshotConflictError):
        checkpoint.persist_budget({**budget, "limit": {**budget["limit"], "deadline_at": "2200-01-01T00:00:00Z"}})
    checkpoint.release()
    with pytest.raises(WorkflowLeaseLost):
        checkpoint.persist_budget(budget)


def test_v4_durable_runner_replays_pending_events_before_dispatch(tmp_path: Path) -> None:
    from services.workflow_runner import WorkflowCheckpoint, run_durable_attempt

    store = RequestStateStore(str(tmp_path))
    bus = InMemoryStreamEventBus()
    original = WorkflowCheckpoint(store, "tenant", "v4-replay", "arq", "run")
    original.reserve("topic", config=Configuration(v4_research_budget_enabled=True))
    assert original.claim()[1]
    committed = original.persist(
        {
            "phase": "task_execution",
            "v4": {"replan": {"dag_revision": 1, "wave_1_committed": True, "decision_id": "wave1-r1"}},
        },
        expected_dag_revision=0,
        expected_wave_1_committed=False,
        pending_events=({"type": "research_replanned", "event_key": "wave1-r1", "decision_id": "wave1-r1"},),
    )
    original.release()
    recovered = WorkflowCheckpoint(store, "tenant", "v4-replay", "arq", "run")

    class Agent:
        def run_stream_resume(self, request_id: str, *, cancel_event: threading.Event):
            published = bus.replay_since("tenant", request_id)
            assert [event.payload for event in published] == committed["v4"]["pending_events"]
            assert recovered.load()["v4"].get("pending_events", []) == []
            budget_event = {"type": "budget_updated", "event_key": "wave2-budget", "stage": "research"}
            saved = recovered.persist({}, pending_events=(budget_event,))
            yield budget_event
            published = bus.replay_since("tenant", request_id)
            assert [event.payload["type"] for event in published] == ["research_replanned", "budget_updated"]
            assert published[-1].payload["state_version"] == saved["state_version"]
            assert recovered.load()["v4"].get("pending_events", []) == []
            yield {"type": "todo_list", "tasks": [{"id": 2}]}
            yield {"type": "done"}

    assert run_durable_attempt(recovered, Agent, bus, threading.Event())["status"] == "completed"
    assert [event.payload["type"] for event in bus.replay_since("tenant", "v4-replay")] == [
        "research_replanned", "budget_updated", "todo_list", "done",
    ]


def test_v4_durable_runner_retries_saved_event_after_lost_ack(tmp_path: Path, monkeypatch) -> None:
    from services.workflow_runner import (
        WorkflowCheckpoint,
        WorkflowEventDeliveryError,
        run_durable_attempt,
    )

    store = RequestStateStore(str(tmp_path))
    bus = InMemoryStreamEventBus()
    checkpoint = WorkflowCheckpoint(store, "tenant", "v4-lost-ack", "arq", "run")
    checkpoint.reserve("topic", config=Configuration(v4_research_budget_enabled=True))
    assert checkpoint.claim()[1]
    committed = checkpoint.persist(
        {"phase": "task_execution"},
        pending_events=({"type": "budget_updated", "event_key": "budget-1", "stage": "research"},),
    )
    checkpoint.release()

    def lost_ack(event_key: str) -> None:
        raise SnapshotConflictError(event_key)

    def unexpected_agent() -> None:
        pytest.fail("Research resumed before the pending event was acknowledged")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "acknowledge_event", lost_ack)
        with pytest.raises(WorkflowEventDeliveryError):
            run_durable_attempt(checkpoint, unexpected_agent, bus, threading.Event())
    event = committed["v4"]["pending_events"][0]
    assert checkpoint.load()["v4"]["pending_events"] == [event]

    class Agent:
        def run_stream_resume(self, request_id: str):
            assert checkpoint.load()["v4"]["pending_events"] == []
            yield {"type": "done"}

    assert run_durable_attempt(checkpoint, Agent, bus, threading.Event())["status"] == "completed"
    events = bus.replay_since("tenant", "v4-lost-ack")
    assert [item.payload for item in events[:2]] == [event, event]
    assert events[0].event_id != events[1].event_id
    assert events[2].payload["type"] == "done"


def test_v4_durable_terminal_delivery_failure_preserves_completed_report(tmp_path: Path) -> None:
    from services.workflow_runner import (
        WorkflowCheckpoint,
        WorkflowEventDeliveryError,
        run_durable_attempt,
    )

    class FailFirstDoneBus(InMemoryStreamEventBus):
        def __init__(self) -> None:
            super().__init__()
            self.fail_done = True

        def publish(self, tenant_id: str, request_id: str, payload: dict[str, Any]) -> str:
            if payload.get("type") == "done" and self.fail_done:
                self.fail_done = False
                raise RuntimeError("Event bus interrupted")
            return super().publish(tenant_id, request_id, payload)

    store = RequestStateStore(str(tmp_path))
    bus = FailFirstDoneBus()
    checkpoint = WorkflowCheckpoint(store, "tenant", "v4-terminal-failure", "arq", "run")
    checkpoint.reserve("topic", config=Configuration(v4_research_budget_enabled=True))

    class Agent:
        def run_stream(self, topic: str):
            checkpoint.persist({"phase": "completed", "status": "success", "report_markdown": "# Complete"})
            yield {"type": "final_report", "report": "# Complete"}
            yield {"type": "done"}

    with pytest.raises(WorkflowEventDeliveryError):
        run_durable_attempt(checkpoint, Agent, bus, threading.Event())
    assert checkpoint.load()["status"] == "success"

    def unexpected_agent() -> None:
        pytest.fail("Terminal event recovery started research work")

    assert run_durable_attempt(checkpoint, unexpected_agent, bus, threading.Event())["status"] == "success"
    events = bus.replay_since("tenant", "v4-terminal-failure")
    assert [item.payload["type"] for item in events] == ["final_report", "done"]


def test_v4_local_duplicate_repairs_terminal_delivery(tmp_path: Path) -> None:
    from services.workflow_runner import WorkflowCheckpoint, _workflow_run_id

    store = RequestStateStore(str(tmp_path))
    bus = InMemoryStreamEventBus()
    run_id = _workflow_run_id("local_durable", "tenant", "v4-duplicate-terminal")
    checkpoint = WorkflowCheckpoint(store, "tenant", "v4-duplicate-terminal", "local_durable", run_id)
    config = Configuration(v4_research_budget_enabled=True)
    checkpoint.reserve("topic", config=config)
    assert checkpoint.claim()[1]
    checkpoint.persist({"phase": "completed", "status": "success", "report_markdown": "# Complete"})
    checkpoint.release()

    def unexpected_agent() -> None:
        pytest.fail("A duplicate terminal request started research work")

    runner = LocalDurableWorkflowRunner(agent_factory=unexpected_agent, event_bus=bus, state_store=store)
    handle = runner.submit("topic", request_id="v4-duplicate-terminal", tenant_id="tenant", config=config)
    runner.shutdown()
    assert handle.metadata["duplicate"] is True
    assert handle.status == "success"
    assert [item.payload["type"] for item in bus.replay_since("tenant", "v4-duplicate-terminal")] == [
        "final_report", "done",
    ]


def test_plan_checkpoint_waits_for_matching_revision(tmp_path: Path) -> None:
    from services.workflow_runner import WorkflowCheckpoint, run_durable_attempt

    store = RequestStateStore(str(tmp_path))
    bus = InMemoryStreamEventBus()
    checkpoint = WorkflowCheckpoint(store, "tenant", "plan", "temporal", "run-1")
    checkpoint.reserve("topic", plan_first=True)
    calls: list[str] = []

    class Agent:
        def plan_only(self, topic: str):
            calls.append("plan")
            return {"todo_items": [{"id": 1, "title": "Task"}]}

        def run_stream_resume(self, request_id: str, *, cancel_event: threading.Event):
            calls.append("execution")
            yield {"type": "done"}

    waiting = run_durable_attempt(checkpoint, Agent, bus, threading.Event())
    assert waiting["status"] == "awaiting_approval"
    event = bus.replay_since("tenant", "plan")[-1].payload
    assert checkpoint.signal("approve_plan", {"revision": event["revision"] - 1})["accepted"] is False
    assert checkpoint.signal("approve_plan", {"revision": event["revision"]})["accepted"] is True
    assert run_durable_attempt(checkpoint, Agent, bus, threading.Event())["status"] == "completed"
    assert calls == ["plan", "execution"]


def test_live_lease_renews_and_stale_worker_cannot_write(tmp_path: Path, monkeypatch) -> None:
    from services import workflow_runner as workflow

    now = 100.0
    monkeypatch.setattr(workflow, "time", lambda: now)
    store = RequestStateStore(str(tmp_path))
    owner = workflow.WorkflowCheckpoint(store, "tenant", "lease", "temporal", "run", lease_seconds=10)
    recovery = workflow.WorkflowCheckpoint(store, "tenant", "lease", "temporal", "run", lease_seconds=10)
    owner.reserve("topic")
    assert owner.claim()[1]
    now = 106
    owner.refresh(threading.Event())
    now = 111
    assert recovery.claim()[1] is False
    now = 117
    assert recovery.claim()[1]
    with pytest.raises(workflow.WorkflowLeaseLost):
        owner.persist({"status": "completed"})


def test_durable_tool_approval_does_not_complete_and_cancel_closes_it(tmp_path: Path) -> None:
    from services.workflow_runner import WorkflowCheckpoint, run_durable_attempt

    store = RequestStateStore(str(tmp_path))
    bus = InMemoryStreamEventBus()
    checkpoint = WorkflowCheckpoint(store, "tenant", "tool", "temporal", "run")
    checkpoint.reserve("topic")
    constructions = []

    class Agent:
        def __init__(self):
            constructions.append(True)

        def run_stream(self, topic: str):
            checkpoint.persist({
                "phase": "awaiting_tool_approval", "status": "awaiting_approval",
                "pending_tool_approval": {
                    "approval_id": "A1", "status": "awaiting_approval", "expires_at": time.time() + 60,
                },
            })
            yield {"type": "tool_approval_required", "approval_id": "A1"}

    assert run_durable_attempt(checkpoint, Agent, bus, threading.Event())["status"] == "awaiting_approval"
    assert [item.payload["type"] for item in bus.replay_since("tenant", "tool")] == ["tool_approval_required"]
    assert checkpoint.signal("approve_tool", {"approval_id": "A1"})["accepted"] is False
    assert checkpoint.signal("cancel", {})["accepted"] is True
    assert run_durable_attempt(checkpoint, Agent, bus, threading.Event())["status"] == "cancelled"
    assert len(constructions) == 1
    assert bus.replay_since("tenant", "tool")[-1].payload["type"] == "cancelled"


def test_temporal_explicit_fallback_executes_and_reports_real_backend(tmp_path: Path, monkeypatch) -> None:
    from services.workflow_runner import (
        TemporalWorkflowRunner,
        WorkflowUnavailableError,
    )

    class Agent:
        def __init__(self, config: Configuration):
            pass

        def run_stream(self, topic: str):
            yield {"type": "final_report", "report": topic}
            yield {"type": "done"}

    def unavailable():
        raise ImportError("temporalio")

    monkeypatch.setattr(TemporalWorkflowRunner, "_ensure_temporalio", staticmethod(unavailable))
    store = RequestStateStore(str(tmp_path))
    bus = InMemoryStreamEventBus()
    config = Configuration(workflow_runner="temporal")
    runner = build_workflow_runner(config, agent_factory=Agent, event_bus=bus, state_store=store)
    with pytest.raises(WorkflowUnavailableError):
        runner.submit("test", request_id="fallback", tenant_id="tenant", config=config)
    assert store.load("fallback", tenant_id="tenant") is None

    config = Configuration(workflow_runner="temporal", workflow_runner_fallback_inline=True)
    runner = build_workflow_runner(config, agent_factory=Agent, event_bus=bus, state_store=store)
    handle = runner.submit("test", request_id="fallback", tenant_id="tenant", config=config)
    items = list(bus.tail_since("tenant", "fallback", block_ms=1000))
    runner.shutdown()
    assert handle.runner_backend == "local_durable"
    assert handle.metadata["degraded_reason"] == "temporal_unavailable_fallback_inline"
    assert items[-1].payload["type"] == "done"
