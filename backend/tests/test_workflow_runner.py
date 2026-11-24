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
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.zsets: dict[str, dict[str, float]] = {}
            self.counts: dict[str, int] = {}

        def incr(self, key: str) -> int:
            if self.fail:
                raise ConnectionError("redis down")
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

        def decr(self, key: str) -> int:
            if self.fail:
                raise ConnectionError("redis down")
            self.counts[key] = self.counts.get(key, 0) - 1
            return self.counts[key]

        def expire(self, key: str, ttl: int) -> None:
            del key, ttl

        def delete(self, key: str) -> None:
            self.counts.pop(key, None)

        def zremrangebyscore(self, key: str, min_s: Any, max_s: float) -> None:
            del min_s
            bucket = self.zsets.setdefault(key, {})
            for member in [m for m, score in bucket.items() if score <= max_s]:
                del bucket[member]

        def zcard(self, key: str) -> int:
            return len(self.zsets.get(key, {}))

        def zadd(self, key: str, mapping: dict[str, float]) -> None:
            self.zsets.setdefault(key, {}).update(mapping)

        def zrange(self, key: str, start: int, end: int, withscores: bool = False) -> list[Any]:
            items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
            return items[start : end + 1] if withscores else [k for k, _ in items[start : end + 1]]

    limiter = TenantRateLimiter(rpm_limit=2, max_concurrent=5, redis_client=_FakeRedisRL())
    ok1, _, _ = limiter.try_acquire("t1")
    limiter.release("t1")
    ok2, _, _ = limiter.try_acquire("t1")
    limiter.release("t1")
    ok3, retry_after, reason = limiter.try_acquire("t1")
    assert ok1 is True and ok2 is True
    assert ok3 is False and reason == "rate_limit_rpm" and retry_after >= 1

    # Fallback on Redis failure keeps working in-process
    degraded_limiter = TenantRateLimiter(
        rpm_limit=1,
        max_concurrent=1,
        redis_client=_FakeRedisRL(fail=True),
    )
    a1, _, _ = degraded_limiter.try_acquire("t2")
    a2, _, reason2 = degraded_limiter.try_acquire("t2")
    assert a1 is True and a2 is False and reason2 == "max_concurrent_requests"


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
