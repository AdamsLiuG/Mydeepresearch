"""Tests for P2-WP3 multi-tenant auth, rate limiting, cache isolation, EventBus & Postgres store."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Configuration, SearchAPI  # noqa: E402
from main import create_app  # noqa: E402
from models import SummaryStateOutput, TodoItem  # noqa: E402
from services.event_bus import InMemoryStreamEventBus, RedisStreamEventBus  # noqa: E402
from services.search import dispatch_search  # noqa: E402
from services.storage_backends import PostgresRequestStateStore  # noqa: E402


class _StubMultiTenantAgent:
    def __init__(self, config: Configuration, request_id: str | None = None, tenant_id: str | None = None) -> None:
        self.config = config
        self.request_id = request_id or "req-stub"
        self.tenant_id = tenant_id or "default"

    def run(self, topic: str) -> SummaryStateOutput:
        return SummaryStateOutput(
            running_summary=f"report for {self.tenant_id}",
            report_markdown=f"# {topic}\n\nTenant: {self.tenant_id}",
            todo_items=[
                TodoItem(
                    id=1,
                    title="Tenant Task",
                    intent="Verify tenant",
                    query=topic,
                    status="completed",
                )
            ],
        )

    def run_stream(self, topic: str):
        yield {"type": "status", "message": "starting", "tenant_id": self.tenant_id}
        yield {"type": "task_status", "task_id": 1, "status": "in_progress"}
        yield {
            "type": "final_report",
            "report": f"# {topic}",
            "todo_items": [],
        }
        yield {"type": "done"}

    def run_resume(self, request_id: str) -> SummaryStateOutput:
        return SummaryStateOutput(
            running_summary=f"# Resumed {request_id}",
            report_markdown=f"# Resumed {request_id}",
            todo_items=[],
        )

    def drilldown_section(self, request_id: str, *, section_title: str, question: str) -> dict:
        return {
            "request_id": request_id,
            "section_title": section_title,
            "question": question,
            "answer_markdown": f"Drilldown for {self.tenant_id}",
        }


class MultiTenantGatewayTests(unittest.TestCase):
    def test_auth_enforces_401_and_supports_x_api_key_and_bearer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Configuration(
                auth_enabled=True,
                tenant_api_keys={"key-alpha": "tenant-alpha", "key-beta": "tenant-beta"},
                request_state_dir=tmpdir,
                llm_api_key="test-key",
            )
            app = create_app(base_config=cfg, agent_factory=_StubMultiTenantAgent)
            client = TestClient(app)

            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/metrics/json").status_code, 401)
            self.assertEqual(client.get("/requests").status_code, 401)
            self.assertEqual(
                client.post(
                    "/research",
                    json={"topic": "RAG eval"},
                    headers={"X-API-Key": "wrong-key"},
                ).status_code,
                401,
            )

            resp_alpha = client.post(
                "/research",
                json={"topic": "RAG eval"},
                headers={"X-API-Key": "key-alpha"},
            )
            self.assertEqual(resp_alpha.status_code, 200)
            self.assertIn("tenant-alpha", resp_alpha.json()["report_markdown"])

            resp_beta = client.post(
                "/research",
                json={"topic": "RAG eval"},
                headers={"Authorization": "Bearer key-beta"},
            )
            self.assertEqual(resp_beta.status_code, 200)
            self.assertIn("tenant-beta", resp_beta.json()["report_markdown"])

    def test_per_tenant_rate_limit_returns_429_with_retry_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Configuration(
                auth_enabled=True,
                tenant_api_keys={"key-alpha": "tenant-alpha", "key-beta": "tenant-beta"},
                tenant_rate_limit_rpm=2,
                request_state_dir=tmpdir,
                llm_api_key="test-key",
            )
            app = create_app(base_config=cfg, agent_factory=_StubMultiTenantAgent)
            client = TestClient(app)

            for _ in range(2):
                ok = client.post(
                    "/research",
                    json={"topic": "Topic"},
                    headers={"X-API-Key": "key-alpha"},
                )
                self.assertEqual(ok.status_code, 200)

            limited = client.post(
                "/research",
                json={"topic": "Topic"},
                headers={"X-API-Key": "key-alpha"},
            )
            self.assertEqual(limited.status_code, 429)
            self.assertIn("Retry-After", limited.headers)
            self.assertGreaterEqual(int(limited.headers["Retry-After"]), 1)

            # Separate tenant retains independent quota.
            other = client.post(
                "/research",
                json={"topic": "Topic"},
                headers={"X-API-Key": "key-beta"},
            )
            self.assertEqual(other.status_code, 200)

    def test_cross_tenant_snapshot_and_hitl_access_returns_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Configuration(
                auth_enabled=True,
                tenant_api_keys={"key-alpha": "tenant-alpha", "key-beta": "tenant-beta"},
                request_state_dir=tmpdir,
                llm_api_key="test-key",
            )
            app = create_app(base_config=cfg, agent_factory=_StubMultiTenantAgent)
            store = app.state.request_state_store
            assert store is not None
            store.save(
                "req-alpha-1",
                {
                    "request_id": "req-alpha-1",
                    "topic": "Secret Alpha Research",
                    "phase": "awaiting_approval",
                    "status": "awaiting_approval",
                    "todo_items": [{"id": 1, "title": "Task 1", "intent": "Intent 1", "query": "q1"}],
                    "completed": False,
                },
                tenant_id="tenant-alpha",
            )

            client = TestClient(app)
            alpha_list = client.get("/requests", headers={"X-API-Key": "key-alpha"})
            beta_list = client.get("/requests", headers={"X-API-Key": "key-beta"})
            self.assertEqual([item["request_id"] for item in alpha_list.json()["items"]], ["req-alpha-1"])
            self.assertEqual(beta_list.json()["items"], [])

            self.assertEqual(
                client.get("/requests/req-alpha-1", headers={"X-API-Key": "key-alpha"}).status_code,
                200,
            )
            for req in (
                lambda: client.get("/requests/req-alpha-1", headers={"X-API-Key": "key-beta"}),
                lambda: client.put(
                    "/requests/req-alpha-1/plan",
                    headers={"X-API-Key": "key-beta"},
                    json={"tasks": [{"title": "T1", "intent": "I1", "query": "q1"}]},
                ),
                lambda: client.post(
                    "/requests/req-alpha-1/drilldown",
                    headers={"X-API-Key": "key-beta"},
                    json={"section_title": "Sec", "question": "Q?"},
                ),
                lambda: client.post(
                    "/requests/req-alpha-1/resume",
                    headers={"X-API-Key": "key-beta"},
                ),
                lambda: client.get(
                    "/requests/req-alpha-1/events",
                    headers={"X-API-Key": "key-beta"},
                ),
            ):
                self.assertEqual(req().status_code, 404)

    def test_sse_stream_publishes_event_ids_and_supports_last_event_id_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Configuration(
                auth_enabled=True,
                tenant_api_keys={"key-alpha": "tenant-alpha", "key-beta": "tenant-beta"},
                request_state_dir=tmpdir,
                llm_api_key="test-key",
            )
            app = create_app(base_config=cfg, agent_factory=_StubMultiTenantAgent)
            client = TestClient(app)

            stream_resp = client.post(
                "/research/stream",
                json={"topic": "SSE Replay"},
                headers={"X-API-Key": "key-alpha", "X-Request-ID": "req-stream-1"},
            )
            self.assertEqual(stream_resp.status_code, 200)
            self.assertIn("id: 1\n", stream_resp.text)
            self.assertIn("id: 2\n", stream_resp.text)

            replay_json = client.get(
                "/requests/req-stream-1/events",
                params={"last_event_id": "2"},
                headers={"X-API-Key": "key-alpha"},
            )
            self.assertEqual(replay_json.status_code, 200)
            events = replay_json.json()["events"]
            self.assertEqual([e["id"] for e in events], ["3", "4"])
            self.assertEqual(events[0]["data"]["type"], "final_report")

            replay_sse = client.get(
                "/requests/req-stream-1/events",
                headers={
                    "X-API-Key": "key-alpha",
                    "Last-Event-ID": "3",
                    "Accept": "text/event-stream",
                },
            )
            self.assertEqual(replay_sse.status_code, 200)
            self.assertIn("id: 4\n", replay_sse.text)
            self.assertNotIn("id: 3\n", replay_sse.text)

            # Another tenant sees zero buffered events.
            beta_replay = client.get(
                "/requests/req-stream-1/events",
                headers={"X-API-Key": "key-beta"},
            )
            self.assertEqual(beta_replay.status_code, 200)
            self.assertEqual(beta_replay.json()["events"], [])


class SearchCacheAndBackendAdapterTests(unittest.TestCase):
    def test_search_cache_isolates_by_tenant_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Configuration(
                search_api=SearchAPI.DUCKDUCKGO,
                search_cache_enabled=True,
                search_cache_dir=tmpdir,
                llm_api_key="test-key",
            )
            calls = {"count": 0}

            def _fake_provider(query: str, backend: str, config, loop_count: int, **kwargs):
                calls["count"] += 1
                return (
                    {
                        "results": [
                            {
                                "title": f"Result {calls['count']}",
                                "url": f"https://example.com/{calls['count']}",
                                "content": f"Content {calls['count']} for {query}",
                            }
                        ],
                    },
                    [],
                    None,
                    backend,
                )

            with patch("services.search._execute_search_backend", side_effect=_fake_provider):
                res_a1 = dispatch_search(
                    "agent memory",
                    cfg,
                    0,
                    cache_context={"tenant_id": "tenant-alpha", "research_topic": "agents"},
                )[0]
                res_a2 = dispatch_search(
                    "agent memory",
                    cfg,
                    0,
                    cache_context={"tenant_id": "tenant-alpha", "research_topic": "agents"},
                )[0]
                res_b1 = dispatch_search(
                    "agent memory",
                    cfg,
                    0,
                    cache_context={"tenant_id": "tenant-beta", "research_topic": "agents"},
                )[0]

            self.assertEqual(calls["count"], 2)
            self.assertEqual(res_a1["results"][0]["title"], res_a2["results"][0]["title"])
            self.assertNotEqual(res_a1["results"][0]["title"], res_b1["results"][0]["title"])

    def test_inmemory_and_redis_event_bus_replay(self) -> None:
        mem_bus = InMemoryStreamEventBus(max_events_per_request=20)
        e1 = mem_bus.publish("t1", "r1", {"type": "a"})
        e2 = mem_bus.publish("t1", "r1", {"type": "b"})
        self.assertEqual(e1, "1")
        self.assertEqual(e2, "2")
        self.assertEqual([item.event_id for item in mem_bus.replay_since("t1", "r1", "1")], ["2"])

        class _FakeRedis:
            def __init__(self) -> None:
                self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
                self.counter = 0

            def xadd(self, key: str, fields: dict[str, str], maxlen=None, approximate=True) -> str:
                self.counter += 1
                msg_id = f"1700000000000-{self.counter}"
                self.streams.setdefault(key, []).append((msg_id, fields))
                return msg_id

            def xrange(self, key: str, min: str = "-", max: str = "+"):
                items = self.streams.get(key, [])
                if min.startswith("("):
                    after = min[1:]
                    return [(mid, f) for mid, f in items if mid > after]
                return list(items)

        fake_redis = _FakeRedis()
        redis_bus = RedisStreamEventBus(client=fake_redis)
        m1 = redis_bus.publish("t1", "r1", {"type": "status", "step": 1})
        m2 = redis_bus.publish("t1", "r1", {"type": "done", "step": 2})
        replayed = redis_bus.replay_since("t1", "r1", last_event_id=m1)
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0].event_id, m2)
        self.assertEqual(replayed[0].payload["step"], 2)

    def test_postgres_request_state_store_uses_parameterized_queries(self) -> None:
        executed: list[tuple[str, tuple | None]] = []

        class _FakeCursor:
            def __init__(self, store_rows: dict[tuple[str, str], str]) -> None:
                self._rows = store_rows
                self._fetched = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, sql: str, params=None):
                executed.append((sql, params))
                normalized = " ".join(sql.split()).upper()
                if normalized.startswith("INSERT INTO"):
                    req_id, tenant_id, _topic, _phase, _status, _upd, payload_json = params
                    self._rows[(req_id, tenant_id)] = payload_json
                elif "WHERE REQUEST_ID = %S AND TENANT_ID = %S" in normalized:
                    req_id, tenant_id = params
                    val = self._rows.get((req_id, tenant_id))
                    self._fetched = [(tenant_id, val)] if val is not None else []
                elif "WHERE REQUEST_ID = %S" in normalized:
                    req_id = params[0]
                    matches = [(t, v) for (r, t), v in self._rows.items() if r == req_id]
                    self._fetched = matches[:1]
                elif "FROM DEEPRESEARCH_REQUEST_STATES WHERE TENANT_ID = %S" in normalized:
                    tenant_id, _limit = params
                    self._fetched = [(v,) for (_r, t), v in self._rows.items() if t == tenant_id]

            def fetchone(self):
                return self._fetched[0] if self._fetched else None

            def fetchall(self):
                return list(self._fetched)

        rows_backing: dict[tuple[str, str], str] = {}

        class _FakeConn:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def cursor(self):
                return _FakeCursor(rows_backing)

            def commit(self):
                return None

        store = PostgresRequestStateStore(
            dsn="postgresql://user:pass@localhost:5432/db",
            connection_factory=_FakeConn,
        )
        store.save(
            "req-pg-1",
            {"topic": "PG test", "phase": "completed", "status": "completed", "completed": True},
            tenant_id="tenant-pg",
        )
        loaded = store.load("req-pg-1", tenant_id="tenant-pg")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["tenant_id"], "tenant-pg")
        self.assertIsNone(store.load("req-pg-1", tenant_id="other-tenant"))

        recent = store.list_recent(tenant_id="tenant-pg")
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["request_id"], "req-pg-1")
        # Confirm every non-DDL query bound parameters safely via %s placeholders.
        dml_queries = [q for q, p in executed if p is not None]
        self.assertTrue(all("%s" in q for q in dml_queries))
