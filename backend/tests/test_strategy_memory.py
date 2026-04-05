import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

hello_agents_stub = sys.modules.get("hello_agents") or types.ModuleType("hello_agents")


class DummyToolAwareSimpleAgent:
    pass


hello_agents_stub.ToolAwareSimpleAgent = DummyToolAwareSimpleAgent
sys.modules.setdefault("hello_agents", hello_agents_stub)

from config import Configuration
from metrics import RequestTrace, metrics_registry
from services import strategy_memory as strategy_memory_module
from services.strategy_memory import StrategyMemoryService
from services.strategy_synthesizer import (
    StrategyCard,
    StrategySourceRequest,
    StrategySynthesizer,
)


def keyword_embedding(text: str) -> list[float]:
    normalized = str(text or "").lower()
    tokens = [
        "mcp",
        "protocol",
        "deployment",
        "official",
        "anti",
        "planning",
        "reflection",
        "failure",
        "repair",
        "memory",
    ]
    vector = [float(normalized.count(token)) for token in tokens]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [1.0] + [0.0 for _ in range(len(tokens) - 1)]
    return [value / norm for value in vector]


class FakeCollection:
    def __init__(self) -> None:
        self.records = {}

    def upsert(self, *, ids, documents, metadatas, embeddings) -> None:
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.records[item_id] = {
                "id": item_id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def delete(self, *, ids) -> None:
        for item_id in ids:
            self.records.pop(item_id, None)

    def query(self, *, query_embeddings, n_results, include):
        query_embedding = query_embeddings[0]
        ranked = []
        for record in self.records.values():
            embedding = record["embedding"]
            dot = sum(left * right for left, right in zip(query_embedding, embedding))
            left_norm = math.sqrt(sum(value * value for value in query_embedding))
            right_norm = math.sqrt(sum(value * value for value in embedding))
            similarity = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
            ranked.append((1.0 - similarity, record))
        ranked.sort(key=lambda item: item[0])
        top = ranked[:n_results]
        return {
            "documents": [[item[1]["document"] for item in top]],
            "distances": [[item[0] for item in top]],
            "metadatas": [[item[1]["metadata"] for item in top]],
        }


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self.collections:
            self.collections[name] = FakeCollection()
        return self.collections[name]

    def delete_collection(self, name) -> None:
        self.collections.pop(name, None)


class FakeSynthesizer:
    def synthesize(self, source_request: StrategySourceRequest) -> list[StrategyCard]:
        cards: list[StrategyCard] = []
        for kind in source_request.requested_kinds:
            title_prefix = {
                "planning_pattern": "Planning strategy",
                "reflection_pattern": "Reflection strategy",
                "anti_pattern": "Anti pattern",
            }[kind]
            cards.append(
                StrategyCard(
                    strategy_id=f"{source_request.request_id}::{kind}",
                    strategy_kind=kind,
                    stage_scope="planning" if kind == "planning_pattern" else "reflection",
                    title=f"{title_prefix} {source_request.request_id}",
                    applicable_when=f"{source_request.topic} 出现 deployment / official docs / repair 信号时适用",
                    match_signals=[
                        f"topic:{source_request.topic}",
                        "缺少官方文档",
                        "deployment monitoring",
                    ],
                    recommended_actions=(
                        ["优先查官方文档", "增加 deployment / observability 维度", "交叉验证 maintainer sources"]
                        if kind != "anti_pattern"
                        else ["不要只依赖二手博客"]
                    ),
                    query_templates=[
                        "official docs deployment observability",
                        "maintainer blog release notes",
                    ],
                    preferred_sources=["official docs", "maintainer blog"],
                    pitfalls_to_avoid=[
                        "不要只依赖二手博客",
                        "不要忽略 repair 和 failure signal",
                    ],
                    origin_request_id=source_request.request_id,
                    origin_status=source_request.status,
                    origin_review_status=source_request.review_status,
                    origin_task_ids=[
                        int(task["id"])
                        for task in source_request.tasks
                        if str(task.get("id") or "").isdigit()
                    ],
                    created_at="2026-04-03T00:00:00+00:00",
                )
            )
        return cards


class StubSynthAgent:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts = []
        self.clear_history_calls = 0

    def run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response

    def clear_history(self) -> None:
        self.clear_history_calls += 1


class StrategyMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.requests_dir = self.root / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.root / "strategy-memory"
        self.config = Configuration.from_env(
            overrides={
                "request_state_enabled": True,
                "request_state_dir": str(self.requests_dir),
                "strategy_memory_enabled": True,
                "strategy_memory_dir": str(self.memory_dir),
            },
            load_env_file=False,
        )
        self.observer = RequestTrace(
            request_id="req-current",
            topic="MCP protocol deployment",
            search_api="duckduckgo",
            provider="custom",
            model="demo-model",
            pricing_catalog={},
        )
        self.embedding_patches = [
            patch.object(strategy_memory_module, "embeddings_available", return_value=True),
            patch.object(
                strategy_memory_module,
                "encode_text",
                side_effect=lambda text, **_: keyword_embedding(text),
            ),
            patch.object(
                strategy_memory_module,
                "encode_texts",
                side_effect=lambda texts, **_: [keyword_embedding(text) for text in texts],
            ),
        ]
        for active_patch in self.embedding_patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.embedding_patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def _write_snapshot(
        self,
        request_id: str,
        *,
        topic: str,
        status: str,
        report_markdown: str = "# report",
        review_status: str = "passed",
        report_repair_cycles: int = 0,
        todo_items: list[dict] | None = None,
    ) -> None:
        (self.requests_dir / f"{request_id}.json").write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "topic": topic,
                    "status": status,
                    "phase": "completed" if status != "failed" else "failed",
                    "updated_at": "2026-04-03T12:00:00+00:00",
                    "report_markdown": report_markdown,
                    "review_summary": {
                        "overall_status": review_status,
                        "reason": "ok",
                        "issue_count": 0,
                    },
                    "report_repair_cycles": report_repair_cycles,
                    "todo_items": todo_items
                    or [
                        {
                            "id": 1,
                            "title": "Deployment overview",
                            "intent": "梳理部署与 observability",
                            "query": "mcp protocol deployment observability",
                            "status": "completed" if status != "failed" else "failed",
                            "summary": "官方文档和 deployment guidance 很关键。",
                            "notices": ["official_docs_missing"],
                            "review_issues": [],
                            "react_gap_signals": ["official_docs_missing"],
                            "react_last_action": "rewrite_query",
                            "react_stop_reason": "",
                        }
                    ],
                    "request_metrics": {
                        "degraded_reasons": [],
                        "reflection_gap_signals": ["official_docs_missing"],
                        "reflection_reason": "仍缺少 deployment 视角",
                        "report_repair_cycles": report_repair_cycles,
                        "report_repair_triggered": report_repair_cycles > 0,
                        "report_repair_added_tasks": 1 if report_repair_cycles > 0 else 0,
                        "updated_at": "2026-04-03T12:00:00+00:00",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_reconcile_builds_positive_and_anti_cards_and_queries_by_stage(self):
        self._write_snapshot(
            "req-success",
            topic="MCP protocol deployment",
            status="success",
        )
        self._write_snapshot(
            "req-failed",
            topic="MCP protocol deployment failure",
            status="failed",
            report_markdown="",
        )
        service = StrategyMemoryService(
            self.config,
            synthesizer=FakeSynthesizer(),
            client=FakeChromaClient(),
        )

        planning_context = service.search_for_planning(
            "MCP protocol deployment official docs",
            current_request_id="req-current",
            observer=self.observer,
        )
        reflection_context = service.search_for_reflection(
            "MCP protocol deployment official docs",
            gap_signals=["official_docs_missing", "deployment_gap"],
            task_titles=["Deployment overview"],
            current_request_id="req-current",
            observer=self.observer,
        )

        self.assertIn("planning_pattern", planning_context)
        self.assertIn("reflection_pattern", reflection_context)
        self.assertIn("anti_pattern", reflection_context)
        snapshot = self.observer.snapshot()
        self.assertEqual(snapshot["strategy_memory_queries"], 2)
        self.assertGreaterEqual(snapshot["strategy_memory_hits"], 2)
        self.assertEqual(snapshot["strategy_memory_prompt_injections"], 2)

    def test_failed_or_repair_history_only_generates_anti_pattern(self):
        self._write_snapshot(
            "req-repair",
            topic="MCP protocol deployment repair",
            status="partial_success",
            report_repair_cycles=1,
        )
        service = StrategyMemoryService(
            self.config,
            synthesizer=FakeSynthesizer(),
            client=FakeChromaClient(),
        )

        service.ensure_reconciled(observer=self.observer)
        manifest = service._load_manifest()

        self.assertEqual(manifest["requests"]["req-repair"]["card_ids"], ["req-repair::anti_pattern"])
        planning_context = service.search_for_planning(
            "MCP protocol deployment repair",
            current_request_id="req-current",
            observer=self.observer,
        )
        self.assertIn("anti_pattern", planning_context)
        self.assertNotIn("planning_pattern", planning_context)

    def test_search_excludes_current_request_and_refresh_request_reindexes_cards(self):
        self._write_snapshot(
            "req-current",
            topic="MCP protocol deployment",
            status="success",
        )
        self._write_snapshot(
            "req-history",
            topic="MCP protocol deployment official docs",
            status="success",
        )
        service = StrategyMemoryService(
            self.config,
            synthesizer=FakeSynthesizer(),
            client=FakeChromaClient(),
        )

        context = service.search_for_planning(
            "MCP protocol deployment official docs",
            current_request_id="req-current",
            observer=self.observer,
        )
        self.assertIn("req-history", context)
        self.assertNotIn("req-current", context)

        self._write_snapshot(
            "req-history",
            topic="MCP protocol deployment observability",
            status="success",
        )
        service.refresh_request("req-history", observer=self.observer)
        refreshed = service.search_for_planning(
            "MCP protocol deployment observability",
            current_request_id="req-current",
            observer=self.observer,
        )
        self.assertIn("observability", refreshed)


class StrategySynthesizerTests(unittest.TestCase):
    def test_synthesizer_parses_valid_json_array(self):
        response = json.dumps(
            [
                {
                    "strategy_kind": "planning_pattern",
                    "title": "优先官方文档",
                    "applicable_when": "主题涉及协议或部署细节时",
                    "match_signals": ["缺少官方文档"],
                    "recommended_actions": ["先查官方文档"],
                    "query_templates": ["official docs deployment observability"],
                    "preferred_sources": ["official docs"],
                    "pitfalls_to_avoid": ["不要只看二手博客"],
                }
            ],
            ensure_ascii=False,
        )
        agent = StubSynthAgent(response)
        synthesizer = StrategySynthesizer(
            lambda: agent,
            Configuration.from_env(load_env_file=False),
        )

        cards = synthesizer.synthesize(
            StrategySourceRequest(
                request_id="req-1",
                topic="MCP protocol deployment",
                status="success",
                review_status="passed",
                report_available=True,
                completed_task_count=1,
                failed_task_count=0,
                repair_cycles=0,
                requested_kinds=["planning_pattern"],
                tasks=[{"id": 1, "title": "Deployment", "intent": "梳理部署", "status": "completed"}],
            )
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].strategy_kind, "planning_pattern")
        self.assertEqual(cards[0].origin_request_id, "req-1")
        self.assertEqual(agent.clear_history_calls, 1)

    def test_synthesizer_rejects_non_array_json(self):
        agent = StubSynthAgent('{"strategy_kind":"planning_pattern"}')
        synthesizer = StrategySynthesizer(
            lambda: agent,
            Configuration.from_env(load_env_file=False),
        )

        cards = synthesizer.synthesize(
            StrategySourceRequest(
                request_id="req-2",
                topic="MCP protocol deployment",
                status="success",
                review_status="passed",
                report_available=True,
                completed_task_count=1,
                failed_task_count=0,
                repair_cycles=0,
                requested_kinds=["planning_pattern"],
            )
        )

        self.assertEqual(cards, [])


if __name__ == "__main__":
    unittest.main()
