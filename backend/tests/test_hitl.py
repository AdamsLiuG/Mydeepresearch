import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

agent_stub = sys.modules.get("agent") or types.ModuleType("agent")


class _StubImportSafeAgent:
    def __init__(self, config, request_id=None):
        self.config = config
        self.request_id = request_id


if not hasattr(agent_stub, "DeepResearchAgent"):
    agent_stub.DeepResearchAgent = _StubImportSafeAgent
sys.modules.setdefault("agent", agent_stub)

import main
from config import Configuration
from models import SummaryState, TodoItem
from services.evidence import EvidenceStore
from services.hitl import HitlService
from services.request_state import RequestStateStore

_ORIGINAL_FROM_ENV = main.Configuration.from_env


@contextmanager
def _isolated_configuration():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(
            main.Configuration,
            "from_env",
            side_effect=lambda overrides=None, load_env_file=True: _ORIGINAL_FROM_ENV(
                overrides=overrides,
                load_env_file=False,
            ),
        ):
            with patch.object(main, "_warmup_semantic_cache_model", return_value=None):
                yield


class HitlUnitAndApiTests(unittest.TestCase):
    def test_clarify_endpoint_detects_underspecified_vs_specific_topics(self):
        with _isolated_configuration():
            with TestClient(main.create_app()) as client:
                broad_resp = client.post("/research/clarify", json={"topic": "向量数据库"})
                specific_resp = client.post(
                    "/research/clarify",
                    json={"topic": "2026年生产环境高并发部署 Qdrant 与 Milvus 成本架构对比"},
                )

        self.assertEqual(broad_resp.status_code, 200)
        broad = broad_resp.json()
        self.assertTrue(broad["clarification_needed"])
        self.assertGreaterEqual(len(broad["questions"]), 2)
        self.assertGreater(broad["estimated_duration_seconds"], 0)

        self.assertEqual(specific_resp.status_code, 200)
        specific = specific_resp.json()
        self.assertFalse(specific["clarification_needed"])

    def test_hitl_plan_edit_and_resume_workflow_executes_user_edited_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = Configuration.from_env(
                overrides={
                    "request_state_enabled": True,
                    "request_state_dir": tmpdir,
                    "enable_notes": False,
                },
                load_env_file=False,
            )

            class FakeAgent:
                def __init__(self, config, request_id=None):
                    self.config = config
                    self.request_id = request_id or "req-hitl-1"
                    self.store = RequestStateStore(tmpdir)

                def plan_only(self, topic: str):
                    items = [
                        {
                            "id": 1,
                            "title": "初步背景",
                            "intent": "梳理背景",
                            "query": f"{topic} 背景",
                            "status": "pending",
                            "origin": "planned",
                            "round": 1,
                            "depends_on": [],
                        }
                    ]
                    self.store.save(
                        self.request_id,
                        {
                            "snapshot_version": 2,
                            "request_id": self.request_id,
                            "topic": topic,
                            "phase": "awaiting_approval",
                            "status": "awaiting_approval",
                            "todo_items": items,
                        },
                    )
                    return {
                        "request_id": self.request_id,
                        "topic": topic,
                        "status": "awaiting_approval",
                        "phase": "awaiting_approval",
                        "todo_items": items,
                        "estimated_duration_seconds": 45,
                    }

                def run_resume(self, request_id: str):
                    snap = self.store.load(request_id)
                    if not snap:
                        raise ValueError("not found")
                    titles = [t["title"] for t in snap["todo_items"]]
                    return type(
                        "Out",
                        (),
                        {
                            "report_markdown": f"# 基于确认计划\n\n已执行：{', '.join(titles)}",
                            "running_summary": "",
                            "todo_items": [
                                TodoItem(
                                    id=t["id"],
                                    title=t["title"],
                                    intent=t["intent"],
                                    query=t["query"],
                                    status="completed",
                                    depends_on=t.get("depends_on", []),
                                )
                                for t in snap["todo_items"]
                            ],
                        },
                    )()

            with _isolated_configuration():
                with patch.object(main, "DeepResearchAgent", FakeAgent):
                    with TestClient(main.create_app(base_config=base_config)) as client:
                        plan_resp = client.post(
                            "/research/plan",
                            json={"topic": "AI Agent 架构"},
                            headers={"X-Request-ID": "req-hitl-1"},
                        )
                        self.assertEqual(plan_resp.status_code, 200)
                        self.assertEqual(plan_resp.json()["status"], "awaiting_approval")

                        edit_resp = client.put(
                            "/requests/req-hitl-1/plan",
                            json={
                                "tasks": [
                                    {
                                        "id": 1,
                                        "title": "用户新增-成本分析",
                                        "intent": "核算推理成本",
                                        "query": "AI Agent 推理成本",
                                        "depends_on": [2],
                                    },
                                    {
                                        "id": 2,
                                        "title": "用户保留-核心架构",
                                        "intent": "梳理核心架构",
                                        "query": "AI Agent 架构",
                                        "depends_on": [1],
                                    },
                                ]
                            },
                        )
                        self.assertEqual(edit_resp.status_code, 200)
                        edited = edit_resp.json()
                        self.assertEqual(edited["status"], "awaiting_approval")
                        # Cyclic dependency between task 1 and task 2 was broken safely
                        self.assertEqual(edited["todo_items"][1]["depends_on"], [])

                        resume_resp = client.post("/requests/req-hitl-1/resume", json={})
                        self.assertEqual(resume_resp.status_code, 200)
                        self.assertIn("用户新增-成本分析", resume_resp.json()["report_markdown"])
                        self.assertEqual(
                            resume_resp.json()["todo_items"][0]["depends_on"], [2]
                        )

    def test_drilldown_section_merges_grounded_addendum_and_strips_hallucinated_citations(self):
        import agent as agent_module

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RequestStateStore(tmpdir)
            store.save(
                "req-drill-1",
                {
                    "snapshot_version": 2,
                    "request_id": "req-drill-1",
                    "topic": "vLLM 优化",
                    "phase": "completed",
                    "status": "success",
                    "report_markdown": "# 报告\n\n## 背景概览\n初始背景 [T1-S1]。\n\n## 风险与挑战\n现有风险。\n\n## 参考来源\n- [T1-S1] https://docs.vllm.ai",
                    "todo_items": [
                        {
                            "id": 1,
                            "title": "背景梳理",
                            "intent": "背景",
                            "query": "vLLM",
                            "status": "completed",
                            "summary": "PagedAttention 降低显存碎片 [T1-S1]",
                            "evidence_items": [
                                {
                                    "source_id": "T1-S1",
                                    "title": "vLLM Docs",
                                    "url": "https://docs.vllm.ai",
                                    "domain": "docs.vllm.ai",
                                }
                            ],
                        }
                    ],
                },
            )

            cfg = Configuration.from_env(
                overrides={
                    "request_state_enabled": True,
                    "request_state_dir": tmpdir,
                    "enable_notes": False,
                },
                load_env_file=False,
            )
            instance = agent_module.DeepResearchAgent.__new__(agent_module.DeepResearchAgent)
            instance.config = cfg
            instance.request_id = "req-drill-1"
            instance._request_state_store = store
            instance._evidence_store = EvidenceStore()
            instance._request_trace = None
            instance._note_memory = type(
                "NoopNote",
                (),
                {"refresh_notes": staticmethod(lambda *a, **kw: None)},
            )()
            instance._start_request_trace = lambda topic: type(
                "Trace",
                (),
                {
                    "set_task_totals": staticmethod(lambda **kw: None),
                    "attach_result": staticmethod(lambda **kw: None),
                    "complete_request": staticmethod(lambda **kw: {}),
                    "snapshot": staticmethod(lambda: {"request_id": "req-drill-1", "elapsed_ms": 10.0}),
                },
            )()
            instance._request_status = lambda state, report: "success"
            instance._serialize_task = lambda task: {"id": task.id, "title": task.title, "status": task.status}

            def fake_exec(state: SummaryState, task: TodoItem, *, emit_stream: bool, step=None):
                task.status = "completed"
                task.evidence_items = [
                    {
                        "source_id": f"T{task.id}-S1",
                        "title": "Chunked Prefill Paper",
                        "url": "https://arxiv.org/abs/2401.00001",
                        "domain": "arxiv.org",
                    }
                ]
                task.summary = f"补充分析指出 Chunked Prefill 优化吞吐 [T{task.id}-S1]，忽略虚假引用 [T99-S9]。"
                if False:
                    yield {}

            instance._execute_task = fake_exec

            result = instance.drilldown_section(
                "req-drill-1",
                section_title="背景概览",
                question="Chunked Prefill 对首字延迟有什么影响？",
            )

            self.assertEqual(result["task_id"], 2)
            self.assertIn("[T2-S1]", result["addendum_markdown"])
            self.assertNotIn("[T99-S9]", result["addendum_markdown"])
            updated = result["updated_report_markdown"]
            self.assertIn("### 深入补充：背景概览", updated)
            self.assertLess(updated.index("### 深入补充：背景概览"), updated.index("## 风险与挑战"))
            self.assertIn("Chunked Prefill Paper", updated)

            # HitlService merge helper also places addendum before references when heading is unseen
            fallback_merge = HitlService.merge_section_drilldown(
                "# 报告\n\n正文\n\n## 参考来源\n- [T1-S1] a",
                section_title="未知章节",
                addendum_markdown="新增补充",
            )
            self.assertLess(
                fallback_merge.index("### 深入补充：未知章节"),
                fallback_merge.index("## 参考来源"),
            )


if __name__ == "__main__":
    unittest.main()
