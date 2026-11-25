import json
import sys
import types
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

hello_agents_stub = sys.modules.get("hello_agents") or types.ModuleType("hello_agents")


class _StubToolRegistry:
    def register_tool(self, *args, **kwargs):
        return None


class _StubSearchTool:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubToolAwareSimpleAgent:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run(self, prompt: str, **kwargs):
        return prompt

    def clear_history(self):
        return None


hello_agents_stub.HelloAgentsLLM = type("HelloAgentsLLM", (), {})
hello_agents_stub.ToolAwareSimpleAgent = _StubToolAwareSimpleAgent
sys.modules.setdefault("hello_agents", hello_agents_stub)

tools_stub = sys.modules.get("hello_agents.tools") or types.ModuleType("hello_agents.tools")
tools_stub.ToolRegistry = _StubToolRegistry
if not hasattr(tools_stub, "SearchTool"):
    tools_stub.SearchTool = _StubSearchTool
sys.modules.setdefault("hello_agents.tools", tools_stub)
sys.modules.setdefault("hello_agents.tools.builtin", types.ModuleType("hello_agents.tools.builtin"))
note_stub = types.ModuleType("hello_agents.tools.builtin.note_tool")
note_stub.NoteTool = type("_StubNoteTool", (), {})
sys.modules.setdefault("hello_agents.tools.builtin.note_tool", note_stub)

from config import Configuration
from models import DagAction, DagPatch, SummaryState, TodoItem
from services.dag_scheduler import (
    apply_dag_patch,
    build_upstream_context,
    partition_into_waves,
    sanitize_task_dependencies,
    select_wave_one,
)
from services.evidence import EvidenceStore
from services.planner import PlanningService


class DagSchedulerTests(unittest.TestCase):
    def test_sanitize_removes_self_unknown_and_cyclic_dependencies(self):
        t1 = TodoItem(id=1, title="t1", intent="i1", query="q1", depends_on=[1, 99, 2])
        t2 = TodoItem(id=2, title="t2", intent="i2", query="q2", depends_on=[1, 2])
        t3 = TodoItem(id=3, title="t3", intent="i3", query="q3", depends_on=[1, 2, 1])

        sanitize_task_dependencies([t1, t2, t3])
        # t1 accepts dep 2 (valid non-self), then t2 cannot point back to 1 (cycle broken)
        self.assertEqual(t1.depends_on, [2])
        self.assertEqual(t2.depends_on, [])
        self.assertEqual(t3.depends_on, [1, 2])

    def test_partition_into_waves_orders_multi_wave_and_preserves_single_wave_fast_path(self):
        plain = [
            TodoItem(id=1, title="A", intent="i", query="q"),
            TodoItem(id=2, title="B", intent="i", query="q"),
        ]
        self.assertEqual(len(partition_into_waves(plain)), 1)

        t1 = TodoItem(id=1, title="筛选厂商", intent="筛选", query="q1", depends_on=[])
        t2 = TodoItem(id=2, title="行业背景", intent="背景", query="q2", depends_on=[])
        t3 = TodoItem(id=3, title="深度对比", intent="对比", query="q3", depends_on=[1])
        t4 = TodoItem(id=4, title="成本评估", intent="成本", query="q4", depends_on=[1, 3])

        waves = partition_into_waves([t1, t2, t3, t4])
        self.assertEqual([[t.id for t in w] for w in waves], [[1, 2], [3], [4]])

    def test_v4_wave_one_selection_does_not_change_legacy_static_waves(self):
        tasks = [
            TodoItem(id=4, title="A", intent="i", query="q4"),
            TodoItem(id=2, title="B", intent="i", query="q2"),
            TodoItem(id=1, title="C", intent="i", query="q1"),
            TodoItem(id=3, title="D", intent="i", query="q3", depends_on=[1]),
        ]
        tasks[0].coverage_keys = ["vendors"]
        tasks[1].coverage_keys = ["cost"]
        tasks[2].coverage_keys = ["vendors", "cost"]

        selected = select_wave_one(tasks, "standard")
        legacy_waves = partition_into_waves(tasks)

        self.assertEqual([task.id for task in selected], [1, 2])
        self.assertEqual([[task.id for task in wave] for wave in legacy_waves], [[4, 2, 1], [3]])

    def test_invalid_v4_patch_keeps_legacy_task_graph_unchanged(self):
        tasks = [
            TodoItem(id=1, title="Root", intent="Root intent", query="root query", status="completed"),
            TodoItem(id=2, title="Child", intent="Child intent", query="child query", depends_on=[1]),
        ]
        patch = DagPatch(
            base_revision=0,
            decision_id="wave1-r1",
            actions=[
                DagAction(
                    action="PIVOT",
                    task_id=2,
                    parent_task_ids=[1],
                    reason_code="conflict_density",
                    after={"query": "root query"},
                )
            ],
        )

        result = apply_dag_patch(tasks, patch, current_revision=0)

        self.assertFalse(result.applied)
        self.assertEqual([task.query for task in tasks], ["root query", "child query"])
        self.assertEqual(tasks[1].depends_on, [1])

    def test_planner_parses_depends_on_and_sanitizes_cycles(self):
        payload = {
            "tasks": [
                {"id": 1, "title": "梳理主流引擎", "intent": "列出候选引擎", "query": "inference engines", "depends_on": []},
                {"id": 2, "title": "对比基准吞吐", "intent": "基于名单测算吞吐", "query": "throughput benchmark", "depends_on": [1]},
                {"id": 3, "title": "闭环循环测试", "intent": "无效反向依赖", "query": "cycle", "depends_on": [3, 99, 2]},
            ]
        }

        class FakeAgent:
            def run(self, prompt: str, **kwargs):
                return json.dumps(payload, ensure_ascii=False)

            def clear_history(self):
                return None

        planner = PlanningService(FakeAgent(), Configuration.from_env(load_env_file=False))
        state = SummaryState(research_topic="大模型推理框架")
        items = planner.plan_todo_list(state)

        self.assertEqual([t.id for t in items], [1, 2, 3])
        self.assertEqual(items[0].depends_on, [])
        self.assertEqual(items[1].depends_on, [1])
        self.assertEqual(items[2].depends_on, [2])

    def test_two_wave_workflow_injects_upstream_context_and_emits_wave_events(self):
        import agent as agent_module

        cfg = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "skill_pack_enabled": False,
                "request_reflection_enabled": False,
                "review_stage_enabled": False,
                "report_repair_enabled": False,
                "request_state_enabled": False,
            },
            load_env_file=False,
        )
        instance = agent_module.DeepResearchAgent.__new__(agent_module.DeepResearchAgent)
        instance.config = cfg
        instance.request_id = "req-dag-2wave"
        instance._request_trace = None
        instance._skill_pack = SimpleNamespace(enabled=False)
        instance._skill_selection = None
        instance._evidence_store = EvidenceStore()
        instance._request_state_store = None
        instance._state_lock = Lock()
        instance._cancel_event = None
        instance._note_memory = SimpleNamespace(
            search_for_planning=lambda *a, **kw: "",
            refresh_notes=lambda *a, **kw: None,
        )
        instance._strategy_memory = SimpleNamespace(
            search_for_planning=lambda *a, **kw: "",
            search_for_reflection=lambda *a, **kw: "",
            refresh_request=lambda *a, **kw: None,
            run_maintenance=lambda *a, **kw: None,
        )
        instance._tool_tracker = SimpleNamespace(
            drain=lambda *a, **kw: [],
            set_event_sink=lambda *a, **kw: None,
        )
        instance._tool_event_sink_enabled = False
        instance.note_tool = None

        t1 = TodoItem(id=1, title="候选框架筛选", intent="找出热门框架", query="llm frameworks")
        t2 = TodoItem(
            id=2,
            title="吞吐细节拆解",
            intent="分析入选框架吞吐",
            query="throughput details",
            depends_on=[1],
        )
        instance.planner = SimpleNamespace(
            plan_todo_list=lambda state, **kw: [t1, t2],
            create_fallback_task=lambda state: t1,
        )
        instance.reporting = SimpleNamespace(
            generate_report=lambda state, **kw: "# 2-Wave 报告\n\n完成。",
        )
        observed_upstream: dict[int, str] = {}

        def fake_execute_task(state, task, *, emit_stream: bool, step=None):
            observed_upstream[task.id] = task.upstream_context
            task.status = "completed"
            task.summary = (
                "第一波已识别核心框架 vLLM 与 SGLang [T1-S1]"
                if task.id == 1
                else "第二波完成对比评估"
            )
            if False:
                yield {}

        instance._execute_task = fake_execute_task
        instance._warm_topic_search_cache = lambda state, observer=None: None

        events = list(instance.run_stream("大模型推理框架"))
        wave_started = [ev for ev in events if ev.get("type") == "wave_started"]
        wave_completed = [ev for ev in events if ev.get("type") == "wave_completed"]

        self.assertEqual([ev["wave"] for ev in wave_started], [1, 2])
        self.assertEqual([ev["wave"] for ev in wave_completed], [1, 2])
        self.assertEqual(wave_started[0]["task_ids"], [1])
        self.assertEqual(wave_started[1]["task_ids"], [2])
        self.assertEqual(observed_upstream[1], "")
        self.assertIn("vLLM 与 SGLang", observed_upstream[2])

        # Verify build_upstream_context formatting & truncation helper directly
        ctx = build_upstream_context(t2, [t1, t2], max_chars=200)
        self.assertIn("前置任务 1", ctx)
        # Verify snapshot round-trip preserves depends_on
        serialized = instance._serialize_task(t2)
        self.assertEqual(serialized["depends_on"], [1])
        restored = instance._task_from_payload(serialized)
        self.assertEqual(restored.depends_on, [1])


if __name__ == "__main__":
    unittest.main()
