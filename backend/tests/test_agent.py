import sys
import time
import types
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


hello_agents_stub = sys.modules.get("hello_agents") or types.ModuleType("hello_agents")


class DummyLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyToolAwareSimpleAgent:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run(self, prompt: str, **kwargs):
        return prompt

    def stream_run(self, prompt: str, **kwargs):
        if False:
            yield prompt

    def clear_history(self):
        return None


hello_agents_stub.HelloAgentsLLM = DummyLLM
hello_agents_stub.ToolAwareSimpleAgent = DummyToolAwareSimpleAgent
sys.modules.setdefault("hello_agents", hello_agents_stub)

tools_stub = sys.modules.get("hello_agents.tools") or types.ModuleType("hello_agents.tools")


class DummyToolRegistry:
    def register_tool(self, tool):
        self.tool = tool


class DummySearchTool:
    def __init__(self, backend="hybrid"):
        self.backend = backend

    def run(self, payload):
        return payload


tools_stub.ToolRegistry = DummyToolRegistry
tools_stub.SearchTool = DummySearchTool
sys.modules.setdefault("hello_agents.tools", tools_stub)

tools_builtin_stub = types.ModuleType("hello_agents.tools.builtin")
sys.modules.setdefault("hello_agents.tools.builtin", tools_builtin_stub)

note_tool_stub = types.ModuleType("hello_agents.tools.builtin.note_tool")


class DummyNoteTool:
    def __init__(self, workspace=None):
        self.workspace = workspace

    def run(self, payload):
        return "OK"


note_tool_stub.NoteTool = DummyNoteTool
sys.modules.setdefault("hello_agents.tools.builtin.note_tool", note_tool_stub)

import agent as agent_module
import task_executor as task_executor_module
from config import Configuration
from metrics import RequestTrace, metrics_registry
from models import SummaryState, TodoItem
from services.evidence import EvidenceStore
from services.reflection import ReflectionAssessment


class AgentDecompositionTests(unittest.TestCase):
    def test_agent_uses_dedicated_orchestration_mixins(self):
        from repair_orchestrator import RepairOrchestratorMixin
        from state_manager import StateManagerMixin
        from stream_coordinator import StreamCoordinatorMixin
        from task_executor import TaskExecutorMixin

        self.assertTrue(issubclass(agent_module.DeepResearchAgent, StateManagerMixin))
        self.assertTrue(issubclass(agent_module.DeepResearchAgent, StreamCoordinatorMixin))
        self.assertTrue(issubclass(agent_module.DeepResearchAgent, RepairOrchestratorMixin))
        self.assertTrue(issubclass(agent_module.DeepResearchAgent, TaskExecutorMixin))

    def test_task_execution_helpers_live_on_task_executor_mixin(self):
        from task_executor import TaskExecutorMixin

        expected_helpers = [
            "_run_with_timeout",
            "_dispatch_search_with_guardrails",
            "_normalize_query_candidate",
            "_task_search_queries",
            "_observe_task_evidence",
            "_fallback_task_react_decision",
            "_plan_task_react_decision",
            "_search_with_fallback_queries",
            "_execute_task",
            "_record_task_failure",
        ]

        for helper_name in expected_helpers:
            self.assertIn(helper_name, TaskExecutorMixin.__dict__)
            self.assertNotIn(helper_name, agent_module.DeepResearchAgent.__dict__)


class StubTracker:
    def drain(self, state, *, step=None):
        return []

    def as_dicts(self):
        return []

    def set_event_sink(self, sink):
        self.sink = sink


class AgentExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()

    def test_web_leads_cannot_close_react_coverage_or_drive_wave_one(self):
        instance = self._build_agent()
        instance._evidence_store = EvidenceStore(web_extraction_enabled=True)
        task = TodoItem(id=1, title="Report", intent="Find measurements", query="measurements")
        state = SummaryState(research_topic="Measurements", todo_items=[task])
        leads = instance._evidence_store.record_search_results(
            task_id=1, query=task.query, backend="searxng",
            search_payload={"results": [{
                "url": "https://example.org/research", "title": "Report",
                "content": "Provider lead " * 40, "raw_content": "UNTRUSTED",
            }]},
        )
        self.assertIn("thin_content", instance._observe_task_evidence(state, task, leads).gap_signals)
        self.assertFalse(instance._observe_task_evidence(state, task, leads).evidence_sufficiency)
        self.assertEqual(instance._wave_material(state), {"facts": [], "sources": []})

    def _build_agent(self):
        instance = agent_module.DeepResearchAgent.__new__(agent_module.DeepResearchAgent)
        instance.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_reflection_enabled": False,
                "review_stage_enabled": False,
                "request_state_enabled": False,
                "task_react_enabled": False,
                "report_repair_enabled": False,
                "search_cache_enabled": False,
                "semantic_cache_enabled": False,
            },
            load_env_file=False,
        )
        instance.request_id = "req-test"
        instance.note_tool = None
        instance.tools_registry = None
        instance._request_state_store = None
        instance._evidence_store = EvidenceStore()
        instance._tool_tracker = StubTracker()
        instance._tool_event_sink_enabled = False
        instance._state_lock = Lock()
        instance._last_search_notices = []
        instance._request_trace = None
        instance._note_memory = SimpleNamespace(
            search_for_planning=lambda *args, **kwargs: "",
            search_for_task=lambda *args, **kwargs: "",
            refresh_notes=lambda *args, **kwargs: None,
        )
        instance._strategy_memory = SimpleNamespace(
            search_for_planning=lambda *args, **kwargs: "",
            search_for_reflection=lambda *args, **kwargs: "",
            refresh_request=lambda *args, **kwargs: None,
        )
        instance._skill_pack = SimpleNamespace(enabled=False)
        instance._skill_selection = None
        instance.planner = SimpleNamespace(
            plan_todo_list=lambda state, observer=None, historical_memory_context=None, strategy_memory_context=None, skill_pack_context=None: [
                TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")
            ],
            plan_additional_tasks=lambda state, **kwargs: [],
            create_fallback_task=lambda state: TodoItem(
                id=1,
                title="兜底任务",
                intent="收集背景",
                query=state.research_topic,
            ),
        )
        instance.reflection = SimpleNamespace(
            assess_request=lambda state, gap_signals, strategy_memory_context=None, skill_pack_context=None, observer=None: None
        )
        instance.reporting = SimpleNamespace(
            generate_report=lambda state, observer=None, skill_pack_context=None: "# report"
        )
        instance.summarizer = SimpleNamespace(
            summarize_task=lambda state, task, context, observer=None, historical_memory_context=None, skill_pack_context=None: "summary"
        )
        instance.task_react_agent = SimpleNamespace(
            run=lambda prompt: '{"action":"stop","reason":"stub"}',
            clear_history=lambda: None,
        )
        return instance

    def test_run_consumes_task_generator_for_sync_endpoint(self):
        agent = self._build_agent()
        calls = []

        def fake_execute_task(state, task, emit_stream=False, step=None):
            calls.append(task.id)
            task.status = "completed"
            task.summary = "summary"
            task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        agent._execute_task = fake_execute_task

        result = agent.run("AI agent")

        self.assertEqual(calls, [1])
        self.assertEqual(result.todo_items[0].status, "completed")
        self.assertEqual(result.todo_items[0].summary, "summary")

    def test_v4_wave_barrier_and_resume_do_not_repeat_decision_or_tasks(self):
        from tempfile import TemporaryDirectory

        from services.request_state import RequestStateStore

        with TemporaryDirectory() as directory:
            store = RequestStateStore(directory)
            agent = self._build_agent()
            agent._request_state_store = store
            agent.config = agent.config.model_copy(update={
                "v4_research_budget_enabled": True,
                "v4_replan_enabled": True,
                "v4_default_research_depth": "quick",
            })
            agent.planner.plan_todo_list = lambda *args, **kwargs: [
                TodoItem(id=i, title=f"Angle {i}", intent="Test", query=f"Topic {i}") for i in (1, 2)
            ]
            calls = []

            def batch(state, tasks, channels):
                calls.append([task.id for task in tasks])
                for task in tasks:
                    task.status = "completed"
                yield from ()

            agent._execute_task_batch_stream = batch
            events = agent.run_stream("Topic")
            first_todo = None
            for event in events:
                if event["type"] == "todo_list":
                    first_todo = event
                if event["type"] == "budget_updated" and event.get("boundary") == "wave_1":
                    break
            events.close()
            snapshot = store.load(agent.request_id)
            self.assertEqual(first_todo["dag_revision"], 1)
            self.assertIn("state_version", first_todo)
            self.assertEqual(calls, [[1]])
            self.assertTrue(snapshot["v4"]["replan"]["wave_1_committed"])
            self.assertEqual(snapshot["v4"]["budget"]["snapshot"]["spent"]["tasks"], 2)
            recovered = self._build_agent()
            recovered._request_state_store = store
            recovered._execute_task_batch_stream = batch
            payloads = list(recovered.run_stream_resume(recovered.request_id))
            self.assertEqual(calls, [[1], [2]])
            self.assertEqual([event["type"] for event in payloads[-2:]], ["final_report", "done"])
            self.assertEqual(payloads[-2]["v4"]["budget"]["limit"]["depth"], "quick")

    def test_v4_zero_tasks_returns_a_deterministic_partial_report(self):
        agent = self._build_agent()
        agent.config = agent.config.model_copy(update={
            "v4_research_budget_enabled": True,
            "v4_research_budget_limits": {"tasks": 0, "llm_calls": 0},
        })
        agent.reporting.generate_report = lambda *args, **kwargs: self.fail("report provider called")
        events = list(agent.run_stream("Budget test"))
        self.assertEqual([event["type"] for event in events[-2:]], ["final_report", "done"])
        self.assertIn("Budget test", events[-2]["report"])
        self.assertEqual(agent._request_trace.snapshot()["status"], "partial_success")

    def test_v4_plan_draft_is_approved_once_without_truncating_the_edited_graph(self):
        from tempfile import TemporaryDirectory

        from services.request_state import RequestStateStore

        with TemporaryDirectory() as directory:
            store = RequestStateStore(directory)
            agent = self._build_agent()
            agent._request_state_store = store
            agent.config = agent.config.model_copy(update={"v4_research_budget_enabled": True, "v4_replan_enabled": True})
            draft = agent.plan_only("Topic")
            self.assertEqual(draft["plan_revision"], 0)
            self.assertEqual(draft["v4"]["budget"]["spent"]["tasks"], 0)
            snapshot = store.load(agent.request_id)
            snapshot["todo_items"][0]["id"] = 9
            store.save(agent.request_id, snapshot, expected_version=snapshot["state_version"])

            def execute(state, tasks):
                for task in tasks:
                    task.status = "completed"

            agent._execute_task_batch_sync = execute
            result = agent.run_resume(agent.request_id)
            self.assertEqual([task.id for task in result.todo_items], [9])
            self.assertEqual(result.v4["budget"]["spent"]["tasks"], 1)
            saved = store.load(agent.request_id)
            self.assertEqual(saved["v4"]["replan"]["reason_code"], "plan_locked")
            self.assertEqual(agent.run_resume(agent.request_id).v4["budget"]["spent"]["tasks"], 1)

    def test_run_applies_task_budget_limit(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "max_agent_tasks": 2,
                "task_react_enabled": False,
                "report_repair_enabled": False,
            },
            load_env_file=False,
        )
        agent.planner = SimpleNamespace(
            plan_todo_list=lambda state, observer=None, historical_memory_context=None, strategy_memory_context=None, skill_pack_context=None: [
                TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent background"),
                TodoItem(id=2, title="任务2", intent="梳理挑战", query="AI agent challenges"),
                TodoItem(id=3, title="任务3", intent="梳理案例", query="AI agent case studies"),
            ],
            create_fallback_task=lambda state: TodoItem(
                id=99,
                title="兜底任务",
                intent="收集背景",
                query=state.research_topic,
            ),
        )

        calls = []

        def fake_execute_task(state, task, emit_stream=False, step=None):
            calls.append(task.id)
            task.status = "completed"
            task.summary = "summary"
            task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        agent._execute_task = fake_execute_task

        result = agent.run("AI agent")

        self.assertEqual(calls, [1, 2])
        self.assertEqual([task.id for task in result.todo_items], [1, 2])
        self.assertTrue(any("任务数超过预算" in notice for notice in result.todo_items[0].notices))

    def test_run_passes_historical_memory_context_into_planner(self):
        agent = self._build_agent()
        captured = {}
        agent._note_memory = SimpleNamespace(
            search_for_planning=lambda *args, **kwargs: "历史研究记忆：MCP protocol",
            search_for_task=lambda *args, **kwargs: "",
            refresh_notes=lambda *args, **kwargs: None,
        )
        agent._strategy_memory = SimpleNamespace(
            search_for_planning=lambda *args, **kwargs: "历史策略记忆：优先官方文档与失败反模式。",
            search_for_reflection=lambda *args, **kwargs: "",
            refresh_request=lambda *args, **kwargs: None,
        )

        def fake_plan_todo_list(
            state,
            observer=None,
            historical_memory_context=None,
            strategy_memory_context=None,
            skill_pack_context=None,
        ):
            captured["historical_memory_context"] = historical_memory_context
            captured["strategy_memory_context"] = strategy_memory_context
            return [TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")]

        agent.planner = SimpleNamespace(
            plan_todo_list=fake_plan_todo_list,
            plan_additional_tasks=lambda state, **kwargs: [],
            create_fallback_task=lambda state: TodoItem(
                id=1,
                title="兜底任务",
                intent="收集背景",
                query=state.research_topic,
            ),
        )

        def fake_execute_task(state, task, emit_stream=False, step=None):
            task.status = "completed"
            task.summary = "summary"
            task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        agent._execute_task = fake_execute_task

        agent.run("AI agent")

        self.assertEqual(captured["historical_memory_context"], "历史研究记忆：MCP protocol")
        self.assertEqual(captured["strategy_memory_context"], "历史策略记忆：优先官方文档与失败反模式。")

    def test_run_warms_topic_cache_before_executing_tasks(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_reflection_enabled": False,
                "review_stage_enabled": False,
                "request_state_enabled": False,
                "task_react_enabled": False,
                "report_repair_enabled": False,
                "search_cache_enabled": True,
                "semantic_cache_enabled": True,
            },
            load_env_file=False,
        )
        topic = "探索多模态大模型在 2025 年的关键突破"
        dispatch_calls = []

        def fake_dispatch_search(query, config, loop_count, observer=None, cache_context=None):
            dispatch_calls.append((query, cache_context))
            if observer is not None:
                observer.record_search_attempt(
                    cache_hit=False,
                    success=True,
                    cache_strategy="miss",
                )
            return (
                {
                    "results": [
                        {
                            "title": "Topic Overview",
                            "url": "https://example.com/topic",
                            "content": "content",
                        }
                    ]
                },
                [],
                None,
                "advanced[searxng]",
                False,
                "miss",
            )

        def fake_execute_task(state, task, emit_stream=False, step=None):
            self.assertTrue(state.topic_cache_warmup_completed)
            self.assertEqual(dispatch_calls[0][0], agent._topic_canonical_query(state))
            self.assertEqual(dispatch_calls[0][1], {"research_topic": topic})
            task.status = "completed"
            task.summary = "summary"
            task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        agent._execute_task = fake_execute_task

        with patch.object(task_executor_module, "dispatch_search", side_effect=fake_dispatch_search) as mock_dispatch:
            result = agent.run(topic)

        self.assertEqual(mock_dispatch.call_count, 1)
        self.assertEqual(dispatch_calls[0][0], topic)
        self.assertEqual(result.todo_items[0].status, "completed")

    def test_run_passes_strategy_memory_context_into_reflection(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_reflection_enabled": True,
                "task_react_enabled": False,
                "report_repair_enabled": False,
            },
            load_env_file=False,
        )
        captured = {}
        agent._strategy_memory = SimpleNamespace(
            search_for_planning=lambda *args, **kwargs: "",
            search_for_reflection=lambda *args, **kwargs: "历史策略记忆：部署失败时优先补官方 deployment 文档。",
            refresh_request=lambda *args, **kwargs: None,
        )

        def fake_execute_task(state, task, emit_stream=False, step=None):
            task.status = "completed"
            task.summary = "暂无可用信息"
            task.sources_summary = ""
            if False:
                yield {}

        agent._execute_task = fake_execute_task
        agent.reflection = SimpleNamespace(
            assess_request=lambda state, gap_signals, strategy_memory_context=None, skill_pack_context=None, observer=None: (
                captured.setdefault("strategy_memory_context", strategy_memory_context),
                ReflectionAssessment(
                    coverage_status="sufficient",
                    reason="已捕获策略记忆上下文。",
                    gap_signals=gap_signals,
                    missing_angles=[],
                ),
            )[1]
        )

        agent.run("AI agent")

        self.assertEqual(
            captured["strategy_memory_context"],
            "历史策略记忆：部署失败时优先补官方 deployment 文档。",
        )

    def test_run_refreshes_strategy_memory_after_completion(self):
        agent = self._build_agent()
        refreshed = {}
        agent._strategy_memory = SimpleNamespace(
            search_for_planning=lambda *args, **kwargs: "",
            search_for_reflection=lambda *args, **kwargs: "",
            refresh_request=lambda request_id, observer=None: refreshed.setdefault("request_id", request_id),
        )

        def fake_execute_task(state, task, emit_stream=False, step=None):
            task.status = "completed"
            task.summary = "summary"
            task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        agent._execute_task = fake_execute_task

        agent.run("AI agent")

        self.assertEqual(refreshed["request_id"], "req-test")

    def test_reflection_agent_uses_content_only_llm_and_disables_tools(self):
        config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_state_enabled": False,
                "use_tool_calling": True,
            },
            load_env_file=False,
        )

        agent = agent_module.DeepResearchAgent(config=config, request_id="req-init")

        self.assertIs(agent.reflection_agent.kwargs["llm"], agent._content_only_llm)
        self.assertFalse(agent.reflection_agent.kwargs["enable_tool_calling"])

    def test_vllm_provider_disables_auto_tool_calling_for_agents(self):
        config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_state_enabled": False,
                "llm_provider": "vllm",
                "use_tool_calling": True,
            },
            load_env_file=False,
        )

        agent = agent_module.DeepResearchAgent(config=config, request_id="req-vllm")

        self.assertFalse(agent.todo_agent.kwargs["enable_tool_calling"])
        self.assertFalse(agent.report_agent.kwargs["enable_tool_calling"])
        self.assertFalse(agent.review_agent.kwargs["enable_tool_calling"])
        self.assertFalse(agent.task_react_agent.kwargs["enable_tool_calling"])

    def test_safe_llm_bypasses_env_proxy_for_private_base_urls(self):
        self.assertTrue(
            agent_module.SafeHelloAgentsLLM._should_bypass_env_proxy("http://192.168.1.136:8081/v1")
        )
        self.assertTrue(
            agent_module.SafeHelloAgentsLLM._should_bypass_env_proxy("http://127.0.0.1:8000/v1")
        )
        self.assertTrue(
            agent_module.SafeHelloAgentsLLM._should_bypass_env_proxy("http://localhost:8000/v1")
        )
        self.assertFalse(
            agent_module.SafeHelloAgentsLLM._should_bypass_env_proxy("https://api.openai.com/v1")
        )

    def test_execute_task_marks_search_failures_without_raising(self):
        agent = self._build_agent()
        observer = RequestTrace(
            request_id="req-search",
            topic="AI agent",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        agent._request_trace = observer

        state = SummaryState(research_topic="AI agent")
        task = TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")

        with patch.object(task_executor_module, "dispatch_search", side_effect=RuntimeError("search offline")):
            events = list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    task,
                    emit_stream=False,
                )
            )

        self.assertEqual(events, [])
        self.assertEqual(task.status, "failed")
        self.assertIn("搜索失败", task.notices[0])
        self.assertEqual(observer.snapshot()["failed_tasks"], 1)

    def test_execute_task_marks_summary_failures_without_raising(self):
        agent = self._build_agent()
        observer = RequestTrace(
            request_id="req-summary",
            topic="AI agent",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        agent._request_trace = observer

        def failing_summary(state, task, context, observer=None, historical_memory_context=None):
            raise RuntimeError("summary offline")

        agent.summarizer = SimpleNamespace(summarize_task=failing_summary)

        state = SummaryState(research_topic="AI agent")
        task = TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")

        with patch.object(
            task_executor_module,
            "dispatch_search",
            return_value=(
                {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "content": "content",
                        }
                    ]
                },
                [],
                None,
                "duckduckgo",
                False,
                "miss",
            ),
        ):
            events = list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    task,
                    emit_stream=False,
                )
            )

        self.assertEqual(events, [])
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.summary, "总结阶段失败，请参考已收集来源。")
        self.assertTrue(task.sources_summary)
        self.assertEqual(observer.snapshot()["failed_tasks"], 1)

    def test_drain_tool_events_refreshes_note_memory(self):
        agent = self._build_agent()
        refreshed = {}

        class TrackerWithNote:
            def drain(self, state, *, step=None):
                return [{"type": "tool_call", "note_id": "note_123"}]

            def as_dicts(self):
                return []

            def set_event_sink(self, sink):
                self.sink = sink

        agent._tool_tracker = TrackerWithNote()
        agent._note_memory = SimpleNamespace(
            search_for_planning=lambda *args, **kwargs: "",
            search_for_task=lambda *args, **kwargs: "",
            refresh_notes=lambda note_ids, observer=None: refreshed.setdefault("note_ids", list(note_ids)),
        )

        events = agent._drain_tool_events(SummaryState(research_topic="AI agent"))

        self.assertEqual(events[0]["note_id"], "note_123")
        self.assertEqual(refreshed["note_ids"], ["note_123"])

    def test_execute_task_retries_search_tool_failures_before_recovering(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "search_tool_retry_attempts": 1,
                "search_tool_retry_backoff_seconds": 0.0,
                "task_react_enabled": False,
                "report_repair_enabled": False,
            },
            load_env_file=False,
        )
        observer = RequestTrace(
            request_id="req-search-retry",
            topic="AI agent",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        agent._request_trace = observer

        state = SummaryState(research_topic="AI agent")
        task = TodoItem(
            id=1,
            title="任务1",
            intent="梳理背景",
            query="AI agent system design",
        )

        responses = iter(
            [
                RuntimeError("provider offline"),
                (
                    {
                        "results": [
                            {
                                "title": "Example",
                                "url": "https://example.com",
                                "content": "content",
                            }
                        ]
                    },
                    [],
                    None,
                    "duckduckgo",
                    False,
                    "miss",
                ),
            ]
        )

        def flaky_dispatch(*args, **kwargs):
            outcome = next(responses)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.object(task_executor_module, "dispatch_search", side_effect=flaky_dispatch) as mock_dispatch:
            events = list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    task,
                    emit_stream=False,
                )
            )

        self.assertEqual(events, [])
        self.assertEqual(task.status, "completed")
        self.assertTrue(any("搜索工具调用失败，准备第 1 次重试" in notice for notice in task.notices))
        self.assertEqual(mock_dispatch.call_count, 2)
        self.assertEqual(observer.snapshot()["completed_tasks"], 1)

    def test_execute_task_marks_search_timeout_after_retry_budget_exhausted(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "task_query_rewrite_enabled": False,
                "search_tool_timeout_seconds": 0.01,
                "search_tool_retry_attempts": 1,
                "search_tool_retry_backoff_seconds": 0.0,
                "task_react_enabled": False,
                "report_repair_enabled": False,
            },
            load_env_file=False,
        )
        observer = RequestTrace(
            request_id="req-search-timeout",
            topic="",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        agent._request_trace = observer

        state = SummaryState(research_topic="")
        task = TodoItem(
            id=1,
            title="任务1",
            intent="梳理背景",
            query="AI agent system design architecture",
        )

        def slow_dispatch(*args, **kwargs):
            time.sleep(0.05)
            return (
                {"results": []},
                [],
                None,
                "duckduckgo",
                False,
                "miss",
            )

        with patch.object(task_executor_module, "dispatch_search", side_effect=slow_dispatch) as mock_dispatch:
            events = list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    task,
                    emit_stream=False,
                )
            )

        self.assertEqual(events, [])
        self.assertEqual(task.status, "failed")
        self.assertTrue(any("超时" in notice for notice in task.notices))
        self.assertEqual(mock_dispatch.call_count, 2)
        self.assertEqual(observer.snapshot()["failed_tasks"], 1)

    def test_normalize_query_candidate_strips_request_style_prefixes(self):
        agent = self._build_agent()

        self.assertEqual(
            agent._normalize_query_candidate("请简要研究 vLLM PagedAttention 论文 原理"),
            "vLLM PagedAttention 论文 原理",
        )
        self.assertEqual(
            agent._normalize_query_candidate("请分析：Transformer 推理优化"),
            "Transformer 推理优化",
        )
        self.assertEqual(
            agent._normalize_query_candidate("briefly research TensorRT-LLM deployment"),
            "TensorRT-LLM deployment",
        )
        self.assertEqual(
            agent._normalize_query_candidate("研究方法 对比"),
            "研究方法 对比",
        )
        self.assertEqual(
            agent._normalize_query_candidate(
                "[TOOL_CALL:note:{\"note_id\":\"note_123\"}] 按任务顺序执行 search_web 并更新笔记状态，query: MCP protocol architecture"
            ),
            "MCP protocol architecture",
        )

    def test_task_search_queries_clean_request_style_original_query(self):
        agent = self._build_agent()
        state = SummaryState(research_topic="探索大模型推理服务的关键优化")
        task = TodoItem(
            id=1,
            title="PagedAttention 原理",
            intent="说明其在推理服务中的作用",
            query="请简要研究 vLLM PagedAttention 论文与其在推理服务中的作用 PagedAttention 原理",
        )

        candidates = agent._task_search_queries(state, task)

        self.assertTrue(candidates)
        self.assertEqual(
            candidates[0],
            ("探索大模型推理服务的关键优化 PagedAttention 原理", "original"),
        )
        self.assertTrue(all("请简要研究" not in query for query, _ in candidates))
        self.assertTrue(all("note_" not in query for query, _ in candidates))

    def test_execute_task_retries_with_broader_query_before_skipping(self):
        agent = self._build_agent()
        observer = RequestTrace(
            request_id="req-retry",
            topic="探索多模态大模型在2025年的关键进展",
            search_api="searxng",
            provider="custom",
            model="Qwen/Qwen3.5-27B",
            pricing_catalog={},
        )
        agent._request_trace = observer

        state = SummaryState(research_topic="探索多模态大模型在2025年的关键进展")
        task = TodoItem(id=2, title="性能基准对比", intent="评估主流模型能力水平与资源消耗", query="性能基准对比")

        with patch.object(
            task_executor_module,
            "dispatch_search",
            side_effect=[
                ({"results": []}, [], None, "searxng", False, "miss"),
                (
                    {
                        "results": [
                            {
                                "title": "Benchmark",
                                "url": "https://example.com/bench",
                                "content": "content",
                            }
                        ]
                    },
                    [],
                    None,
                    "searxng",
                    False,
                    "miss",
                ),
            ],
        ) as mock_dispatch:
            events = list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    task,
                    emit_stream=False,
                )
            )

        self.assertEqual(events, [])
        self.assertEqual(task.status, "completed")
        self.assertEqual(
            task.query,
            "探索多模态大模型在2025年的关键进展 性能基准对比 评估主流模型能力水平与资源消耗",
        )
        self.assertTrue(any("重写检索词" in notice for notice in task.notices))
        self.assertEqual(mock_dispatch.call_count, 2)
        self.assertEqual(observer.snapshot()["completed_tasks"], 1)

    def test_execute_task_records_semantic_cache_hits_on_second_request(self):
        agent = self._build_agent()
        state = SummaryState(research_topic="多模态大模型前沿技术")
        first_task = TodoItem(
            id=1,
            title="架构创新与模型设计",
            intent="梳理多模态大模型的核心架构演进与融合机制",
            query="多模态大模型 架构 视觉语言融合 最新研究 2024 2025",
        )
        second_task = TodoItem(
            id=1,
            title="架构创新调研",
            intent="梳理核心架构设计、跨模态融合与注意力机制创新",
            query="多模态大模型 架构设计 2024 2025 跨模态融合 注意力机制",
        )

        dispatch_results = iter(
            [
                (
                    {
                        "results": [
                            {
                                "title": "Architecture",
                                "url": "https://example.com/arch",
                                "content": "content",
                            }
                        ]
                    },
                    [],
                    None,
                    "advanced[searxng, tavily]",
                    False,
                    "miss",
                ),
                (
                    {
                        "results": [
                            {
                                "title": "Architecture",
                                "url": "https://example.com/arch",
                                "content": "content",
                            }
                        ]
                    },
                    [],
                    None,
                    "advanced[searxng, tavily]",
                    True,
                    "semantic",
                ),
            ]
        )

        def fake_dispatch_search(query, config, loop_count, observer=None, cache_context=None):
            result = next(dispatch_results)
            if observer is not None:
                observer.record_search_attempt(
                    cache_hit=result[4],
                    success=True,
                    cache_strategy=result[5],
                )
            return result

        with patch.object(task_executor_module, "dispatch_search", side_effect=fake_dispatch_search):
            first_observer = RequestTrace(
                request_id="req-semantic-first",
                topic="多模态大模型前沿技术",
                search_api="advanced",
                provider="custom",
                model="Qwen/Qwen3.5-27B",
                pricing_catalog={},
            )
            agent._request_trace = first_observer
            list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    first_task,
                    emit_stream=False,
                )
            )

            second_observer = RequestTrace(
                request_id="req-semantic-second",
                topic="多模态大模型前沿技术",
                search_api="advanced",
                provider="custom",
                model="Qwen/Qwen3.5-27B",
                pricing_catalog={},
            )
            agent._request_trace = second_observer
            list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    second_task,
                    emit_stream=False,
                )
            )

        self.assertEqual(first_observer.snapshot()["cache_semantic_hits"], 0)
        self.assertEqual(second_observer.snapshot()["cache_semantic_hits"], 1)
        search_stages = [
            stage
            for stage in second_observer.snapshot()["stages"]
            if stage.get("stage") == "search"
        ]
        self.assertTrue(search_stages)
        self.assertEqual(search_stages[-1]["metadata"]["cache_strategy"], "semantic")

    def test_execute_task_batch_sync_warms_topic_cache_only_once(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_reflection_enabled": False,
                "review_stage_enabled": False,
                "request_state_enabled": False,
                "task_react_enabled": False,
                "report_repair_enabled": False,
                "search_cache_enabled": True,
                "semantic_cache_enabled": True,
            },
            load_env_file=False,
        )
        observer = RequestTrace(
            request_id="req-topic-warmup",
            topic="探索多模态大模型在 2025 年的关键突破",
            search_api="advanced",
            provider="custom",
            model="Qwen/Qwen3.5-27B",
            pricing_catalog={},
        )
        agent._request_trace = observer
        state = SummaryState(research_topic="探索多模态大模型在 2025 年的关键突破")
        first_task = TodoItem(id=1, title="任务1", intent="梳理背景", query="任务1")
        second_task = TodoItem(id=2, title="任务2", intent="梳理进展", query="任务2")

        def fake_execute_task(state, task, emit_stream=False, step=None):
            task.status = "completed"
            task.summary = "summary"
            task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        agent._execute_task = fake_execute_task

        def fake_dispatch_search(query, config, loop_count, observer=None, cache_context=None):
            if observer is not None:
                observer.record_search_attempt(
                    cache_hit=True,
                    success=True,
                    cache_strategy="exact",
                )
            return (
                {"results": [{"title": "Topic", "url": "https://example.com", "content": "content"}]},
                [],
                None,
                "advanced[searxng]",
                True,
                "exact",
            )

        with patch.object(task_executor_module, "dispatch_search", side_effect=fake_dispatch_search) as mock_dispatch:
            agent._execute_task_batch_sync(state, [first_task])
            agent._execute_task_batch_sync(state, [second_task])

        self.assertTrue(state.topic_cache_warmup_completed)
        self.assertEqual(mock_dispatch.call_count, 1)

    def test_execute_task_react_stops_after_sufficient_evidence_in_round_one(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "task_react_enabled": True,
                "task_react_max_rounds": 2,
                "review_min_sources_per_task": 2,
                "review_min_domains_per_task": 2,
            },
            load_env_file=False,
        )
        observer = RequestTrace(
            request_id="req-react-stop",
            topic="AI agent",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        agent._request_trace = observer

        state = SummaryState(research_topic="AI agent")
        task = TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")

        with patch.object(
            task_executor_module,
            "dispatch_search",
            return_value=(
                {
                    "results": [
                        {
                            "title": "Official Guide",
                            "url": "https://docs.example.com/guide",
                            "content": "detailed content",
                        },
                        {
                            "title": "News Coverage",
                            "url": "https://news.example.org/story",
                            "content": "recent coverage",
                        },
                    ]
                },
                [],
                None,
                "duckduckgo",
                False,
                "miss",
            ),
        ) as mock_dispatch:
            events = list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    task,
                    emit_stream=False,
                )
            )

        self.assertEqual(events, [])
        self.assertEqual(mock_dispatch.call_count, 1)
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.react_rounds, 1)
        self.assertEqual(task.react_stop_reason, "evidence_sufficient")
        self.assertEqual(observer.snapshot()["task_react_rounds"], 1)
        self.assertEqual(observer.snapshot()["task_react_stop_count"], 1)

    def test_execute_task_react_diversifies_sources_when_evidence_is_single_sourced(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "task_react_enabled": True,
                "task_react_max_rounds": 2,
                "task_react_max_additional_searches_per_task": 1,
                "review_min_sources_per_task": 2,
                "review_min_domains_per_task": 2,
            },
            load_env_file=False,
        )
        agent.task_react_agent = SimpleNamespace(
            run=lambda prompt: (
                '{"action":"diversify_source_query","query":"AI agent official documentation report","reason":"补充权威来源"}'
            ),
            clear_history=lambda: None,
        )
        observer = RequestTrace(
            request_id="req-react-diversify",
            topic="AI agent",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        agent._request_trace = observer

        state = SummaryState(research_topic="AI agent")
        task = TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")

        with patch.object(
            task_executor_module,
            "dispatch_search",
            side_effect=[
                (
                    {
                        "results": [
                            {
                                "title": "Community Blog",
                                "url": "https://blog.example.com/post",
                                "content": "blog content",
                            }
                        ]
                    },
                    [],
                    None,
                    "duckduckgo",
                    False,
                    "miss",
                ),
                (
                    {
                        "results": [
                            {
                                "title": "Official Docs",
                                "url": "https://docs.example.org/official",
                                "content": "official content",
                            }
                        ]
                    },
                    [],
                    None,
                    "duckduckgo",
                    False,
                    "miss",
                ),
            ],
        ) as mock_dispatch:
            list(
                agent_module.DeepResearchAgent._execute_task(
                    agent,
                    state,
                    task,
                    emit_stream=False,
                )
            )

        self.assertEqual(mock_dispatch.call_count, 2)
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.react_rounds, 2)
        self.assertEqual(task.react_last_action, "diversify_source_query")
        self.assertEqual(task.react_additional_search_count, 1)
        self.assertIn("补充多样化来源检索", "\n".join(task.notices))
        self.assertEqual(observer.snapshot()["task_react_rounds"], 2)
        self.assertEqual(observer.snapshot()["task_react_continue_count"], 1)

    def test_run_report_repair_adds_targeted_tasks_and_reruns_review_once(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "task_react_enabled": False,
                "report_repair_enabled": True,
                "report_repair_max_tasks": 2,
                "report_repair_max_cycles": 1,
                "review_stage_enabled": True,
            },
            load_env_file=False,
        )

        execution_calls = []

        def fake_execute_task(state, task, emit_stream=False, step=None):
            execution_calls.append(task.id)
            task.status = "completed"
            task.summary = "summary"
            task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        review_calls = []

        def fake_review(state, observer):
            review_calls.append(len(state.todo_items))
            if len(review_calls) == 1:
                summary = {
                    "overall_status": "warning",
                    "reason": "missing angle",
                    "issue_count": 1,
                    "severity_counts": {"high": 1, "medium": 0, "low": 0},
                    "issues": [],
                    "repair_candidates": [
                        {
                            "task_id": 1,
                            "severity": "high",
                            "check": "missing_angle",
                            "message": "缺少工程落地维度",
                            "source_ids": [],
                        }
                    ],
                }
            else:
                summary = {
                    "overall_status": "passed",
                    "reason": "fixed",
                    "issue_count": 0,
                    "severity_counts": {"high": 0, "medium": 0, "low": 0},
                    "issues": [],
                    "repair_candidates": [],
                }
            state.review_summary = summary
            state.review_completed = True
            return summary

        agent._execute_task = fake_execute_task
        agent._run_review_stage = fake_review
        agent.planner = SimpleNamespace(
            plan_todo_list=lambda state, observer=None, historical_memory_context=None, strategy_memory_context=None, skill_pack_context=None: [
                TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")
            ],
            plan_additional_tasks=lambda state, **kwargs: [],
            plan_repair_tasks=lambda state, **kwargs: [
                TodoItem(
                    id=2,
                    title="工程落地",
                    intent="补充部署与监控维度",
                    query="AI agent deployment monitoring",
                    origin="repair",
                    round=2,
                )
            ],
            create_fallback_task=lambda state: TodoItem(
                id=1,
                title="兜底任务",
                intent="收集背景",
                query=state.research_topic,
            ),
        )

        result = agent.run("AI agent")

        self.assertEqual(execution_calls, [1, 2])
        self.assertEqual(review_calls, [1, 2])
        self.assertEqual(len(result.todo_items), 2)
        self.assertEqual(result.todo_items[1].origin, "repair")
        self.assertTrue(agent._request_trace.snapshot()["report_repair_triggered"])
        self.assertEqual(agent._request_trace.snapshot()["report_repair_added_tasks"], 1)

    def test_run_triggers_reflection_and_executes_additional_tasks(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_reflection_enabled": True,
                "max_agent_tasks": 3,
                "reflection_max_additional_tasks": 2,
                "task_react_enabled": False,
                "report_repair_enabled": False,
            },
            load_env_file=False,
        )
        calls = []

        def fake_execute_task(state, task, emit_stream=False, step=None):
            calls.append(task.id)
            task.status = "completed"
            if task.id == 1:
                task.summary = "暂无可用信息"
                task.sources_summary = ""
            else:
                task.summary = "supplemental summary"
                task.sources_summary = "* Example : https://example.com/replan"
            if False:
                yield {}

        agent._execute_task = fake_execute_task
        agent.reflection = SimpleNamespace(
            assess_request=lambda state, gap_signals, strategy_memory_context=None, skill_pack_context=None, observer=None: ReflectionAssessment(
                coverage_status="needs_more_research",
                reason="首轮研究仍缺少工程实践维度。",
                gap_signals=gap_signals,
                missing_angles=["工程实践与部署经验"],
            )
        )
        agent.planner = SimpleNamespace(
            plan_todo_list=lambda state, observer=None, historical_memory_context=None, strategy_memory_context=None, skill_pack_context=None: [
                TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")
            ],
            plan_additional_tasks=lambda state, **kwargs: [
                TodoItem(
                    id=2,
                    title="工程实践",
                    intent="补充部署与监控维度",
                    query="AI agent deployment monitoring",
                    origin="replanned",
                    round=2,
                )
            ],
            create_fallback_task=lambda state: TodoItem(
                id=1,
                title="兜底任务",
                intent="收集背景",
                query=state.research_topic,
            ),
        )

        result = agent.run("AI agent")

        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(result.todo_items), 2)
        self.assertEqual(result.todo_items[1].origin, "replanned")
        self.assertEqual(result.todo_items[1].round, 2)
        self.assertTrue(agent._request_trace.snapshot()["reflection_triggered"])
        self.assertEqual(agent._request_trace.snapshot()["reflection_added_tasks"], 1)

    def test_run_skips_reflection_when_task_budget_is_exhausted(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_reflection_enabled": True,
                "max_agent_tasks": 1,
                "task_react_enabled": False,
                "report_repair_enabled": False,
            },
            load_env_file=False,
        )

        def fake_execute_task(state, task, emit_stream=False, step=None):
            task.status = "completed"
            task.summary = "暂无可用信息"
            task.sources_summary = ""
            if False:
                yield {}

        agent._execute_task = fake_execute_task
        agent.reflection = SimpleNamespace(
            assess_request=lambda state, gap_signals, strategy_memory_context=None, skill_pack_context=None, observer=None: (_ for _ in ()).throw(
                AssertionError("reflection should not run when budget is exhausted")
            )
        )

        result = agent.run("AI agent")

        self.assertEqual(len(result.todo_items), 1)
        self.assertTrue(agent._request_trace.snapshot()["reflection_triggered"])
        self.assertEqual(agent._request_trace.snapshot()["reflection_added_tasks"], 0)
        self.assertIn("预算已满", agent._request_trace.snapshot()["reflection_reason"])

    def test_reflection_gap_signals_ignore_generic_degraded_reasons_without_fallback(self):
        agent = self._build_agent()
        observer = RequestTrace(
            request_id="req-reflection-skip",
            topic="AI agent",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        observer.record_degraded("task_react_incomplete:max_rounds_reached")
        agent._request_trace = observer
        state = SummaryState(
            research_topic="AI agent",
            todo_items=[
                TodoItem(
                    id=1,
                    title="任务1",
                    intent="梳理背景",
                    query="AI agent",
                    status="completed",
                    summary="完整总结",
                    sources_summary="* Example : https://example.com",
                )
            ],
        )

        self.assertEqual(agent._reflection_gap_signals(state), [])

    def test_run_stream_emits_reflection_stage_and_updated_todo_list(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_reflection_enabled": True,
                "max_agent_tasks": 3,
                "reflection_max_additional_tasks": 2,
                "task_react_enabled": False,
                "report_repair_enabled": False,
            },
            load_env_file=False,
        )

        def fake_execute_task(state, task, emit_stream=False, step=None):
            task.status = "completed"
            if task.id == 1:
                task.summary = "暂无可用信息"
                task.sources_summary = ""
            else:
                task.summary = "补充任务总结"
                task.sources_summary = "* Example : https://example.com/replan"
            if False:
                yield {}

        agent._execute_task = fake_execute_task
        agent.reflection = SimpleNamespace(
            assess_request=lambda state, gap_signals, strategy_memory_context=None, skill_pack_context=None, observer=None: ReflectionAssessment(
                coverage_status="needs_more_research",
                reason="需要补充落地视角。",
                gap_signals=gap_signals,
                missing_angles=["落地与监控"],
            )
        )
        agent.planner = SimpleNamespace(
            plan_todo_list=lambda state, observer=None, historical_memory_context=None, strategy_memory_context=None, skill_pack_context=None: [
                TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")
            ],
            plan_additional_tasks=lambda state, **kwargs: [
                TodoItem(
                    id=2,
                    title="落地监控",
                    intent="补充部署与监控实践",
                    query="AI agent observability",
                    origin="replanned",
                    round=2,
                )
            ],
            create_fallback_task=lambda state: TodoItem(
                id=1,
                title="兜底任务",
                intent="收集背景",
                query=state.research_topic,
            ),
        )

        events = list(agent.run_stream("AI agent"))

        reflection_started = [
            event for event in events if event.get("type") == "stage_started" and event.get("stage") == "reflection"
        ]
        reflection_completed = [
            event for event in events if event.get("type") == "stage_completed" and event.get("stage") == "reflection"
        ]
        todo_lists = [event for event in events if event.get("type") == "todo_list"]
        metrics_events = [event for event in events if event.get("type") == "metrics_snapshot"]

        self.assertTrue(reflection_started)
        self.assertTrue(reflection_completed)
        self.assertEqual(len(todo_lists), 2)
        self.assertEqual(len(todo_lists[-1]["tasks"]), 2)
        self.assertEqual(metrics_events[-1]["request_metrics"]["reflection_added_tasks"], 1)

    def test_state_from_snapshot_hydrates_evidence_store_and_sources_summary(self):
        agent = self._build_agent()
        payload = {
            "topic": "resume topic",
            "phase": "reporting",
            "todo_items": [
                {
                    "id": 2,
                    "title": "任务2",
                    "intent": "恢复引用",
                    "query": "resume grounding",
                    "status": "completed",
                    "summary": "# 任务总结\n\n## 关键发现\n1. resume 必须恢复 evidence [T2-S2]",
                    "summary_payload": {
                        "key_findings": [
                            {"text": "resume 必须恢复 evidence", "source_ids": ["T2-S2"]}
                        ],
                        "evidence_gaps": [],
                    },
                    "sources_summary": "",
                    "evidence_items": [
                        {
                            "source_id": "T2-S2",
                            "title": "Recovered Resume Source",
                            "url": "https://example.com/resume",
                            "snippet": "resume snapshot evidence",
                            "domain": "example.com",
                            "source_type": "web",
                            "quality_label": "medium",
                            "freshness_label": "recent",
                        }
                    ],
                }
            ],
        }

        state, phase = agent._state_from_snapshot(payload)
        refs = agent._evidence_store.build_reference_map(["T2-S2", "T2-S9"])

        self.assertEqual(phase, "reporting")
        self.assertEqual(state.todo_items[0].evidence_items[0]["source_id"], "T2-S2")
        self.assertIn("[T2-S2]", state.todo_items[0].sources_summary)
        self.assertEqual(refs[0]["title"], "Recovered Resume Source")
        self.assertEqual(refs[0]["url"], "https://example.com/resume")

    def test_three_tier_budget_prevents_reflection_and_repair_starvation(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "max_agent_tasks": 5,
                "request_reflection_enabled": True,
                "reflection_max_additional_tasks": 2,
                "report_repair_enabled": True,
                "report_repair_max_tasks": 2,
                "report_repair_max_cycles": 1,
                "task_react_enabled": False,
            },
            load_env_file=False,
        )
        agent.planner = SimpleNamespace(
            plan_todo_list=lambda state, **kwargs: [
                TodoItem(id=i, title=f"任务{i}", intent=f"维度{i}", query=f"topic angle {i}")
                for i in range(1, 6)
            ],
            plan_additional_tasks=lambda state, **kwargs: [
                TodoItem(
                    id=5,
                    title="反思补充1",
                    intent="补缺1",
                    query="topic reflection 1",
                    origin="replanned",
                    round=2,
                ),
                TodoItem(
                    id=6,
                    title="反思补充2",
                    intent="补缺2",
                    query="topic reflection 2",
                    origin="replanned",
                    round=2,
                ),
            ][: kwargs["max_additional_tasks"]],
            plan_repair_tasks=lambda state, **kwargs: [
                TodoItem(
                    id=7,
                    title="定向修补1",
                    intent="修补证据",
                    query="topic repair 1",
                    origin="repair",
                    round=3,
                )
            ][: kwargs["max_additional_tasks"]],
            create_fallback_task=lambda state: TodoItem(
                id=99, title="兜底", intent="兜底", query=state.research_topic
            ),
        )
        agent.reflection = SimpleNamespace(
            assess_request=lambda state, **kwargs: SimpleNamespace(
                needs_more_research=True,
                coverage_status="insufficient",
                missing_angles=["补缺1", "补缺2"],
                reason="覆盖不足",
                gap_signals=kwargs.get("gap_signals") or [],
            )
        )
        review_calls = {"count": 0}

        def fake_review(state, observer=None, skill_pack_context=None):
            review_calls["count"] += 1
            if review_calls["count"] == 1:
                state.review_summary = {
                    "overall_status": "blocked",
                    "reason": "证据缺口",
                    "issue_count": 1,
                    "severity_counts": {"high": 1, "medium": 0, "low": 0},
                    "issues": [],
                    "repair_candidates": [
                        {
                            "task_id": 1,
                            "severity": "high",
                            "check": "insufficient_sources",
                            "message": "来源不足",
                            "source_ids": [],
                        }
                    ],
                }
            else:
                state.review_summary = {
                    "overall_status": "passed",
                    "reason": "修补完成",
                    "issue_count": 0,
                    "severity_counts": {"high": 0, "medium": 0, "low": 0},
                    "issues": [],
                    "repair_candidates": [],
                }
            state.review_completed = True
            return state.review_summary

        agent.reviewer = SimpleNamespace(review_request=fake_review)

        def fake_execute_task(state, task, emit_stream=False, step=None):
            if task.id == 1:
                task.status = "failed"
            else:
                task.status = "completed"
                task.summary = "有引用总结 [T1-S1]"
                task.sources_summary = "* Example : https://example.com"
            if False:
                yield {}

        agent._execute_task = fake_execute_task

        result = agent.run("AI agent")
        snapshot = agent._request_trace.snapshot()

        # Initial 5 tasks trimmed to 4 -> Reflection adds 2 (total 6) -> Repair still gets 1 (total 7)
        self.assertEqual(len(result.todo_items), 7)
        self.assertEqual(snapshot["reflection_added_tasks"], 2)
        self.assertTrue(snapshot["report_repair_triggered"])
        self.assertEqual(snapshot["report_repair_added_tasks"], 1)
        repair_spans = [
            span for span in snapshot["stages"] if span.get("stage") == "repair"
        ]
        self.assertEqual(len(repair_spans), 1)

    def test_reflection_gap_signals_detect_low_quality_single_domain_and_stale_evidence(self):
        agent = self._build_agent()
        agent._request_trace = RequestTrace(
            request_id="req-quality-gaps",
            topic="2026 最新多模态模型进展",
            search_api="duckduckgo",
            provider="ollama",
            model="llama3.2",
            pricing_catalog={},
        )
        state = SummaryState(
            research_topic="2026 最新多模态模型进展",
            todo_items=[
                TodoItem(
                    id=1,
                    title="任务1",
                    intent="进展",
                    query="2026 最新多模态",
                    status="completed",
                    summary="总结1",
                    sources_summary="* Post1 : https://forum.example.com/1",
                    evidence_items=[
                        {
                            "source_id": "T1-S1",
                            "domain": "forum.example.com",
                            "source_type": "forum_general",
                            "quality_label": "low",
                            "freshness_label": "stale",
                        }
                    ],
                ),
                TodoItem(
                    id=2,
                    title="任务2",
                    intent="局限",
                    query="2026 最新多模态局限",
                    status="completed",
                    summary="总结2",
                    sources_summary="* Post2 : https://forum.example.com/2",
                    evidence_items=[
                        {
                            "source_id": "T2-S1",
                            "domain": "forum.example.com",
                            "source_type": "content_farm_or_aggregator",
                            "quality_label": "low",
                            "freshness_label": "aging",
                        }
                    ],
                ),
            ],
        )

        signals = agent._reflection_gap_signals(state)
        self.assertIn("low_global_domain_diversity", signals)
        self.assertIn("low_evidence_authority", signals)
        self.assertIn("stale_global_evidence", signals)

    def test_safe_llm_captures_stream_usage_and_falls_back_when_stream_options_rejected(self):
        from types import SimpleNamespace

        class FakeCompletions:
            def __init__(self, *, reject_stream_options: bool = False):
                self.reject_stream_options = reject_stream_options
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if self.reject_stream_options and "stream_options" in kwargs:
                    raise RuntimeError("unknown parameter: stream_options")
                return [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="你好", reasoning=None))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[],
                        usage={"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
                    ),
                ]

        llm = agent_module.SafeHelloAgentsLLM.__new__(agent_module.SafeHelloAgentsLLM)
        llm.model = "demo-model"
        llm.provider = "custom"
        llm.temperature = 0.0
        llm.max_tokens = 256
        llm.allow_reasoning_fallback = True
        llm._usage_local = agent_module.local()
        fake = FakeCompletions(reject_stream_options=False)
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        output = llm.invoke([{"role": "user", "content": "hi"}])
        usage = llm.consume_last_usage()

        self.assertEqual(output, "你好")
        self.assertEqual(fake.calls[0].get("stream_options"), {"include_usage": True})
        self.assertEqual(usage, {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165})
        self.assertIsNone(llm.consume_last_usage())

        fake_reject = FakeCompletions(reject_stream_options=True)
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=fake_reject))
        output2 = llm.invoke([{"role": "user", "content": "hi"}])
        self.assertEqual(output2, "你好")
        self.assertEqual(len(fake_reject.calls), 2)
        self.assertNotIn("stream_options", fake_reject.calls[1])

    def test_model_tiering_routes_fast_and_strong_llms_by_stage(self):
        cfg = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "request_state_enabled": False,
                "llm_provider": "custom",
                "llm_model_id": "Qwen/Qwen3-32B",
                "llm_base_url": "http://localhost:8001/v1",
                "fast_llm_model_id": "Qwen/Qwen3-7B-Instruct",
                "fast_llm_base_url": "http://localhost:8002/v1",
            },
            load_env_file=False,
        )
        created_models: list[tuple[str, bool]] = []

        def fake_safe_llm(**kwargs):
            model = str(kwargs.get("model") or "")
            allow_reasoning = bool(kwargs.get("allow_reasoning_fallback", True))
            created_models.append((model, allow_reasoning))
            return type("FakeLLM", (), {"model": model, "allow_reasoning_fallback": allow_reasoning})()

        with patch.object(agent_module, "SafeHelloAgentsLLM", side_effect=fake_safe_llm):
            tiered_agent = agent_module.DeepResearchAgent(config=cfg)

        def _agent_llm_model(sub_agent):
            llm_obj = getattr(sub_agent, "llm", None) or sub_agent.kwargs["llm"]
            return llm_obj.model

        self.assertEqual(_agent_llm_model(tiered_agent.todo_agent), "Qwen/Qwen3-7B-Instruct")
        self.assertEqual(_agent_llm_model(tiered_agent.reflection_agent), "Qwen/Qwen3-7B-Instruct")
        self.assertEqual(_agent_llm_model(tiered_agent.task_react_agent), "Qwen/Qwen3-7B-Instruct")
        self.assertEqual(_agent_llm_model(tiered_agent._summarizer_factory()), "Qwen/Qwen3-7B-Instruct")
        self.assertEqual(_agent_llm_model(tiered_agent.report_agent), "Qwen/Qwen3-32B")
        self.assertEqual(_agent_llm_model(tiered_agent.review_agent), "Qwen/Qwen3-32B")

    def test_high_confidence_early_stop_skips_extra_react_rounds(self):
        agent = self._build_agent()
        agent.config = Configuration.from_env(
            overrides={
                "enable_notes": False,
                "task_react_enabled": True,
                "task_react_max_rounds": 3,
                "review_min_sources_per_task": 2,
                "review_min_domains_per_task": 2,
            },
            load_env_file=False,
        )
        state = SummaryState(research_topic="Python 3.13 特性")
        task = TodoItem(id=1, title="官方文档与论文", intent="收集权威资料", query="Python 3.13")
        evidence_items = [
            {
                "source_id": "T1-S1",
                "domain": "docs.python.org",
                "quality_label": "high",
                "freshness_label": "fresh",
                "snippet": "Python 3.13 release highlights" * 10,
            },
            {
                "source_id": "T1-S2",
                "domain": "arxiv.org",
                "quality_label": "high",
                "freshness_label": "fresh",
                "snippet": "Benchmark paper" * 10,
            },
            {
                "source_id": "T1-S3",
                "domain": "github.com",
                "quality_label": "medium",
                "freshness_label": "fresh",
                "snippet": "Release notes" * 10,
            },
        ]
        observation = agent._observe_task_evidence(state, task, evidence_items)
        self.assertTrue(observation.evidence_sufficiency)
        self.assertEqual(observation.stop_reason, "high_confidence_early_stop")

    def test_stream_coordinator_binds_thread_local_task_id_and_respects_cancel_event(self):
        from threading import Event

        agent = self._build_agent()
        state = SummaryState(research_topic="并发流式测试")
        t1 = TodoItem(id=1, title="任务1", intent="i1", query="q1")
        t2 = TodoItem(id=2, title="任务2", intent="i2", query="q2")
        state.todo_items = [t1, t2]

        agent._start_request_trace(state.research_topic)
        channel_map: dict[int, dict] = {}
        agent._assign_stream_channels(state.todo_items, channel_map, start_step=1)

        sink_holder = {"sink": None}
        agent._set_tool_event_sink = lambda sink: sink_holder.__setitem__("sink", sink)
        agent._warm_topic_search_cache = lambda s, observer=None: None

        def fake_exec(s, task, emit_stream=True, step=None):
            # Simulate a tool event emitted from worker thread WITHOUT explicit task_id
            if sink_holder["sink"] is not None:
                sink_holder["sink"]({"type": "tool_call", "tool": "search_web"})
            task.status = "completed"
            yield {"type": "task_status", "task_id": task.id, "status": "completed"}

        agent._execute_task = fake_exec
        events = list(agent._execute_task_batch_stream(state, [t1, t2], channel_map))
        tool_events = [ev for ev in events if ev.get("type") == "tool_call"]
        self.assertEqual(len(tool_events), 2)
        self.assertEqual({ev["task_id"] for ev in tool_events}, {1, 2})
        self.assertEqual({ev["stream_token"] for ev in tool_events}, {"task_1", "task_2"})

        # Pre-set cancel_event short-circuits workflow before planning/reporting
        cancel_ev = Event()
        cancel_ev.set()
        report_called = {"flag": False}
        agent.reporting = type(
            "StubReporting",
            (),
            {"generate_report": staticmethod(lambda *a, **kw: (report_called.__setitem__("flag", True) or ""))},
        )()
        cancelled_events = list(agent.run_stream("取消测试", cancel_event=cancel_ev))
        self.assertFalse(report_called["flag"])
        self.assertFalse(any(ev.get("type") == "final_report" for ev in cancelled_events))
        self.assertIn("client_disconnected", agent._request_trace.degraded_reasons)


if __name__ == "__main__":
    unittest.main()
    
