import sys
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

    def run(self, prompt: str):
        return prompt

    def stream_run(self, prompt: str):
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
from config import Configuration
from metrics import RequestTrace, metrics_registry
from models import SummaryState, TodoItem


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

    def _build_agent(self):
        instance = agent_module.DeepResearchAgent.__new__(agent_module.DeepResearchAgent)
        instance.config = Configuration.from_env(
            overrides={"enable_notes": False},
            load_env_file=False,
        )
        instance.request_id = "req-test"
        instance.note_tool = None
        instance.tools_registry = None
        instance._tool_tracker = StubTracker()
        instance._tool_event_sink_enabled = False
        instance._state_lock = Lock()
        instance._last_search_notices = []
        instance._request_trace = None
        instance.planner = SimpleNamespace(
            plan_todo_list=lambda state, observer=None: [
                TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")
            ],
            create_fallback_task=lambda state: TodoItem(
                id=1,
                title="兜底任务",
                intent="收集背景",
                query=state.research_topic,
            ),
        )
        instance.reporting = SimpleNamespace(
            generate_report=lambda state, observer=None: "# report"
        )
        instance.summarizer = SimpleNamespace(
            summarize_task=lambda state, task, context, observer=None: "summary"
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

        with patch.object(agent_module, "dispatch_search", side_effect=RuntimeError("search offline")):
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

        def failing_summary(state, task, context, observer=None):
            raise RuntimeError("summary offline")

        agent.summarizer = SimpleNamespace(summarize_task=failing_summary)

        state = SummaryState(research_topic="AI agent")
        task = TodoItem(id=1, title="任务1", intent="梳理背景", query="AI agent")

        with patch.object(
            agent_module,
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
            agent_module,
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
            "探索多模态大模型在2025年的关键进展 性能基准对比",
        )
        self.assertTrue(any("更宽泛检索词" in notice for notice in task.notices))
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

        with patch.object(agent_module, "dispatch_search", side_effect=fake_dispatch_search):
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


if __name__ == "__main__":
    unittest.main()
    
