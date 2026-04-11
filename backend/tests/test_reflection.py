import sys
import types
import unittest
from pathlib import Path

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
from models import SummaryState, TodoItem
from services.reflection import ReflectionService


class StubReflectionAgent:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.clear_history_calls = 0

    def run(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def clear_history(self):
        self.clear_history_calls += 1


class ReflectionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()
        self.config = Configuration.from_env(overrides={}, load_env_file=False)

    def _build_state(self) -> SummaryState:
        return SummaryState(
            research_topic="AI agent 落地现状",
            research_loop_count=2,
            todo_items=[
                TodoItem(
                    id=1,
                    title="背景梳理",
                    intent="总结当前技术和生态",
                    query="AI agent landscape",
                    status="completed",
                    summary="已覆盖定义、主要框架和典型应用场景。",
                    sources_summary="* Example : https://example.com/overview",
                    notices=["cache_hit"],
                    evidence_items=[
                        {"source_id": "T1-S1", "domain": "example.com"},
                        {"source_id": "T1-S2", "domain": "docs.example.com"},
                    ],
                    origin="planned",
                    round=1,
                ),
                TodoItem(
                    id=2,
                    title="工程落地",
                    intent="评估部署、监控与运维挑战",
                    query="AI agent deployment monitoring",
                    status="failed",
                    summary="暂无可用信息",
                    sources_summary="",
                    evidence_items=[],
                    origin="planned",
                    round=1,
                ),
            ],
        )

    def _build_observer(self) -> RequestTrace:
        return RequestTrace(
            request_id="req-reflection",
            topic="AI agent 落地现状",
            search_api="advanced",
            provider="vllm",
            model="test-model",
            pricing_catalog={},
        )

    def test_build_reflection_context_and_prompt_are_explicit(self):
        service = ReflectionService(StubReflectionAgent([]), self.config)
        state = self._build_state()

        context = service._build_reflection_context(state, gap_signals=["task_2_failed"])
        prompt = service._build_prompt(
            context,
            strategy_memory_context="历史策略记忆：当失败任务暴露部署缺口时，优先补官方部署与监控资料。",
        )

        self.assertEqual(context["task_counts"]["total"], 2)
        self.assertEqual(context["task_snapshots"][0]["evidence_count"], 2)
        self.assertFalse(context["task_snapshots"][0]["summary_missing"])
        self.assertTrue(context["task_snapshots"][1]["sources_missing"])
        self.assertIn("<JSON_CONTEXT>", prompt)
        self.assertIn('"task_snapshots"', prompt)
        self.assertIn("任务统计：共 2 个任务", prompt)
        self.assertIn("工程落地", prompt)
        self.assertIn("must_output_single_json_object", prompt)
        self.assertIn("STRATEGY_MEMORY", prompt)
        self.assertIn("历史策略记忆", prompt)
        self.assertNotIn('"query"', prompt)

    def test_assess_request_accepts_valid_strict_json(self):
        agent = StubReflectionAgent(
            [
                '{"coverage_status":"sufficient","reason":"现有任务已覆盖关键维度。","gap_signals":["task_2_failed"],"missing_angles":[]}'
            ]
        )
        service = ReflectionService(agent, self.config)

        assessment = service.assess_request(
            self._build_state(),
            gap_signals=["task_2_failed"],
            strategy_memory_context="历史策略记忆：部署类失败通常需要补可观测性与官方文档。",
            observer=self._build_observer(),
        )

        self.assertEqual(assessment.coverage_status, "sufficient")
        self.assertEqual(assessment.reason, "现有任务已覆盖关键维度。")
        self.assertEqual(assessment.gap_signals, ["task_2_failed"])
        self.assertEqual(agent.calls[0]["kwargs"]["response_format"], {"type": "json_object"})
        self.assertEqual(agent.clear_history_calls, 1)

    def test_assess_request_retries_without_response_format_when_unsupported(self):
        agent = StubReflectionAgent(
            [
                TypeError("run() got an unexpected keyword argument 'response_format'"),
                '{"coverage_status":"sufficient","reason":"回退后输出有效 JSON。","gap_signals":["task_2_failed"],"missing_angles":[]}',
            ]
        )
        service = ReflectionService(agent, self.config)

        assessment = service.assess_request(
            self._build_state(),
            gap_signals=["task_2_failed"],
        )

        self.assertEqual(assessment.coverage_status, "sufficient")
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.calls[0]["kwargs"]["response_format"], {"type": "json_object"})
        self.assertEqual(agent.calls[1]["kwargs"], {})

    def test_assess_request_rejects_missing_angles_for_needs_more_research(self):
        agent = StubReflectionAgent(
            [
                '{"coverage_status":"needs_more_research","reason":"仍有缺口。","gap_signals":["task_2_failed"],"missing_angles":[]}'
            ]
        )
        observer = self._build_observer()
        service = ReflectionService(agent, self.config)

        assessment = service.assess_request(
            self._build_state(),
            gap_signals=["task_2_failed"],
            observer=observer,
        )

        self.assertEqual(assessment.coverage_status, "sufficient")
        self.assertEqual(assessment.reason, "reflection 输出不符合严格 JSON，已跳过补充研究。")
        self.assertIn("reflection_invalid_output", observer.snapshot()["degraded_reasons"])

    def test_assess_request_rejects_wrapped_or_tool_call_output(self):
        invalid_outputs = (
            '```json\n{"coverage_status":"sufficient","reason":"ok","gap_signals":[],"missing_angles":[]}\n```',
            '结论如下：{"coverage_status":"sufficient","reason":"ok","gap_signals":[],"missing_angles":[]}',
            '{"coverage_status":"sufficient","reason":"ok","gap_signals":[],"missing_angles":[]} [TOOL_CALL:note:{}]',
        )

        for output in invalid_outputs:
            with self.subTest(output=output):
                agent = StubReflectionAgent([output])
                observer = self._build_observer()
                service = ReflectionService(agent, self.config)

                assessment = service.assess_request(
                    self._build_state(),
                    gap_signals=["task_2_failed"],
                    observer=observer,
                )

                self.assertEqual(assessment.coverage_status, "sufficient")
                self.assertEqual(
                    assessment.reason,
                    "reflection 输出不符合严格 JSON，已跳过补充研究。",
                )
                self.assertIn("reflection_invalid_output", observer.snapshot()["degraded_reasons"])


if __name__ == "__main__":
    unittest.main()
