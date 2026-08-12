import sys
import tempfile
import types
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
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
from metrics import RequestTrace
from services.skill_pack import NEUTRAL_PACK_ID, SkillPackService

REAL_PACK_DIR = str((BACKEND_SRC.parent / "skills" / "research").resolve())


def _trace() -> RequestTrace:
    return RequestTrace(
        request_id="req-skill",
        topic="话题",
        search_api="duckduckgo",
        provider="openai",
        model="gpt-test",
        pricing_catalog=None,
    )


def _agent(config: Configuration) -> agent_module.DeepResearchAgent:
    instance = agent_module.DeepResearchAgent.__new__(agent_module.DeepResearchAgent)
    instance.config = config
    instance._skill_pack = SkillPackService(config)
    instance._skill_selection = None
    return instance


def _config(dir_path: str, *, enabled: bool = True, min_score: float = 0.20) -> Configuration:
    return Configuration.from_env(
        overrides={
            "skill_pack_enabled": enabled,
            "skill_pack_dir": dir_path,
            "skill_pack_min_match_score": min_score,
            "skill_prompt_char_limit": 1200,
        },
        load_env_file=False,
    )


class SelectionWiringTests(unittest.TestCase):
    def test_matching_topic_selects_pack_and_records_trace(self):
        agent = _agent(_config(REAL_PACK_DIR))
        observer = _trace()

        agent._select_skill_pack("某新规的适用范围与生效时间以及合规义务", observer)

        self.assertEqual(agent._skill_selection.pack.pack_id, "policy-analysis")
        self.assertEqual(observer.skill_pack_id, "policy-analysis")
        self.assertEqual(observer.skill_pack_version, "1")
        self.assertGreater(observer.skill_pack_match_score, 0.0)
        self.assertEqual(observer.skill_pack_fallback_reason, "")

    def test_injection_flows_into_stage_context(self):
        agent = _agent(_config(REAL_PACK_DIR))
        agent._select_skill_pack("向量数据库技术选型与性能权衡的方案对比", _trace())

        block = agent._skill_stage_context("planning")

        self.assertIn('<RESEARCH_SKILL pack="tech-selection"', block)
        self.assertIn("不是当前主题事实", block)

    def test_unmatched_topic_falls_back_to_neutral_with_reason(self):
        agent = _agent(_config(REAL_PACK_DIR))
        observer = _trace()

        agent._select_skill_pack("今天天气怎么样", observer)

        self.assertEqual(agent._skill_selection.pack.pack_id, NEUTRAL_PACK_ID)
        self.assertEqual(observer.skill_pack_id, NEUTRAL_PACK_ID)
        self.assertEqual(observer.skill_pack_fallback_reason, "no_pack_met_min_match_score")
        # The neutral pack still injects its generic method fragment.
        self.assertIn("RESEARCH_SKILL", agent._skill_stage_context("planning"))

    def test_disabled_service_selects_nothing_and_injects_nothing(self):
        agent = _agent(_config(REAL_PACK_DIR, enabled=False))
        observer = _trace()

        agent._select_skill_pack("某新规的适用范围与生效时间与合规义务", observer)

        self.assertIsNone(agent._skill_selection)
        self.assertEqual(observer.skill_pack_id, "")
        self.assertEqual(agent._skill_stage_context("planning"), "")


class ShadowAndDegradeTests(unittest.TestCase):
    def _copy_pack(self, root: Path, pack_id: str, *, status: str | None = None) -> None:
        source = Path(REAL_PACK_DIR) / pack_id / "SKILL.md"
        text = source.read_text(encoding="utf-8")
        if status is not None:
            text = text.replace("status: active", f"status: {status}", 1)
        target = root / pack_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(text, encoding="utf-8")

    def test_shadow_pack_is_recorded_but_not_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_pack(root, "policy-analysis", status="shadow")
            self._copy_pack(root, NEUTRAL_PACK_ID)
            agent = _agent(_config(str(root)))
            observer = _trace()

            agent._select_skill_pack("某新规的适用范围与生效时间与合规义务", observer)

            self.assertEqual(observer.skill_pack_id, "policy-analysis")
            self.assertTrue(observer.skill_pack_shadow)
            # Shadow: selection recorded, but nothing is injected downstream.
            self.assertEqual(agent._skill_stage_context("planning"), "")

    def test_missing_neutral_records_unavailable_degrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_pack(root, "policy-analysis")  # no neutral fallback
            agent = _agent(_config(str(root)))
            observer = _trace()

            agent._select_skill_pack("完全无关的主题", observer)

            self.assertIsNone(agent._skill_selection)
            self.assertIn("skill_pack_unavailable", observer.degraded_reasons)


if __name__ == "__main__":
    unittest.main()
