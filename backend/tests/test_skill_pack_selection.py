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


class DummyToolAwareSimpleAgent:
    pass


hello_agents_stub.ToolAwareSimpleAgent = DummyToolAwareSimpleAgent
sys.modules.setdefault("hello_agents", hello_agents_stub)

from config import Configuration
from services.skill_pack import (
    NEUTRAL_PACK_ID,
    SKILL_STAGES,
    SkillPackService,
)

REAL_PACK_DIR = str((BACKEND_SRC.parent / "skills" / "research").resolve())


def _service(dir_path: str, *, min_score: float = 0.20, char_limit: int = 1200) -> SkillPackService:
    config = Configuration.from_env(
        overrides={
            "skill_pack_enabled": True,
            "skill_pack_dir": dir_path,
            "skill_pack_min_match_score": min_score,
            "skill_prompt_char_limit": char_limit,
        },
        load_env_file=False,
    )
    return SkillPackService(config)


class SelectionTests(unittest.TestCase):
    """Selection is conservative: mis-selecting a pack is worse than using the neutral one."""

    def setUp(self):
        self.svc = _service(REAL_PACK_DIR)

    def test_policy_topic_selects_policy_pack(self):
        sel = self.svc.select("某新规的适用范围与生效时间以及合规义务")
        self.assertEqual(sel.pack.pack_id, "policy-analysis")
        self.assertFalse(sel.is_fallback)
        self.assertGreater(sel.match_score, 0.0)

    def test_tech_topic_selects_tech_pack(self):
        sel = self.svc.select("向量数据库技术选型与性能权衡的方案对比")
        self.assertEqual(sel.pack.pack_id, "tech-selection")

    def test_competitor_topic_selects_competitor_pack(self):
        sel = self.svc.select("国内云厂商竞品格局与市场份额")
        self.assertEqual(sel.pack.pack_id, "competitor-scan")

    def test_unrelated_topic_falls_back_to_neutral(self):
        sel = self.svc.select("今天天气怎么样")
        self.assertEqual(sel.pack.pack_id, NEUTRAL_PACK_ID)
        self.assertTrue(sel.is_fallback)
        self.assertEqual(sel.fallback_reason, "no_pack_met_min_match_score")

    def test_negative_signal_disqualifies_a_pack(self):
        # A topic full of policy words but also an explicit tech-selection negative signal
        # must not be routed to the tech pack.
        sel = self.svc.select("政策 法规 合规 架构 组件 框架 性能 权衡 选型 方案对比")
        self.assertNotEqual(sel.pack.pack_id, "tech-selection")

    def test_render_block_carries_disclaimer_and_no_leak(self):
        pack = self.svc.select("某新规的适用范围与生效时间").pack
        block = self.svc.render_block(pack, "planning")
        self.assertIn(f'pack="{pack.pack_id}"', block)
        self.assertIn("不是当前主题事实", block)
        self.assertNotIn("[TOOL_CALL:", block)
        self.assertNotIn("http", block)

    def test_render_block_empty_for_stage_outside_scope(self):
        pack = self.svc.neutral_pack()
        # Neutral covers all five stages, so fabricate a missing stage lookup.
        self.assertEqual(self.svc.render_block(pack, "nonexistent_stage"), "")

    def test_all_five_stages_render(self):
        pack = self.svc.neutral_pack()
        for stage in SKILL_STAGES:
            self.assertTrue(self.svc.render_block(pack, stage).strip())


class ShadowAndFallbackTests(unittest.TestCase):
    def test_shadow_status_pack_is_marked_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_pack(root, "policy-analysis", status="shadow")
            self._copy_pack(root, NEUTRAL_PACK_ID)
            svc = _service(str(root))
            sel = svc.select("某新规的适用范围与生效时间与合规义务")
            self.assertEqual(sel.pack.pack_id, "policy-analysis")
            self.assertTrue(sel.shadow)

    def test_missing_neutral_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_pack(root, "policy-analysis")  # no neutral present
            svc = _service(str(root))
            sel = svc.select("完全无关的主题")
            self.assertIsNone(sel)

    def test_disabled_pack_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_pack(root, "policy-analysis", status="disabled")
            self._copy_pack(root, NEUTRAL_PACK_ID)
            svc = _service(str(root))
            sel = svc.select("某新规的适用范围与生效时间与合规义务")
            self.assertEqual(sel.pack.pack_id, NEUTRAL_PACK_ID)

    def _copy_pack(self, root: Path, pack_id: str, *, status: str | None = None) -> None:
        source = Path(REAL_PACK_DIR) / pack_id / "SKILL.md"
        text = source.read_text(encoding="utf-8")
        if status is not None:
            text = text.replace("status: active", f"status: {status}", 1)
        target_dir = root / pack_id
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "SKILL.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
