import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from services.skill_pack import (
    NEUTRAL_PACK_ID,
    SKILL_STAGES,
    SkillPackError,
    compute_content_hash,
    enforce_quality_gate_direction,
    load_pack,
    validate_pack,
)

REAL_PACK_DIR = BACKEND_SRC.parent / "skills" / "research"


def _frontmatter_lines(meta: dict) -> str:
    """Render a minimal-but-valid frontmatter block from a nested dict."""

    import yaml

    return yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()


def _pack_text(meta: dict, sections: dict[str, str]) -> str:
    body = "\n\n".join(f"## {stage}\n{text}" for stage, text in sections.items())
    return f"---\n{_frontmatter_lines(meta)}\n---\n\n{body}\n"


def _valid_meta(**overrides) -> dict:
    meta = {
        "pack_id": "policy-analysis",
        "schema_version": 1,
        "version": 1,
        "content_hash": "",
        "maturity": "draft",
        "status": "active",
        "supersedes": "",
        "title": "政策分析",
        "topic_domain": "policy_analysis",
        "applicable_when": "涉及政策法规时",
        "match_signals": ["政策", "法规"],
        "negative_signals": ["架构"],
        "min_match_score": 0.25,
        "stage_scope": list(SKILL_STAGES),
        "quality_gates": {},
        "budget_knobs": {},
        "layout": {},
        "derived_from": "",
        "eval": {},
        "owner": "team",
    }
    meta.update(overrides)
    return meta


def _valid_sections(stages=SKILL_STAGES) -> dict[str, str]:
    return {stage: f"这是 {stage} 阶段的方法约定，只讲做法与取舍。" for stage in stages}


class _Harness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, meta: dict, sections: dict[str, str], *, dir_name: str | None = None) -> Path:
        directory = self.root / (dir_name or meta["pack_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(_pack_text(meta, sections), encoding="utf-8")
        return directory

    def _errors(self, meta, sections, *, dir_name=None, char_limit=1200):
        directory = self._write(meta, sections, dir_name=dir_name)
        pack = load_pack(directory / "SKILL.md")
        return pack, validate_pack(pack, char_limit=char_limit, directory_name=directory.name)


class RealPacksTests(_Harness):
    """The shipped packs must all be valid and the neutral fallback must exist."""

    def test_all_shipped_packs_validate(self):
        self.assertTrue(REAL_PACK_DIR.is_dir())
        seen = set()
        for directory in sorted(REAL_PACK_DIR.iterdir()):
            if not directory.is_dir():
                continue
            pack = load_pack(directory / "SKILL.md")
            errors = validate_pack(pack, char_limit=1200, directory_name=directory.name)
            self.assertEqual(errors, [], f"{directory.name}: {errors}")
            seen.add(pack.pack_id)
        self.assertIn(NEUTRAL_PACK_ID, seen)


class IdentityRuleTests(_Harness):
    def test_rule01_pack_id_must_match_directory(self):
        _, errors = self._errors(_valid_meta(pack_id="policy-analysis"), _valid_sections(), dir_name="tech-selection")
        self.assertTrue(any("does not match directory" in e for e in errors))

    def test_rule01b_pack_id_must_be_kebab_case(self):
        _, errors = self._errors(_valid_meta(pack_id="Policy_Analysis"), _valid_sections(), dir_name="Policy_Analysis")
        self.assertTrue(any("kebab-case" in e for e in errors))

    def test_rule02_pinned_hash_mismatch_is_rejected(self):
        _, errors = self._errors(_valid_meta(content_hash="deadbeefdeadbeef"), _valid_sections())
        self.assertTrue(any("content_hash does not match" in e for e in errors))

    def test_rule03_matching_pinned_hash_passes(self):
        # Load once to learn the computed hash, then pin it and confirm it validates.
        pack, _ = self._errors(_valid_meta(), _valid_sections())
        meta = _valid_meta(content_hash=pack.content_hash)
        _, errors = self._errors(meta, _valid_sections())
        self.assertEqual(errors, [])

    def test_bad_status_is_rejected(self):
        _, errors = self._errors(_valid_meta(status="published"), _valid_sections())
        self.assertTrue(any("status" in e for e in errors))


class StructureRuleTests(_Harness):
    def test_rule04_stage_scope_must_be_known(self):
        meta = _valid_meta(stage_scope=list(SKILL_STAGES) + ["summarize"])
        sections = _valid_sections()
        sections["summarize"] = "未知阶段"
        _, errors = self._errors(meta, sections)
        self.assertTrue(any("unknown stages" in e for e in errors))

    def test_rule05_body_sections_must_equal_stage_scope(self):
        meta = _valid_meta(stage_scope=list(SKILL_STAGES))
        sections = _valid_sections([s for s in SKILL_STAGES if s != "report"])
        _, errors = self._errors(meta, sections)
        self.assertTrue(any("must equal stage_scope" in e for e in errors))

    def test_rule06_fragment_over_char_limit_is_rejected(self):
        sections = _valid_sections()
        sections["planning"] = "方" * 50
        _, errors = self._errors(_valid_meta(), sections, char_limit=20)
        self.assertTrue(any("char limit" in e for e in errors))


class SafetyRuleTests(_Harness):
    def test_rule07_quality_gate_key_must_be_whitelisted(self):
        _, errors = self._errors(_valid_meta(quality_gates={"max_agent_tasks": 3}), _valid_sections())
        self.assertTrue(any("quality_gates key" in e for e in errors))

    def test_rule07b_quality_gate_direction_enforced(self):
        # increase-direction gate: override must be >= baseline to be a tightening.
        self.assertTrue(enforce_quality_gate_direction("review_min_sources_per_task", 2, 3))
        self.assertFalse(enforce_quality_gate_direction("review_min_sources_per_task", 2, 1))
        # decrease-direction gate: stricter means a smaller freshness window.
        self.assertTrue(enforce_quality_gate_direction("freshness_reference_days", 365, 180))
        self.assertFalse(enforce_quality_gate_direction("freshness_reference_days", 365, 720))

    def test_rule08_budget_and_layout_keys_must_be_whitelisted(self):
        _, errors = self._errors(_valid_meta(budget_knobs={"unknown_knob": 1}), _valid_sections())
        self.assertTrue(any("budget_knobs key" in e for e in errors))
        _, errors = self._errors(_valid_meta(layout={"unknown_layout": "x"}), _valid_sections())
        self.assertTrue(any("layout key" in e for e in errors))

    def test_rule09_fragment_must_not_contain_tool_directive(self):
        sections = _valid_sections()
        sections["planning"] = "先调用 [TOOL_CALL: search_web] 再总结"
        _, errors = self._errors(_valid_meta(), sections)
        self.assertTrue(any("tool directive" in e for e in errors))

    def test_rule10_fragment_must_not_contain_source_id_url_or_number(self):
        sections = _valid_sections()
        sections["planning"] = "见 [T1-S2] 的说法"
        _, errors = self._errors(_valid_meta(), sections)
        self.assertTrue(any("source id" in e for e in errors))

        sections = _valid_sections()
        sections["planning"] = "参考 https://example.com 的数据"
        _, errors = self._errors(_valid_meta(), sections)
        self.assertTrue(any("URL" in e for e in errors))

        sections = _valid_sections()
        sections["planning"] = "市场规模达到 12345 亿元"
        _, errors = self._errors(_valid_meta(), sections)
        self.assertTrue(any("numeric finding" in e for e in errors))


class GovernanceRuleTests(_Harness):
    def test_rule11_match_and_negative_signals_must_not_overlap(self):
        meta = _valid_meta(match_signals=["政策", "架构"], negative_signals=["架构"])
        _, errors = self._errors(meta, _valid_sections())
        self.assertTrue(any("overlap" in e for e in errors))

    def test_rule12_promoted_requires_eval_pass(self):
        _, errors = self._errors(_valid_meta(maturity="promoted", eval={}), _valid_sections())
        self.assertTrue(any("promoted pack requires an eval" in e for e in errors))
        _, errors = self._errors(
            _valid_meta(maturity="promoted", eval={"verdict": "pass"}), _valid_sections()
        )
        self.assertFalse(any("promoted pack requires an eval" in e for e in errors))

    def test_rule13_neutral_must_have_empty_overrides(self):
        meta = _valid_meta(
            pack_id=NEUTRAL_PACK_ID,
            topic_domain="general",
            match_signals=[],
            negative_signals=[],
            min_match_score=0.0,
            quality_gates={"review_min_sources_per_task": 3},
        )
        _, errors = self._errors(meta, _valid_sections(), dir_name=NEUTRAL_PACK_ID)
        self.assertTrue(any("must not override any knobs" in e for e in errors))

    def test_neutral_with_match_signals_is_rejected(self):
        meta = _valid_meta(
            pack_id=NEUTRAL_PACK_ID,
            topic_domain="general",
            match_signals=["政策"],
            negative_signals=[],
            min_match_score=0.0,
        )
        _, errors = self._errors(meta, _valid_sections(), dir_name=NEUTRAL_PACK_ID)
        self.assertTrue(any("must not declare match_signals" in e for e in errors))


class LoaderErrorTests(_Harness):
    def test_missing_frontmatter_raises(self):
        directory = self.root / "broken"
        directory.mkdir()
        (directory / "SKILL.md").write_text("## planning\n无 frontmatter", encoding="utf-8")
        with self.assertRaises(SkillPackError):
            load_pack(directory / "SKILL.md")

    def test_content_hash_is_stable_across_loads(self):
        directory = self._write(_valid_meta(), _valid_sections())
        first = load_pack(directory / "SKILL.md")
        second = load_pack(directory / "SKILL.md")
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.content_hash, compute_content_hash(second))


if __name__ == "__main__":
    unittest.main()
