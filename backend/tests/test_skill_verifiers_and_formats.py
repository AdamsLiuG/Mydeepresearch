"""Tests for Skill System v2: retrieval profiles, executable verifiers, format skills, composable & hybrid routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"
for entry in (str(BACKEND_ROOT), str(BACKEND_SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from config import Configuration  # noqa: E402
from evals.judges.heuristic import HeuristicJudge  # noqa: E402
from evals.judges.hybrid import HybridJudge  # noqa: E402
from evals.schema import BenchmarkCase  # noqa: E402
from main import create_app  # noqa: E402
from models import SummaryState, TodoItem  # noqa: E402
from services.evidence import EvidenceStore  # noqa: E402
from services.reporter import ReportingService  # noqa: E402
from services.reviewer import ReviewService  # noqa: E402
from services.search import (  # noqa: E402
    _compute_fusion_rrf_score,
    _fusion_source_prior,
)
from services.skill_pack import (  # noqa: E402
    SkillPackService,
    apply_skill_pack_overrides,
    load_pack,
    validate_pack,
)

REAL_PACK_DIR = str((BACKEND_ROOT / "skills" / "research").resolve())
REAL_FORMAT_DIR = str((BACKEND_ROOT / "skills" / "formats").resolve())


def _v2_service(**overrides) -> SkillPackService:
    params = {
        "skill_pack_enabled": True,
        "skill_pack_dir": REAL_PACK_DIR,
        "skill_format_dir": REAL_FORMAT_DIR,
        "skill_pack_min_match_score": 0.25,
    }
    params.update(overrides)
    cfg = Configuration.from_env(overrides=params, load_env_file=False)
    llm_router = params.pop("llm_router", None)
    return SkillPackService(cfg, llm_router=llm_router)


class SkillSystemV2Tests(unittest.TestCase):
    def test_all_shipped_domain_and_format_packs_validate_with_pinned_hashes(self) -> None:
        svc = _v2_service()
        catalog = svc.list_catalog()
        domain_ids = {item["pack_id"] for item in catalog["domain_packs"]}
        format_ids = {item["pack_id"] for item in catalog["format_packs"]}

        self.assertIn("financial-research", domain_ids)
        self.assertEqual(format_ids, {"exec-brief", "engineering-rfc", "deep-whitepaper"})
        self.assertEqual(
            set(catalog["registered_verifiers"]),
            {"numeric-audit", "timeline-audit", "comparison-matrix"},
        )

        for folder in (Path(REAL_PACK_DIR), Path(REAL_FORMAT_DIR)):
            for child in sorted(folder.iterdir()):
                if not child.is_dir():
                    continue
                pack = load_pack(child / "SKILL.md")
                errors = validate_pack(pack, char_limit=1200, directory_name=child.name)
                self.assertEqual(errors, [], f"{folder.name}/{child.name}: {errors}")
                if pack.pack_id != "_neutral":
                    self.assertEqual(pack.pinned_content_hash, pack.content_hash)

    def test_retrieval_profile_rejects_unsafe_hosts_and_adjusts_rrf_prior(self) -> None:
        base_pack = load_pack(Path(REAL_PACK_DIR) / "financial-research" / "SKILL.md")
        bad_pack = load_pack(Path(REAL_PACK_DIR) / "financial-research" / "SKILL.md")
        bad_pack.retrieval_profile = {
            "authoritative_hosts": ["http://localhost:8000", "127.0.0.1"],
        }
        errors = validate_pack(bad_pack, char_limit=1200, directory_name="financial-research")
        self.assertTrue(any("unsafe/invalid host hint" in err for err in errors))

        sec_prior = _fusion_source_prior("https://www.sec.gov/Archives/edgar/10k.htm", base_pack.retrieval_profile)
        dep_prior = _fusion_source_prior("https://zhuanlan.jianshu.com/p/123", base_pack.retrieval_profile)
        self.assertGreater(sec_prior, 0.04)
        self.assertLess(dep_prior, -0.02)

        sec_score = _compute_fusion_rrf_score(
            {"url": "https://www.sec.gov/10k", "_backend_ranks": [("tavily", 0, 0)]},
            retrieval_profile=base_pack.retrieval_profile,
        )
        ddg_score = _compute_fusion_rrf_score(
            {"url": "https://zhuanlan.jianshu.com/p/123", "_backend_ranks": [("duckduckgo", 0, 0)]},
            retrieval_profile=base_pack.retrieval_profile,
        )
        self.assertGreater(sec_score, ddg_score)

    def test_financial_research_routing_and_composable_secondary_constraint(self) -> None:
        svc = _v2_service()
        fin_sel = svc.select("某上市云计算公司 2025 财报营收、毛利率与估值分析")
        self.assertIsNotNone(fin_sel)
        assert fin_sel is not None
        self.assertEqual(fin_sel.pack.pack_id, "financial-research")
        self.assertIn("numeric-audit", fin_sel.active_verifiers)
        self.assertIn("timeline-audit", fin_sel.active_verifiers)

        # Tech selection primary + policy secondary constraint via positive signals
        combo = svc.select(
            "向量数据库组件选型、框架性能与权衡方案对比，同时评估数据条例与监管生效范围"
        )
        assert combo is not None
        self.assertEqual(combo.pack.pack_id, "tech-selection")
        self.assertIsNotNone(combo.secondary_pack)
        assert combo.secondary_pack is not None
        self.assertEqual(combo.secondary_pack.pack_id, "policy-analysis")

        # Secondary skill renders only for review/report and tightens quality gates monotonically
        self.assertEqual(svc.render_secondary_block(combo.secondary_pack, "planning"), "")
        self.assertIn("<SECONDARY_RESEARCH_SKILL", svc.render_secondary_block(combo.secondary_pack, "review"))

        base_cfg = Configuration.from_env(
            overrides={"review_min_sources_per_task": 2, "review_min_domains_per_task": 1, "freshness_reference_days": 365},
            load_env_file=False,
        )
        effective, applied = apply_skill_pack_overrides(
            base_cfg,
            combo.pack,
            secondary_pack=combo.secondary_pack,
        )
        self.assertEqual(effective.review_min_domains_per_task, 2)
        self.assertEqual(effective.review_min_sources_per_task, 3)
        self.assertEqual(effective.freshness_reference_days, 180)
        self.assertEqual(applied["review_min_sources_per_task"], 3)
        self.assertTrue(bool(effective.active_retrieval_profile))
        self.assertIn("timeline-audit", effective.active_skill_verifiers)
        self.assertIn("numeric-audit", effective.active_skill_verifiers)

    def test_executable_verifiers_flag_numeric_timeline_and_comparison_issues(self) -> None:
        state = SummaryState(
            research_topic="CloudDB vs VectorStore 性能选型对比",
            todo_items=[
                TodoItem(
                    id=1,
                    title="CloudDB 指标分析",
                    intent="对比 CloudDB 性能与份额",
                    query="CloudDB benchmark",
                    status="completed",
                    summary=(
                        "CloudDB 2026 年最新吞吐从 100 增长到 150，增幅达 120%。"
                        "市场占比分别为 A 45%、B 40%、C 30%。"
                    ),
                    evidence_items=[
                        {
                            "source_id": "T1-S1",
                            "title": "2022 CloudDB report",
                            "domain": "vendor1.com",
                            "url": "https://vendor1.com/2022",
                            "snippet": "2022 annual benchmark summary showed 100 QPS.",
                            "published_at": "2022-05-01",
                            "quality_label": "high",
                            "freshness_label": "fresh",
                        },
                        {
                            "source_id": "T1-S2",
                            "title": "Secondary note",
                            "domain": "vendor2.com",
                            "url": "https://vendor2.com/note",
                            "snippet": "Released in 2023.",
                            "published_at": "2023-01-10",
                            "quality_label": "medium",
                            "freshness_label": "recent",
                        },
                    ],
                ),
                TodoItem(
                    id=2,
                    title="VectorStore 指标分析",
                    intent="对比 VectorStore 性能",
                    query="VectorStore benchmark",
                    status="completed",
                    summary="VectorStore 文档暂无足够公开数据。",
                    evidence_items=[],
                ),
            ],
        )

        cfg = Configuration(review_agent_enabled=False, llm_api_key="test-key")
        reviewer = ReviewService(None, cfg)
        summary = reviewer.review_request(
            state,
            active_verifiers=["numeric-audit", "timeline-audit", "comparison-matrix"],
        )

        self.assertEqual(summary["verifiers_executed"], ["numeric-audit", "timeline-audit", "comparison-matrix"])
        self.assertGreaterEqual(summary["verifier_issue_count"], 3)
        checks = {item["check"] for item in summary["issues"]}
        self.assertIn("numeric_audit_mismatch", checks)
        self.assertIn("timeline_inconsistency", checks)
        self.assertIn("comparison_asymmetry", checks)

    def test_format_skill_selection_rendering_and_stream_outline(self) -> None:
        svc = _v2_service()
        explicit = svc.select("Qdrant vs Milvus 选型", output_format="engineering-rfc")
        assert explicit is not None and explicit.format_pack is not None
        self.assertEqual(explicit.format_pack.pack_id, "engineering-rfc")
        rendered = svc.render_format_block(explicit.format_pack, "report")
        self.assertIn('<OUTPUT_FORMAT_SKILL format="engineering-rfc"', rendered)
        self.assertEqual(svc.render_format_block(explicit.format_pack, "planning"), "")

        auto_brief = svc.select("请给我一份高管摘要与一页纸决策简报，介绍 MCP 进展")
        assert auto_brief is not None and auto_brief.format_pack is not None
        self.assertEqual(auto_brief.format_pack.pack_id, "exec-brief")

        class _FakeReportAgent:
            def run(self, prompt: str, **kwargs):
                return "# 决策简报\n\n核心结论 [T1-S1]"

            def clear_history(self):
                return None

        reporter = ReportingService(
            _FakeReportAgent(),  # type: ignore[arg-type]
            Configuration(llm_api_key="k"),
            evidence_store=EvidenceStore(),
        )
        state = SummaryState(
            research_topic="MCP 进展",
            todo_items=[
                TodoItem(
                    id=1,
                    title="MCP 标准化进展",
                    intent="梳理协议进度",
                    query="MCP spec",
                    status="completed",
                    summary="核心发现 [T1-S1]",
                    evidence_items=[{"source_id": "T1-S1", "title": "Spec", "url": "https://example.com"}],
                )
            ],
        )
        events = list(
            reporter.stream_generate_report(
                state,
                outline_sections=explicit.format_pack.outline_sections,
            )
        )
        self.assertEqual(events[0]["type"], "report_outline")
        self.assertEqual(events[0]["sections"][:2], ["问题背景与工程约束", "候选架构对比矩阵"])

    def test_hybrid_semantic_and_llm_router_tiers_and_skills_endpoint(self) -> None:
        # Mock encode_text so an oblique topic routes to policy-analysis via Tier-2 semantic similarity.
        from unittest.mock import patch

        def _fake_encode(text: str):
            if "监管" in text or "法规" in text or "GDPR" in text:
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]

        svc = _v2_service(skill_pack_semantic_routing_enabled=True, skill_pack_min_match_score=0.40)
        with patch("services.embeddings.encode_text", side_effect=_fake_encode):
            sem = svc.select("欧盟 GDPR 罚款条款与企业出海边界检查")
        assert sem is not None
        self.assertEqual(sem.pack.pack_id, "policy-analysis")
        self.assertEqual(sem.routing_method, "hybrid_semantic")

        # Tier-3 LLM router fallback
        llm_svc = _v2_service(
            skill_pack_llm_router_enabled=True,
            skill_pack_min_match_score=0.50,
            llm_router=lambda topic, candidates: {"pack_id": "financial-research", "confidence": 0.88},
        )
        routed = llm_svc.select("分析某大型云厂商的年度资本开支与现金状况")
        assert routed is not None
        self.assertEqual(routed.pack.pack_id, "financial-research")
        self.assertEqual(routed.routing_method, "llm_tiebreak")

        # GET /skills HTTP route
        cfg = Configuration.from_env(
            overrides={
                "skill_pack_enabled": True,
                "skill_pack_dir": REAL_PACK_DIR,
                "skill_format_dir": REAL_FORMAT_DIR,
            },
            load_env_file=False,
        )
        app = create_app(base_config=cfg)
        client = TestClient(app)
        resp = client.get("/skills")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["enabled"])
        self.assertGreaterEqual(len(body["domain_packs"]), 6)
        self.assertEqual(len(body["format_packs"]), 3)

    def test_hybrid_judge_maps_heuristic_citation_keys_accurately(self) -> None:
        case = BenchmarkCase(
            id="case-1",
            topic="RAG 架构对比",
            expected_keywords=["RAG", "RRF"],
            expected_sections=["背景", "核心洞见"],
        )
        report = (
            "## 背景\nRAG 系统综述。\n\n"
            "## 核心洞见\n- 混合 RRF 检索显著提升召回 [T1-S1]\n\n"
            "## 参考来源\n- [T1-S1] https://example.com\n"
        )
        judge = HybridJudge(heuristic_judge=HeuristicJudge(), llm_judge=None)
        result = judge.evaluate(case=case, report_markdown=report, todo_items=[], trace_snapshot={})
        self.assertEqual(result["reference_match_rate"], 1.0)
        self.assertEqual(result["grounded_bullet_ratio"], 1.0)
        self.assertEqual(result["heuristic_composite_score"], 1.0)
        self.assertEqual(result["hybrid_score"], 1.0)
