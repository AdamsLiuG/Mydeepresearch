import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for candidate in (BACKEND_SRC, BACKEND_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from config import Configuration
from evals.judges.heuristic import HeuristicJudge
from metrics import RequestTrace, build_token_usage, metrics_registry
from models import SummaryState, TodoItem
from services.context_builder import (
    ContextPrefixBuilder,
    ProviderCacheAdapter,
)
from services.dag_scheduler import build_upstream_context
from services.evidence import (
    EvidenceLookupTool,
    EvidenceStore,
    detect_source_type,
    extract_citation_ids,
)
from services.evidence_index import (
    EvidenceMemoryService,
    EvidenceRetrievalService,
    EvidenceRuntimeIndex,
)
from services.fact_ledger import (
    FactLedger,
    FactLedgerCompactor,
    build_evidence_index_cards,
    render_evidence_index_cards_block,
)
from services.hitl import HitlService
from services.memory_sanitizer import neutralize_source_ids
from services.reporter import ReportingService
from services.skill_pack import _validate_fragment_safety
from services.text_processing import strip_citation_markers
from state_manager import StateManagerMixin


class UnifiedEvidenceAndContextTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()
        self.config = Configuration.from_env(load_env_file=False)

    def test_unified_citation_tokens_across_all_sanitizers(self) -> None:
        sample = "公网 [T1-S1]、MCP 研报 [T1-M2] 与沙箱计算 [T2-C1] 均可识别，非法 [T9-S9] 应剔除。"
        self.assertEqual(set(extract_citation_ids(sample)), {"T1-S1", "T1-M2", "T2-C1", "T9-S9"})
        stripped = strip_citation_markers(sample)
        self.assertNotIn("[T1-M2]", stripped)
        self.assertNotIn("[T2-C1]", stripped)

        neutralized = neutralize_source_ids("历史记忆引用 [T1-M2] 和 [T2-C1]")
        self.assertNotIn("[T1-M2]", neutralized)
        self.assertNotIn("[T2-C1]", neutralized)

        sanitized_hitl = HitlService.sanitize_citations(sample, {"T1-S1", "T1-M2", "T2-C1"})
        self.assertIn("[T1-M2]", sanitized_hitl)
        self.assertIn("[T2-C1]", sanitized_hitl)
        self.assertNotIn("[T9-S9]", sanitized_hitl)

        sanitized_report = ReportingService._sanitize_chunk_citations(sample, {"T1-S1", "T1-M2", "T2-C1"})
        self.assertIn("[T1-M2]", sanitized_report)
        self.assertIn("[T2-C1]", sanitized_report)
        self.assertNotIn("[T9-S9]", sanitized_report)

        errors = _validate_fragment_safety("task_execution", "[T1-M1] leaked citation") + _validate_fragment_safety(
            "task_execution",
            "[T2-C1] leaked computation",
        )
        self.assertEqual(len(errors), 2)

        judge = HeuristicJudge()
        eval_res = judge.evaluate(
            case=SimpleNamespace(expected_sections=["核心洞见"], expected_keywords=["结论"]),
            report_markdown=(
                "# 报告\n\n"
                "## 核心洞见\n- 财务结论 [T1-M1] 与计算结果 [T2-C1] 一致。\n\n"
                "## 参考来源\n- [T1-M1] Filing\n- [T2-C1] Sandbox\n"
            ),
            todo_items=[
                SimpleNamespace(
                    id=1,
                    status="completed",
                    evidence_items=[
                        {"source_id": "T1-M1", "url": "finarag://sha1#page=1"},
                        {"source_id": "T2-C1", "url": "sandbox://req/T2-C1"},
                    ],
                )
            ],
            trace_snapshot={},
        )
        self.assertEqual(eval_res["citation_marker_count"], 2)
        self.assertEqual(eval_res["reference_match_rate"], 1.0)
        self.assertEqual(eval_res["grounded_bullet_ratio"], 1.0)

    def test_evidence_store_mcp_compute_cards_and_lazy_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_cfg = Configuration.from_env(
                overrides={
                    "evidence_runtime_enabled": True,
                    "evidence_memory_enabled": True,
                    "evidence_memory_dir": tmpdir,
                },
                load_env_file=False,
            )
            memory = EvidenceMemoryService(mem_cfg)
            runtime = EvidenceRuntimeIndex(mem_cfg, archive_loader=memory)
            store = EvidenceStore(listeners=[runtime, memory])

            web_items = store.record_search_results(
                task_id=1,
                query="vLLM latency",
                search_payload={
                    "results": [
                        {
                            "title": "vLLM Docs",
                            "url": "https://docs.vllm.ai/en/latest/bench",
                            "content": "vLLM achieves 24% latency reduction with PagedAttention. " * 6,
                        }
                    ]
                },
                backend="tavily",
            )
            mcp_items = store.record_mcp_results(
                task_id=1,
                query="NVIDIA FY2025 margin",
                server_id="finarag",
                tool_name="finarag_filings",
                items=[
                    {
                        "title": "NVIDIA 10-K FY2025",
                        "url": "finarag://abc123#page=12",
                        "snippet": "Gross margin reached 75.0% in FY2025.",
                        "full_content": "NVIDIA reported gross margin of 75.0% in FY2025. " * 6,
                        "source_type": "mcp_curated_filing",
                        "provenance_meta": {"pdf_sha1": "abc123", "page_index_1based": 12},
                    }
                ],
                tenant_id="tenant-a",
            )
            comp_item = store.record_computation_evidence(
                task_id=2,
                title="CAGR Check",
                derived_from_source_ids=["T1-S1", "T1-M1"],
                computation_artifact={"result": {"cagr_pct": 18.5}, "execution_hash": "deadbeef1234"},
                summary_text="Computed 3-year CAGR = 18.5% based on audited disclosures.",
            )

            self.assertEqual(web_items[0]["source_id"], "T1-S1")
            self.assertEqual(mcp_items[0]["source_id"], "T1-M1")
            self.assertEqual(comp_item["source_id"], "T2-C1")
            self.assertEqual(detect_source_type({"source_origin": "mcp", "source_type": "mcp_curated_filing"}), "mcp_curated_filing")
            self.assertEqual(detect_source_type({"source_origin": "sandbox", "source_type": "sandbox_computation"}), "sandbox_computation")

            # Private MCP & sandbox records must never be persisted to shared global archive
            memory.on_records_changed(
                task_id=1,
                records=[SimpleNamespace(source_origin="mcp", url="finarag://abc123#page=12", snippet="secret")],
                reason="search",
            )
            memory.on_records_changed(
                task_id=2,
                records=[SimpleNamespace(source_origin="sandbox", url="sandbox://compute/2/1", snippet="computed")],
                reason="search",
            )
            manifest_raw = (
                memory._manifest_path.read_text(encoding="utf-8")
                if memory._manifest_path.exists()
                else ""
            )
            self.assertNotIn("finarag://abc123", manifest_raw)
            self.assertNotIn("sandbox://compute", manifest_raw)

            cards = build_evidence_index_cards(store, runtime_index=runtime)
            self.assertEqual(len(cards), 3)
            rendered_cards = render_evidence_index_cards_block(cards)
            self.assertIn("[T1-M1]", rendered_cards)
            self.assertIn("[T2-C1]", rendered_cards)

            retrieval_svc = EvidenceRetrievalService(
                runtime_index=runtime,
                memory_service=memory,
                config=mem_cfg,
            )
            lookup_tool = EvidenceLookupTool(evidence_store=store, retrieval_service=retrieval_svc)
            cards_payload = json.loads(lookup_tool.run({"mode": "index_cards", "task_id": 1}))
            self.assertEqual(len(cards_payload["index_cards"]), 2)

            lazy_payload = json.loads(
                lookup_tool.run(
                    {
                        "mode": "lazy_chunks",
                        "source_ids": ["T1-M1", "T2-C1"],
                        "chunk_indices": [1],
                        "char_budget": 1000,
                    }
                )
            )
            self.assertTrue(any(item["source_id"] == "T1-M1" for item in lazy_payload["results"]))

            filtered = retrieval_svc.query(
                "margin",
                task_id=0,
                scope="current",
                allowed_source_ids=["T1-M1"],
            )
            self.assertTrue(filtered.hits)
            self.assertTrue(all(hit.source_id == "T1-M1" for hit in filtered.hits))

    def test_context_prefix_builder_and_provider_adapters(self) -> None:
        cfg_openai = Configuration.from_env(
            overrides={"kv_cache_provider_mode": "openai", "kv_cache_min_segment_tokens": 10},
            load_env_file=False,
        )
        builder = ContextPrefixBuilder(cfg_openai)
        shared_ledger = "<FACT_LEDGER>\n- C-001 (T1, conf=0.80, refs=T1-S1): PagedAttention cuts latency 24%\n</FACT_LEDGER>"
        env1 = builder.build_envelope(
            stage="task_summarization",
            tenant_id="tenant-alpha",
            request_id="req-1",
            system_role="你是总结专家",
            static_schema_and_rules="输出严格 JSON。" * 10,
            research_topic="AI Infra Inference",
            skill_pack_context="## task_execution\n核验延迟数据",
            frozen_index_and_ledger=shared_ledger,
            volatile_blocks=[("task_delta", "任务 1：评估 vLLM")],
        )
        env2 = builder.build_envelope(
            stage="task_summarization",
            tenant_id="tenant-alpha",
            request_id="req-1",
            system_role="你是总结专家",
            static_schema_and_rules="输出严格 JSON。" * 10,
            research_topic="AI Infra Inference",
            skill_pack_context="## task_execution\n核验延迟数据",
            frozen_index_and_ledger=shared_ledger,
            volatile_blocks=[("task_delta", "任务 2：评估 TensorRT-LLM")],
        )
        env_other_tenant = builder.build_envelope(
            stage="task_summarization",
            tenant_id="tenant-beta",
            request_id="req-2",
            system_role="你是总结专家",
            static_schema_and_rules="输出严格 JSON。" * 10,
            research_topic="AI Infra Inference",
            skill_pack_context="## task_execution\n核验延迟数据",
            frozen_index_and_ledger=shared_ledger,
            volatile_blocks=[("task_delta", "任务 1：评估 vLLM")],
        )

        self.assertEqual(env1.prefix_fingerprint, env2.prefix_fingerprint)
        msgs1, kwargs1 = ProviderCacheAdapter.render_messages_and_kwargs(
            env1, provider_mode="openai", config=cfg_openai
        )
        msgs2, kwargs2 = ProviderCacheAdapter.render_messages_and_kwargs(
            env2, provider_mode="openai", config=cfg_openai
        )
        _, kwargs_beta = ProviderCacheAdapter.render_messages_and_kwargs(
            env_other_tenant, provider_mode="openai", config=cfg_openai
        )
        self.assertEqual(msgs1[0]["content"], msgs2[0]["content"])
        self.assertEqual(kwargs1["prompt_cache_key"], kwargs2["prompt_cache_key"])
        self.assertNotEqual(kwargs1["prompt_cache_key"], kwargs_beta["prompt_cache_key"])

        cfg_anthropic = Configuration.from_env(
            overrides={"kv_cache_provider_mode": "anthropic", "kv_cache_min_segment_tokens": 10},
            load_env_file=False,
        )
        anth_msgs, anth_kwargs = ProviderCacheAdapter.render_messages_and_kwargs(
            env1, provider_mode="anthropic", config=cfg_anthropic
        )
        self.assertEqual(anth_kwargs, {})
        self.assertEqual(anth_msgs[0]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(anth_msgs[1]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", anth_msgs[1]["content"][1])

    def test_safe_llm_cache_usage_extraction_and_one_shot_downgrade(self) -> None:
        import agent as agent_module

        class RejectCacheCompletions:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if "prompt_cache_key" in kwargs:
                    raise RuntimeError("unknown parameter: prompt_cache_key")
                return [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", reasoning=None))],
                        usage={
                            "prompt_tokens": 1000,
                            "completion_tokens": 100,
                            "total_tokens": 1100,
                            "cache_read_input_tokens": 800,
                            "cache_creation_input_tokens": 200,
                        },
                    )
                ]

        llm = agent_module.SafeHelloAgentsLLM.__new__(agent_module.SafeHelloAgentsLLM)
        llm.model = "demo"
        llm.provider = "openai"
        llm.temperature = 0.0
        llm.max_tokens = 256
        llm.allow_reasoning_fallback = True
        llm._usage_local = agent_module.local()
        llm._cache_header_unsupported = False
        fake = RejectCacheCompletions()
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        out = llm.invoke(
            [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}],
            prompt_cache_key="dr_key_1",
        )
        self.assertEqual(out, "ok")
        self.assertTrue(llm._cache_header_unsupported)
        self.assertEqual(fake.calls[-1]["messages"][0]["content"], "hello")
        self.assertNotIn("prompt_cache_key", fake.calls[-1])
        usage = llm.consume_last_usage()
        self.assertEqual(usage["cached_tokens"], 800)
        self.assertEqual(usage["cache_creation_tokens"], 200)

        # Subsequent call skips sending prompt_cache_key immediately
        llm.invoke([{"role": "user", "content": "hi2"}], prompt_cache_key="dr_key_2")
        self.assertNotIn("prompt_cache_key", fake.calls[-1])

    def test_fact_ledger_compactor_dedup_conflict_and_upstream_context(self) -> None:
        compactor = FactLedgerCompactor()
        t1 = TodoItem(
            id=1,
            title="GPU 供给",
            intent="统计出货",
            query="gpu shipment",
            status="completed",
            summary="2025年 NVIDIA 毛利率为 75% [T1-S1]",
            evidence_items=[{"source_id": "T1-S1"}],
            fact_claims=[
                {
                    "entity": "NVIDIA",
                    "metric": "毛利率",
                    "value": "75%",
                    "period": "2025",
                    "source_ids": ["T1-S1"],
                    "confidence": 0.80,
                    "statement": "NVIDIA 2025 年毛利率为 75%",
                }
            ],
        )
        t2 = TodoItem(
            id=2,
            title="财报复核",
            intent="交叉验证",
            query="nvidia margin",
            status="completed",
            summary="不同口径测算 NVIDIA 毛利率为 72% [T2-S1]",
            evidence_items=[{"source_id": "T2-S1"}],
            fact_claims=[
                {
                    "entity": "NVIDIA",
                    "metric": "毛利率",
                    "value": "75%",
                    "period": "2025",
                    "source_ids": ["T2-S1"],
                    "confidence": 0.80,
                    "statement": "NVIDIA FY2025 毛利率确认 75%",
                },
                {
                    "entity": "NVIDIA",
                    "metric": "毛利率",
                    "value": "72%",
                    "period": "2025",
                    "source_ids": ["T2-S1"],
                    "confidence": 0.70,
                    "statement": "调整后口径显示 NVIDIA 毛利率为 72%",
                },
            ],
        )
        ledger = compactor.update_ledger(None, [t1], valid_source_ids={"T1-S1"})
        self.assertEqual(len(ledger.entries), 1)
        first_prefix = ledger.render_frozen_markdown()

        ledger = compactor.update_ledger(ledger, [t1, t2], valid_source_ids={"T1-S1", "T2-S1"})
        self.assertEqual(len(ledger.entries), 2)
        self.assertEqual(ledger.entries[0].stance, "disputed")
        self.assertEqual(ledger.entries[1].stance, "disputed")
        self.assertTrue(ledger.conflict_groups)
        self.assertIn("T2-S1", ledger.entries[0].source_ids)
        self.assertIn("[CONFLICT:", ledger.render_frozen_markdown(max_entries=1))
        self.assertNotEqual(first_prefix, ledger.render_frozen_markdown())

        t3 = TodoItem(
            id=3,
            title="下游结论",
            intent="汇总",
            query="summary",
            depends_on=[1],
        )
        upstream_ctx = build_upstream_context(
            t3,
            [t1, t2, t3],
            fact_ledger=ledger.to_dict(),
            compact_enabled=True,
        )
        self.assertIn("前置任务 1（GPU 供给）结论：", upstream_ctx)
        self.assertIn("C-001", upstream_ctx)

    def test_metrics_kv_cache_cost_and_snapshot_roundtrip(self) -> None:
        catalog = {
            "openai": {
                "gpt-test": {
                    "prompt_per_1k_tokens": 0.01,
                    "completion_per_1k_tokens": 0.03,
                    "cached_prompt_per_1k_tokens": 0.0025,
                    "cache_creation_per_1k_tokens": 0.0125,
                }
            }
        }
        usage = build_token_usage(
            prompt_text="ignored",
            completion_text="ignored",
            provider="openai",
            model="gpt-test",
            pricing_catalog=catalog,
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cached_tokens": 800,
                "cache_creation_tokens": 0,
            },
        )
        # 200 uncached ($0.002) + 800 cached ($0.002) + 100 completion ($0.003) = $0.007
        self.assertAlmostEqual(usage["estimated_cost"], 0.007, places=6)
        self.assertEqual(usage["uncached_prompt_tokens"], 200)
        self.assertEqual(usage["cache_hit_ratio"], 0.8)

        trace = RequestTrace(
            request_id="req-kv",
            topic="kv",
            search_api="duckduckgo",
            provider="openai",
            model="gpt-test",
            pricing_catalog=catalog,
        )
        trace.record_llm_call(
            success=True,
            prompt_text="p",
            completion_text="c",
            usage={"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 800},
            stage="task_summarization",
        )
        snap = trace.snapshot()
        self.assertEqual(snap["cached_prompt_tokens"], 800)
        self.assertEqual(snap["uncached_prompt_tokens"], 200)
        self.assertAlmostEqual(snap["cache_hit_ratio"], 0.8, places=4)
        self.assertEqual(snap["kv_cache_by_stage"]["task_summarization"]["calls"], 1)
        self.assertAlmostEqual(metrics_registry.snapshot()["kv_cache_hit_ratio"], 0.8, places=4)

        # Snapshot roundtrip via StateManagerMixin
        class _JsonMemoryStore:
            def __init__(self) -> None:
                self._data: dict[str, str] = {}

            def save(self, req_id: str, payload: dict, *, tenant_id: str = "default") -> None:
                self._data[f"{tenant_id}:{req_id}"] = json.dumps(payload, ensure_ascii=False)

            def load(self, req_id: str, *, tenant_id: str = "default") -> dict | None:
                raw = self._data.get(f"{tenant_id}:{req_id}")
                return json.loads(raw) if raw else None

        class DummyManager(StateManagerMixin):
            def __init__(self) -> None:
                self.request_id = "req-kv"
                self.tenant_id = "t1"
                self.config = Configuration.from_env(load_env_file=False)
                self._request_state_store = _JsonMemoryStore()
                self._evidence_store = EvidenceStore()
                self._request_trace = trace
                self._cache_stats = {}

            def _serialize_task(self, task: TodoItem) -> dict:
                return {
                    "id": task.id,
                    "title": task.title,
                    "intent": task.intent,
                    "query": task.query,
                    "status": task.status,
                    "fact_claims": list(task.fact_claims),
                }

            def _refresh_note_memory_for_state(self, state, observer=None):
                return None

        mgr = DummyManager()
        state = SummaryState(
            research_topic="KV test",
            todo_items=[TodoItem(id=1, title="t1", intent="i1", query="q1", fact_claims=[{"entity": "X"}])],
            fact_ledger={"version": 2, "entries": [], "conflict_groups": {}, "evidence_gaps": []},
            evidence_index_cards=[{"source_id": "T1-S1"}],
            frozen_prefix_version=3,
        )
        mgr._persist_request_state(state, phase="review", status="in_progress")
        loaded_payload = mgr._load_resume_snapshot("req-kv")
        restored_state, phase = mgr._state_from_snapshot(loaded_payload)
        self.assertEqual(phase, "review")
        self.assertEqual(restored_state.frozen_prefix_version, 3)
        self.assertEqual(FactLedger.from_dict(restored_state.fact_ledger).version, 2)
        self.assertEqual(restored_state.evidence_index_cards, [{"source_id": "T1-S1"}])
        self.assertEqual(restored_state.todo_items[0].fact_claims, [{"entity": "X"}])


if __name__ == "__main__":
    unittest.main()
