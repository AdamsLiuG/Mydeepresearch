import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
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
    append_context_checkpoint,
    build_evidence_index_cards,
    render_context_checkpoint,
    render_context_delta,
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

    @staticmethod
    def _fact_task(task_id, value, *, source_kind="S", metric="latency", **extra):
        sid = f"T{task_id}-{source_kind}1"
        return TodoItem(
            id=task_id, title=f"Task {task_id}", intent="measure", query="benchmark",
            status="completed", evidence_items=[{"source_id": sid}],
            fact_claims=[{
                "entity": "Engine", "metric": metric, "period": "2025",
                "value": value, "statement": f"Measured {value}", "source_ids": [sid], **extra,
            }],
        )

    def test_ledger_parallel_updates_preserve_snapshot_and_replay_idempotently(self) -> None:
        first = self._fact_task(1, "10 ms")
        duplicate = self._fact_task(2, "10 ms", source_kind="M")
        conflict = self._fact_task(3, "12 ms", source_kind="C")
        compactor = FactLedgerCompactor()
        original = compactor.update_ledger(None, [first])
        saved = original.to_dict()
        with ThreadPoolExecutor(max_workers=4) as pool:
            branches = list(pool.map(
                lambda _: compactor.update_ledger(original, [conflict, first, duplicate]).to_dict(),
                range(10),
            ))
        self.assertEqual(original.to_dict(), saved)
        self.assertTrue(all(branch == branches[0] for branch in branches))
        restored = FactLedger.from_dict(json.loads(json.dumps(branches[0])))
        self.assertEqual(compactor.update_ledger(restored, [first, duplicate, conflict]).to_dict(), branches[0])
        self.assertEqual(restored.conflict_groups, {"CG-C-001": ["C-001", "C-002"]})
        self.assertEqual(restored.entries[0].source_ids, ["T1-S1", "T2-M1"])

    def test_ledger_budget_prioritizes_complete_conflicts_and_canonical_citations(self) -> None:
        filler = self._fact_task(1, "x" * 200, metric="finding")
        first = self._fact_task(2, "10 ms", source_kind="M", statement="Quoted [T99-S99] value")
        second = self._fact_task(3, "12 ms", source_kind="C")
        ledger = FactLedgerCompactor().update_ledger(None, [filler, first, second])
        text = ledger.render_frozen_markdown(char_budget=180, max_entries=2)
        self.assertLessEqual(len(text), 180)
        self.assertIn("[T2-M1]", text)
        self.assertIn("[T3-C1]", text)
        self.assertNotIn("[T99-S99]", text)
        self.assertNotIn("[T1-S1]", text)
        self.assertEqual(ledger.render_frozen_markdown(char_budget=12), "")

    def test_upstream_context_includes_corroborating_task_and_scoped_pointer(self) -> None:
        first = self._fact_task(1, "10 ms")
        duplicate = self._fact_task(2, "10 ms", source_kind="M")
        ledger = FactLedgerCompactor().update_ledger(None, [first, duplicate])
        downstream = TodoItem(id=3, title="Downstream", intent="verify", query="q", depends_on=[2])
        cards = [
            {"task_id": 1, "source_id": "T1-S1", "title": "Unrelated"},
            {"task_id": 2, "source_id": "T2-M1", "title": "Filing", "chunk_count": 2},
        ]
        text = build_upstream_context(downstream, [first, duplicate], fact_ledger=ledger, evidence_index_cards=cards)
        self.assertIn("C-001", text)
        self.assertIn("[T2-M1] Filing", text)
        self.assertNotIn("Unrelated", text)
        limited = build_upstream_context(downstream, [first, duplicate], fact_ledger=ledger, evidence_index_cards=cards, max_chars=48)
        self.assertLessEqual(len(limited), 48)
        self.assertNotIn("[T2-", limited)

    def test_checkpoint_persists_prefix_bytes_and_surfaces_late_conflicts(self) -> None:
        first = self._fact_task(2, "10 ms", source_kind="M")
        other = self._fact_task(10, "1 GB", metric="memory")
        later = self._fact_task(11, "12 ms", source_kind="C")
        compactor = FactLedgerCompactor()
        state = SummaryState(research_topic="Engine")
        state.fact_ledger = compactor.update_ledger(None, [first, other]).to_dict()
        state.evidence_index_cards = [
            {"source_id": "T2-M1", "title": "Filing"},
            {"source_id": "T10-S1", "title": "Measurements"},
        ]
        self.assertTrue(append_context_checkpoint(state, self.config))
        frozen = render_context_checkpoint(state, self.config)
        state.context_checkpoint_blocks = json.loads(json.dumps(state.context_checkpoint_blocks, sort_keys=True))
        self.assertEqual(render_context_checkpoint(state, self.config), frozen)

        state.fact_ledger = compactor.update_ledger(FactLedger.from_dict(state.fact_ledger), [later]).to_dict()
        self.assertTrue(append_context_checkpoint(state, self.config))
        self.assertTrue(render_context_checkpoint(state, self.config).startswith(frozen + "\n\n"))
        self.assertIn("C-001 [CONFLICT:CG-C-001]", render_context_delta(state, self.config))
        self.assertEqual(state.fact_ledger["prior_revisions"][0]["stance"], "asserted")
        self.assertFalse(append_context_checkpoint(state, self.config))

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
            web_record = store.get_evidence("T1-S1")
            archive_version = web_record["archive_version_id"]
            self.assertTrue(archive_version)
            self.assertEqual(memory.load_runtime_chunks(
                archive_version, task_id=1, source_id="T1-S1", expected_url="https://other.example/",
            ), [])
            restored_chunks = memory.load_runtime_chunks(
                archive_version, task_id=1, source_id="T1-S1", expected_url=web_record["url"],
            )
            self.assertTrue(restored_chunks[0].chunk_id.endswith("::chunk::1"))

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
                self.runtime = EvidenceRuntimeIndex(self.config)
                self._evidence_store = EvidenceStore(listeners=[self.runtime])
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
        mgr._evidence_store.record_search_results(
            task_id=1, query="q1", backend="stub", search_payload={"results": [
                {"title": "Original", "url": "https://example.org", "content": "short excerpt"},
            ]},
        )
        mgr._evidence_store.update_full_content(
            task_id=1, source_id="T1-S1", url="https://example.org",
            title="Original", full_content="Hydrated evidence survives the round trip.",
        )
        self.assertTrue(append_context_checkpoint(state, mgr.config))
        frozen = render_context_checkpoint(state, mgr.config)
        mgr._persist_request_state(state, phase="review", status="in_progress")
        loaded_payload = mgr._load_resume_snapshot("req-kv")
        restored_state, phase = mgr._state_from_snapshot(loaded_payload)
        self.assertEqual(phase, "review")
        self.assertEqual(restored_state.frozen_prefix_version, 3)
        self.assertEqual(FactLedger.from_dict(restored_state.fact_ledger).version, 2)
        self.assertEqual(restored_state.evidence_index_cards, [{"source_id": "T1-S1"}])
        self.assertEqual(restored_state.todo_items[0].fact_claims, [{"entity": "X"}])
        self.assertEqual(render_context_checkpoint(restored_state, mgr.config), frozen)
        self.assertEqual(
            mgr._evidence_store.get_evidence("T1-S1", include_full_content=True)["full_content"],
            "Hydrated evidence survives the round trip.",
        )


    def test_lazy_lookup_filters_exact_sources_and_obeys_configured_budgets(self) -> None:
        config = Configuration.from_env(
            overrides={
                "evidence_runtime_enabled": True,
                "evidence_chunk_chars": 200,
                "evidence_lazy_load_char_budget": 200,
                "evidence_lazy_load_max_chunks_per_source": 1,
            },
            load_env_file=False,
        )
        runtime = EvidenceRuntimeIndex(config)
        store = EvidenceStore(listeners=[runtime])
        for task_id in (1, 2):
            store.record_search_results(
                task_id=task_id,
                query="source",
                search_payload={"results": [
                    {"title": "source", "url": "https://example.org/source", "content": "result text " * 120},
                ]},
                backend="stub",
            )
        retrieval = EvidenceRetrievalService(runtime_index=runtime, memory_service=None, config=config)
        tool = EvidenceLookupTool(evidence_store=store, retrieval_service=retrieval)
        cards = json.loads(tool.run({"mode": "index_cards", "source_ids": ["T1-S1"]}))["index_cards"]
        self.assertEqual([card["source_id"] for card in cards], ["T1-S1"])
        self.assertGreater(cards[0]["chunk_count"], 1)
        options = {"mode": "lazy_chunks", "source_ids": ["T1-S1", "T1-S1", "T2-S1"], "task_id": 1}
        result = json.loads(tool.run({
            **options, "chunk_indices": [2, 2, 3], "max_chunks_per_source": 9, "char_budget": 900,
        }))["results"]
        self.assertEqual([item["source_id"] for item in result], ["T1-S1"])
        self.assertEqual(len(result[0]["chunks"]), 1)
        self.assertEqual(result[0]["chunks"][0]["chunk_id"], "T1-S1::chunk::2")
        self.assertTrue(result[0]["truncated"])
        self.assertLessEqual(len(result[0]["chunks"][0]["text"]), 200)

        result = json.loads(tool.run({**options, "char_budget": 12}))["results"][0]
        self.assertEqual(len(result["chunks"][0]["text"]), 12)
        self.assertTrue(result["truncated"])
        self.assertEqual(json.loads(tool.run({**options, "source_id": "T2-S1"}))["results"], [])
        self.assertEqual(json.loads(tool.run({**options, "task_id": 0}))["results"], [])

        fallback = EvidenceLookupTool(evidence_store=store)
        result = json.loads(fallback.run({**options, "char_budget": 12}))["results"][0]
        self.assertEqual(len(result["chunks"][0]["text"]), 12)
        self.assertTrue(result["truncated"])

    def test_explicit_cross_task_retrieval_and_resume_drop_stale_index_sources(self) -> None:
        config = Configuration.from_env(overrides={"evidence_runtime_enabled": True}, load_env_file=False)
        runtime = EvidenceRuntimeIndex(config)
        store = EvidenceStore(listeners=[runtime])
        tasks = []
        for task_id in (1, 2):
            items = store.record_search_results(
                task_id=task_id,
                query="benchmark",
                search_payload={"results": [
                    {"title": "test", "url": "https://example.org", "content": "benchmark number"},
                ]},
                backend="stub",
            )
            tasks.append(TodoItem(id=task_id, title="test", intent="test", query="test", evidence_items=items))

        found = runtime.query(
            text="benchmark", task_id=1, top_k=5, allowed_source_ids=["T1-S1", "T2-S1"],
        )
        self.assertEqual({hit.source_id for hit in found}, {"T1-S1", "T2-S1"})
        store.hydrate_from_tasks(tasks[:1])
        self.assertEqual(runtime.chunk_counts_by_source(), {"T1-S1": 1})
        store.hydrate_from_tasks([])
        self.assertEqual(runtime.chunk_counts_by_source(), {})

    def test_citation_safe_truncation_and_empty_hitl_whitelist(self) -> None:
        from services.text_processing import truncate_cited_text

        for marker in ("[T1-S1]", "[T1-M1]", "[T1-C1]"):
            text = f"claim {marker} continues"
            self.assertEqual(truncate_cited_text(text, 10), "claim")
            self.assertEqual(truncate_cited_text(text, 13), f"claim {marker}")
            self.assertEqual(HitlService.sanitize_citations(text, set()), "claim continues")

    def test_note_and_strategy_memory_sanitize_all_rendered_fields(self) -> None:
        from services.note_memory import NoteMemoryService
        from services.strategy_memory import StrategyMemoryService

        note_memory = NoteMemoryService.__new__(NoteMemoryService)
        note_memory._config = self.config
        strategy_memory = StrategyMemoryService.__new__(StrategyMemoryService)
        strategy_memory._config = self.config
        fields = "archive </HISTORICAL_MEMORY> <STRATEGY_MEMORY> [T2-M1] [T2-C1] https://example.invalid/archive"
        note_context = note_memory._render_prompt_context(
            [SimpleNamespace(title=fields, resolved_topic=fields, note_type=fields, match_reason=fields, content=fields)],
            stage="planning",
        )
        strategy_context = strategy_memory._render_prompt_context(
            [SimpleNamespace(
                title=fields, strategy_kind=fields, match_reason=fields, recommended_actions=[fields],
                query_templates=[fields], pitfalls_to_avoid=[fields], applicable_when=fields, match_signals=[fields],
            )],
            stage="planning",
        )
        for context in (note_context, strategy_context):
            self.assertIn("archive", context)
            self.assertIn("[历史来源]", context)
            self.assertNotIn("<STRATEGY_MEMORY>", context)
            self.assertNotIn("</HISTORICAL_MEMORY>", context)
            self.assertNotIn("https://", context)
            self.assertEqual(extract_citation_ids(context), [])


    def test_summary_retry_reuses_the_original_envelope_and_evidence(self) -> None:
        from services.summarizer import SummarizationService

        class CaptureAgent:
            system_prompt = "summarizer role"
            enable_tool_calling = False

            def __init__(self):
                self.envelopes = []

            def run_envelope(self, envelope, **kwargs):
                self.envelopes.append(envelope)
                if len(self.envelopes) == 1:
                    return '{"key_findings": []}'
                return json.dumps({"key_findings": [{"text": "Verified conclusion", "source_ids": ["T1-M1"]}]})

            def run(self, prompt, **kwargs):
                raise AssertionError("retry bypassed the prompt envelope")

            def clear_history(self):
                pass

        agent = CaptureAgent()
        service = SummarizationService(lambda: agent, self.config)
        task = TodoItem(id=1, title="Task", intent="Verify", query="query", evidence_items=[{"source_id": "T1-M1"}])
        result = service.summarize_task(SummaryState(research_topic="Research"), task, "original evidence")
        self.assertEqual(agent.envelopes[0].system_text(), "summarizer role\n\n" + agent.envelopes[0].system_segments[-1].text)
        self.assertTrue(agent.envelopes[1].user_text().startswith(agent.envelopes[0].user_text()))
        self.assertEqual(agent.envelopes[1].prefix_fingerprint, agent.envelopes[0].prefix_fingerprint)
        self.assertIn("[T1-M1]", result.markdown)

    def test_optional_model_fact_fields_are_grounded_and_default_to_deterministic(self) -> None:
        from services.summarizer import SummarizationService

        finding = {"text": "Engine latency is 10 ms", "source_ids": ["T1-M1"]}
        raw = {"key_findings": [finding], "fact_claims": [{
            "finding_index": 1, "entity": "Engine", "metric": "p99 latency", "value": "10",
            "unit": "ms", "period": "2025", "source_ids": ["T1-M1", "T2-C1"],
        }]}
        task = self._fact_task(1, "10 ms", source_kind="M")
        disabled = SummarizationService(lambda: None, self.config)
        self.assertFalse(self.config.fact_ledger_llm_compaction_enabled)
        self.assertEqual(disabled._build_fact_claims(task, [finding], raw)[0]["metric"], "latency")
        enabled = SummarizationService(lambda: None, self.config.model_copy(update={
            "fact_ledger_llm_compaction_enabled": True,
        }))
        facts = enabled._build_fact_claims(task, [finding], raw)
        self.assertEqual(facts[0]["metric"], "p99 latency")
        self.assertEqual(facts[0]["source_ids"], ["T1-M1"])
        raw["fact_claims"][0]["source_ids"] = ["T2-C1"]
        self.assertEqual(enabled._build_fact_claims(task, [finding], raw)[0]["metric"], "latency")

    def test_report_prefix_excludes_multiline_topic_and_sync_filters_all_citations(self) -> None:
        class MarkdownAgent:
            def run(self, prompt, **kwargs):
                return "Preserve [T1-C1], remove [T2-M2] and [T3-S3]."

            def clear_history(self):
                pass

        task = TodoItem(id=1, title="Task", intent="Verify", query="query", evidence_items=[{"source_id": "T1-C1"}])
        state = SummaryState(research_topic="Test topic\nPrivate line", todo_items=[task])
        service = ReportingService(MarkdownAgent(), self.config, evidence_store=EvidenceStore())
        first = service._build_envelope(state)
        state.research_topic = "Different topic"
        second = service._build_envelope(state)
        self.assertEqual(first.system_fingerprint, second.system_fingerprint)
        self.assertNotIn("Private line", first.system_text())
        result = service.generate_report(state)
        self.assertIn("[T1-C1]", result)
        self.assertNotIn("[T2-M2]", result)
        self.assertNotIn("[T3-S3]", result)

    def test_planning_and_reflection_put_schema_before_date_or_tasks(self) -> None:
        from unittest.mock import patch

        from services.planner import PlanningService
        from services.reflection import ReflectionService

        class CaptureAgent:
            system_prompt = "static role"

            def run_envelope(self, envelope, **kwargs):
                self.envelope = envelope
                return '{"tasks": []}'

            def clear_history(self):
                pass

        planner_agent = CaptureAgent()
        planner = PlanningService(planner_agent, self.config)
        state = SummaryState(research_topic="Private topic")
        with patch("services.planner.get_current_date", return_value="date-one"):
            planner.plan_todo_list(state)
        first = planner_agent.envelope
        with patch("services.planner.get_current_date", return_value="date-two"):
            planner.plan_todo_list(state)
        second = planner_agent.envelope
        self.assertIn("<FORMAT>", first.system_text())
        self.assertNotIn("date-one", first.system_text())
        self.assertEqual(first.prefix_fingerprint, second.prefix_fingerprint)
        self.assertIn("date-two", second.volatile_segments[0].text)

        task = TodoItem(id=1, title="Task", intent="Verify", query="query", summary="Distinct legacy evidence")
        state.todo_items = [task]
        reflection = ReflectionService(CaptureAgent(), self.config)
        context = reflection._build_reflection_context(state, gap_signals=[])
        envelope = reflection._build_envelope(state, context)
        self.assertIn("<FORMAT>", envelope.system_text())
        self.assertEqual(envelope.user_text().count("Distinct legacy evidence"), 1)

    def test_compact_review_hydrates_flagged_sources_and_keeps_uncovered_tasks(self) -> None:
        from services.reviewer import ReviewIssue, ReviewService

        class Retrieval:
            def get_source_chunks(self, *, source_ids):
                self.source_ids = source_ids
                return [SimpleNamespace(source_id="T1-M1", chunks=[{"text": "disputed raw figure"}])]

            def query(self, *args, **kwargs):
                return SimpleNamespace(hits=[])

        tasks = [
            TodoItem(id=1, title="Covered", intent="Verify", query="query", status="completed",
                     summary="covered legacy prose", evidence_items=[{"source_id": "T1-M1"}]),
            TodoItem(id=2, title="Uncovered", intent="Verify", query="query", status="completed",
                     summary="uncovered legacy prose", evidence_items=[{"source_id": "T2-C1"}]),
        ]
        ledger = {
            "entries": [{"claim_id": "C-001", "task_id": 1, "statement": "compact fact", "source_ids": ["T1-M1"]}],
        }
        state = SummaryState(research_topic="test", todo_items=tasks, fact_ledger=ledger)
        retrieval = Retrieval()
        reviewer = ReviewService(None, self.config, retrieval_service=retrieval)
        issues = [ReviewIssue(1, "medium", "numeric_audit_mismatch", "disputed", ["T1-M1"])]
        envelope = reviewer._build_envelope(state, issues)
        self.assertEqual(retrieval.source_ids, ["T1-M1"])
        self.assertIn("[T1-M1] disputed raw figure", envelope.user_text())
        self.assertNotIn("covered legacy prose", envelope.user_text().replace("uncovered legacy prose", ""))
        self.assertIn("uncovered legacy prose", envelope.user_text())

        ledger["entries"].append({
            "claim_id": "C-002", "task_id": 1, "statement": "omitted fact", "source_ids": ["T1-M1"],
        })
        capped_config = self.config.model_copy(update={"fact_ledger_max_entries": 1})
        capped_prompt = ReviewService(None, capped_config)._build_prompt(state, issues)
        self.assertIn("covered legacy prose", capped_prompt.replace("uncovered legacy prose", ""))

    def test_numeric_review_does_not_match_the_tail_of_another_figure(self) -> None:
        from services.reviewer import _number_token_supported_in_text

        self.assertFalse(_number_token_supported_in_text("24%", "actual reduction: 124%"))
        self.assertFalse(_number_token_supported_in_text("24%", "actual reduction: 24.5%"))
        self.assertFalse(_number_token_supported_in_text("24ms", "actual latency: 24 seconds"))
        self.assertTrue(_number_token_supported_in_text("24%", "actual reduction: 24.0 %"))

class Direction03CacheTests(unittest.TestCase):
    def test_anthropic_usage_includes_cache_in_prompt_and_bills_creation_once(self):
        from agent import SafeHelloAgentsLLM

        raw_usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=800,
            cache_creation_input_tokens=200,
        )
        usage = SafeHelloAgentsLLM._extract_chunk_usage(SimpleNamespace(usage=raw_usage))
        self.assertEqual(usage["prompt_tokens"], 1100)
        self.assertEqual(usage["total_tokens"], 1120)

        recorded = build_token_usage(
            prompt_text="ignored",
            completion_text="ignored",
            provider="anthropic",
            model="demo",
            pricing_catalog={
                "anthropic": {"demo": {
                    "prompt_per_1k_tokens": 1.0,
                    "cached_prompt_per_1k_tokens": 0.1,
                    "cache_creation_per_1k_tokens": 1.25,
                    "completion_per_1k_tokens": 2.0,
                }}
            },
            usage=vars(raw_usage),
        )
        self.assertEqual(recorded["uncached_prompt_tokens"], 300)
        self.assertEqual(recorded["prompt_cache_hit_ratio"], 0.7273)
        self.assertAlmostEqual(recorded["estimated_cost"], 0.47)

    def test_stream_usage_combines_partial_frames_and_ignores_malformed_fields(self):
        llm = self._make_llm(
            [
                SimpleNamespace(choices=[], usage={"input_tokens": 8, "cache_read_input_tokens": 40}),
                SimpleNamespace(choices=[], usage={"output_tokens": 5, "cache_creation_input_tokens": "bad"}),
            ]
        )
        llm.invoke([{"role": "user", "content": "test"}])
        self.assertEqual(llm.consume_last_usage(), {
            "prompt_tokens": 48, "completion_tokens": 5, "total_tokens": 53, "cached_tokens": 40,
        })

    def test_cache_isolation_survives_plain_message_fallback_without_volatile_history(self):
        from agent import CacheAwareToolAgent

        config = Configuration(kv_cache_min_segment_tokens=0, kv_cache_prompt_cache_key_enabled=False)
        builder = ContextPrefixBuilder(config)
        env = builder.build_envelope(
            stage="report", tenant_id="a", request_id="r", system_role="role",
            research_topic="shared", volatile_blocks=[("date", "2026-11-25")],
        )
        next_request = builder.build_envelope(
            stage="report", tenant_id="a", request_id="another", system_role="role",
            research_topic="shared", volatile_blocks=[("date", "2026-11-26")],
        )
        self.assertEqual(env.prefix_fingerprint, next_request.prefix_fingerprint)
        self.assertNotEqual(
            ProviderCacheAdapter.build_tenant_cache_key(env),
            ProviderCacheAdapter.build_tenant_cache_key(next_request),
        )
        rendered, _ = ProviderCacheAdapter.render_messages_and_kwargs(env, provider_mode="openai", config=config)
        other, _ = ProviderCacheAdapter.render_messages_and_kwargs(next_request, provider_mode="openai", config=config)
        self.assertEqual(rendered[0], other[0])
        self.assertNotEqual(rendered[1]["content"].splitlines()[0], other[1]["content"].splitlines()[0])
        self.assertNotIn("another", other[1]["content"])
        self.assertNotIn("2026-11-26", other[0]["content"])

        stage = CacheAwareToolAgent.__new__(CacheAwareToolAgent)
        stage.system_prompt = "role"
        stage.enable_tool_calling = False
        stage._history = [SimpleNamespace(role="user", content="previous volatile task")]
        messages, _ = stage._prepare_envelope_messages(env, config=config)
        self.assertNotIn("previous volatile task", str(messages))
        self.assertEqual(messages, rendered)

    def test_global_stage_metrics_weight_actual_prompt_tokens(self):
        metrics_registry.reset()
        for request, prompt, cached in (("a", 100, 100), ("b", 900, 0)):
            trace = RequestTrace(request_id=request, topic="topic", search_api="test", provider="openai", model="demo", pricing_catalog=None)
            trace.record_llm_call(
                success=True, prompt_text="", completion_text="",
                usage={"prompt_tokens": prompt, "cached_tokens": cached}, stage="report",
            )
        snap = metrics_registry.snapshot()
        self.assertEqual(snap["prompt_cache_hit_ratio"], 0.1)
        self.assertEqual(snap["kv_cache_by_stage"]["report"]["calls"], 2)
        self.assertEqual(snap["kv_cache_by_stage"]["report"]["cache_hit_ratio"], 0.1)

    def test_envelope_accounting_includes_tool_iterations(self):
        from agent import CacheAwareToolAgent

        class LLM:
            def __init__(self):
                self.usage = None
                self.index = 0

            def invoke(self, messages, **kwargs):
                self.index += 1
                self.usage = {
                    "prompt_tokens": self.index * 100,
                    "completion_tokens": 10,
                    "cached_tokens": 150 if self.index == 2 else 0,
                }
                return "call tool" if self.index == 1 else "done"

            def consume_last_usage(self):
                usage, self.usage = self.usage, None
                return usage

        stage = CacheAwareToolAgent.__new__(CacheAwareToolAgent)
        stage.llm = LLM()
        stage.system_prompt = "role"
        stage.enable_tool_calling = True
        stage.tool_registry = None
        stage._parse_tool_calls = lambda response: (
            [{"tool_name": "test", "parameters": "{}"}] if response == "call tool" else []
        )
        stage._execute_tool_call = lambda name, parameters: "data"
        envelope = ContextPrefixBuilder(Configuration()).build_envelope(stage="report", research_topic="topic")
        self.assertEqual(stage.run_envelope(envelope), "done")
        usage = stage.consume_last_usage()
        self.assertEqual(usage["prompt_tokens"], 300)
        self.assertEqual(usage["cached_tokens"], 150)
        self.assertEqual(usage["provider_calls"], 2)
        self.assertIsNone(stage.consume_last_usage())

    def test_cache_rejection_downgrades_once_without_retrying_stream_usage(self):
        class Completions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if "prompt_cache_key" in kwargs:
                    raise RuntimeError("extra_forbidden: prompt_cache_key")
                return []

        fake = Completions()
        llm = self._make_llm([])
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        messages = [{"role": "user", "content": "test"}]
        llm.invoke(messages, prompt_cache_key="test")
        self.assertTrue(llm.consume_cache_downgrade())
        llm.invoke(messages, prompt_cache_key="test")
        self.assertFalse(llm.consume_cache_downgrade())
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(all("stream_options" in call for call in fake.calls))

    @staticmethod
    def _make_llm(chunks):
        from agent import SafeHelloAgentsLLM, local

        llm = SafeHelloAgentsLLM.__new__(SafeHelloAgentsLLM)
        llm.model = "demo"
        llm.provider = "test"
        llm.temperature = 0.0
        llm.max_tokens = 64
        llm.allow_reasoning_fallback = False
        llm._usage_local = local()
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: chunks)))
        return llm

class ScopedToolPrefixTests(unittest.TestCase):
    def test_scoped_tools_are_after_request_scope_and_plain_global_prompt_can_share(self):
        config = Configuration()
        builder = ContextPrefixBuilder(config)
        first = builder.build_envelope(stage="review", tenant_id="A", request_id="one", system_role="role", cache_key_scope="global")
        second = builder.build_envelope(stage="review", tenant_id="B", request_id="one", system_role="role", cache_key_scope="global")
        expected_global_key = ProviderCacheAdapter.build_tenant_cache_key(first)
        self.assertEqual(ProviderCacheAdapter.build_tenant_cache_key(second), expected_global_key)
        plain_first, _ = ProviderCacheAdapter.render_messages_and_kwargs(
            first, provider_mode="openai", request_tool_section="private tool", config=config,
        )
        plain_second, _ = ProviderCacheAdapter.render_messages_and_kwargs(
            second, provider_mode="openai", request_tool_section="private tool", config=config,
        )
        self.assertEqual(plain_first[0], plain_second[0])
        self.assertNotEqual(plain_first[1], plain_second[1])
        self.assertNotIn("private tool", plain_first[0]["content"])
        _, options = ProviderCacheAdapter.render_messages_and_kwargs(
            first, provider_mode="openai", request_tool_section="private tool", config=config,
        )
        self.assertNotEqual(options["prompt_cache_key"], expected_global_key)

class CacheProviderModeTests(unittest.TestCase):
    def test_auto_mode_uses_stage_model_and_endpoint(self):
        from agent import CacheAwareToolAgent

        config = Configuration(llm_provider="openai", kv_cache_min_segment_tokens=0)
        agent = CacheAwareToolAgent.__new__(CacheAwareToolAgent)
        agent.llm = SimpleNamespace(
            base_url="https://api.anthropic.com/v1",
            provider="custom",
            model="claude-sonnet-4-5",
        )
        agent.system_prompt = "role"
        agent.enable_tool_calling = False
        envelope = ContextPrefixBuilder(config).build_envelope(stage="reflection", research_topic="topic")
        messages, kwargs = agent._prepare_envelope_messages(envelope, config=config)
        self.assertNotIn("prompt_cache_key", kwargs)
        self.assertEqual(messages[0]["content"][0]["cache_control"], {"type": "ephemeral"})

if __name__ == "__main__":
    unittest.main()
