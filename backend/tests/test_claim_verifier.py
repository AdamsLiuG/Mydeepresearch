from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for root in (SRC_ROOT, BACKEND_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from config import Configuration
from evals.judges.hybrid import HybridJudge
from evals.metrics.factscore import (
    compute_atomic_factscore,
    summarize_conflict_resolution,
)
from evals.report_metrics import MetricRecord, aggregate_records
from evals.schema import BenchmarkCase
from main import _warmup_nli_model, create_app
from metrics import RequestTrace
from models import SummaryState, SummaryStateOutput, TodoItem
from services.claim_verifier import (
    AtomicClaimDecomposer,
    AtomicClaimRecord,
    AtomicProposition,
    ClaimVerificationOrchestrator,
    ConflictGraphBuilder,
    LoadedNLIModel,
    LocalNLIScorer,
    ScopedEvidenceRetriever,
    _resolve_label_mapping,
    clear_nli_model_cache,
    load_nli_model,
)
from services.evidence_index import EvidenceRetrievalService, EvidenceRuntimeIndex
from services.reporter import ReportingService
from services.request_state import RequestStateStore
from services.reviewer import ReviewService
from state_manager import StateManagerMixin


class _MockLLM:
    def __init__(self, content: str, *, raise_exc: Exception | None = None) -> None:
        self.content = content
        self.raise_exc = raise_exc
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> SimpleNamespace:
        self.calls.append(messages)
        if self.raise_exc is not None:
            raise self.raise_exc
        return SimpleNamespace(content=self.content)


class _FakeEvidenceStore:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = list(records)

    def lookup(self, *, source_ids=None, include_full_content: bool = False):  # noqa: ANN001
        if source_ids is None:
            return list(self._records)
        wanted = set(source_ids)
        return [r for r in self._records if r.get("source_id") in wanted]

    def build_reference_map(self, source_ids=None):  # noqa: ANN001
        wanted = set(source_ids) if source_ids is not None else None
        return [
            {
                "source_id": str(r.get("source_id") or ""),
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "domain": str(r.get("domain") or ""),
            }
            for r in self._records
            if r.get("source_id") and (wanted is None or r.get("source_id") in wanted)
        ]

    def list_task_evidence(self, task_id: int) -> list[dict[str, Any]]:
        return [r for r in self._records if int(r.get("task_id") or 0) == int(task_id)]


class _PersistedStateHost(StateManagerMixin):
    def __init__(self, config: Configuration, request_id: str = "req-roundtrip") -> None:
        self.config = config
        self.request_id = request_id
        self._request_trace = None
        self._request_state_store = RequestStateStore(config.request_state_dir)
        self._evidence_store = SimpleNamespace(
            hydrate_from_tasks=lambda items: None,
            list_task_evidence=lambda task_id: [],
        )

    @staticmethod
    def _refresh_note_memory_for_state(state: SummaryState, observer=None) -> None:  # noqa: ANN001
        return None

    @staticmethod
    def _serialize_task(task: TodoItem) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "notices": list(task.notices or []),
            "evidence_items": list(task.evidence_items or []),
            "claims": list(task.claims or []),
            "atomic_claims": list(task.atomic_claims or []),
            "review_issues": list(task.review_issues or []),
            "review_status": task.review_status,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "origin": task.origin,
            "round": task.round,
        }


def test_atomic_decomposer_subset_citation_conservation_and_fallback() -> None:
    payload = {
        "claims": [
            {
                "parent_index": 0,
                "atomics": [
                    {
                        "text": "Acme 2025年营收达到120亿元",
                        "source_ids": ["T1-S1", "T99-HALLUCINATED"],
                        "proposition": {
                            "entity": "Acme",
                            "metric": "revenue",
                            "period": "2025",
                            "numeric_value": 120e8,
                            "unit": "currency",
                            "polarity": "positive",
                        },
                    },
                    {
                        "text": "Acme 毛利率提升至42%",
                        "source_ids": ["T1-S2"],
                    },
                ],
            }
        ]
    }
    decomposer = AtomicClaimDecomposer(decompose_with_fast_llm=True)
    parent_claims = [
        {
            "text": "Acme 2025年营收达到120亿元，同时毛利率提升至42%。",
            "source_ids": ["T1-S1", "T1-S2"],
        }
    ]
    records = decomposer.decompose(
        task_id=1,
        parent_claims=parent_claims,
        valid_source_ids={"T1-S1", "T1-S2", "T2-S1"},
        fast_llm=_MockLLM(json.dumps(payload, ensure_ascii=False)),
        max_atomic_per_task=10,
    )
    assert len(records) == 2
    assert records[0].source_ids == ["T1-S1"]
    assert records[1].source_ids == ["T1-S2"]
    assert records[0].proposition.entity == "acme"
    assert records[0].proposition.metric == "revenue"
    assert records[0].proposition.period == "2025"

    # Non-JSON / failing LLM falls back deterministically to syntactic splitting
    fallback_records = decomposer.decompose(
        task_id=1,
        parent_claims=[
            {
                "text": "Beta 2025年收入为80亿元 [T1-S1]；同时其净利率下降至12% [T1-S2]",
                "source_ids": ["T1-S1", "T1-S2"],
            }
        ],
        valid_source_ids={"T1-S1", "T1-S2"},
        fast_llm=_MockLLM("not json", raise_exc=RuntimeError("llm down")),
        max_atomic_per_task=10,
    )
    assert len(fallback_records) == 2
    assert fallback_records[0].source_ids == ["T1-S1"]
    assert fallback_records[1].source_ids == ["T1-S2"]


def test_scoped_evidence_retriever_isolation_and_cross_task_wave_recall() -> None:
    rt_cfg = Configuration(
        evidence_chunk_chars=300,
        evidence_chunk_overlap=40,
        evidence_runtime_max_chunks=8,
    )
    index = EvidenceRuntimeIndex(rt_cfg)
    index.on_records_changed(
        task_id=1,
        records=[
            SimpleNamespace(
                source_id="T1-S1",
                task_id=1,
                title="Alpha 官方年报",
                url="https://alpha.example.com/report",
                domain="alpha.example.com",
                snippet="Alpha 2025年营业收入实现150亿元。",
                full_content="Alpha 2025年营业收入实现150亿元，同比增长25%。",
                quality_label="high",
                freshness_label="recent",
            ),
            SimpleNamespace(
                source_id="T1-S2",
                task_id=1,
                title="干扰来源",
                url="https://noise.example.com/rumor",
                domain="noise.example.com",
                snippet="Alpha 2025年营业收入传闻高达999亿元。",
                full_content="Alpha 2025年营业收入传闻高达999亿元。",
                quality_label="low",
                freshness_label="recent",
            ),
        ],
        reason="search",
    )
    retrieval = EvidenceRetrievalService(runtime_index=index, memory_service=None, config=rt_cfg)
    retriever = ScopedEvidenceRetriever()

    # Cross-task citation (task 2 cites T1-S1 only): must never recall T1-S2
    atomic = AtomicClaimRecord(
        atomic_id="2-C0-A0",
        task_id=2,
        parent_claim_index=0,
        text="Alpha 2025年营业收入实现150亿元",
        source_ids=["T1-S1"],
    )
    hits = retriever.fetch_for_claim(
        atomic,
        retrieval_service=retrieval,
        evidence_store=None,
        cited_records_fallback=[],
        top_k_per_source=2,
    )
    assert hits
    assert all(h.source_id == "T1-S1" for h in hits)
    assert all(h.origin == "current" for h in hits)
    assert all("999亿元" not in h.text for h in hits)


def test_label_mapping_load_singleton_and_offline_guard(monkeypatch, tmp_path: Path) -> None:
    clear_nli_model_cache()
    try:
        # Ordering A: 0=contradiction, 1=entailment, 2=neutral
        map_a = _resolve_label_mapping({0: "contradiction", 1: "entailment", 2: "neutral"})
        # Ordering B: 0=ENTAILMENT, 1=NEUTRAL, 2=CONTRADICTION
        map_b = _resolve_label_mapping({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"})
        assert map_a == {"contradiction": 0, "entailment": 1, "neutral": 2}
        assert map_b == {"entailment": 0, "neutral": 1, "contradiction": 2}

        scorer = LocalNLIScorer(Configuration())
        probs_a = scorer._logits_to_prob_dict([0.1, 3.2, 0.2], map_a)
        probs_b = scorer._logits_to_prob_dict([3.2, 0.2, 0.1], map_b)
        assert probs_a["entailment"] > 0.75
        assert abs(probs_a["entailment"] - probs_b["entailment"]) < 1e-6

        # Uncached model with allow_download=False returns None cleanly
        assert (
            load_nli_model(
                "nonexistent/offline-model-xyz",
                allow_download=False,
            )
            is None
        )

        # Fake CrossEncoder on an existing local directory caches as process singleton
        local_model_dir = tmp_path / "local-nli"
        local_model_dir.mkdir()
        init_calls: list[str] = []

        class _FakeCrossEncoder:
            def __init__(self, name: str, **kwargs: Any) -> None:
                init_calls.append(name)
                self.model = SimpleNamespace(
                    config=SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
                )

            def predict(self, batch: list[tuple[str, str]], **kwargs: Any) -> list[list[float]]:
                return [[0.1, 3.0, 0.2] for _ in batch]

        import services.claim_verifier as cv_mod

        monkeypatch.setattr(cv_mod, "CrossEncoder", _FakeCrossEncoder)
        m1 = load_nli_model(str(local_model_dir), backend="cross_encoder", device="cpu", allow_download=False)
        m2 = load_nli_model(str(local_model_dir), backend="cross_encoder", device="cpu", allow_download=False)
        assert m1 is not None
        assert m1 is m2
        assert len(init_calls) == 1
    finally:
        clear_nli_model_cache()


def test_nli_budget_timeout_degrades_cleanly_to_numeric_fallback() -> None:
    cfg = Configuration(
        nli_verifier_enabled=True,
        nli_stage_budget_seconds=0.1,
        nli_conflict_graph_enabled=False,
    )

    def _slow_predict(batch: list[tuple[str, str]]) -> list[list[float]]:
        time.sleep(0.25)
        return [[0.0, 3.0, 0.0] for _ in batch]

    slow_model = LoadedNLIModel(
        backend="mock_slow",
        model_name="slow",
        label_mapping={"contradiction": 0, "entailment": 1, "neutral": 2},
        predict_logits_fn=_slow_predict,
    )
    orchestrator = ClaimVerificationOrchestrator(
        cfg,
        scorer=LocalNLIScorer(cfg, loaded_model=slow_model),
        decomposer=AtomicClaimDecomposer(decompose_with_fast_llm=False),
    )
    state = SummaryState(
        research_topic="Test",
        todo_items=[
            TodoItem(
                id=1,
                title="财务指标核查",
                intent="核查营收",
                query="revenue",
                evidence_items=[
                    {
                        "source_id": "T1-S1",
                        "title": "财报",
                        "url": "https://example.com/1",
                        "domain": "example.com",
                        "snippet": "2025年营收为100亿元。",
                    }
                ],
            )
        ],
    )
    fallback_called: list[str] = []

    def _fallback(text: str, records: list[dict[str, Any]]) -> str | None:
        fallback_called.append(text)
        return "数值不匹配: 100 vs 200"

    observer = RequestTrace(
        request_id="req-timeout",
        topic="Test",
        search_api="duckduckgo",
        provider="ollama",
        model="llama3.2",
        pricing_catalog={},
    )
    out = orchestrator.verify_request(
        state,
        claims_by_task={1: [{"text": "Acme 2025年营收为200亿元", "source_ids": ["T1-S1"], "support_status": "supported"}]},
        observer=observer,
        fallback_claim_verifier=_fallback,
    )
    assert out["summary"]["degraded"] is True
    assert out["summary"]["skipped_count"] >= 1
    assert state.todo_items[0].atomic_claims[0]["nli_label"] == "SKIPPED"
    assert fallback_called
    assert any(i["origin"] == "claim_verifier:numeric_fallback" for i in out["issues"])
    assert observer.snapshot()["claim_verification_summary"]["degraded"] is True


def test_conflict_graph_numeric_semantic_and_adjudication_modes() -> None:
    builder = ConflictGraphBuilder(
        Configuration(nli_conflict_numeric_rel_tol=0.05, nli_conflict_min_score=0.70)
    )
    evidence_by_id = {
        "T1-S1": {
            "source_id": "T1-S1",
            "title": "国家统计局公报",
            "domain": "stats.gov.cn",
            "source_type": "government",
            "source_reliability_score": 9.8,
            "published_at": "2025-03-01",
            "snippet": "2025年营收100亿元。",
        },
        "T2-S1": {
            "source_id": "T2-S1",
            "title": "普通论坛帖子",
            "domain": "bbs.example.com",
            "source_type": "forum_general",
            "source_reliability_score": 3.0,
            "published_at": "2025-03-02",
            "snippet": "2025年营收140亿元。",
        },
        "T1-S2": {
            "source_id": "T1-S2",
            "title": "同行A新闻",
            "domain": "news1.example.com",
            "source_type": "news_primary",
            "source_reliability_score": 7.5,
            "published_at": "2025-01-10",
            "snippet": "Beta 2025年毛利率20%。",
        },
        "T2-S2": {
            "source_id": "T2-S2",
            "title": "同行B新闻",
            "domain": "news2.example.com",
            "source_type": "news_primary",
            "source_reliability_score": 7.5,
            "published_at": "2025-04-15",
            "snippet": "Beta 2025年毛利率28%。",
        },
        "T1-S3": {
            "source_id": "T1-S3",
            "title": "官方机构A",
            "domain": "org1.gov",
            "source_type": "official_documentation",
            "source_reliability_score": 9.0,
            "published_at": "2025-05-01",
            "snippet": "Gamma 已正式生产交付。",
        },
        "T2-S3": {
            "source_id": "T2-S3",
            "title": "官方机构B",
            "domain": "org2.org",
            "source_type": "official_documentation",
            "source_reliability_score": 9.0,
            "published_at": "2025-05-02",
            "snippet": "Gamma 尚未生产交付，仍处于内部测试。",
        },
    }
    atomics = [
        # Cluster 1: authority resolution (government vs forum_general)
        AtomicClaimRecord(
            atomic_id="1-C0-A0",
            task_id=1,
            parent_claim_index=0,
            text="Alpha 2025年营收为100亿元",
            source_ids=["T1-S1"],
            best_source_id="T1-S1",
            nli_label="ENTAILED",
            raw_probs={"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05},
            calibrated_confidence=0.9,
            proposition=AtomicProposition(entity="alpha", metric="revenue", period="2025", numeric_value=100.0, unit="currency"),
        ),
        AtomicClaimRecord(
            atomic_id="2-C0-A0",
            task_id=2,
            parent_claim_index=0,
            text="Alpha 2025年营收为140亿元",
            source_ids=["T2-S1"],
            best_source_id="T2-S1",
            nli_label="ENTAILED",
            raw_probs={"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10},
            calibrated_confidence=0.85,
            proposition=AtomicProposition(entity="alpha", metric="revenue", period="2025", numeric_value=140.0, unit="currency"),
        ),
        # Cluster 2: recency resolution (same tier, Jan vs Apr)
        AtomicClaimRecord(
            atomic_id="1-C1-A0",
            task_id=1,
            parent_claim_index=1,
            text="Beta 2025年毛利率为20%",
            source_ids=["T1-S2"],
            best_source_id="T1-S2",
            nli_label="ENTAILED",
            raw_probs={"entailment": 0.85},
            calibrated_confidence=0.85,
            proposition=AtomicProposition(entity="beta", metric="gross_margin", period="2025", numeric_value=20.0, unit="%"),
        ),
        AtomicClaimRecord(
            atomic_id="2-C1-A0",
            task_id=2,
            parent_claim_index=1,
            text="Beta 2025年毛利率为28%",
            source_ids=["T2-S2"],
            best_source_id="T2-S2",
            nli_label="ENTAILED",
            raw_probs={"entailment": 0.85},
            calibrated_confidence=0.85,
            proposition=AtomicProposition(entity="beta", metric="gross_margin", period="2025", numeric_value=28.0, unit="%"),
        ),
        # Cluster 3: semantic contradiction with unresolved_divergence (same tier & month)
        AtomicClaimRecord(
            atomic_id="1-C2-A0",
            task_id=1,
            parent_claim_index=2,
            text="Gamma 已正式生产交付",
            source_ids=["T1-S3"],
            best_source_id="T1-S3",
            nli_label="ENTAILED",
            raw_probs={"entailment": 0.85},
            calibrated_confidence=0.85,
            proposition=AtomicProposition(entity="gamma", metric="delivery", period="2025", polarity="positive"),
        ),
        AtomicClaimRecord(
            atomic_id="2-C2-A0",
            task_id=2,
            parent_claim_index=2,
            text="Gamma 尚未生产交付",
            source_ids=["T2-S3"],
            best_source_id="T2-S3",
            nli_label="ENTAILED",
            raw_probs={"entailment": 0.85},
            calibrated_confidence=0.85,
            proposition=AtomicProposition(entity="gamma", metric="delivery", period="2025", polarity="negative"),
        ),
    ]
    clusters = builder.build(atomics, evidence_by_id=evidence_by_id)
    by_entity = {c.entity: c for c in clusters}
    assert set(by_entity) == {"alpha", "beta", "gamma"}

    assert by_entity["alpha"].conflict_type == "numeric_divergence"
    assert by_entity["alpha"].adjudication_status == "resolved_by_authority"
    assert by_entity["alpha"].recommended_source_id == "T1-S1"

    assert by_entity["beta"].adjudication_status == "resolved_by_recency"
    assert by_entity["beta"].recommended_source_id == "T2-S2"

    assert by_entity["gamma"].conflict_type == "semantic_contradiction"
    assert by_entity["gamma"].adjudication_status == "unresolved_divergence"
    assert by_entity["gamma"].recommended_source_id is None


def test_review_service_single_track_repair_mapping_and_state_roundtrip(tmp_path: Path) -> None:
    cfg = Configuration(
        nli_verifier_enabled=True,
        request_state_enabled=True,
        request_state_dir=str(tmp_path / "states"),
        notes_workspace=str(tmp_path / "notes"),
    )
    legacy_calls: list[str] = []

    def _legacy_claim_verifier(claim_text: str, cited: list[dict[str, Any]]) -> tuple[bool, str] | None:
        legacy_calls.append(claim_text)
        return True, ""

    def _predict_nli(batch: list[tuple[str, str]]) -> list[list[float]]:
        out: list[list[float]] = []
        for premise, hyp in batch:
            if "40%" in hyp and "15%" in premise:
                out.append([3.5, 0.1, 0.1])  # CONTRADICTED
            else:
                out.append([0.1, 3.2, 0.2])  # ENTAILED
        return out

    fake_model = LoadedNLIModel(
        backend="mock_nli",
        model_name="mock-nli",
        label_mapping={"contradiction": 0, "entailment": 1, "neutral": 2},
        predict_logits_fn=_predict_nli,
    )
    evidence_items = [
        {
            "source_id": "T1-S1",
            "task_id": 1,
            "title": "来源A",
            "url": "https://a.example.com",
            "domain": "a.example.com",
            "source_type": "official_documentation",
            "source_reliability_score": 8.0,
            "published_at": "2025-05-01",
            "full_content": "Acme 2025年市场份额仅为15%。Acme 在华东市场营收为100亿元。",
        },
        {
            "source_id": "T2-S1",
            "task_id": 2,
            "title": "来源B",
            "url": "https://b.example.com",
            "domain": "b.example.com",
            "source_type": "official_documentation",
            "source_reliability_score": 8.0,
            "published_at": "2025-05-01",
            "full_content": "Acme 在华东市场营收为130亿元。",
        },
    ]
    state = SummaryState(
        research_topic="Acme 调研",
        todo_items=[
            TodoItem(
                id=1,
                title="任务一",
                intent="调研",
                query="q1",
                status="completed",
                summary="- Acme 2025年市场份额达到40% [T1-S1]\n- Acme 在华东市场营收为100亿元 [T1-S1]",
                evidence_items=[evidence_items[0]],
            ),
            TodoItem(
                id=2,
                title="任务二",
                intent="对比",
                query="q2",
                status="completed",
                summary="- Acme 在华东市场营收为130亿元 [T2-S1]",
                evidence_items=[evidence_items[1]],
            ),
        ],
    )
    store = _FakeEvidenceStore(evidence_items)
    orchestrator = ClaimVerificationOrchestrator(
        cfg,
        evidence_store=store,
        scorer=LocalNLIScorer(cfg, loaded_model=fake_model),
        decomposer=AtomicClaimDecomposer(decompose_with_fast_llm=False),
    )
    reviewer = ReviewService(
        None,
        cfg,
        claim_verifier=_legacy_claim_verifier,
        claim_verification_orchestrator=orchestrator,
        evidence_store=store,
    )
    result = reviewer.review_request(state)
    # Single-track guarantee: per-claim callback was NOT invoked alongside orchestrator
    assert legacy_calls == []
    assert any(i.get("check") == "claim_contradicted" for i in result["issues"])
    assert any(i.get("check") == "cross_source_conflict" for i in result["issues"])
    assert any(c.get("check") == "invalid_citation" for c in result.get("repair_candidates", []))
    assert any(c.get("check") == "weak_evidence" for c in result.get("repair_candidates", []))

    assert result["claim_verification_summary"]["contradicted_count"] == 1
    assert result["evidence_conflict_count"] == 1
    assert result["unresolved_conflict_count"] == 1
    assert state.claim_verification_summary["contradicted_count"] == 1
    assert len(state.evidence_conflicts) == 1
    assert len(state.todo_items[0].atomic_claims) >= 1

    # Snapshot persistence roundtrip
    host = _PersistedStateHost(cfg, request_id="req-roundtrip")
    host._persist_request_state(
        state,
        phase="completed",
        status="completed",
    )
    loaded = host._request_state_store.load("req-roundtrip")
    assert loaded is not None
    restored, _ = host._state_from_snapshot(loaded)
    assert restored.evidence_conflicts == state.evidence_conflicts
    assert restored.claim_verification_summary == state.claim_verification_summary
    assert restored.todo_items[0].atomic_claims == state.todo_items[0].atomic_claims


def test_reporter_renders_conflict_section_with_reference_conservation() -> None:
    state = SummaryState(
        research_topic="多源分歧报告测试",
        todo_items=[
            TodoItem(
                id=1,
                title="核心财务分析",
                intent="分析收入",
                query="q1",
                status="completed",
                summary="某厂商 2025年营收达到 100亿元 [T1-S1]；另一机构预测为 140亿元 [T1-S2]。",
                evidence_items=[
                    {
                        "source_id": "T1-S1",
                        "title": "官方年报",
                        "url": "https://official.example.com/report",
                        "domain": "official.example.com",
                        "snippet": "2025年营收为100亿元。",
                    },
                    {
                        "source_id": "T1-S2",
                        "title": "第三方估算",
                        "url": "https://thirdparty.example.com/est",
                        "domain": "thirdparty.example.com",
                        "snippet": "2025年营收约为140亿元。",
                    },
                ],
            )
        ],
        evidence_conflicts=[
            {
                "conflict_id": "CONF-1234",
                "entity": "vendor",
                "metric": "revenue",
                "period": "2025",
                "conflict_type": "numeric_divergence",
                "severity": "low",
                "contradiction_score": 0.78,
                "sides": [
                    {
                        "source_id": "T1-S1",
                        "task_id": 1,
                        "title": "官方年报",
                        "url": "https://official.example.com/report",
                        "domain": "official.example.com",
                        "source_type": "official_documentation",
                        "quality_label": "high",
                        "published_at": "2025-03-01",
                        "claim_text": "营收 100亿元",
                        "normalized_value": 100.0,
                        "unit": "currency",
                    },
                    {
                        "source_id": "T1-S2",
                        "task_id": 1,
                        "title": "第三方估算",
                        "url": "https://thirdparty.example.com/est",
                        "domain": "thirdparty.example.com",
                        "source_type": "web_general",
                        "quality_label": "medium",
                        "published_at": "2025-02-20",
                        "claim_text": "营收 140亿元",
                        "normalized_value": 140.0,
                        "unit": "currency",
                    },
                    {
                        "source_id": "T9-ORPHAN",
                        "task_id": 9,
                        "title": "被过滤孤儿引用",
                        "url": "https://orphan.example.com",
                        "domain": "orphan.example.com",
                        "source_type": "web_general",
                        "quality_label": "low",
                        "published_at": None,
                        "claim_text": "不应保留",
                    },
                ],
                "adjudication_status": "resolved_by_authority",
                "recommended_source_id": "T1-S1",
                "adjudication_rationale": "优先采信官方年报 [T1-S1]。",
            }
        ],
    )

    payload = {
        "background_overview": "多源营收数据存在显著口径差异 [T1-S1]。",
        "key_findings": [{"claim": "官方年报显示营收为100亿元，第三方估算为140亿元 [T1-S1] [T1-S2]", "source_ids": ["T1-S1", "T1-S2"]}],
        "evidence_and_data": [{"point": "年报与第三方口径对照 [T1-S1] [T1-S2]", "source_ids": ["T1-S1", "T1-S2"]}],
        "dimension_sections": [],
        "risks_and_challenges": [{"risk": "统计口径不一导致估值偏差 [T1-S2]", "source_ids": ["T1-S2"]}],
    }

    for layout_mode in ("fixed", "flexible"):
        cfg = Configuration(use_tool_calling=False, report_layout_mode=layout_mode)
        reporter = ReportingService(
            None,
            cfg,
            evidence_store=_FakeEvidenceStore(state.todo_items[0].evidence_items),
        )
        rendered = reporter._render_structured_report(payload, state=state)
        assert "## 多源口径分歧与证据对照" in rendered
        assert "vendor / revenue / 2025" in rendered
        assert "[T1-S1]" in rendered
        assert "[T1-S2]" in rendered
        # Orphan citation not present in allowed references must never be leaked
        assert "T9-ORPHAN" not in rendered


def test_factscore_metrics_report_metrics_and_hybrid_judge() -> None:
    todo_items = [
        {
            "id": 1,
            "atomic_claims": [
                {"nli_label": "ENTAILED"},
                {"nli_label": "ENTAILED"},
                {"nli_label": "CONTRADICTED"},
                {"nli_label": "SKIPPED"},
            ],
        }
    ]
    fs = compute_atomic_factscore(todo_items)
    assert fs["evaluated_atomic_claims"] == 3
    assert fs["entailed_count"] == 2
    assert fs["contradicted_count"] == 1
    assert fs["skipped_count"] == 1
    assert abs(float(fs["atomic_factscore"]) - 0.6667) < 1e-4

    cs = summarize_conflict_resolution(
        [
            {"adjudication_status": "resolved_by_authority", "severity": "low"},
            {"adjudication_status": "unresolved_divergence", "severity": "high"},
        ]
    )
    assert cs["conflict_count"] == 2
    assert cs["resolved_by_authority"] == 1
    assert cs["unresolved_divergence"] == 1
    assert cs["resolution_rate"] == 0.5

    agg = aggregate_records(
        [
            MetricRecord(
                request_id="c1",
                status="success",
                phase="completed",
                elapsed_ms=100.0,
                cache_hits=0,
                cache_misses=0,
                cache_exact_hits=0,
                cache_semantic_hits=0,
                cache_approximate_hits=0,
                cache_approximate_dense_hits=0,
                cache_approximate_sparse_hits=0,
                cache_approximate_hybrid_hits=0,
                todo_items=[
                    {
                        "id": 1,
                        "status": "completed",
                        "claims": [
                            {"support_status": "supported"},
                            {"support_status": "contradicted"},
                        ],
                    }
                ],
                source_label="unit",
            )
        ]
    )
    assert agg["reviewed_claim_total"] == 2
    assert agg["citation_validity_rate"] == 0.5
    assert agg["claim_support_counts"]["contradicted"] == 1

    judge = HybridJudge()
    result = judge.evaluate(
        case=BenchmarkCase(id="case-nli", topic="Test"),
        report_markdown="## 执行摘要\n内容。\n\n## 参考来源\n1. [来源](https://a.example.com) `[T1-S1]`",
        todo_items=[],
        trace_snapshot={
            "claim_verification_summary": {
                "total_atomic_claims": 1,
                "atomic_faithfulness_rate": 0.91,
            }
        },
    )
    assert result["atomic_faithfulness_rate"] == 0.91


def test_nli_config_from_env_warmup_and_api_response(monkeypatch) -> None:
    monkeypatch.setenv("NLI_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("NLI_MODEL_BACKEND", "cross_encoder")
    monkeypatch.setenv("NLI_DEVICE", "cpu")
    monkeypatch.setenv("NLI_WARMUP_ENABLED", "false")
    cfg = Configuration.from_env()
    assert cfg.nli_verifier_enabled is True
    assert cfg.nli_model_backend == "cross_encoder"
    assert cfg.nli_device == "cpu"

    # Warmup helper safely skips when disabled or offline without raising
    _warmup_nli_model(Configuration(nli_verifier_enabled=True, nli_warmup_enabled=True, nli_model_name="nonexistent/offline"))

    class _ConflictStubAgent:
        def __init__(self, config: Configuration, **kwargs: Any) -> None:
            self.config = config

        def run(self, topic: str, **kwargs: Any) -> SummaryStateOutput:
            return SummaryStateOutput(
                report_markdown="## 执行摘要\n测试报告",
                todo_items=[
                    TodoItem(
                        id=1,
                        title="t1",
                        intent="i1",
                        query="q1",
                        status="completed",
                        atomic_claims=[{"atomic_id": "1-C0-A0", "nli_label": "ENTAILED"}],
                    )
                ],
                evidence_conflicts=[{"conflict_id": "CONF-API1", "adjudication_status": "resolved_by_authority"}],
                claim_verification_summary={"total_atomic_claims": 1, "entailed_count": 1, "atomic_faithfulness_rate": 1.0},
            )

    app = create_app(base_config=Configuration(), agent_factory=_ConflictStubAgent)
    with TestClient(app) as client:
        resp = client.post("/research", json={"topic": "NLI API check"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["evidence_conflicts"][0]["conflict_id"] == "CONF-API1"
        assert body["claim_verification_summary"]["atomic_faithfulness_rate"] == 1.0
        assert body["todo_items"][0]["atomic_claims"][0]["nli_label"] == "ENTAILED"
