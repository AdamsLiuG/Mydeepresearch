from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

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
from main import ResearchResponse, _warmup_nli_model, create_app
from metrics import RequestTrace, metrics_registry
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
from services.event_bus import RedisStreamEventBus
from services.evidence import EvidenceStore
from services.evidence_index import EvidenceRetrievalService, EvidenceRuntimeIndex
from services.reporter import ReportingService
from services.request_state import RequestStateStore, SnapshotConflictError
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


def test_enabled_claim_verification_omits_l0_with_forged_state_body():
    from unittest.mock import Mock

    config = Configuration.from_env(overrides={"nli_verifier_enabled": True}, load_env_file=False)
    store = EvidenceStore(web_extraction_enabled=True)
    evidence = store.record_search_results(task_id=1, query="test", backend="fixture", search_payload={
        "results": [{"url": "https://example.org", "content": "Processing time improved by 91%."}],
    })
    task = TodoItem(
        id=1, title="Benchmark", intent="Check latency", query="latency", status="completed",
        evidence_items=[{**evidence[0], "full_content": "Processing time improved by 91%."}],
    )
    state = SummaryState(research_topic="Benchmark", todo_items=[task])
    scorer = Mock()
    scorer.score_pairs_batched.return_value = None
    fallback = Mock()
    fallback.return_value = None
    orchestrator = ClaimVerificationOrchestrator(config, evidence_store=store, scorer=scorer)
    finding = {"text": "Processing time improved by 91%.", "source_ids": ["T1-S1"]}
    orchestrator.verify_request(state, claims_by_task={1: [finding]}, fallback_claim_verifier=fallback)
    scorer.score_pairs_batched.assert_not_called()
    fallback.assert_not_called()
    self_check = orchestrator.as_review_callback()(finding["text"], task.evidence_items)
    assert self_check is None
    scorer.score_pairs_batched.assert_not_called()


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


def test_atomic_citations_follow_their_span_and_table_context() -> None:
    claims = AtomicClaimDecomposer(decompose_with_fast_llm=False).decompose(
        task_id=2,
        parent_claims=[{
            "text": (
                "## Results\n"
                "- Acme 营收 1.25 亿。 [T1-S1][T1-M1] 同时 Acme margin reached 24% [T2-C1].\n"
                "| Entity | 2025 revenue | Source |\n"
                "| --- | ---: | --- |\n"
                "| Beta | USD 2 million | [T1-M1] |\n"
                "```\nAcme made 0 [T1-S1]\n```"
            ),
            "source_ids": ["T1-S1", "T1-M1", "T2-C1"],
        }],
        valid_source_ids={"T1-S1", "T1-M1", "T2-C1"},
    )
    assert [claim.source_ids for claim in claims] == [["T1-S1", "T1-M1"], ["T2-C1"], ["T1-M1"]]
    assert "1.25" in claims[0].text and "24%" in claims[1].text
    assert "Beta" in claims[2].text and "2025 revenue" in claims[2].text


def test_llm_partial_decomposition_cannot_drop_or_rebind_inline_clauses() -> None:
    payload = {"claims": [
        {"parent_index": 0, "atomics": [{"text": "   "}]},
        {"parent_index": 1, "atomics": [
            {"text": "Acme revenue was USD 2 million [T1-M1]", "source_ids": ["T1-M1"]},
        ]},
    ]}
    claims = AtomicClaimDecomposer().decompose(
        task_id=1,
        parent_claims=[
            {"text": "Acme latency was 60 ms [T1-S1]", "source_ids": ["T1-S1"]},
            {
                "text": "Acme revenue was USD 2 million [T1-S1]; margin was 30% [T1-M1]",
                "source_ids": ["T1-S1", "T1-M1"],
            },
        ],
        valid_source_ids={"T1-S1", "T1-M1"},
        fast_llm=_MockLLM(json.dumps(payload)),
    )
    assert [claim.source_ids for claim in claims] == [["T1-S1"], ["T1-S1"], ["T1-M1"]]
    assert "30%" in claims[2].text


def test_numeric_grounding_normalizes_dimensions_and_scale() -> None:
    from services.claim_verifier import (
        _extract_numeric_and_unit,
        numeric_grounding_mismatch,
    )

    assert _extract_numeric_and_unit("2025年Q4营收2,025万元") == (20_250_000, "cny")
    assert _extract_numeric_and_unit("FY2025 revenue was USD 1.2 billion") == (1_200_000_000, "usd")
    assert _extract_numeric_and_unit("毛利率增长2个百分点") == (2, "pp")
    assert _extract_numeric_and_unit("Acme v2.5 has 1 亿 users") == (100_000_000, "scalar")
    assert _extract_numeric_and_unit("营收为2025") == (2025, "scalar")
    assert numeric_grounding_mismatch("营收为1.2亿美元", ["Revenue reached $120 million."]) is None
    assert numeric_grounding_mismatch("营收为1亿元", ["营收为10000万元"]) is None
    assert numeric_grounding_mismatch("Margin reached 42%", ["Margin reached 0.42."]) is None
    assert numeric_grounding_mismatch("Latency was 0.5 seconds", ["Latency was 500 ms."]) is None
    assert numeric_grounding_mismatch("Margin rose 2 pp", ["Margin rose 2 percentage points."]) is None
    assert numeric_grounding_mismatch("FY2025 margin reached 98%", ["FY2025 margin reached 42%."])
    assert numeric_grounding_mismatch("FY2025 margin reached 42%", ["FY2024 margin reached 42%."])
    assert numeric_grounding_mismatch("Margin rose 2 pp", ["Margin rose 2%."])
    assert numeric_grounding_mismatch("Revenue was USD 1 million", ["Revenue was EUR 1 million."])
    assert numeric_grounding_mismatch("Revenue was USD 1 billion", ["Revenue was USD 1 million."])


def test_scorer_rejects_ambiguous_labels_and_keeps_completed_batches() -> None:
    assert _resolve_label_mapping({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}) == {}
    calls = 0

    def predict(batch):
        nonlocal calls
        calls += 1
        return [[0.0, 5.0, 0.0]] if calls == 1 else [[float("nan"), 0.0, 0.0]]

    model = LoadedNLIModel(
        backend="mock",
        model_name="mock",
        label_mapping={"contradiction": 0, "entailment": 1, "neutral": 2},
        predict_logits_fn=predict,
    )
    scorer = LocalNLIScorer(Configuration(nli_batch_size=1), loaded_model=model)
    result = scorer.score_pairs_batched([("A", "A"), ("B", "B")], deadline_monotonic=time.monotonic() + 2)
    assert result and len(result) == 1 and result[0]["entailment"] > 0.9


def test_conflict_graph_only_attributes_backed_conflicting_sources() -> None:
    from services.claim_verifier import normalize_conflict_key

    assert normalize_conflict_key("Acme", "营收同比增长率", "2025 年第四季度") == (
        "acme", "revenue_growth_rate", "2025Q4",
    )
    assert normalize_conflict_key("Acme", "revenue_growth_rate", "Q4 2025") == (
        "acme", "revenue_growth_rate", "2025Q4",
    )
    records = {
        "T1-S1": {"source_type": "forum_general"},
        "T1-M1": {"source_origin": "mcp", "source_type": "mcp_curated_filing"},
        "T1-C1": {"source_origin": "sandbox", "source_type": "computed_sandbox", "derived_from_source_ids": ["T1-M1"]},
        "T2-S1": {"source_type": "government"},
    }

    def claim(sid, value, unit, label, task_id=1):
        return AtomicClaimRecord(
            atomic_id=f"{task_id}-{sid}", task_id=task_id, parent_claim_index=0,
            text="Acme Q4 gross margin", source_ids=[sid], best_source_id=sid, nli_label=label,
            calibrated_confidence=0.9,
            proposition=AtomicProposition(entity="Acme", metric="gross margin", period="Q4 2025",
                                          numeric_value=value, unit=unit),
        )

    claims = [
        claim("T1-S1", 42, "%", "ENTAILED"), claim("T1-M1", 0.42, "scalar", "ENTAILED"),
        claim("T1-C1", 2, "pp", "ENTAILED"), claim("T2-S1", 16, "%", "SKIPPED"),
    ]
    builder = ConflictGraphBuilder(Configuration(nli_conflict_numeric_rel_tol=0))
    assert builder.build(claims, evidence_by_id=records) == []
    claims[-1].nli_label = "ENTAILED"
    clusters = builder.build(claims, evidence_by_id=records)
    assert len(clusters) == 1
    assert {side.source_id for side in clusters[0].sides} == {"T1-S1", "T1-M1", "T2-S1"}
    assert len(clusters[0].edges) == 2
    assert clusters[0].recommended_source_id is None
    assert clusters[0].to_dict() == builder.build(list(reversed(claims)), evidence_by_id=records)[0].to_dict()


def test_scoped_retriever_fills_a_missing_source_without_broadening_scope() -> None:
    from services.evidence_index import EvidenceRetrievalHit

    evidence = [
        {"source_id": "T1-M1", "full_content": "Acme margin was 42%."},
        {"source_id": "T2-S1", "full_content": "Acme margin was 88%."},
    ]

    class Store(_FakeEvidenceStore):
        def lookup(self, *, source_ids=None, include_full_content=False):
            assert source_ids == ["T1-M1"]
            return super().lookup(source_ids=source_ids, include_full_content=include_full_content)

    def hit(sid, origin):
        return EvidenceRetrievalHit(
            origin=origin, citation_eligible=origin == "current", task_id=1,
            source_id=sid, archive_doc_key=None, archive_version_id=None,
            chunk_id=f"{sid}-chunk", text="Acme margin 21%", url="", title="", domain="",
            quality_label="medium", freshness_label="unknown", score=1.0,
        )

    retriever = SimpleNamespace(query=lambda *args, **kwargs: SimpleNamespace(
        hits=[hit("T1-S1", "current"), hit("T1-M1", "history"), hit("T2-S1", "current")],
    ))
    atomic = AtomicClaimRecord(
        atomic_id="1-C0-A0", task_id=1, parent_claim_index=0,
        text="Acme margin was 42%", source_ids=["T1-S1", "T1-M1"],
    )
    hits = ScopedEvidenceRetriever().fetch_for_claim(
        atomic, retrieval_service=retriever, evidence_store=Store(evidence), cited_records_fallback=[],
    )
    assert [(item.source_id, item.origin) for item in hits] == [("T1-S1", "current"), ("T1-M1", "current")]
    assert "42%" in hits[1].text


def test_nli_numeric_guard_and_atomic_cap_reach_parent_rollup() -> None:
    cfg = Configuration(nli_verifier_enabled=True, nli_max_atomic_claims_per_task=2)
    score = LocalNLIScorer(cfg, loaded_model=LoadedNLIModel(
        backend="test", model_name="test", label_mapping={"contradiction": 0, "entailment": 1, "neutral": 2},
        predict_logits_fn=lambda batch: [[0, 8, 0] for _ in batch],
    ))
    state = SummaryState(research_topic="Test", todo_items=[TodoItem(
        id=1, title="Data", intent="Check", query="Acme",
        evidence_items=[{"source_id": "T1-S1", "snippet": "Acme margin was 42%. Acme revenue was USD 2 million."}],
    )])
    claims = {1: [
        {"text": text, "source_ids": ["T1-S1"], "support_status": "supported"}
        for text in ("Acme margin was 42%", "Acme revenue was USD 2 billion", "Acme margin was 80%")
    ]}
    out = ClaimVerificationOrchestrator(cfg, scorer=score).verify_request(state, claims_by_task=claims)
    assert [row["nli_label"] for row in state.todo_items[0].atomic_claims] == ["ENTAILED", "NEUTRAL", "SKIPPED"]
    assert [row["support_status"] for row in claims[1]] == ["supported", "unverified_claim", "unverified_claim"]
    assert out["summary"]["degraded"] and out["summary"]["atomic_faithfulness_rate"] == 0.5


def test_direction04_snapshot_and_event_payloads_are_tenant_scoped(tmp_path) -> None:
    config = Configuration(
        auth_enabled=True,
        tenant_api_keys={"key-alpha": "alpha", "key-beta": "beta"},
        request_state_dir=str(tmp_path),
    )
    app = create_app(base_config=config)
    atomic = {"atomic_id": "1-C0-A0", "nli_label": "ENTAILED", "matched_excerpt": "private excerpt"}
    conflicts = [{"conflict_id": "CONF-1", "sides": [{"evidence_excerpt": "private conflict"}]}]
    summary = {"entailed_count": 1, "total_atomic_claims": 1}
    app.state.request_state_store.save(
        "req-nli",
        {"todo_items": [{"atomic_claims": [atomic]}], "evidence_conflicts": conflicts,
         "claim_verification_summary": summary},
        tenant_id="alpha",
    )
    app.state.event_bus.publish(
        "alpha", "req-nli",
        {"type": "final_report", "evidence_conflicts": conflicts, "claim_verification_summary": summary},
    )
    with TestClient(app) as client:
        for path in ("/requests/req-nli", "/requests/req-nli/events"):
            assert client.get(path).status_code == 401
            assert client.get(path, headers={"X-API-Key": "key-beta"}).status_code == 404
        headers = {"X-API-Key": "key-alpha"}
        detail = client.get("/requests/req-nli", headers=headers).json()
        assert detail["evidence_conflicts"] == conflicts
        assert detail["claim_verification_summary"] == summary
        assert detail["todo_items"][0]["atomic_claims"] == [atomic]
        events = client.get("/requests/req-nli/events", headers=headers).json()["events"]
        assert events[0]["data"]["evidence_conflicts"] == conflicts
        assert events[0]["data"]["claim_verification_summary"] == summary


def test_direction04_metrics_only_returns_the_callers_request_traces(tmp_path) -> None:
    config = Configuration(
        auth_enabled=True,
        tenant_api_keys={"key-alpha": "alpha", "key-beta": "beta"},
        request_state_dir=str(tmp_path),
    )
    for tenant in ("alpha", "beta"):
        metrics_registry.store_request(
            {"request_id": f"nli-{tenant}", "tenant_id": tenant,
             "todo_items": [{"atomic_claims": [{"matched_excerpt": f"{tenant} excerpt"}]}]},
        )
    with TestClient(create_app(base_config=config)) as client:
        assert client.get("/metrics/json").status_code == 401
        requests = client.get("/metrics/json", headers={"X-API-Key": "key-alpha"}).json()["recent_requests"]
    assert [entry["request_id"] for entry in requests] == ["nli-alpha"]
    assert requests[0]["todo_items"][0]["atomic_claims"][0]["matched_excerpt"] == "alpha excerpt"


def test_direction04_snapshot_retains_owner_for_versioned_and_unversioned_saves(tmp_path) -> None:
    store = RequestStateStore(str(tmp_path))
    original = store.save("req-nli", {"evidence_conflicts": [{"conflict_id": "CONF-1"}]}, tenant_id="alpha")
    for expected_version in (None, original["state_version"]):
        with pytest.raises(SnapshotConflictError):
            store.save("req-nli", {"evidence_conflicts": []}, tenant_id="beta", expected_version=expected_version)
    assert store.load("req-nli", tenant_id="alpha") == original
    assert store.load("req-nli", tenant_id="beta") is None


def test_direction04_resume_restores_summary_without_recording_another_verification() -> None:
    observer = RequestTrace(
        request_id="req-resume-nli", topic="topic", search_api="test", provider=None,
        model=None, pricing_catalog=None,
    )
    state = SummaryState(
        review_completed=True,
        claim_verification_summary={"total_atomic_claims": 2, "entailed_count": 1, "skipped_count": 1},
    )
    state.review_summary = {"overall_status": "passed", "claim_verification_summary": state.claim_verification_summary}
    before = metrics_registry.snapshot(include_recent_requests=False)
    StateManagerMixin()._restore_task_counters(observer, state)
    snapshot = observer.snapshot()
    after = metrics_registry.snapshot(include_recent_requests=False)
    assert snapshot["claim_verification_summary"] == state.claim_verification_summary
    assert snapshot["review_summary"] == state.review_summary
    assert after["counters"]["nli_claims_verified_total"] == before["counters"]["nli_claims_verified_total"]
    assert after["latencies_ms"]["nli_latency_ms"] == before["latencies_ms"]["nli_latency_ms"]


def test_direction04_configuration_validates_nli_options_and_preserves_safe_defaults() -> None:
    assert Configuration().nli_verifier_enabled is False
    assert Configuration().nli_warmup_enabled is False
    assert Configuration(nli_model_backend=" Transformers_NLI ", nli_device=" CPU ").nli_model_backend == "transformers_nli"
    with pytest.raises(ValidationError):
        Configuration(nli_model_backend="unknown")
    with pytest.raises(ValidationError):
        Configuration(nli_device="gpu")
    with pytest.raises(ValidationError):
        Configuration(nli_stage_budget_seconds=float("inf"))


def test_direction04_api_response_fields_have_blueprint_defaults() -> None:
    payload = ResearchResponse(report_markdown="report")
    assert payload.evidence_conflicts == []
    assert payload.claim_verification_summary == {}


def test_direction04_snapshot_defaults_handle_legacy_and_malformed_atomic_fields() -> None:
    host = StateManagerMixin()
    host._evidence_store = SimpleNamespace(hydrate_from_tasks=lambda tasks: None)
    state, _ = host._state_from_snapshot(
        {"todo_items": [{"id": 0}, {"id": 0, "atomic_claims": "invalid"}]},
    )
    assert state.evidence_conflicts == []
    assert state.claim_verification_summary == {}
    assert [task.atomic_claims for task in state.todo_items] == [[], []]


def test_direction04_research_runs_with_the_application_nli_configuration(tmp_path) -> None:
    class _ConfigAgent:
        def __init__(self, config, request_id=None):
            self.config = config

        def run(self, topic):
            return SimpleNamespace(
                report_markdown="report",
                todo_items=[],
                claim_verification_summary={
                    "enabled": self.config.nli_verifier_enabled,
                    "decompose_with_fast_llm": self.config.nli_decompose_with_fast_llm,
                },
            )

    config = Configuration(
        nli_verifier_enabled=True,
        nli_decompose_with_fast_llm=False,
        request_state_dir=str(tmp_path),
    )
    with TestClient(create_app(base_config=config, agent_factory=_ConfigAgent)) as client:
        response = client.post("/research", json={"topic": "topic"})
    assert response.status_code == 200
    assert response.json()["claim_verification_summary"] == {
        "enabled": True, "decompose_with_fast_llm": False,
    }


def test_direction04_plan_edits_invalidate_results_from_previous_tasks(tmp_path) -> None:
    config = Configuration(request_state_dir=str(tmp_path))
    app = create_app(base_config=config)
    store = app.state.request_state_store
    store.save(
        "req-plan-review",
        {
            "phase": "completed",
            "status": "success",
            "review_completed": True,
            "evidence_conflicts": [{"conflict_id": "CONF-old"}],
            "claim_verification_summary": {"total_atomic_claims": 1},
            "review_summary": {"claim_verification_summary": {"total_atomic_claims": 1}},
            "todo_items": [{"id": 1, "atomic_claims": [{"atomic_id": "1-C0-A0"}]}],
        },
    )
    with TestClient(app) as client:
        response = client.put("/requests/req-plan-review/plan", json={
            "tasks": [{"title": "Current task", "intent": "Verify", "query": "current"}],
        })
        assert response.status_code == 200
        snapshot = client.get("/requests/req-plan-review").json()
    assert snapshot["review_completed"] is False
    assert snapshot["evidence_conflicts"] == []
    assert snapshot["claim_verification_summary"] == {}
    assert snapshot["review_summary"] == {}
    assert not snapshot["todo_items"][0].get("atomic_claims")


def test_direction04_redis_replay_keys_keep_tenant_and_request_boundaries() -> None:
    key = RedisStreamEventBus._stream_key
    assert key("alpha", "req-nli") == "deepresearch:stream:alpha:req-nli"
    assert key("org:alpha", "req-nli") != key("org", "alpha:req-nli")
    assert key("org:alpha", "req-nli") != key("org%3Aalpha", "req-nli")


def test_scoped_retrieval_keeps_all_cited_sources_and_configured_chunks(monkeypatch):
    from types import SimpleNamespace

    from config import Configuration
    from services import evidence_index
    from services.evidence_index import EvidenceRetrievalService, EvidenceRuntimeIndex

    monkeypatch.setattr(evidence_index, "encode_texts", lambda texts, **_: [None] * len(texts))
    monkeypatch.setattr(evidence_index, "encode_text", lambda *_args, **_kwargs: None)
    cfg = Configuration(evidence_runtime_enabled=True, nli_top_k_chunks_per_source=3)
    index = EvidenceRuntimeIndex(cfg)
    allowed = ["T1-S1", "T1-M1", "T1-C1", "T2-M1", "T2-C1"]
    for sid in [*allowed, "T2-S5"]:
        task_id = int(sid[1])
        index.on_records_changed(
            task_id=task_id,
            records=[SimpleNamespace(
                source_id=sid,
                multimodal_blocks=[{"text": f"Revenue observation {i}"} for i in range(3)],
            )],
            reason="compute" if "-C" in sid else ("mcp" if "-M" in sid else "search"),
        )

    class History:
        def query(self, **_):
            raise AssertionError("Whitelisted retrieval must never query history")

    retrieval = EvidenceRetrievalService(runtime_index=index, memory_service=History(), config=cfg)
    result = retrieval.query(
        "revenue", task_id=2, scope="hybrid", mode="repair",
        top_k=15, allowed_source_ids=allowed,
    )
    assert (result.scope, result.mode) == ("current", "grounding")
    assert {hit.source_id for hit in result.hits} == set(allowed)
    assert all(sum(hit.source_id == sid for hit in result.hits) == 3 for sid in allowed)
    assert not retrieval.query("revenue", task_id=2, allowed_source_ids=[]).hits


def test_factscore_uses_typed_atomic_records_and_excludes_skipped():
    from evals.judges.hybrid import HybridJudge
    from evals.schema import BenchmarkCase
    from models import TodoItem

    task = TodoItem(id=1, title="test", intent="test", query="test", atomic_claims=[
        {"nli_label": "ENTAILED"},
        {"nli_label": "NEUTRAL"},
        {"nli_label": "SKIPPED"},
    ])
    result = HybridJudge().evaluate(
        case=BenchmarkCase(id="case", topic="topic"),
        report_markdown="report",
        todo_items=[task],
        trace_snapshot={"claim_verification_summary": {"atomic_faithfulness_rate": 1, "total_atomic_claims": 1}},
    )
    assert result["atomic_faithfulness_rate"] == 0.5
    assert result["evaluated_atomic_claims"] == 2
    assert result["skipped_count"] == 1

    task.atomic_claims = [{"nli_label": "SKIPPED"}]
    result = HybridJudge().evaluate(
        case=BenchmarkCase(id="case", topic="topic"),
        report_markdown="report",
        todo_items=[task],
        trace_snapshot=None,
    )
    assert result["atomic_faithfulness_rate"] is None


def test_eval_benchmark_and_project_report_aggregate_factscore_and_conflict_recall():
    from types import SimpleNamespace

    from config import Configuration
    from evals.judges.hybrid import HybridJudge
    from evals.report_metrics import aggregate_records, normalize_eval_result
    from evals.runner import run_benchmark_suite
    from evals.schema import BenchmarkCase
    from models import TodoItem

    class Agent:
        def __init__(self, **_):
            pass

        def run(self, topic):
            labels = ["ENTAILED", "SKIPPED"] if topic == "first" else ["CONTRADICTED"] * 2 + ["NEUTRAL"]
            return SimpleNamespace(
                report_markdown="Report",
                todo_items=[TodoItem(
                    id=1, title="test", intent="test", query="test",
                    atomic_claims=[{"nli_label": label} for label in labels],
                    claims=[{"support_status": "supported" if topic == "first" else "contradicted"}],
                )],
                evidence_conflicts=[{
                    "conflict_id": "same", "entity": "acmecorp", "metric": "revenue",
                    "period": "2025Q4", "adjudication_status": "unresolved_divergence", "severity": "high",
                }] if topic == "first" else [],
            )

    first = {"entity": "ACME Corp", "metric": "营收", "period": "2025年第四季度"}
    second = {"entity": "ACME Corp", "metric": "净利润", "period": "2025 Q4"}
    third = {"entity": "ACME Corp", "metric": "毛利率", "period": "2025 Q4"}
    benchmark = run_benchmark_suite(
        [
            BenchmarkCase(id="first", topic="first", metadata={"expected_conflicts": [first, first]}),
            BenchmarkCase(id="second", topic="second", metadata={"expected_conflicts": [first, second, third]}),
        ],
        config=Configuration(),
        agent_factory=Agent,
        judge=HybridJudge(),
    )
    result_a, result_b = benchmark["results"]
    assert result_a["metrics"]["conflict_recall"] == 1.0
    assert result_b["metrics"]["conflict_recall"] == 0.0
    assert len(result_a["todo_items"][0]["atomic_claims"]) == 2
    assert result_a["evidence_conflicts"][0]["conflict_id"] == "same"
    assert benchmark["summary"]["atomic_factscore"] == 0.25
    assert benchmark["summary"]["conflict_recall"] == 0.25

    records = [normalize_eval_result(item, source_label="unit", fallback_latency_keys=())
               for item in benchmark["results"]]
    assert all(record is not None for record in records)
    aggregate = aggregate_records(records)
    assert aggregate["atomic_factscore"] == 0.25
    assert aggregate["contradiction_rate"] == 0.5
    assert aggregate["skipped_count"] == 1
    assert aggregate["conflict_recall"] == 0.25
    assert aggregate["conflict_count"] == 1
    assert aggregate["citation_validity_rate"] == 0.5


def test_eval_conflict_resolution_ignores_malformed_and_duplicate_clusters():
    from evals.metrics.factscore import summarize_conflict_resolution

    resolved = {"conflict_id": "one", "adjudication_status": "resolved_by_authority", "severity": "low"}
    summary = summarize_conflict_resolution([resolved, resolved.copy(), None])
    assert summary["conflict_count"] == 1
    assert summary["resolved_by_authority"] == 1
    assert summary["resolution_rate"] == 1.0


def test_project_report_recovers_atomic_counts_from_summary_only():
    from evals.report_metrics import aggregate_records, normalize_request_snapshot

    record = normalize_request_snapshot(
        {
            "request_id": "old-export",
            "request_metrics": {
                "claim_verification_summary": {
                    "entailed_count": 3,
                    "contradicted_count": 1,
                    "neutral_count": 2,
                    "skipped_count": 7,
                }
            },
        },
        source_label="unit",
    )
    assert record is not None
    aggregate = aggregate_records([record])
    assert aggregate["atomic_factscore"] == 0.5
    assert aggregate["evaluated_atomic_claims"] == 6
    assert aggregate["total_atomic_claims"] == 13
    assert aggregate["conflict_recall"] is None


def test_scoped_evidence_table_chunk_keeps_columns_with_late_rows(monkeypatch):
    from types import SimpleNamespace

    from config import Configuration
    from services import evidence_index
    from services.evidence_index import EvidenceRuntimeIndex

    monkeypatch.setattr(evidence_index, "encode_texts", lambda texts, **_: [None] * len(texts))
    monkeypatch.setattr(evidence_index, "encode_text", lambda *_args, **_kwargs: None)
    cfg = Configuration(evidence_runtime_enabled=True, evidence_chunk_chars=240, evidence_chunk_overlap=0)
    index = EvidenceRuntimeIndex(cfg)
    table = (
        "## Financial highlights\n\n"
        "| Company | Gross margin (%) |\n| --- | ---: |\n"
        + "\n".join(f"| Company Number {i} | 12.3 |" for i in range(25))
        + "\n| TargetCorp | 42.5 |"
    )
    index.on_records_changed(
        task_id=1,
        records=[SimpleNamespace(source_id="T1-S1", full_content=table)],
        reason="fetch",
    )
    hit, = index.query(text="TargetCorp", top_k=1, allowed_source_ids=["T1-S1"])
    assert "| TargetCorp | 42.5 |" in hit.text
    assert "| Company | Gross margin (%) |\n| --- | ---: |" in hit.text


def test_table_cells_bind_each_value_to_its_own_citation() -> None:
    claims = AtomicClaimDecomposer().decompose(
        task_id=1,
        parent_claims=[{
            "text": "Company | Revenue | Margin\n--- | --- | ---\nAcme | USD 2 million [T1-M1] | 42% [T1-C1]",
            "source_ids": ["T1-M1", "T1-C1"],
        }],
        valid_source_ids={"T1-M1", "T1-C1"},
    )
    assert [(claim.proposition.metric, claim.source_ids) for claim in claims] == [
        ("revenue", ["T1-M1"]), ("gross_margin", ["T1-C1"]),
    ]
    assert all("Acme" in claim.text for claim in claims)
def test_reviewer_preserves_citation_spans_validates_inline_ids_and_uses_current_record() -> None:
    from types import SimpleNamespace

    from config import Configuration
    from models import SummaryState, TodoItem
    from services.reviewer import ReviewService

    source = {"source_id": "T1-S1", "snippet": "A sufficiently long old excerpt says 20%."}
    full_source = {**source, "full_content": "The final metric published for the interval was 24%."}
    task = TodoItem(
        id=1, title="Results", intent="Review results", query="q", status="completed",
        evidence_items=[source, {"source_id": "T1-M1"}],
        claims=[
            {"text": "Reported margin was 24% [T1-S1]; export status is stable [T1-M1]"},
            {"text": "The export was available [T9-C9]", "source_ids": ["T1-S1"]},
            {"text": "The published margin was 24%", "source_ids": ["T1-S1"]},
        ],
    )
    service = ReviewService(
        None, Configuration(nli_verifier_enabled=False, review_agent_enabled=False),
        evidence_store=SimpleNamespace(lookup=lambda **_: [full_source]),
    )
    service.review_request(SummaryState(todo_items=[task], research_topic="Results"))
    assert task.claims[0]["source_ids"] == ["T1-S1", "T1-M1"]
    assert "[T1-S1];" in task.claims[0]["text"]
    assert task.claims[1]["support_status"] == "invalid_citation"
    assert task.claims[2]["support_status"] == "supported"


def test_reviewer_ignores_table_headers_and_fenced_examples_as_claims() -> None:
    from services.reviewer import ReviewService

    claims = ReviewService._extract_claim_texts(
        "## Findings\n\n"
        "| Issuer | FY2025 margin | Source |\n"
        "| :--- | ---: | --- |\n"
        "| Example Corp | 18% | [T1-M1] |\n\n"
        "```text\nexample statement without a source\n```\n\n"
        "+ 第二个来源支持此结论 [T1-C1]"
    )
    assert len(claims) == 2
    assert "FY2025 margin" in claims[0] and "[T1-M1]" in claims[0]
    assert claims[1] == "第二个来源支持此结论 [T1-C1]"


def test_reviewer_rejects_claim_citing_unverified_computation_across_tasks() -> None:
    from config import Configuration
    from models import SummaryState, TodoItem
    from services.reviewer import ReviewService

    original = TodoItem(
        id=1, title="Original", intent="Compute", query="q", status="completed",
        evidence_items=[{"source_id": "T1-C1", "source_origin": "sandbox", "snippet": "margin was 92%"}],
    )
    dependent = TodoItem(
        id=2, title="Dependent", intent="Compare", query="q", status="completed",
        claims=[{"text": "Margin was 92% [T1-C1]"}],
    )
    service = ReviewService(None, Configuration(review_agent_enabled=False))
    service.review_request(SummaryState(research_topic="Margins", todo_items=[original, dependent]))
    assert dependent.claims[0]["support_status"] == "invalid_citation"


def test_reporting_fallback_omits_contradicted_claim_from_both_summary_paths() -> None:
    from config import Configuration
    from models import SummaryState, TodoItem
    from services.reporter import ReportingService

    contradicted = "Acme reported a 92% operating margin"
    task = TodoItem(
        id=1, title="Acme", intent="Measure margins", query="q", status="completed",
        review_status="warning",
        claims=[
            {"text": contradicted, "source_ids": ["T1-S1"], "support_status": "contradicted", "nli_verdict": "CONTRADICTED"},
            {"text": "Acme reported a 12% operating margin", "source_ids": ["T1-S1"], "support_status": "supported"},
        ],
        summary_payload={
            "executive_summary": contradicted,
            "key_findings": [{"text": contradicted, "source_ids": ["T1-S1"]}],
        },
        evidence_items=[{"source_id": "T1-S1", "title": "Acme filing", "snippet": "Acme reported a 12% operating margin."}],
    )
    state = SummaryState(research_topic="Acme", todo_items=[task])
    service = ReportingService(None, Configuration(), evidence_store=None)
    payload = {
        "background_overview": contradicted,
        "key_findings": [{"claim": contradicted, "source_ids": ["T1-S1"]}],
    }
    assert contradicted not in service._task_prompt_summary(task)
    report = service._render_structured_report(payload, state=state)
    assert contradicted not in report
    assert "Acme reported a 12% operating margin" in report


def test_reporting_markdown_preserves_fences_and_renders_all_conflict_edges() -> None:
    import re
    from types import SimpleNamespace

    from config import Configuration
    from models import SummaryState, TodoItem
    from services.reporter import ReportingService

    markdown = (
        "## 风险与挑战\n\n"
        "| Issuer | Margin |\n| --- | --- |\n| Acme | 12% |\n\n"
        "```json\n{\"text\": \"[T9-S9]\"}\n## 参考来源\n```\n\n"
        "## 参考来源\n"
    )
    evidence = [
        {"source_id": source_id, "title": name, "url": "https://example.com/report"}
        for source_id, name in (("T1-S1", "filing"), ("T1-M1", "archive"), ("T1-C1", "computation"))
    ]
    conflict = {
        "conflict_id": "CONF-markdown",
        "entity": "Acme | Global",
        "metric": "margin",
        "period": "2025",
        "sides": [
            {"source_id": "T1-S1", "title": r"Filing \| revision", "claim_text": "Margin was 12%."},
            {"source_id": "T1-M1", "title": "Archive", "claim_text": "Margin was 24%."},
            {"source_id": "T1-C1", "title": "Calculation", "claim_text": "Margin was 33%."},
            {"source_id": "T9-C9", "title": "Foreign record", "claim_text": "unavailable"},
        ],
        "edges": [
            {"source_id_a": "T1-S1", "source_id_b": "T1-M1"},
            {"source_id_a": "T1-M1", "source_id_b": "T1-C1"},
            {"source_id_a": "T1-C1", "source_id_b": "T9-C9"},
        ],
        "recommended_source_id": "T9-C9",
        "adjudication_status": "resolved_by_authority",
        "adjudication_rationale": "Foreign record [T9-C9]",
    }
    task = TodoItem(id=1, title="Acme", intent="Measure margins", query="q", evidence_items=evidence)
    state = SummaryState(research_topic="Acme", todo_items=[task], evidence_conflicts=[conflict])
    agent = SimpleNamespace(run=lambda *_args, **_kwargs: markdown, clear_history=lambda: None)
    store = SimpleNamespace(list_task_evidence=lambda *_: evidence, build_reference_map=lambda _: evidence)
    service = ReportingService(agent, Configuration(), evidence_store=store)
    events = list(service.stream_generate_report(state))
    report = "".join(event["content"] for event in events if event["type"] == "report_chunk")
    assert "多源口径分歧与证据对照" in events[0]["sections"]
    assert "```json\n{\"text\": \"[T9-S9]\"}\n## 参考来源\n```" in report
    assert "| Issuer | Margin |\n| --- | --- |\n| Acme | 12% |" in report
    section, references = report.rsplit("## 参考来源", 1)
    rows = [line for line in section.splitlines() if line.startswith(r"| Acme \| Global")]
    assert len(rows) == 2
    assert all(len(re.split(r"(?<!\\)\|", row)) == 6 for row in rows)
    assert "[T1-C1]" in section and "[T1-C1]" in references
    assert "Foreign record" not in report and "[T9-C9]" not in report
    exposed = service.build_workbench_payload(state, report)["evidence_conflicts"][0]
    assert {side["source_id"] for side in exposed["sides"]} == {"T1-S1", "T1-M1", "T1-C1"}
    assert exposed["recommended_source_id"] is None


def test_structured_markdown_preserves_multiline_tables_and_code() -> None:
    from config import Configuration
    from models import SummaryState, TodoItem
    from services.reporter import ReportingService

    table = "| Issuer | Margin |\n| --- | --- |\n| Acme | 12% |"
    code = "```text\nreview status blocked\n    2 | 3\n```"
    task = TodoItem(
        id=1, title="Acme", intent="Measure margins", query="q",
        evidence_items=[{"source_id": "T1-S1", "title": "Acme"}],
    )
    service = ReportingService(None, Configuration(report_layout_mode="flexible"), evidence_store=None)
    report = service._render_structured_report(
        {"custom_sections": [{
            "title": "Margin details", "content_type": "paragraph",
            "content": f"{table}\n\n{code}", "source_ids": ["T1-S1"],
        }]},
        state=SummaryState(todo_items=[task]),
    )
    assert table in report
    assert f"{code}\n\n[T1-S1]" in report


def test_reviewer_uses_separate_citation_line_with_preceding_claim() -> None:
    from services.reviewer import ReviewService

    claims = ReviewService._extract_claim_texts(
        "- Reported issuer margin was 12%.\n  [T1-S1] [T1-M1]\n"
        "- The calculation is documented [T1-C1]"
    )
    assert claims == [
        "Reported issuer margin was 12%. [T1-S1] [T1-M1]",
        "The calculation is documented [T1-C1]",
    ]


def test_numeric_grounding_checks_quarter_and_currency_in_the_same_excerpt() -> None:
    from services.claim_verifier import numeric_grounding_mismatch

    assert numeric_grounding_mismatch("2025Q4营收2亿元", ["2025年第四季度营收20,000万元"]) is None
    assert numeric_grounding_mismatch("Q4 2025 revenue reached $1 billion", ["2025 Q3 revenue was USD 1 billion."])
    assert numeric_grounding_mismatch("Acme's gross margin was 42%", ["2025 margin was 42 percent."]) is None
    assert numeric_grounding_mismatch("FY2025 revenue reached USD 2 million", [
        "FY2024 revenue was USD 2 million.",
        "FY2025 revenue was USD 1 million.",
    ])


def test_binary_non_entailment_does_not_become_contradiction() -> None:
    assert _resolve_label_mapping({0: "unsupported", 1: "supported"}) == {"neutral": 0, "entailment": 1}


def test_report_atomic_binding_uses_only_entailed_best_source():
    from services.claim_verifier import report_atomic_sources

    record = {"nli_label": "ENTAILED", "best_source_id": "T1-S1", "source_ids": ["T1-S1", "T2-S1"]}
    assert report_atomic_sources(record, {"T1-S1", "T2-S1"}) == ("T1-S1",)
    assert report_atomic_sources(record, {"T2-S1"}) == ()
    assert report_atomic_sources({**record, "nli_label": "SKIPPED"}, {"T1-S1"}) == ()
