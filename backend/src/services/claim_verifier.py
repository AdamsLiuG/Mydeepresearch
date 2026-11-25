"""Atomic claim decomposition, source-scoped NLI verification, and cross-source conflict graph."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Literal, TypeVar

try:  # pragma: no cover - optional runtime imports
    from huggingface_hub import constants as huggingface_constants
except Exception:  # pragma: no cover - optional runtime imports
    huggingface_constants = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime imports
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional runtime imports
    CrossEncoder = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime imports
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional runtime imports
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

from metrics import metrics_registry
from services.evidence import detect_source_type
from services.evidence_index import EvidenceRetrievalHit, EvidenceRetrievalService
from services.text_processing import (
    CITATION_PATTERN,
    iter_markdown_lines,
    normalize_agent_markdown,
)
from utils import truncate_text

logger = logging.getLogger(__name__)

NliLabel = Literal["ENTAILED", "CONTRADICTED", "NEUTRAL", "SKIPPED"]
ConflictType = Literal[
    "numeric_divergence",
    "semantic_contradiction",
    "scope_mismatch",
    "temporal_shift",
]
AdjudicationStatus = Literal[
    "resolved_by_authority",
    "resolved_by_recency",
    "unresolved_divergence",
]

SOURCE_TIER_WEIGHT: dict[str, float] = {
    "government": 1.00,
    "standards_spec": 0.96,
    "official_documentation": 0.94,
    "mcp_curated_filing": 0.95,
    "mcp_filing": 0.95,
    "mcp_internal_metrics": 0.88,
    "computed_sandbox": 0.92,
    "sandbox_computation": 0.92,
    "peer_reviewed_paper": 0.92,
    "official_org": 0.90,
    "official_product": 0.88,
    "repository_official": 0.82,
    "preprint_paper": 0.80,
    "news_primary": 0.72,
    "company_blog": 0.68,
    "mcp_tool": 0.70,
    "technical_blog": 0.60,
    "reference_curated": 0.58,
    "mcp_external": 0.55,
    "forum_expert": 0.45,
    "web_general": 0.40,
    "web": 0.40,
    "forum_general": 0.25,
    "social_general": 0.20,
    "content_farm_or_aggregator": 0.10,
}

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("营业收入", "总营收", "营收", "销售额", "revenue", "sales"),
    "gross_margin": ("毛利率", "gross margin", "gross profit margin"),
    "net_margin": ("净利率", "净利润率", "net margin", "net profit margin"),
    "net_income": ("净利润", "净利", "盈利", "net income", "net profit"),
    "growth_rate": ("同比增速", "同比增长", "年复合增长率", "增速", "增长率", "growth rate", "cagr", "yoy"),
    "market_share": ("市场份额", "市占率", "占比", "份额", "market share"),
    "latency": ("p99延迟", "p95延迟", "首字延迟", "延迟", "ttft", "latency"),
    "throughput": ("吞吐量", "吞吐", "qps", "tps", "throughput"),
    "funding": ("融资", "估值", "funding", "valuation"),
}

_NEGATIVE_MARKERS = (
    "尚未",
    "不支持",
    "暂不",
    "未",
    "无法",
    "下降",
    "减少",
    "低于",
    "推迟",
    "取消",
    "否定",
    "内部测试",
    "实验阶段",
    "not ",
    "no ",
    "unable",
    "decline",
    "decrease",
    "without",
    "lack",
)
_POSITIVE_MARKERS = (
    "实现",
    "支持",
    "已",
    "增长",
    "提升",
    "突破",
    "领先",
    "正式",
    "全面",
    "生产",
    "increase",
    "growth",
    "support",
    "achieved",
    "production",
)

_NLI_LOCK = Lock()
_NLI_CACHE: dict[str, LoadedNLIModel] = {}
_NLI_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="claim-verifier")
_NLI_WORK_SLOTS = BoundedSemaphore(4)
_Result = TypeVar("_Result")


def _within_budget(work: Callable[[], _Result], deadline: float) -> _Result | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not _NLI_WORK_SLOTS.acquire(blocking=False):
        return None
    try:
        future = _NLI_EXECUTOR.submit(work)
    except RuntimeError:
        _NLI_WORK_SLOTS.release()
        return None
    future.add_done_callback(lambda _: _NLI_WORK_SLOTS.release())
    try:
        return future.result(timeout=max(0, deadline - time.monotonic()))
    except Exception as exc:
        future.cancel()
        logger.debug("Claim verification work interrupted: %s", exc)
        return None


@dataclass(kw_only=True)
class AtomicProposition:
    """Structured proposition skeleton extracted from an atomic claim."""

    entity: str = ""
    metric: str = ""
    period: str = ""
    numeric_value: float | None = None
    unit: str = ""
    polarity: Literal["positive", "negative", "neutral"] = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "metric": self.metric,
            "period": self.period,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "polarity": self.polarity,
        }


@dataclass(kw_only=True)
class AtomicClaimRecord:
    """Atomic sub-claim along with scoped retrieval hits and NLI verification verdict."""

    atomic_id: str
    task_id: int
    parent_claim_index: int
    text: str
    source_ids: list[str]
    proposition: AtomicProposition = field(default_factory=AtomicProposition)
    nli_label: NliLabel = "SKIPPED"
    raw_probs: dict[str, float] = field(default_factory=dict)
    calibrated_confidence: float = 0.0
    best_source_id: str | None = None
    matched_chunk_ids: list[str] = field(default_factory=list)
    matched_excerpt: str = ""
    verifier_backend: str = "none"
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "atomic_id": self.atomic_id,
            "task_id": self.task_id,
            "parent_claim_index": self.parent_claim_index,
            "text": self.text,
            "source_ids": list(self.source_ids),
            "proposition": self.proposition.to_dict(),
            "nli_label": self.nli_label,
            "raw_probs": {k: round(float(v), 4) for k, v in (self.raw_probs or {}).items()},
            "calibrated_confidence": round(float(self.calibrated_confidence), 4),
            "best_source_id": self.best_source_id,
            "matched_chunk_ids": list(self.matched_chunk_ids),
            "matched_excerpt": self.matched_excerpt,
            "verifier_backend": self.verifier_backend,
            "degraded_reason": self.degraded_reason,
        }


@dataclass(kw_only=True)
class ConflictSide:
    """Single source's stance and authority metadata inside an EvidenceConflictCluster."""

    source_id: str
    task_id: int
    title: str
    url: str
    domain: str
    source_type: str
    quality_label: str
    published_at: str | None
    claim_text: str
    normalized_value: float | None = None
    unit: str = ""
    evidence_excerpt: str = ""
    authority_rank: float = 0.0
    nli_support_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "task_id": self.task_id,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "source_type": self.source_type,
            "quality_label": self.quality_label,
            "published_at": self.published_at,
            "claim_text": self.claim_text,
            "normalized_value": self.normalized_value,
            "unit": self.unit,
            "evidence_excerpt": self.evidence_excerpt,
            "authority_rank": round(float(self.authority_rank), 4),
            "nli_support_score": round(float(self.nli_support_score), 4),
        }


@dataclass(kw_only=True)
class EvidenceConflictCluster:
    """Cluster of cross-source conflicting statements with adjudication rationale."""

    conflict_id: str
    entity: str
    metric: str
    period: str
    conflict_type: ConflictType
    severity: Literal["high", "medium", "low"]
    contradiction_score: float
    sides: list[ConflictSide]
    edges: list[dict[str, Any]]
    adjudication_status: AdjudicationStatus
    recommended_source_id: str | None = None
    adjudication_rationale: str = ""
    affected_task_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "entity": self.entity,
            "metric": self.metric,
            "period": self.period,
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "contradiction_score": round(float(self.contradiction_score), 4),
            "sides": [side.to_dict() for side in self.sides],
            "edges": [dict(edge) for edge in self.edges],
            "adjudication_status": self.adjudication_status,
            "recommended_source_id": self.recommended_source_id,
            "adjudication_rationale": self.adjudication_rationale,
            "affected_task_ids": list(self.affected_task_ids),
        }


@dataclass
class LoadedNLIModel:
    """Container holding a cached NLI predictor and normalized label index mapping."""

    backend: str
    model_name: str
    label_mapping: dict[str, int]
    predict_logits_fn: Callable[[list[tuple[str, str]]], list[list[float]]]
    device: str = "cpu"
    predict_lock: Any = field(default_factory=Lock, repr=False, compare=False)


def clear_nli_model_cache() -> None:
    """Clear the process-level NLI model singleton cache (used by tests)."""

    with _NLI_LOCK:
        _NLI_CACHE.clear()


def _resolve_label_mapping(id2label: dict[Any, Any] | None) -> dict[str, int]:
    """Normalize NLI checkpoint id2label indices to canonical entailment/contradiction/neutral keys."""

    mapping: dict[str, int] = {}
    for raw_idx, raw_name in (id2label or {}).items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        name = str(raw_name or "").strip().lower()
        if "entail" in name or name in {"support", "supported", "yes"}:
            mapping["entailment"] = idx
        elif "contra" in name or name in {"refute", "refuted"}:
            mapping["contradiction"] = idx
        elif "neutral" in name or "unknown" in name or name in {"unsupport", "unsupported", "no"}:
            mapping["neutral"] = idx
    if "entailment" in mapping and len(mapping) >= 2 and all(idx >= 0 for idx in mapping.values()):
        return mapping
    return {}


def _pick_device(device: str) -> str:
    requested = str(device or "auto").strip().lower() or "auto"
    if requested != "auto":
        return requested
    if torch is not None:
        try:
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                return "cuda"
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


def _has_local_hf_cache(model_name: str) -> bool:
    normalized = str(model_name or "").strip()
    if not normalized:
        return False
    direct_path = Path(normalized).expanduser()
    if direct_path.exists():
        return True

    hf_snapshot_dir = f"models--{normalized.replace('/', '--')}"
    cache_roots: list[Path] = []
    hf_hub_env = str(os.getenv("HF_HUB_CACHE", "") or "").strip()
    if hf_hub_env:
        cache_roots.append(Path(hf_hub_env).expanduser())
    hf_home_env = str(os.getenv("HF_HOME", "") or "").strip()
    if hf_home_env:
        cache_roots.append(Path(hf_home_env).expanduser() / "hub")
    const_cache = str(getattr(huggingface_constants, "HF_HUB_CACHE", "") or "").strip()
    if const_cache:
        cache_roots.append(Path(const_cache).expanduser())
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in cache_roots:
        if (root / hf_snapshot_dir).exists():
            return True
    return False


def _load_backend_impl(
    model_name: str,
    *,
    backend: str,
    device: str,
    max_length: int,
    local_files_only: bool,
) -> LoadedNLIModel | None:
    resolved_backend = str(backend or "auto").strip().lower() or "auto"
    if resolved_backend in {"auto", "cross_encoder"} and CrossEncoder is not None:
        try:
            kwargs: dict[str, Any] = {
                "device": device,
                "max_length": max_length,
                "local_files_only": local_files_only,
            }
            ce_model = CrossEncoder(model_name, **kwargs)

            config_obj = getattr(getattr(ce_model, "model", None), "config", None) or getattr(
                ce_model, "config", None
            )
            label_mapping = _resolve_label_mapping(getattr(config_obj, "id2label", None))
            if not label_mapping:
                raise ValueError("NLI checkpoint lacks an entailment label mapping")

            def _predict_ce(batch: list[tuple[str, str]]) -> list[list[float]]:
                raw = ce_model.predict(batch, apply_softmax=False, activation_fn=torch.nn.Identity() if torch is not None else None)
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                if raw and not isinstance(raw[0], (list, tuple)):
                    # A binary logit represents the log-odds of class 1.
                    return [[0.0, float(val)] for val in raw]
                return [[float(x) for x in row] for row in raw]

            return LoadedNLIModel(
                backend="cross_encoder",
                model_name=model_name,
                label_mapping=label_mapping,
                predict_logits_fn=_predict_ce,
                device=device,
            )
        except Exception as exc:
            logger.debug("CrossEncoder load failed (%s): %s", model_name, exc)

    if (
        resolved_backend in {"auto", "transformers_nli"}
        and AutoModelForSequenceClassification is not None
        and AutoTokenizer is not None
        and torch is not None
    ):
        try:
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            hf_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            if hasattr(hf_model, "to"):
                hf_model = hf_model.to(device)
            if hasattr(hf_model, "eval"):
                hf_model.eval()
            label_mapping = _resolve_label_mapping(
                getattr(getattr(hf_model, "config", None), "id2label", None)
            )

            if not label_mapping:
                raise ValueError("NLI checkpoint lacks an entailment label mapping")

            def _predict_hf(batch: list[tuple[str, str]]) -> list[list[float]]:
                encoded = tok(
                    [p for p, _ in batch],
                    [h for _, h in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                if device != "cpu" and hasattr(encoded, "to"):
                    encoded = encoded.to(device)
                with torch.no_grad():
                    outputs = hf_model(**encoded)
                    logits = outputs.logits.detach().cpu().tolist()
                return [[float(x) for x in row] for row in logits]

            return LoadedNLIModel(
                backend="transformers_nli",
                model_name=model_name,
                label_mapping=label_mapping,
                predict_logits_fn=_predict_hf,
                device=device,
            )
        except Exception as exc:
            logger.debug("Transformers NLI load failed (%s): %s", model_name, exc)

    return None


def load_nli_model(
    model_name: str,
    *,
    backend: str = "auto",
    device: str = "auto",
    max_length: int = 512,
    allow_download: bool = False,
) -> LoadedNLIModel | None:
    """Thread-safe process-level singleton NLI loader with strict offline cache guard."""

    normalized = str(model_name or "").strip()
    if not normalized or backend == "disabled":
        return None

    cache_key = f"{backend}::{normalized}::{device}::{max_length}"
    with _NLI_LOCK:
        if cache_key in _NLI_CACHE:
            return _NLI_CACHE[cache_key]

        has_cache = _has_local_hf_cache(normalized)
        if not has_cache and not allow_download:
            return None

        resolved_device = _pick_device(device)
        loaded = _load_backend_impl(
            normalized,
            backend=backend,
            device=resolved_device,
            max_length=max_length,
            local_files_only=has_cache or not allow_download,
        )
        if loaded is not None:
            _NLI_CACHE[cache_key] = loaded
        return loaded


class LocalNLIScorer:
    """Batched local NLI scorer with temperature calibration and stage budget cutoff."""

    def __init__(
        self,
        config: Any,
        *,
        loaded_model: LoadedNLIModel | None = None,
        temperature: float = 1.15,
    ) -> None:
        self._config = config
        self._loaded_model = loaded_model
        self._temperature = max(0.1, float(temperature or 1.15))
        self.last_backend_used: str = "none"

    @staticmethod
    def _softmax_calibrated(logits: list[float], temperature: float = 1.15) -> list[float]:
        if not logits:
            return []
        scaled = [float(val) / max(float(temperature or 1.15), 0.1) for val in logits]
        peak = max(scaled)
        exps = [math.exp(val - peak) for val in scaled]
        denom = sum(exps) or 1.0
        return [val / denom for val in exps]

    def _resolve_model(self) -> LoadedNLIModel | None:
        if self._loaded_model is not None:
            return self._loaded_model
        model = load_nli_model(
            str(getattr(self._config, "nli_model_name", "") or ""),
            backend=str(getattr(self._config, "nli_model_backend", "auto") or "auto"),
            device=str(getattr(self._config, "nli_device", "auto") or "auto"),
            max_length=int(getattr(self._config, "nli_max_length", 512) or 512),
        )
        return model

    def _logits_to_prob_dict(
        self,
        logits: list[float],
        label_mapping: dict[str, int],
    ) -> dict[str, float]:
        probs = self._softmax_calibrated(logits, temperature=self._temperature)
        n = len(probs)
        ent_idx = label_mapping.get("entailment", -1)
        con_idx = label_mapping.get("contradiction", -1)
        neu_idx = label_mapping.get("neutral")
        p_ent = probs[ent_idx] if 0 <= ent_idx < n else 0.0
        p_con = probs[con_idx] if 0 <= con_idx < n else 0.0
        if neu_idx is not None and 0 <= neu_idx < n:
            p_neu = probs[neu_idx]
        else:
            p_neu = max(0.0, 1.0 - p_ent - p_con)
        return {
            "entailment": float(p_ent),
            "contradiction": float(p_con),
            "neutral": float(p_neu),
        }

    def _predict_batch(
        self, model: LoadedNLIModel, batch: list[tuple[str, str]],
    ) -> list[dict[str, float]] | None:
        if not model.predict_lock.acquire(blocking=False):
            return None
        try:
            rows = model.predict_logits_fn(batch)
        finally:
            model.predict_lock.release()
        if not isinstance(rows, list) or len(rows) != len(batch):
            return None
        indices = set(model.label_mapping.values())
        if "entailment" not in model.label_mapping:
            return None
        if any(
            not isinstance(row, (list, tuple))
            or indices != set(range(len(row)))
            or len(model.label_mapping) != len(row)
            or not all(math.isfinite(value) for value in row)
            for row in rows
        ):
            return None
        return [self._logits_to_prob_dict(list(row), model.label_mapping) for row in rows]

    def score_pairs_batched(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        deadline_monotonic: float,
    ) -> list[dict[str, float]] | None:
        if not pairs:
            return []
        self.last_backend_used = "none"
        model = _within_budget(self._resolve_model, deadline_monotonic)
        if model is None:
            return None
        self.last_backend_used = model.backend

        batch_size = max(1, int(getattr(self._config, "nli_batch_size", 16) or 16))
        if model.device == "cpu":
            batch_size = min(batch_size, 8)
        results: list[dict[str, float]] = []
        for offset in range(0, len(pairs), batch_size):
            batch = list(pairs[offset : offset + batch_size])
            probs_batch = _within_budget(lambda batch=batch: self._predict_batch(model, batch), deadline_monotonic)
            if probs_batch is None:
                break
            results.extend(probs_batch)
        return results or None


def _normalize_entity(raw: str) -> str:
    cleaned = re.sub(r"[\s_\-]+", "", str(raw or "").strip().lower())
    return cleaned[:64]


def _normalize_period(raw_text: str) -> str:
    text = unicodedata.normalize("NFKC", str(raw_text or ""))
    q_match = re.search(
        r"((?:19|20)\d{2})\s*(?:年\s*)?(?:Q\s*([1-4])|第?\s*([一二三四1-4])季度)"
        r"|Q([1-4])\s*[-/ ]\s*((?:19|20)\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    if q_match:
        year, english_q, chinese_q, leading_q, trailing_year = q_match.groups()
        quarter = english_q or chinese_q or leading_q
        quarter = {"一": "1", "二": "2", "三": "3", "四": "4"}.get(quarter, quarter)
        return f"{year or trailing_year}Q{quarter}"
    year_text = re.sub(r"(?<![a-zA-Z])(?:USD|RMB|CNY|EUR|GBP|JPY|HKD|[$¥€£])\s*\d[\d,.]*", "", text)
    fy_match = re.search(r"(?<![a-zA-Z0-9.])(?:FY\s*)?((?:19|20)\d{2})(?![\d.]|\s*(?:%|元|万|亿|USD|CNY))",
                         year_text, flags=re.IGNORECASE)
    if fy_match:
        return fy_match[1]
    ver_match = re.search(r"\bv(\d+(?:\.\d+){1,2})\b", text, flags=re.IGNORECASE)
    return f"v{ver_match[1]}" if ver_match else ""


def _normalize_metric(raw_text: str) -> str:
    lowered = unicodedata.normalize("NFKC", str(raw_text or "")).strip().lower()
    lowered = re.sub(r"[_-]+", " ", lowered)
    if not lowered:
        return ""
    # Keep distinct percentiles, growth vs. level, and financing vs. valuation.
    percentile = re.search(r"\bp\s*(\d{2,3})\s*[- ]?\s*(?:latency|延迟)", lowered)
    if percentile:
        return f"p{percentile[1]}_latency"
    if re.search(r"首字延迟|\bttft\b", lowered):
        return "ttft"
    if re.search(r"估值|\bvaluation\b", lowered):
        return "valuation"
    matched = [
        metric for metric, aliases in _METRIC_ALIASES.items()
        if lowered == metric.replace("_", " ")
        or any(
            alias in lowered if not alias.isascii() else bool(re.search(rf"\b{re.escape(alias)}\b", lowered))
            for alias in aliases
        )
    ]
    if "growth_rate" in matched:
        base = next((metric for metric in matched if metric != "growth_rate"), "")
        if base:
            return f"{base}_growth_rate"
    if matched:
        return matched[0]
    if re.search(r"\bmargin\s*:", lowered):
        return "gross_margin"
    return re.sub(r"\s+", "_", lowered)[:40] if len(lowered) <= 30 else ""


def normalize_conflict_key(entity: str, metric: str, period: str) -> tuple[str, str, str]:
    return (
        _normalize_entity(unicodedata.normalize("NFKC", entity)),
        _normalize_metric(metric),
        _normalize_period(period) or re.sub(r"\s+", "", period).casefold(),
    )


_PERIOD_YEAR = re.compile(
    r"(?<![a-z0-9.])(?:FY\s*)?(?:19|20)\d{2}\s*(?:年\s*)?(?:Q\s*[1-4](?!\d)|第?\s*[一二三四1-4]季度)|"
    r"(?<![a-z0-9.])FY\s*(?:19|20)\d{2}(?!\d)|"
    r"(?<!\d)(?:19|20)\d{2}(?=\s*年)|"
    r"(?<!\w)Q[1-4]\s*[-/ ]\s*(?:19|20)\d{2}(?!\d)",
    re.IGNORECASE,
)
_VERSION = re.compile(r"(?<![a-z0-9])v\d+(?:\.\d+)+", re.IGNORECASE)
_CURRENCY = r"(?:USD|CNY|RMB|EUR|GBP|JPY|HKD|US\$|HK\$|[$¥€£]|人民币|美元|港元|欧元|日元|英镑)"
_SCALE = r"(?:万亿|亿|万|千|trillion\b|billion\b|million\b|thousand\b|tn\b|bn\b|mn\b|[kmbt]\b)"
_QUANTITY = re.compile(
    rf"(?<![a-z0-9.])(?P<prefix>{_CURRENCY})?\s*(?P<number>[+\-−]?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?|[+\-−]?\.\d+)"
    rf"(?:\s*(?P<scale>{_SCALE})(?![a-zA-Z]))?"
    rf"(?:\s*(?P<unit>percentage\s+points?\b|percent(?:age)?\b|p\.?p\.?\b|basis\s+points?\b|bps?\b|个百分点|基点|%|"
    rf"{_CURRENCY}(?![a-zA-Z])|元|milliseconds?\b|seconds?\b|ms\b|s\b|x\b|倍))?",
    re.IGNORECASE,
)
_SCALE_FACTORS = {
    "k": 1e3, "thousand": 1e3, "千": 1e3, "万": 1e4,
    "m": 1e6, "mn": 1e6, "million": 1e6, "亿": 1e8,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12, "万亿": 1e12,
}
_CURRENCY_UNITS = {
    "$": "usd", "us$": "usd", "usd": "usd", "美元": "usd",
    "¥": "cny", "元": "cny", "cny": "cny", "rmb": "cny", "人民币": "cny",
    "€": "eur", "eur": "eur", "欧元": "eur", "£": "gbp", "gbp": "gbp", "英镑": "gbp",
    "jpy": "jpy", "日元": "jpy", "hkd": "hkd", "hk$": "hkd", "港元": "hkd",
}
_RATE_CONTEXT = re.compile(r"%|率|份额|占比|margin|rate|share|percent|cagr|yoy", re.IGNORECASE)
_YEAR_COLUMN = re.compile(r"\b(?:year|date|period|quarter|fy)\b|年度|年份|期间|日期|季度", re.I)


def web_fact_excerpts(record: dict[str, Any]) -> list[str] | None:
    """Return verified web prose and independent table rows for numeric grounding."""
    if record.get("provenance_meta", {}).get("web_extraction", {}).get("tier") not in {"L1", "L2"}:
        return None
    excerpts = [record["full_content"]] if record.get("full_content") else []
    for block in record.get("multimodal_blocks") or ():
        if block.get("modality") != "table":
            if text := block.get("text"):
                excerpts.append(text)
            continue
        payload = block.get("structured_payload") or {}
        if payload.get("schema_version") != 1:
            if text := block.get("text") or block.get("markdown"):
                excerpts.append(text)
            continue
        unit = str(payload.get("unit_text") or "").strip()
        currency_scale = re.fullmatch(rf"({_CURRENCY})\s+({_SCALE})", unit, re.I)
        if currency_scale:
            unit = f"{currency_scale[2]} {currency_scale[1]}"
        if not _QUANTITY.fullmatch(f"1 {unit}"):
            unit = ""
        headers = payload.get("headers") or []
        for row in payload.get("rows") or []:
            cells = []
            for col, value in enumerate(row):
                label = headers[col] if col < len(headers) else ""
                if unit and not _YEAR_COLUMN.search(label) and re.fullmatch(
                    r"[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value,
                ):
                    value = f"{value} {unit}"
                cells.append(value)
            excerpts.append(" | ".join(cells))
    return excerpts


def _normalize_quantity(value: float, unit: str) -> tuple[float, str]:
    unit = unit.strip().lower()
    if unit in {"%", "percent", "percentage"}:
        return value, "%"
    if unit.rstrip(".") in {"pp", "p.p"} or unit in {"个百分点", "percentage point", "percentage points"}:
        return value, "pp"
    if unit in {"bp", "bps", "基点", "basis point", "basis points"}:
        return value / 100, "pp"
    if unit in {"s", "second", "seconds"}:
        return value * 1000, "ms"
    if unit in {"ms", "millisecond", "milliseconds"}:
        return value, "ms"
    if unit in {"倍", "x"}:
        return value, "x"
    if unit in _SCALE_FACTORS:
        return value * _SCALE_FACTORS[unit], "scalar"
    if unit in {"万元", "亿元", "万亿元", "亿美元", "万美元"}:
        scale, currency = re.fullmatch(r"(万亿|万|亿)(.+)", unit).groups()
        return value * _SCALE_FACTORS[scale], _CURRENCY_UNITS[currency]
    return value, _CURRENCY_UNITS.get(unit, unit or "scalar")


def _extract_quantities(raw_text: str) -> list[tuple[float, str]]:
    text = unicodedata.normalize("NFKC", CITATION_PATTERN.sub(" ", str(raw_text or "")))
    text = _VERSION.sub(" ", text)
    text = re.sub(r"(?<![a-z0-9])p\s*\d{2,3}(?=\s*(?:latency|延迟))", " ", text, flags=re.IGNORECASE)
    quantities: list[tuple[float, str]] = []
    cursor = 0
    while cursor < len(text):
        period = _PERIOD_YEAR.match(text, cursor)
        if period:
            canonical = _normalize_period(period[0])
            quantities.append((float(canonical[:4]), "year"))
            if len(canonical) == 6:
                quantities.append((float(canonical.replace("Q", "")), "quarter"))
            cursor = period.end()
            continue
        match = _QUANTITY.match(text, cursor)
        if not match:
            cursor += 1
            continue
        cursor = match.end()
        value = float(match["number"].replace(",", "").replace("−", "-"))
        scale = (match["scale"] or "").lower()
        suffix = (match["unit"] or "").lower()
        prefix = match["prefix"] or ""
        if prefix and suffix in _CURRENCY_UNITS and _CURRENCY_UNITS[prefix.lower()] != _CURRENCY_UNITS[suffix]:
            continue
        if not (scale or suffix or prefix) and value.is_integer() and 1900 <= value <= 2100:
            context = text[:match.start("number")].rstrip()
            if not re.search(r"(?:[=：:为达到至]|was|is|of|reached|totaled)$", context, re.IGNORECASE):
                quantities.append((value, "year"))
                continue
        normalized = _normalize_quantity(value * _SCALE_FACTORS.get(scale, 1), prefix or suffix)
        if math.isfinite(normalized[0]):
            quantities.append(normalized)
    return quantities


def _extract_numeric_and_unit(raw_text: str) -> tuple[float | None, str]:
    return next(
        ((value, unit) for value, unit in _extract_quantities(raw_text) if unit not in {"year", "quarter"}),
        (None, ""),
    )


def _comparable_values(
    left: tuple[float, str], right: tuple[float, str], *, rate_context: bool,
) -> tuple[float, float] | None:
    value, unit = left
    other, other_unit = right
    if unit == other_unit:
        return value, other
    if rate_context and {unit, other_unit} == {"%", "scalar"}:
        scalar = value if unit == "scalar" else other
        if abs(scalar) <= 1:
            return (value * 100, other) if unit == "scalar" else (value, other * 100)
    return None


def numeric_grounding_mismatch(claim_text: str, evidence_texts: Sequence[str]) -> str | None:
    """Require one cited excerpt to contain all expressed quantities in compatible units."""
    expected = _extract_quantities(claim_text)
    if not expected:
        return None
    is_rate = bool(_RATE_CONTEXT.search(claim_text))
    evidence = [_extract_quantities(text) for text in evidence_texts]

    def matches(left: tuple[float, str], right: tuple[float, str]) -> bool:
        comparable = _comparable_values(left, right, rate_context=is_rate)
        if comparable is None:
            return False
        if left[1] in {"year", "quarter"}:
            return comparable[0] == comparable[1]
        return math.isclose(*comparable, rel_tol=0.005, abs_tol=1e-9)

    for quantities in evidence:
        if all(any(matches(item, candidate) for candidate in quantities) for item in expected):
            return None
    values = "、".join(f"{value:g}{'' if unit in {'year', 'scalar'} else ' ' + unit}" for value, unit in expected)
    return f"引用证据未同时支持所述数值或单位（{values}）"


def _infer_polarity(text: str) -> Literal["positive", "negative", "neutral"]:
    lowered = str(text or "").lower()
    if any(tok in lowered for tok in _NEGATIVE_MARKERS):
        return "negative"
    if any(tok in lowered for tok in _POSITIVE_MARKERS):
        return "positive"
    return "neutral"


def _extract_entity_candidate(text: str) -> str:
    cleaned = CITATION_PATTERN.sub("", str(text or "")).strip()
    # Look for capitalized/alphanumeric product/org names or leading Chinese subject chunk
    en_match = re.search(r"\b([A-Za-z][A-Za-z0-9.\-]{1,24})\b", cleaned)
    if en_match:
        return _normalize_entity(en_match.group(1))
    zh_match = re.match(r"^([\u4e00-\u9fffA-Za-z0-9·]{2,12}?)(?:在|于|的|为|是|实现|营业收入|营收|毛利率|净利|延迟|已|未|\s+(?=20\d{2}))", cleaned)
    if zh_match:
        return _normalize_entity(zh_match.group(1))
    head = cleaned[:10]
    return _normalize_entity(head)


def _extract_proposition_from_text(text: str, raw_prop: dict[str, Any] | None = None) -> AtomicProposition:
    prop_data = raw_prop if isinstance(raw_prop, dict) else {}
    entity = _normalize_entity(str(prop_data.get("entity") or "")) or _extract_entity_candidate(text)
    metric = _normalize_metric(str(prop_data.get("metric") or "")) or _normalize_metric(text)
    period = _normalize_period(str(prop_data.get("period") or "")) or _normalize_period(text)
    num_val = prop_data.get("numeric_value")
    unit = str(prop_data.get("unit") or "").strip()
    text_value, text_unit = _extract_numeric_and_unit(text)
    if text_value is not None:
        num_val, unit = text_value, text_unit
    elif num_val is not None:
        try:
            num_val, unit = _normalize_quantity(float(num_val), unit)
        except (TypeError, ValueError, OverflowError):
            num_val = None
        if num_val is not None and not math.isfinite(num_val):
            num_val = None
    polarity = str(prop_data.get("polarity") or "").strip().lower()
    if polarity not in {"positive", "negative", "neutral"}:
        polarity = _infer_polarity(text)
    return AtomicProposition(
        entity=entity,
        metric=metric,
        period=period,
        numeric_value=num_val,
        unit=unit,
        polarity=polarity,  # type: ignore[arg-type]
    )


def _extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class AtomicClaimDecomposer:
    """Step 1: Decompose compound parent claims into atomic claims with citation conservation."""

    def __init__(self, *, decompose_with_fast_llm: bool = True) -> None:
        self._decompose_with_fast_llm = bool(decompose_with_fast_llm)

    @staticmethod
    def _conserve_source_ids(
        candidate_sids: Sequence[str] | None,
        *,
        parent_sids: list[str],
        valid_source_ids: set[str],
    ) -> list[str]:
        parent_valid = [s for s in parent_sids if s in valid_source_ids]
        if not candidate_sids:
            return list(parent_valid)
        parent_set = set(parent_valid)
        kept: list[str] = []
        seen: set[str] = set()
        for raw_sid in candidate_sids:
            sid = str(raw_sid or "").strip().strip("[]")
            if sid and sid in parent_set and sid not in seen:
                kept.append(sid)
                seen.add(sid)
        return kept or list(parent_valid)

    @staticmethod
    def _syntactic_split(text: str) -> list[str]:
        lines = list(iter_markdown_lines(text))
        clauses: list[str] = []
        headers: list[str] = []
        separator = re.compile(r"^\s*:?-{3,}:?\s*$")
        boundaries = re.compile(
            r"(?:[；;。!?！？]+|\.(?=\s|$))"
            rf"(?:[ \t]*(?:{CITATION_PATTERN.pattern}))*"
            r"|[，,]?\s*(?:并且|而且|同时|但是|然而|以及|且|但)"
            r"|[，,]?\s*\b(?:however|while|whereas|but|and\s+also)\b"
            r"|\s+\band\b(?=[^.!?;]*\b(?:was|were|is|are|has|had|reached|rose|fell|grew)\b)",
            re.IGNORECASE,
        )

        for index, (raw_line, fenced) in enumerate(lines):
            line = raw_line.strip()
            if fenced or not line or re.match(r"^#{1,6}\s", line):
                headers = []
                continue
            line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)
            if "|" in line:
                cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip("|"))]
                if len(cells) >= 2 and all(separator.fullmatch(cell) for cell in cells):
                    continue
                next_line = lines[index + 1][0].strip().strip("|") if index + 1 < len(lines) else ""
                if "|" in next_line and all(separator.fullmatch(cell) for cell in next_line.split("|")):
                    headers = cells
                    continue
                if headers and len(headers) == len(cells):
                    fields = []
                    shared_citations = []
                    for header, cell in zip(headers, cells):
                        plain = CITATION_PATTERN.sub("", cell).strip()
                        if not plain:
                            shared_citations.extend(CITATION_PATTERN.findall(cell))
                        else:
                            fields.append((header, cell))
                    if fields:
                        subject_header, subject = fields[0]
                        if subject_header.strip().casefold() in {"entity", "company", "name", "metric", "指标", "主体", "公司", "名称"}:
                            fields = fields[1:] or [(subject_header, subject)]
                            subject = CITATION_PATTERN.sub("", subject).strip()
                        else:
                            subject = ""
                        for header, cell in fields:
                            suffix = "".join(f"[{sid}]" for sid in shared_citations)
                            clauses.append(f"{subject} {header}: {cell} {suffix}".strip())
                else:
                    clauses.append(" | ".join(cells))
                continue
            headers = []
            start = 0
            for boundary in boundaries.finditer(line):
                markers = CITATION_PATTERN.findall(boundary[0])
                clause = line[start:boundary.start()].strip(" ，,;；。")
                if markers:
                    clause += " " + "".join(f"[{sid}]" for sid in markers)
                if clause.strip():
                    clauses.append(clause.strip())
                start = boundary.end()
            tail = line[start:].strip(" ，,;；。")
            if tail:
                clauses.append(tail)

        merged: list[str] = []
        for clause in clauses:
            if not CITATION_PATTERN.sub("", clause).strip(" \t.,，。;；()（）"):
                if merged:
                    merged[-1] += " " + clause
            else:
                merged.append(clause)
        return merged

    def _decompose_via_fast_llm(
        self,
        *,
        task_id: int,
        parent_claims: list[dict[str, Any]],
        fast_llm: Any,
    ) -> dict[int, list[dict[str, Any]]] | None:
        prompt_items = [
            {
                "index": idx,
                "text": str(claim.get("text") or ""),
                "source_ids": list(claim.get("source_ids") or []),
            }
            for idx, claim in enumerate(parent_claims)
            if str(claim.get("text") or "").strip()
        ]
        if not prompt_items:
            return None

        system_msg = (
            "You decompose compound research claims into atomic verifiable claims. "
            "For each atomic sub-claim, assign only source_ids from its parent's source_ids subset. "
            "Output strict JSON only: {\"claims\": [{\"parent_index\": 0, \"atomics\": [{\"text\": \"...\", "
            "\"source_ids\": [\"T1-S1\"], \"proposition\": {\"entity\": \"\", \"metric\": \"\", "
            "\"period\": \"\", \"numeric_value\": null, \"unit\": \"\", \"polarity\": \"neutral\"}}]}]}."
        )
        user_msg = json.dumps({"task_id": task_id, "claims": prompt_items}, ensure_ascii=False)
        try:
            if hasattr(fast_llm, "invoke"):
                resp = fast_llm.invoke(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                )
            elif callable(fast_llm):
                resp = fast_llm(f"{system_msg}\n\n{user_msg}")
            else:
                return None
            raw_text = getattr(resp, "content", resp)
            parsed = _extract_first_json_object(str(raw_text or ""))
            if not parsed or not isinstance(parsed.get("claims"), list):
                return None
            by_parent: dict[int, list[dict[str, Any]]] = {}
            for entry in parsed["claims"]:
                if not isinstance(entry, dict):
                    continue
                try:
                    p_idx = int(entry.get("parent_index", -1))
                except (TypeError, ValueError):
                    continue
                atomics = entry.get("atomics")
                if p_idx >= 0 and isinstance(atomics, list):
                    by_parent[p_idx] = [a for a in atomics if isinstance(a, dict)]
            return by_parent or None
        except Exception as exc:
            logger.debug("Fast LLM atomic decomposition degraded task=%s: %s", task_id, exc)
            return None

    def decompose(
        self,
        *,
        task_id: int,
        parent_claims: list[dict[str, Any]],
        valid_source_ids: set[str],
        fast_llm: Any | None = None,
        max_atomic_per_task: int | None = 12,
    ) -> list[AtomicClaimRecord]:
        if not parent_claims or (max_atomic_per_task is not None and max_atomic_per_task <= 0):
            return []

        splits = [self._syntactic_split(str(claim.get("text") or "")) for claim in parent_claims]
        has_compound = any(
            len(str(claim.get("text") or "")) >= 35 or len(claim.get("source_ids") or []) > 1 or len(parts) > 1
            for claim, parts in zip(parent_claims, splits)
        )
        llm_results: dict[int, list[dict[str, Any]]] | None = None
        if self._decompose_with_fast_llm and fast_llm is not None and has_compound:
            llm_results = self._decompose_via_fast_llm(
                task_id=task_id,
                parent_claims=parent_claims,
                fast_llm=fast_llm,
            )

        records: list[AtomicClaimRecord] = []
        for p_idx, (claim, clauses) in enumerate(zip(parent_claims, splits)):
            if not clauses:
                continue
            parent_sids = list(dict.fromkeys(str(s).strip() for s in (claim.get("source_ids") or []) if s))
            inline_ids = [CITATION_PATTERN.findall(clause) for clause in clauses]
            candidates: list[tuple[str, Sequence[str] | None, dict[str, Any] | None]] = []
            raw_atomics = (llm_results or {}).get(p_idx, [])
            # Explicit spans control attribution.
            if not any(inline_ids):
                for atomic_item in raw_atomics:
                    text = str(atomic_item.get("text") or "").strip()
                    source_ids = atomic_item.get("source_ids")
                    if CITATION_PATTERN.sub("", text).strip():
                        candidates.append((
                            text,
                            source_ids if isinstance(source_ids, (list, tuple)) else None,
                            atomic_item.get("proposition"),
                        ))
                parent_numbers = set(_extract_quantities(str(claim.get("text") or "")))
                child_numbers = {value for text, _, _ in candidates for value in _extract_quantities(text)}
                if len(candidates) < len(clauses) or not parent_numbers.issubset(child_numbers):
                    candidates.clear()
            use_rule_split = not candidates
            if use_rule_split:
                trailing = inline_ids[-1] if not any(inline_ids[:-1]) else []
                explicitly_scoped = any(inline_ids)
                candidates = [
                    (clause, bound_ids or trailing or ([] if explicitly_scoped else None), None)
                    for clause, bound_ids in zip(clauses, inline_ids)
                ]

            for a_idx, (text, source_ids, raw_prop) in enumerate(candidates):
                text = normalize_agent_markdown(CITATION_PATTERN.sub("", text)).strip()
                if not text:
                    continue
                effective_sids = self._conserve_source_ids(
                    source_ids,
                    parent_sids=parent_sids,
                    valid_source_ids=valid_source_ids,
                )
                if use_rule_split and source_ids is not None and not set(source_ids).intersection(parent_sids, valid_source_ids):
                    effective_sids = []
                records.append(
                    AtomicClaimRecord(
                        atomic_id=f"{task_id}-C{p_idx}-A{a_idx}",
                        task_id=task_id,
                        parent_claim_index=p_idx,
                        text=text,
                        source_ids=effective_sids,
                        proposition=_extract_proposition_from_text(text, raw_prop),
                    )
                )
                if max_atomic_per_task is not None and len(records) >= max_atomic_per_task:
                    return records
        return records


def _inline_chunk_and_bm25_top_k(
    *,
    query: str,
    records: list[dict[str, Any]],
    top_k_per_source: int,
    web_extraction_enabled: bool = False,
) -> list[EvidenceRetrievalHit]:
    query_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?", query.lower()))
    hits: list[EvidenceRetrievalHit] = []
    for rec in records:
        sid = str(rec.get("source_id") or "").strip()
        if not sid:
            continue
        texts: list[tuple[str, str]] = []
        body = rec.get("full_content")
        if not (web_extraction_enabled and rec.get("source_origin") == "web"):
            body = body or rec.get("raw_content") or rec.get("snippet")
        full = str(body or "").strip()
        if full:
            for idx in range(0, min(len(full), 6000), 600):
                window = full[idx : idx + 680].strip()
                if window:
                    texts.append((f"{sid}::inline::{len(texts) + 1}", window))
        for mm_idx, block in enumerate(rec.get("multimodal_blocks") or [], start=1):
            if isinstance(block, dict):
                mm_text = str(block.get("markdown") or block.get("text") or block.get("summary") or "").strip()
                if mm_text:
                    texts.append((f"{sid}::mm::{block.get('anchor_id') or mm_idx}", mm_text))

        scored_chunks: list[tuple[float, str, str]] = []
        for chunk_id, chunk_text in texts:
            chunk_lower = chunk_text.lower()
            overlap = sum(1 for tok in query_tokens if tok in chunk_lower) if query_tokens else 0
            score = (overlap / max(len(query_tokens), 1)) + (0.01 if overlap == 0 else 0.2)
            scored_chunks.append((score, chunk_id, chunk_text))
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        for score, chunk_id, chunk_text in scored_chunks[: max(1, top_k_per_source)]:
            hits.append(
                EvidenceRetrievalHit(
                    origin="current",
                    citation_eligible=True,
                    task_id=int(rec.get("task_id") or 0),
                    source_id=sid,
                    archive_doc_key=rec.get("archive_doc_key"),
                    archive_version_id=rec.get("archive_version_id"),
                    chunk_id=chunk_id,
                    text=chunk_text,
                    url=str(rec.get("url") or ""),
                    title=str(rec.get("title") or sid),
                    domain=str(rec.get("domain") or ""),
                    quality_label=str(rec.get("quality_label") or "medium"),
                    freshness_label=str(rec.get("freshness_label") or "unknown"),
                    score=float(score),
                )
            )
    return hits


class ScopedEvidenceRetriever:
    """Step 2: Recall top-K evidence chunks strictly inside an atomic claim's cited source_ids."""

    def fetch_for_claim(
        self,
        atomic_claim: AtomicClaimRecord,
        *,
        retrieval_service: EvidenceRetrievalService | None,
        evidence_store: Any | None,
        cited_records_fallback: list[dict[str, Any]],
        top_k_per_source: int = 3,
    ) -> list[EvidenceRetrievalHit]:
        allowed_list = list(dict.fromkeys(sid.strip() for sid in atomic_claim.source_ids if sid.strip()))
        web_enabled = getattr(evidence_store, "web_extraction_enabled", False)
        if web_enabled:
            allowed_list = [sid for sid in allowed_list if evidence_store.is_eligible_for_facts(sid)]
        if not allowed_list:
            return []
        allowed_set = set(allowed_list)

        by_source: dict[str, list[EvidenceRetrievalHit]] = defaultdict(list)
        if retrieval_service is not None:
            result = retrieval_service.query(
                atomic_claim.text,
                task_id=atomic_claim.task_id,
                scope="current",
                mode="grounding",
                top_k=len(allowed_list) * top_k_per_source,
                allowed_source_ids=allowed_list,
            )
            seen_chunks: set[tuple[str, str]] = set()
            for hit in result.hits:
                key = (hit.source_id, hit.chunk_id)
                if (
                    hit.source_id in allowed_set and hit.origin == "current" and hit.citation_eligible
                    and hit.text.strip() and key not in seen_chunks
                    and len(by_source[hit.source_id]) < top_k_per_source
                ):
                    by_source[hit.source_id].append(hit)
                    seen_chunks.add(key)

        missing = [sid for sid in allowed_list if not by_source.get(sid)]
        if missing:
            records = {
                rec["source_id"]: rec for rec in cited_records_fallback
                if not web_enabled and isinstance(rec, dict) and rec.get("source_id") in missing
            }
            if evidence_store is not None:
                records.update({
                    rec["source_id"]: rec
                    for rec in evidence_store.lookup(source_ids=missing, include_full_content=True)
                    if isinstance(rec, dict) and rec.get("source_id") in missing
                    and (rec.get("full_content") or rec.get("raw_content") or rec.get("snippet") or rec.get("multimodal_blocks"))
                })
            for hit in _inline_chunk_and_bm25_top_k(
                query=atomic_claim.text,
                records=list(records.values()),
                top_k_per_source=top_k_per_source,
                web_extraction_enabled=web_enabled,
            ):
                by_source[hit.source_id].append(hit)
        return [hit for sid in allowed_list for hit in by_source[sid]]


def _authority_source_type(record: dict[str, Any]) -> str:
    source_type = str(record.get("source_type") or "").strip().lower()
    origin = str(record.get("source_origin") or "").strip().lower()
    if origin == "sandbox":
        return "computed_sandbox" if record.get("derived_from_source_ids") else "web_general"
    if origin == "mcp" or source_type.startswith("mcp_"):
        return source_type if source_type.startswith("mcp_") and source_type in SOURCE_TIER_WEIGHT else "mcp_tool"
    if source_type not in SOURCE_TIER_WEIGHT or source_type in {"web", "web_general"}:
        return detect_source_type(record)
    return source_type


def _compute_authority_rank(record: dict[str, Any], nli_support: float) -> float:
    tier = SOURCE_TIER_WEIGHT.get(_authority_source_type(record), 0.40)
    try:
        reliability = float(record.get("source_reliability_score", 5.0))
    except (TypeError, ValueError):
        reliability = 5.0
    if not math.isfinite(reliability):
        reliability = 5.0
    reliability_norm = max(0.0, min(1.0, reliability / 10.0))
    return (0.65 * tier) + (0.25 * reliability_norm) + (0.10 * max(0.0, min(1.0, nli_support)))


def _publication_day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _adjudicate_cluster(sides: list[ConflictSide]) -> tuple[AdjudicationStatus, str | None, str]:
    ranked = sorted(
        sides,
        key=lambda s: (s.authority_rank, _publication_day(s.published_at) or date.min, s.source_id),
        reverse=True,
    )
    top, runner_up = ranked[0], ranked[1]
    tier_gap = top.authority_rank - runner_up.authority_rank

    if tier_gap >= 0.22:
        return (
            "resolved_by_authority",
            top.source_id,
            f"优先采信权威等级更高的来源《{top.title}》（{top.domain or top.source_type}）[{top.source_id}]，"
            f"同时保留《{runner_up.title}》[{runner_up.source_id}] 口径供参考。",
        )
    dated = [(day, side) for side in ranked if (day := _publication_day(side.published_at))]
    if len(dated) == len(ranked):
        dated.sort(key=lambda item: item[0], reverse=True)
    if (
        len(dated) == len(ranked)
        and (dated[0][0].year, dated[0][0].month) != (dated[1][0].year, dated[1][0].month)
        and top.authority_rank - dated[0][1].authority_rank < 0.22
    ):
        newer = dated[0][1]
        older = runner_up if newer is top else top
        return (
            "resolved_by_recency",
            newer.source_id,
            f"两个来源权威度接近，基于发布时间优先采信较新口径（{newer.published_at}，[{newer.source_id}]），"
            f"早先数据（{older.published_at}，[{older.source_id}]）标注为历史口径。",
        )
    return (
        "unresolved_divergence",
        None,
        f"多个同等级或高可信度来源存在实质口径分歧（[{top.source_id}] 与 [{runner_up.source_id}]），"
        "暂不做单边否定，在报告中并列呈现双方依据与边界。",
    )


class ConflictGraphBuilder:
    """Step 4: Cluster supported/neutral atomic claims by proposition and detect cross-source conflicts."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def build(
        self,
        atomic_claims: list[AtomicClaimRecord],
        *,
        evidence_by_id: dict[str, dict[str, Any]],
        scorer: LocalNLIScorer | None = None,
        deadline_monotonic: float | None = None,
    ) -> list[EvidenceConflictCluster]:
        eligible = [
            claim
            for claim in atomic_claims
            if claim.nli_label == "ENTAILED"
            and claim.best_source_id in claim.source_ids
            and claim.best_source_id in evidence_by_id
            and claim.proposition.entity
            and (claim.proposition.metric or claim.proposition.numeric_value is not None)
        ]
        if len(eligible) < 2:
            return []

        buckets: dict[tuple[str, str, str], list[AtomicClaimRecord]] = defaultdict(list)
        for claim in eligible:
            prop = claim.proposition
            key = normalize_conflict_key(prop.entity, prop.metric or "statement", prop.period or "general")
            buckets[key].append(claim)

        rel_tol = float(getattr(self._config, "nli_conflict_numeric_rel_tol", 0.05))
        min_nli_score = float(getattr(self._config, "nli_conflict_min_score", 0.70))

        clusters: list[EvidenceConflictCluster] = []
        for (entity, metric, period), group_claims in sorted(buckets.items()):
            # Pick representative atomic claim per distinct source_id
            by_source: dict[str, AtomicClaimRecord] = {}
            for c in sorted(group_claims, key=lambda claim: (-claim.calibrated_confidence, claim.atomic_id, claim.text)):
                sid = c.best_source_id or (c.source_ids[0] if c.source_ids else "")
                if not sid:
                    continue
                prev = by_source.get(sid)
                if prev is None or c.calibrated_confidence > prev.calibrated_confidence:
                    by_source[sid] = c
            if len(by_source) < 2:
                continue

            sides_list: list[ConflictSide] = []
            for sid, c in sorted(by_source.items()):
                rec = evidence_by_id.get(sid) or {}
                nli_supp = float(c.raw_probs.get("entailment", c.calibrated_confidence))
                sides_list.append(
                    ConflictSide(
                        source_id=sid,
                        task_id=c.task_id,
                        title=str(rec.get("title") or sid),
                        url=str(rec.get("url") or ""),
                        domain=str(rec.get("domain") or ""),
                        source_type=_authority_source_type(rec),
                        quality_label=str(rec.get("quality_label") or "medium"),
                        published_at=str(rec["published_at"]) if rec.get("published_at") else None,
                        claim_text=c.text,
                        normalized_value=c.proposition.numeric_value,
                        unit=c.proposition.unit,
                        evidence_excerpt=truncate_text(
                            c.matched_excerpt or str(rec.get("snippet") or rec.get("full_content") or ""),
                            220,
                        ),
                        authority_rank=_compute_authority_rank(rec, nli_supp),
                        nli_support_score=nli_supp,
                    )
                )

            edges: list[dict[str, Any]] = []
            conflict_types: set[str] = set()
            max_score = 0.0
            cross_nli_pairs: list[tuple[int, int]] = []

            for i in range(len(sides_list)):
                for j in range(i + 1, len(sides_list)):
                    sa, sb = sides_list[i], sides_list[j]
                    if sa.url and sa.url == sb.url:
                        continue
                    if (
                        sa.source_id in (evidence_by_id[sb.source_id].get("derived_from_source_ids") or [])
                        or sb.source_id in (evidence_by_id[sa.source_id].get("derived_from_source_ids") or [])
                    ):
                        continue
                    va, vb = sa.normalized_value, sb.normalized_value
                    comparable = (
                        _comparable_values((va, sa.unit), (vb, sb.unit), rate_context=bool(_RATE_CONTEXT.search(metric)))
                        if va is not None and vb is not None else None
                    )
                    if comparable is not None:
                        left, right = comparable
                        delta_ratio = abs(left - right) / max(abs(left), abs(right), 1e-9)
                        if delta_ratio > rel_tol:
                            score = min(1.0, 0.70 + 0.30 * min(delta_ratio, 1.0))
                            max_score = max(max_score, score)
                            conflict_types.add("numeric_divergence")
                            edges.append(
                                {
                                    "source_id_a": sa.source_id,
                                    "source_id_b": sb.source_id,
                                    "contradiction_score": round(score, 4),
                                    "delta_ratio": round(delta_ratio, 4),
                                    "kind": "numeric_divergence",
                                }
                            )
                        continue
                    if va is not None and vb is not None:
                        continue
                    # Polarity / Semantic branch
                    ca, cb = by_source[sa.source_id], by_source[sb.source_id]
                    if (
                        {ca.proposition.polarity, cb.proposition.polarity} == {"positive", "negative"}
                    ):
                        score = 0.80
                        max_score = max(max_score, score)
                        conflict_types.add("semantic_contradiction")
                        edges.append(
                            {
                                "source_id_a": sa.source_id,
                                "source_id_b": sb.source_id,
                                "contradiction_score": score,
                                "delta_ratio": None,
                                "kind": "semantic_contradiction",
                            }
                        )
                    elif scorer is not None and deadline_monotonic is not None:
                        cross_nli_pairs.append((i, j))

            if cross_nli_pairs and scorer is not None and deadline_monotonic is not None:
                nli_inputs: list[tuple[str, str]] = []
                for i, j in cross_nli_pairs:
                    sa, sb = sides_list[i], sides_list[j]
                    nli_inputs.append((sa.evidence_excerpt or sa.claim_text, sb.claim_text))
                    nli_inputs.append((sb.evidence_excerpt or sb.claim_text, sa.claim_text))
                nli_scores = scorer.score_pairs_batched(
                    nli_inputs,
                    deadline_monotonic=deadline_monotonic,
                )
                if nli_scores:
                    for idx, (i, j) in enumerate(cross_nli_pairs):
                        pair_scores = nli_scores[2 * idx:2 * idx + 2]
                        if not pair_scores:
                            break
                        pair_score = max(score["contradiction"] for score in pair_scores)
                        if pair_score >= min_nli_score:
                            max_score = max(max_score, pair_score)
                            conflict_types.add("semantic_contradiction")
                            edges.append(
                                {
                                    "source_id_a": sides_list[i].source_id,
                                    "source_id_b": sides_list[j].source_id,
                                    "contradiction_score": round(pair_score, 4),
                                    "delta_ratio": None,
                                    "kind": "semantic_contradiction",
                                }
                            )

            if not edges:
                continue

            edge_sources = {edge[key] for edge in edges for key in ("source_id_a", "source_id_b")}
            sides_list = [side for side in sides_list if side.source_id in edge_sources]
            adjudication_status, recommended_sid, rationale = _adjudicate_cluster(sides_list)
            if recommended_sid and not all(recommended_sid in (edge["source_id_a"], edge["source_id_b"]) for edge in edges):
                adjudication_status = "unresolved_divergence"
                recommended_sid = None
                rationale = "多个来源组合存在分歧，报告并列呈现各方依据与口径。"
            ctype: ConflictType = (
                "numeric_divergence" if "numeric_divergence" in conflict_types else "semantic_contradiction"
            )
            if adjudication_status in {"resolved_by_authority", "resolved_by_recency"}:
                severity: Literal["high", "medium", "low"] = "low"
            elif ctype == "numeric_divergence":
                severity = "high"
            else:
                severity = "medium"

            digest = hashlib.sha1(f"{entity}|{metric}|{period}".encode()).hexdigest()[:8]
            clusters.append(
                EvidenceConflictCluster(
                    conflict_id=f"CONF-{digest}",
                    entity=entity,
                    metric=metric,
                    period=period,
                    conflict_type=ctype,
                    severity=severity,
                    contradiction_score=round(max_score, 4),
                    sides=sides_list,
                    edges=edges,
                    adjudication_status=adjudication_status,
                    recommended_source_id=recommended_sid,
                    adjudication_rationale=rationale,
                    affected_task_ids=sorted({
                        claim.task_id for claim in group_claims
                        if claim.best_source_id in edge_sources and claim.task_id > 0
                    }),
                )
            )
        return clusters


class ClaimVerificationOrchestrator:
    """Orchestrates Step 1..4 atomic claim verification & conflict graph inside ReviewService."""

    def __init__(
        self,
        config: Any,
        *,
        retrieval_service: EvidenceRetrievalService | None = None,
        evidence_store: Any | None = None,
        fast_llm: Any | None = None,
        decomposer: AtomicClaimDecomposer | None = None,
        retriever: ScopedEvidenceRetriever | None = None,
        scorer: LocalNLIScorer | None = None,
        conflict_builder: ConflictGraphBuilder | None = None,
    ) -> None:
        self._config = config
        self._retrieval_service = retrieval_service
        self._evidence_store = evidence_store
        self._fast_llm = fast_llm
        self._decomposer = decomposer or AtomicClaimDecomposer(
            decompose_with_fast_llm=bool(getattr(config, "nli_decompose_with_fast_llm", True)),
        )
        self._retriever = retriever or ScopedEvidenceRetriever()
        self._scorer = scorer or LocalNLIScorer(config)
        self._conflict_builder = conflict_builder or ConflictGraphBuilder(config)

    def _collect_evidence_map(self, state: Any) -> dict[str, dict[str, Any]]:
        store = self._evidence_store
        if getattr(store, "web_extraction_enabled", False):
            return {
                item["source_id"]: item for item in store.lookup()
                if store.is_eligible_for_facts(item["source_id"])
            }
        by_id: dict[str, dict[str, Any]] = {}
        for task in getattr(state, "todo_items", []) or []:
            for item in getattr(task, "evidence_items", None) or []:
                if isinstance(item, dict) and item.get("source_id"):
                    by_id[str(item["source_id"]).strip()] = item
        if self._evidence_store is not None:
            for item in self._evidence_store.lookup() or []:
                if isinstance(item, dict) and item.get("source_id"):
                    sid = str(item["source_id"]).strip()
                    by_id[sid] = {**by_id.get(sid, {}), **item}
        return by_id

    def verify_request(
        self,
        state: Any,
        *,
        claims_by_task: dict[int, list[dict[str, Any]]],
        observer: Any | None = None,
        fallback_claim_verifier: Callable[[str, list[dict[str, Any]]], str | None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        enabled = bool(getattr(self._config, "nli_verifier_enabled", False))
        if not enabled:
            return {"issues": [], "summary": {}, "conflicts": []}

        budget_sec = float(getattr(self._config, "nli_stage_budget_seconds", 8.0))
        deadline = time.monotonic() + budget_sec
        evidence_by_id = self._collect_evidence_map(state)
        web_enabled = getattr(self._evidence_store, "web_extraction_enabled", False)
        request_valid_ids = set(evidence_by_id.keys())
        max_atomics = int(getattr(self._config, "nli_max_atomic_claims_per_task", 12) or 12)
        top_k = int(getattr(self._config, "nli_top_k_chunks_per_source", 3) or 3)
        entail_thresh = float(getattr(self._config, "nli_entailment_threshold", 0.60))
        contra_thresh = float(getattr(self._config, "nli_contradiction_threshold", 0.65))

        all_atomics: list[AtomicClaimRecord] = []
        candidates: list[AtomicClaimRecord] = []
        atomics_by_task_parent: dict[tuple[int, int], list[AtomicClaimRecord]] = defaultdict(list)

        for task in getattr(state, "todo_items", []) or []:
            task_claims = list(claims_by_task.get(task.id) or [])
            verifiable_pairs: list[tuple[int, dict[str, Any]]] = [
                (idx, claim)
                for idx, claim in enumerate(task_claims)
                if str(claim.get("support_status") or "") not in {"missing_citation", "invalid_citation"}
                and bool(claim.get("source_ids"))
                and (not web_enabled or set(claim["source_ids"]) <= request_valid_ids)
            ]
            if not verifiable_pairs:
                continue
            subset_claims = [c for _, c in verifiable_pairs]
            index_remap = {sub_idx: orig_idx for sub_idx, (orig_idx, _) in enumerate(verifiable_pairs)}
            decompose = partial(
                self._decomposer.decompose,
                task_id=task.id,
                parent_claims=[dict(claim) for claim in subset_claims],
                valid_source_ids=request_valid_ids,
                max_atomic_per_task=None,
            )
            task_atomics = None
            if self._fast_llm is not None and getattr(self._config, "nli_decompose_with_fast_llm", True):
                task_atomics = _within_budget(partial(decompose, fast_llm=self._fast_llm), deadline)
            if task_atomics is None:
                task_atomics = decompose()

            for idx, atomic in enumerate(task_atomics):
                orig_idx = index_remap[atomic.parent_claim_index]
                atomic.parent_claim_index = orig_idx
                atomic.atomic_id = f"{task.id}-C{orig_idx}-A{len(atomics_by_task_parent[(task.id, orig_idx)])}"
                all_atomics.append(atomic)
                atomics_by_task_parent[(task.id, orig_idx)].append(atomic)
                if idx < max_atomics:
                    candidates.append(atomic)
                else:
                    atomic.degraded_reason = "nli_atomic_limit"

        # Step 2: prepare only the source-scoped pairs that fit the stage budget.
        pair_meta: list[tuple[AtomicClaimRecord, EvidenceRetrievalHit]] = []
        hits_by_atomic: dict[str, list[EvidenceRetrievalHit]] = {}
        for atomic in candidates:
            cited_fallback = [evidence_by_id[sid] for sid in atomic.source_ids if sid in evidence_by_id]
            hits = _within_budget(
                lambda claim=atomic, records=cited_fallback: self._retriever.fetch_for_claim(
                    claim,
                    retrieval_service=self._retrieval_service,
                    evidence_store=self._evidence_store,
                    cited_records_fallback=records,
                    top_k_per_source=top_k,
                ),
                deadline,
            )
            hits_by_atomic[atomic.atomic_id] = hits or []
            if hits:
                pair_meta.extend((atomic, hit) for hit in hits if hit.text)

        # Step 3: retain completed batches and fall back for incomplete atomics.
        scored_probs = self._scorer.score_pairs_batched(
            [(hit.text, atomic.text) for atomic, hit in pair_meta],
            deadline_monotonic=deadline,
        ) if pair_meta else None
        backend_used = self._scorer.last_backend_used if scored_probs else "none"
        scoring_reason = (
            "nli_budget_exhausted" if time.monotonic() >= deadline else "nli_model_unavailable"
        )
        counts = defaultdict(int)
        for atomic, _ in pair_meta:
            counts[atomic.atomic_id] += 1

        probs_by_atomic: dict[str, list[tuple[EvidenceRetrievalHit, dict[str, float]]]] = defaultdict(list)
        if scored_probs is not None:
            for (atomic, hit), probs in zip(pair_meta, scored_probs):
                probs_by_atomic[atomic.atomic_id].append((hit, probs))

        issues: list[dict[str, Any]] = []
        fallback_failures: set[str] = set()
        fallback_records: dict[str, dict[str, Any]] = {}
        entailed_cnt = contradicted_cnt = neutral_cnt = skipped_cnt = 0

        for atomic in all_atomics:
            hit_probs = probs_by_atomic.get(atomic.atomic_id)
            if hit_probs and len(hit_probs) == counts[atomic.atomic_id]:
                supported_hits = [
                    pair for pair in hit_probs
                    if pair[1]["entailment"] >= entail_thresh
                    and numeric_grounding_mismatch(atomic.text, [pair[0].text]) is None
                ]
                max_ent_pair = max(supported_hits, key=lambda hp: hp[1]["entailment"], default=None)
                max_con_pair = max(hit_probs, key=lambda hp: hp[1]["contradiction"])
                max_ent = max_ent_pair[1]["entailment"] if max_ent_pair else 0.0
                max_con = max_con_pair[1]["contradiction"]

                if max_con >= contra_thresh and max_con > max_ent + 0.10:
                    atomic.nli_label = "CONTRADICTED"
                    atomic.raw_probs = max_con_pair[1]
                    atomic.calibrated_confidence = max_con
                    atomic.best_source_id = max_con_pair[0].source_id
                    atomic.matched_chunk_ids = [max_con_pair[0].chunk_id]
                    atomic.matched_excerpt = truncate_text(max_con_pair[0].text, 220)
                    atomic.verifier_backend = backend_used
                    contradicted_cnt += 1
                    metrics_registry.increment("nli_contradictions_total")
                elif max_ent_pair is not None:
                    atomic.nli_label = "ENTAILED"
                    atomic.raw_probs = max_ent_pair[1]
                    atomic.calibrated_confidence = max_ent
                    atomic.best_source_id = max_ent_pair[0].source_id
                    atomic.matched_chunk_ids = [max_ent_pair[0].chunk_id]
                    atomic.matched_excerpt = truncate_text(max_ent_pair[0].text, 220)
                    atomic.verifier_backend = backend_used
                    entailed_cnt += 1
                else:
                    atomic.nli_label = "NEUTRAL"
                    best_pair = max(hit_probs, key=lambda hp: hp[1]["neutral"])
                    atomic.raw_probs = best_pair[1]
                    atomic.calibrated_confidence = best_pair[1]["neutral"]
                    atomic.best_source_id = best_pair[0].source_id
                    atomic.matched_chunk_ids = [best_pair[0].chunk_id]
                    atomic.matched_excerpt = truncate_text(best_pair[0].text, 220)
                    atomic.verifier_backend = backend_used
                    neutral_cnt += 1
                metrics_registry.increment("nli_claims_verified_total")
            else:
                atomic.nli_label = "SKIPPED"
                atomic.verifier_backend = "numeric_fallback" if fallback_claim_verifier else "none"
                atomic.degraded_reason = atomic.degraded_reason or (
                    scoring_reason if hits_by_atomic.get(atomic.atomic_id) or time.monotonic() >= deadline
                    else "no_scoped_chunks"
                )
                skipped_cnt += 1
                reason = None
                if not atomic.source_ids:
                    reason = "未找到有效引用"
                elif fallback_claim_verifier is not None:
                    missing = [sid for sid in atomic.source_ids if sid not in fallback_records]
                    for sid in missing:
                        fallback_records[sid] = evidence_by_id.get(sid, {})
                    if missing and self._evidence_store is not None:
                        for record in self._evidence_store.lookup(source_ids=missing, include_full_content=True):
                            sid = record.get("source_id")
                            if sid in missing:
                                fallback_records[sid] = {**fallback_records[sid], **record}
                    reason = fallback_claim_verifier(
                        atomic.text, [fallback_records[sid] for sid in atomic.source_ids],
                    )
                if reason:
                    fallback_failures.add(atomic.atomic_id)
                    issues.append(
                        {
                            "task_id": atomic.task_id,
                            "severity": "medium",
                            "check": "claim_evidence_mismatch",
                            "message": f"任务 {atomic.task_id} 原子结论与引用证据不一致：{reason}",
                            "source_ids": list(atomic.source_ids),
                            "origin": "claim_verifier:numeric_fallback",
                        }
                    )

        # Roll up atomic judgments onto parent claims in-place + attach task.atomic_claims
        for task in getattr(state, "todo_items", []) or []:
            task_atomics_list = [a for a in all_atomics if a.task_id == task.id]
            task.atomic_claims = [a.to_dict() for a in task_atomics_list]
            parent_list = claims_by_task.get(task.id) or []
            for p_idx, parent_claim in enumerate(parent_list):
                children = atomics_by_task_parent.get((task.id, p_idx))
                if not children:
                    parent_claim.pop("atomic_claim_ids", None)
                    parent_claim.pop("nli_verdict", None)
                    continue
                parent_claim["atomic_claim_ids"] = [c.atomic_id for c in children]
                labels = {c.nli_label for c in children}
                if "CONTRADICTED" in labels:
                    parent_claim["support_status"] = "contradicted"
                    parent_claim["nli_verdict"] = "CONTRADICTED"
                    bad_child = next(c for c in children if c.nli_label == "CONTRADICTED")
                    issues.append(
                        {
                            "task_id": task.id,
                            "severity": "high",
                            "check": "claim_contradicted",
                            "message": (
                                f"任务 {task.id} 原子结论「{truncate_text(bad_child.text, 80)}」"
                                f"与所引来源 [{bad_child.best_source_id or ','.join(bad_child.source_ids)}] 原文存在直接语义矛盾。"
                            ),
                            "source_ids": list(bad_child.source_ids),
                            "origin": "claim_verifier:nli",
                        }
                    )
                elif "NEUTRAL" in labels:
                    parent_claim["support_status"] = "unverified_claim"
                    parent_claim["nli_verdict"] = "NEUTRAL"
                    neu_child = next(c for c in children if c.nli_label == "NEUTRAL")
                    issues.append(
                        {
                            "task_id": task.id,
                            "severity": "medium",
                            "check": "claim_evidence_mismatch",
                            "message": (
                                f"任务 {task.id} 原子结论「{truncate_text(neu_child.text, 80)}」"
                                f"在所引来源 [{','.join(neu_child.source_ids)}] 中缺乏充分直接支撑。"
                            ),
                            "source_ids": list(neu_child.source_ids),
                            "origin": "claim_verifier:nli",
                        }
                    )
                elif labels == {"ENTAILED"}:
                    parent_claim["support_status"] = "supported"
                    parent_claim["nli_verdict"] = "ENTAILED"
                elif "SKIPPED" in labels:
                    parent_claim["nli_verdict"] = "SKIPPED"
                    if any(
                        c.atomic_id in fallback_failures
                        or c.degraded_reason in {"no_scoped_chunks", "nli_atomic_limit"}
                        for c in children
                    ):
                        parent_claim["support_status"] = "unverified_claim"

        degraded_reasons = {atomic.degraded_reason for atomic in all_atomics if atomic.degraded_reason}
        if degraded_reasons:
            metrics_registry.increment("nli_degraded_total")
            if observer is not None and hasattr(observer, "record_degraded"):
                for reason in sorted(degraded_reasons):
                    observer.record_degraded(reason)

        # Step 4: Cross-source conflict graph
        clusters: list[EvidenceConflictCluster] = []
        if bool(getattr(self._config, "nli_conflict_graph_enabled", True)):
            conflict_scorer = self._scorer if scored_probs and time.monotonic() < deadline else None
            clusters = self._conflict_builder.build(
                all_atomics,
                evidence_by_id=evidence_by_id,
                scorer=conflict_scorer,
                deadline_monotonic=deadline,
            )
        state.evidence_conflicts = [c.to_dict() for c in clusters]
        if clusters:
            metrics_registry.increment("evidence_conflicts_total", len(clusters))

        unresolved_cnt = 0
        for cluster in clusters:
            if cluster.adjudication_status == "unresolved_divergence":
                unresolved_cnt += 1
            side_sids = [s.source_id for s in cluster.sides]
            target_task = cluster.affected_task_ids[0] if cluster.affected_task_ids else None
            issues.append(
                {
                    "task_id": target_task,
                    "severity": cluster.severity,
                    "check": "cross_source_conflict",
                    "message": (
                        f"跨源冲突 [{cluster.conflict_id}]（{cluster.entity} / {cluster.metric}）："
                        f"{cluster.adjudication_rationale}"
                    ),
                    "source_ids": side_sids,
                    "origin": "claim_verifier:conflict_graph",
                }
            )

        denom = entailed_cnt + contradicted_cnt + neutral_cnt
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        summary = {
            "total_parent_claims": sum(len(v) for v in claims_by_task.values()),
            "total_atomic_claims": len(all_atomics),
            "entailed_count": entailed_cnt,
            "contradicted_count": contradicted_cnt,
            "neutral_count": neutral_cnt,
            "skipped_count": skipped_cnt,
            "atomic_faithfulness_rate": entailed_cnt / denom if denom else 0.0,
            "conflict_count": len(clusters),
            "unresolved_conflict_count": unresolved_cnt,
            "backend_used": backend_used if denom else ("numeric_fallback" if skipped_cnt and fallback_claim_verifier else "none"),
            "elapsed_ms": elapsed_ms,
            "degraded": bool(degraded_reasons),
        }
        state.claim_verification_summary = summary
        if observer is not None and hasattr(observer, "record_nli_verification"):
            observer.record_nli_verification(summary, elapsed_ms=elapsed_ms)
        return {
            "issues": issues,
            "summary": summary,
            "conflicts": state.evidence_conflicts,
        }

    def as_review_callback(self) -> Callable[[str, list[dict[str, Any]]], tuple[bool, str] | None]:
        """Compatibility adapter matching legacy ReviewService(claim_verifier=...) signature."""

        def _callback(claim_text: str, cited_records: list[dict[str, Any]]) -> tuple[bool, str] | None:
            if not bool(getattr(self._config, "nli_verifier_enabled", False)):
                return None
            valid_ids = {str(r.get("source_id") or "").strip() for r in cited_records if r.get("source_id")}
            web_enabled = getattr(self._evidence_store, "web_extraction_enabled", False)
            if web_enabled:
                valid_ids = {sid for sid in valid_ids if self._evidence_store.is_eligible_for_facts(sid)}
            if not valid_ids:
                return None
            atomics = self._decomposer.decompose(
                task_id=0,
                parent_claims=[{"text": claim_text, "source_ids": sorted(valid_ids)}],
                valid_source_ids=valid_ids,
                fast_llm=None,
                max_atomic_per_task=None,
            )
            if not atomics or len(atomics) > int(getattr(self._config, "nli_max_atomic_claims_per_task", 12)):
                return None
            deadline = time.monotonic() + float(
                getattr(self._config, "nli_stage_budget_seconds", 8.0) or 8.0
            )
            pairs: list[tuple[AtomicClaimRecord, EvidenceRetrievalHit]] = []
            for atomic in atomics:
                hits = self._retriever.fetch_for_claim(
                    atomic,
                    retrieval_service=None,
                    evidence_store=self._evidence_store if web_enabled else None,
                    cited_records_fallback=cited_records,
                    top_k_per_source=int(getattr(self._config, "nli_top_k_chunks_per_source", 3) or 3),
                )
                for h in hits:
                    if h.text:
                        pairs.append((atomic, h))
            if not pairs:
                return None
            probs = self._scorer.score_pairs_batched(
                [(h.text, a.text) for a, h in pairs],
                deadline_monotonic=deadline,
            )
            if not probs or len(probs) != len(pairs):
                return None
            entail_thresh = float(getattr(self._config, "nli_entailment_threshold", 0.60))
            contra_thresh = float(getattr(self._config, "nli_contradiction_threshold", 0.65))
            by_aid: dict[str, list[tuple[EvidenceRetrievalHit, dict[str, float]]]] = defaultdict(list)
            for (atomic, hit), p in zip(pairs, probs):
                by_aid[atomic.atomic_id].append((hit, p))
            for atomic in atomics:
                plist = by_aid.get(atomic.atomic_id) or []
                if not plist:
                    return None
                grounded_scores = [
                    p["entailment"] for hit, p in plist
                    if numeric_grounding_mismatch(atomic.text, [hit.text]) is None
                ]
                max_ent = max(grounded_scores, default=0.0)
                max_con = max(p["contradiction"] for _, p in plist)
                if max_con >= contra_thresh and max_con > max_ent + 0.10:
                    return False, f"原子结论「{truncate_text(atomic.text, 60)}」与引用原文矛盾"
                if not grounded_scores or max_ent < entail_thresh:
                    return False, f"原子结论「{truncate_text(atomic.text, 60)}」未获引用原文充分蕴含"
            return True, ""

        return _callback
