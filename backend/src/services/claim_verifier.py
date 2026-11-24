"""Atomic claim decomposition, source-scoped NLI verification, and cross-source conflict graph."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Literal

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
from services.evidence_index import EvidenceRetrievalHit, EvidenceRetrievalService
from services.text_processing import CITATION_PATTERN, normalize_agent_markdown
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
    "gross_margin": ("毛利率", "gross margin"),
    "net_income": ("净利润", "净利", "盈利", "net income", "net profit"),
    "growth_rate": ("同比增速", "同比增长", "年复合增长率", "增速", "增长率", "cagr", "yoy"),
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
        elif "contra" in name or name in {"refute", "refuted", "unsupport", "unsupported", "no"}:
            mapping["contradiction"] = idx
        elif "neutral" in name or "unknown" in name:
            mapping["neutral"] = idx
    if len(mapping) >= 2:
        return mapping
    return {"entailment": 1, "contradiction": 0}


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
    if resolved_backend in {"auto", "cross_encoder", "minicheck"} and CrossEncoder is not None:
        try:
            kwargs: dict[str, Any] = {
                "device": device,
                "max_length": max_length,
                "local_files_only": local_files_only,
            }
            try:
                ce_model = CrossEncoder(model_name, **kwargs)
            except TypeError:
                kwargs.pop("local_files_only", None)
                try:
                    ce_model = CrossEncoder(model_name, **kwargs)
                except TypeError:
                    ce_model = CrossEncoder(model_name)

            config_obj = getattr(getattr(ce_model, "model", None), "config", None) or getattr(
                ce_model, "config", None
            )
            label_mapping = _resolve_label_mapping(getattr(config_obj, "id2label", None))

            def _predict_ce(batch: list[tuple[str, str]]) -> list[list[float]]:
                raw = ce_model.predict(batch, apply_softmax=False)
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                if raw and not isinstance(raw[0], (list, tuple)):
                    # Binary single logit fallback -> [1 - p, p] style logits
                    return [[-float(val), float(val)] for val in raw]
                return [[float(x) for x in row] for row in raw]

            return LoadedNLIModel(
                backend="cross_encoder" if resolved_backend == "auto" else resolved_backend,
                model_name=model_name,
                label_mapping=label_mapping,
                predict_logits_fn=_predict_ce,
            )
        except Exception as exc:
            logger.debug("CrossEncoder load failed (%s): %s", model_name, exc)

    if (
        resolved_backend in {"auto", "transformers_nli", "minicheck"}
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
            self.last_backend_used = self._loaded_model.backend
            return self._loaded_model
        model = load_nli_model(
            str(getattr(self._config, "nli_model_name", "") or ""),
            backend=str(getattr(self._config, "nli_model_backend", "auto") or "auto"),
            device=str(getattr(self._config, "nli_device", "auto") or "auto"),
            max_length=int(getattr(self._config, "nli_max_length", 512) or 512),
        )
        if model is not None:
            self.last_backend_used = model.backend
        return model

    def _logits_to_prob_dict(
        self,
        logits: list[float],
        label_mapping: dict[str, int],
    ) -> dict[str, float]:
        probs = self._softmax_calibrated(logits, temperature=self._temperature)
        n = len(probs)
        ent_idx = label_mapping.get("entailment", 0)
        con_idx = label_mapping.get("contradiction", 1 if n > 1 else 0)
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

    def _predict_batch_with_timeout(
        self,
        model: LoadedNLIModel,
        batch: list[tuple[str, str]],
        *,
        timeout: float,
    ) -> list[dict[str, float]] | None:
        if timeout <= 0.01:
            return None
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(model.predict_logits_fn, batch)
        try:
            logits_rows = future.result(timeout=timeout)
        except (FuturesTimeoutError, Exception) as exc:
            logger.debug("NLI batch inference interrupted: %s", exc)
            return None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if not isinstance(logits_rows, list) or len(logits_rows) != len(batch):
            return None
        return [self._logits_to_prob_dict(list(row), model.label_mapping) for row in logits_rows]

    def score_pairs_batched(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        deadline_monotonic: float,
    ) -> list[dict[str, float]] | None:
        if not pairs:
            return []
        model = self._resolve_model()
        if model is None:
            return None

        batch_size = max(1, int(getattr(self._config, "nli_batch_size", 16) or 16))
        if model and str(getattr(self._config, "nli_device", "auto")).lower() == "cpu":
            batch_size = min(batch_size, 8)
        results: list[dict[str, float]] = []
        for offset in range(0, len(pairs), batch_size):
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0.02:
                return None
            batch = list(pairs[offset : offset + batch_size])
            probs_batch = self._predict_batch_with_timeout(model, batch, timeout=remaining)
            if probs_batch is None:
                return None
            results.extend(probs_batch)
        return results


def _normalize_entity(raw: str) -> str:
    cleaned = re.sub(r"[\s_\-]+", "", str(raw or "").strip().lower())
    return cleaned[:64]


def _normalize_period(raw_text: str) -> str:
    text = str(raw_text or "")
    q_match = re.search(
        r"(20\d{2}|19\d{2})\s*(?:年)?\s*[Qq]\s*([1-4])|(20\d{2})\s*年第([一二三四1-4])季度",
        text,
    )
    if q_match:
        if q_match.group(1):
            return f"{q_match.group(1)}Q{q_match.group(2)}"
        cn_map = {"一": "1", "二": "2", "三": "3", "四": "4"}
        q_num = cn_map.get(q_match.group(4), q_match.group(4))
        return f"{q_match.group(3)}Q{q_num}"

    fy_match = re.search(r"(?:FY)?(20\d{2}|19\d{2})\s*年?", text, flags=re.IGNORECASE)
    if fy_match:
        return fy_match.group(1)

    ver_match = re.search(r"\bv(\d+(?:\.\d+){1,2})\b", text, flags=re.IGNORECASE)
    if ver_match:
        return f"v{ver_match.group(1)}"
    return ""


def _normalize_metric(raw_text: str) -> str:
    lowered = str(raw_text or "").strip().lower()
    if not lowered:
        return ""
    for canonical, aliases in _METRIC_ALIASES.items():
        if canonical == lowered or any(alias in lowered for alias in aliases):
            return canonical
    return re.sub(r"[\s_\-]+", "_", lowered)[:40] if len(lowered) <= 30 else ""


def _extract_numeric_and_unit(raw_text: str) -> tuple[float | None, str]:
    match = re.search(
        r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|％|个百分点|亿元|亿美元|万元|万|亿|美元|USD|RMB|元|ms|s|x|倍)?",
        str(raw_text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None, ""
    # Skip standalone 4-digit year values (e.g. 2024, 2025) when unit is empty
    raw_num = match.group(1).replace(",", "")
    unit = (match.group(2) or "").strip()
    try:
        val = float(raw_num)
    except ValueError:
        return None, ""
    if not unit and val.is_integer() and 1900 <= val <= 2100:
        remainder = str(raw_text or "")[match.end() :]
        second = _extract_numeric_and_unit(remainder)
        if second[0] is not None:
            return second
        return None, ""

    unit_lower = unit.lower()
    if unit_lower in {"%", "％", "个百分点"}:
        return val, "%"
    if unit_lower in {"万"}:
        return val * 1e4, "scalar"
    if unit_lower in {"亿", "亿元"}:
        return val * 1e8, "currency"
    if unit_lower in {"亿美元"}:
        return val * 1e8, "usd"
    return val, unit_lower or "scalar"


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
    zh_match = re.match(r"^([\u4e00-\u9fffA-Za-z0-9·]{2,12}?)(?:在|于|的|为|是|实现|营收|毛利率|净利|延迟|已|未)", cleaned)
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
    if num_val is None:
        num_val, unit = _extract_numeric_and_unit(text)
    else:
        try:
            num_val = float(num_val)
        except (TypeError, ValueError):
            num_val, unit = _extract_numeric_and_unit(text)
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
        normalized = normalize_agent_markdown(str(text or "").strip())
        if not normalized:
            return []
        raw_parts = re.split(
            r"(?:[；;。!?！？]+|\s*(?:，|,)\s*(?:且|同时|并且|并|而|但|however|while|whereas)\b|\b(?:however|while|whereas)\b)",
            normalized,
            flags=re.IGNORECASE,
        )
        clauses = [part.strip(" ，,;；。") for part in raw_parts if part and part.strip(" ，,;；。")]
        return clauses or [normalized]

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
        max_atomic_per_task: int = 12,
    ) -> list[AtomicClaimRecord]:
        if not parent_claims or max_atomic_per_task <= 0:
            return []

        has_compound = any(
            len(str(c.get("text") or "")) >= 35
            or len(c.get("source_ids") or []) > 1
            or bool(re.search(r"[；;且]|however|while", str(c.get("text") or ""), flags=re.IGNORECASE))
            for c in parent_claims
        )
        llm_results: dict[int, list[dict[str, Any]]] | None = None
        if self._decompose_with_fast_llm and fast_llm is not None and has_compound:
            llm_results = self._decompose_via_fast_llm(
                task_id=task_id,
                parent_claims=parent_claims,
                fast_llm=fast_llm,
            )

        records: list[AtomicClaimRecord] = []
        for p_idx, claim in enumerate(parent_claims):
            if len(records) >= max_atomic_per_task:
                break
            parent_text = str(claim.get("text") or "").strip()
            parent_sids = [str(s).strip() for s in (claim.get("source_ids") or []) if str(s).strip()]
            if not parent_text:
                continue

            raw_atomics = (llm_results or {}).get(p_idx)
            if raw_atomics:
                for a_idx, atomic_item in enumerate(raw_atomics):
                    if len(records) >= max_atomic_per_task:
                        break
                    sub_text = normalize_agent_markdown(str(atomic_item.get("text") or "").strip())
                    if not sub_text:
                        continue
                    effective_sids = self._conserve_source_ids(
                        atomic_item.get("source_ids"),
                        parent_sids=parent_sids,
                        valid_source_ids=valid_source_ids,
                    )
                    prop = _extract_proposition_from_text(
                        sub_text,
                        atomic_item.get("proposition"),
                    )
                    records.append(
                        AtomicClaimRecord(
                            atomic_id=f"{task_id}-C{p_idx}-A{a_idx}",
                            task_id=task_id,
                            parent_claim_index=p_idx,
                            text=sub_text,
                            source_ids=effective_sids,
                            proposition=prop,
                        )
                    )
                continue

            clauses = self._syntactic_split(parent_text)
            for a_idx, clause in enumerate(clauses):
                if len(records) >= max_atomic_per_task:
                    break
                inline_sids = CITATION_PATTERN.findall(clause)
                effective_sids = self._conserve_source_ids(
                    inline_sids or parent_sids,
                    parent_sids=parent_sids,
                    valid_source_ids=valid_source_ids,
                )
                clean_clause = normalize_agent_markdown(CITATION_PATTERN.sub("", clause).strip()) or clause
                prop = _extract_proposition_from_text(clean_clause)
                records.append(
                    AtomicClaimRecord(
                        atomic_id=f"{task_id}-C{p_idx}-A{a_idx}",
                        task_id=task_id,
                        parent_claim_index=p_idx,
                        text=clean_clause,
                        source_ids=effective_sids,
                        proposition=prop,
                    )
                )
        return records


def _inline_chunk_and_bm25_top_k(
    *,
    query: str,
    records: list[dict[str, Any]],
    top_k_per_source: int,
) -> list[EvidenceRetrievalHit]:
    query_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?", query.lower()))
    hits: list[EvidenceRetrievalHit] = []
    for rec in records:
        sid = str(rec.get("source_id") or "").strip()
        if not sid:
            continue
        texts: list[tuple[str, str]] = []
        full = str(rec.get("full_content") or rec.get("raw_content") or rec.get("snippet") or "").strip()
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
        allowed_list = [str(s).strip() for s in (atomic_claim.source_ids or []) if str(s).strip()]
        if not allowed_list:
            return []
        allowed_set = set(allowed_list)

        hits: list[EvidenceRetrievalHit] = []
        if retrieval_service is not None:
            result = retrieval_service.query(
                atomic_claim.text,
                task_id=atomic_claim.task_id,
                scope="current",
                mode="grounding",
                top_k=max(1, len(allowed_list) * max(1, int(top_k_per_source or 3))),
                allowed_source_ids=allowed_list,
            )
            hits = [h for h in result.hits if h.source_id in allowed_set and h.origin == "current"]

        if not hits:
            records: list[dict[str, Any]] = []
            if evidence_store is not None and hasattr(evidence_store, "lookup"):
                records = [
                    rec
                    for rec in (evidence_store.lookup(include_full_content=True) or [])
                    if isinstance(rec, dict) and rec.get("source_id") in allowed_set
                ]
            if not records:
                records = [
                    rec
                    for rec in (cited_records_fallback or [])
                    if isinstance(rec, dict) and rec.get("source_id") in allowed_set
                ]
            hits = _inline_chunk_and_bm25_top_k(
                query=atomic_claim.text,
                records=records,
                top_k_per_source=max(1, int(top_k_per_source or 3)),
            )
        return hits


def _compute_authority_rank(record: dict[str, Any], nli_support: float) -> float:
    source_type = str(record.get("source_type") or "web_general").strip()
    tier = SOURCE_TIER_WEIGHT.get(source_type, 0.40)
    reliability = float(record.get("source_reliability_score") or 5.0)
    reliability_norm = max(0.0, min(1.0, reliability / 10.0))
    return (0.65 * tier) + (0.25 * reliability_norm) + (0.10 * max(0.0, min(1.0, nli_support)))


def _adjudicate_cluster(sides: list[ConflictSide]) -> tuple[AdjudicationStatus, str | None, str]:
    ranked = sorted(
        sides,
        key=lambda s: (s.authority_rank, s.published_at or ""),
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
    if top.published_at and runner_up.published_at and top.published_at[:7] != runner_up.published_at[:7]:
        newer = top if top.published_at > runner_up.published_at else runner_up
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
            if claim.nli_label != "CONTRADICTED"
            and (claim.best_source_id or (claim.source_ids[0] if claim.source_ids else None))
            and claim.proposition.entity
            and (claim.proposition.metric or claim.proposition.numeric_value is not None)
        ]
        if len(eligible) < 2:
            return []

        buckets: dict[tuple[str, str, str], list[AtomicClaimRecord]] = defaultdict(list)
        for claim in eligible:
            key = (
                claim.proposition.entity,
                claim.proposition.metric or "statement",
                claim.proposition.period or "general",
            )
            buckets[key].append(claim)

        rel_tol = float(getattr(self._config, "nli_conflict_numeric_rel_tol", 0.05) or 0.05)
        min_nli_score = float(getattr(self._config, "nli_conflict_min_score", 0.70) or 0.70)

        clusters: list[EvidenceConflictCluster] = []
        for (entity, metric, period), group_claims in sorted(buckets.items()):
            # Pick representative atomic claim per distinct source_id
            by_source: dict[str, AtomicClaimRecord] = {}
            for c in group_claims:
                sid = c.best_source_id or (c.source_ids[0] if c.source_ids else "")
                if not sid:
                    continue
                prev = by_source.get(sid)
                if prev is None or c.calibrated_confidence > prev.calibrated_confidence:
                    by_source[sid] = c
            if len(by_source) < 2:
                continue

            sides_list: list[ConflictSide] = []
            for sid, c in by_source.items():
                rec = evidence_by_id.get(sid) or {}
                nli_supp = float((c.raw_probs or {}).get("entailment") or c.calibrated_confidence or 0.5)
                sides_list.append(
                    ConflictSide(
                        source_id=sid,
                        task_id=c.task_id,
                        title=str(rec.get("title") or sid),
                        url=str(rec.get("url") or ""),
                        domain=str(rec.get("domain") or ""),
                        source_type=str(rec.get("source_type") or "web_general"),
                        quality_label=str(rec.get("quality_label") or "medium"),
                        published_at=rec.get("published_at"),
                        claim_text=c.text,
                        normalized_value=c.proposition.numeric_value,
                        unit=c.proposition.unit,
                        evidence_excerpt=truncate_text(
                            c.matched_excerpt or str(rec.get("snippet") or ""),
                            220,
                        ),
                        authority_rank=_compute_authority_rank(rec, nli_supp),
                        nli_support_score=nli_supp,
                    )
                )

            edges: list[dict[str, Any]] = []
            conflict_types: set[str] = set()
            max_score = 0.0
            cross_nli_pairs: list[tuple[int, int, str, str]] = []

            for i in range(len(sides_list)):
                for j in range(i + 1, len(sides_list)):
                    sa, sb = sides_list[i], sides_list[j]
                    va, vb = sa.normalized_value, sb.normalized_value
                    if (
                        va is not None
                        and vb is not None
                        and (not sa.unit or not sb.unit or sa.unit == sb.unit)
                    ):
                        delta_ratio = abs(va - vb) / max(abs(va), abs(vb), 1e-9)
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
                        cross_nli_pairs.append(
                            (i, j, sa.evidence_excerpt or sa.claim_text, sb.claim_text)
                        )

            if cross_nli_pairs and scorer is not None and deadline_monotonic is not None:
                nli_inputs: list[tuple[str, str]] = []
                for _, _, premise_a, hyp_b in cross_nli_pairs:
                    nli_inputs.append((premise_a, hyp_b))
                    nli_inputs.append((hyp_b, premise_a))
                nli_scores = scorer.score_pairs_batched(
                    nli_inputs,
                    deadline_monotonic=deadline_monotonic,
                )
                if nli_scores and len(nli_scores) == len(nli_inputs):
                    for idx, (i, j, _, _) in enumerate(cross_nli_pairs):
                        fwd = nli_scores[2 * idx]["contradiction"]
                        rev = nli_scores[2 * idx + 1]["contradiction"]
                        pair_score = max(fwd, rev)
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

            adjudication_status, recommended_sid, rationale = _adjudicate_cluster(sides_list)
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
                    affected_task_ids=sorted({s.task_id for s in sides_list if s.task_id > 0}),
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
        by_id: dict[str, dict[str, Any]] = {}
        for task in getattr(state, "todo_items", []) or []:
            for item in getattr(task, "evidence_items", None) or []:
                if isinstance(item, dict) and item.get("source_id"):
                    by_id[str(item["source_id"]).strip()] = item
        if self._evidence_store is not None and hasattr(self._evidence_store, "lookup"):
            for item in self._evidence_store.lookup(include_full_content=True) or []:
                if isinstance(item, dict) and item.get("source_id"):
                    sid = str(item["source_id"]).strip()
                    by_id[sid] = item
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

        budget_sec = max(0.1, float(getattr(self._config, "nli_stage_budget_seconds", 8.0) or 8.0))
        deadline = time.monotonic() + budget_sec
        evidence_by_id = self._collect_evidence_map(state)
        request_valid_ids = set(evidence_by_id.keys())
        max_atomics = int(getattr(self._config, "nli_max_atomic_claims_per_task", 12) or 12)
        top_k = int(getattr(self._config, "nli_top_k_chunks_per_source", 3) or 3)
        entail_thresh = float(getattr(self._config, "nli_entailment_threshold", 0.60) or 0.60)
        contra_thresh = float(getattr(self._config, "nli_contradiction_threshold", 0.65) or 0.65)

        all_atomics: list[AtomicClaimRecord] = []
        atomics_by_task_parent: dict[tuple[int, int], list[AtomicClaimRecord]] = defaultdict(list)

        for task in getattr(state, "todo_items", []) or []:
            task_claims = list(claims_by_task.get(task.id) or [])
            verifiable_pairs: list[tuple[int, dict[str, Any]]] = [
                (idx, claim)
                for idx, claim in enumerate(task_claims)
                if str(claim.get("support_status") or "") not in {"missing_citation", "invalid_citation"}
                and bool(claim.get("source_ids"))
            ]
            if not verifiable_pairs:
                continue
            subset_claims = [c for _, c in verifiable_pairs]
            index_remap = {sub_idx: orig_idx for sub_idx, (orig_idx, _) in enumerate(verifiable_pairs)}
            task_atomics = self._decomposer.decompose(
                task_id=task.id,
                parent_claims=subset_claims,
                valid_source_ids=request_valid_ids,
                fast_llm=self._fast_llm,
                max_atomic_per_task=max_atomics,
            )
            for atomic in task_atomics:
                orig_idx = index_remap.get(atomic.parent_claim_index, atomic.parent_claim_index)
                atomic.parent_claim_index = orig_idx
                atomic.atomic_id = f"{task.id}-C{orig_idx}-A{len(atomics_by_task_parent[(task.id, orig_idx)])}"
                all_atomics.append(atomic)
                atomics_by_task_parent[(task.id, orig_idx)].append(atomic)

        # Step 2: Scoped retrieval per atomic claim -> prepare NLI pairs
        pair_meta: list[tuple[AtomicClaimRecord, EvidenceRetrievalHit]] = []
        hits_by_atomic: dict[str, list[EvidenceRetrievalHit]] = {}
        for atomic in all_atomics:
            cited_fallback = [evidence_by_id[sid] for sid in atomic.source_ids if sid in evidence_by_id]
            hits = self._retriever.fetch_for_claim(
                atomic,
                retrieval_service=self._retrieval_service,
                evidence_store=self._evidence_store,
                cited_records_fallback=cited_fallback,
                top_k_per_source=top_k,
            )
            hits_by_atomic[atomic.atomic_id] = hits
            for hit in hits:
                if hit.text:
                    pair_meta.append((atomic, hit))

        # Step 3: Batch NLI scoring
        degraded = False
        degraded_reason: str | None = None
        scored_probs: list[dict[str, float]] | None = None
        if pair_meta:
            scored_probs = self._scorer.score_pairs_batched(
                [(hit.text, atomic.text) for atomic, hit in pair_meta],
                deadline_monotonic=deadline,
            )
            if scored_probs is None:
                degraded = True
                degraded_reason = (
                    "nli_budget_exhausted"
                    if time.monotonic() >= deadline
                    else "nli_model_unavailable"
                )
                if observer is not None and hasattr(observer, "record_degraded"):
                    observer.record_degraded(degraded_reason)
                metrics_registry.increment("nli_degraded_total")

        probs_by_atomic: dict[str, list[tuple[EvidenceRetrievalHit, dict[str, float]]]] = defaultdict(list)
        if scored_probs is not None:
            for (atomic, hit), probs in zip(pair_meta, scored_probs):
                probs_by_atomic[atomic.atomic_id].append((hit, probs))

        issues: list[dict[str, Any]] = []
        entailed_cnt = contradicted_cnt = neutral_cnt = skipped_cnt = 0

        for atomic in all_atomics:
            hit_probs = probs_by_atomic.get(atomic.atomic_id)
            if hit_probs:
                max_ent_pair = max(hit_probs, key=lambda hp: hp[1]["entailment"])
                max_con_pair = max(hit_probs, key=lambda hp: hp[1]["contradiction"])
                max_ent = max_ent_pair[1]["entailment"]
                max_con = max_con_pair[1]["contradiction"]

                if max_con >= contra_thresh and max_con > max_ent + 0.10:
                    atomic.nli_label = "CONTRADICTED"
                    atomic.raw_probs = max_con_pair[1]
                    atomic.calibrated_confidence = max_con
                    atomic.best_source_id = max_con_pair[0].source_id
                    atomic.matched_chunk_ids = [max_con_pair[0].chunk_id]
                    atomic.matched_excerpt = truncate_text(max_con_pair[0].text, 220)
                    atomic.verifier_backend = self._scorer.last_backend_used
                    contradicted_cnt += 1
                    metrics_registry.increment("nli_contradictions_total")
                elif max_ent >= entail_thresh:
                    atomic.nli_label = "ENTAILED"
                    atomic.raw_probs = max_ent_pair[1]
                    atomic.calibrated_confidence = max_ent
                    atomic.best_source_id = max_ent_pair[0].source_id
                    atomic.matched_chunk_ids = [max_ent_pair[0].chunk_id]
                    atomic.matched_excerpt = truncate_text(max_ent_pair[0].text, 220)
                    atomic.verifier_backend = self._scorer.last_backend_used
                    entailed_cnt += 1
                else:
                    atomic.nli_label = "NEUTRAL"
                    best_pair = max_ent_pair
                    atomic.raw_probs = best_pair[1]
                    atomic.calibrated_confidence = max(max_ent, best_pair[1].get("neutral", 0.0))
                    atomic.best_source_id = best_pair[0].source_id
                    atomic.matched_chunk_ids = [best_pair[0].chunk_id]
                    atomic.matched_excerpt = truncate_text(best_pair[0].text, 220)
                    atomic.verifier_backend = self._scorer.last_backend_used
                    neutral_cnt += 1
                metrics_registry.increment("nli_claims_verified_total")
            else:
                atomic.nli_label = "SKIPPED"
                atomic.verifier_backend = "numeric_fallback" if fallback_claim_verifier else "none"
                atomic.degraded_reason = degraded_reason or (
                    "no_scoped_chunks" if not hits_by_atomic.get(atomic.atomic_id) else "nli_unavailable"
                )
                skipped_cnt += 1
                if fallback_claim_verifier is not None:
                    cited_recs = [evidence_by_id[s] for s in atomic.source_ids if s in evidence_by_id]
                    reason = fallback_claim_verifier(atomic.text, cited_recs)
                    if reason:
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

        # Step 4: Cross-source conflict graph
        clusters: list[EvidenceConflictCluster] = []
        if bool(getattr(self._config, "nli_conflict_graph_enabled", True)):
            conflict_scorer = None if degraded else self._scorer
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
            "atomic_faithfulness_rate": round(entailed_cnt / denom, 4) if denom > 0 else 0.0,
            "conflict_count": len(clusters),
            "unresolved_conflict_count": unresolved_cnt,
            "backend_used": self._scorer.last_backend_used,
            "elapsed_ms": elapsed_ms,
            "degraded": degraded,
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
            if not valid_ids:
                return None
            atomics = self._decomposer.decompose(
                task_id=0,
                parent_claims=[{"text": claim_text, "source_ids": sorted(valid_ids)}],
                valid_source_ids=valid_ids,
                fast_llm=None,
                max_atomic_per_task=int(getattr(self._config, "nli_max_atomic_claims_per_task", 12) or 12),
            )
            if not atomics:
                return None
            deadline = time.monotonic() + float(
                getattr(self._config, "nli_stage_budget_seconds", 8.0) or 8.0
            )
            pairs: list[tuple[AtomicClaimRecord, EvidenceRetrievalHit]] = []
            for atomic in atomics:
                hits = self._retriever.fetch_for_claim(
                    atomic,
                    retrieval_service=self._retrieval_service,
                    evidence_store=self._evidence_store,
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
            if probs is None:
                return None
            entail_thresh = float(getattr(self._config, "nli_entailment_threshold", 0.60) or 0.60)
            contra_thresh = float(getattr(self._config, "nli_contradiction_threshold", 0.65) or 0.65)
            by_aid: dict[str, list[dict[str, float]]] = defaultdict(list)
            for (atomic, _), p in zip(pairs, probs):
                by_aid[atomic.atomic_id].append(p)
            for atomic in atomics:
                plist = by_aid.get(atomic.atomic_id) or []
                if not plist:
                    continue
                max_ent = max(x["entailment"] for x in plist)
                max_con = max(x["contradiction"] for x in plist)
                if max_con >= contra_thresh and max_con > max_ent + 0.10:
                    return False, f"原子结论「{truncate_text(atomic.text, 60)}」与引用原文矛盾"
                if max_ent < entail_thresh:
                    return False, f"原子结论「{truncate_text(atomic.text, 60)}」未获引用原文充分蕴含"
            return True, ""

        return _callback
