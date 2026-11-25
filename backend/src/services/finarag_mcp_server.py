"""FinaRAG stdio / Streamable-HTTP MCP Server with Fail-Closed tenant ACL."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import math
import os
import re
import stat
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Literal, Sequence, TextIO, Union

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+|[\u4e00-\u9fff]+")
_REPORT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_PROTOCOL_VERSION = "2025-03-26"
_MAX_REQUEST_BYTES = 2 * 1024 * 1024


class _ReportUnavailable(ValueError):
    pass


class FinaragInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True, str_max_length=4000)


CompanyName = Annotated[str, Field(min_length=1, max_length=300)]
PageNumber = Annotated[int, Field(ge=1)]


class FinaragListCompaniesInput(FinaragInput):
    industry: str | None = Field(default=None, description="Filter by subset.csv major_industry")
    currency: str | None = Field(default=None, description="Filter by currency such as USD/EUR/CHF")
    keyword: str | None = Field(default=None, description="Substring match against company_name")
    limit: int = Field(default=50, ge=1, le=200)


class FinaragCompanyMeta(BaseModel):
    company_name: str
    sha1_name: str
    currency: str | None = None
    major_industry: str | None = None
    pages_amount: int = 0
    tables_amount: int = 0


class FinaragListCompaniesOutput(BaseModel):
    companies: list[FinaragCompanyMeta] = Field(default_factory=list)
    total: int = 0


class FinaragFilingsInput(FinaragInput):
    company_name: CompanyName = Field(description="Exact company_name in FinaRAG repository")
    query: str = Field(..., min_length=1, description="Filing semantic or lexical retrieval query")
    retrieval_mode: Literal["vector", "hybrid", "bm25"] = Field(default="vector")
    top_n: int = Field(default=6, ge=1, le=20)
    return_parent_pages: bool = Field(default=True)
    llm_reranking_sample_size: int = Field(default=28, ge=5, le=60)
    llm_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    use_serialized_tables: bool = Field(default=False)


class FinaragPassageItem(BaseModel):
    page_index_1based: int
    page_index_0based: int
    distance: float
    relevance_score: float | None = None
    combined_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    text: str
    chunk_types: list[str] = Field(default_factory=list)
    citation_uri: str


class FinaragFilingsOutput(BaseModel):
    company_name: str
    pdf_sha1: str
    currency: str | None = None
    retrieval_mode: str
    passages: list[FinaragPassageItem] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)


class FinaragStructuredQueryInput(FinaragInput):
    company_name: CompanyName
    question: str = Field(..., min_length=1)
    answer_schema: Literal["number", "name", "names", "boolean"] = Field(default="number")
    top_n_retrieval: int = Field(default=6, ge=1, le=15)
    return_parent_pages: bool = Field(default=True)
    llm_reranking: bool = Field(default=False)
    use_serialized_tables: bool = Field(default=False)


class FinaragPageRef(BaseModel):
    pdf_sha1: str
    page_index: int
    citation_uri: str


class FinaragPageExcerpt(BaseModel):
    page_index: int
    text: str


class FinaragStructuredQueryOutput(BaseModel):
    company_name: str
    pdf_sha1: str
    answer_schema: str
    final_answer: Union[float, int, bool, str, list[str]]
    reasoning_summary: str
    step_by_step_analysis: str
    relevant_pages: list[int] = Field(default_factory=list)
    references: list[FinaragPageRef] = Field(default_factory=list)
    excerpts: list[FinaragPageExcerpt] = Field(default_factory=list)
    answer_reason: str


class FinaragGetPagesInput(FinaragInput):
    company_name: CompanyName
    page_numbers: list[PageNumber] = Field(..., min_length=1, max_length=10)
    include_serialized_tables: bool = Field(default=True)


class FinaragGetPagesOutput(BaseModel):
    company_name: str
    pdf_sha1: str
    pages: list[FinaragPageExcerpt] = Field(default_factory=list)


def _verify_hmac_token(
    *,
    token: str,
    secret: str,
    expected_server_id: str = "finarag",
    expected_tenant_id: str | None = None,
) -> dict[str, Any]:
    raw = str(token or "").strip()
    parts = raw.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        raise PermissionError("invalid scoped tenant token format")
    payload_b64, sig_b64 = parts[1], parts[2]
    expected_sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    if not sig_b64.isascii() or not hmac.compare_digest(sig_b64, expected_sig):
        raise PermissionError("invalid scoped tenant token signature")
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception as exc:
        raise PermissionError("malformed scoped tenant token payload") from exc
    if not isinstance(claims, dict):
        raise PermissionError("invalid scoped tenant token claims")
    exp = claims.get("exp")
    if type(exp) is not int or exp <= time.time():
        raise PermissionError("scoped tenant token expired")
    if expected_server_id and str(claims.get("server_id") or "") != expected_server_id:
        raise PermissionError("scoped tenant token server mismatch")
    if expected_tenant_id and str(claims.get("tenant_id") or "") != expected_tenant_id:
        raise PermissionError("scoped tenant token tenant mismatch")
    return claims


class TenantACL:
    """Fail-Closed per-tenant company whitelist."""

    def __init__(
        self,
        rules: dict[str, list[str]] | None = None,
        *,
        enforce: bool = True,
    ) -> None:
        self._enforce = bool(enforce)
        self._rules: dict[str, set[str]] = {}
        for tenant, companies in (rules or {}).items():
            norm_tenant = str(tenant or "").strip()
            if not norm_tenant:
                continue
            self._rules[norm_tenant] = {
                str(comp).strip().casefold() for comp in (companies or []) if str(comp).strip()
            }

    @classmethod
    def from_path(cls, path: str | Path | None, *, enforce: bool = True) -> TenantACL:
        if not path:
            return cls(rules=None, enforce=enforce)
        resolved = Path(path)
        if not resolved.exists():
            return cls(rules=None, enforce=enforce)
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("tenants"), dict):
            normalized: dict[str, list[str]] = {}
            for tid, entry in raw["tenants"].items():
                if isinstance(entry, dict):
                    companies = entry.get("companies", entry.get("allowed_companies"))
                    if isinstance(companies, list):
                        normalized[str(tid)] = companies
                elif isinstance(entry, list):
                    normalized[str(tid)] = list(entry)
            return cls(normalized, enforce=enforce)
        if isinstance(raw, dict):
            return cls(
                {str(k): list(v) for k, v in raw.items() if isinstance(v, list)},
                enforce=enforce,
            )
        return cls(rules=None, enforce=enforce)

    def allowed_companies(self, tenant_id: str) -> set[str] | None:
        """Return allowed lowercase company names, None for unrestricted, or empty set on deny."""
        if not self._enforce and not self._rules:
            return None
        tid = str(tenant_id or "").strip()
        if not tid:
            return set()
        if tid in self._rules:
            allowed = set(self._rules[tid])
            if "*" in allowed:
                return None
            return allowed
        if "*" in self._rules:
            wildcard_allowed = set(self._rules["*"])
            return None if "*" in wildcard_allowed else wildcard_allowed
        # Fail-Closed: unknown tenant receives empty allowed set when ACL is enforced
        return set() if self._enforce else None

    def is_company_allowed(self, tenant_id: str, company_name: str) -> bool:
        allowed = self.allowed_companies(tenant_id)
        if allowed is None:
            return True
        return company_name.strip().casefold() in allowed


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(str(text or "").lower()):
        if re.match(r"[\u4e00-\u9fff]+", tok):
            tokens.extend(list(tok))
            if len(tok) >= 2:
                tokens.extend(tok[i : i + 2] for i in range(len(tok) - 1))
        else:
            tokens.append(tok)
    return tokens


def _normalize_vector(raw: Sequence[float]) -> list[float]:
    cleaned: list[float] = []
    for value in raw:
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = 0.0
        cleaned.append(num if math.isfinite(num) else 0.0)
    norm = math.sqrt(sum(v * v for v in cleaned))
    if norm <= 0.0:
        return [0.0 for _ in cleaned]
    return [v / norm for v in cleaned]


def _hashed_dense_vector(text: str, *, dim: int = 128) -> list[float]:
    """Deterministic zero-dependency L2-normalized embedding over tokens & char 3-grams."""
    vec = [0.0] * dim
    tokens = _tokenize(text)
    cleaned = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    features: list[tuple[str, float]] = [(tok, 1.0) for tok in tokens]
    if len(cleaned) >= 3:
        features.extend((f"3g:{cleaned[i : i + 3]}", 0.5) for i in range(len(cleaned) - 2))
    elif cleaned:
        features.append((f"raw:{cleaned}", 0.5))
    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[bucket] += sign * weight
    return _normalize_vector(vec)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or len(vec_a) != len(vec_b):
        return 0.0
    dot = math.fsum(a * b for a, b in zip(vec_a, vec_b))
    return max(0.0, min(1.0, dot)) if math.isfinite(dot) else 0.0


class FinaRAGAdapter:
    """In-process / CLI-compatible retrieval adapter over FinaRAG chunked reports."""

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        acl: TenantACL | None = None,
        token_secret: str | None = None,
        answer_fn: Callable[[str, list[dict[str, Any]], str], tuple[Any, str]] | None = None,
        hybrid_rrf_enabled: bool | None = None,
        rrf_k: int | None = None,
        embedder: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        self._root = Path(
            data_root
            or os.getenv("FINARAG_DATA_ROOT")
            or "/media/main/hjz/llm/FinaRAG"
        ).expanduser().resolve()
        self._acl = acl or TenantACL.from_path(
            os.getenv("FINARAG_TENANT_ACL_PATH"),
            enforce=True,
        )
        self._token_secret = token_secret if token_secret is not None else os.getenv("MCP_TENANT_TOKEN_SECRET")
        self._answer_fn = answer_fn
        if hybrid_rrf_enabled is None:
            env_rrf = str(os.getenv("V4_FINARAG_HYBRID_RRF_ENABLED") or "").strip().lower()
            self._hybrid_rrf_enabled = env_rrf in {"1", "true", "yes", "on"}
        else:
            self._hybrid_rrf_enabled = bool(hybrid_rrf_enabled)
        raw_k = rrf_k if rrf_k is not None else (os.getenv("V4_FINARAG_RRF_K") or 60)
        try:
            parsed_k = int(raw_k)
        except (TypeError, ValueError):
            parsed_k = 60
        self._rrf_k = max(1, min(200, parsed_k))
        self._embedder = embedder
        self._subset_meta: dict[str, dict[str, Any]] | None = None
        self._reports_cache: dict[bool, dict[str, dict[str, Any]]] = {}

    def authorize_request(
        self,
        *,
        tenant_id: str,
        scoped_token: str | None,
        request_id: str | None = None,
        require_token: bool = False,
    ) -> str:
        resolved_tenant = str(tenant_id or "").strip() or "default"
        if self._token_secret or require_token:
            if not scoped_token or not self._token_secret:
                raise PermissionError("missing scoped tenant token")
            claims = _verify_hmac_token(
                token=scoped_token,
                secret=self._token_secret,
                expected_server_id="finarag",
                expected_tenant_id=resolved_tenant,
            )
            if claims.get("scope") != "mcp:read" or claims.get("request_id") != request_id or not request_id:
                raise PermissionError("invalid FinaRAG token scope or request")
            resolved_tenant = str(claims.get("tenant_id") or resolved_tenant)
        elif resolved_tenant != "default":
            raise PermissionError("tenant identity requires a scoped token")
        return resolved_tenant

    @contextmanager
    def _open_data_file(self, path: Path) -> Iterator[TextIO]:
        parts = path.relative_to(self._root).parts
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError("Invalid FinaRAG data path")
        # Resolve each component through its parent descriptor and refuse symlinks.
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent = os.open(self._root, directory_flags)
        try:
            for part in parts[:-1]:
                child = os.open(part, directory_flags, dir_fd=parent)
                os.close(parent)
                parent = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, encoding="utf-8") as file:
                if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                    raise ValueError("Invalid FinaRAG data file")
                yield file
        finally:
            os.close(parent)

    def _load_subset_meta(self) -> dict[str, dict[str, Any]]:
        if self._subset_meta is not None:
            return self._subset_meta
        meta: dict[str, dict[str, Any]] = {}
        subset_csv = self._root / "subset.csv"
        try:
            with self._open_data_file(subset_csv) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    cname = str(row.get("company_name") or "").strip()
                    if cname:
                        meta[cname.casefold()] = {
                            "company_name": cname,
                            "sha1": str(row.get("sha1") or "").strip(),
                            "currency": str(row.get("cur") or row.get("currency") or "").strip() or None,
                            "major_industry": str(row.get("major_industry") or "").strip() or None,
                        }
        except FileNotFoundError:
            pass
        self._subset_meta = meta
        return meta

    def _load_reports(self, *, use_serialized_tables: bool = False) -> dict[str, dict[str, Any]]:
        if use_serialized_tables in self._reports_cache:
            return self._reports_cache[use_serialized_tables]

        candidates = [
            self._root / "databases" / "chunked_reports",
            self._root / "chunked_reports",
        ]
        if use_serialized_tables:
            candidates.insert(0, self._root / "databases_ser_tab" / "chunked_reports")

        reports: dict[str, dict[str, Any]] = {}
        subset = self._load_subset_meta()
        has_subset = (self._root / "subset.csv").exists()
        for report_dir in candidates:
            if report_dir.resolve() != report_dir:
                continue
            for json_path in sorted(report_dir.glob("*.json")):
                try:
                    with self._open_data_file(json_path) as file:
                        payload = json.load(file)
                except (OSError, ValueError, RecursionError):
                    continue
                if not isinstance(payload, dict):
                    continue
                metainfo = payload.get("metainfo")
                if not isinstance(metainfo, dict) or not isinstance(payload.get("content"), dict):
                    continue
                company_name = str(metainfo.get("company_name") or "").strip()
                if not company_name:
                    continue
                key = company_name.casefold()
                expected_sha = (subset.get(key) or {}).get("sha1")
                sha = str(metainfo.get("sha1_name") or metainfo.get("sha1") or expected_sha or "").strip()
                if has_subset and (key not in subset or (expected_sha and sha != expected_sha)):
                    continue
                if not _REPORT_ID_RE.fullmatch(sha):
                    continue
                reports.setdefault(key, {**payload, "metainfo": {**metainfo, "sha1_name": sha}})

        self._reports_cache[use_serialized_tables] = reports
        return reports

    def _resolve_company_doc(
        self,
        *,
        tenant_id: str,
        company_name: str,
        use_serialized_tables: bool = False,
    ) -> dict[str, Any]:
        if not self._acl.is_company_allowed(tenant_id, company_name):
            raise PermissionError(f"tenant '{tenant_id}' is not authorized to access '{company_name}'")
        reports = self._load_reports(use_serialized_tables=use_serialized_tables)
        key = company_name.strip().casefold()
        doc = reports.get(key)
        if doc is None:
            raise _ReportUnavailable("FinaRAG report unavailable")
        return doc

    def list_companies(
        self,
        params: FinaragListCompaniesInput,
        *,
        tenant_id: str,
    ) -> FinaragListCompaniesOutput:
        reports = self._load_reports(use_serialized_tables=False)
        subset = self._load_subset_meta()
        items: list[FinaragCompanyMeta] = []

        keys = sorted(set(reports.keys()) | set(subset.keys()))
        for key in keys:
            doc = reports.get(key) or {}
            metainfo = doc.get("metainfo") or {}
            sub = subset.get(key) or {}
            cname = str(metainfo.get("company_name") or sub.get("company_name") or "").strip()
            if not cname:
                continue
            if not self._acl.is_company_allowed(tenant_id, cname):
                continue
            currency = str(
                metainfo.get("currency") or sub.get("currency") or ""
            ).strip() or None
            industry = str(
                metainfo.get("major_industry") or sub.get("major_industry") or ""
            ).strip() or None
            if params.industry and (industry or "").casefold() != params.industry.strip().casefold():
                continue
            if params.currency and (currency or "").casefold() != params.currency.strip().casefold():
                continue
            if params.keyword and params.keyword.strip().casefold() not in cname.casefold():
                continue
            content = doc.get("content") or {}
            pages = content.get("pages") or []
            sha1 = str(metainfo.get("sha1_name") or metainfo.get("sha1") or sub.get("sha1") or "").strip()
            items.append(
                FinaragCompanyMeta(
                    company_name=cname,
                    sha1_name=sha1,
                    currency=currency,
                    major_industry=industry,
                    pages_amount=int(metainfo.get("pages_amount") or len(pages)),
                    tables_amount=int(metainfo.get("tables_amount") or 0),
                )
            )
        return FinaragListCompaniesOutput(
            companies=items[: params.limit],
            total=len(items),
        )

    def _embed_text(self, text: str) -> list[float]:
        if self._embedder is not None:
            return _normalize_vector(self._embedder(text))
        return _hashed_dense_vector(text)

    def retrieve_filings(
        self,
        params: FinaragFilingsInput,
        *,
        tenant_id: str,
    ) -> FinaragFilingsOutput:
        doc = self._resolve_company_doc(
            tenant_id=tenant_id,
            company_name=params.company_name,
            use_serialized_tables=params.use_serialized_tables,
        )
        metainfo = doc.get("metainfo") or {}
        content = doc.get("content") or {}
        sha1 = str(metainfo.get("sha1_name") or metainfo.get("sha1") or "unknown").strip()
        canon_name = str(metainfo.get("company_name") or params.company_name).strip()
        currency = str(metainfo.get("currency") or "").strip() or None

        pages = self._valid_passages(content.get("pages"))
        pages_by_no = {page["page"]: page["text"] for page in pages}
        chunks = self._valid_passages(content.get("chunks")) or pages
        terms = set(_tokenize(params.query))
        counts = [Counter(_tokenize(chunk["text"])) for chunk in chunks]
        average_length = sum(c.total() for c in counts) / max(1, len(counts)) or 1.0
        weights = {
            term: math.log1p((len(counts) - frequency + 0.5) / (frequency + 0.5))
            for term in terms
            for frequency in [sum(term in count for count in counts)]
        }

        bm25_scores: list[float] = []
        for count in counts:
            normalization = 1.5 * (0.25 + 0.75 * count.total() / average_length)
            bm25_scores.append(
                sum(
                    weights[term] * count[term] * 2.5 / (count[term] + normalization)
                    for term in terms
                    if term in count
                )
            )

        use_hybrid_rrf = self._hybrid_rrf_enabled and params.retrieval_mode in {"hybrid", "vector"}
        passages: list[FinaragPassageItem] = []
        seen_pages: set[int] = set()

        if not use_hybrid_rrf:
            scored = [
                (score, chunk)
                for score, chunk in zip(bm25_scores, chunks)
                if score > 0
            ]
            scored.sort(key=lambda item: (-item[0], item[1]["page"]))
            for score, chunk in scored:
                page_1based = chunk["page"]
                page_0based = page_1based - 1
                if params.return_parent_pages and page_1based in pages_by_no:
                    if page_1based in seen_pages:
                        continue
                    seen_pages.add(page_1based)
                    passage_text = pages_by_no[page_1based]
                else:
                    passage_text = str(chunk.get("text") or "")
                chunk_types = [str(chunk.get("type") or "content")]
                passages.append(
                    FinaragPassageItem(
                        page_index_1based=page_1based,
                        page_index_0based=page_0based,
                        distance=round(max(0.0, 1.0 - min(score, 1.0)), 4),
                        relevance_score=round(min(1.0, score), 4),
                        combined_score=round(score, 4),
                        text=passage_text,
                        chunk_types=chunk_types,
                        citation_uri=f"finarag://{sha1}#page={page_1based}",
                    )
                )
                if len(passages) >= params.top_n:
                    break

            return FinaragFilingsOutput(
                company_name=canon_name,
                pdf_sha1=sha1,
                currency=currency,
                retrieval_mode="bm25",
                passages=passages,
                notices=(
                    [f"{params.retrieval_mode} retrieval is unavailable. Used local BM25."]
                    if params.retrieval_mode != "bm25" else []
                ),
            )

        query_vec = self._embed_text(params.query)
        dense_scores = [_cosine_similarity(query_vec, self._embed_text(chunk["text"])) for chunk in chunks]

        bm25_order = sorted(
            (idx for idx, s in enumerate(bm25_scores) if s > 0.0),
            key=lambda idx: (-bm25_scores[idx], chunks[idx]["page"], idx),
        )
        dense_order = sorted(
            (idx for idx, s in enumerate(dense_scores) if s > 0.0),
            key=lambda idx: (-dense_scores[idx], chunks[idx]["page"], idx),
        )
        bm25_ranks = {idx: pos for pos, idx in enumerate(bm25_order, start=1)}
        dense_ranks = {idx: pos for pos, idx in enumerate(dense_order, start=1)}

        fused: list[tuple[float, float, float, int]] = []
        k = self._rrf_k
        for idx in sorted(set(bm25_ranks) | set(dense_ranks)):
            r_bm25 = bm25_ranks.get(idx)
            r_dense = dense_ranks.get(idx)
            rrf = (1.0 / (k + r_bm25) if r_bm25 is not None else 0.0) + (
                1.0 / (k + r_dense) if r_dense is not None else 0.0
            )
            fused.append((rrf, bm25_scores[idx], dense_scores[idx], idx))

        fused.sort(key=lambda item: (-item[0], -item[1], -item[2], chunks[item[3]]["page"], item[3]))

        for rrf_score, b_score, d_score, idx in fused:
            chunk = chunks[idx]
            page_1based = chunk["page"]
            page_0based = page_1based - 1
            if params.return_parent_pages and page_1based in pages_by_no:
                if page_1based in seen_pages:
                    continue
                seen_pages.add(page_1based)
                passage_text = pages_by_no[page_1based]
            else:
                passage_text = str(chunk.get("text") or "")
            chunk_types = [str(chunk.get("type") or "content")]
            signal = max(d_score, min(b_score, 1.0))
            passages.append(
                FinaragPassageItem(
                    page_index_1based=page_1based,
                    page_index_0based=page_0based,
                    distance=round(max(0.0, 1.0 - min(signal, 1.0)), 4),
                    relevance_score=round(min(1.0, signal), 4),
                    combined_score=round(rrf_score, 6),
                    bm25_rank=bm25_ranks.get(idx),
                    dense_rank=dense_ranks.get(idx),
                    rrf_score=round(rrf_score, 6),
                    text=passage_text,
                    chunk_types=chunk_types,
                    citation_uri=f"finarag://{sha1}#page={page_1based}",
                )
            )
            if len(passages) >= params.top_n:
                break

        return FinaragFilingsOutput(
            company_name=canon_name,
            pdf_sha1=sha1,
            currency=currency,
            retrieval_mode="hybrid_rrf",
            passages=passages,
            notices=[],
        )

    @staticmethod
    def _valid_passages(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        return [
            item for item in raw if isinstance(item, dict)
            and type(item.get("page")) is int and item["page"] > 0
            and isinstance(item.get("text"), str) and item["text"].strip()
        ]

    def structured_query(
        self,
        params: FinaragStructuredQueryInput,
        *,
        tenant_id: str,
    ) -> FinaragStructuredQueryOutput:
        filings = self.retrieve_filings(
            FinaragFilingsInput(
                company_name=params.company_name,
                query=params.question,
                retrieval_mode="hybrid",
                top_n=params.top_n_retrieval,
                return_parent_pages=params.return_parent_pages,
                use_serialized_tables=params.use_serialized_tables,
            ),
            tenant_id=tenant_id,
        )
        excerpts = [
            FinaragPageExcerpt(page_index=p.page_index_1based, text=p.text)
            for p in filings.passages
        ]
        relevant_pages = [p.page_index_1based for p in filings.passages]
        refs = [
            FinaragPageRef(
                pdf_sha1=filings.pdf_sha1,
                page_index=p.page_index_1based,
                citation_uri=p.citation_uri,
            )
            for p in filings.passages
        ]

        final_answer: Any = "N/A"
        reasoning = "Structured answer generation is unavailable. Retrieved filing excerpts are provided."
        if self._answer_fn is not None and filings.passages:
            final_answer, reasoning = self._answer_fn(
                params.question,
                [p.model_dump() for p in filings.passages],
                params.answer_schema,
            )

        return FinaragStructuredQueryOutput(
            company_name=filings.company_name,
            pdf_sha1=filings.pdf_sha1,
            answer_schema=params.answer_schema,
            final_answer=final_answer,
            reasoning_summary=reasoning,
            step_by_step_analysis=f"Retrieved {len(excerpts)} filing page(s) for {filings.company_name}.",
            relevant_pages=relevant_pages,
            references=refs,
            excerpts=excerpts,
            answer_reason=(
                "no_matching_pages" if not excerpts else
                "grounded_filings_lookup" if self._answer_fn is not None else "answerer_unavailable"
            ),
        )

    def get_report_pages(
        self,
        params: FinaragGetPagesInput,
        *,
        tenant_id: str,
    ) -> FinaragGetPagesOutput:
        doc = self._resolve_company_doc(
            tenant_id=tenant_id,
            company_name=params.company_name,
            use_serialized_tables=params.include_serialized_tables,
        )
        metainfo = doc.get("metainfo") or {}
        content = doc.get("content") or {}
        sha1 = str(metainfo.get("sha1_name") or metainfo.get("sha1") or "unknown").strip()
        canon_name = str(metainfo.get("company_name") or params.company_name).strip()
        pages_by_no = {p["page"]: p["text"] for p in self._valid_passages(content.get("pages"))}
        selected = [
            FinaragPageExcerpt(page_index=int(pno), text=pages_by_no[int(pno)])
            for pno in dict.fromkeys(params.page_numbers)
            if int(pno) in pages_by_no
        ]
        return FinaragGetPagesOutput(
            company_name=canon_name,
            pdf_sha1=sha1,
            pages=selected,
        )


class FinaRAGMCPServer:
    """Standards-compliant MCP JSON-RPC 2.0 server exposing FinaRAG read-only tools."""

    SERVER_ID = "finarag"

    def __init__(self, adapter: FinaRAGAdapter | None = None) -> None:
        self._adapter = adapter or FinaRAGAdapter()
        self._tools: list[dict[str, Any]] = [
            {
                "name": "finarag_list_companies",
                "description": "List companies and filing metadata accessible to the current tenant.",
                "inputSchema": FinaragListCompaniesInput.model_json_schema(),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
            {
                "name": "finarag_filings",
                "description": "Search private financial filings for a company and return page-grounded excerpts.",
                "inputSchema": FinaragFilingsInput.model_json_schema(),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
            {
                "name": "finarag_structured_query",
                "description": "Answer a structured financial question over company filings with page references.",
                "inputSchema": FinaragStructuredQueryInput.model_json_schema(),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
            {
                "name": "finarag_get_report_pages",
                "description": "Fetch exact 1-based page texts from a company filing by page numbers.",
                "inputSchema": FinaragGetPagesInput.model_json_schema(),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
        ]

    @staticmethod
    def _error(request_id: str | int | None, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def handle_jsonrpc(
        self, payload: Any, *, headers: dict[str, str] | None = None, require_token: bool = False,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return self._error(None, -32600, "Invalid Request")
        request_id = payload.get("id")
        method = payload.get("method")
        notification = "id" not in payload
        if (
            payload.get("jsonrpc") != "2.0" or not isinstance(method, str) or not method
            or (not notification and type(request_id) not in {str, int})
        ):
            return self._error(None, -32600, "Invalid Request")
        if notification:
            return None

        params = payload.get("params", {})
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "Invalid params")
        if method == "initialize":
            if (
                not isinstance(params.get("protocolVersion"), str)
                or not isinstance(params.get("capabilities"), dict)
                or not isinstance(params.get("clientInfo"), dict)
                or not all(isinstance(params["clientInfo"].get(k), str) for k in ("name", "version"))
            ):
                return self._error(request_id, -32602, "Invalid initialize params")
            result = {
                "protocolVersion": _PROTOCOL_VERSION,
                "serverInfo": {"name": "finarag-mcp", "version": "1.0.0"},
                "capabilities": {"tools": {"listChanged": False}},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            if "cursor" in params:
                return self._error(request_id, -32602, "Invalid tools cursor")
            result = {"tools": self._tools}
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            meta = params.get("_meta", {})
            if not isinstance(tool_name, str) or not isinstance(arguments, dict) or not isinstance(meta, dict):
                return self._error(request_id, -32602, "Invalid tool call params")
            if tool_name not in {spec["name"] for spec in self._tools}:
                return self._error(request_id, -32602, "Unknown FinaRAG tool")

            norm_headers = {k.lower(): v for k, v in (headers or {}).items()}
            auth_parts = norm_headers.get("authorization", "").split(None, 1)
            bearer = auth_parts[1] if len(auth_parts) == 2 and auth_parts[0].lower() == "bearer" else None
            identities = {
                "tenant_id": norm_headers.get("x-tenant-id"),
                "request_id": norm_headers.get("x-request-id"),
                "scoped_token": bearer,
            }
            for key, value in identities.items():
                metadata_value = meta.get(key)
                if metadata_value is not None and (not isinstance(metadata_value, str) or value and metadata_value != value):
                    return self._error(request_id, -32003, "Conflicting MCP identity metadata")
                identities[key] = value or metadata_value
            try:
                tenant = self._adapter.authorize_request(
                    tenant_id=identities["tenant_id"] or "default",
                    scoped_token=identities["scoped_token"],
                    request_id=identities["request_id"],
                    require_token=require_token,
                )
                structured = self._dispatch_tool(tool_name=tool_name, arguments=arguments, tenant_id=tenant)
                result = {
                    "content": [{"type": "text", "text": json.dumps(structured, ensure_ascii=False, allow_nan=False)}],
                    "structuredContent": structured,
                    "isError": False,
                }
            except PermissionError as exc:
                return self._error(request_id, -32003, str(exc))
            except ValidationError:
                return self._error(request_id, -32602, "Invalid tool arguments")
            except _ReportUnavailable:
                result = {"content": [{"type": "text", "text": "FinaRAG report unavailable"}], "isError": True}
            except Exception:
                return self._error(request_id, -32603, "Internal error")
        else:
            return self._error(request_id, -32601, "Method not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _dispatch_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        if tool_name == "finarag_list_companies":
            parsed = FinaragListCompaniesInput.model_validate(arguments)
            return self._adapter.list_companies(parsed, tenant_id=tenant_id).model_dump()
        if tool_name == "finarag_filings":
            parsed = FinaragFilingsInput.model_validate(arguments)
            return self._adapter.retrieve_filings(parsed, tenant_id=tenant_id).model_dump()
        if tool_name == "finarag_structured_query":
            parsed = FinaragStructuredQueryInput.model_validate(arguments)
            return self._adapter.structured_query(parsed, tenant_id=tenant_id).model_dump()
        if tool_name == "finarag_get_report_pages":
            parsed = FinaragGetPagesInput.model_validate(arguments)
            return self._adapter.get_report_pages(parsed, tenant_id=tenant_id).model_dump()
        raise ValueError(f"Unknown FinaRAG tool: {tool_name}")

    def create_http_router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/mcp")
        async def handle_mcp_post(request: Request) -> Response:
            origin = request.headers.get("origin")
            if origin:
                return Response(status_code=403)
            if request.headers.get("mcp-protocol-version", _PROTOCOL_VERSION) != _PROTOCOL_VERSION:
                return Response(status_code=400)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > _MAX_REQUEST_BYTES:
                    return Response(status_code=413)
            try:
                payload = json.loads(body)
            except (ValueError, RecursionError):
                return JSONResponse(self._error(None, -32700, "Parse error"), status_code=400)
            reply = await run_in_threadpool(
                self.handle_jsonrpc, payload, headers=dict(request.headers), require_token=True,
            )
            return Response(status_code=202) if reply is None else JSONResponse(reply)

        return router

    def run_stdio(self) -> None:
        for line in iter(lambda: sys.stdin.buffer.readline(_MAX_REQUEST_BYTES + 1), b""):
            if len(line) > _MAX_REQUEST_BYTES:
                return
            try:
                payload = json.loads(line)
            except (ValueError, RecursionError):
                reply = self._error(None, -32700, "Parse error")
            else:
                reply = self.handle_jsonrpc(payload)
            if reply is not None:
                sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="FinaRAG MCP Server")
    parser.add_argument("--transport", choices=("stdio", "http", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    server = FinaRAGMCPServer()
    if args.transport == "stdio":
        server.run_stdio()
        return

    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="FinaRAG MCP Server")
    app.include_router(server.create_http_router())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
