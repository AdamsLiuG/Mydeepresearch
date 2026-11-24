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
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal, Union

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+|[\u4e00-\u9fff]+")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?")


class FinaragListCompaniesInput(BaseModel):
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


class FinaragFilingsInput(BaseModel):
    company_name: str = Field(..., min_length=1, description="Exact company_name in FinaRAG repository")
    query: str = Field(..., min_length=1, description="Filing semantic or lexical retrieval query")
    retrieval_mode: Literal["vector", "hybrid", "bm25"] = Field(default="hybrid")
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


class FinaragStructuredQueryInput(BaseModel):
    company_name: str = Field(..., min_length=1)
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


class FinaragGetPagesInput(BaseModel):
    company_name: str = Field(..., min_length=1)
    page_numbers: list[int] = Field(..., min_length=1, max_length=10)
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
    if not hmac.compare_digest(sig_b64, expected_sig):
        raise PermissionError("invalid scoped tenant token signature")
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception as exc:
        raise PermissionError("malformed scoped tenant token payload") from exc
    if not isinstance(claims, dict):
        raise PermissionError("invalid scoped tenant token claims")
    exp = float(claims.get("exp") or 0)
    if exp < time.time():
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
                    normalized[str(tid)] = list(entry.get("companies") or entry.get("allowed_companies") or [])
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


class FinaRAGAdapter:
    """In-process / CLI-compatible retrieval adapter over FinaRAG chunked reports."""

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        acl: TenantACL | None = None,
        token_secret: str | None = None,
        answer_fn: Callable[[str, list[dict[str, Any]], str], tuple[Any, str]] | None = None,
    ) -> None:
        self._root = Path(
            data_root
            or os.getenv("FINARAG_DATA_ROOT")
            or "/media/main/hjz/llm/FinaRAG"
        )
        self._acl = acl or TenantACL.from_path(
            os.getenv("FINARAG_TENANT_ACL_PATH"),
            enforce=bool(os.getenv("FINARAG_TENANT_ACL_PATH")),
        )
        self._token_secret = token_secret if token_secret is not None else os.getenv("MCP_TENANT_TOKEN_SECRET")
        self._answer_fn = answer_fn
        self._subset_meta: dict[str, dict[str, Any]] | None = None
        self._reports_cache: dict[bool, dict[str, dict[str, Any]]] = {}

    def authorize_request(
        self,
        *,
        tenant_id: str,
        scoped_token: str | None,
    ) -> str:
        resolved_tenant = str(tenant_id or "").strip() or "default"
        if self._token_secret:
            if not scoped_token:
                raise PermissionError("missing scoped tenant token")
            claims = _verify_hmac_token(
                token=scoped_token,
                secret=self._token_secret,
                expected_server_id="finarag",
                expected_tenant_id=resolved_tenant,
            )
            resolved_tenant = str(claims.get("tenant_id") or resolved_tenant)
        return resolved_tenant

    def _load_subset_meta(self) -> dict[str, dict[str, Any]]:
        if self._subset_meta is not None:
            return self._subset_meta
        meta: dict[str, dict[str, Any]] = {}
        subset_csv = self._root / "subset.csv"
        if subset_csv.exists():
            with subset_csv.open("r", encoding="utf-8", errors="replace") as fh:
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
        self._subset_meta = meta
        return meta

    def _load_reports(self, *, use_serialized_tables: bool = False) -> dict[str, dict[str, Any]]:
        if use_serialized_tables in self._reports_cache:
            return self._reports_cache[use_serialized_tables]

        suffix = "_ser_tab" if use_serialized_tables else ""
        candidates = [
            self._root / f"databases{suffix}" / "chunked_reports",
            self._root / "databases" / "chunked_reports",
            self._root / "chunked_reports",
        ]
        if use_serialized_tables:
            candidates.insert(0, self._root / "databases_ser_tab" / "chunked_reports")

        reports: dict[str, dict[str, Any]] = {}
        for report_dir in candidates:
            if not report_dir.exists() or not report_dir.is_dir():
                continue
            for json_path in sorted(report_dir.glob("*.json")):
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                metainfo = payload.get("metainfo") or {}
                company_name = str(metainfo.get("company_name") or "").strip()
                if not company_name:
                    continue
                reports[company_name.casefold()] = payload
            if reports:
                break

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
            for cand_key, cand_doc in reports.items():
                if key in cand_key or cand_key in key:
                    if not self._acl.is_company_allowed(
                        tenant_id,
                        str((cand_doc.get("metainfo") or {}).get("company_name") or cand_key),
                    ):
                        raise PermissionError(
                            f"tenant '{tenant_id}' is not authorized to access matching company"
                        )
                    doc = cand_doc
                    break
        if doc is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
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
                    sha1_name=sha1 or hashlib.sha1(cname.encode("utf-8")).hexdigest()[:12],
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

        chunks = list(content.get("chunks") or [])
        pages = list(content.get("pages") or [])
        pages_by_no = {int(p.get("page") or 0): str(p.get("text") or "") for p in pages if isinstance(p, dict)}

        query_tokens = _tokenize(params.query)
        q_set = set(query_tokens)
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text") or "")
            if not text.strip():
                continue
            c_tokens = _tokenize(text)
            if not c_tokens:
                continue
            tf_hits = sum(1 for tok in c_tokens if tok in q_set)
            unique_hits = len(q_set & set(c_tokens))
            raw_score = (unique_hits * 2.5 + tf_hits * 0.4) / math.sqrt(len(c_tokens) + 8.0)
            scored.append((round(raw_score, 4), {**chunk, "_idx": idx}))

        if not scored and pages:
            for page in pages[: params.top_n]:
                pno = int(page.get("page") or 1)
                scored.append((0.1, {"page": pno, "text": str(page.get("text") or ""), "type": "page"}))

        scored.sort(key=lambda item: (-item[0], int(item[1].get("page") or 0)))

        passages: list[FinaragPassageItem] = []
        seen_pages: set[int] = set()
        for score, chunk in scored:
            raw_page = int(chunk.get("page") or 1)
            page_1based = max(1, raw_page)
            page_0based = max(0, page_1based - 1)
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
            retrieval_mode=params.retrieval_mode,
            passages=passages,
            notices=[],
        )

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
        reasoning = "Synthesized directly from retrieved filing excerpts."
        if self._answer_fn is not None:
            final_answer, reasoning = self._answer_fn(
                params.question,
                [p.model_dump() for p in filings.passages],
                params.answer_schema,
            )
        elif filings.passages:
            top_text = filings.passages[0].text
            if params.answer_schema == "number":
                nums = _NUMBER_RE.findall(top_text)
                if nums:
                    tok = nums[0].replace(",", "").rstrip("%")
                    try:
                        final_answer = float(tok) if "." in tok else int(tok)
                    except ValueError:
                        final_answer = nums[0]
            elif params.answer_schema == "boolean":
                lowered = top_text.lower()
                final_answer = "yes" in lowered or "true" in lowered or "increased" in lowered
            else:
                final_answer = top_text.splitlines()[0][:180] if top_text.splitlines() else "N/A"

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
            answer_reason="grounded_filings_lookup" if excerpts else "no_matching_pages",
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
        pages_by_no = {
            int(p.get("page") or 0): str(p.get("text") or "")
            for p in (content.get("pages") or [])
            if isinstance(p, dict)
        }
        selected = [
            FinaragPageExcerpt(page_index=int(pno), text=pages_by_no[int(pno)])
            for pno in params.page_numbers
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

    def handle_jsonrpc(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        req_id = payload.get("id")
        method = str(payload.get("method") or "").strip()
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": "finarag-mcp", "version": "1.0.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self._tools},
            }

        if method == "tools/call":
            tool_name = str(params.get("name") or "").strip()
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
            norm_headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
            tenant_id = str(
                meta.get("tenant_id") or norm_headers.get("x-tenant-id") or "default"
            ).strip()
            scoped_token = str(
                meta.get("scoped_token")
                or norm_headers.get("authorization", "").removeprefix("Bearer ").strip()
            ).strip() or None

            try:
                resolved_tenant = self._adapter.authorize_request(
                    tenant_id=tenant_id,
                    scoped_token=scoped_token,
                )
                structured = self._dispatch_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    tenant_id=resolved_tenant,
                )
            except PermissionError as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32003, "message": str(exc)},
                }
            except ValueError as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": str(exc)},
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(structured, ensure_ascii=False),
                        }
                    ],
                    "structuredContent": structured,
                    "isError": False,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

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
        async def handle_mcp_post(request: Request) -> JSONResponse:
            payload = await request.json()
            reply = self.handle_jsonrpc(payload, headers=dict(request.headers))
            return JSONResponse(content=reply or {})

        return router

    def run_stdio(self) -> None:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
                sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue
            if not isinstance(payload, dict):
                continue
            reply = self.handle_jsonrpc(payload)
            if reply is not None:
                sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="FinaRAG MCP Server")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
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
