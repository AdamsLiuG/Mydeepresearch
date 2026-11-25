"""Receipt-gated temporal graph storage and deterministic tenant-local retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Literal, Protocol

from models import EvidenceRef, GraphPurpose, VerifiedFactReceipt
from services.memory_sanitizer import (
    WEB_SANITIZER_VERSION,
    collapse_whitespace,
    sanitize_web_text,
)
from services.memory_scope import (
    MemoryScope,
    checked_path,
    open_partition_directory,
    partition_root,
    valid_internal_id,
    valid_scope,
)

SCHEMA_VERSION = 1
NORMALIZATION_VERSION = 1
MAX_RECEIPTS = 64
MAX_BATCH_BYTES = 128 * 1024
MAX_HISTORICAL_RECEIPTS = 128
MAX_CANDIDATES = 100
MAX_HYDRATIONS = 32
MAX_DIRECTION_ROWS = 16
UTC = timezone.utc
ENTITY_TYPES = frozenset({"organization", "product", "person", "place", "policy"})
RELATIONSHIPS = frozenset({"subsidiary_of", "supplies_to", "operates_in", "succeeds"})
SINGLE_VALUED_RELATIONSHIPS = frozenset({"succeeds"})
PURPOSES = frozenset({"plan", "task", "report"})
TERMINAL_STATUSES = frozenset({"completed", "done", "partial_success", "success"})
_HASH = re.compile(r"[a-f0-9]{64}")
_INTERNAL_KEY = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_SCALES = {"万": Decimal(10_000), "亿": Decimal(100_000_000)}
_CURRENCIES = frozenset({"CNY", "USD", "EUR", "JPY", "GBP", "HKD", "CAD", "AUD", "CHF", "KRW"})
_CURRENCY_ALIASES = {"人民币": "CNY", "元": "CNY", "美元": "USD", "欧元": "EUR", "日元": "JPY"}
_METRIC_ALIASES = {
    "营收": "revenue", "营业收入": "revenue", "总收入": "revenue",
    "利润": "profit", "净利润": "net_profit", "利润率": "profit_margin",
    "延迟": "latency", "数量": "count", "百分比": "percentage",
}
_METRIC_DIMENSIONS = {
    "revenue": "currency", "profit": "currency", "net_profit": "currency",
    "cost": "currency", "market_cap": "currency", "profit_margin": "ratio",
    "growth_rate": "ratio", "market_share": "ratio", "percentage": "ratio",
    "ratio": "ratio", "latency": "duration", "duration": "duration",
    "count": "count", "employees": "count", "population": "count",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_value)


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError("unsupported graph value")


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _id(prefix: str, *values: Any) -> str:
    return f"{prefix}_{_sha(values)}"


def _clean(value: str, *, limit: int = 512, multiline: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("invalid graph text")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n"))
    if any(unicodedata.category(char) in {"Cc", "Cf"} and char not in "\n\t" for char in normalized):
        raise ValueError("invalid graph text")
    normalized = "\n".join(collapse_whitespace(line) for line in normalized.split("\n")).strip()
    if sanitize_web_text(normalized, max_chars=limit) != normalized:
        raise ValueError("unsafe graph text")
    return normalized if multiline else collapse_whitespace(normalized)


def _name(value: str) -> str:
    return unicodedata.normalize("NFKC", _clean(value)).casefold()


def _metric(value: str) -> str:
    normalized = _name(value).replace(" ", "_")
    key = _METRIC_ALIASES.get(normalized, normalized)
    if not _INTERNAL_KEY.fullmatch(key):
        raise ValueError("invalid metric")
    return key


def _utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError("graph system time must be UTC")
    return value.isoformat(timespec="microseconds")


def _bounded_integer(value: Any, high: int = 2**63 - 1) -> bool:
    return type(value) is int and 0 < value <= high


def _business_date(value: date | None) -> bool:
    return value is None or type(value) is date


@dataclass(frozen=True)
class GraphEntity:
    entity_type: str
    name: str
    disambiguation: str = ""
    trusted_namespace: str = ""
    trusted_identifier: str = ""

    def __post_init__(self) -> None:
        if self.entity_type not in ENTITY_TYPES:
            raise ValueError("invalid entity type")
        for key in ("name", "disambiguation", "trusted_namespace", "trusted_identifier"):
            object.__setattr__(self, key, _clean(getattr(self, key)))
        if not self.name or bool(self.trusted_namespace) != bool(self.trusted_identifier):
            raise ValueError("invalid canonical entity identity")

    @property
    def canonical_key(self) -> str:
        if self.trusted_namespace:
            return _sha(("identifier", self.trusted_namespace, self.trusted_identifier))
        return _sha(("name", _name(self.name), _name(self.disambiguation)))

    @property
    def entity_id(self) -> str:
        return _id("entity", self.entity_type, self.canonical_key)


@dataclass(frozen=True)
class ValidPeriod:
    valid_from: date | None = None
    valid_to: date | None = None
    label: str = ""
    calendar_id: str = ""
    grain: str = ""

    def __post_init__(self) -> None:
        if not _business_date(self.valid_from) or not _business_date(self.valid_to):
            raise ValueError("invalid business date")
        if self.valid_to is not None and (self.valid_from is None or self.valid_to <= self.valid_from):
            raise ValueError("invalid half-open period")
        object.__setattr__(self, "label", _clean(self.label))
        object.__setattr__(self, "calendar_id", _clean(self.calendar_id))
        object.__setattr__(self, "grain", _name(self.grain))
        if self.grain and self.grain not in {"day", "month", "quarter", "year", "week"}:
            raise ValueError("invalid period grain")
        if self.label.casefold().startswith("fy") and self.valid_from and not self.calendar_id:
            raise ValueError("fiscal dates require a verified calendar")

    @property
    def mode(self) -> str:
        if self.valid_from is None:
            return "unknown"
        return "open" if self.valid_to is None else "finite"

    @property
    def bounds(self) -> tuple[str | None, str | None]:
        return (
            self.valid_from.isoformat() if self.valid_from else None,
            self.valid_to.isoformat() if self.valid_to else None,
        )

    def intersects(self, other: ValidPeriod) -> ValidPeriod | None:
        if self.valid_from is None or other.valid_from is None:
            return None
        start = max(self.valid_from, other.valid_from)
        stops = [item for item in (self.valid_to, other.valid_to) if item is not None]
        stop = min(stops) if stops else None
        return ValidPeriod(start, stop) if stop is None or start < stop else None

    def matches(self, valid_at: date | None, period_range: tuple[date, date] | None) -> bool:
        if valid_at is not None:
            return self.valid_from is not None and self.valid_from <= valid_at and (
                self.valid_to is None or valid_at < self.valid_to
            )
        return period_range is None or self.intersects(ValidPeriod(*period_range)) is not None


def _scope_fields(value: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    normalized = tuple(sorted((_name(key), _name(item)) for key, item in value))
    if len(normalized) > 16 or any(not key or not item for key, item in normalized):
        raise ValueError("invalid comparison scope")
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("duplicate comparison scope field")
    return normalized


def _decimal_text(value: Decimal) -> str:
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _normalize_value(metric: str, value: str, unit: str) -> tuple[str, str, str, str]:
    raw_value = unicodedata.normalize("NFKC", _clean(value, limit=64))
    raw_unit = unicodedata.normalize("NFKC", _clean(unit, limit=32))
    if not _DECIMAL.fullmatch(raw_value):
        raise ValueError("observation must have one finite decimal value")
    factor = Decimal(1)
    for scale, multiplier in _SCALES.items():
        if raw_unit.startswith(scale):
            factor = multiplier
            raw_unit = raw_unit[len(scale):]
            break
    currency = _CURRENCY_ALIASES.get(raw_unit, raw_unit.upper())
    if currency in _CURRENCIES:
        dimension, canonical_unit = "currency", currency
    elif raw_unit in {"s", "ms", "秒", "毫秒"} and factor == 1:
        currency, dimension, canonical_unit = "", "duration", "s"
        if raw_unit in {"ms", "毫秒"}:
            factor = Decimal("0.001")
    elif raw_unit in {"%", "percent", "ratio", "比例"} and factor == 1:
        currency, dimension, canonical_unit = "", "ratio", "ratio"
        if raw_unit in {"%", "percent"}:
            factor = Decimal("0.01")
    elif raw_unit in {"", "count", "人", "个", "件"}:
        currency, dimension, canonical_unit = "", "count", "count"
    else:
        raise ValueError("unregistered observation unit")
    if _METRIC_DIMENSIONS.get(metric, dimension) != dimension:
        raise ValueError("unit does not match metric")
    with localcontext() as context:
        context.prec = 100
        normalized = Decimal(raw_value) * factor
    return _decimal_text(normalized), dimension, canonical_unit, currency


@dataclass(frozen=True)
class ObservationFact:
    subject: GraphEntity
    metric_key: str
    value: str
    unit: str
    period: ValidPeriod
    scope: tuple[tuple[str, str], ...] = ()
    precision: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_key", _metric(self.metric_key))
        object.__setattr__(self, "value", _clean(self.value, limit=64))
        object.__setattr__(self, "unit", _clean(self.unit, limit=32))
        object.__setattr__(self, "scope", _scope_fields(self.scope))
        if self.period.mode == "open":
            raise ValueError("observation period must be finite or unknown")
        if self.precision is not None and (type(self.precision) is not int or not 0 <= self.precision <= 64):
            raise ValueError("invalid explicit precision")
        self.normalized

    @property
    def normalized(self) -> tuple[str, str, str, str]:
        return _normalize_value(self.metric_key, self.value, self.unit)

    @property
    def scope_key(self) -> str:
        return _sha(self.scope) if self.scope else ""

    @property
    def comparison_key(self) -> tuple[str, ...]:
        return (self.metric_key, self.scope_key, *self.normalized[1:], self.period.calendar_id, self.period.grain)

    @property
    def comparable(self) -> bool:
        return bool(self.scope and self.period.mode == "finite" and self.period.calendar_id and self.period.grain)

    @property
    def observation_id(self) -> str:
        return _id(
            "observation", self.subject.entity_id, self.comparison_key, self.normalized[0], self.period.bounds,
            self.period.label if self.period.mode == "unknown" else "",
        )

    @property
    def target_id(self) -> str:
        return _id("edge", self.subject.entity_id, "has_observation", self.observation_id)


@dataclass(frozen=True)
class RelationshipFact:
    subject: GraphEntity
    predicate: str
    object: GraphEntity
    period: ValidPeriod
    scope: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.predicate not in RELATIONSHIPS or self.subject.entity_id == self.object.entity_id:
            raise ValueError("invalid directional relation")
        object.__setattr__(self, "scope", _scope_fields(self.scope))

    @property
    def target_id(self) -> str:
        return _id(
            "edge", self.subject.entity_id, self.predicate, self.object.entity_id, self.period.bounds, self.scope,
            self.period.label if self.period.mode == "unknown" else "",
        )


@dataclass(frozen=True)
class AliasFact:
    subject: GraphEntity
    alias: str
    period: ValidPeriod = ValidPeriod()

    def __post_init__(self) -> None:
        object.__setattr__(self, "alias", _clean(self.alias))
        if not self.alias:
            raise ValueError("invalid alias")

    @property
    def target_id(self) -> str:
        return _id("alias", self.subject.entity_id, _name(self.alias), self.period.bounds)


TypedFact = ObservationFact | RelationshipFact | AliasFact


def _fact_data(fact: TypedFact) -> dict[str, Any]:
    if isinstance(fact, ObservationFact):
        return {"type": "observation", **asdict(fact), "normalized": fact.normalized}
    if isinstance(fact, RelationshipFact):
        return {"type": "relationship", **asdict(fact)}
    if isinstance(fact, AliasFact):
        return {"type": "alias", **asdict(fact)}
    raise ValueError("unknown graph fact type")


def _fact_from_payload(payload: dict[str, Any]) -> TypedFact:
    payload = payload.copy()
    fact_type = payload.pop("type")
    payload.pop("normalized", None)
    payload["subject"] = GraphEntity(**payload["subject"])
    period = payload["period"].copy()
    for name in ("valid_from", "valid_to"):
        period[name] = date.fromisoformat(period[name]) if period[name] else None
    payload["period"] = ValidPeriod(**period)
    if fact_type == "relationship":
        payload["object"] = GraphEntity(**payload["object"])
        return RelationshipFact(**payload)
    if fact_type == "observation":
        return ObservationFact(**payload)
    if fact_type == "alias":
        return AliasFact(**payload)
    raise ValueError("invalid stored graph fact")


def fact_fingerprint(ledger_version: int, statement: str, fact: TypedFact) -> str:
    if not _bounded_integer(ledger_version):
        raise ValueError("invalid ledger version")
    normalized = _clean(statement, limit=4096)
    if not normalized:
        raise ValueError("empty fact statement")
    return _sha((ledger_version, normalized, _fact_data(fact)))


def receipt_id_for(scope: MemoryScope, ledger_version: int, claim_id: str, fingerprint: str) -> str:
    return _id("receipt", scope.request_id, claim_id, ledger_version, fingerprint)


@dataclass(frozen=True)
class EvidenceFragment:
    chunk_id: str
    anchor: str
    text: str


def evidence_digest(fragments: tuple[EvidenceFragment, ...]) -> str:
    return _sha([asdict(fragment) for fragment in fragments])


@dataclass(frozen=True)
class GraphEvidence:
    tenant_id: str
    request_id: str
    source_id: str
    artifact_version: str
    fragments: tuple[EvidenceFragment, ...]
    level: str
    trusted: bool
    visible: bool
    facts_eligible: bool
    sanitizer_version: int
    isolated: bool = False
    origin: str = "web"
    acl_reference: str = ""


@dataclass(frozen=True)
class BoundAtom:
    atom_id: str
    ledger_version: int
    claim_id: str
    fact_fingerprint: str
    nli_label: str
    confidence: float
    verifier_version: str
    support: tuple[EvidenceRef, ...]
    is_fallback: bool = False
    verified_at: str = ""
    decomposition_version: int = 1


@dataclass(frozen=True)
class PublicationBinding:
    ledger_version: int
    claim_id: str
    fact_fingerprint: str
    support: tuple[EvidenceRef, ...]
    allowed: bool = True


@dataclass(frozen=True)
class VerifiedCandidate:
    ledger_version: int
    claim_id: str
    statement: str
    fact: TypedFact
    required_atomic_claim_ids: tuple[str, ...]
    atoms: tuple[BoundAtom, ...]
    publication: PublicationBinding | None
    deterministically_checked: bool = False
    unresolved_conflict: bool = False
    conflict_id: str = ""


@dataclass(frozen=True)
class Retirement:
    request_id: str
    receipt_id: str
    reason: Literal["superseded", "revoked"]


@dataclass(frozen=True)
class TerminalBatch:
    batch_id: str
    digest: str
    source_epoch: int
    ledger_version: int
    report_version: int
    publication_digest: str
    candidates: tuple[VerifiedCandidate, ...]
    retirements: tuple[Retirement, ...] = ()
    status: str = "pending"


@dataclass(frozen=True)
class TerminalGraphSnapshot:
    tenant_id: str
    request_id: str
    status: str
    policy_version: int
    owner_epoch: int | None
    batch: TerminalBatch
    historical_receipt_ids: tuple[str, ...] = ()
    active_receipt_ids: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    normalization_version: int = NORMALIZATION_VERSION


def _batch_content(
    scope: MemoryScope, candidates: tuple[VerifiedCandidate, ...], retirements: tuple[Retirement, ...],
    source_epoch: int, ledger_version: int, report_version: int,
) -> tuple[str, str, str]:
    if not all(_bounded_integer(number) for number in (source_epoch, ledger_version, report_version)):
        raise ValueError("invalid graph epoch")
    if len(candidates) > MAX_RECEIPTS or len(retirements) > MAX_HISTORICAL_RECEIPTS:
        raise ValueError("graph batch exceeds receipt limits")
    publications = [asdict(item.publication) if item.publication else None for item in candidates]
    publication_digest = _sha(sorted(publications, key=_json))
    payload = _json({
        "tenant_id": scope.tenant_id,
        "request_id": scope.request_id,
        "source_epoch": source_epoch,
        "ledger_version": ledger_version,
        "report_version": report_version,
        "publication_digest": publication_digest,
        "candidates": [asdict(value) for value in candidates],
        "retirements": sorted((asdict(value) for value in retirements), key=_json),
    }).encode("utf-8")
    if len(payload) > MAX_BATCH_BYTES:
        raise ValueError("graph batch exceeds size limit")
    digest = hashlib.sha256(payload).hexdigest()
    return _id("batch", scope.tenant_id, scope.request_id, source_epoch, report_version, digest), digest, publication_digest


def seal_terminal_batch(
    scope: MemoryScope, candidates: tuple[VerifiedCandidate, ...], *,
    source_epoch: int, ledger_version: int, report_version: int,
    retirements: tuple[Retirement, ...] = (),
) -> TerminalBatch:
    """Bind trusted final publication data to a batch before the terminal snapshot CAS."""
    batch_id, digest, publication = _batch_content(
        scope, tuple(candidates), tuple(retirements), source_epoch, ledger_version, report_version,
    )
    return TerminalBatch(batch_id, digest, source_epoch, ledger_version, report_version, publication,
                         tuple(candidates), tuple(retirements))


class GraphAuthority(Protocol):
    """Trusted adapter for the current request ACL, terminal snapshot and EvidenceStore."""

    def scope_allowed(self, scope: MemoryScope, *, purpose: str, deadline: float) -> bool: ...
    def authorize_origin(self, scope: MemoryScope, origin_request_id: str, *, purpose: str, deadline: float) -> bool: ...
    def load_terminal(
        self, scope: MemoryScope, origin_request_id: str, *, deadline: float,
    ) -> TerminalGraphSnapshot | None: ...
    def load_evidence(
        self, scope: MemoryScope, reference: EvidenceRef, *, purpose: str, deadline: float,
    ) -> GraphEvidence | None: ...


@dataclass(frozen=True)
class GraphWatermark:
    as_of: datetime
    graph_revision: int


@dataclass(frozen=True)
class GraphApplyResult:
    outcome: str
    inserted: int = 0
    revoked: int = 0
    superseded: int = 0
    watermark: GraphWatermark | None = None
    receipt_ids: tuple[str, ...] = ()
    rejected: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class ReceiptDecision:
    receipt: VerifiedFactReceipt | None
    reason: str = ""
    provenance: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class GraphQuery:
    seed_entities: tuple[str, ...]
    metrics: tuple[str, ...] = ()
    predicates: tuple[str, ...] = ()
    period_range: tuple[date, date] | None = None
    valid_at: date | None = None
    as_of: datetime | None = None
    max_hops: Literal[1, 2] = 2
    limit: int = 20
    graph_revision: int | None = None


@dataclass(frozen=True)
class GraphMatch:
    seed: GraphEntity
    facts: tuple[TypedFact, ...]
    path: tuple[str, ...]
    common_period: ValidPeriod | None
    support: tuple[tuple[EvidenceRef, ...], ...]
    stability: str
    confidence_ppm: int


@dataclass(frozen=True)
class GraphResult:
    status: str
    matches: tuple[GraphMatch, ...] = ()
    watermark: GraphWatermark | None = None
    ambiguities: tuple[tuple[GraphEntity, ...], ...] = ()
    candidates: int = 0
    hydrated_sources: int = 0


@dataclass(frozen=True)
class GraphProjection:
    phase: str
    status: str
    records: tuple[dict[str, Any], ...] = ()

    def encode(self) -> str:
        value = _json(asdict(self))
        return value.replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e")


@dataclass(frozen=True)
class EvolutionQuery:
    entity_ids: tuple[str, ...]
    metric_key: str
    scope_key: str | None = None
    period_range: tuple[date, date] | None = None
    as_of: datetime | None = None
    limit: int = 24
    graph_revision: int | None = None


@dataclass(frozen=True)
class EvolutionChange:
    old: GraphMatch
    new: GraphMatch
    absolute_delta: str
    percent_change: str | None
    unit: str
    percentage_points: str | None = None
    kind: Literal["temporal", "cross_entity"] = "temporal"


@dataclass(frozen=True)
class EvolutionSeries:
    entity: GraphEntity
    comparison_key: tuple[str, ...]
    points: tuple[GraphMatch, ...]
    status: str


@dataclass(frozen=True)
class EvolutionResult:
    status: str
    series: tuple[EvolutionSeries, ...] = ()
    changes: tuple[EvolutionChange, ...] = ()
    cross_entity: tuple[EvolutionChange, ...] = ()
    watermark: GraphWatermark | None = None
    candidates: int = 0
    hydrated_sources: int = 0

GRAPH_TABLES = (
    "graph_tenant_revisions", "graph_source_epochs", "graph_entities", "graph_observations",
    "graph_edges", "graph_receipts", "graph_aliases", "graph_provenance", "graph_conflicts",
    "graph_conflict_members", "graph_applied_batches",
)
GRAPH_SCHEMA_SQL = (
    """CREATE TABLE IF NOT EXISTS graph_tenant_revisions (
        tenant_id TEXT PRIMARY KEY, revision BIGINT NOT NULL, recorded_at TEXT NOT NULL,
        schema_version INTEGER NOT NULL, normalization_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS graph_source_epochs (
        tenant_id TEXT NOT NULL, request_id TEXT NOT NULL, source_epoch BIGINT NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
        PRIMARY KEY (tenant_id, request_id)
    )""",
    """CREATE TABLE IF NOT EXISTS graph_entities (
        tenant_id TEXT NOT NULL, entity_id TEXT NOT NULL, entity_type TEXT NOT NULL,
        canonical_key TEXT NOT NULL, normalized_name TEXT NOT NULL, name TEXT NOT NULL,
        trusted_namespace TEXT NOT NULL, normalization_version INTEGER NOT NULL,
        PRIMARY KEY (tenant_id, entity_id), UNIQUE (tenant_id, entity_type, canonical_key)
    )""",
    """CREATE INDEX IF NOT EXISTS graph_entity_name ON
        graph_entities (tenant_id, normalized_name, entity_type, canonical_key, entity_id)""",
    """CREATE TABLE IF NOT EXISTS graph_observations (
        tenant_id TEXT NOT NULL, observation_id TEXT NOT NULL, entity_id TEXT NOT NULL,
        metric_key TEXT NOT NULL, scope_key TEXT NOT NULL, unit_key TEXT NOT NULL,
        currency TEXT NOT NULL, calendar_id TEXT NOT NULL, period_grain TEXT NOT NULL,
        valid_from TEXT, valid_to TEXT, normalized_value TEXT NOT NULL, comparison_bucket TEXT,
        PRIMARY KEY (tenant_id, observation_id),
        FOREIGN KEY (tenant_id, entity_id) REFERENCES graph_entities (tenant_id, entity_id)
    )""",
    """CREATE INDEX IF NOT EXISTS graph_observation_time ON
        graph_observations (tenant_id, entity_id, metric_key, scope_key, valid_from, observation_id)""",
    """CREATE TABLE IF NOT EXISTS graph_edges (
        tenant_id TEXT NOT NULL, edge_id TEXT NOT NULL, subject_id TEXT NOT NULL,
        predicate TEXT NOT NULL, object_entity_id TEXT, observation_id TEXT, valid_from TEXT, valid_to TEXT,
        comparison_bucket TEXT, canonical_key TEXT NOT NULL,
        PRIMARY KEY (tenant_id, edge_id),
        CHECK ((object_entity_id IS NULL) <> (observation_id IS NULL)),
        CHECK ((predicate = 'has_observation') = (observation_id IS NOT NULL)),
        FOREIGN KEY (tenant_id, subject_id) REFERENCES graph_entities (tenant_id, entity_id),
        FOREIGN KEY (tenant_id, object_entity_id) REFERENCES graph_entities (tenant_id, entity_id),
        FOREIGN KEY (tenant_id, observation_id) REFERENCES graph_observations (tenant_id, observation_id)
    )""",
    """CREATE INDEX IF NOT EXISTS graph_edge_forward ON
        graph_edges (tenant_id, subject_id, predicate, valid_from, edge_id)""",
    """CREATE INDEX IF NOT EXISTS graph_edge_reverse ON
        graph_edges (tenant_id, object_entity_id, predicate, valid_from, edge_id)""",
    """CREATE TABLE IF NOT EXISTS graph_receipts (
        tenant_id TEXT NOT NULL, receipt_id TEXT NOT NULL, request_id TEXT NOT NULL, claim_id TEXT NOT NULL,
        ledger_version BIGINT NOT NULL, fact_fingerprint TEXT NOT NULL, receipt_digest TEXT NOT NULL,
        receipt_payload TEXT NOT NULL, fact_payload TEXT NOT NULL, edge_id TEXT, confidence_ppm INTEGER NOT NULL,
        nli_label TEXT NOT NULL CHECK (nli_label = 'ENTAILED'), source_epoch BIGINT NOT NULL,
        verifier_version TEXT NOT NULL, verification_policy_version INTEGER NOT NULL,
        verified_at TEXT NOT NULL, recorded_at TEXT NOT NULL, ended_at TEXT, start_revision BIGINT NOT NULL,
        end_revision BIGINT, retirement TEXT CHECK (retirement IN ('superseded', 'revoked')),
        PRIMARY KEY (tenant_id, request_id, claim_id, ledger_version, fact_fingerprint),
        UNIQUE (tenant_id, receipt_id),
        FOREIGN KEY (tenant_id, edge_id) REFERENCES graph_edges (tenant_id, edge_id)
    )""",
    """CREATE INDEX IF NOT EXISTS graph_receipt_edge ON
        graph_receipts (tenant_id, edge_id, start_revision, receipt_id)""",
    """CREATE TABLE IF NOT EXISTS graph_aliases (
        tenant_id TEXT NOT NULL, normalized_alias TEXT NOT NULL, entity_id TEXT NOT NULL,
        alias_version_id TEXT NOT NULL, entity_type TEXT NOT NULL, receipt_id TEXT NOT NULL,
        valid_from TEXT, valid_to TEXT, alias_kind TEXT NOT NULL CHECK (alias_kind IN ('primary', 'explicit')),
        PRIMARY KEY (tenant_id, normalized_alias, entity_id, alias_version_id),
        FOREIGN KEY (tenant_id, entity_id) REFERENCES graph_entities (tenant_id, entity_id),
        FOREIGN KEY (tenant_id, receipt_id) REFERENCES graph_receipts (tenant_id, receipt_id)
            ON DELETE CASCADE
    )""",
    """CREATE INDEX IF NOT EXISTS graph_alias_name ON
        graph_aliases (tenant_id, normalized_alias, entity_type, entity_id, alias_version_id)""",
    """CREATE INDEX IF NOT EXISTS graph_alias_entity ON
        graph_aliases (tenant_id, entity_id, normalized_alias, alias_version_id)""",
    """CREATE TABLE IF NOT EXISTS graph_provenance (
        tenant_id TEXT NOT NULL, receipt_id TEXT NOT NULL, source_id TEXT NOT NULL,
        artifact_version TEXT NOT NULL, chunk_id TEXT NOT NULL, request_id TEXT NOT NULL,
        claim_id TEXT NOT NULL, conflict_id TEXT NOT NULL, anchor TEXT NOT NULL,
        content_sha256 TEXT NOT NULL, acl_reference TEXT NOT NULL,
        PRIMARY KEY (tenant_id, receipt_id, source_id, artifact_version, chunk_id),
        FOREIGN KEY (tenant_id, receipt_id) REFERENCES graph_receipts (tenant_id, receipt_id)
            ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS graph_conflicts (
        tenant_id TEXT NOT NULL, conflict_id TEXT NOT NULL, key_version INTEGER NOT NULL,
        PRIMARY KEY (tenant_id, conflict_id)
    )""",
    """CREATE TABLE IF NOT EXISTS graph_conflict_members (
        tenant_id TEXT NOT NULL, conflict_id TEXT NOT NULL, receipt_id TEXT NOT NULL,
        PRIMARY KEY (tenant_id, conflict_id, receipt_id),
        FOREIGN KEY (tenant_id, conflict_id) REFERENCES graph_conflicts (tenant_id, conflict_id),
        FOREIGN KEY (tenant_id, receipt_id) REFERENCES graph_receipts (tenant_id, receipt_id)
            ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS graph_applied_batches (
        tenant_id TEXT NOT NULL, request_id TEXT NOT NULL, batch_id TEXT NOT NULL,
        digest TEXT NOT NULL, source_epoch BIGINT NOT NULL, graph_revision BIGINT NOT NULL,
        result_payload TEXT NOT NULL,
        PRIMARY KEY (tenant_id, request_id, batch_id)
    )""",
    """CREATE INDEX IF NOT EXISTS graph_applied_epoch ON
        graph_applied_batches (tenant_id, request_id, source_epoch)""",
)


class GraphStore(Protocol):
    backend: str

    def transaction(
        self, scope: MemoryScope, *, write: bool, deadline: float,
    ) -> AbstractContextManager[SqlGraphTransaction]: ...


class GraphIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class _StoredReceipt:
    receipt_id: str
    receipt: VerifiedFactReceipt
    fact: TypedFact
    start_revision: int
    end_revision: int | None
    retirement: str | None
    source_epoch: int


def _receipt_from_payload(value: str) -> VerifiedFactReceipt:
    data = json.loads(value)
    data["atomic_claim_ids"] = tuple(data["atomic_claim_ids"])
    data["support"] = tuple(
        EvidenceRef(**{**item, "chunk_ids": tuple(item["chunk_ids"])}) for item in data["support"]
    )
    return VerifiedFactReceipt(**data)


def _stored_receipt(row: tuple[Any, ...]) -> _StoredReceipt:
    return _StoredReceipt(
        str(row[0]), _receipt_from_payload(row[1]), _fact_from_payload(json.loads(row[2])),
        int(row[3]), row[4], row[5], int(row[6]),
    )


_RECEIPT_FIELDS = (
    "r.receipt_id, r.receipt_payload, r.fact_payload, r.start_revision, r.end_revision, r.retirement, r.source_epoch"
)


class SqlGraphTransaction:
    """Shared tenant-bound data operations for SQLite and Postgres."""

    def __init__(self, connection: Any, tenant_id: str, *, placeholder: str = "?") -> None:
        self.connection = connection
        self.tenant_id = tenant_id
        self.placeholder = placeholder

    def _execute(self, sql: str, *parameters: Any) -> Any:
        return self.connection.execute(sql.replace("?", self.placeholder), (self.tenant_id, *parameters))

    def _insert(self, table: str, columns: str, *values: Any, conflict: str = "") -> None:
        parameters = ", ".join("?" for _ in range(len(values) + 1))
        self._execute(f"INSERT INTO {table} (tenant_id, {columns}) VALUES ({parameters}) {conflict}", *values)

    def watermark(self, now: datetime) -> GraphWatermark:
        row = self._execute(
            """SELECT revision, recorded_at, schema_version, normalization_version
               FROM graph_tenant_revisions WHERE tenant_id = ?""",
        ).fetchone()
        if row is None:
            return GraphWatermark(now, 0)
        if int(row[2]) != SCHEMA_VERSION or int(row[3]) != NORMALIZATION_VERSION:
            raise GraphIntegrityError("unsupported graph schema")
        return GraphWatermark(datetime.fromisoformat(row[1]), int(row[0]))

    def allocate_revision(self, now: datetime) -> GraphWatermark:
        previous = self.watermark(now)
        current = max(now, previous.as_of + timedelta(microseconds=1)) if previous.graph_revision else now
        revision = previous.graph_revision + 1
        self._insert(
            "graph_tenant_revisions", "revision, recorded_at, schema_version, normalization_version",
            revision, _utc(current), SCHEMA_VERSION, NORMALIZATION_VERSION,
            conflict="ON CONFLICT (tenant_id) DO UPDATE SET revision = excluded.revision, recorded_at = excluded.recorded_at",
        )
        return GraphWatermark(current, revision)

    def get_batch(self, request_id: str, batch_id: str) -> tuple[str, GraphApplyResult] | None:
        row = self._execute(
            "SELECT digest, result_payload FROM graph_applied_batches WHERE tenant_id = ? AND request_id = ? AND batch_id = ?",
            request_id, batch_id,
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[1])
        watermark = data["watermark"]
        data["watermark"] = GraphWatermark(datetime.fromisoformat(watermark["as_of"]), watermark["graph_revision"])
        data["receipt_ids"] = tuple(data["receipt_ids"])
        data["rejected"] = tuple(tuple(item) for item in data["rejected"])
        return row[0], GraphApplyResult(**data)

    def get_source_epoch(self, request_id: str) -> int | None:
        row = self._execute(
            "SELECT source_epoch, deleted FROM graph_source_epochs WHERE tenant_id = ? AND request_id = ?", request_id,
        ).fetchone()
        return None if row is None else 2**63 - 1 if row[1] else int(row[0])

    def record_batch(self, request_id: str, batch: TerminalBatch, result: GraphApplyResult) -> None:
        assert result.watermark is not None
        self._insert(
            "graph_applied_batches", "request_id, batch_id, digest, source_epoch, graph_revision, result_payload",
            request_id, batch.batch_id, batch.digest, batch.source_epoch, result.watermark.graph_revision, _json(asdict(result)),
        )
        self._insert(
            "graph_source_epochs", "request_id, source_epoch", request_id, batch.source_epoch,
            conflict="ON CONFLICT (tenant_id, request_id) DO UPDATE SET source_epoch = excluded.source_epoch",
        )

    def _store_entity(self, entity: GraphEntity) -> None:
        self._insert(
            "graph_entities", "entity_id, entity_type, canonical_key, normalized_name, name, trusted_namespace, normalization_version",
            entity.entity_id, entity.entity_type, entity.canonical_key, _name(entity.name), entity.name,
            entity.trusted_namespace, NORMALIZATION_VERSION, conflict="ON CONFLICT (tenant_id, entity_id) DO NOTHING",
        )
        row = self._execute(
            "SELECT entity_type, canonical_key FROM graph_entities WHERE tenant_id = ? AND entity_id = ?", entity.entity_id,
        ).fetchone()
        if tuple(row) != (entity.entity_type, entity.canonical_key):
            raise GraphIntegrityError("entity key collision")

    def _comparison_bucket(self, fact: TypedFact) -> str | None:
        if isinstance(fact, ObservationFact) and fact.comparable:
            keys = ("observation", fact.subject.entity_id, fact.comparison_key, fact.period.bounds)
        elif isinstance(fact, RelationshipFact) and fact.predicate in SINGLE_VALUED_RELATIONSHIPS and fact.period.mode != "unknown":
            keys = ("relation", fact.subject.entity_id, fact.predicate, fact.scope)
        else:
            return None
        return _id("conflict", NORMALIZATION_VERSION, self.tenant_id, *keys)

    def _store_target(self, fact: TypedFact, bucket: str | None) -> str | None:
        self._store_entity(fact.subject)
        if isinstance(fact, AliasFact):
            return None
        observation_id = object_entity_id = None
        if isinstance(fact, ObservationFact):
            observation_id = fact.observation_id
            normalized_value, dimension, unit_key, currency = fact.normalized
            predicate = "has_observation"
            self._insert(
                "graph_observations",
                "observation_id, entity_id, metric_key, scope_key, unit_key, currency, calendar_id, period_grain, valid_from, valid_to, normalized_value, comparison_bucket",
                observation_id, fact.subject.entity_id, fact.metric_key, fact.scope_key, unit_key, currency,
                fact.period.calendar_id, fact.period.grain, *fact.period.bounds, normalized_value, bucket,
                conflict="ON CONFLICT (tenant_id, observation_id) DO NOTHING",
            )
            canonical = _sha((fact.comparison_key, normalized_value, dimension, fact.period.bounds))
        else:
            self._store_entity(fact.object)
            object_entity_id = fact.object.entity_id
            predicate = fact.predicate
            canonical = _sha((predicate, object_entity_id, fact.scope, fact.period.bounds))
        self._insert(
            "graph_edges", "edge_id, subject_id, predicate, object_entity_id, observation_id, valid_from, valid_to, comparison_bucket, canonical_key",
            fact.target_id, fact.subject.entity_id, predicate, object_entity_id, observation_id, *fact.period.bounds,
            bucket, canonical, conflict="ON CONFLICT (tenant_id, edge_id) DO NOTHING",
        )
        row = self._execute(
            "SELECT canonical_key FROM graph_edges WHERE tenant_id = ? AND edge_id = ?", fact.target_id,
        ).fetchone()
        if row[0] != canonical:
            raise GraphIntegrityError("edge key collision")
        return fact.target_id

    def get_receipt(self, receipt_id: str) -> _StoredReceipt | None:
        row = self._execute(
            f"SELECT {_RECEIPT_FIELDS} FROM graph_receipts r WHERE r.tenant_id = ? AND r.receipt_id = ?",
            receipt_id,
        ).fetchone()
        return _stored_receipt(row) if row else None

    def active_receipts(self, request_id: str) -> tuple[str, ...]:
        return tuple(row[0] for row in self._execute(
            """SELECT receipt_id FROM graph_receipts WHERE tenant_id = ? AND request_id = ?
               AND end_revision IS NULL ORDER BY receipt_id LIMIT ?""", request_id, MAX_RECEIPTS + 1,
        ).fetchall())

    def store_receipt(
        self, receipt_id: str, decision: ReceiptDecision, candidate: VerifiedCandidate,
        watermark: GraphWatermark, source_epoch: int, policy_version: int,
    ) -> bool:
        receipt = decision.receipt
        assert receipt is not None
        receipt_payload = _json(asdict(receipt))
        fact_payload = _json(_fact_data(candidate.fact))
        digest = _sha((receipt_payload, fact_payload))
        row = self._execute(
            "SELECT receipt_digest, retirement FROM graph_receipts WHERE tenant_id = ? AND receipt_id = ?", receipt_id,
        ).fetchone()
        if row is not None:
            if row[0] != digest or row[1] is not None:
                raise GraphIntegrityError("receipt key collision or retired receipt replay")
            return False

        fact = candidate.fact
        bucket = self._comparison_bucket(fact)
        edge_id = self._store_target(fact, bucket)
        self._insert(
            "graph_receipts",
            """receipt_id, request_id, claim_id, ledger_version, fact_fingerprint, receipt_digest,
               receipt_payload, fact_payload, edge_id, confidence_ppm, nli_label, source_epoch, verifier_version,
               verification_policy_version, verified_at, recorded_at, start_revision""",
            receipt_id, receipt.request_id, receipt.claim_id, receipt.ledger_version, receipt.fact_fingerprint,
            digest, receipt_payload, fact_payload, edge_id, receipt.confidence_ppm, receipt.nli_label, source_epoch,
            receipt.verifier_version, policy_version, receipt.verified_at, _utc(watermark.as_of), watermark.graph_revision,
        )
        aliases = [(fact.subject, fact.subject.name, "primary")]
        if isinstance(fact, RelationshipFact):
            aliases.append((fact.object, fact.object.name, "primary"))
        if isinstance(fact, AliasFact):
            aliases.append((fact.subject, fact.alias, "explicit"))
        for entity, alias, kind in aliases:
            normalized_alias = _name(alias)
            self._insert(
                "graph_aliases",
                "normalized_alias, entity_id, alias_version_id, entity_type, receipt_id, valid_from, valid_to, alias_kind",
                normalized_alias, entity.entity_id, _id("name", receipt_id, kind), entity.entity_type,
                receipt_id, *fact.period.bounds, kind,
            )
        for fragment in decision.provenance:
            self._insert(
                "graph_provenance",
                "receipt_id, source_id, artifact_version, chunk_id, request_id, claim_id, conflict_id, anchor, content_sha256, acl_reference",
                receipt_id, fragment["source_id"], fragment["artifact_version"], fragment["chunk_id"],
                receipt.request_id, receipt.claim_id, candidate.conflict_id, fragment["anchor"],
                fragment["content_sha256"], fragment["acl_reference"],
                conflict="ON CONFLICT (tenant_id, receipt_id, source_id, artifact_version, chunk_id) DO NOTHING",
            )
        if bucket:
            self._insert("graph_conflicts", "conflict_id, key_version", bucket, NORMALIZATION_VERSION,
                         conflict="ON CONFLICT (tenant_id, conflict_id) DO NOTHING")
            self._insert("graph_conflict_members", "conflict_id, receipt_id", bucket, receipt_id)
        return True

    def retire(self, retirement: Retirement, watermark: GraphWatermark) -> bool:
        row = self._execute(
            """WITH tenant AS (SELECT CAST(? AS TEXT) AS id)
               UPDATE graph_receipts SET retirement = ?, ended_at = COALESCE(ended_at, ?),
                   end_revision = COALESCE(end_revision, ?)
               WHERE tenant_id = (SELECT id FROM tenant) AND request_id = ? AND receipt_id = ?
                 AND (retirement IS NULL OR (retirement = 'superseded' AND ? = 'revoked'))
               RETURNING receipt_id""",
            retirement.reason, _utc(watermark.as_of), watermark.graph_revision, retirement.request_id,
            retirement.receipt_id, retirement.reason,
        ).fetchone()
        return row is not None

    @staticmethod
    def _visibility() -> str:
        return """r.start_revision <= ? AND r.recorded_at <= ?
                  AND (r.retirement IS NULL OR (r.retirement = 'superseded'
                        AND (r.end_revision > ? OR r.ended_at > ?)))"""

    @staticmethod
    def _visibility_values(watermark: GraphWatermark) -> tuple[Any, ...]:
        return (watermark.graph_revision, _utc(watermark.as_of)) * 2

    @staticmethod
    def _time_predicate(
        query: GraphQuery, alias: str = "e",
    ) -> tuple[str, tuple[Any, ...]]:
        if query.valid_at is not None:
            moment = query.valid_at.isoformat()
            return f"AND {alias}.valid_from <= ? AND ({alias}.valid_to IS NULL OR {alias}.valid_to > ?)", (moment, moment)
        if query.period_range is not None:
            start, stop = [item.isoformat() for item in query.period_range]
            return f"AND {alias}.valid_from < ? AND ({alias}.valid_to IS NULL OR {alias}.valid_to > ?)", (stop, start)
        return "", ()

    def seed_rows(
        self, seed: str, by_id: bool, query: GraphQuery, watermark: GraphWatermark, cap: int,
    ) -> list[tuple[str, _StoredReceipt]]:
        seed_column = "a.entity_id" if by_id else "a.normalized_alias"
        dates, date_args = self._time_predicate(query, "a")
        rows = self._execute(
            f"""SELECT a.entity_id, {_RECEIPT_FIELDS} FROM graph_aliases a
                JOIN graph_receipts r ON r.tenant_id = a.tenant_id AND r.receipt_id = a.receipt_id
                WHERE a.tenant_id = ? AND {seed_column} = ? AND {self._visibility()}
                AND (a.alias_kind = 'primary' OR (a.alias_kind = 'explicit' {dates}))
                ORDER BY a.entity_type, a.entity_id, a.alias_version_id LIMIT ?""",
            seed, *self._visibility_values(watermark), *date_args, cap,
        ).fetchall()
        return [(row[0], _stored_receipt(row[1:])) for row in rows]

    def adjacency(
        self, entity_id: str, reverse: bool, query: GraphQuery, watermark: GraphWatermark, cap: int,
    ) -> list[tuple[str, str | None, _StoredReceipt]]:
        endpoint = "object_entity_id" if reverse else "subject_id"
        dates, date_args = self._time_predicate(query)
        predicates = query.predicates
        predicate_filter = ""
        if predicates:
            predicate_filter = f"AND e.predicate IN ({', '.join('?' for _ in predicates)})"
        rows = self._execute(
            f"""SELECT e.edge_id, e.comparison_bucket, {_RECEIPT_FIELDS} FROM graph_edges e
                JOIN graph_receipts r ON r.tenant_id = e.tenant_id AND r.edge_id = e.edge_id
                WHERE e.tenant_id = ? AND e.{endpoint} = ? AND {self._visibility()}
                  {dates} {predicate_filter}
                ORDER BY COALESCE(e.valid_from, '') DESC, e.edge_id, r.receipt_id LIMIT ?""",
            entity_id, *self._visibility_values(watermark), *date_args, *predicates, cap,
        ).fetchall()
        return [(row[0], row[1], _stored_receipt(row[2:])) for row in rows]

    def conflict_rows(
        self, bucket: str, query: GraphQuery, watermark: GraphWatermark, cap: int,
    ) -> list[_StoredReceipt]:
        dates, date_args = self._time_predicate(query)
        rows = self._execute(
            f"""SELECT {_RECEIPT_FIELDS} FROM graph_conflict_members m
                JOIN graph_receipts r ON r.tenant_id = m.tenant_id AND r.receipt_id = m.receipt_id
                JOIN graph_edges e ON e.tenant_id = r.tenant_id AND e.edge_id = r.edge_id
                WHERE m.tenant_id = ? AND m.conflict_id = ? AND {self._visibility()} {dates}
                ORDER BY m.receipt_id LIMIT ?""",
            bucket, *self._visibility_values(watermark), *date_args, cap,
        ).fetchall()
        return [_stored_receipt(row) for row in rows]

    def count_receipts(self, maximum: int) -> int:
        return int(self._execute(
            """SELECT COUNT(*) FROM (
               SELECT receipt_id FROM graph_receipts WHERE tenant_id = ? LIMIT ?) AS receipts""", maximum,
        ).fetchone()[0])

    def purge_origin(self, request_id: str, *, limit: int, dry_run: bool) -> int:
        rows = self._execute(
            "SELECT receipt_id FROM graph_receipts WHERE tenant_id = ? AND request_id = ? ORDER BY receipt_id LIMIT ?",
            request_id, limit,
        ).fetchall()
        if dry_run:
            return len(rows)
        self._insert(
            "graph_source_epochs", "request_id, source_epoch, deleted", request_id, 0, 1,
            conflict="ON CONFLICT (tenant_id, request_id) DO UPDATE SET deleted = excluded.deleted",
        )
        receipt_ids = tuple(row[0] for row in rows)
        if receipt_ids:
            placeholders = ", ".join("?" for _ in receipt_ids)
            self._execute(
                f"DELETE FROM graph_receipts WHERE tenant_id = ? AND request_id = ? AND receipt_id IN ({placeholders})",
                request_id, *receipt_ids,
            )
        self._execute("DELETE FROM graph_applied_batches WHERE tenant_id = ? AND request_id = ?", request_id)
        for table, key, dependent, dependent_key in (
            ("graph_conflicts", "conflict_id", "graph_conflict_members", "conflict_id"),
            ("graph_edges", "edge_id", "graph_receipts", "edge_id"),
            ("graph_observations", "observation_id", "graph_edges", "observation_id"),
        ):
            self._execute(
                f"""DELETE FROM {table} WHERE tenant_id = ? AND NOT EXISTS (
                    SELECT 1 FROM {dependent} WHERE {dependent}.tenant_id = {table}.tenant_id
                    AND {dependent}.{dependent_key} = {table}.{key})""",
            )
        self._execute(
            """DELETE FROM graph_entities WHERE tenant_id = ?
               AND NOT EXISTS (SELECT 1 FROM graph_aliases a WHERE a.tenant_id = graph_entities.tenant_id
                               AND a.entity_id = graph_entities.entity_id)
               AND NOT EXISTS (SELECT 1 FROM graph_edges e WHERE e.tenant_id = graph_entities.tenant_id
                               AND (e.subject_id = graph_entities.entity_id OR e.object_entity_id = graph_entities.entity_id))""",
        )
        return len(rows)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return remaining


@contextmanager
def _sqlite_transaction(
    connection: sqlite3.Connection, scope: MemoryScope, *, write: bool, deadline: float, initialize: bool = False,
) -> Iterator[SqlGraphTransaction]:
    connection.execute(f"PRAGMA busy_timeout = {max(1, min(250, int(_remaining(deadline) * 1000)))}")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 500)
    try:
        connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        if initialize:
            for statement in GRAPH_SCHEMA_SQL:
                connection.execute(statement)
        yield SqlGraphTransaction(connection, scope.tenant_id)
        _remaining(deadline)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.set_progress_handler(None, 0)


class SQLiteGraphStore:
    backend = "sqlite"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @contextmanager
    def transaction(
        self, scope: MemoryScope, *, write: bool, deadline: float,
    ) -> Iterator[SqlGraphTransaction]:
        if not valid_scope(scope):
            raise PermissionError("invalid graph scope")
        root = partition_root(self.root, scope, create=write, version="v1")
        for name in ("graph.sqlite3", "graph.sqlite3-wal", "graph.sqlite3-shm", "graph.sqlite3-journal"):
            checked_path(root, name)
        with open_partition_directory(root) as fd:
            root_uri = Path(f"/proc/self/fd/{fd}")
            if not root_uri.is_dir():
                raise OSError("graph directory handles unavailable")
            path = root_uri / "graph.sqlite3"
            mode = "rwc" if write else "ro"
            with closing(sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True, isolation_level=None)) as connection:
                connection.execute(f"PRAGMA busy_timeout = {max(1, min(250, int(_remaining(deadline) * 1000)))}")
                if write and connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
                    raise OSError("graph WAL unavailable")
                with _sqlite_transaction(connection, scope, write=write, deadline=deadline, initialize=write) as transaction:
                    yield transaction


@dataclass
class _VolatilePartition:
    lock: RLock = field(default_factory=RLock)
    connection: sqlite3.Connection | None = None


class InMemoryGraphStore:
    """Each tenant owns a separate volatile transaction space with its own quota."""

    backend = "memory"

    def __init__(self, *, max_receipts_per_tenant: int = 20_000) -> None:
        if not _bounded_integer(max_receipts_per_tenant, 1_000_000):
            raise ValueError("invalid volatile graph quota")
        self._quota = max_receipts_per_tenant
        self._partitions: dict[str, _VolatilePartition] = {}
        self._lock = Lock()
        self._closed = False

    @contextmanager
    def transaction(
        self, scope: MemoryScope, *, write: bool, deadline: float,
    ) -> Iterator[SqlGraphTransaction]:
        if not valid_scope(scope) or scope.owner_epoch is not None:
            raise PermissionError("volatile graph requires an inline scope")
        with self._lock:
            if self._closed:
                raise OSError("volatile graph closed")
            partition = self._partitions.get(scope.tenant_id)
            if partition is None:
                partition = _VolatilePartition()
                if write:
                    self._partitions[scope.tenant_id] = partition
        if not partition.lock.acquire(timeout=_remaining(deadline)):
            raise TimeoutError
        try:
            if partition.connection is None and not write:
                with closing(sqlite3.connect(":memory:", isolation_level=None)) as conn:
                    with _sqlite_transaction(conn, scope, write=False, deadline=deadline, initialize=True) as txn:
                        yield txn
                return
            if partition.connection is None:
                conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
                try:
                    for statement in GRAPH_SCHEMA_SQL:
                        conn.execute(statement)
                except BaseException:
                    conn.close()
                    raise
                partition.connection = conn
            with _sqlite_transaction(partition.connection, scope, write=write, deadline=deadline) as txn:
                yield txn
                if write and txn.count_receipts(self._quota + 1) > self._quota:
                    raise GraphIntegrityError("volatile tenant graph quota reached")
        finally:
            partition.lock.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            partitions = tuple(self._partitions.values())
            self._partitions.clear()
        for partition in partitions:
            with partition.lock:
                if partition.connection is not None:
                    partition.connection.close()
                    partition.connection = None


def _valid_ref(reference: EvidenceRef, scope: MemoryScope) -> bool:
    return (
        reference.request_id == scope.request_id
        and valid_internal_id(reference.source_id)
        and valid_internal_id(reference.artifact_version)
        and _HASH.fullmatch(reference.content_sha256) is not None
        and 0 < len(reference.chunk_ids) <= 8
        and len(set(reference.chunk_ids)) == len(reference.chunk_ids)
        and all(valid_internal_id(chunk) for chunk in reference.chunk_ids)
    )


def _valid_graph_id(value: str, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix + "_") and bool(_HASH.fullmatch(value[len(prefix) + 1:]))


def _valid_snapshot(snapshot: TerminalGraphSnapshot | None, scope: MemoryScope, origin: str) -> bool:
    return (
        isinstance(snapshot, TerminalGraphSnapshot)
        and snapshot.tenant_id == scope.tenant_id
        and snapshot.request_id == origin
        and snapshot.policy_version == scope.policy_version
        and snapshot.schema_version == SCHEMA_VERSION
        and snapshot.normalization_version == NORMALIZATION_VERSION
        and snapshot.status in TERMINAL_STATUSES
        and snapshot.batch.status in {"pending", "applied", "volatile_applied", "retry_deferred"}
        and _bounded_integer(snapshot.batch.source_epoch)
        and len(snapshot.active_receipt_ids) <= MAX_RECEIPTS
        and len(snapshot.historical_receipt_ids) <= MAX_HISTORICAL_RECEIPTS
        and all(_valid_graph_id(item, "receipt") for item in (*snapshot.active_receipt_ids, *snapshot.historical_receipt_ids))
        and not (set(snapshot.active_receipt_ids) & set(snapshot.historical_receipt_ids))
    )


def _same_snapshot(left: TerminalGraphSnapshot | None, right: TerminalGraphSnapshot | None) -> bool:
    if left is None or right is None:
        return left is right
    return replace(left, batch=replace(left.batch, status=right.batch.status)) == right


def _check_read_query(query: GraphQuery) -> GraphQuery:
    if type(query.max_hops) is not int or query.max_hops not in (1, 2) or not _bounded_integer(query.limit):
        raise ValueError("invalid graph query limits")
    if query.as_of is not None:
        _utc(query.as_of)
    if query.graph_revision is not None and (
        type(query.graph_revision) is not int or not 0 <= query.graph_revision < 2**63
    ):
        raise ValueError("invalid graph revision")
    if not _business_date(query.valid_at) or (query.period_range is not None and query.valid_at is not None):
        raise ValueError("invalid graph query time")
    if query.period_range is not None:
        period = ValidPeriod(*query.period_range)
        if period.mode != "finite" or period.label or period.calendar_id or period.grain:
            raise ValueError("invalid date query range")
    seeds = tuple(item for item in query.seed_entities[:8] if _name(item))
    if any(seed.startswith("entity_") and not _valid_graph_id(seed, "entity") for seed in seeds):
        raise ValueError("invalid graph entity identifier")
    predicates = tuple(dict.fromkeys(query.predicates[:8]))
    if any(item not in RELATIONSHIPS | {"has_observation"} for item in predicates):
        raise ValueError("invalid graph predicate")
    return replace(
        query, seed_entities=seeds, metrics=tuple(dict.fromkeys(_metric(metric) for metric in query.metrics[:8])),
        predicates=predicates, limit=min(query.limit, 20),
    )


@dataclass(frozen=True)
class _VisibleReceipt:
    stored: _StoredReceipt

    @property
    def fact(self) -> TypedFact:
        return self.stored.fact

    @property
    def support(self) -> tuple[EvidenceRef, ...]:
        return self.stored.receipt.support

    @property
    def confidence(self) -> int:
        return self.stored.receipt.confidence_ppm


@dataclass
class _ReadBudget:
    max_candidates: int
    candidates: int = 0
    truncated: bool = False
    exhausted_sources: bool = False
    sources: set[tuple[str, str, str]] = field(default_factory=set)

    def rows(self, maximum: int) -> int:
        count = min(maximum, self.max_candidates - self.candidates)
        if not count:
            self.truncated = True
        return count

    def consume(self, count: int, limit: int) -> None:
        self.candidates += count
        self.truncated |= count == limit

    def hydrate(self, reference: EvidenceRef) -> bool:
        key = (reference.request_id, reference.source_id, reference.artifact_version)
        if key not in self.sources and len(self.sources) >= MAX_HYDRATIONS:
            self.exhausted_sources = self.truncated = True
            return False
        self.sources.add(key)
        return True


class KnowledgeGraphService:
    """Consume committed publication batches and revalidate every returned support group."""

    def __init__(
        self, store: GraphStore, authority: GraphAuthority, *,
        min_entailment: float = 0.85, nli_entailment_threshold: float = 0.85,
        max_candidates: int = MAX_CANDIDATES, max_projection_tokens: int = 512,
        query_timeout_ms: int = 100, verification_policy_version: int = 1,
        tenants: tuple[str, ...] | None = None, metrics: Any = None,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        threshold = max(Decimal(str(min_entailment)), Decimal(str(nli_entailment_threshold)))
        if not Decimal("0.85") <= threshold <= 1:
            raise ValueError("invalid graph entailment threshold")
        if not _bounded_integer(max_candidates, MAX_CANDIDATES) or not _bounded_integer(max_projection_tokens, 512):
            raise ValueError("invalid graph limits")
        if not _bounded_integer(query_timeout_ms, 250) or not _bounded_integer(verification_policy_version):
            raise ValueError("invalid graph policy")
        self.store = store
        self.authority = authority
        self.threshold = threshold
        self.max_candidates = max_candidates
        self.max_projection_tokens = max_projection_tokens
        self.query_timeout_ms = query_timeout_ms
        self.verification_policy_version = verification_policy_version
        self.tenants = frozenset(tenants) if tenants is not None else None
        self.metrics = metrics
        self.utc_now = utc_now

    def _allowed(self, scope: MemoryScope, purpose: str, deadline: float) -> bool:
        return (
            valid_scope(scope)
            and (self.tenants is None or scope.tenant_id in self.tenants)
            and _remaining(deadline) > 0
            and self.authority.scope_allowed(scope, purpose=purpose, deadline=deadline) is True
        )

    def _validate_evidence(
        self, scope: MemoryScope, reference: EvidenceRef, purpose: str, deadline: float,
    ) -> tuple[str, tuple[dict[str, str], ...]]:
        _remaining(deadline)
        evidence = self.authority.load_evidence(scope, reference, purpose=purpose, deadline=deadline)
        _remaining(deadline)
        if not isinstance(evidence, GraphEvidence) or (
            evidence.tenant_id, evidence.request_id, evidence.source_id, evidence.artifact_version
        ) != (scope.tenant_id, reference.request_id, reference.source_id, reference.artifact_version):
            return "ineligible", ()
        if (
            evidence.origin != "web" or evidence.level not in {"L1", "L2"} or evidence.isolated
            or evidence.trusted is not True or evidence.visible is not True or evidence.facts_eligible is not True
            or evidence.sanitizer_version != WEB_SANITIZER_VERSION
        ):
            return "ineligible", ()
        if len(evidence.fragments) > 256:
            return "ineligible", ()
        by_chunk = {item.chunk_id: item for item in evidence.fragments}
        if len(by_chunk) != len(evidence.fragments) or set(reference.chunk_ids).difference(by_chunk):
            return "ineligible", ()
        fragments = tuple(by_chunk[chunk] for chunk in reference.chunk_ids)
        try:
            for item in fragments:
                if not _clean(item.anchor) or not _clean(item.text, limit=50_000, multiline=True):
                    return "sanitizer", ()
        except ValueError:
            return "sanitizer", ()
        if reference.content_sha256 != evidence_digest(fragments):
            return "ineligible", ()
        return "", tuple(
            {
                "source_id": reference.source_id,
                "artifact_version": reference.artifact_version,
                "chunk_id": fragment.chunk_id,
                "anchor": fragment.anchor,
                "content_sha256": reference.content_sha256,
                "acl_reference": evidence.acl_reference,
            }
            for fragment in fragments
        )

    def issue_receipt(
        self, scope: MemoryScope, candidate: VerifiedCandidate, *, deadline: float,
    ) -> ReceiptDecision:
        """Evaluate a trusted Review/Repair and publication mapping without writing the graph."""
        if not self._allowed(scope, "maintenance", deadline):
            return ReceiptDecision(None, "missing_scope")
        if candidate.unresolved_conflict:
            return ReceiptDecision(None, "conflict")
        if candidate.deterministically_checked is not True:
            return ReceiptDecision(None, "unsupported")
        try:
            fingerprint = fact_fingerprint(candidate.ledger_version, candidate.statement, candidate.fact)
        except ValueError:
            return ReceiptDecision(None, "sanitizer")
        binding = (candidate.ledger_version, candidate.claim_id, fingerprint)
        atoms = candidate.atoms
        required_ids = candidate.required_atomic_claim_ids
        if (
            not valid_internal_id(candidate.claim_id)
            or (candidate.conflict_id and not valid_internal_id(candidate.conflict_id))
            or not 0 < len(required_ids) <= 8
            or tuple(sorted(required_ids)) != tuple(sorted(atom.atom_id for atom in atoms))
            or len(set(required_ids)) != len(required_ids)
            or any(not valid_internal_id(atom_id) for atom_id in required_ids)
            or any((atom.ledger_version, atom.claim_id, atom.fact_fingerprint) != binding for atom in atoms)
        ):
            return ReceiptDecision(None, "unbound")
        publication = candidate.publication
        if (
            publication is None or publication.allowed is not True
            or (publication.ledger_version, publication.claim_id, publication.fact_fingerprint) != binding
        ):
            return ReceiptDecision(None, "report_blocked")

        confidence = Decimal(1)
        verifier_versions = set()
        verified_times = set()
        refs = set()
        for atom in atoms:
            if (
                atom.nli_label != "ENTAILED" or atom.is_fallback or not valid_internal_id(atom.verifier_version)
                or not _bounded_integer(atom.decomposition_version) or not atom.support
                or any(not _valid_ref(ref, scope) for ref in atom.support)
            ):
                return ReceiptDecision(None, "unsupported")
            try:
                score = Decimal(str(atom.confidence))
                if not score.is_finite() or not self.threshold <= score <= 1:
                    return ReceiptDecision(None, "confidence")
                if atom.verified_at:
                    verified_times.add(_utc(datetime.fromisoformat(atom.verified_at)))
            except (InvalidOperation, ValueError):
                return ReceiptDecision(None, "confidence")
            confidence = min(confidence, score)
            verifier_versions.add((atom.verifier_version, atom.decomposition_version))
            refs.update(atom.support)

        if len(verifier_versions) != 1:
            return ReceiptDecision(None, "unbound")
        refs = tuple(sorted(refs, key=lambda ref: (ref.source_id, ref.artifact_version, ref.chunk_ids, ref.content_sha256)))
        distinct_chunks = {(ref.source_id, ref.artifact_version, chunk) for ref in refs for chunk in ref.chunk_ids}
        if set(publication.support) != set(refs):
            return ReceiptDecision(None, "report_blocked")
        if not refs or len(refs) > 8 or sum(len(ref.chunk_ids) for ref in refs) != len(distinct_chunks) or len(distinct_chunks) > 8:
            return ReceiptDecision(None, "unsupported")
        if self.authority.authorize_origin(scope, scope.request_id, purpose="maintenance", deadline=deadline) is not True:
            return ReceiptDecision(None, "ineligible")

        provenance = []
        for reference in refs:
            rejection, fragments = self._validate_evidence(scope, reference, "maintenance", deadline)
            if rejection:
                return ReceiptDecision(None, rejection)
            provenance.extend(fragments)
        verifier_version, decomposition_version = verifier_versions.pop()
        version = f"{verifier_version}/atom-{decomposition_version}/policy-{self.verification_policy_version}"
        verified_at = max(verified_times, default="")
        if not verified_at:
            return ReceiptDecision(None, "unsupported")
        receipt = VerifiedFactReceipt(
            scope.tenant_id, scope.request_id, candidate.ledger_version, candidate.claim_id, fingerprint,
            tuple(sorted(required_ids)), "ENTAILED", int(confidence * 1_000_000), version, verified_at, refs,
        )
        return ReceiptDecision(receipt, provenance=tuple(provenance))

    def _validate_batch(self, scope: MemoryScope, snapshot: TerminalGraphSnapshot, batch_id: str) -> TerminalBatch:
        batch = snapshot.batch
        if snapshot.owner_epoch != scope.owner_epoch or batch.batch_id != batch_id:
            raise GraphIntegrityError("graph batch scope mismatch")
        key = _batch_content(
            scope, batch.candidates, batch.retirements, batch.source_epoch, batch.ledger_version, batch.report_version,
        )
        if key != (batch.batch_id, batch.digest, batch.publication_digest):
            raise GraphIntegrityError("graph batch digest mismatch")
        if any(candidate.ledger_version != batch.ledger_version for candidate in batch.candidates):
            raise GraphIntegrityError("graph batch ledger mismatch")
        if len({candidate.claim_id for candidate in batch.candidates}) != len(batch.candidates):
            raise GraphIntegrityError("ambiguous graph batch claim mapping")
        if len({r.receipt_id for r in batch.retirements}) != len(batch.retirements):
            raise GraphIntegrityError("ambiguous retirement")
        if any(
            not valid_internal_id(r.request_id) or not _valid_graph_id(r.receipt_id, "receipt")
            or r.reason not in {"superseded", "revoked"} for r in batch.retirements
        ):
            raise GraphIntegrityError("invalid graph retirement")
        return batch

    def apply_terminal_batch(
        self, scope: MemoryScope, batch_id: str, *, deadline: float,
    ) -> GraphApplyResult:
        """Read certified terminal state and atomically merge only fully verified receipts."""
        deadline = min(deadline, time.monotonic() + 0.25)
        try:
            result = self._apply_batch(scope, batch_id, deadline)
        except TimeoutError:
            result = GraphApplyResult("timeout")
        except Exception:
            result = GraphApplyResult("error")
        if valid_scope(scope) and self.metrics is not None:
            for reason, amount in result.rejected:
                self.metrics.record_v4_graph_receipt("rejected", reason=reason, amount=amount, tenant_id=scope.tenant_id)
            if result.outcome == "duplicate":
                self.metrics.record_v4_graph_apply("duplicate", tenant_id=scope.tenant_id)
            else:
                for outcome, amount in (("inserted", result.inserted), ("revoked", result.revoked), ("superseded", result.superseded)):
                    if amount:
                        self.metrics.record_v4_graph_apply(outcome, amount=amount, tenant_id=scope.tenant_id)
                if result.outcome in {"stale", "error", "timeout"}:
                    self.metrics.record_v4_graph_apply(
                        "stale" if result.outcome == "stale" else "error", tenant_id=scope.tenant_id,
                    )
        return result

    def _apply_batch(self, scope: MemoryScope, batch_id: str, deadline: float) -> GraphApplyResult:
        if not _valid_graph_id(batch_id, "batch") or not self._allowed(scope, "maintenance", deadline):
            return GraphApplyResult("missing_scope")
        snapshot = self.authority.load_terminal(scope, scope.request_id, deadline=deadline)
        if not _valid_snapshot(snapshot, scope, scope.request_id):
            return GraphApplyResult("stale")
        assert snapshot is not None
        if snapshot.batch.batch_id != batch_id:
            return GraphApplyResult("stale")
        batch = self._validate_batch(scope, snapshot, batch_id)
        with self.store.transaction(scope, write=True, deadline=deadline) as txn:
            txn.watermark(self.utc_now())
            current = self.authority.load_terminal(scope, scope.request_id, deadline=deadline)
            if not _same_snapshot(current, snapshot) or not self._allowed(scope, "maintenance", deadline):
                return GraphApplyResult("stale")
            current_epoch = txn.get_source_epoch(scope.request_id)
            if current_epoch is not None and current_epoch > batch.source_epoch:
                return GraphApplyResult("stale")
            existing = txn.get_batch(scope.request_id, batch_id)
            if existing is not None:
                if existing[0] != batch.digest:
                    raise GraphIntegrityError("applied batch has conflicting digest")
                return replace(existing[1], outcome="duplicate", inserted=0, revoked=0, superseded=0, rejected=())
            if current_epoch == batch.source_epoch:
                return GraphApplyResult("stale")

            accepted = []
            rejected: dict[str, int] = defaultdict(int)
            for candidate in batch.candidates:
                decision = self.issue_receipt(scope, candidate, deadline=deadline)
                if decision.receipt is None:
                    rejected[decision.reason] += 1
                    continue
                receipt_id = receipt_id_for(scope, candidate.ledger_version, candidate.claim_id, decision.receipt.fact_fingerprint)
                if receipt_id not in snapshot.active_receipt_ids:
                    rejected["report_blocked"] += 1
                    continue
                accepted.append((receipt_id, decision, candidate))

            retirements = {item.receipt_id: item for item in batch.retirements}
            for item in batch.retirements:
                if item.request_id == scope.request_id:
                    if item.receipt_id in snapshot.active_receipt_ids:
                        raise GraphIntegrityError("terminal snapshot authorizes a retired receipt")
                    continue
                origin = self.authority.load_terminal(scope, item.request_id, deadline=deadline)
                if not _valid_snapshot(origin, scope, item.request_id):
                    raise GraphIntegrityError("cross-request retirement lacks authorization")
                assert origin is not None
                if (
                    item.receipt_id in origin.active_receipt_ids
                    or (item.reason == "superseded") != (item.receipt_id in origin.historical_receipt_ids)
                    or self.authority.authorize_origin(scope, item.request_id, purpose="maintenance", deadline=deadline) is not True
                ):
                    raise GraphIntegrityError("cross-request retirement lacks authorization")
            allowed_ids = set(snapshot.active_receipt_ids)
            historical_ids = set(snapshot.historical_receipt_ids)
            for receipt_id in txn.active_receipts(scope.request_id):
                if receipt_id not in allowed_ids:
                    retirements.setdefault(receipt_id, Retirement(
                        scope.request_id, receipt_id, "superseded" if receipt_id in historical_ids else "revoked",
                    ))

            watermark = txn.allocate_revision(self.utc_now())
            retired: dict[str, int] = defaultdict(int)
            for item in sorted(retirements.values(), key=lambda value: value.receipt_id):
                if txn.retire(item, watermark):
                    retired[item.reason] += 1
            inserted = sum(
                txn.store_receipt(receipt_id, decision, candidate, watermark, batch.source_epoch, self.verification_policy_version)
                for receipt_id, decision, candidate in accepted
            )
            if not _same_snapshot(self.authority.load_terminal(scope, scope.request_id, deadline=deadline), snapshot):
                raise GraphIntegrityError("terminal snapshot changed before commit")
            _remaining(deadline)
            result = GraphApplyResult(
                "volatile_applied" if self.store.backend == "memory" else "applied",
                inserted, retired["revoked"], retired["superseded"], watermark,
                tuple(item[0] for item in accepted), tuple(sorted(rejected.items())),
            )
            txn.record_batch(scope.request_id, batch, result)
            return result

    def purge_origin(
        self, scope: MemoryScope, origin_request_id: str, *, deadline: float, dry_run: bool = True, limit: int = 64,
    ) -> int:
        """Physically remove a bounded set of receipts after authoritative deletion."""
        check = getattr(self.authority, "origin_deleted", None)
        if (
            check is None or not valid_internal_id(origin_request_id) or not _bounded_integer(limit, 256)
            or not self._allowed(scope, "maintenance", deadline)
            or check(scope, origin_request_id, deadline=deadline) is not True
        ):
            return 0
        with self.store.transaction(scope, write=not dry_run, deadline=deadline) as txn:
            if check(scope, origin_request_id, deadline=deadline) is not True:
                return 0
            return txn.purge_origin(origin_request_id, limit=limit, dry_run=dry_run)

    def query(self, scope: MemoryScope, query: GraphQuery, *, purpose: GraphPurpose, deadline: float) -> GraphResult:
        started = time.monotonic()
        deadline = min(deadline, started + self.query_timeout_ms / 1000)
        try:
            checked = _check_read_query(query)
        except (TypeError, ValueError):
            return GraphResult("invalid_query")
        result = self._query(scope, checked, purpose=purpose, deadline=deadline, result_limit=checked.limit)
        self._record_query(scope, purpose, started, result.status)
        return result

    def _record_query(
        self, scope: MemoryScope, purpose: str, started: float, status: str, *, projected_tokens: int = 0,
    ) -> None:
        if self.metrics is not None and valid_scope(scope) and purpose in PURPOSES:
            outcome = status if status in {"hit", "ambiguous", "truncated", "timeout", "unavailable"} else "empty"
            self.metrics.record_v4_graph_query(
                "planning" if purpose == "plan" else purpose, self.store.backend, outcome,
                (time.monotonic() - started) * 1000, tenant_id=scope.tenant_id, projected_tokens=projected_tokens,
            )

    def project(
        self, scope: MemoryScope, query: GraphQuery, *, purpose: GraphPurpose, deadline: float,
        token_limit: int | None = None,
    ) -> GraphProjection:
        """Encode atomic, citation-free records using a conservative UTF-8 size bound."""
        started = time.monotonic()
        deadline = min(deadline, started + self.query_timeout_ms / 1000)
        if purpose not in PURPOSES or (token_limit is not None and not _bounded_integer(token_limit)):
            return GraphProjection("graph", "invalid_query")
        phase = "planning" if purpose == "plan" else purpose
        try:
            checked = _check_read_query(query)
        except (TypeError, ValueError):
            return GraphProjection(phase, "invalid_query")
        result = self._query(scope, checked, purpose=purpose, deadline=deadline, result_limit=checked.limit)
        limit = min(self.max_projection_tokens, token_limit or self.max_projection_tokens, 512 if purpose == "report" else 256)
        projection = GraphProjection(phase, result.status)
        try:
            for match in result.matches:
                _remaining(deadline)
                record = _public_match(match)
                proposed = replace(projection, records=(*projection.records, record))
                if len(proposed.encode().encode("utf-8")) <= limit:
                    projection = proposed
                    continue
                projection = replace(projection, status="truncated")
                if match.stability == "disputed":
                    hint = replace(projection, records=(*projection.records, {"stability": "disputed"}))
                    if len(hint.encode().encode("utf-8")) <= limit:
                        projection = hint
                break
            while projection.records and len(projection.encode().encode("utf-8")) > limit:
                projection = replace(projection, records=projection.records[:-1])
            _remaining(deadline)
        except TimeoutError:
            projection = GraphProjection(phase, "timeout")
        self._record_query(
            scope, purpose, started, projection.status,
            projected_tokens=len(projection.encode().encode("utf-8")) if projection.records else 0,
        )
        return projection

    def _query(
        self, scope: MemoryScope, query: GraphQuery, *, purpose: GraphPurpose, deadline: float,
        result_limit: int, observations_only: bool = False,
    ) -> GraphResult:
        if purpose not in PURPOSES:
            return GraphResult("invalid_query")
        try:
            if not self._allowed(scope, purpose, deadline):
                return GraphResult("empty")
            now = self.utc_now()
            with self.store.transaction(scope, write=False, deadline=deadline) as txn:
                committed = txn.watermark(now)
                watermark = GraphWatermark(
                    query.as_of or max(now, committed.as_of),
                    min(committed.graph_revision, query.graph_revision) if query.graph_revision is not None else committed.graph_revision,
                )
                view = _GraphReadView(self, txn, scope, query, purpose, watermark, deadline)
                matches, ambiguities = view.traverse(observations_only=observations_only)
                matches = tuple(item.match for item in matches if view.recheck(item.receipts))
                ambiguities = tuple(
                    tuple(option[0] for option in group if view.recheck(option[1]))
                    for group in ambiguities
                )
                ambiguities = tuple(group for group in ambiguities if len(group) > 1)
                if not self._allowed(scope, purpose, deadline):
                    return GraphResult("empty")
                status = "hit" if matches else "empty"
                if ambiguities:
                    status = "ambiguous"
                if view.budget.truncated or len(matches) > result_limit:
                    status = "truncated"
                return GraphResult(
                    status, matches[:result_limit], watermark, ambiguities, view.budget.candidates,
                    len(view.budget.sources),
                )
        except TimeoutError:
            return GraphResult("timeout")
        except FileNotFoundError:
            return GraphResult("empty")
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_INTERRUPT}:
                return GraphResult("timeout")
            return GraphResult("unavailable")
        except Exception:
            return GraphResult("unavailable")

    def _receipt_visible(
        self, scope: MemoryScope, item: _StoredReceipt, purpose: str, deadline: float,
        budget: _ReadBudget, snapshots: dict[str, TerminalGraphSnapshot | None],
    ) -> bool:
        _remaining(deadline)
        receipt = item.receipt
        if (
            receipt.tenant_id != scope.tenant_id or receipt.nli_label != "ENTAILED"
            or receipt.confidence_ppm < self.threshold * 1_000_000
            or item.retirement == "revoked"
        ):
            return False
        origin = receipt.request_id
        if origin not in snapshots:
            snapshots[origin] = self.authority.load_terminal(scope, origin, deadline=deadline)
        snapshot = snapshots[origin]
        if not _valid_snapshot(snapshot, scope, origin):
            return False
        assert snapshot is not None
        if snapshot.batch.source_epoch < item.source_epoch:
            return False
        allowed_ids = snapshot.historical_receipt_ids if item.retirement == "superseded" else snapshot.active_receipt_ids
        if item.receipt_id not in allowed_ids:
            return False
        if self.authority.authorize_origin(scope, origin, purpose=purpose, deadline=deadline) is not True:
            return False
        for reference in receipt.support:
            if not _valid_ref(reference, replace(scope, request_id=origin)) or not budget.hydrate(reference):
                return False
            reason, _ = self._validate_evidence(scope, reference, purpose, deadline)
            if reason:
                return False
        return bool(receipt.support)

    def evolution(
        self, scope: MemoryScope, spec: EvolutionQuery, *, purpose: GraphPurpose, deadline: float,
    ) -> EvolutionResult:
        started = time.monotonic()
        deadline = min(deadline, started + self.query_timeout_ms / 1000)
        try:
            if not _bounded_integer(spec.limit) or any(
                not _valid_graph_id(entity_id, "entity") for entity_id in spec.entity_ids[:4]
            ):
                raise ValueError("invalid evolution query")
            if spec.scope_key is not None and not _HASH.fullmatch(spec.scope_key):
                raise ValueError("invalid comparison scope")
            query = _check_read_query(GraphQuery(
                spec.entity_ids[:4], metrics=(spec.metric_key,), predicates=("has_observation",),
                period_range=spec.period_range, as_of=spec.as_of, max_hops=1, graph_revision=spec.graph_revision,
            ))
        except (TypeError, ValueError):
            return EvolutionResult("invalid_query")
        result = self._query(scope, query, purpose=purpose, deadline=deadline, result_limit=100, observations_only=True)
        try:
            _remaining(deadline)
            evolution = _evolution_result(result, spec)
            _remaining(deadline)
        except TimeoutError:
            evolution = EvolutionResult("timeout")
        self._record_query(scope, purpose, started, evolution.status)
        return evolution


@dataclass(frozen=True)
class _GraphMatchCandidate:
    match: GraphMatch
    receipts: tuple[_StoredReceipt, ...]


class _GraphReadView:
    def __init__(
        self, service: KnowledgeGraphService, transaction: SqlGraphTransaction, scope: MemoryScope,
        query: GraphQuery, purpose: GraphPurpose, watermark: GraphWatermark, deadline: float,
    ) -> None:
        self.service = service
        self.txn = transaction
        self.scope = scope
        self.query = query
        self.purpose = purpose
        self.watermark = watermark
        self.deadline = deadline
        self.budget = _ReadBudget(service.max_candidates)
        self._snapshots: dict[str, TerminalGraphSnapshot | None] = {}
        self._recheck_snapshots: dict[str, TerminalGraphSnapshot | None] = {}
        self._visible: dict[str, _VisibleReceipt | None] = {}
        self._rechecked: dict[str, bool] = {}
        self._neighbors: dict[str, tuple[tuple[str, str | None, _VisibleReceipt], ...]] = {}
        self._buckets: dict[str, tuple[tuple[_VisibleReceipt, ...], bool]] = {}

    def _visible_receipt(self, item: _StoredReceipt) -> _VisibleReceipt | None:
        if item.receipt_id not in self._visible:
            visible = self.service._receipt_visible(
                self.scope, item, self.purpose, self.deadline, self.budget, self._snapshots,
            )
            self._visible[item.receipt_id] = _VisibleReceipt(item) if visible else None
        return self._visible[item.receipt_id]

    def recheck(self, items: tuple[_StoredReceipt, ...]) -> bool:
        for item in items:
            if item.receipt_id not in self._rechecked:
                origin = item.receipt.request_id
                self._rechecked[item.receipt_id] = self.service._receipt_visible(
                    self.scope, item, self.purpose, self.deadline, self.budget, self._recheck_snapshots,
                ) and _same_snapshot(self._recheck_snapshots[origin], self._snapshots[origin])
            if not self._rechecked[item.receipt_id]:
                return False
        return True

    def resolve(self, seed: str) -> tuple[tuple[GraphEntity, tuple[_StoredReceipt, ...]], ...]:
        by_id = _valid_graph_id(seed, "entity")
        cap = self.budget.rows(MAX_DIRECTION_ROWS)
        if not cap:
            return ()
        rows = self.txn.seed_rows(seed if by_id else _name(seed), by_id, self.query, self.watermark, cap)
        self.budget.consume(len(rows), cap)
        matches: dict[str, tuple[GraphEntity, tuple[_StoredReceipt, ...]]] = {}
        for entity_id, row in rows:
            if entity_id in matches or self._visible_receipt(row) is None:
                continue
            fact = row.fact
            entity = fact.object if isinstance(fact, RelationshipFact) and fact.object.entity_id == entity_id else fact.subject
            matches[entity_id] = (entity, (row,))
        if len(rows) == cap or self.budget.exhausted_sources:
            return ()
        return tuple(sorted(matches.values(), key=lambda item: (item[0].entity_type, item[0].canonical_key, item[0].entity_id)))

    def neighbors(self, entity_id: str) -> tuple[tuple[str, str | None, _VisibleReceipt], ...]:
        if entity_id not in self._neighbors:
            neighbors: dict[str, tuple[str, str | None, _VisibleReceipt]] = {}
            for reverse in (False, True):
                cap = self.budget.rows(MAX_DIRECTION_ROWS)
                if not cap:
                    break
                rows = self.txn.adjacency(entity_id, reverse, self.query, self.watermark, cap)
                self.budget.consume(len(rows), cap)
                for edge_id, bucket, row in rows:
                    visible = self._visible_receipt(row)
                    if visible is not None and (
                        edge_id not in neighbors
                        or _support_rank(visible) < _support_rank(neighbors[edge_id][2])
                    ):
                        neighbors[edge_id] = (edge_id, bucket, visible)
            self._neighbors[entity_id] = tuple(neighbors.values())
        return self._neighbors[entity_id]

    def _bucket(
        self, bucket: str, reference: _VisibleReceipt,
    ) -> tuple[tuple[_VisibleReceipt, ...], bool]:
        if bucket not in self._buckets:
            cap = self.budget.rows(self.budget.max_candidates)
            if not cap:
                return (), False
            rows = self.txn.conflict_rows(bucket, self.query, self.watermark, cap)
            self.budget.consume(len(rows), cap)
            visible = tuple(item for row in rows if (item := self._visible_receipt(row)) is not None)
            self._buckets[bucket] = visible, len(rows) < cap and not self.budget.exhausted_sources
        candidates, complete = self._buckets[bucket]
        if not complete:
            return (), False
        by_target = {reference.fact.target_id: reference}
        for candidate in candidates:
            fact = candidate.fact
            if isinstance(fact, RelationshipFact) and (
                reference.fact.period.intersects(fact.period) is None
                or isinstance(reference.fact, RelationshipFact) and reference.fact.object.entity_id == fact.object.entity_id
            ):
                continue
            current = by_target.get(fact.target_id)
            if current is None or _support_rank(candidate) < _support_rank(current):
                by_target[fact.target_id] = candidate
        return tuple(by_target.values()), True

    def traverse(
        self, *, observations_only: bool,
    ) -> tuple[list[_GraphMatchCandidate], list[tuple[tuple[GraphEntity, tuple[_StoredReceipt, ...]], ...]]]:
        matches: list[tuple[int, _GraphMatchCandidate]] = []
        ambiguities = []
        for seed_index, raw_seed in enumerate(self.query.seed_entities):
            resolved = self.resolve(raw_seed)
            if len(resolved) > 1:
                ambiguities.append(resolved)
                continue
            if not resolved:
                continue
            seed, seed_receipts = resolved[0]
            start = _Frontier(seed, (seed.entity_id,))
            frontier = [start]
            for _ in range(self.query.max_hops):
                next_hop = []
                for node in frontier:
                    for edge_id, bucket, visible in self.neighbors(node.entity.entity_id):
                        _remaining(self.deadline)
                        fact = visible.fact
                        if isinstance(fact, ObservationFact) and self.query.metrics and fact.metric_key not in self.query.metrics:
                            continue
                        common = _common_period(node.supported, fact.period)
                        if common is False:
                            continue
                        if common is None and (self.query.valid_at is not None or self.query.period_range is not None):
                            continue
                        other = None
                        if isinstance(fact, RelationshipFact):
                            other = fact.object if node.entity.entity_id == fact.subject.entity_id else fact.subject
                            if other.entity_id in node.visited:
                                continue
                        group, complete = self._bucket(bucket, visible) if bucket else ((visible,), True)
                        if not complete:
                            continue
                        disputed = len(group) > 1
                        path = tuple(item.fact.target_id for item in node.supported) + (edge_id,)
                        if not observations_only or isinstance(fact, ObservationFact):
                            match = GraphMatch(
                                seed,
                                tuple(item.fact for item in group),
                                path, common,
                                tuple(item.support for item in (*node.supported, *group)),
                                "disputed" if disputed else "unknown_period" if common is None else "verified",
                                min(item.confidence for item in (*node.supported, *group)),
                            )
                            receipts = tuple(dict.fromkeys(item.stored.receipt_id for item in (*node.supported, *group)))
                            by_id = {item.stored.receipt_id: item.stored for item in (*node.supported, *group)}
                            matched = _GraphMatchCandidate(match, seed_receipts + tuple(by_id[key] for key in receipts))
                            matches.append((seed_index, matched))
                        if other is not None and not disputed:
                            next_hop.append(_Frontier(other, (*node.visited, other.entity_id), (*node.supported, visible)))
                frontier = next_hop
        matches.sort(key=lambda item: (item[0], *_match_rank(item[1].match)))
        unique = {}
        for _, candidate in matches:
            match = candidate.match
            group = tuple(sorted(fact.target_id for fact in match.facts))
            key = (match.seed.entity_id, match.path[:-1], group)
            unique.setdefault(key, candidate)
        return list(unique.values()), ambiguities


@dataclass(frozen=True)
class _Frontier:
    entity: GraphEntity
    visited: tuple[str, ...]
    supported: tuple[_VisibleReceipt, ...] = ()


def _support_rank(value: _VisibleReceipt) -> tuple[int, str]:
    return -value.confidence, value.stored.receipt_id


def _match_rank(match: GraphMatch) -> tuple[Any, ...]:
    start = match.common_period.valid_from if match.common_period else None
    return (
        len(match.path), -(start.toordinal() if start else 0), -match.confidence_ppm,
        match.seed.entity_id, match.path,
    )


def _public_match(match: GraphMatch) -> dict[str, Any]:
    records = []
    for fact in match.facts:
        item = {"entity": fact.subject.name}
        start, end = fact.period.bounds
        if start is not None:
            item["from"] = start
        if end is not None:
            item["to"] = end
        if fact.period.label:
            item["period"] = fact.period.label
        if isinstance(fact, ObservationFact):
            item.update(metric=fact.metric_key, value=fact.normalized[0], unit=fact.normalized[2])
        elif isinstance(fact, RelationshipFact):
            item.update(relation=fact.predicate, target=fact.object.name)
        if isinstance(fact, (ObservationFact, RelationshipFact)) and fact.scope:
            item["scope"] = dict(fact.scope)
        records.append(item)
    return {"facts": records, "stability": match.stability}


def _common_period(rows: tuple[_VisibleReceipt, ...], period: ValidPeriod) -> ValidPeriod | None | Literal[False]:
    periods = tuple(item.fact.period for item in rows) + (period,)
    known = [item for item in periods if item.mode != "unknown"]
    if not known:
        return None
    common = known[0]
    for item in known[1:]:
        intersection = common.intersects(item)
        if intersection is None:
            return False
        common = intersection
    return common if len(known) == len(periods) else None


def _series_sort(point: GraphMatch) -> tuple[str, str, str]:
    fact = point.facts[0]
    return fact.period.bounds[0] or "", fact.period.bounds[1] or "", fact.target_id


def _difference(
    old: GraphMatch, new: GraphMatch, *, kind: Literal["temporal", "cross_entity"],
) -> EvolutionChange:
    first, second = old.facts[0], new.facts[0]
    assert isinstance(first, ObservationFact) and isinstance(second, ObservationFact)
    with localcontext() as context:
        context.prec = 100
        old_value = Decimal(first.normalized[0])
        absolute = Decimal(second.normalized[0]) - old_value
        percent = _decimal_text(absolute / old_value * 100) if old_value > 0 else None
        percentage_points = _decimal_text(absolute * 100) if first.normalized[1] == "ratio" else None
    return EvolutionChange(old, new, _decimal_text(absolute), percent, first.normalized[2], percentage_points, kind)


def _within_window(period: ValidPeriod, window: tuple[date, date] | None) -> bool:
    return (
        period.valid_from is not None and period.valid_to is not None
        and (window is None or (window[0] <= period.valid_from and period.valid_to <= window[1]))
    )


def _evolution_result(result: GraphResult, spec: EvolutionQuery) -> EvolutionResult:
    if result.status not in {"hit", "truncated"}:
        return EvolutionResult(result.status, watermark=result.watermark)
    groups: dict[tuple[str, tuple[str, ...]], list[GraphMatch]] = defaultdict(list)
    for match in result.matches:
        fact = match.facts[0]
        assert isinstance(fact, ObservationFact)
        if spec.scope_key is None or fact.scope_key == spec.scope_key:
            groups[(fact.subject.entity_id, fact.comparison_key)].append(match)
    series, changes, cross_entity = [], [], []
    same_period: dict[tuple[tuple[str, ...], tuple[str | None, str | None]], list[GraphMatch]] = defaultdict(list)
    remaining_points = min(24, spec.limit)
    truncated = result.status == "truncated"
    for (_, key), points in sorted(groups.items()):
        ordered = sorted(points, key=_series_sort)
        selected = []
        for point in ordered:
            if len(point.facts) > remaining_points:
                truncated = True
                break
            selected.append(point)
            remaining_points -= len(point.facts)
        if not selected:
            continue
        first = selected[0].facts[0]
        assert isinstance(first, ObservationFact)
        status = "disputed" if any(point.stability == "disputed" for point in selected) else "verified"
        aligned = []
        for point in selected:
            fact = point.facts[0]
            assert isinstance(fact, ObservationFact)
            comparison_allowed = (
                fact.comparable and point.stability == "verified" and _within_window(fact.period, spec.period_range)
                and not any(
                    point is not other and fact.period.intersects(other.facts[0].period) is not None
                    for other in ordered
                )
            )
            aligned.append(comparison_allowed)
            if comparison_allowed:
                same_period[(key, fact.period.bounds)].append(point)
            elif status == "verified":
                status = "incomparable"
        for index, (old, new) in enumerate(zip(selected, selected[1:])):
            if (
                aligned[index] and aligned[index + 1]
                and old.facts[0].period.valid_to == new.facts[0].period.valid_from
            ):
                changes.append(_difference(old, new, kind="temporal"))
            elif status == "verified":
                status = "incomparable"
        series.append(EvolutionSeries(first.subject, key, tuple(selected), status))
    for _, points in sorted(same_period.items()):
        for index, first in enumerate(points):
            for second in points[index + 1:]:
                cross_entity.append(_difference(first, second, kind="cross_entity"))
    return EvolutionResult(
        "truncated" if truncated else "hit" if series else "empty", tuple(series),
        () if truncated else tuple(changes), () if truncated else tuple(cross_entity),
        result.watermark, result.candidates, result.hydrated_sources,
    )


def evidence_ref_to_payload(reference: EvidenceRef) -> dict[str, Any]:
    """Serialize an EvidenceRef into a JSON-safe dict."""
    return {
        "request_id": reference.request_id,
        "source_id": reference.source_id,
        "chunk_ids": list(reference.chunk_ids),
        "artifact_version": reference.artifact_version,
        "content_sha256": reference.content_sha256,
    }


def evidence_ref_from_payload(payload: dict[str, Any]) -> EvidenceRef:
    """Deserialize an EvidenceRef from a JSON payload."""
    return EvidenceRef(
        request_id=str(payload["request_id"]),
        source_id=str(payload["source_id"]),
        chunk_ids=tuple(str(item) for item in payload.get("chunk_ids") or ()),
        artifact_version=str(payload["artifact_version"]),
        content_sha256=str(payload["content_sha256"]),
    )


def bound_atom_to_payload(atom: BoundAtom) -> dict[str, Any]:
    """Serialize a BoundAtom for terminal batch persistence."""
    return {
        "atom_id": atom.atom_id,
        "ledger_version": int(atom.ledger_version),
        "claim_id": atom.claim_id,
        "fact_fingerprint": atom.fact_fingerprint,
        "nli_label": atom.nli_label,
        "confidence": float(atom.confidence),
        "verifier_version": atom.verifier_version,
        "support": [evidence_ref_to_payload(ref) for ref in atom.support],
        "is_fallback": bool(atom.is_fallback),
        "verified_at": atom.verified_at,
        "decomposition_version": int(atom.decomposition_version),
    }


def bound_atom_from_payload(payload: dict[str, Any]) -> BoundAtom:
    """Deserialize a BoundAtom from a terminal batch payload."""
    return BoundAtom(
        atom_id=str(payload["atom_id"]),
        ledger_version=int(payload["ledger_version"]),
        claim_id=str(payload["claim_id"]),
        fact_fingerprint=str(payload["fact_fingerprint"]),
        nli_label=str(payload["nli_label"]),
        confidence=float(payload["confidence"]),
        verifier_version=str(payload["verifier_version"]),
        support=tuple(evidence_ref_from_payload(item) for item in payload.get("support") or ()),
        is_fallback=bool(payload.get("is_fallback", False)),
        verified_at=str(payload.get("verified_at") or ""),
        decomposition_version=int(payload.get("decomposition_version", 1)),
    )


def publication_binding_to_payload(binding: PublicationBinding) -> dict[str, Any]:
    """Serialize a PublicationBinding for terminal batch persistence."""
    return {
        "ledger_version": int(binding.ledger_version),
        "claim_id": binding.claim_id,
        "fact_fingerprint": binding.fact_fingerprint,
        "support": [evidence_ref_to_payload(ref) for ref in binding.support],
        "allowed": bool(binding.allowed),
    }


def publication_binding_from_payload(payload: dict[str, Any]) -> PublicationBinding:
    """Deserialize a PublicationBinding from a terminal batch payload."""
    return PublicationBinding(
        ledger_version=int(payload["ledger_version"]),
        claim_id=str(payload["claim_id"]),
        fact_fingerprint=str(payload["fact_fingerprint"]),
        support=tuple(evidence_ref_from_payload(item) for item in payload.get("support") or ()),
        allowed=bool(payload.get("allowed", True)),
    )


def candidate_to_payload(candidate: VerifiedCandidate) -> dict[str, Any]:
    """Serialize a VerifiedCandidate including its typed fact."""
    return {
        "ledger_version": int(candidate.ledger_version),
        "claim_id": candidate.claim_id,
        "statement": candidate.statement,
        "fact": json.loads(_json(_fact_data(candidate.fact))),
        "required_atomic_claim_ids": list(candidate.required_atomic_claim_ids),
        "atoms": [bound_atom_to_payload(atom) for atom in candidate.atoms],
        "publication": publication_binding_to_payload(candidate.publication),
        "deterministically_checked": bool(candidate.deterministically_checked),
        "unresolved_conflict": bool(candidate.unresolved_conflict),
        "conflict_id": candidate.conflict_id,
    }


def candidate_from_payload(payload: dict[str, Any]) -> VerifiedCandidate:
    """Deserialize a VerifiedCandidate from a persisted payload."""
    return VerifiedCandidate(
        ledger_version=int(payload["ledger_version"]),
        claim_id=str(payload["claim_id"]),
        statement=str(payload["statement"]),
        fact=_fact_from_payload(dict(payload["fact"])),
        required_atomic_claim_ids=tuple(str(item) for item in payload.get("required_atomic_claim_ids") or ()),
        atoms=tuple(bound_atom_from_payload(item) for item in payload.get("atoms") or ()),
        publication=publication_binding_from_payload(dict(payload["publication"])),
        deterministically_checked=bool(payload.get("deterministically_checked", False)),
        unresolved_conflict=bool(payload.get("unresolved_conflict", False)),
        conflict_id=str(payload.get("conflict_id") or ""),
    )


def graph_evidence_to_payload(evidence: GraphEvidence) -> dict[str, Any]:
    """Serialize GraphEvidence for snapshot reconstruction."""
    return {
        "tenant_id": evidence.tenant_id,
        "request_id": evidence.request_id,
        "source_id": evidence.source_id,
        "artifact_version": evidence.artifact_version,
        "fragments": [asdict(fragment) for fragment in evidence.fragments],
        "level": evidence.level,
        "trusted": bool(evidence.trusted),
        "visible": bool(evidence.visible),
        "facts_eligible": bool(evidence.facts_eligible),
        "sanitizer_version": int(evidence.sanitizer_version),
        "isolated": bool(evidence.isolated),
        "origin": evidence.origin,
        "acl_reference": evidence.acl_reference,
    }


def graph_evidence_from_payload(payload: dict[str, Any]) -> GraphEvidence | None:
    """Deserialize GraphEvidence from a snapshot payload, failing closed on corruption."""
    if not isinstance(payload, dict):
        return None
    try:
        fragments = tuple(
            EvidenceFragment(
                chunk_id=str(item["chunk_id"]),
                anchor=str(item["anchor"]),
                text=str(item["text"]),
            )
            for item in payload.get("fragments") or ()
        )
        return GraphEvidence(
            tenant_id=str(payload["tenant_id"]),
            request_id=str(payload["request_id"]),
            source_id=str(payload["source_id"]),
            artifact_version=str(payload["artifact_version"]),
            fragments=fragments,
            level=str(payload["level"]),
            trusted=bool(payload["trusted"]),
            visible=bool(payload["visible"]),
            facts_eligible=bool(payload["facts_eligible"]),
            sanitizer_version=int(payload["sanitizer_version"]),
            isolated=bool(payload.get("isolated", False)),
            origin=str(payload.get("origin") or "web"),
            acl_reference=str(payload.get("acl_reference") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def terminal_snapshot_to_payload(snapshot: TerminalGraphSnapshot) -> dict[str, Any]:
    """Serialize a TerminalGraphSnapshot into a CAS outbox JSON payload."""
    batch = snapshot.batch
    return {
        "schema_version": int(snapshot.schema_version),
        "normalization_version": int(snapshot.normalization_version),
        "tenant_id": snapshot.tenant_id,
        "request_id": snapshot.request_id,
        "status": snapshot.status,
        "policy_version": int(snapshot.policy_version),
        "owner_epoch": snapshot.owner_epoch,
        "historical_receipt_ids": list(snapshot.historical_receipt_ids),
        "active_receipt_ids": list(snapshot.active_receipt_ids),
        "batch": {
            "batch_id": batch.batch_id,
            "digest": batch.digest,
            "source_epoch": int(batch.source_epoch),
            "ledger_version": int(batch.ledger_version),
            "report_version": int(batch.report_version),
            "publication_digest": batch.publication_digest,
            "candidates": [candidate_to_payload(candidate) for candidate in batch.candidates],
            "retirements": [asdict(retirement) for retirement in batch.retirements],
            "status": batch.status,
        },
    }


def terminal_snapshot_from_payload(payload: dict[str, Any] | None) -> TerminalGraphSnapshot | None:
    """Deserialize a TerminalGraphSnapshot from a CAS outbox JSON payload."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("terminal_snapshot") if isinstance(payload.get("terminal_snapshot"), dict) else payload
    raw_batch = raw.get("batch") if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw_batch, dict):
        return None
    try:
        candidates = tuple(candidate_from_payload(item) for item in raw_batch.get("candidates") or ())
        retirements = tuple(
            Retirement(
                request_id=str(item["request_id"]),
                receipt_id=str(item["receipt_id"]),
                reason=str(item["reason"]),
            )
            for item in raw_batch.get("retirements") or ()
        )
        batch = TerminalBatch(
            batch_id=str(raw_batch["batch_id"]),
            digest=str(raw_batch["digest"]),
            source_epoch=int(raw_batch["source_epoch"]),
            ledger_version=int(raw_batch["ledger_version"]),
            report_version=int(raw_batch["report_version"]),
            publication_digest=str(raw_batch["publication_digest"]),
            candidates=candidates,
            retirements=retirements,
            status=str(raw_batch.get("status") or "pending"),
        )
        owner_epoch = raw.get("owner_epoch")
        return TerminalGraphSnapshot(
            tenant_id=str(raw["tenant_id"]),
            request_id=str(raw["request_id"]),
            status=str(raw["status"]),
            policy_version=int(raw["policy_version"]),
            owner_epoch=int(owner_epoch) if owner_epoch is not None else None,
            batch=batch,
            historical_receipt_ids=tuple(str(item) for item in raw.get("historical_receipt_ids") or ()),
            active_receipt_ids=tuple(str(item) for item in raw.get("active_receipt_ids") or ()),
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
            normalization_version=int(raw.get("normalization_version", NORMALIZATION_VERSION)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_graph_store(config: Any) -> GraphStore:
    """Instantiate the configured tenant-scoped graph store backend."""
    backend = str(getattr(config, "v4_knowledge_graph_backend", "sqlite") or "sqlite").strip().lower()
    if backend == "memory":
        return InMemoryGraphStore()
    if backend == "postgres":
        from services.storage_backends import PostgresGraphStore

        dsn = str(getattr(config, "database_url", None) or "").strip()
        try:
            store = PostgresGraphStore(dsn=dsn)
            store.probe(deadline=time.monotonic() + 2.0)
            return store
        except Exception:
            if getattr(config, "v4_graph_allow_in_memory_fallback", False):
                return InMemoryGraphStore()
            raise
    return SQLiteGraphStore(str(getattr(config, "v4_knowledge_graph_dir", ".memory/knowledge_graph")))

