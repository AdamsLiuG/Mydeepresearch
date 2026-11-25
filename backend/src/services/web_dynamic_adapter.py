"""Optional dynamic extraction through a deployment-controlled egress boundary."""

from __future__ import annotations

import json
import re
import secrets
import ssl
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from html import escape
from ipaddress import ip_address
from threading import Condition, Event, Lock
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import parse_qsl, unquote, urlsplit

import h11

from config import validate_reader_token
from metrics import metrics_registry
from services.web_fetch_gateway import (
    MAX_HEADER_BYTES,
    FetchedDocument,
    FetchQuota,
    FetchScope,
    GatewayError,
    _header_value,
    _HTTPResponse,
    _open_pinned,
    _send,
    _Target,
    _WireReader,
    normalize_url,
)

if TYPE_CHECKING:
    from services.web_fetch_gateway import WebFetchGateway

_PAGE_BYTES = 4 * 1024 * 1024
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_RESOURCE_TYPES = _HTML_TYPES | {"application/javascript", "text/javascript", "text/css", "application/json", "text/plain"}
_RESOURCE_COUNT = 32
_BROWSER_RESOURCES = {"script", "stylesheet", "xhr", "fetch"}
_AUTH_KEYS = {"access_token", "api_key", "apikey", "auth", "authorization", "key", "password", "session", "sessionid", "sig", "signature", "token"}
_AMBIGUOUS_PATH = re.compile(r"%(?:2f|5c|25|3b|3f|23|00)|//|[;\\]", re.IGNORECASE)
_ENDPOINT_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _path(url: str) -> str:
    path = urlsplit(url).path or "/"
    if _AMBIGUOUS_PATH.search(path):
        raise ValueError("ambiguous path")
    decoded = unquote(path, errors="strict")
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise ValueError("ambiguous path segment")
    if any(ord(character) < 32 or ord(character) == 127 for character in decoded):
        raise ValueError("control character in path")
    return path


def _admitted_url(raw: str) -> str:
    # The raw representation also matters if a normalizer cleans up dot segments.
    if not isinstance(raw, str):
        raise GatewayError("url_denied")
    _path(raw)
    url = normalize_url(raw)
    _path(url)
    query = urlsplit(url).query.replace(";", "&")
    for key, _ in parse_qsl(query, keep_blank_values=True):
        key = key.casefold().replace("-", "_")
        if key in _AUTH_KEYS or key.startswith(("x_amz_", "x_goog_", "oauth_")):
            raise GatewayError("url_denied")
    return url


def _configured_origin(raw: str) -> str:
    if not isinstance(raw, str) or "?" in raw or "#" in raw:
        raise ValueError("invalid origin")
    try:
        url = normalize_url(raw)
    except GatewayError as exc:
        raise ValueError("invalid origin") from exc
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.path not in {"", "/"} or parts.query:
        raise ValueError("invalid origin")
    return _origin(url)


@dataclass(frozen=True)
class DynamicOriginRule:
    origin: str
    path_prefix: str
    resource_origins: frozenset[str]

    def allows(self, raw: str, *, resource: bool) -> bool:
        try:
            url = _admitted_url(raw)
            origin = _origin(url)
            path = _path(url)
        except (GatewayError, UnicodeError, ValueError):
            return False
        if origin != self.origin:
            return resource and origin in self.resource_origins
        return self.path_prefix == "/" or path == self.path_prefix or path.startswith(f"{self.path_prefix}/")


@dataclass(frozen=True)
class DynamicTargetPolicy:
    rules: tuple[tuple[str, tuple[DynamicOriginRule, ...]], ...]

    @classmethod
    def from_config(cls, rules: Mapping[str, Any]) -> DynamicTargetPolicy:
        if not isinstance(rules, Mapping) or len(rules) > 256:
            raise ValueError("invalid dynamic origin rules")
        tenants = []
        for tenant, entries in rules.items():
            if not isinstance(tenant, str) or not tenant or len(tenant) > 128:
                raise ValueError("invalid dynamic tenant")
            if not isinstance(entries, list) or len(entries) > 64:
                raise ValueError("invalid origin list")
            tenant_rules = []
            for entry in entries:
                if not isinstance(entry, Mapping) or set(entry) - {"origin", "path_prefix", "resource_origins"}:
                    raise ValueError("invalid origin rule")
                origin = _configured_origin(entry.get("origin"))
                prefix = entry.get("path_prefix", "/")
                if not isinstance(prefix, str) or not prefix.startswith("/") or "?" in prefix or "#" in prefix:
                    raise ValueError("invalid path prefix")
                try:
                    normalized_prefix = _admitted_url(origin + prefix)
                except GatewayError as exc:
                    raise ValueError("invalid path prefix") from exc
                prefix = _path(normalized_prefix).rstrip("/") or "/"
                resources = entry.get("resource_origins", [])
                if not isinstance(resources, list) or len(resources) > _RESOURCE_COUNT:
                    raise ValueError("invalid resource origin list")
                allowed_resources = frozenset({origin, *(_configured_origin(value) for value in resources)})
                tenant_rules.append(DynamicOriginRule(origin, prefix, allowed_resources))
            tenants.append((tenant, tuple(sorted(tenant_rules, key=lambda rule: -len(rule.path_prefix)))))
        return cls(tuple(tenants))

    def admit(self, url: str, tenant_id: str) -> DynamicOriginRule:
        try:
            url = _admitted_url(url)
        except (UnicodeError, ValueError) as exc:
            raise GatewayError("url_denied") from exc
        for tenant, rules in self.rules:
            if tenant == tenant_id:
                for rule in rules:
                    if rule.allows(url, resource=False):
                        return rule
                break
        raise GatewayError("origin_not_allowed")


def validate_reader_endpoint(raw: str) -> str:
    """Validate only a fixed deployment service URL, including private service names."""
    if (
        not isinstance(raw, str) or not raw or len(raw) > 2048 or "?" in raw or "#" in raw
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError("invalid reader endpoint")
    parts = urlsplit(raw)
    hostname = parts.hostname
    if parts.scheme != "https" or parts.username is not None or parts.password is not None or not hostname:
        raise ValueError("invalid reader endpoint")
    if parts.port not in {None, 443} or parts.netloc.lower() not in {hostname, f"{hostname}:443"}:
        raise ValueError("invalid reader endpoint")
    if not all(_ENDPOINT_HOST.fullmatch(label) for label in hostname.split(".")):
        raise ValueError("invalid reader service name")
    path = _path(raw)
    if not parts.path or re.search(r"[^A-Za-z0-9_.~/-]", path):
        raise ValueError("invalid reader endpoint path")
    return f"https://{hostname}{path}"


@dataclass(frozen=True)
class ReaderEndpoint:
    url: str = field(repr=False)
    token: str = field(repr=False)
    cost_microusd: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", validate_reader_endpoint(self.url))
        validate_reader_token(self.token)
        if type(self.cost_microusd) is not int or self.cost_microusd < 0:
            raise ValueError("invalid reader price bound")


@dataclass(frozen=True)
class DynamicCapabilities:
    policy_version: int
    forced_gateway_egress: bool
    independent_dns_udp_background_block: bool
    scoped_authentication: bool
    bounded_cancellation_cleanup: bool

    def supported(self) -> bool:
        return type(self.policy_version) is int and self.policy_version == 1 and all(value is True for value in (
            self.forced_gateway_egress,
            self.independent_dns_udp_background_block,
            self.scoped_authentication,
            self.bounded_cancellation_cleanup,
        ))


@dataclass(frozen=True)
class ReaderResponse:
    status_code: int
    content_type: str
    body: bytes


class ReaderDeployment(Protocol):
    """Deployment-owned trust root, fixed connection, and isolated Reader egress.

    ``post`` performs exactly one bounded cancellable HTTPS POST with a fixed
    destination and authority, server hostname/certificate verification and a
    deployment trust root. It must disable proxy environment, pooling, retries
    and redirects, and send credentials only to the configured endpoint.
    The binding is transport metadata and never enters the JSON or page.
    """

    def capabilities(self) -> DynamicCapabilities | None: ...

    def bind(self, egress: DynamicEgress) -> AbstractContextManager[Any]: ...

    def post(
        self, endpoint: ReaderEndpoint, *, body: bytes, binding: Any,
        scope: FetchScope, max_response_bytes: int,
    ) -> ReaderResponse: ...


class ReaderHttpsTransport:
    """A one-shot HTTPS POST to the Reader service identity and pinned address."""

    def __init__(self, address: str, tls_context: ssl.SSLContext) -> None:
        self._address = str(ip_address(address))
        self._tls_context = tls_context
        self._check_trust()

    def _check_trust(self) -> None:
        if not self._tls_context.check_hostname or self._tls_context.verify_mode != ssl.CERT_REQUIRED:
            raise ValueError("verified Reader service TLS is required")

    def post(
        self, endpoint: ReaderEndpoint, *, body: bytes, binding: Any,
        scope: FetchScope, max_response_bytes: int,
    ) -> ReaderResponse:
        self._check_trust()
        scope.check()
        if not isinstance(binding, str) or not re.fullmatch(r"[A-Za-z0-9._~-]{1,512}", binding):
            raise GatewayError("adapter_unavailable")
        parts = urlsplit(endpoint.url)
        target = _Target(endpoint.url, "https", parts.netloc, 443, parts.netloc, parts.path)
        connection = h11.Connection(h11.CLIENT, max_incomplete_event_size=MAX_HEADER_BYTES)
        headers = [
            (b"Host", parts.netloc.encode("ascii")),
            (b"Authorization", f"Bearer {endpoint.token}".encode("ascii")),
            (b"X-Research-Egress", binding.encode("ascii")),
            (b"Content-Type", b"application/json"),
            (b"Content-Length", str(len(body)).encode("ascii")),
            (b"Accept", b"application/json"),
            (b"Accept-Encoding", b"identity"),
            (b"Connection", b"close"),
        ]
        wire_limit = min(max_response_bytes, _PAGE_BYTES)
        with _open_pinned(target, self._address, scope, tls_context=self._tls_context) as socket:
            request = h11.Request(method=b"POST", target=parts.path.encode("ascii"), headers=headers)
            _send(socket, connection.send(request) + connection.send(h11.Data(data=body)) + connection.send(h11.EndOfMessage()), scope)
            response = _HTTPResponse(socket, scope, _WireReader(wire_limit, None), connection)
            head = response.head()
            if head.status_code in {401, 403, 407}:
                raise GatewayError("access_denied")
            if head.status_code != 200:
                raise GatewayError("response_invalid")
            headers = tuple((name.decode("ascii"), value.decode("latin-1")) for name, value in head.headers)
            if _header_value(headers, "content-encoding").strip().lower() not in {"", "identity"}:
                raise GatewayError("content_type_denied")
            content_type = _header_value(headers, "content-type")
            if _media_type(content_type) != "application/json":
                raise GatewayError("content_type_denied")
            return ReaderResponse(head.status_code, content_type, response.body("identity", wire_limit))


@dataclass(frozen=True)
class PinnedReaderDeployment:
    """Combine a pinned Reader connection with a deployment-owned scope broker."""

    transport: ReaderHttpsTransport
    attest_capabilities: Callable[[], DynamicCapabilities | None]
    bind_scope: Callable[[DynamicEgress], AbstractContextManager[str]]

    def capabilities(self) -> DynamicCapabilities | None:
        return self.attest_capabilities()

    def bind(self, egress: DynamicEgress) -> AbstractContextManager[str]:
        return self.bind_scope(egress)

    def post(
        self, endpoint: ReaderEndpoint, *, body: bytes, binding: Any,
        scope: FetchScope, max_response_bytes: int,
    ) -> ReaderResponse:
        return self.transport.post(endpoint, body=body, binding=binding, scope=scope, max_response_bytes=max_response_bytes)


class DynamicConcurrency:
    def __init__(self) -> None:
        self._condition = Condition()
        self._queue: deque[object] = deque()
        self._active = 0
        self._tenants: dict[str, int] = {}

    @contextmanager
    def acquire(
        self, tenant: str, scope: FetchScope, *, adapter: str,
        max_concurrency: int = 2, per_tenant: int = 1,
    ) -> Iterator[None]:
        scope.check()
        start = scope.monotonic()
        deadline = min(scope.deadline, start + 0.250)
        token = object()
        with self._condition:
            if len(self._queue) >= 8:
                _metric("queue", adapter, elapsed_ms=0)
                raise GatewayError("queue_full")
            self._queue.append(token)
            try:
                while True:
                    scope.check()
                    remaining = deadline - scope.monotonic()
                    if remaining <= 0:
                        raise GatewayError("queue_full")
                    if (
                        self._queue[0] is token and self._active < max_concurrency
                        and self._tenants.get(tenant, 0) < per_tenant
                    ):
                        self._active += 1
                        self._tenants[tenant] = self._tenants.get(tenant, 0) + 1
                        break
                    self._condition.wait(min(0.025, remaining))
            finally:
                self._queue.remove(token)
                self._condition.notify_all()
                _metric("queue", adapter, elapsed_ms=(scope.monotonic() - start) * 1000)
        try:
            scope.check()
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._tenants[tenant] -= 1
                if self._tenants[tenant] == 0:
                    del self._tenants[tenant]
                self._condition.notify_all()


def _metric(event: str, adapter: str, **kwargs: Any) -> None:
    metrics_registry.record_web_extraction(event, adapter=adapter, **kwargs)


_CONCURRENCY = DynamicConcurrency()


@dataclass(frozen=True)
class DynamicAttestation:
    final_url: str
    content_type: str
    redirects: tuple[str, ...]
    resources: tuple[str, ...]


class DynamicEgress:
    """One server-owned capability and ledger bound to a single page and owner."""

    def __init__(
        self, gateway: WebFetchGateway, rule: DynamicOriginRule, url: str,
        *, tenant_id: str, request_id: str, authorize_redirect: Callable[[str], bool] | None,
    ) -> None:
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.url = url
        self._gateway = gateway
        self._rule = rule
        self._dispatch_scope: FetchScope | None = None
        self._closed = Event()
        self._lock = Lock()
        self._document_started = False
        self._document: FetchedDocument | None = None
        self._resources: set[str] = set()
        self._main_error: GatewayError | None = None
        self._quota = FetchQuota(_PAGE_BYTES, _PAGE_BYTES, _RESOURCE_COUNT)
        self._authorize_redirect = authorize_redirect
        self._associate_url: Callable[[str], bool] | None = None
        self._approved_redirects: set[str] = set()

    def close(self) -> None:
        self._closed.set()

    def activate(self, scope: FetchScope, *, associate_url: Callable[[str], bool] | None = None) -> None:
        self._dispatch_scope = scope
        self._associate_url = associate_url

    def fetch(self, raw: str, *, kind: str, method: str, resource_type: str | None = None) -> FetchedDocument:
        dispatch_scope = self._dispatch_scope
        if (
            dispatch_scope is None or self._closed.is_set() or method not in {"GET", "HEAD"}
            or kind not in {"document", "subresource"}
        ):
            raise GatewayError("access_denied")
        dispatch_scope.check()
        resource = kind == "subresource"
        if dispatch_scope.adapter == "playwright" and resource and resource_type not in _BROWSER_RESOURCES:
            raise GatewayError("access_denied")
        try:
            url = _admitted_url(raw)
        except (UnicodeError, ValueError) as exc:
            raise GatewayError("url_denied") from exc
        with self._lock:
            if not self._rule.allows(url, resource=resource):
                _metric("egress_denied", dispatch_scope.adapter, reason="origin_not_allowed")
                raise GatewayError("origin_not_allowed")
            if not resource:
                if self._document_started or url != self.url or method != "GET":
                    raise GatewayError("access_denied")
                self._document_started = True
            elif self._document is None:
                raise GatewayError("access_denied")

        def authorized(candidate: str) -> bool:
            if self._closed.is_set() or not self._rule.allows(candidate, resource=resource):
                return False
            if resource or candidate == url:
                return True
            with self._lock:
                if candidate not in self._approved_redirects:
                    if self._authorize_redirect is not None and not self._authorize_redirect(candidate):
                        return False
                    if self._associate_url is not None and not self._associate_url(candidate):
                        return False
                    self._approved_redirects.add(candidate)
            return True

        scope = replace(
            dispatch_scope,
            cancelled=lambda: self._closed.is_set() or dispatch_scope.cancelled(),
        )
        try:
            document = self._gateway.fetch(
                url, scope=scope, method=method, authorize=authorized, quota=self._quota,
                count_request=resource, allowed_types=_RESOURCE_TYPES if resource else _HTML_TYPES,
                max_wire_bytes=_PAGE_BYTES, max_body_bytes=_PAGE_BYTES,
            )
            scope.check()
        except GatewayError as exc:
            if not resource:
                self._main_error = exc
            raise
        with self._lock:
            if resource:
                self._resources.update((url, *document.redirects, document.final_url))
            else:
                self._document = document
        return document

    def attest(self) -> DynamicAttestation:
        with self._lock:
            if self._main_error is not None:
                raise self._main_error
            if self._document is None:
                raise GatewayError("response_invalid")
            return DynamicAttestation(
                self._document.final_url, _media_type(self._document.content_type),
                self._document.redirects, tuple(sorted(self._resources)),
            )


class DynamicEgressBroker:
    """Map ephemeral capabilities to page scopes in the trusted proxy process.

    The proxy supplies tenant and request identity from its authenticated
    service channel. Headers controlled by a target page cannot supply them.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._scopes: dict[str, DynamicEgress] = {}

    @contextmanager
    def bind(self, egress: DynamicEgress) -> Iterator[str]:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._scopes[token] = egress
        try:
            yield token
        finally:
            with self._lock:
                self._scopes.pop(token, None)

    def fetch(
        self, token: str, *, tenant_id: str, request_id: str, url: str, kind: str, method: str,
        resource_type: str | None = None,
    ) -> FetchedDocument:
        with self._lock:
            egress = self._scopes.get(token) if isinstance(token, str) and len(token) == 43 else None
        if egress is None or (egress.tenant_id, egress.request_id) != (tenant_id, request_id):
            raise GatewayError("access_denied")
        if resource_type is None:
            return egress.fetch(url, kind=kind, method=method)
        return egress.fetch(url, kind=kind, method=method, resource_type=resource_type)


_VISIBLE_TAGS = frozenset((
    "a", "abbr", "article", "aside", "b", "blockquote", "body", "caption", "cite", "code", "dd", "div", "dl", "dt",
    "em", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "html", "i", "li", "main",
    "mark", "nav", "ol", "p", "pre", "q", "s", "section", "small", "span", "strong", "sub", "summary", "sup",
    "table", "tbody", "td", "tfoot", "th", "thead", "time", "tr", "u", "ul",
))
_IGNORED_TAGS = frozenset((
    "audio", "button", "canvas", "embed", "form", "head", "iframe", "input", "noscript", "object", "script",
    "select", "style", "svg", "template", "textarea", "video",
))
_DOM_NODES = 25_000
_DOM_DEPTH = 64
_DOM_CHARS = 500_000


def _visible_html(root: Any, *, scope: FetchScope) -> bytes:
    parts: list[str] = []
    nodes = 0
    characters = 0

    def render(node: Any, depth: int) -> None:
        nonlocal nodes, characters
        nodes += 1
        if nodes % 128 == 0:
            scope.check()
        if depth > _DOM_DEPTH or nodes > _DOM_NODES:
            raise GatewayError("size_limit")
        if isinstance(node, str):
            characters += len(node)
            if characters > _DOM_CHARS:
                raise GatewayError("size_limit")
            parts.append(escape(node, quote=False))
            return
        if not isinstance(node, dict) or set(node) - {"tag", "visible", "children", "attrs"}:
            raise GatewayError("visibility_uncertain")
        tag, visible, children = node.get("tag"), node.get("visible"), node.get("children")
        if (
            not isinstance(tag, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", tag)
            or type(visible) is not bool or not isinstance(children, list)
        ):
            raise GatewayError("visibility_uncertain")
        if not visible or tag in _IGNORED_TAGS:
            return
        if tag == "br":
            if children:
                raise GatewayError("visibility_uncertain")
            parts.append("<br>")
            return
        tag = tag if tag in _VISIBLE_TAGS else "span"
        attrs = node.get("attrs", {})
        if not isinstance(attrs, dict) or set(attrs) - {"href", "rowspan", "colspan", "scope", "id"}:
            raise GatewayError("visibility_uncertain")
        safe_attrs = ""
        for name, value in attrs.items():
            if not isinstance(value, str) or len(value) > 2048:
                raise GatewayError("visibility_uncertain")
            if name == "href" and tag != "a":
                continue
            if name in {"rowspan", "colspan", "scope"} and tag not in {"td", "th"}:
                continue
            characters += len(value)
            if characters > _DOM_CHARS:
                raise GatewayError("size_limit")
            safe_attrs += f' {name}="{escape(value, quote=True)}"'
        parts.append(f"<{tag}{safe_attrs}>")
        for child in children:
            render(child, depth + 1)
        parts.append(f"</{tag}>")

    if not isinstance(root, dict) or root.get("visible") is not True:
        raise GatewayError("visibility_uncertain")
    render(root, 0)
    body = "".join(parts).encode("utf-8")
    if len(body) > _PAGE_BYTES:
        raise GatewayError("size_limit")
    scope.check()
    return body


def _validate_response(value: Any, attested: DynamicAttestation, *, scope: FetchScope) -> FetchedDocument:
    required = {"version", "final_url", "redirects", "resources", "content_type"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - {"visible_dom", "quality_counts"}:
        raise GatewayError("response_invalid")
    if type(value["version"]) is not int or value["version"] != 1:
        raise GatewayError("response_invalid")
    result_type = value["content_type"]
    if (
        value["final_url"] != attested.final_url or value["redirects"] != list(attested.redirects)
        or not isinstance(value["resources"], list) or len(value["resources"]) > _RESOURCE_COUNT
        or any(not isinstance(url, str) for url in value["resources"])
        or sorted(value["resources"]) != list(attested.resources)
        or not isinstance(result_type, str) or result_type not in _HTML_TYPES or result_type != attested.content_type
    ):
        raise GatewayError("response_invalid")
    quality = value.get("quality_counts", {})
    if (
        not isinstance(quality, dict) or len(quality) > 32
        or any(len(key) > 64 or type(count) is not int or not 0 <= count <= _DOM_CHARS for key, count in quality.items())
    ):
        raise GatewayError("response_invalid")
    try:
        body = _visible_html(value.get("visible_dom"), scope=scope)
    except UnicodeError as exc:
        raise GatewayError("response_invalid") from exc
    return FetchedDocument(attested.final_url, "text/html", body, redirects=attested.redirects)


class _DynamicAdapter:
    name: str

    def __init__(
        self, gateway: WebFetchGateway, policy: DynamicTargetPolicy, *,
        max_concurrency: int = 2, per_tenant: int = 1, timeout_seconds: float = 12,
    ) -> None:
        if (
            type(max_concurrency) is not int or not 1 <= max_concurrency <= 2
            or type(per_tenant) is not int or per_tenant != 1
            or type(timeout_seconds) not in {int, float} or not 0 < timeout_seconds <= 12
        ):
            raise ValueError("invalid dynamic limits")
        self._gateway = gateway
        self._policy = policy
        self._limits = max_concurrency, per_tenant, timeout_seconds

    def can_reuse(self, url: str, *, tenant_id: str) -> bool:
        try:
            self._policy.admit(url, tenant_id)
            self._require_capabilities(self._deployment)
        except GatewayError:
            return False
        return True

    def _admit(
        self, raw: str, scope: FetchScope, tenant_id: str, request_id: str,
    ) -> tuple[str, DynamicOriginRule, FetchScope]:
        scope.check()
        if not tenant_id or not request_id:
            raise GatewayError("access_denied")
        rule = self._policy.admit(raw, tenant_id)
        url = _admitted_url(raw)
        scope = replace(scope, adapter=self.name, deadline=min(scope.deadline, scope.monotonic() + self._limits[2]))
        return url, rule, scope

    def _acquire(self, tenant_id: str, scope: FetchScope) -> AbstractContextManager[None]:
        return _CONCURRENCY.acquire(
            tenant_id, scope, adapter=self.name, max_concurrency=self._limits[0], per_tenant=self._limits[1],
        )

    @staticmethod
    def _require_capabilities(deployment: Any) -> None:
        try:
            capabilities = deployment.capabilities() if deployment is not None else None
        except Exception as exc:
            raise GatewayError("adapter_unavailable") from exc
        if not isinstance(capabilities, DynamicCapabilities) or not capabilities.supported():
            raise GatewayError("adapter_unavailable")

    @contextmanager
    def _attempt(
        self, scope: FetchScope, demand: dict[str, int], on_dispatch: Callable[[], None] | None,
        dispatch_attempt: Callable[[dict[str, int]], AbstractContextManager] | None = None,
    ) -> Iterator[tuple[FetchScope, Callable[[str], bool] | None]]:
        scope.check()
        try:
            with (dispatch_attempt or scope.attempt)(demand) as attempt:
                start = scope.monotonic()
                deadline = scope.deadline
                attempt_deadline = getattr(attempt, "deadline", None)
                if isinstance(attempt_deadline, (int, float)):
                    deadline = min(deadline, attempt_deadline)
                timeout = getattr(attempt, "timeout", None)
                if isinstance(timeout, (int, float)):
                    deadline = min(deadline, start + timeout)
                scope = replace(scope, deadline=deadline)
                scope.check()
                if on_dispatch is not None:
                    on_dispatch()
                if dispatch_attempt is None:
                    _metric("l2_selected", self.name)
                _metric("l2_inflight", self.name, amount=1)
                try:
                    yield scope, getattr(attempt, "associate_dynamic_url", None)
                finally:
                    _metric("l2_inflight", self.name, amount=-1)
        except GatewayError:
            raise
        except TimeoutError as exc:
            raise GatewayError("timeout") from exc
        except Exception as exc:
            if getattr(exc, "budget_denied", False):
                if str(exc) == "not_eligible":
                    raise GatewayError("not_eligible") from None
                raise GatewayError("budget_denied") from exc
            scope.check()
            raise GatewayError("network_error") from exc


class ReaderHttpAdapter(_DynamicAdapter):
    name = "reader_http"

    def __init__(
        self, gateway: WebFetchGateway, policy: DynamicTargetPolicy, endpoint: ReaderEndpoint,
        *, deployment: ReaderDeployment | None = None,
        max_concurrency: int = 2, per_tenant: int = 1, timeout_seconds: float = 12,
    ) -> None:
        super().__init__(gateway, policy, max_concurrency=max_concurrency, per_tenant=per_tenant, timeout_seconds=timeout_seconds)
        self._endpoint = endpoint
        self._deployment = deployment

    def extract(
        self, url: str, *, scope: FetchScope, tenant_id: str, request_id: str,
        on_dispatch: Callable[[], None] | None = None,
        authorize_redirect: Callable[[str], bool] | None = None,
        dispatch_attempt: Callable[[dict[str, int]], AbstractContextManager] | None = None,
    ) -> FetchedDocument:
        url, rule, scope = self._admit(url, scope, tenant_id, request_id)
        deployment = self._deployment
        self._require_capabilities(deployment)
        with self._acquire(tenant_id, scope):
            egress = DynamicEgress(
                self._gateway, rule, url, tenant_id=tenant_id, request_id=request_id,
                authorize_redirect=authorize_redirect,
            )
            demand = {"web_fetches": 1, "dynamic_fetches": 1}
            if self._endpoint.cost_microusd:
                demand["cost_microusd"] = self._endpoint.cost_microusd
            try:
                with ExitStack() as binding_stack:
                    binding = binding_stack.enter_context(deployment.bind(egress))
                    try:
                        with self._attempt(scope, demand, on_dispatch, dispatch_attempt) as (dispatch_scope, associate_url):
                            egress.activate(dispatch_scope, associate_url=associate_url)
                            try:
                                response = self._post(deployment, binding, url, dispatch_scope)
                            finally:
                                egress.close()
                                binding_stack.close()
                            return _validate_response(self._decode(response), egress.attest(), scope=dispatch_scope)
                    finally:
                        egress.close()
            except GatewayError:
                raise
            except TimeoutError as exc:
                raise GatewayError("timeout") from exc
            except Exception as exc:
                scope.check()
                raise GatewayError("network_error") from exc

    def _post(self, deployment: ReaderDeployment, binding: Any, url: str, scope: FetchScope) -> ReaderResponse:
        scope.check()
        start = scope.monotonic()
        body = json.dumps({
            "url": url,
            "deadline_ms": max(0, int((scope.deadline - start) * 1000)),
            "max_content_bytes": _PAGE_BYTES,
        }, separators=(",", ":")).encode()
        _metric("http_attempt", self.name, kind="reader_service")
        try:
            return deployment.post(self._endpoint, body=body, binding=binding, scope=scope, max_response_bytes=_PAGE_BYTES)
        finally:
            _metric("http_elapsed", self.name, kind="reader_service", elapsed_ms=(scope.monotonic() - start) * 1000)

    @staticmethod
    def _decode(response: ReaderResponse) -> Any:
        if (
            not isinstance(response, ReaderResponse) or response.status_code != 200
            or not isinstance(response.content_type, str)
            or _media_type(response.content_type) != "application/json"
            or not isinstance(response.body, bytes)
        ):
            raise GatewayError("response_invalid")
        if len(response.body) > _PAGE_BYTES:
            raise GatewayError("size_limit")
        try:
            return json.loads(response.body.decode("utf-8"))
        except (ValueError, RecursionError) as exc:
            raise GatewayError("response_invalid") from exc


class PlaywrightWorker(Protocol):
    def render(self, url: str, *, egress: DynamicEgress, scope: FetchScope) -> Mapping[str, Any]: ...


class PlaywrightDeployment(Protocol):
    """Own the OS network boundary, process sandbox, and forced cleanup.

    Capability validation is local and bounded. The worker uses an isolated
    browser process and per-task egress authority, and its context manager
    revokes, closes and reaps the task by the deadline or after cancellation.
    The worker supplies computed-visible main-document structure, never HTML
    or text gathered in the page's JavaScript execution world.
    """

    def capabilities(self) -> DynamicCapabilities | None: ...

    def open_worker(self, egress: DynamicEgress, *, scope: FetchScope) -> AbstractContextManager[PlaywrightWorker]: ...


class PlaywrightAdapter(_DynamicAdapter):
    name = "playwright"

    def __init__(
        self, gateway: WebFetchGateway, policy: DynamicTargetPolicy, *,
        deployment: PlaywrightDeployment | None = None, enabled: bool = False,
        max_concurrency: int = 2, per_tenant: int = 1, timeout_seconds: float = 12,
    ) -> None:
        super().__init__(gateway, policy, max_concurrency=max_concurrency, per_tenant=per_tenant, timeout_seconds=timeout_seconds)
        self._deployment = deployment
        self._enabled = enabled

    def can_reuse(self, url: str, *, tenant_id: str) -> bool:
        return self._enabled and super().can_reuse(url, tenant_id=tenant_id)

    def extract(
        self, url: str, *, scope: FetchScope, tenant_id: str, request_id: str,
        on_dispatch: Callable[[], None] | None = None,
        authorize_redirect: Callable[[str], bool] | None = None,
        can_dispatch: Callable[[dict[str, int]], bool] | None = None,
        dispatch_attempt: Callable[[dict[str, int]], AbstractContextManager] | None = None,
    ) -> FetchedDocument:
        if not self._enabled:
            raise GatewayError("adapter_unavailable")
        url, rule, scope = self._admit(url, scope, tenant_id, request_id)
        deployment = self._deployment
        self._require_capabilities(deployment)
        with self._acquire(tenant_id, scope):
            if can_dispatch is None or not can_dispatch({"web_fetches": 1}):
                raise GatewayError("budget_denied")
            with self._attempt(
                scope, {"dynamic_fetches": 1}, on_dispatch, dispatch_attempt,
            ) as (dispatch_scope, associate_url):
                egress = DynamicEgress(
                    self._gateway, rule, url, tenant_id=tenant_id, request_id=request_id,
                    authorize_redirect=authorize_redirect,
                )
                egress.activate(dispatch_scope, associate_url=associate_url)
                started = scope.monotonic()
                try:
                    with deployment.open_worker(egress, scope=dispatch_scope) as worker:
                        try:
                            response = worker.render(url, egress=egress, scope=dispatch_scope)
                        finally:
                            egress.close()
                    return _validate_response(response, egress.attest(), scope=dispatch_scope)
                except ImportError as exc:
                    raise GatewayError("adapter_unavailable") from exc
                finally:
                    egress.close()
                    _metric("render", self.name, elapsed_ms=(scope.monotonic() - started) * 1000)


# The sampler runs in a CDP isolated world. Only its bounded, structured return
# value is sent back to the controller.
_VISIBLE_DOM_SCRIPT = r"""(() => {
    const root = document.body;
    if (!root) return { dom: null, text: 0 };
    const skipped = new Set([
        'button', 'embed', 'form', 'iframe', 'input', 'noscript', 'object', 'script',
        'select', 'style', 'template', 'textarea', 'canvas', 'svg', 'video', 'audio'
    ]);
    const documentBox = {
        left: -window.scrollX,
        right: Math.max(document.documentElement.scrollWidth, window.innerWidth) - window.scrollX,
        top: -window.scrollY,
        bottom: Math.max(document.documentElement.scrollHeight, window.innerHeight) - window.scrollY,
    };
    const layout = (rect, box) => rect.width > 0 && rect.height > 0 &&
        rect.right > box.left && rect.left < box.right && rect.bottom > box.top && rect.top < box.bottom;
    const clipped = (box, rect, x, y) => ({
        left: x ? Math.max(box.left, rect.left) : box.left,
        right: x ? Math.min(box.right, rect.right) : box.right,
        top: y ? Math.max(box.top, rect.top) : box.top,
        bottom: y ? Math.min(box.bottom, rect.bottom) : box.bottom,
    });
    const noTextColor = /^(transparent|rgba\([^)]*,\s*0(?:\.0+)?\))$/i;
    let nodes = 0;
    let chars = 0;
    let text = 0;

    function visit(node, depth, box, parentStyle) {
        if (++nodes > 25000 || depth > 64 || chars > 500000) throw new Error('bounded DOM');
        if (node.nodeType === Node.TEXT_NODE) {
            const value = node.nodeValue;
            if (!value || noTextColor.test(parentStyle.color) || noTextColor.test(parentStyle.webkitTextFillColor)) return null;
            if (value.trim()) {
                const range = document.createRange();
                range.selectNodeContents(node);
                if (!Array.from(range.getClientRects()).some(rect => layout(rect, box))) return null;
                text += value.trim().length;
            }
            chars += value.length;
            if (chars > 500000) throw new Error('bounded DOM');
            return value;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return null;
        const tag = node.localName;
        if (skipped.has(tag) || node.hasAttribute('hidden') || node.hasAttribute('inert') ||
            (node.getAttribute('aria-hidden') || '').toLowerCase() === 'true') return null;
        const style = getComputedStyle(node);
        if (style.display === 'none' || style.visibility !== 'visible' || Number(style.opacity) === 0 ||
            style.contentVisibility === 'hidden' || style.clip !== 'auto' || style.clipPath !== 'none') return null;
        const rect = node.getBoundingClientRect();
        if (style.display !== 'contents' && style.display !== 'inline' && tag !== 'br' && !layout(rect, box)) return null;
        if (tag === 'br') return {tag, visible: true, children: []};
        const attrs = {};
        for (const name of ['href', 'id', 'rowspan', 'colspan', 'scope']) {
            const value = node.getAttribute(name);
            if (value && value.length <= 2048) {
                attrs[name] = value;
                chars += value.length;
            }
        }
        if (chars > 500000) throw new Error('bounded DOM');
        const children = [];
        const clips = overflow => overflow === 'clip' || overflow === 'hidden';
        const nextBox = clipped(box, rect, clips(style.overflowX), clips(style.overflowY));
        const summary = tag === 'details' && !node.open
            ? Array.from(node.children).find(child => child.localName === 'summary')
            : null;
        const childNodes = tag === 'details' && !node.open ? (summary ? [summary] : []) : node.childNodes;
        for (const child of childNodes) {
            const value = visit(child, depth + 1, nextBox, style);
            if (value !== null) children.push(value);
        }
        return {tag, visible: true, attrs, children};
    }
    return {dom: visit(root, 0, documentBox, null), text};
})()"""

_QUIET_DOM_SCRIPT = """new Promise(resolve => {
    const waitForChange = %s;
    const duration = %d;
    let quiet;
    const finish = () => {
        observer.disconnect();
        clearTimeout(quiet);
        clearTimeout(end);
        resolve(true);
    };
    const reset = () => {
        clearTimeout(quiet);
        quiet = setTimeout(finish, 200);
    };
    const observer = new MutationObserver(reset);
    observer.observe(document, {subtree: true, childList: true, characterData: true, attributes: true});
    const end = setTimeout(finish, duration);
    if (!waitForChange) reset();
})"""


@dataclass(frozen=True)
class ChromiumSample:
    final_url: str
    visible_dom: dict[str, Any]


async def sample_isolated_chromium(
    url: str, *, scope: FetchScope, capabilities: DynamicCapabilities, proxy_server: str,
) -> ChromiumSample:
    """Run in a supervised worker with an enforced OS network boundary.

    The worker's supervisor owns proxy-only reachability, ephemeral trust and
    profile data, deadline enforcement, and process termination. Its proxy
    translates approved protocol traffic into the bound DynamicEgress capability.
    """
    import asyncio

    scope.check()
    if not capabilities.supported():
        raise GatewayError("adapter_unavailable")
    try:
        proxy = urlsplit(proxy_server)
        if (
            proxy.scheme != "http" or proxy.hostname not in {"127.0.0.1", "::1"} or not proxy.port
            or proxy.username is not None or proxy.password is not None or proxy.path or proxy.query or proxy.fragment
        ):
            raise ValueError("proxy endpoint")
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeout
        from playwright.async_api import async_playwright
    except (ImportError, ValueError) as exc:
        raise GatewayError("adapter_unavailable") from exc

    restricted_reason: str | None = None

    async def render() -> ChromiumSample:
        nonlocal restricted_reason
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(
                    headless=True, chromium_sandbox=True, proxy={"server": proxy_server},
                    args=["--disable-background-networking", "--disable-extensions", "--disable-quic", "--proxy-bypass-list=<-loopback>"],
                    timeout=milliseconds(),
                )
            except PlaywrightError as exc:
                scope.check()
                raise GatewayError("adapter_unavailable") from exc
            try:
                context = await browser.new_context(accept_downloads=False, service_workers="block")
                try:
                    page = await context.new_page()
                    if not hasattr(context, "route_web_socket"):
                        raise GatewayError("adapter_unavailable")
                    document_requested = False

                    async def stop_page(reason: str) -> None:
                        nonlocal restricted_reason
                        restricted_reason = reason
                        await page.close()

                    async def route_resource(route: Any) -> None:
                        nonlocal document_requested
                        request = route.request
                        try:
                            main_frame = request.frame == page.main_frame
                        except PlaywrightError:
                            main_frame = False
                        permitted = (
                            main_frame and request.method in {"GET", "HEAD"}
                            and request.resource_type in {"document", "script", "stylesheet", "xhr", "fetch"}
                        )
                        if permitted and request.is_navigation_request():
                            permitted = not document_requested and request.resource_type == "document" and request.url == url
                            document_requested |= permitted
                        if permitted:
                            await route.continue_()
                        else:
                            await route.abort()

                    async def block_websocket(websocket: Any) -> None:
                        await websocket.close()

                    async def handle_new_page(child: Any) -> None:
                        await child.close()
                        await stop_page("access_denied")

                    async def handle_download(download: Any) -> None:
                        await stop_page("download_blocked")

                    await context.route("**/*", route_resource)
                    await context.route_web_socket("**/*", block_websocket)
                    context.on("page", handle_new_page)
                    page.on("download", handle_download)
                    await page.goto(url, wait_until="domcontentloaded", timeout=milliseconds())
                    scope.check()
                    cdp = await context.new_cdp_session(page)
                    return await _chromium_samples(cdp, page, scope)
                finally:
                    await context.close()
            finally:
                await browser.close()

    def milliseconds() -> float:
        scope.check()
        return max(1, (scope.deadline - scope.monotonic()) * 1000)

    try:
        return await asyncio.wait_for(render(), timeout=milliseconds() / 1000)
    except (asyncio.TimeoutError, PlaywrightTimeout) as exc:
        raise GatewayError(restricted_reason or "timeout") from exc
    except PlaywrightError as exc:
        scope.check()
        raise GatewayError(restricted_reason or "network_error") from exc


async def _chromium_samples(cdp: Any, page: Any, scope: FetchScope) -> ChromiumSample:
    frame = (await cdp.send("Page.getFrameTree"))["frameTree"]["frame"]
    world = await cdp.send("Page.createIsolatedWorld", {"frameId": frame["id"], "worldName": "research-dom"})
    context_id = world["executionContextId"]
    sample_deadline = min(scope.deadline, scope.monotonic() + 2)
    initial_url = page.url

    async def evaluate(expression: str, *, promise: bool = False) -> Any:
        scope.check()
        result = await cdp.send("Runtime.evaluate", {
            "expression": expression, "contextId": context_id, "returnByValue": True, "awaitPromise": promise,
        })
        scope.check()
        if page.url != initial_url or "exceptionDetails" in result:
            raise GatewayError("visibility_uncertain")
        return result.get("result", {}).get("value")

    first = await evaluate(_VISIBLE_DOM_SCRIPT)
    if not isinstance(first, dict):
        raise GatewayError("visibility_uncertain")
    remaining_ms = int((sample_deadline - scope.monotonic()) * 1000)
    if remaining_ms > 0:
        await evaluate(_QUIET_DOM_SCRIPT % ("false" if first.get("text") else "true", remaining_ms), promise=True)
        second = await evaluate(_VISIBLE_DOM_SCRIPT)
        if not isinstance(second, dict):
            raise GatewayError("visibility_uncertain")
        if second.get("text") or not first.get("text"):
            first = second
    if not isinstance(first.get("dom"), dict):
        raise GatewayError("visibility_uncertain")
    return ChromiumSample(normalize_url(initial_url), first["dom"])
