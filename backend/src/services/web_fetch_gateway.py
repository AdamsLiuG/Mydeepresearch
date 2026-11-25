"""Bounded public web transport shared by static extraction and dynamic adapters."""

from __future__ import annotations

import errno
import ipaddress
import math
import re
import secrets
import select
import socket
import ssl
import struct
import time
import unicodedata
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import islice
from threading import BoundedSemaphore, Lock
from urllib.parse import quote, urlsplit

import h11
import idna

from metrics import metrics_registry

MAX_STATIC_BYTES = 2 * 1024 * 1024
MAX_DYNAMIC_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 32 * 1024
MAX_URL_LENGTH = 2048

_PORTS = {"http": 80, "https": 443}
_REDIRECTS = {301, 302, 303, 307, 308}
_DEFAULT_TYPES = frozenset({
    "text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/json", "application/pdf",
})
_RESERVED_SUFFIXES = (
    "localhost", "local", "internal", "invalid", "test", "example", "onion", "home.arpa", "corp", "home", "lan",
)
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z", re.ASCII)
_INVALID_PERCENT = re.compile(r"%(?![0-9a-fA-F]{2})")
_NUMERIC_LABEL = re.compile(r"(?:[0-9]+|0x[0-9a-f]*)\Z", re.ASCII)
_PATH_SAFE = "/:@!$&'()*+,;=-._~%"
_QUERY_SAFE = _PATH_SAFE + "?[]"
_DNS_MAX_ADDRESSES = 16
_DNS_MAX_ALIASES = 8
_DNS_SLOTS = BoundedSemaphore(8)
_IO_POLL_SECONDS = 0.05
_CHUNK_SIZE = 8192
_DENIED_REASONS = frozenset({
    "url_denied", "dns_denied", "origin_not_allowed", "redirect_denied", "access_denied",
    "download_blocked", "content_type_denied", "size_limit",
})
_IPV4_DENY = tuple(ipaddress.ip_network(network) for network in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.0.0.0/24", "192.0.2.0/24", "192.88.99.0/24", "192.168.0.0/16", "198.18.0.0/15",
    "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/3",
))
_IPV6_PUBLIC = ipaddress.ip_network("2000::/3")
_IPV6_DENY = tuple(ipaddress.ip_network(network) for network in (
    "2001::/23", "2001:db8::/32", "2002::/16", "3fff::/20",
))
_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class GatewayError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _public_ip(address: _IPAddress) -> bool:
    if isinstance(address, ipaddress.IPv6Address):
        if address.scope_id:
            return False
        if address.ipv4_mapped:
            return _public_ip(address.ipv4_mapped)
        return address in _IPV6_PUBLIC and not any(address in network for network in _IPV6_DENY)
    return not any(address in network for network in _IPV4_DENY)


def _canonical_hostname(hostname: str) -> str:
    try:
        name = idna.encode(hostname.replace("\u3002", ".").replace("\uff0e", ".").replace("\uff61", ".").lower())
        host = name.decode("ascii").removesuffix(".")
    except (UnicodeError, ValueError):
        raise GatewayError("url_denied") from None
    labels = host.split(".")
    if (
        len(labels) < 2
        or any(not _LABEL.fullmatch(label) for label in labels)
        or _NUMERIC_LABEL.fullmatch(labels[-1])
        or any(host == suffix or host.endswith("." + suffix) for suffix in _RESERVED_SUFFIXES)
    ):
        raise GatewayError("url_denied")
    return host


def _invalid_url_text(value: str) -> bool:
    return (
        len(value) > MAX_URL_LENGTH
        or any(char == "\\" or char.isspace() or unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value)
        or bool(_INVALID_PERCENT.search(value))
    )


@dataclass(frozen=True)
class _Target:
    url: str
    scheme: str
    host: str
    port: int
    authority: str
    request_target: str


def _target(url: str) -> _Target:
    if not isinstance(url, str) or _invalid_url_text(url):
        raise GatewayError("url_denied")
    try:
        parsed = urlsplit(url)
        port = _PORTS.get(parsed.scheme)
        authority = parsed.netloc
        if port is None or not authority or "@" in authority or "%" in authority:
            raise GatewayError("url_denied")
        if authority.startswith("["):
            raw_host, separator, tail = authority[1:].partition("]")
            if not separator or (tail and tail != f":{port}"):
                raise GatewayError("url_denied")
            literal = ipaddress.IPv6Address(raw_host)
            if not _public_ip(literal):
                raise GatewayError("url_denied")
            host = str(literal)
            authority = f"[{host}]"
        else:
            raw_host, separator, tail = authority.partition(":")
            if separator and tail != str(port):
                raise GatewayError("url_denied")
            try:
                literal_v4 = ipaddress.IPv4Address(raw_host)
            except ipaddress.AddressValueError:
                host = _canonical_hostname(raw_host)
            else:
                if not _public_ip(literal_v4):
                    raise GatewayError("url_denied")
                host = str(literal_v4)
            authority = host
    except ValueError:
        raise GatewayError("url_denied") from None
    path = quote(parsed.path or "/", safe=_PATH_SAFE)
    query = quote(parsed.query, safe=_QUERY_SAFE)
    request_target = path + ("?" + query if "?" in url.partition("#")[0] else "")
    canonical = f"{parsed.scheme}://{authority}{request_target}"
    if len(canonical) > MAX_URL_LENGTH:
        raise GatewayError("url_denied")
    return _Target(canonical, parsed.scheme, host, port, authority, request_target)


def normalize_url(url: str) -> str:
    """Canonicalize one HTTP(S) URL without performing DNS or network I/O."""
    return _target(url).url


def _remove_dot_segments(path: str) -> str:
    parts = path.split("/")
    resolved = [""]
    for index, part in enumerate(parts[1:], 1):
        if part == ".." and len(resolved) > 1:
            resolved.pop()
        if part not in {".", ".."}:
            resolved.append(part)
        elif index == len(parts) - 1:
            resolved.append("")
    return "/".join(resolved) or "/"


def _redirect_target(current: _Target, location: str) -> _Target:
    if not location or _invalid_url_text(location):
        raise GatewayError("redirect_denied")
    try:
        reference = urlsplit(location)
    except ValueError:
        raise GatewayError("redirect_denied") from None
    if reference.scheme:
        if not reference.netloc:
            raise GatewayError("redirect_denied")
        candidate = location
    elif location.startswith("//"):
        if not reference.netloc:
            raise GatewayError("redirect_denied")
        candidate = f"{current.scheme}:{location}"
    else:
        path, _, prior_query = current.request_target.partition("?")
        has_query = "?" in location.partition("#")[0]
        if reference.path:
            path = reference.path if reference.path.startswith("/") else path.rsplit("/", 1)[0] + "/" + reference.path
            path = _remove_dot_segments(path)
        elif not has_query:
            has_query = "?" in current.request_target
        query = reference.query if "?" in location.partition("#")[0] or reference.path else prior_query
        candidate = f"{current.scheme}://{current.authority}{path}" + ("?" + query if has_query else "")
    try:
        target = _target(candidate)
    except GatewayError:
        raise GatewayError("redirect_denied") from None
    if current.scheme == "https" and target.scheme != "https":
        raise GatewayError("redirect_denied")
    return target


def _allow_owner() -> bool:
    return True


def _not_cancelled() -> bool:
    return False


@dataclass(frozen=True)
class FetchScope:
    deadline: float
    check_owner: Callable[[], bool] = _allow_owner
    cancelled: Callable[[], bool] = _not_cancelled
    attempt: Callable[[dict[str, int]], AbstractContextManager] = nullcontext
    adapter: str = "static"
    monotonic: Callable[[], float] = time.monotonic
    check_policy: Callable[[], None] | None = None

    def check(self) -> None:
        if self.cancelled():
            raise GatewayError("cancelled")
        if not self.check_owner():
            raise GatewayError("lease_lost")
        if self.check_policy is not None:
            self.check_policy()
        if not math.isfinite(self.deadline) or self.monotonic() >= self.deadline:
            raise GatewayError("timeout")

    def bounded(self, seconds: float) -> FetchScope:
        return replace(self, deadline=min(self.deadline, self.monotonic() + seconds))


@dataclass
class FetchQuota:
    """Concurrency-safe cumulative limits for all fetches in a page."""

    max_wire_bytes: int
    max_body_bytes: int
    max_requests: int | None = None
    wire_bytes: int = field(default=0, init=False)
    body_bytes: int = field(default=0, init=False)
    requests: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def remaining_wire(self) -> int:
        with self._lock:
            return max(0, self.max_wire_bytes - self.wire_bytes)

    def receive(self, sock: socket.socket, limit: int) -> bytes:
        with self._lock:
            limit = min(limit, self.max_wire_bytes - self.wire_bytes)
            if limit <= 0:
                raise GatewayError("size_limit")
            data = sock.recv(limit)
            self.wire_bytes += len(data)
            return data

    def decode(self, data: bytes, limit: int, decoder: zlib.Decompress | None) -> bytes:
        with self._lock:
            decoded = _decode_chunk(data, min(limit, self.max_body_bytes - self.body_bytes), decoder)
            self.body_bytes += len(decoded)
            return decoded

    @contextmanager
    def request_slot(self) -> Iterator[Callable[[], None]]:
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise GatewayError("size_limit")
            self.requests += 1
        started = False

        def start() -> None:
            nonlocal started
            started = True

        try:
            yield start
        finally:
            if not started:
                with self._lock:
                    self.requests -= 1


@dataclass(frozen=True)
class FetchedDocument:
    final_url: str
    content_type: str
    body: bytes
    status_code: int = 200
    wire_bytes: int = 0
    headers: tuple[tuple[str, str], ...] = ()
    redirects: tuple[str, ...] = ()


def _wait(sock: socket.socket, scope: FetchScope, *, read: bool) -> None:
    while True:
        scope.check()
        timeout = min(_IO_POLL_SECONDS, max(0, scope.deadline - scope.monotonic()))
        ready_read, ready_write, _ = select.select([sock] if read else [], [] if read else [sock], [], timeout)
        if ready_read or ready_write:
            scope.check()
            return


def _socket_io(sock: socket.socket, scope: FetchScope, operation: Callable, *, read: bool) -> bytes | int:
    while True:
        scope.check()
        try:
            return operation()
        except ssl.SSLWantReadError:
            _wait(sock, scope, read=True)
        except ssl.SSLWantWriteError:
            _wait(sock, scope, read=False)
        except BlockingIOError:
            _wait(sock, scope, read=read)


def _send(sock: socket.socket, payload: bytes, scope: FetchScope) -> None:
    remaining = memoryview(payload)
    while remaining:
        sent = _socket_io(sock, scope, lambda: sock.send(remaining), read=False)
        if not sent:
            raise OSError("closed socket")
        remaining = remaining[sent:]


def _recv(sock: socket.socket, length: int, scope: FetchScope) -> bytes:
    return _socket_io(sock, scope, lambda: sock.recv(length), read=True)


def _receive_exact(sock: socket.socket, length: int, scope: FetchScope) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = _recv(sock, length - len(data), scope)
        if not chunk:
            raise OSError("closed socket")
        data.extend(chunk)
    return bytes(data)


def _socket_address(address: _IPAddress, port: int) -> tuple[int, tuple]:
    if isinstance(address, ipaddress.IPv6Address):
        return socket.AF_INET6, (str(address), port, 0, 0)
    return socket.AF_INET, (str(address), port)


def _connect_socket(sock: socket.socket, address: tuple, scope: FetchScope) -> None:
    scope.check()
    result = sock.connect_ex(address)
    if result in {errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR}:
        _wait(sock, scope, read=False)
        result = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    if result and result != errno.EISCONN:
        raise OSError(result, "connection failed")


def _check_peer(sock: socket.socket, address: str, port: int) -> None:
    actual = sock.getpeername()
    if ipaddress.ip_address(actual[0]) != ipaddress.ip_address(address) or actual[1] != port:
        raise GatewayError("network_error")


@lru_cache(maxsize=1)
def _tls_context() -> ssl.SSLContext:
    return ssl.create_default_context()


@contextmanager
def _open_pinned(
    target: _Target, address: str, scope: FetchScope, *, tls_context: ssl.SSLContext | None = None,
) -> Iterator[socket.socket]:
    context = (tls_context or _tls_context()) if target.scheme == "https" else None
    if context is not None and (not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED):
        raise GatewayError("network_error")
    family, endpoint = _socket_address(ipaddress.ip_address(address), target.port)
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setblocking(False)
        _connect_socket(sock, endpoint, scope)
        _check_peer(sock, address, target.port)
        if context is None:
            yield sock
        else:
            try:
                with context.wrap_socket(
                    sock, server_hostname=target.host, do_handshake_on_connect=False, suppress_ragged_eofs=False,
                ) as tls:
                    _socket_io(tls, scope, tls.do_handshake, read=True)
                    yield tls
            except OSError:
                scope.check()
                raise GatewayError("network_error") from None


@lru_cache(maxsize=1)
def _dns_server() -> _IPAddress:
    with open("/etc/resolv.conf", encoding="ascii", errors="replace") as resolv_conf:
        for line in resolv_conf.read(8192).splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                try:
                    address = ipaddress.ip_address(parts[1])
                    if not getattr(address, "scope_id", None):
                        return address
                except ValueError:
                    continue
    raise GatewayError("dns_denied")


def _dns_name(packet: bytes, position: int) -> tuple[str, int]:
    labels = []
    next_position = None
    name_length = 0
    for _ in range(128):
        if position >= len(packet):
            break
        size = packet[position]
        if not size:
            return ".".join(labels).lower(), next_position or position + 1
        if size & 0xC0 == 0xC0 and position + 1 < len(packet):
            pointer = ((size & 0x3F) << 8) | packet[position + 1]
            if pointer >= position or pointer < 12:
                break
            if next_position is None:
                next_position = position + 2
            position = pointer
            continue
        position += 1
        name_length += size + 1
        if size > 63 or position + size > len(packet) or name_length > 253:
            break
        try:
            labels.append(packet[position:position + size].decode("ascii"))
        except UnicodeDecodeError:
            break
        position += size
    raise GatewayError("dns_denied")


def _dns_answer(payload: bytes, hostname: str, rr_type: int, query_id: int) -> tuple[dict[str, str], dict[str, list[str]]]:
    identifier, flags, questions, answers, authorities, additional = struct.unpack_from("!6H", payload)
    if (
        identifier != query_id or flags & 0xFA0F != 0x8000 or questions != 1
        or answers + authorities + additional > 128
    ):
        raise GatewayError("dns_denied")
    question, offset = _dns_name(payload, 12)
    if question != hostname or struct.unpack_from("!2H", payload, offset) != (rr_type, 1):
        raise GatewayError("dns_denied")
    offset += 4
    aliases: dict[str, str] = {}
    addresses: dict[str, list[str]] = {}
    for index in range(answers + authorities + additional):
        owner, offset = _dns_name(payload, offset)
        record, rr_class, _, size = struct.unpack_from("!HHIH", payload, offset)
        start, offset = offset + 10, offset + 10 + size
        if offset > len(payload):
            raise GatewayError("dns_denied")
        if index >= answers:
            continue
        if rr_class != 1:
            raise GatewayError("dns_denied")
        if record == 5:  # CNAME
            alias, end = _dns_name(payload, start)
            try:
                canonical_alias = _canonical_hostname(alias)
            except GatewayError:
                raise GatewayError("dns_denied") from None
            if end != offset or canonical_alias != alias or aliases.setdefault(owner, alias) != alias:
                raise GatewayError("dns_denied")
        if record in {1, 28}:
            if size != (4 if record == 1 else 16):
                raise GatewayError("dns_denied")
            address = ipaddress.ip_address(payload[start:offset])
            if not _public_ip(address):
                raise GatewayError("dns_denied")
            if record == rr_type:
                addresses.setdefault(owner, []).append(str(address))
    if offset != len(payload):
        raise GatewayError("dns_denied")
    return aliases, addresses


def _dns_query(hostname: str, rr_type: int, scope: FetchScope) -> tuple[dict[str, str], dict[str, list[str]]]:
    family, endpoint = _socket_address(_dns_server(), 53)
    query_id = secrets.randbits(16)
    qname = b"".join(bytes([len(label)]) + label.encode("ascii") for label in hostname.split(".")) + b"\0"
    query = struct.pack("!6H", query_id, 0x0100, 1, 0, 0, 0) + qname + struct.pack("!2H", rr_type, 1)
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        _connect_socket(sock, endpoint, scope)
        _send(sock, query, scope)
        payload = _recv(sock, 65535, scope)
    identifier, flags = struct.unpack_from("!2H", payload)
    if identifier != query_id:
        raise GatewayError("dns_denied")
    if flags & 0x0200:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setblocking(False)
            _connect_socket(sock, endpoint, scope)
            _send(sock, struct.pack("!H", len(query)) + query, scope)
            length = struct.unpack("!H", _receive_exact(sock, 2, scope))[0]
            payload = _receive_exact(sock, length, scope)
    return _dns_answer(payload, hostname, rr_type, query_id)


def _dns_family(hostname: str, rr_type: int, scope: FetchScope) -> tuple[tuple[str, ...], list[str]]:
    name = hostname
    chain = [name]
    while True:
        scope.check()
        previous = name
        aliases, addresses = _dns_query(name, rr_type, scope)
        while name in aliases:
            if name in addresses or aliases[name] in chain or len(chain) > _DNS_MAX_ALIASES:
                raise GatewayError("dns_denied")
            name = aliases[name]
            chain.append(name)
        if name in addresses or not aliases:
            return tuple(chain), addresses.get(name, [])
        if previous == name:
            raise GatewayError("dns_denied")


def _resolve_dns(hostname: str, scope: FetchScope) -> tuple[str, ...]:
    scope = scope.bounded(2.0)
    while True:
        scope.check()
        if _DNS_SLOTS.acquire(timeout=min(_IO_POLL_SECONDS, max(0, scope.deadline - scope.monotonic()))):
            break
    try:
        chain_v4, ipv4 = _dns_family(hostname, 1, scope)
        chain_v6, ipv6 = _dns_family(hostname, 28, scope)
        scope.check()
        if chain_v4 != chain_v6 or not (ipv4 or ipv6) or len(ipv4) + len(ipv6) > _DNS_MAX_ADDRESSES:
            raise GatewayError("dns_denied")
        return tuple(ipv4 + ipv6)
    except (OSError, ValueError, struct.error):
        raise GatewayError("dns_denied") from None
    finally:
        _DNS_SLOTS.release()


@dataclass
class _WireReader:
    limit: int
    quota: FetchQuota | None
    total: int = 0

    def receive(self, sock: socket.socket, scope: FetchScope, max_chunk: int = _CHUNK_SIZE) -> bytes:
        count = min(max_chunk, self.limit - self.total)
        if count <= 0:
            raise GatewayError("size_limit")
        if self.quota is not None:
            data = _socket_io(sock, scope, lambda: self.quota.receive(sock, count), read=True)
        else:
            data = _recv(sock, count, scope)
        self.total += len(data)
        scope.check()
        return data


def _decode_chunk(data: bytes, limit: int, decoder: zlib.Decompress | None) -> bytes:
    try:
        decoded = decoder.decompress(data, limit + 1) if decoder is not None else data
    except zlib.error:
        raise GatewayError("response_invalid") from None
    if len(decoded) > limit:
        raise GatewayError("size_limit")
    if decoder is not None and decoder.unused_data:
        raise GatewayError("response_invalid")
    return decoded


class _HTTPResponse:
    def __init__(self, sock: socket.socket, scope: FetchScope, reader: _WireReader, connection: h11.Connection):
        self.sock = sock
        self.scope = scope
        self.reader = reader
        self.connection = connection

    def next_event(self, *, reading_head: bool = False) -> h11.Event:
        while True:
            self.scope.check()
            try:
                event = self.connection.next_event()
            except h11.RemoteProtocolError as exc:
                raise GatewayError("size_limit" if exc.error_status_hint == 431 else "response_invalid") from None
            if event is h11.PAUSED or isinstance(event, h11.ConnectionClosed):
                raise GatewayError("response_invalid")
            if event is not h11.NEED_DATA:
                return event
            if reading_head:
                remaining = MAX_HEADER_BYTES - (self.reader.total - self.header_start)
                if remaining <= 0:
                    raise GatewayError("size_limit")
                chunk = self.reader.receive(self.sock, self.scope, remaining)
            else:
                chunk = self.reader.receive(self.sock, self.scope)
            self.connection.receive_data(chunk)

    def head(self) -> h11.Response:
        self.header_start = self.reader.total
        while True:
            event = self.next_event(reading_head=True)
            if isinstance(event, h11.Response):
                return event
            if not isinstance(event, h11.InformationalResponse) or event.status_code == 101:
                raise GatewayError("response_invalid")

    def body(self, encoding: str, limit: int) -> bytes:
        if encoding not in {"gzip", "", "identity"}:
            raise GatewayError("content_type_denied")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
        body = bytearray()
        while True:
            event = self.next_event()
            if isinstance(event, h11.EndOfMessage):
                if decoder is not None and not decoder.eof:
                    raise GatewayError("response_invalid")
                return bytes(body)
            if not isinstance(event, h11.Data):
                raise GatewayError("response_invalid")
            remaining = limit - len(body)
            data = bytes(event.data)
            if self.reader.quota is not None:
                body.extend(self.reader.quota.decode(data, remaining, decoder))
            else:
                body.extend(_decode_chunk(data, remaining, decoder))


@dataclass(frozen=True)
class _Hop:
    content_type: str
    body: bytes
    status: int
    headers: tuple[tuple[str, str], ...]
    location: str = ""


def _header_value(headers: tuple[tuple[str, str], ...], name: str) -> str:
    values = [value for key, value in headers if key == name]
    if len(values) > 1:
        raise GatewayError("response_invalid")
    return values[0] if values else ""


def _exchange(
    sock: socket.socket,
    target: _Target,
    method: str,
    scope: FetchScope,
    reader: _WireReader,
    allowed_types: frozenset[str],
    body_limit: int,
    content_limits: Mapping[str, int],
) -> _Hop:
    connection = h11.Connection(h11.CLIENT, max_incomplete_event_size=MAX_HEADER_BYTES)
    request = h11.Request(
        method=method.encode("ascii"),
        target=target.request_target.encode("ascii"),
        headers=[
            (b"Host", target.authority.encode("ascii")),
            (b"User-Agent", b"DeepResearch"),
            (b"Accept", b"*/*"),
            (b"Accept-Encoding", b"gzip, identity"),
            (b"Connection", b"close"),
        ],
    )
    _send(sock, connection.send(request) + connection.send(h11.EndOfMessage()), scope)
    response = _HTTPResponse(sock, scope, reader, connection)
    head = response.head()
    headers = tuple((key.decode("ascii"), value.decode("latin-1")) for key, value in head.headers)
    status = head.status_code
    disposition = _header_value(headers, "content-disposition").split(";", 1)[0].strip().lower()
    if disposition and disposition != "inline":
        raise GatewayError("download_blocked")
    if _header_value(headers, "upgrade") or status == 206:
        raise GatewayError("response_invalid")
    if status in _REDIRECTS:
        location = _header_value(headers, "location")
        if not location:
            raise GatewayError("redirect_denied")
        return _Hop("", b"", status, (), location)
    if status in {401, 403, 407}:
        raise GatewayError("access_denied")
    if not 200 <= status < 300:
        raise GatewayError("response_invalid")
    content_type = _header_value(headers, "content-type")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in allowed_types:
        raise GatewayError("content_type_denied")
    encoding = _header_value(headers, "content-encoding").strip().lower()
    body_limit = min(body_limit, content_limits.get(media_type, body_limit))
    length = _header_value(headers, "content-length")
    if length and _header_value(headers, "transfer-encoding"):
        raise GatewayError("response_invalid")
    if length and method != "HEAD":
        if len(length) > 15 or int(length) > reader.limit - reader.total + len(connection.trailing_data[0]):
            raise GatewayError("size_limit")
        if encoding in {"", "identity"} and int(length) > body_limit:
            raise GatewayError("size_limit")
    body = b"" if method == "HEAD" else response.body(encoding, body_limit)
    return _Hop(content_type, body, status, (("content-type", content_type),))


class _ConnectFailure(Exception):
    """A pre-request connection error for which a pinned fallback is permitted."""


def _attempt_scope(scope: FetchScope, attempt: object) -> FetchScope:
    remaining = getattr(attempt, "timeout", None)
    if isinstance(remaining, (int, float)):
        scope = scope.bounded(remaining)
    deadline = getattr(attempt, "deadline", None)
    if isinstance(deadline, (int, float)):
        scope = replace(scope, deadline=min(scope.deadline, deadline))
    return scope


def _require_authorized(target: _Target, authorize: Callable[[str], bool] | None) -> None:
    if authorize is not None and not authorize(target.url):
        raise GatewayError("origin_not_allowed")


class WebFetchGateway:
    def __init__(
        self,
        *,
        resolver: Callable[[str, FetchScope], Iterable[str]] = _resolve_dns,
        connect: Callable[[_Target, str, FetchScope], AbstractContextManager[socket.socket]] = _open_pinned,
        denied_networks: Iterable[str] = (),
    ):
        self._resolver = resolver
        self._connect = connect
        self._denied_networks = tuple(ipaddress.ip_network(network) for network in denied_networks)

    def _resolve(self, target: _Target, scope: FetchScope) -> tuple[str, ...]:
        try:
            ipaddress.ip_address(target.host)
        except ValueError:
            try:
                # Bound the number of values consumed even from an injected resolver.
                resolved = tuple(islice(self._resolver(target.host, scope), _DNS_MAX_ADDRESSES + 1))
            except (OSError, ValueError):
                raise GatewayError("dns_denied") from None
        else:
            resolved = (target.host,)
        scope.check()
        if not resolved or len(resolved) > _DNS_MAX_ADDRESSES:
            raise GatewayError("dns_denied")
        addresses: list[str] = []
        for candidate in resolved:
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                raise GatewayError("dns_denied") from None
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                classification = address.ipv4_mapped
            else:
                classification = address
            if not _public_ip(address) or any(
                address in network or classification in network for network in self._denied_networks
            ):
                raise GatewayError("dns_denied")
            if str(address) not in addresses:
                addresses.append(str(address))
        return tuple(addresses)

    @contextmanager
    def _open_attempt(
        self,
        target: _Target,
        address: str,
        scope: FetchScope,
        authorize: Callable[[str], bool] | None,
        quota: FetchQuota | None,
        count_request: bool,
    ) -> Iterator[tuple[socket.socket, FetchScope]]:
        scope.check()
        if quota is not None and quota.remaining_wire() == 0:
            raise GatewayError("size_limit")
        lease_request = quota.request_slot() if quota is not None and count_request else nullcontext(lambda: None)
        with lease_request as mark_request:
            try:
                with scope.attempt({"web_fetches": 1}) as attempt:
                    physical_scope = _attempt_scope(scope, attempt)
                    physical_scope.check()
                    _require_authorized(target, authorize)
                    physical_scope.check()
                    mark_request()
                    kind = "subresource" if count_request else "static" if scope.adapter == "static" else "document"
                    metrics_registry.record_web_extraction("http_attempt", adapter=scope.adapter, kind=kind)
                    started = scope.monotonic()
                    connected = False
                    try:
                        with self._connect(target, address, physical_scope) as sock:
                            connected = True
                            _check_peer(sock, address, target.port)
                            yield sock, physical_scope
                    except ssl.SSLError:
                        raise GatewayError("network_error") from None
                    except OSError as exc:
                        physical_scope.check()
                        if not connected:
                            raise _ConnectFailure() from exc
                        raise GatewayError("network_error") from None
                    finally:
                        metrics_registry.record_web_extraction(
                            "http_elapsed", adapter=scope.adapter, elapsed_ms=(scope.monotonic() - started) * 1000,
                        )
            except GatewayError:
                raise
            except Exception as exc:
                if getattr(exc, "budget_denied", False):
                    scope.check()
                    raise GatewayError("budget_denied") from None
                raise

    def fetch(
        self,
        url: str,
        *,
        scope: FetchScope,
        method: str = "GET",
        authorize: Callable[[str], bool] | None = None,
        quota: FetchQuota | None = None,
        count_request: bool = False,
        allowed_types: Iterable[str] | None = None,
        max_wire_bytes: int = MAX_STATIC_BYTES,
        max_body_bytes: int = MAX_STATIC_BYTES,
        content_limits: Mapping[str, int] | None = None,
    ) -> FetchedDocument:
        """Open an authorized public URL and return a complete bounded response."""
        try:
            if method not in {"GET", "HEAD"}:
                raise GatewayError("access_denied")
            return self._fetch(
                url, scope, method, authorize, quota, count_request,
                _DEFAULT_TYPES if allowed_types is None else frozenset(allowed_types),
                min(max_wire_bytes, MAX_DYNAMIC_BYTES if quota is not None else MAX_STATIC_BYTES),
                min(max_body_bytes, MAX_DYNAMIC_BYTES if quota is not None else MAX_STATIC_BYTES),
                content_limits or {},
            )
        except GatewayError as exc:
            if exc.reason in _DENIED_REASONS:
                metrics_registry.record_web_extraction("egress_denied", reason=exc.reason)
            raise

    def _fetch(
        self,
        url: str,
        scope: FetchScope,
        method: str,
        authorize: Callable[[str], bool] | None,
        quota: FetchQuota | None,
        count_request: bool,
        types: frozenset[str],
        wire_limit: int,
        body_limit: int,
        content_limits: Mapping[str, int],
    ) -> FetchedDocument:
        scope.check()
        if wire_limit <= 0 or body_limit <= 0:
            raise GatewayError("size_limit")
        target = _target(url)
        seen = {target.url}
        redirects = []
        reader = _WireReader(wire_limit, quota)
        for _ in range(4):
            scope.check()
            _require_authorized(target, authorize)
            addresses = self._resolve(target, scope)
            for index, address in enumerate(addresses[:2]):
                try:
                    with self._open_attempt(target, address, scope, authorize, quota, count_request) as (sock, physical_scope):
                        hop = _exchange(sock, target, method, physical_scope, reader, types, body_limit, content_limits)
                        physical_scope.check()
                    break
                except _ConnectFailure:
                    if index == 1 or len(addresses) == 1:
                        raise GatewayError("network_error") from None
            scope.check()
            if not hop.location:
                return FetchedDocument(
                    target.url, hop.content_type, hop.body, hop.status, reader.total, hop.headers, tuple(redirects),
                )
            if len(redirects) == 3:
                raise GatewayError("redirect_denied")
            target = _redirect_target(target, hop.location)
            if target.url in seen:
                raise GatewayError("redirect_denied")
            seen.add(target.url)
            redirects.append(target.url)
        raise GatewayError("redirect_denied")
