"""Trusted tenant scopes, live memory authorization and confined partition I/O."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from models import MemoryPurpose
from services.memory_sanitizer import (
    collapse_whitespace,
    sanitize_memory_text,
    sanitize_web_text,
)

POLICY_VERSION = 1
PURPOSES = frozenset({"plan", "task", "report", "strategy_synthesis", "maintenance"})
_INTERNAL_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_LAYOUT_MARKER = "tenant_partition_v1"
_MANIFEST = "tenant.json"


def valid_internal_id(value: object) -> bool:
    return isinstance(value, str) and bool(_INTERNAL_ID.fullmatch(value)) and ".." not in value


def tenant_namespace(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not tenant_id or not tenant_id.isprintable():
        raise ValueError("invalid tenant identity")
    return "t_" + hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryScope:
    tenant_id: str
    namespace: str
    request_id: str
    policy_version: int
    owner_epoch: int | None

    @classmethod
    def trusted(
        cls, tenant_id: str, request_id: str, *, policy_version: int = POLICY_VERSION,
        owner_epoch: int | None = None,
    ) -> MemoryScope:
        scope = cls(tenant_id, tenant_namespace(tenant_id), request_id, policy_version, owner_epoch)
        if not valid_scope(scope):
            raise ValueError("invalid memory scope")
        return scope

    def frozen_partition(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "tenant_id": self.tenant_id, "namespace": self.namespace,
            "policy_version": self.policy_version, "strict": True,
        }


def valid_scope(scope: MemoryScope | None) -> bool:
    if (
        not isinstance(scope, MemoryScope)
        or not valid_internal_id(scope.request_id)
        or type(scope.policy_version) is not int
        or scope.policy_version != POLICY_VERSION
        or (scope.owner_epoch is not None and (type(scope.owner_epoch) is not int or scope.owner_epoch < 1))
    ):
        return False
    try:
        return scope.namespace == tenant_namespace(scope.tenant_id)
    except ValueError:
        return False


class MemoryPolicy(Protocol):
    def scope_is_current(self, scope: MemoryScope) -> bool: ...

    def allows(
        self, scope: MemoryScope, origin_request_id: str, purpose: MemoryPurpose, *,
        object_id: str | None = None, content_version: str | None = None,
    ) -> bool: ...


@dataclass(frozen=True)
class MemoryGrant:
    """Current authoritative ownership and reuse eligibility for one object."""

    tenant_id: str
    origin_request_id: str
    object_id: str | None
    content_version: str | None
    purposes: frozenset[MemoryPurpose]
    shareable: bool
    active: bool


@dataclass(frozen=True)
class LiveMemoryPolicy:
    validate_owner: Callable[[MemoryScope], bool]
    load_grant: Callable[[MemoryScope, str, str | None], MemoryGrant | None]

    def scope_is_current(self, scope: MemoryScope) -> bool:
        return valid_scope(scope) and self.validate_owner(scope) is True

    def allows(
        self, scope: MemoryScope, origin_request_id: str, purpose: MemoryPurpose, *,
        object_id: str | None = None, content_version: str | None = None,
    ) -> bool:
        if not self.scope_is_current(scope):
            return False
        grant = self.load_grant(scope, origin_request_id, object_id)
        return bool(
            grant and grant.active is True and grant.tenant_id == scope.tenant_id
            and grant.origin_request_id == origin_request_id and grant.object_id == object_id
            and grant.content_version == content_version and purpose in grant.purposes
            and (origin_request_id == scope.request_id or grant.shareable is True)
        )


def scope_is_current(policy: MemoryPolicy | None, scope: MemoryScope | None) -> bool:
    if policy is None or not valid_scope(scope):
        return False
    try:
        check = getattr(policy, "scope_is_current", None)
        if check is not None:
            return check(scope) is True
        return policy.allows(scope, scope.request_id, "plan", object_id=None, content_version=None) is True
    except Exception:
        return False


def authorize(
    policy: MemoryPolicy | None, scope: MemoryScope | None, origin_request_id: str, purpose: MemoryPurpose, *,
    object_id: str | None = None, content_version: str | None = None,
) -> bool:
    if purpose not in PURPOSES or not valid_internal_id(origin_request_id) or not scope_is_current(policy, scope):
        return False
    if object_id is not None and not valid_internal_id(object_id):
        return False
    try:
        return policy.allows(
            scope, origin_request_id, purpose, object_id=object_id, content_version=content_version,
        ) is True
    except Exception:
        return False


def strict_memory_text(value: Any) -> str | None:
    """Preserve clean text and reject content requiring injection removal."""
    if not isinstance(value, str) or len(value) > 2_000_000:
        return None
    dropped = []
    sanitized = sanitize_web_text(value, max_chars=max(1, len(value)), on_drop=lambda: dropped.append(True))
    if (
        dropped or collapse_whitespace(value) != collapse_whitespace(sanitized)
        or sanitize_memory_text(value) != collapse_whitespace(value)
    ):
        return None
    return value


def _verify_components(parts: tuple[str, ...]) -> None:
    if any(not valid_internal_id(part) for part in parts):
        raise ValueError("invalid partition component")


def _resolved_root(root: str | Path) -> Path:
    raw = Path(root)
    if ".." in raw.parts:
        raise ValueError("invalid partition root")
    return raw.absolute()


@contextmanager
def open_partition_directory(
    root: str | Path, *parts: str, create: bool = False,
) -> Iterator[int]:
    """Walk and pin each directory from the filesystem root without following links."""
    _verify_components(parts)
    path = _resolved_root(root)
    fd = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in (*path.parts[1:], *parts):
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def checked_path(root: str | Path, *parts: str) -> Path:
    """Validate a path for libraries requiring filenames. Confined I/O uses directory handles."""
    _verify_components(parts)
    path = _resolved_root(root)
    final = path.joinpath(*parts)
    for component in (*reversed(final.parents), final):
        try:
            info = component.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise PermissionError("partition symlink")
        if component == final:
            if not (stat.S_ISDIR(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink == 1)):
                raise PermissionError("invalid partition item")
        elif not stat.S_ISDIR(info.st_mode):
            raise PermissionError("invalid partition directory")
    return final


def _checked_regular(fd: int) -> os.stat_result:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionError("invalid partition file")
    return info


def read_partition_text(root: str | Path, *parts: str, max_bytes: int = 8_388_608) -> str:
    _verify_components(parts)
    with open_partition_directory(root, *parts[:-1]) as parent:
        fd = os.open(parts[-1], os.O_RDONLY | _FILE_FLAGS, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            info = _checked_regular(stream.fileno())
            if info.st_size > max_bytes:
                raise ValueError("partition file too large")
            data = stream.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("partition file too large")
            _checked_regular(stream.fileno())
            return data.decode("utf-8")


def _check_leaf(parent: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionError("invalid partition file")


def write_partition_text(root: str | Path, *parts: str, text: str) -> None:
    _verify_components(parts)
    with open_partition_directory(root, *parts[:-1], create=True) as parent:
        name = parts[-1]
        _check_leaf(parent, name)
        temp = "staging-" + uuid4().hex
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_FLAGS, 0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            _check_leaf(parent, name)
            os.replace(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        finally:
            try:
                os.unlink(temp, dir_fd=parent)
            except FileNotFoundError:
                pass


def remove_partition_file(root: str | Path, *parts: str) -> None:
    _verify_components(parts)
    with open_partition_directory(root, *parts[:-1]) as parent:
        _check_leaf(parent, parts[-1])
        os.unlink(parts[-1], dir_fd=parent)
        os.fsync(parent)


@contextmanager
def partition_lock(root: str | Path) -> Iterator[None]:
    """Serialize cooperating writers across threads and processes."""
    with open_partition_directory(root, create=True) as parent:
        fd = os.open("partition.lock", os.O_RDWR | os.O_CREAT | _FILE_FLAGS, 0o600, dir_fd=parent)
        try:
            _checked_regular(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)


def strict_partition_required(
    config: Any, root: str | Path, scope: MemoryScope | None = None, *, frozen_strict: bool = False,
) -> bool:
    if frozen_strict or getattr(config, "v4_tenant_memory_partition_enabled", False):
        return True
    if not root:
        return False
    try:
        with open_partition_directory(root, "v4", "tenants") as directory:
            if scope is not None and valid_scope(scope):
                try:
                    os.stat(scope.namespace, dir_fd=directory, follow_symlinks=False)
                    return True
                except FileNotFoundError:
                    pass
            os.stat(_LAYOUT_MARKER, dir_fd=directory, follow_symlinks=False)
            return True
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True


def partition_root(
    root: str | Path, scope: MemoryScope, *, create: bool = False, version: str = "v4",
) -> Path:
    """Resolve a tenant directory and verify its authoritative identity manifest."""
    if not root or not valid_scope(scope):
        raise PermissionError("missing tenant scope")
    base = checked_path(root, version, "tenants")
    tenant_dir = checked_path(base, scope.namespace)
    expected = {"schema_version": 1, "tenant_id": scope.tenant_id, "namespace": scope.namespace}
    if create:
        with partition_lock(base):
            try:
                read_partition_text(base, _LAYOUT_MARKER, max_bytes=1024)
            except FileNotFoundError:
                write_partition_text(base, _LAYOUT_MARKER, text="1")
            if not tenant_dir.exists():
                with open_partition_directory(base, scope.namespace, create=True):
                    write_partition_text(tenant_dir, _MANIFEST, text=json.dumps(expected, ensure_ascii=False))
    try:
        payload = json.loads(read_partition_text(tenant_dir, _MANIFEST, max_bytes=16384))
    except (OSError, ValueError) as error:
        raise PermissionError("partition identity unavailable") from error
    if payload != expected:
        raise PermissionError("partition identity mismatch")
    return tenant_dir
