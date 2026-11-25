"""Helpers for coordinating note tool usage instructions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from hello_agents.tools.builtin.note_tool import NoteTool

from models import MemoryPurpose, TodoItem
from services.memory_scope import (
    MemoryPolicy,
    MemoryScope,
    authorize,
    checked_path,
    partition_lock,
    partition_root,
    read_partition_text,
    remove_partition_file,
    scope_is_current,
    strict_memory_text,
    valid_internal_id,
    write_partition_text,
)


def build_note_guidance(task: TodoItem) -> str:
    """Generate note tool usage guidance for a specific task."""
    tags_list = ["deep_research", f"task_{task.id}"]
    tags_literal = json.dumps(tags_list, ensure_ascii=False)

    if task.note_id:
        read_payload = json.dumps({"action": "read", "note_id": task.note_id}, ensure_ascii=False)
        update_payload = json.dumps(
            {
                "action": "update",
                "note_id": task.note_id,
                "task_id": task.id,
                "title": f"任务 {task.id}: {task.title}",
                "note_type": "task_state",
                "tags": tags_list,
                "content": "请将本轮新增信息补充到任务概览中",
            },
            ensure_ascii=False,
        )

        return (
            "笔记协作指引：\n"
            f"- 当前任务笔记 ID：{task.note_id}。\n"
            f"- 在书写总结前必须调用：[TOOL_CALL:note:{read_payload}] 获取最新内容。\n"
            f"- 完成分析后调用：[TOOL_CALL:note:{update_payload}] 同步增量信息。\n"
            "- 更新时保持原有段落结构，新增内容请在对应段落中补充。\n"
            f"- 建议 tags 保持为 {tags_literal}，保证其他 Agent 可快速定位。\n"
            "- 成功同步到笔记后，再输出面向用户的总结。\n"
        )

    create_payload = json.dumps(
        {
            "action": "create",
            "task_id": task.id,
            "title": f"任务 {task.id}: {task.title}",
            "note_type": "task_state",
            "tags": tags_list,
            "content": "请记录任务概览、来源概览",
        },
        ensure_ascii=False,
    )

    return (
        "笔记协作指引：\n"
        f"- 当前任务尚未建立笔记，请先调用：[TOOL_CALL:note:{create_payload}]。\n"
        "- 创建成功后记录返回的 note_id，并在后续所有更新中复用。\n"
        "- 同步笔记后，再输出面向用户的总结。\n"
    )



class ScopedNoteWorkspace:
    """Tenant-bound note bodies with authoritative service-owned provenance."""

    def __init__(self, root: str | Path, scope: MemoryScope, policy: MemoryPolicy) -> None:
        self._root = root
        self.scope = scope
        self.policy = policy

    @property
    def root(self) -> Path:
        if not scope_is_current(self.policy, self.scope):
            raise PermissionError("note scope unavailable")
        return partition_root(self._root, self.scope)

    def read_record(
        self, note_id: str, *, purpose: MemoryPurpose = "task",
    ) -> tuple[str, dict[str, str]] | None:
        if not valid_internal_id(note_id):
            return None
        try:
            root = self.root
            raw = read_partition_text(root, "provenance", note_id + ".json", max_bytes=4096)
            provenance = json.loads(raw)
            if not isinstance(provenance, dict) or provenance.get("tenant_id") != self.scope.tenant_id:
                return None
            owner = provenance.get("request_id")
            digest = provenance.get("content_sha256")
            if (
                provenance.get("sanitizer_version") != 1 or not isinstance(digest, str)
                or not authorize(self.policy, self.scope, owner, purpose, object_id=note_id, content_version=digest)
            ):
                return None
            body = read_partition_text(root, note_id + ".md", max_bytes=2_000_000)
            if hashlib.sha256(body.encode()).hexdigest() != digest or strict_memory_text(body) is None:
                return None
            if raw == read_partition_text(root, "provenance", note_id + ".json", max_bytes=4096) and authorize(
                self.policy, self.scope, owner, purpose, object_id=note_id, content_version=digest,
            ):
                return body, provenance
        except (OSError, ValueError):
            pass
        return None

    def read(self, note_id: str, *, purpose: MemoryPurpose = "task") -> str | None:
        result = self.read_record(note_id, purpose=purpose)
        return result[0] if result else None

    def path(self, note_id: str, *, owned: bool = False) -> Path | None:
        record = self.read_record(note_id)
        if record is None or (owned and record[1]["request_id"] != self.scope.request_id):
            return None
        try:
            return checked_path(self.root, note_id + ".md")
        except (OSError, ValueError):
            return None

    def write(self, note_id: str, body: str, *, append: bool = False) -> bool:
        if not valid_internal_id(note_id) or strict_memory_text(body) is None:
            return False
        if not authorize(self.policy, self.scope, self.scope.request_id, "task"):
            return False
        try:
            root = partition_root(self._root, self.scope, create=True)
            with partition_lock(root):
                try:
                    read_partition_text(root, "provenance", note_id + ".json", max_bytes=4096)
                except FileNotFoundError:
                    if append or checked_path(root, note_id + ".md").exists():
                        return False
                else:
                    record = self.read_record(note_id)
                    if record is None or record[1]["request_id"] != self.scope.request_id:
                        return False
                    if append:
                        body = record[0] + "\n" + body
                if strict_memory_text(body) is None:
                    return False
                digest = hashlib.sha256(body.encode()).hexdigest()
                if not authorize(self.policy, self.scope, self.scope.request_id, "task"):
                    return False
                write_partition_text(root, note_id + ".md", text=body)
                write_partition_text(root, "provenance", note_id + ".json", text=json.dumps({
                    "tenant_id": self.scope.tenant_id, "request_id": self.scope.request_id,
                    "content_sha256": digest, "sanitizer_version": 1,
                }))
                return True
        except (OSError, ValueError):
            return False

    def delete(self, note_id: str) -> bool:
        if not valid_internal_id(note_id):
            return False
        try:
            root = self.root
            with partition_lock(root):
                if self.path(note_id, owned=True) is None:
                    return False
                remove_partition_file(root, "provenance", note_id + ".json")
                remove_partition_file(root, note_id + ".md")
                return True
        except (OSError, ValueError):
            return False

    def candidate_ids(self, *, after: str = "", limit: int = 64) -> list[str]:
        from services.memory_scope import open_partition_directory

        try:
            with open_partition_directory(self.root) as fd:
                import os

                return sorted(
                    name[:-3] for name in os.listdir(fd)
                    if name.endswith(".md") and name[:-3] > after and valid_internal_id(name[:-3])
                )[:max(0, min(limit, 64))]
        except (OSError, ValueError):
            return []


class ScopedNoteTool(NoteTool):
    """NoteTool parameter protocol backed by confined, scope-checked writes and reads."""

    def __init__(self, workspace: ScopedNoteWorkspace) -> None:
        # NoteTool initialization creates and reads its shared path-based index.
        try:
            super(NoteTool, self).__init__(name="note", description="创建、读取和管理研究笔记")
        except TypeError:
            super().__init__()
            self.name = "note"
            self.description = "创建、读取和管理研究笔记"
        self._scoped = workspace

    def run(self, parameters: dict) -> str:
        if not self.validate_parameters(parameters):
            return "❌ 参数验证失败"
        action = parameters["action"]
        note_id = parameters.get("note_id")
        if action in {"read", "delete", "update", "append"} and not valid_internal_id(note_id):
            return "❌ 笔记不可用"
        if action == "read":
            body = self._scoped.read(note_id)
            return body if body is not None else "❌ 笔记不可用"
        if action == "delete":
            return "✅ 笔记已删除" if self._scoped.delete(note_id) else "❌ 笔记不可用"
        if action == "append":
            ok = self._scoped.write(note_id, parameters.get("content"), append=True)
            return f"✅ 笔记更新成功: {note_id}" if ok else "❌ 笔记不可用"
        if action in {"create", "update"}:
            previous = {}
            if action == "update":
                if self._scoped.path(note_id, owned=True) is None:
                    return "❌ 笔记不可用"
                raw = self._scoped.read(note_id)
                if raw is None:
                    return "❌ 笔记不可用"
                previous = self._markdown_to_note(raw)
            else:
                note_id = "note_" + uuid4().hex
            title = parameters.get("title", previous.get("title"))
            content = parameters.get("content", previous.get("content"))
            note_type = parameters.get("note_type", previous.get("type", "general"))
            tags = parameters.get("tags", previous.get("tags", []))
            if (
                not isinstance(title, str) or not title.strip() or len(title.splitlines()) != 1
                or not isinstance(content, str) or not content.strip()
                or note_type not in ("task_state", "conclusion", "blocker", "action", "reference", "general")
                or not isinstance(tags, list) or any(strict_memory_text(tag) is None for tag in tags)
            ):
                return "❌ 笔记不可用"
            now = datetime.now(timezone.utc).isoformat()
            markdown = self._note_to_markdown({
                "id": note_id, "title": title, "type": note_type, "tags": tags, "content": content,
                "created_at": previous.get("created_at") or now, "updated_at": now,
            })
            if self._scoped.write(note_id, markdown):
                return f"✅ 笔记保存成功\nID: {note_id}"
            return "❌ 笔记不可用"
        if action in {"list", "search", "summary"}:
            limit = parameters.get("limit", 10)
            query = str(parameters.get("query") or "").casefold()
            note_type = parameters.get("note_type")
            items = []
            for candidate in self._scoped.candidate_ids():
                body = self._scoped.read(candidate)
                if body is None or (action == "search" and query not in body.casefold()):
                    continue
                note = self._markdown_to_note(body)
                if note_type and note.get("type") != note_type:
                    continue
                items.append(f"{note.get('title', candidate)}\nID: {candidate}")
            if action == "summary":
                return f"📝 笔记共 {len(items)} 条"
            return "\n\n".join(items[:max(0, min(64, limit))]) or "📝 暂无笔记"
        return "❌ 不支持的操作"
