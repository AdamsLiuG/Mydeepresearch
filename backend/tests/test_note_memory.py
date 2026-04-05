import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from config import Configuration
from metrics import RequestTrace, metrics_registry
from services import note_memory as note_memory_module
from services.note_memory import NoteMemoryService, NoteProvenanceResolver


def keyword_embedding(text: str) -> list[float]:
    normalized = str(text or "").lower()
    tokens = [
        "mcp",
        "protocol",
        "overview",
        "architecture",
        "component",
        "agent",
        "memory",
        "benchmark",
        "task",
        "report",
    ]
    vector = [float(normalized.count(token)) for token in tokens]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [1.0] + [0.0 for _ in range(len(tokens) - 1)]
    return [value / norm for value in vector]


class FakeCollection:
    def __init__(self) -> None:
        self.records = {}

    def upsert(self, *, ids, documents, metadatas, embeddings) -> None:
        for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.records[chunk_id] = {
                "id": chunk_id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def delete(self, *, ids) -> None:
        for chunk_id in ids:
            self.records.pop(chunk_id, None)

    def query(self, *, query_embeddings, n_results, include):
        query_embedding = query_embeddings[0]
        ranked = []
        for record in self.records.values():
            embedding = record["embedding"]
            dot = sum(left * right for left, right in zip(query_embedding, embedding))
            left_norm = math.sqrt(sum(value * value for value in query_embedding))
            right_norm = math.sqrt(sum(value * value for value in embedding))
            similarity = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
            ranked.append((1.0 - similarity, record))
        ranked.sort(key=lambda item: item[0])
        top = ranked[:n_results]
        return {
            "documents": [[item[1]["document"] for item in top]],
            "distances": [[item[0] for item in top]],
            "metadatas": [[item[1]["metadata"] for item in top]],
        }


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self.collections:
            self.collections[name] = FakeCollection()
        return self.collections[name]

    def delete_collection(self, name) -> None:
        self.collections.pop(name, None)


class NoteMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_registry.reset()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.notes_dir = self.root / "notes"
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir = self.root / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.root / "memory"
        self.config = Configuration.from_env(
            overrides={
                "enable_notes": True,
                "notes_workspace": str(self.notes_dir),
                "note_memory_enabled": True,
                "note_memory_dir": str(self.memory_dir),
                "request_state_enabled": True,
                "request_state_dir": str(self.requests_dir),
            },
            load_env_file=False,
        )
        self.observer = RequestTrace(
            request_id="req-current",
            topic="MCP protocol",
            search_api="duckduckgo",
            provider="custom",
            model="demo-model",
            pricing_catalog={},
        )
        self.embedding_patches = [
            patch.object(note_memory_module, "embeddings_available", return_value=True),
            patch.object(note_memory_module, "encode_text", side_effect=lambda text, **_: keyword_embedding(text)),
            patch.object(
                note_memory_module,
                "encode_texts",
                side_effect=lambda texts, **_: [keyword_embedding(text) for text in texts],
            ),
        ]
        for active_patch in self.embedding_patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.embedding_patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def _write_note(self, note_id: str, *, title: str, note_type: str, body: str) -> Path:
        path = self.notes_dir / f"{note_id}.md"
        path.write_text(
            "\n".join(
                [
                    "---",
                    f"id: {note_id}",
                    f"title: {title}",
                    f"type: {note_type}",
                    'tags: ["deep_research"]',
                    "created_at: 2026-04-01T12:00:00",
                    "updated_at: 2026-04-01T12:30:00",
                    "---",
                    "",
                    body.strip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def _write_snapshot(
        self,
        request_id: str,
        *,
        topic: str,
        status: str,
        todo_items: list[dict] | None = None,
        report_note_id: str | None = None,
    ) -> None:
        path = self.requests_dir / f"{request_id}.json"
        path.write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "topic": topic,
                    "status": status,
                    "phase": "completed",
                    "updated_at": "2026-04-01T13:00:00+00:00",
                    "todo_items": todo_items or [],
                    "report_note_id": report_note_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_provenance_resolver_and_note_parser_extract_metadata(self):
        self._write_note(
            "note_task_1",
            title="Task note",
            note_type="task_state",
            body="# Task note\n\n## Key Findings\nMCP architecture components and transport model.",
        )
        self._write_snapshot(
            "req-historical",
            topic="MCP protocol overview",
            status="success",
            todo_items=[{"id": 1, "note_id": "note_task_1"}],
        )

        resolver = NoteProvenanceResolver(str(self.requests_dir))
        provenance = resolver.resolve()
        service = NoteMemoryService(self.config, client=FakeChromaClient())
        document = service._parse_note_file(self.notes_dir / "note_task_1.md")

        self.assertIsNotNone(document)
        self.assertEqual(document.note_type, "task_state")
        self.assertEqual(provenance["note_task_1"]["topic"], "MCP protocol overview")
        self.assertEqual(provenance["note_task_1"]["request_status"], "success")
        self.assertEqual(provenance["note_task_1"]["task_id"], 1)

    def test_search_biases_conclusion_for_planning_and_task_state_for_execution(self):
        self._write_note(
            "note_conclusion",
            title="Historical conclusion",
            note_type="conclusion",
            body="# Historical conclusion\n\n## Overview\nMCP protocol overview, why it matters, and common integration patterns.",
        )
        self._write_note(
            "note_task",
            title="Historical task",
            note_type="task_state",
            body="# Historical task\n\n## Key Findings\nTask note about MCP architecture components and client server roles.",
        )
        self._write_snapshot(
            "req-success",
            topic="MCP protocol overview",
            status="success",
            todo_items=[{"id": 2, "note_id": "note_task"}],
            report_note_id="note_conclusion",
        )

        service = NoteMemoryService(self.config, client=FakeChromaClient())

        planning_context = service.search_for_planning(
            "MCP protocol overview",
            current_request_id="req-current",
            observer=self.observer,
        )
        execution_context = service.search_for_task(
            "MCP protocol overview",
            "architecture",
            "components and roles",
            current_request_id="req-current",
            observer=self.observer,
        )

        self.assertIn("note_type：conclusion", planning_context)
        self.assertIn("note_type：task_state", execution_context)
        snapshot = self.observer.snapshot()
        self.assertEqual(snapshot["note_memory_queries"], 2)
        self.assertGreaterEqual(snapshot["note_memory_hits"], 2)
        self.assertEqual(snapshot["note_memory_prompt_injections"], 2)

    def test_search_excludes_current_request_and_failed_history(self):
        self._write_note(
            "note_success",
            title="Reusable success note",
            note_type="conclusion",
            body="# Conclusion\n\n## Overview\nMCP protocol overview and architecture guidance.",
        )
        self._write_note(
            "note_current",
            title="Current request note",
            note_type="conclusion",
            body="# Conclusion\n\n## Overview\nCurrent request should be excluded from retrieval.",
        )
        self._write_note(
            "note_failed",
            title="Failed request note",
            note_type="conclusion",
            body="# Conclusion\n\n## Overview\nFailed retrieval should not be preferred in v1.",
        )
        self._write_snapshot(
            "req-success",
            topic="MCP protocol overview",
            status="success",
            report_note_id="note_success",
        )
        self._write_snapshot(
            "req-current",
            topic="MCP protocol overview",
            status="partial_success",
            report_note_id="note_current",
        )
        self._write_snapshot(
            "req-failed",
            topic="MCP protocol overview",
            status="failed",
            report_note_id="note_failed",
        )

        service = NoteMemoryService(self.config, client=FakeChromaClient())
        context = service.search_for_planning(
            "MCP protocol overview",
            current_request_id="req-current",
            observer=self.observer,
        )

        self.assertIn("Reusable success note", context)
        self.assertNotIn("Current request note", context)
        self.assertNotIn("Failed request note", context)

    def test_refresh_notes_reindexes_updated_content(self):
        self._write_note(
            "note_refresh",
            title="Refresh me",
            note_type="task_state",
            body="# Task\n\n## Key Findings\nMCP architecture basics.",
        )
        self._write_snapshot(
            "req-refresh",
            topic="MCP architecture",
            status="success",
            todo_items=[{"id": 1, "note_id": "note_refresh"}],
        )

        service = NoteMemoryService(self.config, client=FakeChromaClient())
        first = service.search_for_task(
            "MCP architecture",
            "components",
            "roles",
            current_request_id="req-current",
            observer=self.observer,
        )

        self._write_note(
            "note_refresh",
            title="Refresh me",
            note_type="task_state",
            body="# Task\n\n## Key Findings\nMCP architecture components updated with transport details.",
        )
        service.refresh_notes(["note_refresh"], observer=self.observer)
        second = service.search_for_task(
            "MCP architecture transport",
            "components",
            "roles",
            current_request_id="req-current",
            observer=self.observer,
        )

        self.assertIn("MCP architecture basics", first)
        self.assertIn("transport details", second)

    def test_ensure_reconciled_reuses_persisted_index_without_full_scan(self):
        manifest_path = self.memory_dir / "manifest.json"
        collection_dir = self.memory_dir / "chromadb"
        collection_dir.mkdir(parents=True, exist_ok=True)
        (collection_dir / "chroma.sqlite3").write_text("placeholder", encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "embedding_model": self.config.resolved_note_memory_embedding_model(),
                    "chunking_version": 1,
                    "notes": {
                        "note_cached": {
                            "checksum": "abc",
                            "updated_at": "2026-04-01T12:30:00",
                            "chunk_ids": ["note_cached::chunk::1"],
                            "last_indexed_at": "2026-04-01T13:00:00+00:00",
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        service = NoteMemoryService(self.config, client=FakeChromaClient())
        with patch.object(service, "_reconcile_locked") as mock_reconcile:
            service.ensure_reconciled(observer=self.observer)

        self.assertTrue(service._reconciled)
        self.assertIsNotNone(service._collection)
        mock_reconcile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
