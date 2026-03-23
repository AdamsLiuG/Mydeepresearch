"""Schema definitions for benchmark cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _normalize_list(values: Any, *, field_name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of strings")

    normalized: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


@dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark case used by the offline eval runner."""

    id: str
    topic: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_sections: list[str] = field(default_factory=list)
    freshness_sensitive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkCase":
        """Create a benchmark case from a plain dictionary."""

        if not isinstance(payload, dict):
            raise ValueError("benchmark payload must be an object")

        case_id = str(payload.get("id") or "").strip()
        topic = str(payload.get("topic") or "").strip()
        if not case_id:
            raise ValueError("benchmark case is missing required field: id")
        if not topic:
            raise ValueError(f"benchmark case {case_id!r} is missing required field: topic")

        reserved_keys = {
            "id",
            "topic",
            "expected_keywords",
            "expected_sections",
            "freshness_sensitive",
        }
        metadata = {
            str(key): value
            for key, value in payload.items()
            if key not in reserved_keys
        }

        return cls(
            id=case_id,
            topic=topic,
            expected_keywords=_normalize_list(
                payload.get("expected_keywords"),
                field_name="expected_keywords",
            ),
            expected_sections=_normalize_list(
                payload.get("expected_sections"),
                field_name="expected_sections",
            ),
            freshness_sensitive=_normalize_bool(payload.get("freshness_sensitive")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the benchmark case to a serializable dictionary."""

        return asdict(self)
