"""Base protocol for benchmark judges."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from evals.schema import BenchmarkCase


class Judge(Protocol):
    """Protocol for pluggable benchmark judges."""

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        report_markdown: str,
        todo_items: Sequence[Any],
        trace_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Return deterministic metrics for a benchmark case."""
