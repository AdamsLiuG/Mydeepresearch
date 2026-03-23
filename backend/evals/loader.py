"""Load benchmark cases from JSON or JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from evals.schema import BenchmarkCase


def _read_json_payload(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("cases", "benchmarks", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON benchmark shape in {path}")


def _read_jsonl_payload(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL object at {path}:{line_number}")
        rows.append(payload)
    return rows


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    """Load benchmark cases from a JSON or JSONL file."""

    benchmark_path = Path(path).expanduser().resolve()
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    suffix = benchmark_path.suffix.lower()
    if suffix == ".json":
        payloads = _read_json_payload(benchmark_path)
    elif suffix in {".jsonl", ".ndjson"}:
        payloads = _read_jsonl_payload(benchmark_path)
    else:
        raise ValueError("Benchmark file must be .json, .jsonl, or .ndjson")

    return [BenchmarkCase.from_dict(payload) for payload in payloads]
