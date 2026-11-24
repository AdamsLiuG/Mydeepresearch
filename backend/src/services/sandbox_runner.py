"""Standalone restricted Python execution entrypoint run inside the sandbox container."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import sys
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None  # type: ignore[assignment]


ALLOWED_MODULES = frozenset(
    {
        "math",
        "statistics",
        "decimal",
        "fractions",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "re",
        "json",
        "numpy",
        "pandas",
        "scipy",
    }
)
BANNED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "memoryview",
        "__builtins__",
        "__loader__",
        "__spec__",
    }
)
MAX_CODE_CHARS = 20_000
MAX_AST_NODES = 2_000


def _validate_ast_safety(code: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(code, str) or not code.strip():
        return ["empty code payload"]
    if len(code) > MAX_CODE_CHARS:
        return [f"code exceeds {MAX_CODE_CHARS} chars"]
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > MAX_AST_NODES:
            errors.append(f"AST exceeds {MAX_AST_NODES} nodes")
            break
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_MODULES:
                    errors.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".", 1)[0]
            if node.level != 0 or mod not in ALLOWED_MODULES:
                errors.append(f"disallowed import-from: {node.module}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in {"_subclasses__"}:
                errors.append(f"disallowed dunder/private attribute: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                errors.append(f"disallowed builtin or name: {node.id}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_NAMES:
                errors.append(f"disallowed builtin call: {node.func.id}")
    return errors


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = str(name or "").split(".", 1)[0]
    if level != 0 or root not in ALLOWED_MODULES:
        raise ImportError(f"module '{name}' is not allowed in sandbox")
    return __import__(name, globals_, locals_, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    return {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "ZeroDivisionError": ZeroDivisionError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "ArithmeticError": ArithmeticError,
        "True": True,
        "False": False,
        "None": None,
    }


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 8:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth=_depth + 1) for k, v in list(value.items())[:200]}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth=_depth + 1) for v in list(value)[:500]]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(orient="records"), _depth=_depth + 1)
        except Exception:
            pass
    return str(value)


def _apply_resource_limits(timeout_seconds: float, memory_mb: int) -> None:
    if resource is None:
        return
    try:
        cpu_limit = max(1, int(math.ceil(timeout_seconds)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except Exception:
        pass
    try:
        mem_bytes = max(64, int(memory_mb)) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass


def execute_sandbox_payload(
    payload: dict[str, Any],
    *,
    apply_rlimit: bool = False,
) -> dict[str, Any]:
    """Validate and execute one JSON sandbox payload, returning structured output."""

    code = str(payload.get("code") or "")
    errors = _validate_ast_safety(code)
    if errors:
        return {
            "ok": False,
            "error": "; ".join(errors),
            "stdout": "",
            "stderr": "",
            "result": None,
            "tables_out": [],
            "charts": [],
        }

    if apply_rlimit:
        timeout_seconds = float(payload.get("timeout_seconds") or 10.0)
        memory_mb = int(payload.get("memory_mb") or 512)
        _apply_resource_limits(timeout_seconds, memory_mb)

    tables_dict: dict[str, list[dict[str, Any]]] = {}
    chart_bucket: list[dict[str, Any]] = []
    tables_out_bucket: list[dict[str, Any]] = []
    safe_globals: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "tables": tables_dict,
        "tables_out": tables_out_bucket,
        "charts": chart_bucket,
        "result": None,
    }

    for tbl in payload.get("tables") or []:
        if not isinstance(tbl, dict):
            continue
        name = str(tbl.get("name") or "").strip()
        if not name or not name.isidentifier() or name.startswith("_"):
            continue
        if isinstance(tbl.get("records"), list):
            records = [dict(r) for r in tbl["records"] if isinstance(r, dict)]
        else:
            headers = [str(h) for h in (tbl.get("columns") or tbl.get("headers") or [])]
            records = []
            for row in tbl.get("rows") or []:
                if isinstance(row, dict):
                    records.append(dict(row))
                elif isinstance(row, list):
                    records.append(
                        {
                            (headers[i] if i < len(headers) else f"col_{i + 1}"): val
                            for i, val in enumerate(row)
                        }
                    )
        tables_dict[name] = records
        if name not in safe_globals:
            safe_globals[name] = records

    def emit_chart(
        *,
        title: str = "",
        library: str = "vega-lite",
        spec: dict[str, Any] | None = None,
        caption: str = "",
        source_ids: list[str] | None = None,
    ) -> None:
        chart_bucket.append(
            {
                "chart_id": f"chart-{len(chart_bucket) + 1}",
                "title": str(title or ""),
                "library": str(library or "vega-lite"),
                "spec": dict(spec or {}),
                "caption": str(caption or ""),
                "source_ids": [str(s) for s in (source_ids or []) if s],
            }
        )

    safe_globals["emit_chart"] = emit_chart

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        compiled = compile(ast.parse(code, mode="exec"), "<sandbox>", "exec")
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compiled, safe_globals, safe_globals)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": stdout_buf.getvalue()[:16000],
            "stderr": stderr_buf.getvalue()[:8000],
            "result": None,
            "tables_out": _json_safe(safe_globals.get("tables_out") or []),
            "charts": _json_safe(safe_globals.get("charts") or []),
        }

    return {
        "ok": True,
        "error": None,
        "stdout": stdout_buf.getvalue()[:16000],
        "stderr": stderr_buf.getvalue()[:8000],
        "result": _json_safe(safe_globals.get("result")),
        "tables_out": _json_safe(safe_globals.get("tables_out") or []),
        "charts": _json_safe(safe_globals.get("charts") or []),
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args and os.path.isfile(args[0]):
        with open(args[0], encoding="utf-8") as handle:
            raw_input = handle.read()
    else:
        raw_input = sys.stdin.read()

    try:
        payload = json.loads(raw_input) if raw_input.strip() else {}
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"invalid input JSON: {exc}"}) + "\n")
        return 1

    output = execute_sandbox_payload(
        payload if isinstance(payload, dict) else {},
        apply_rlimit=True,
    )
    sys.stdout.write(json.dumps(output, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
