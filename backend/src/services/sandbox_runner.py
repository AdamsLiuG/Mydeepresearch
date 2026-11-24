"""Standalone restricted Python execution entrypoint run inside the sandbox container."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import sys
from types import SimpleNamespace
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None  # type: ignore[assignment]


MODULE_EXPORTS = {
    "math": "ceil floor trunc sqrt cbrt log log2 log10 log1p exp expm1 pow sin cos tan asin acos atan atan2 hypot pi e tau inf nan isfinite isnan isinf fsum prod factorial gcd lcm comb perm degrees radians",
    "statistics": "mean fmean geometric_mean harmonic_mean median median_low median_high mode multimode quantiles stdev pstdev variance pvariance covariance correlation linear_regression",
    "decimal": "Decimal",
    "fractions": "Fraction",
    "datetime": "date datetime timedelta",
    "collections": "Counter defaultdict deque",
    "itertools": "accumulate chain combinations combinations_with_replacement compress groupby islice pairwise permutations product repeat zip_longest",
    "functools": "reduce",
    "re": "search match fullmatch findall finditer split sub escape",
    "json": "loads dumps",
    "numpy": "array asarray arange linspace zeros ones sum mean median average std var min max amin amax argmin argmax quantile percentile diff cumsum concatenate stack reshape transpose dot corrcoef cov polyfit sqrt log exp round isfinite where clip",
    "pandas": "DataFrame Series concat to_numeric",
    "scipy": "stats",
    "scipy.stats": "describe linregress pearsonr spearmanr ttest_ind ttest_rel iqr zscore",
}
MODULE_EXPORTS = {module: frozenset(names.split()) for module, names in MODULE_EXPORTS.items()}
ALLOWED_MODULES = frozenset(MODULE_EXPORTS)
ALLOWED_ATTRIBUTES = frozenset().union(*MODULE_EXPORTS.values()) | frozenset(
    """
    append extend insert pop remove clear copy count index reverse sort get items keys values
    update setdefault popitem add discard union intersection difference
    strip lstrip rstrip split rsplit splitlines join replace startswith endswith lower upper
    casefold find rfind partition rpartition isdigit isnumeric isalpha isalnum zfill
    group groups groupdict start end span
    real imag numerator denominator quantize normalize
    year month day days seconds microseconds total_seconds isoformat fromisoformat strptime strftime
    tolist to_dict item shape size ndim T astype flatten ravel column_stack
    columns loc iloc groupby agg head tail sort_values reset_index fillna dropna
    value_counts unique nunique min max sum mean median std var quantile round
    slope intercept rvalue pvalue stderr statistic statistic_location
    """.split()
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
MAX_CODE_CHARS = 16_000
MAX_AST_NODES = 2_000
MAX_INPUT_BYTES = 512 * 1024
MAX_OUTPUT_BYTES = 256 * 1024


def _validate_ast_safety(
    code: str,
    *,
    allowed_modules: frozenset[str] = ALLOWED_MODULES,
    banned_names: frozenset[str] = BANNED_NAMES,
    max_code_chars: int = MAX_CODE_CHARS,
    max_ast_nodes: int = MAX_AST_NODES,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(code, str) or not code.strip():
        return ["empty code payload"]
    if len(code) > max_code_chars:
        return [f"code exceeds {max_code_chars} chars"]
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    except (ValueError, RecursionError):
        return ["invalid or overly complex Python code"]

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > max_ast_nodes:
            errors.append(f"AST exceeds {max_ast_nodes} nodes")
            break
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_modules or alias.name not in MODULE_EXPORTS:
                    errors.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod not in allowed_modules or any(
                alias.name not in MODULE_EXPORTS.get(mod, ()) for alias in node.names
            ):
                errors.append(f"disallowed import-from: {node.module}")
        elif isinstance(node, ast.Attribute):
            if node.attr not in ALLOWED_ATTRIBUTES or not isinstance(node.ctx, ast.Load):
                errors.append(f"disallowed attribute: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in banned_names:
                errors.append(f"disallowed builtin or name: {node.id}")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
            errors.append(f"disallowed Python construct: {type(node).__name__}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if any(isinstance(part, ast.Pow) for part in ast.walk(node.right)) or (
                isinstance(node.right, ast.Constant)
                and type(node.right.value) in (int, float)
                and abs(node.right.value) > 1000
            ):
                errors.append("exponent exceeds sandbox limits")
    return errors


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    del globals_, locals_
    exports = MODULE_EXPORTS.get(name, ())
    if level or not exports or any(member not in exports for member in (fromlist or ())):
        raise ImportError(f"module '{name}' is not allowed in sandbox")

    def namespace(module_name: str) -> SimpleNamespace:
        module = __import__(module_name, fromlist=("__name__",))
        return SimpleNamespace(**{
            member: namespace(f"{module_name}.{member}")
            if f"{module_name}.{member}" in MODULE_EXPORTS else getattr(module, member)
            for member in MODULE_EXPORTS[module_name] if hasattr(module, member)
        })

    return namespace(name if fromlist else name.split(".", 1)[0])


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


class OutputLimitExceeded(Exception):
    pass


class _BoundedBuffer(io.StringIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.size = 0
        self.exceeded = False

    def write(self, value: str) -> int:
        self.size += len(value)
        if self.size > self.limit:
            self.exceeded = True
            raise OutputLimitExceeded("sandbox text output exceeds limit")
        return super().write(value)


def _json_safe(value: Any) -> Any:
    remaining = MAX_OUTPUT_BYTES

    def visit(item: Any, depth: int) -> Any:
        nonlocal remaining
        remaining -= 8
        if isinstance(item, str):
            remaining -= len(item)
        if depth > 12 or remaining < 0:
            raise OutputLimitExceeded("sandbox structured output exceeds limit")
        if item is None or isinstance(item, (bool, str)):
            return item
        if isinstance(item, int) and item.bit_length() <= 4096:
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {visit(str(key), depth + 1): visit(entry, depth + 1) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(entry, depth + 1) for entry in item]
        if isinstance(item, (set, frozenset)):
            return sorted((visit(entry, depth + 1) for entry in item), key=repr)
        if type(item).__module__.split(".")[0] in {"numpy", "pandas"}:
            if getattr(item, "size", 1) > 16000:
                raise OutputLimitExceeded("sandbox array output exceeds limit")
            if hasattr(item, "tolist"):
                return visit(item.tolist(), depth + 1)
            if hasattr(item, "to_dict"):
                return visit(item.to_dict(orient="records"), depth + 1)
        raise ValueError("sandbox output must be JSON-serializable")

    return visit(value, 0)


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
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


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
        if tbl.get("records"):
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

    stdout_buf = _BoundedBuffer(16000)
    stderr_buf = _BoundedBuffer(8000)
    try:
        compiled = compile(code, "<sandbox>", "exec")
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compiled, safe_globals, safe_globals)
        if stdout_buf.exceeded or stderr_buf.exceeded:
            raise OutputLimitExceeded("sandbox text output exceeds limit")
        output = _json_safe({
            "ok": True,
            "error": None,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result": safe_globals.get("result"),
            "tables_out": safe_globals.get("tables_out") or [],
            "charts": safe_globals.get("charts") or [],
        })
        if len(json.dumps(output, ensure_ascii=True)) > MAX_OUTPUT_BYTES:
            raise OutputLimitExceeded("sandbox JSON output exceeds byte limit")
        return output
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "error_code": (
                "output_limit" if isinstance(exc, OutputLimitExceeded)
                else "oom" if isinstance(exc, MemoryError)
                else "runtime_error"
            ),
            "stdout": "",
            "stderr": "",
            "result": None,
            "tables_out": [],
            "charts": [],
        }


def main() -> int:
    raw_input = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    try:
        if len(raw_input) > MAX_INPUT_BYTES:
            raise ValueError("sandbox input exceeds byte limit")
        payload = json.loads(raw_input) if raw_input.strip() else {}
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"invalid input JSON: {exc}"}) + "\n")
        return 1

    output = execute_sandbox_payload(
        payload if isinstance(payload, dict) else {},
        apply_rlimit=True,
    )
    sys.stdout.write(json.dumps(output, ensure_ascii=True))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
