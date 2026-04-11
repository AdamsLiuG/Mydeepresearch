"""LLM-based judge used by offline benchmark and eval workflows."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from config import Configuration
from evals.judges.base import Judge
from evals.schema import BenchmarkCase
from utils import strip_thinking_tokens, truncate_text

logger = logging.getLogger(__name__)

_STRICT_JSON_RESPONSE_FORMAT = {"type": "json_object"}
_DEFAULT_JUDGE_VERSION = "llm_judge_v1"
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_TEMPERATURE = 0.0
_REPORT_CHAR_LIMIT = 12000
_TASK_SUMMARY_CHAR_LIMIT = 900
_TASK_SOURCES_CHAR_LIMIT = 900
_MAX_TASKS_IN_PROMPT = 12
_MAX_FINDINGS = 8
_SUPPORTED_VERDICTS = {"pass", "warning", "fail"}
_SUPPORTED_SEVERITIES = {"low", "medium", "high"}
_SUPPORTED_FINDING_CATEGORIES = {
    "unsupported_claim",
    "coverage_gap",
    "citation_mismatch",
    "stale_evidence",
    "overclaim",
    "format_issue",
    "other",
}

InvokeFn = Callable[[str, dict[str, Any] | None], str]


def _coerce_text(value: Any) -> str:
    """Extract text from OpenAI-compatible message payloads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        if isinstance(value.get("reasoning"), str):
            return value["reasoning"]
        if isinstance(value.get("reasoning_content"), str):
            return value["reasoning_content"]
        return "".join(_coerce_text(item) for item in value.values())
    if hasattr(value, "model_dump"):
        return _coerce_text(value.model_dump(exclude_none=True))
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return str(value) if not isinstance(value, (bytes, bytearray)) else ""


def _get_item_value(item: Any, name: str, default: Any = None) -> Any:
    """Return a field from either an object-like or dict-like todo item."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalize_string_list(value: Any, *, limit: int | None = None) -> list[str]:
    """Normalize arbitrary values into a compact list of strings."""
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


class LLMJudge(Judge):
    """Use an LLM to add semantic quality scoring for a benchmark case."""

    def __init__(
        self,
        *,
        config: Configuration | None = None,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        judge_version: str | None = None,
        invocation: InvokeFn | None = None,
        load_env_file: bool = True,
    ) -> None:
        self._config = config or Configuration.from_env(load_env_file=load_env_file)
        self._provider = (
            str(provider or os.getenv("EVAL_LLM_JUDGE_PROVIDER") or self._config.llm_provider or "")
            .strip()
            .lower()
        )
        self._model = str(
            model
            or os.getenv("EVAL_LLM_JUDGE_MODEL_ID")
            or self._config.llm_model_id
            or self._config.local_llm
            or "llm-judge"
        ).strip()
        self._base_url = str(
            base_url
            or os.getenv("EVAL_LLM_JUDGE_BASE_URL")
            or self._resolve_default_base_url(self._provider)
            or ""
        ).strip()
        self._api_key = str(
            api_key
            or os.getenv("EVAL_LLM_JUDGE_API_KEY")
            or self._resolve_default_api_key(self._provider)
            or ""
        ).strip()
        self._timeout_seconds = self._coerce_positive_float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("EVAL_LLM_JUDGE_TIMEOUT_SECONDS"),
            default=_DEFAULT_TIMEOUT_SECONDS,
        )
        self._temperature = self._coerce_float(
            temperature if temperature is not None else os.getenv("EVAL_LLM_JUDGE_TEMPERATURE"),
            default=_DEFAULT_TEMPERATURE,
        )
        self._judge_version = str(
            judge_version or os.getenv("EVAL_LLM_JUDGE_VERSION") or _DEFAULT_JUDGE_VERSION
        ).strip() or _DEFAULT_JUDGE_VERSION
        self._invocation = invocation

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        report_markdown: str,
        todo_items: Sequence[Any],
        trace_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Run the benchmark LLM judge and return structured metrics."""
        trace = dict(trace_snapshot or {})
        report_text = (report_markdown or "").strip()
        base_metrics = self._base_metrics(report_text=report_text, trace=trace)

        if not report_text:
            return {
                **base_metrics,
                "judge_status": "skipped",
                "judge_error": "",
                "overall_verdict": "fail",
                "reason": "报告为空，跳过 LLM judge 语义评估。",
                "findings": [],
                "factuality_score": 0.0,
                "coverage_score": 0.0,
                "citation_grounding_score": 0.0,
                "freshness_score": 0.0,
                "conservativeness_score": 0.0,
                "scoring_notes": {"strengths": [], "weaknesses": ["未生成最终报告"]},
            }

        if not self._is_configured():
            return {
                **base_metrics,
                "judge_status": "skipped",
                "judge_error": "judge_not_configured",
                "overall_verdict": "warning",
                "reason": "LLM judge 未完成配置，已跳过二级评估。",
                "findings": [],
                "factuality_score": 0.0,
                "coverage_score": 0.0,
                "citation_grounding_score": 0.0,
                "freshness_score": 0.0,
                "conservativeness_score": 0.0,
                "scoring_notes": {"strengths": [], "weaknesses": ["judge 未配置"]},
            }

        prompt = self._build_prompt(
            case=case,
            report_markdown=report_text,
            todo_items=todo_items,
            trace_snapshot=trace,
        )
        try:
            raw_response = self._run_with_json_fallback(prompt)
            payload = self._extract_json_payload(raw_response)
            if not isinstance(payload, dict):
                raise ValueError("judge response did not contain a JSON object")
            normalized = self._normalize_payload(payload)
        except Exception as exc:
            logger.warning("LLM judge failed case=%s error=%s", case.id, exc)
            return {
                **base_metrics,
                "judge_status": "error",
                "judge_error": str(exc),
                "overall_verdict": "warning",
                "reason": "LLM judge 执行失败，当前仅保留基础运行指标。",
                "findings": [],
                "factuality_score": 0.0,
                "coverage_score": 0.0,
                "citation_grounding_score": 0.0,
                "freshness_score": 0.0,
                "conservativeness_score": 0.0,
                "scoring_notes": {"strengths": [], "weaknesses": ["judge 执行失败"]},
            }

        return {
            **base_metrics,
            "judge_status": "success",
            "judge_error": "",
            **normalized,
        }

    def _base_metrics(self, *, report_text: str, trace: Mapping[str, Any]) -> dict[str, Any]:
        degraded_flag = bool(
            trace.get("degraded")
            or trace.get("fallback_triggered")
            or trace.get("status") == "partial_success"
        )
        return {
            "report_generated": bool(report_text.strip()),
            "degraded_flag": degraded_flag,
            "total_latency_ms": round(float(trace.get("elapsed_ms") or 0.0), 2),
            "estimated_cost": round(float(trace.get("estimated_cost") or 0.0), 6),
            "judge_model": self._model,
            "judge_provider": self._provider or "custom",
            "judge_version": self._judge_version,
        }

    def _build_prompt(
        self,
        *,
        case: BenchmarkCase,
        report_markdown: str,
        todo_items: Sequence[Any],
        trace_snapshot: Mapping[str, Any],
    ) -> str:
        case_block = {
            "id": case.id,
            "topic": case.topic,
            "expected_sections": list(case.expected_sections or []),
            "expected_keywords": list(case.expected_keywords or []),
            "freshness_sensitive": bool(case.freshness_sensitive),
            "metadata": dict(case.metadata or {}),
        }
        trace_block = {
            "status": str(trace_snapshot.get("status") or "").strip(),
            "elapsed_ms": round(float(trace_snapshot.get("elapsed_ms") or 0.0), 2),
            "estimated_cost": round(float(trace_snapshot.get("estimated_cost") or 0.0), 6),
            "degraded": bool(trace_snapshot.get("degraded")),
            "degraded_reasons": _normalize_string_list(trace_snapshot.get("degraded_reasons")),
            "fallback_triggered": bool(trace_snapshot.get("fallback_triggered")),
            "fallback_reasons": _normalize_string_list(trace_snapshot.get("fallback_reasons")),
        }
        task_block = [self._serialize_task(item) for item in list(todo_items or [])[:_MAX_TASKS_IN_PROMPT]]

        report_excerpt = truncate_text(report_markdown, _REPORT_CHAR_LIMIT, suffix="... [report truncated]")
        return (
            "You are a benchmark evaluator for a research agent.\n"
            "Evaluate the final output quality of a single benchmark case.\n"
            "Only use the supplied benchmark case, report, task outputs, and trace data.\n"
            "Do not invent external facts, do not browse the web, and do not rewrite the report.\n"
            "Return one strict JSON object and nothing else.\n\n"
            "Scoring guide:\n"
            "- factuality_score: Are important claims supported by available evidence?\n"
            "- coverage_score: Does the report cover the main dimensions implied by the case?\n"
            "- citation_grounding_score: Are citations relevant and sufficient for the claims they support?\n"
            "- freshness_score: For freshness-sensitive cases, does the report use recent enough evidence and state uncertainty properly?\n"
            "- conservativeness_score: When evidence is weak, does the report remain careful rather than overclaim?\n\n"
            "Allowed verdicts: pass, warning, fail.\n"
            "Allowed finding severities: low, medium, high.\n"
            "Preferred finding categories: unsupported_claim, coverage_gap, citation_mismatch, stale_evidence, overclaim, format_issue, other.\n\n"
            "Output JSON schema:\n"
            "{\n"
            '  "factuality_score": 0.0,\n'
            '  "coverage_score": 0.0,\n'
            '  "citation_grounding_score": 0.0,\n'
            '  "freshness_score": 0.0,\n'
            '  "conservativeness_score": 0.0,\n'
            '  "overall_verdict": "pass | warning | fail",\n'
            '  "reason": "one sentence summary",\n'
            '  "findings": [\n'
            "    {\n"
            '      "severity": "high | medium | low",\n'
            '      "category": "unsupported_claim | coverage_gap | citation_mismatch | stale_evidence | overclaim | format_issue | other",\n'
            '      "message": "short finding"\n'
            "    }\n"
            "  ],\n"
            '  "scoring_notes": {\n'
            '    "strengths": ["..."],\n'
            '    "weaknesses": ["..."]\n'
            "  }\n"
            "}\n\n"
            f"Benchmark case:\n{json.dumps(case_block, ensure_ascii=False, indent=2)}\n\n"
            f"Request trace:\n{json.dumps(trace_block, ensure_ascii=False, indent=2)}\n\n"
            f"Task outputs:\n{json.dumps(task_block, ensure_ascii=False, indent=2)}\n\n"
            "Final report markdown:\n"
            f"{report_excerpt}\n"
        )

    def _serialize_task(self, item: Any) -> dict[str, Any]:
        claims = self._normalize_claims(_get_item_value(item, "claims"))
        review_issues = self._normalize_review_issues(_get_item_value(item, "review_issues"))
        evidence_items = _get_item_value(item, "evidence_items", [])
        source_ids: list[str] = []
        if isinstance(evidence_items, list):
            source_ids = [
                str(entry.get("source_id") or "").strip()
                for entry in evidence_items
                if isinstance(entry, dict) and str(entry.get("source_id") or "").strip()
            ]
        return {
            "id": _get_item_value(item, "id"),
            "title": str(_get_item_value(item, "title", "") or "").strip(),
            "intent": str(_get_item_value(item, "intent", "") or "").strip(),
            "query": str(_get_item_value(item, "query", "") or "").strip(),
            "status": str(_get_item_value(item, "status", "") or "").strip(),
            "summary_excerpt": truncate_text(
                str(_get_item_value(item, "summary", "") or "").strip(),
                _TASK_SUMMARY_CHAR_LIMIT,
                suffix="... [summary truncated]",
            ),
            "sources_excerpt": truncate_text(
                str(_get_item_value(item, "sources_summary", "") or "").strip(),
                _TASK_SOURCES_CHAR_LIMIT,
                suffix="... [sources truncated]",
            ),
            "review_status": str(_get_item_value(item, "review_status", "") or "").strip(),
            "claims": claims,
            "review_issues": review_issues,
            "source_ids": source_ids[:12],
            "notice_count": len(_get_item_value(item, "notices", []) or []),
        }

    @staticmethod
    def _normalize_claims(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in value[:8]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            normalized.append(
                {
                    "text": truncate_text(text, 240, suffix="... [claim truncated]"),
                    "source_ids": _normalize_string_list(item.get("source_ids"), limit=6),
                    "support_status": str(item.get("support_status") or "").strip(),
                }
            )
        return normalized

    @staticmethod
    def _normalize_review_issues(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in value[:8]:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message") or "").strip()
            if not message:
                continue
            normalized.append(
                {
                    "severity": str(item.get("severity") or "").strip().lower(),
                    "check": str(item.get("check") or "").strip(),
                    "message": truncate_text(message, 220, suffix="... [issue truncated]"),
                    "source_ids": _normalize_string_list(item.get("source_ids"), limit=6),
                    "origin": str(item.get("origin") or "").strip(),
                }
            )
        return normalized

    def _run_with_json_fallback(self, prompt: str) -> str:
        try:
            return self._invoke(prompt, response_format=_STRICT_JSON_RESPONSE_FORMAT)
        except Exception as exc:
            if not self._response_format_is_unsupported(exc):
                raise
            logger.info("LLM judge JSON mode unsupported; retrying without response_format: %s", exc)
            return self._invoke(prompt, response_format=None)

    def _invoke(self, prompt: str, *, response_format: dict[str, Any] | None) -> str:
        if self._invocation is not None:
            return str(self._invocation(prompt, response_format))

        client_kwargs: dict[str, Any] = {
            "api_key": self._api_key or "llm-judge",
            "timeout": self._timeout_seconds,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if self._should_bypass_env_proxy(self._base_url):
            client_kwargs["http_client"] = httpx.Client(trust_env=False)

        client = OpenAI(**client_kwargs)
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a strict JSON benchmark evaluator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        response = client.chat.completions.create(**request_kwargs)
        try:
            message = response.choices[0].message
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"judge returned unexpected response shape: {exc}") from exc

        text = _coerce_text(getattr(message, "content", None))
        if not text:
            text = _coerce_text(getattr(message, "reasoning", None))
        if not text:
            raise ValueError("judge returned empty response")
        return text

    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, Any] | None:
        cleaned = strip_thinking_tokens((text or "").strip())
        decoder = json.JSONDecoder()
        index = 0
        while index < len(cleaned):
            if cleaned[index] != "{":
                index += 1
                continue
            try:
                payload, end = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(payload, dict):
                return payload
            index += max(end, 1)
        return None

    def _normalize_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        factuality = self._normalize_score(payload.get("factuality_score"))
        coverage = self._normalize_score(payload.get("coverage_score"))
        citation_grounding = self._normalize_score(payload.get("citation_grounding_score"))
        freshness = self._normalize_score(payload.get("freshness_score"))
        conservativeness = self._normalize_score(payload.get("conservativeness_score"))

        verdict = str(payload.get("overall_verdict") or "").strip().lower()
        if verdict not in _SUPPORTED_VERDICTS:
            if min(factuality, citation_grounding) < 0.4:
                verdict = "fail"
            elif min(factuality, coverage, citation_grounding, conservativeness) < 0.7:
                verdict = "warning"
            else:
                verdict = "pass"

        findings = self._normalize_findings(payload.get("findings"))
        scoring_notes = self._normalize_scoring_notes(payload.get("scoring_notes"))
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            if verdict == "fail":
                reason = "LLM judge 认为该报告存在高风险语义质量问题。"
            elif verdict == "warning":
                reason = "LLM judge 认为该报告整体可用，但仍存在需要人工关注的问题。"
            else:
                reason = "LLM judge 认为该报告整体表现稳定。"

        return {
            "factuality_score": factuality,
            "coverage_score": coverage,
            "citation_grounding_score": citation_grounding,
            "freshness_score": freshness,
            "conservativeness_score": conservativeness,
            "overall_verdict": verdict,
            "reason": reason,
            "findings": findings,
            "scoring_notes": scoring_notes,
        }

    @staticmethod
    def _normalize_findings(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []

        normalized: list[dict[str, str]] = []
        for item in value[:_MAX_FINDINGS]:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message") or "").strip()
            if not message:
                continue
            severity = str(item.get("severity") or "medium").strip().lower()
            if severity not in _SUPPORTED_SEVERITIES:
                severity = "medium"
            category = str(item.get("category") or "other").strip().lower()
            if category not in _SUPPORTED_FINDING_CATEGORIES:
                category = "other"
            normalized.append(
                {
                    "severity": severity,
                    "category": category,
                    "message": truncate_text(message, 240, suffix="... [finding truncated]"),
                }
            )
        return normalized

    @staticmethod
    def _normalize_scoring_notes(value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {"strengths": [], "weaknesses": []}
        return {
            "strengths": _normalize_string_list(value.get("strengths"), limit=5),
            "weaknesses": _normalize_string_list(value.get("weaknesses"), limit=5),
        }

    @staticmethod
    def _normalize_score(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric < 0.0:
            return 0.0
        if numeric > 1.0:
            return 1.0
        return round(numeric, 4)

    def _is_configured(self) -> bool:
        return bool(self._invocation or (self._model and (self._api_key or self._base_url)))

    def _resolve_default_base_url(self, provider: str) -> str | None:
        if provider == "ollama":
            return self._config.sanitized_ollama_url()
        if provider == "lmstudio":
            return self._config.lmstudio_base_url
        return self._config.llm_base_url

    def _resolve_default_api_key(self, provider: str) -> str | None:
        if self._config.llm_api_key:
            return self._config.llm_api_key
        if provider == "ollama":
            return "ollama"
        if self._base_url:
            return "llm-judge"
        return None

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_positive_float(value: Any, *, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric if numeric > 0 else default

    @staticmethod
    def _response_format_is_unsupported(exc: Exception) -> bool:
        """Return whether the provider rejected the JSON-mode request contract."""
        message = str(exc or "").casefold()
        if "response_format" not in message and "json_object" not in message and "json schema" not in message:
            return isinstance(exc, TypeError) and "response_format" in message

        unsupported_markers = (
            "unsupported",
            "not support",
            "not supported",
            "invalid",
            "unknown",
            "unexpected",
            "extra inputs are not permitted",
            "extra fields not permitted",
        )
        return any(marker in message for marker in unsupported_markers)

    @staticmethod
    def _should_bypass_env_proxy(base_url: str | None) -> bool:
        """Bypass environment proxies for loopback/private OpenAI-compatible endpoints."""
        parsed = urlsplit(str(base_url or "").strip())
        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            return False
        if hostname == "localhost":
            return True
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return False
        return bool(address.is_loopback or address.is_private or address.is_link_local)
