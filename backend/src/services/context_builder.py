"""Three-segment KV-Cache prompt prefix builder and provider cache adapter."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from prompts import REPORT_MAP_ROLE, REPORT_MAP_SCHEMA

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _evidence_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "<", r"\u003C"
    ).replace(">", r"\u003E")


def _web_evidence_frame(items: list[str]) -> str:
    body = "[" + ",".join(items) + "]"
    return f'<WEB_EVIDENCE_DATA chars="{len(body)}">\n{body}\n</WEB_EVIDENCE_DATA>'


def render_web_evidence_data(
    records: Sequence[dict[str, Any]], evidence_store: Any, *, max_chars: int = 6000,
) -> str:
    """Render bounded Store data, retaining a whole table and its anchor or omitting it."""
    if not getattr(evidence_store, "web_extraction_enabled", False):
        return ""
    source_ids = list(dict.fromkeys(
        str(item["source_id"]) for item in records if isinstance(item, dict) and item.get("source_id")
    ))
    if not source_ids:
        return ""
    items: list[str] = []
    for record in evidence_store.lookup(source_ids=source_ids, include_full_content=True):
        sid = record["source_id"]
        eligible = evidence_store.is_eligible_for_facts(sid)
        metadata = record.get("provenance_meta", {}).get("web_extraction", {})
        item = {
            "source_id": sid,
            "tier": metadata.get("tier") if eligible else "L0",
            "fact_eligible": eligible,
            "title": record.get("title", ""),
            "url": record.get("url", ""),
        }
        if not eligible:
            item["snippet"] = record.get("snippet", "")
        rendered = _evidence_json(item)
        if len(_web_evidence_frame([*items, rendered])) > max_chars:
            continue
        if eligible:
            if type(metadata.get("artifact_version")) is int:
                item["artifact_version"] = metadata["artifact_version"]
            for block in record.get("multimodal_blocks") or ():
                if block.get("modality") != "table":
                    continue
                table = {"anchor_id": block.get("anchor_id"), **(block.get("structured_payload") or {})}
                candidate = {**item, "tables": [*item.get("tables", []), table]}
                encoded = _evidence_json(candidate)
                if len(_web_evidence_frame([*items, encoded])) <= max_chars:
                    item, rendered = candidate, encoded
            full_text = record.get("full_content", "")
            if metadata.get("tier") in {"L1", "L2"}:
                passages = [
                    {"anchor_id": anchor["anchor_id"], "text": full_text[anchor["start"]:anchor["end"]]}
                    for anchor in metadata.get("text_anchors", [])
                ]
            else:
                passages = [{"text": full_text or record.get("snippet", "")}]
            for passage in passages:
                if not passage["text"]:
                    continue
                candidate = {**item, "passages": [*item.get("passages", []), passage]}
                encoded = _evidence_json(candidate)
                if len(_web_evidence_frame([*items, encoded])) <= max_chars:
                    item, rendered = candidate, encoded
        items.append(rendered)
    return _web_evidence_frame(items) if items else ""


def estimate_tokens_fast(text: str) -> int:
    """Fast CJK-aware token estimate mirroring metrics._estimate_tokens."""

    raw = str(text or "")
    if not raw:
        return 0
    cjk_count = len(_CJK_RE.findall(raw))
    other_count = len(raw) - cjk_count
    return max(1, int(round((cjk_count * 1.45) + (other_count / 3.8))))


@dataclass(frozen=True)
class PromptSegment:
    """Single labeled segment placed into one of the three prefix tiers."""

    name: str
    tier: str  # "global_static" | "request_frozen" | "wave_checkpoint" | "volatile_suffix"
    text: str
    cacheable: bool = True
    breakpoint: bool = False


@dataclass
class PromptEnvelope:
    """Provider-agnostic 3-segment prompt structure maximizing KV-Cache prefix hits."""

    stage: str
    tenant_id: str
    request_id: str
    system_segments: list[PromptSegment] = field(default_factory=list)
    frozen_segments: list[PromptSegment] = field(default_factory=list)
    volatile_segments: list[PromptSegment] = field(default_factory=list)
    cache_key_scope: str = "request"
    prefix_fingerprint: str = ""
    system_fingerprint: str = ""

    def to_flat_prompt(self, *, include_system_role: bool = False) -> str:
        """Flatten envelope deterministically for FakeAgent/Prompt-Only callers."""

        parts: list[str] = []
        for seg in self.system_segments:
            if not seg.text:
                continue
            if seg.name == "role" and not include_system_role:
                continue
            parts.append(seg.text)
        for seg in (*self.frozen_segments, *self.volatile_segments):
            if seg.text:
                parts.append(seg.text)
        return "\n\n".join(parts).strip()

    def system_text(self) -> str:
        return "\n\n".join(seg.text for seg in self.system_segments if seg.text).strip()

    def user_text(self) -> str:
        frozen = "\n\n".join(seg.text for seg in self.frozen_segments if seg.text).strip()
        volatile = "\n\n".join(seg.text for seg in self.volatile_segments if seg.text).strip()
        return "\n\n".join(part for part in (frozen, volatile) if part).strip()


class ContextPrefixBuilder:
    """Assemble prompts in strictly increasing volatility order (Global -> Frozen -> Volatile)."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def enabled(self) -> bool:
        return bool(getattr(self._config, "kv_cache_prefix_optimization_enabled", True))

    def resolve_provider_mode(self, *, llm: Any = None) -> str:
        if not self.enabled():
            return "disabled"
        explicit = str(getattr(self._config, "kv_cache_provider_mode", "auto") or "auto").strip().lower()
        if explicit in {"openai", "anthropic", "vllm", "disabled"}:
            return explicit
        provider = str(getattr(llm, "provider", None) or getattr(self._config, "llm_provider", "") or "").lower()
        base_url = str(getattr(llm, "base_url", None) or getattr(self._config, "llm_base_url", "") or "").lower()
        model_name = getattr(llm, "model", None)
        if not model_name:
            model_fn = getattr(self._config, "resolved_model", None)
            model_name = model_fn() if callable(model_fn) else getattr(self._config, "llm_model_id", "")
        if provider == "vllm" or ":8000" in base_url:
            return "vllm"
        if "anthropic" in provider or "anthropic" in base_url or "claude" in str(model_name or "").lower():
            return "anthropic"
        return "openai"

    def build_envelope(
        self,
        *,
        stage: str,
        tenant_id: str = "default",
        request_id: str = "local",
        system_role: str = "",
        static_schema_and_rules: str = "",
        research_topic: str = "",
        skill_pack_context: str = "",
        frozen_index_and_ledger: str = "",
        volatile_blocks: list[tuple[str, str]] | None = None,
        cache_key_scope: str = "request",
    ) -> PromptEnvelope:
        sys_segments: list[PromptSegment] = []
        if str(system_role or "").strip():
            sys_segments.append(
                PromptSegment(
                    name="role",
                    tier="global_static",
                    text=str(system_role).strip(),
                    breakpoint=False,
                )
            )
        if str(static_schema_and_rules or "").strip():
            sys_segments.append(
                PromptSegment(
                    name="schema_and_rules",
                    tier="global_static",
                    text=str(static_schema_and_rules).strip(),
                    breakpoint=True,
                )
            )

        frozen_segments: list[PromptSegment] = []
        topic_text = str(research_topic or "").strip()
        if topic_text:
            frozen_segments.append(
                PromptSegment(
                    name="topic",
                    tier="request_frozen",
                    text=f"<FROZEN_TOPIC>\n研究主题：{topic_text}\n</FROZEN_TOPIC>",
                    breakpoint=not (str(skill_pack_context or "").strip() or str(frozen_index_and_ledger or "").strip()),
                )
            )
        skill_text = str(skill_pack_context or "").strip()
        frozen_text = str(frozen_index_and_ledger or "").strip()
        if skill_text:
            frozen_segments.append(
                PromptSegment(
                    name="skill_pack",
                    tier="request_frozen",
                    text=skill_text,
                    breakpoint=not bool(frozen_text),
                )
            )
        if frozen_text:
            frozen_segments.append(
                PromptSegment(
                    name="frozen_checkpoint",
                    tier="wave_checkpoint",
                    text=frozen_text,
                    breakpoint=True,
                )
            )

        volatile_segments = [
            PromptSegment(
                name=str(name or "delta").strip() or "delta",
                tier="volatile_suffix",
                text=str(text).strip(),
                cacheable=False,
                breakpoint=False,
            )
            for name, text in (volatile_blocks or ())
            if str(text or "").strip()
        ]

        sys_material = "\n\n".join(seg.text for seg in sys_segments)
        # Encode the message boundary as well as the rendered text to avoid ambiguous joins.
        prefix_material = json.dumps(
            [sys_material, "\n\n".join(seg.text for seg in frozen_segments)],
            ensure_ascii=False,
        )
        system_fp = hashlib.sha256(sys_material.encode("utf-8")).hexdigest()[:16] if sys_material else ""
        prefix_fp = hashlib.sha256(prefix_material.encode("utf-8")).hexdigest()[:16] if sys_segments or frozen_segments else ""

        return PromptEnvelope(
            stage=stage,
            tenant_id=str(tenant_id or "default").strip() or "default",
            request_id=str(request_id or "local").strip() or "local",
            system_segments=sys_segments,
            frozen_segments=frozen_segments,
            volatile_segments=volatile_segments,
            cache_key_scope=cache_key_scope,
            prefix_fingerprint=prefix_fp,
            system_fingerprint=system_fp,
        )


    def freeze_report_prefix(
        self, *, tenant_id: str, request_id: str, topic: str, ledger_version: int,
        prefix_version: int, format_context: str = "", layout: str = "flexible",
        role: str = REPORT_MAP_ROLE, schema: str = REPORT_MAP_SCHEMA, stage: str = "report_map",
        llm: Any = None, terminology: tuple[tuple[str, str], ...] = (),
    ) -> FrozenReportPrefix:
        """Freeze a provider-rendered projection with no evidence, history, or tools."""
        mode = self.resolve_provider_mode(llm=llm)
        model = str(getattr(llm, "model", None) or self._config.resolved_model() or "")
        checkpoint = report_json({
            "format_hash": hashlib.sha256(format_context.encode("utf-8")).hexdigest(),
            "layout": layout,
            "ledger_version": ledger_version,
            "frozen_prefix_version": prefix_version,
            "template_version": 1,
            "topic": topic[:400],
            "terminology": terminology,
        })
        envelope = self.build_envelope(
            stage=stage, tenant_id=tenant_id, request_id=request_id,
            system_role=role, static_schema_and_rules=schema,
            skill_pack_context=format_context[:1200], frozen_index_and_ledger=checkpoint,
        )
        messages, options = ProviderCacheAdapter.render_messages_and_kwargs(
            envelope, provider_mode=mode, config=self._config,
        )
        fingerprint = hashlib.sha256(
            report_json([mode, model, stage, messages, options, []]).encode("utf-8"),
        ).hexdigest()
        return FrozenReportPrefix(mode, model, fingerprint, report_json(messages), report_json(options))


class ProviderCacheAdapter:
    """Render PromptEnvelope into provider-specific OpenAI, Anthropic, or vLLM payloads."""

    @staticmethod
    def _request_scope(envelope: PromptEnvelope) -> str:
        material = json.dumps([envelope.tenant_id, envelope.request_id], ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def build_tenant_cache_key(envelope: PromptEnvelope, *, request_tools: bool = False) -> str:
        scope = (
            "global" if envelope.cache_key_scope == "global"
            and not envelope.frozen_segments and not envelope.volatile_segments and not request_tools
            else ProviderCacheAdapter._request_scope(envelope)
        )
        raw = json.dumps([scope, envelope.stage, envelope.cache_key_scope, envelope.system_fingerprint])
        return "dr_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def render_messages_and_kwargs(
        cls,
        envelope: PromptEnvelope,
        *,
        provider_mode: str,
        enhanced_tool_section: str = "",
        request_tool_section: str = "",
        history_messages: list[dict[str, Any]] | None = None,
        config: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_text = envelope.system_text()
        if str(enhanced_tool_section or "").strip():
            system_text = "\n\n".join(
                part for part in (system_text, str(enhanced_tool_section).strip()) if part
            )

        frozen_text = "\n\n".join(
            part for part in (request_tool_section.strip(), *(seg.text for seg in envelope.frozen_segments))
            if part
        )
        volatile_text = "\n\n".join(seg.text for seg in envelope.volatile_segments if seg.text).strip()
        if frozen_text or volatile_text:
            # Providers may reuse implicit prefixes even when they ignore prompt_cache_key.
            marker = f"<CACHE_SCOPE>{cls._request_scope(envelope)}</CACHE_SCOPE>"
            if frozen_text:
                frozen_text = f"{marker}\n\n{frozen_text}"
            else:
                volatile_text = f"{marker}\n\n{volatile_text}"
        extra_kwargs: dict[str, Any] = {}

        anthropic_enabled = bool(getattr(config, "kv_cache_anthropic_breakpoints_enabled", True))
        min_tokens = max(0, int(getattr(config, "kv_cache_min_segment_tokens", 1024) or 0))
        prompt_key_enabled = bool(getattr(config, "kv_cache_prompt_cache_key_enabled", True))

        if provider_mode == "anthropic" and anthropic_enabled and not history_messages:
            breakpoints_used = 0
            system_block: dict[str, Any] = {"type": "text", "text": system_text}
            if system_text and estimate_tokens_fast(system_text) >= min_tokens and breakpoints_used < 4:
                system_block["cache_control"] = {"type": "ephemeral"}
                breakpoints_used += 1

            user_blocks: list[dict[str, Any]] = []
            if frozen_text:
                frozen_block: dict[str, Any] = {"type": "text", "text": frozen_text}
                if (
                    estimate_tokens_fast(f"{system_text}\n{frozen_text}") >= min_tokens
                    and breakpoints_used < 4
                ):
                    frozen_block["cache_control"] = {"type": "ephemeral"}
                    breakpoints_used += 1
                user_blocks.append(frozen_block)
            if volatile_text:
                user_blocks.append({"type": "text", "text": volatile_text})

            if breakpoints_used:
                messages = [{"role": "system", "content": [system_block]}] if system_text else []
                messages.append({"role": "user", "content": user_blocks or [{"type": "text", "text": ""}]})
                return messages, extra_kwargs

        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        for msg in history_messages or ():
            if isinstance(msg, dict) and msg.get("role"):
                messages.append({"role": str(msg["role"]), "content": str(msg.get("content") or "")})

        user_text = "\n\n".join(part for part in (frozen_text, volatile_text) if part)
        messages.append({"role": "user", "content": user_text})

        if prompt_key_enabled and provider_mode in {"openai", "vllm"} and envelope.prefix_fingerprint:
            extra_kwargs["prompt_cache_key"] = cls.build_tenant_cache_key(
                envelope, request_tools=bool(request_tool_section.strip())
            )
        return messages, extra_kwargs


def report_json(value: Any) -> str:
    """Encode report contracts deterministically and protect the enclosing data tag."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).replace("<", r"\u003C").replace(">", r"\u003E")


def parse_report_json(text: str, *, byte_limit: int) -> dict[str, Any] | None:
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON member")
        return result

    def invalid_constant(value):
        raise ValueError(value)

    try:
        if len(text) > byte_limit or len(text.encode("utf-8")) > byte_limit:
            return None
        payload = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
        return payload if isinstance(payload, dict) else None
    except (ValueError, RecursionError):
        return None


@dataclass(frozen=True)
class ReportModelRequest:
    kind: str
    data_json: str
    prefix_fingerprint: str
    messages_json: str
    options_json: str
    provider_mode: str
    model: str
    max_tokens: int
    timeout: float

    @property
    def messages(self) -> list[dict[str, Any]]:
        return json.loads(self.messages_json)

    @property
    def options(self) -> dict[str, Any]:
        return json.loads(self.options_json)

    @property
    def input_upper_bound(self) -> int:
        # Byte accounting, including serialized message boundaries, bounds text tokens.
        return len(self.messages_json.encode("utf-8")) + len(self.options_json.encode("utf-8")) + 64


@dataclass(frozen=True)
class FrozenReportPrefix:
    provider_mode: str
    model: str
    prefix_fingerprint: str
    messages_json: str
    options_json: str

    def request(
        self, kind: str, data: dict[str, Any], *, max_tokens: int, timeout: float = 12,
    ) -> ReportModelRequest:
        payload = report_json(data)
        suffix = f"<REPORT_DATA>\n{payload}\n</REPORT_DATA>"
        messages = json.loads(self.messages_json)
        content = messages[-1]["content"]
        if isinstance(content, list):
            content.append({"type": "text", "text": suffix})
        else:
            messages[-1]["content"] = f"{content}\n\n{suffix}"
        return ReportModelRequest(
            kind, payload, self.prefix_fingerprint, report_json(messages), self.options_json,
            self.provider_mode, self.model, max_tokens, timeout,
        )
