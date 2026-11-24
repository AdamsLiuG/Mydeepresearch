"""Three-segment KV-Cache prompt prefix builder and provider cache adapter."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


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

    def resolve_provider_mode(self) -> str:
        if not self.enabled():
            return "disabled"
        explicit = str(getattr(self._config, "kv_cache_provider_mode", "auto") or "auto").strip().lower()
        if explicit in {"openai", "anthropic", "vllm", "disabled"}:
            return explicit
        provider = str(getattr(self._config, "llm_provider", "") or "").strip().lower()
        base_url = str(getattr(self._config, "llm_base_url", "") or "").strip().lower()
        model_fn = getattr(self._config, "resolved_model", None)
        model_name = str(model_fn() if callable(model_fn) else getattr(self._config, "llm_model_id", "") or "").lower()
        if provider == "vllm" or ":8000" in base_url:
            return "vllm"
        if "anthropic" in provider or "anthropic" in base_url or "claude" in model_name:
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

        sys_material = "\n".join(seg.text for seg in sys_segments if seg.text)
        prefix_material = "\n".join(seg.text for seg in (*sys_segments, *frozen_segments) if seg.text)
        system_fp = hashlib.sha256(sys_material.encode("utf-8")).hexdigest()[:16] if sys_material else ""
        prefix_fp = hashlib.sha256(prefix_material.encode("utf-8")).hexdigest()[:16] if prefix_material else ""

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


class ProviderCacheAdapter:
    """Render PromptEnvelope into provider-specific OpenAI, Anthropic, or vLLM payloads."""

    @staticmethod
    def build_tenant_cache_key(envelope: PromptEnvelope) -> str:
        raw = f"{envelope.tenant_id}:{envelope.stage}:{envelope.cache_key_scope}:{envelope.prefix_fingerprint}"
        return "dr_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def render_messages_and_kwargs(
        cls,
        envelope: PromptEnvelope,
        *,
        provider_mode: str,
        enhanced_tool_section: str = "",
        history_messages: list[dict[str, Any]] | None = None,
        config: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_text = envelope.system_text()
        if str(enhanced_tool_section or "").strip():
            system_text = "\n\n".join(
                part for part in (system_text, str(enhanced_tool_section).strip()) if part
            )

        frozen_text = "\n\n".join(seg.text for seg in envelope.frozen_segments if seg.text).strip()
        volatile_text = "\n\n".join(seg.text for seg in envelope.volatile_segments if seg.text).strip()
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

            return [
                {"role": "system", "content": [system_block] if system_text else []},
                {"role": "user", "content": user_blocks or [{"type": "text", "text": ""}]},
            ], extra_kwargs

        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        for msg in history_messages or ():
            if isinstance(msg, dict) and msg.get("role"):
                messages.append({"role": str(msg["role"]), "content": str(msg.get("content") or "")})

        user_text = "\n\n".join(part for part in (frozen_text, volatile_text) if part)
        messages.append({"role": "user", "content": user_text})

        if prompt_key_enabled and provider_mode in {"openai", "vllm"} and envelope.prefix_fingerprint:
            extra_kwargs["prompt_cache_key"] = cls.build_tenant_cache_key(envelope)
        return messages, extra_kwargs
