"""Adapter registration for generic HF causal-LM translators (``LLMTranslator``).

Variants here are informational labels mapped to HF model IDs via
``cfg.extra["model_name"]`` when a caller wants a non-default checkpoint.
"""
from __future__ import annotations

from ..base import BaseTranslator
from ..registry import register_translator
from ..types import TranslatorConfig
from .research_adapter import ResearchTranslatorAdapter

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


@register_translator(
    "llm",
    variants=("qwen2.5-0.5b",),
    description="HF causal-LM zero/few-shot translator (LLMTranslator)",
)
def _build(cfg: TranslatorConfig) -> BaseTranslator:
    model_name = str(cfg.extra.get("model_name", _DEFAULT_MODEL))
    preset_values = dict(cfg.extra.get("preset_values", {}))

    def factory() -> object:
        from pipeline.TranslationModels.LLMTranslator import LLMTranslator

        translator = LLMTranslator(
            model_name=model_name,
            device=cfg.device,
            verbose=cfg.verbose,
        )
        if preset_values:
            translator.configure(**preset_values)
        return translator

    label = cfg.variant or model_name
    return ResearchTranslatorAdapter(
        model_id=f"llm:{label}",
        factory=factory,
    )
