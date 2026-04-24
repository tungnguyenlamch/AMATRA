"""Adapter registration for the Google Translate adapter."""
from __future__ import annotations

from ..base import BaseTranslator
from ..registry import register_translator
from ..types import TranslatorConfig
from .research_adapter import ResearchTranslatorAdapter


@register_translator(
    "google",
    variants=("free", "cloud"),
    description="Google Translate (unofficial googletrans or Cloud API)",
)
def _build(cfg: TranslatorConfig) -> BaseTranslator:
    variant = cfg.variant or "free"
    use_official_api = variant == "cloud"
    source_lang = str(cfg.extra.get("source_lang", "ja"))
    target_lang = str(cfg.extra.get("target_lang", "en"))

    def factory() -> object:
        from pipeline.TranslationModels.GoogleTranslator import GoogleTranslator

        return GoogleTranslator(
            source_lang=source_lang,
            target_lang=target_lang,
            use_official_api=use_official_api,
            verbose=cfg.verbose,
        )

    return ResearchTranslatorAdapter(
        model_id=f"google:{variant}",
        factory=factory,
    )
