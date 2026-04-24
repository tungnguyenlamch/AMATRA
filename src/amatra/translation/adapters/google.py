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
    if variant not in {"free", "cloud"}:
        raise ValueError("google variant must be one of ('free', 'cloud')")
    use_official_api = variant == "cloud"
    source_lang = str(cfg.extra.get("source_lang", "ja"))
    target_lang = str(cfg.extra.get("target_lang", "en"))

    if use_official_api:
        import os

        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError(
                "google:cloud requires GOOGLE_APPLICATION_CREDENTIALS to be set"
            )

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
