"""Adapter registration for the Mitsua ElanMT ja->en translator family."""
from __future__ import annotations

from ..base import BaseTranslator
from ..registry import register_translator
from ..types import TranslatorConfig
from .research_adapter import ResearchTranslatorAdapter

_VARIANTS: tuple[str, ...] = ("tiny", "base", "bt")


@register_translator(
    "elan-mt",
    variants=_VARIANTS,
    description="Mitsua ElanMT ja->en (HuggingFace pipeline)",
)
def _build(cfg: TranslatorConfig) -> BaseTranslator:
    variant = cfg.variant or "base"
    if variant not in _VARIANTS:
        raise ValueError(
            f"elan-mt variant must be one of {_VARIANTS}, got {variant!r}"
        )

    def factory() -> object:
        from pipeline.TranslationModels.ElanMtJaEnTranslator import (
            ElanMtJaEnTranslator,
        )

        return ElanMtJaEnTranslator(
            elan_model=variant,
            device=cfg.device,
            verbose=cfg.verbose,
        )

    return ResearchTranslatorAdapter(
        model_id=f"elan-mt:{variant}",
        factory=factory,
    )
