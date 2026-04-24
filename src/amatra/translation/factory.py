"""Factory entry point consumed by the app."""
from __future__ import annotations

from typing import Any, Mapping

from . import adapters as _adapters  # noqa: F401  triggers adapter registration
from .base import BaseTranslator
from .registry import get_translator_spec
from .types import TranslatorConfig


def load_translator(
    config: Mapping[str, Any] | TranslatorConfig,
) -> BaseTranslator:
    """Build (but do not yet ``load``) a translator from a config dict.

    Example::

        translator = load_translator({"type": "elan-mt", "variant": "base"})
        translator.load()
        result = translator.translate(TranslationRequest(source_texts=["..."]))
    """
    cfg = (
        config
        if isinstance(config, TranslatorConfig)
        else TranslatorConfig.from_mapping(config)
    )
    spec = get_translator_spec(cfg.type)
    return spec.builder(cfg)
