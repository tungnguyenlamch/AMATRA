"""Public API for the translation boundary.

The app and any other consumer should only import symbols from this module.
"""
from __future__ import annotations

from .base import BaseTranslator
from .errors import TranslatorBoundaryError
from .factory import load_translator
from .registry import (
    TranslatorSpec,
    get_translator_spec,
    list_translators,
    register_translator,
)
from .types import TranslationRequest, TranslationResult, TranslatorConfig

__all__ = [
    "BaseTranslator",
    "TranslatorBoundaryError",
    "TranslationRequest",
    "TranslationResult",
    "TranslatorConfig",
    "TranslatorSpec",
    "get_translator_spec",
    "list_translators",
    "load_translator",
    "register_translator",
]
