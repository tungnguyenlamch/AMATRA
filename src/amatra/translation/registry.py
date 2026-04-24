"""Decorator-based translator registry.

Adapter modules call ``@register_translator(...)`` at import time; the factory
looks up entries by ``type`` and delegates construction to the registered
builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .base import BaseTranslator
from .types import TranslatorConfig

Builder = Callable[[TranslatorConfig], BaseTranslator]


@dataclass(frozen=True)
class TranslatorSpec:
    """Static description of a translator the app can present to users."""

    type: str
    variants: tuple[str, ...]
    builder: Builder
    description: str = ""


_REGISTRY: dict[str, TranslatorSpec] = {}


def register_translator(
    type: str,
    *,
    variants: Iterable[str] = (),
    description: str = "",
) -> Callable[[Builder], Builder]:
    """Decorator that registers a translator builder under ``type``."""

    def deco(fn: Builder) -> Builder:
        if type in _REGISTRY:
            raise ValueError(f"translator '{type}' already registered")
        _REGISTRY[type] = TranslatorSpec(
            type=type,
            variants=tuple(variants),
            builder=fn,
            description=description,
        )
        return fn

    return deco


def get_translator_spec(type: str) -> TranslatorSpec:
    if type not in _REGISTRY:
        raise KeyError(
            f"no translator registered for type={type!r}. "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[type]


def list_translators() -> list[TranslatorSpec]:
    """Return registered translators (stable order of registration)."""
    return list(_REGISTRY.values())


def _clear_registry_for_tests() -> None:
    _REGISTRY.clear()
