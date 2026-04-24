"""Typed request/response/config objects for the translation boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TranslationRequest:
    """Input to a translator.

    ``source_texts`` is a sequence (list/tuple) of strings. ``context`` is an
    optional parallel sequence of source-side context strings that postprocess
    steps may consume. ``options`` is a free-form bag for per-call overrides.
    """

    source_texts: Sequence[str]
    source_lang: str | None = "ja"
    target_lang: str | None = "en"
    context: Sequence[str] | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranslationResult:
    """Output of a translator. ``translations`` is aligned with ``source_texts``."""

    translations: list[str]
    source_texts: Sequence[str]
    model_id: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranslatorConfig:
    """Config used by the factory to instantiate a translator.

    ``type`` is the registry key (for example ``"elan-mt"``); ``variant`` picks a
    sub-option of that type (for example ``"base"``). ``extra`` captures any
    other keyword arguments a specific adapter may want.
    """

    type: str
    variant: str | None = None
    device: str = "auto"
    verbose: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "TranslatorConfig":
        if "type" not in cfg:
            raise ValueError("TranslatorConfig requires a 'type' key")
        model_type = cfg["type"]
        variant = cfg.get("variant")
        if isinstance(model_type, str) and ":" in model_type and variant is None:
            model_type, variant = model_type.split(":", 1)
            variant = variant or None
        known = {"type", "variant", "device", "verbose"}
        extra = {k: v for k, v in cfg.items() if k not in known}
        return cls(
            type=model_type,
            variant=variant,
            device=cfg.get("device", "auto"),
            verbose=cfg.get("verbose", False),
            extra=extra,
        )
