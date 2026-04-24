from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from amatra.translation.types import TranslatorConfig


def split_model_id(model_id: str) -> tuple[str, str | None]:
    if ":" in model_id:
        model_type, variant = model_id.split(":", 1)
        return model_type, (variant or None)
    return model_id, None


def load_presets() -> dict[str, Any]:
    with resources.files("amatra.config").joinpath("translator_presets.v1.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)
    return payload["presets"]


@dataclass(frozen=True)
class TranslatorPreset:
    name: str
    values: dict[str, Any]


def get_preset(name: str | None) -> TranslatorPreset | None:
    if not name:
        return None
    presets = load_presets()
    if name not in presets:
        raise ValueError(
            f"unknown preset {name!r}. available: {sorted(presets)}"
        )
    return TranslatorPreset(name=name, values=dict(presets[name]))


def build_translator_config(
    model_id: str,
    *,
    preset_name: str | None = None,
    verbose: bool = False,
) -> TranslatorConfig:
    model_type, variant = split_model_id(model_id)
    preset = get_preset(preset_name)
    extra = {}
    if preset is not None:
        extra["preset"] = preset.name
        extra["preset_values"] = preset.values
    return TranslatorConfig(
        type=model_type,
        variant=variant,
        verbose=verbose,
        extra=extra,
    )
