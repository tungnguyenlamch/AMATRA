from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from amatra.translation.base import BaseTranslator
from amatra.translation.types import TranslationRequest, TranslationResult


@dataclass(frozen=True)
class MockBubble:
    bbox: list[int]
    text: str
    translation: str


class MockTranslator(BaseTranslator):
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self.model_id = "mock:echo"
        self._mapping = mapping or {}
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def translate(self, request: TranslationRequest) -> TranslationResult:
        if not self.is_loaded:
            self.load()
        translations = [
            self._mapping.get(text, f"EN::{text}") for text in request.source_texts
        ]
        return TranslationResult(
            translations=translations,
            source_texts=list(request.source_texts),
            model_id=self.model_id,
            meta={"mock": True},
        )


def bubbles_from_fixture(spec: dict[str, object]) -> list[MockBubble]:
    bubbles = []
    for bubble in spec.get("bubbles", []):
        payload = dict(bubble)
        bubbles.append(
            MockBubble(
                bbox=list(payload["bbox"]),
                text=str(payload["ocr_text"]),
                translation=str(payload["translation"]),
            )
        )
    return bubbles


def make_rect_mask(width: int, height: int, bbox: Iterable[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def render_mock_output(image: np.ndarray, bubbles: list[MockBubble]) -> np.ndarray:
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for bubble in bubbles:
        x1, y1, x2, y2 = bubble.bbox
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=(235, 235, 235), outline=(0, 0, 0))
        draw.text((x1 + 1, y1 + 1), bubble.translation[:8], fill=(0, 0, 0))
    return np.array(canvas)
