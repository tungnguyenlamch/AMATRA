"""Generic adapter that wraps a research ``Translator`` subclass unchanged.

Research classes live in ``src/pipeline/TranslationModels`` and share the
``Translator`` base (``load_model``/``unload_model``/``predict``). This adapter
defers instantiation to a factory callable so construction side-effects (model
downloads, GPU allocation) only happen on ``load``.
"""
from __future__ import annotations

from typing import Callable

from ..base import BaseTranslator
from ..types import TranslationRequest, TranslationResult


class ResearchTranslatorAdapter(BaseTranslator):
    def __init__(
        self,
        model_id: str,
        factory: Callable[[], object],
    ) -> None:
        self.model_id = model_id
        self._factory = factory
        self._impl: object | None = None

    @property
    def is_loaded(self) -> bool:
        return self._impl is not None and bool(getattr(self._impl, "is_loaded", False))

    def load(self) -> None:
        if self._impl is None:
            self._impl = self._factory()
        if not getattr(self._impl, "is_loaded", False):
            self._impl.load_model()  # type: ignore[attr-defined]

    def unload(self) -> None:
        if self._impl is None:
            return
        try:
            self._impl.unload_model()  # type: ignore[attr-defined]
        finally:
            self._impl = None

    def translate(self, request: TranslationRequest) -> TranslationResult:
        if not self.is_loaded:
            self.load()
        assert self._impl is not None
        out = self._impl.predict(list(request.source_texts))  # type: ignore[attr-defined]
        if isinstance(out, str):
            out = [out]
        return TranslationResult(
            translations=list(out),
            source_texts=list(request.source_texts),
            model_id=self.model_id,
        )
