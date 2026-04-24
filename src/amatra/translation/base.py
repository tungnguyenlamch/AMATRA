"""Abstract base class every translation adapter must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .types import TranslationRequest, TranslationResult


class BaseTranslator(ABC):
    """Stable translator contract exposed to the app.

    Implementations are built by the factory/registry and typically wrap a
    concrete research class (see ``ResearchTranslatorAdapter``) or an external
    API client. The app must depend on this interface only.
    """

    model_id: str = "unknown"

    @abstractmethod
    def load(self) -> None:
        """Load any underlying model weights or client into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Free any underlying model weights or client."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the underlying model is ready to serve ``translate``."""

    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Core typed entry point; returns a parallel list of translations."""

    def predict(self, texts: str | Iterable[str], **kwargs) -> list[str]:
        """List-in / list-out convenience wrapper kept for simpler callers."""
        if isinstance(texts, str):
            request = TranslationRequest(source_texts=[texts], **kwargs)
            return self.translate(request).translations
        return self.translate(
            TranslationRequest(source_texts=list(texts), **kwargs)
        ).translations

    def __enter__(self) -> "BaseTranslator":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.model_id!r}, loaded={self.is_loaded})"
