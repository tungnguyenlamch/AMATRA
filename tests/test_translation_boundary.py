from __future__ import annotations

import pytest

from amatra.translation import TranslationRequest, TranslatorBoundaryError, load_translator, register_translator
from amatra.translation.base import BaseTranslator
from amatra.translation.registry import _clear_registry_for_tests
from amatra.translation.types import TranslationResult, TranslatorConfig


@pytest.fixture(autouse=True)
def restore_registry():
    from amatra.translation import adapters  # noqa: F401

    snapshot = dict(__import__("amatra.translation.registry", fromlist=["_REGISTRY"])._REGISTRY)
    yield
    registry = __import__("amatra.translation.registry", fromlist=["_REGISTRY"])
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


def test_duplicate_registration_fails():
    _clear_registry_for_tests()

    @register_translator("dup")
    def build_one(cfg):  # pragma: no cover - exercised via decorator
        raise AssertionError(cfg)

    with pytest.raises(ValueError, match="already registered"):
        @register_translator("dup")
        def build_two(cfg):  # pragma: no cover - exercised via decorator
            raise AssertionError(cfg)


def test_unknown_translator_fails_clearly():
    with pytest.raises(KeyError, match="available"):
        load_translator({"type": "does-not-exist"})


def test_invalid_variant_fails_clearly():
    with pytest.raises(ValueError, match="variant must be one of"):
        load_translator({"type": "elan-mt", "variant": "nope"})


def test_config_parses_combined_model_id():
    config = TranslatorConfig.from_mapping(
        {"type": "elan-mt:tiny", "verbose": True, "custom": "value"}
    )
    assert config.type == "elan-mt"
    assert config.variant == "tiny"
    assert config.extra == {"custom": "value"}


def test_research_adapter_load_and_unload():
    calls = []

    class Impl:
        is_loaded = False

        def load_model(self):
            calls.append("load")
            self.is_loaded = True

        def unload_model(self):
            calls.append("unload")
            self.is_loaded = False

        def predict(self, texts):
            return [text.upper() for text in texts]

    @register_translator("temp-load")
    def build(cfg):
        from amatra.translation.adapters.research_adapter import ResearchTranslatorAdapter

        return ResearchTranslatorAdapter("temp-load:base", factory=Impl)

    translator = load_translator({"type": "temp-load"})
    with translator:
        assert translator.is_loaded
        result = translator.translate(TranslationRequest(source_texts=["a"]))
        assert result.translations == ["A"]
    assert calls == ["load", "unload"]


def test_output_length_mismatch_fails_clearly():
    class BadTranslator(BaseTranslator):
        model_id = "bad:length"

        def __init__(self):
            self._loaded = False

        def load(self):
            self._loaded = True

        def unload(self):
            self._loaded = False

        @property
        def is_loaded(self):
            return self._loaded

        def translate(self, request: TranslationRequest) -> TranslationResult:
            return TranslationResult(
                translations=["one"],
                source_texts=request.source_texts,
                model_id=self.model_id,
            )

    @register_translator("bad-length")
    def build(cfg):
        return BadTranslator()

    translator = load_translator({"type": "bad-length"})
    with translator:
        result = translator.translate(TranslationRequest(source_texts=["a", "b"]))
        assert len(result.translations) == 1

    from amatra.translation.adapters.research_adapter import ResearchTranslatorAdapter

    adapter = ResearchTranslatorAdapter(
        model_id="bad:length",
        factory=lambda: type(
            "Impl",
            (),
            {
                "is_loaded": True,
                "load_model": lambda self: None,
                "unload_model": lambda self: None,
                "predict": lambda self, texts: ["only-one"],
            },
        )(),
    )
    with pytest.raises(TranslatorBoundaryError, match="request_size=2, output_size=1"):
        adapter.translate(TranslationRequest(source_texts=["a", "b"]))
