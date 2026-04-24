# Feature map

Reference for locating code, docs, config, and tests per feature area. Kept
short and pragmatic. If you add a feature, add or update a row here so
future agents (and you) can find it.

> Tests: there is **no** test suite in this repo yet. Where "Tests" is blank
> below, that is the current state, not an oversight on this page. Add tests
> alongside new features and record them here.

## Translation adapter boundary (public API for the app)

Stable `amatra.translation` interface that the app depends on. Research
translators are wrapped by adapters; adding a new model means adding an
adapter, not editing UI code.

| Kind   | Path |
|--------|------|
| Code   | [`src/amatra/translation/__init__.py`](../src/amatra/translation/__init__.py), [`base.py`](../src/amatra/translation/base.py), [`types.py`](../src/amatra/translation/types.py), [`registry.py`](../src/amatra/translation/registry.py), [`factory.py`](../src/amatra/translation/factory.py) |
| Adapters | [`src/amatra/translation/adapters/research_adapter.py`](../src/amatra/translation/adapters/research_adapter.py), [`elan_mt.py`](../src/amatra/translation/adapters/elan_mt.py), [`llm.py`](../src/amatra/translation/adapters/llm.py), [`google.py`](../src/amatra/translation/adapters/google.py) |
| Docs   | [`docs/INTEGRATION.md`](INTEGRATION.md), "How the pieces fit together" in [`README.md`](../README.md) |
| Config | none (registration is via `@register_translator` at import time) |
| Tests  | — |

## Research: translation models

Research-side translator classes. Source of truth for model logic. Must not
import from `amatra.*` or the app.

| Kind   | Path |
|--------|------|
| Base   | [`src/pipeline/BaseModel.py`](../src/pipeline/BaseModel.py), [`src/pipeline/TranslationModels/Translator.py`](../src/pipeline/TranslationModels/Translator.py) |
| Models | [`src/pipeline/TranslationModels/ElanMtJaEnTranslator.py`](../src/pipeline/TranslationModels/ElanMtJaEnTranslator.py), [`LLMTranslator.py`](../src/pipeline/TranslationModels/LLMTranslator.py), [`GoogleTranslator.py`](../src/pipeline/TranslationModels/GoogleTranslator.py), [`ContextAwareLLMTranslator.py`](../src/pipeline/TranslationModels/ContextAwareLLMTranslator.py), [`ThinkingLLMTranslator.py`](../src/pipeline/TranslationModels/ThinkingLLMTranslator.py), [`ElanMtJaEnBatchTranslator.py`](../src/pipeline/TranslationModels/ElanMtJaEnBatchTranslator.py), [`LLMPerImageTranslator.py`](../src/pipeline/TranslationModels/LLMPerImageTranslator.py) |
| Evaluators | [`src/pipeline/Evaluators/Translators/`](../src/pipeline/Evaluators/Translators) (FloresPlus, NTRex, OpenMantra, Opus100) |
| Training utils | [`src/bubble-translation/page-level/finetune_utils/`](../src/bubble-translation/page-level/finetune_utils) |
| Docs   | — (each module is self-describing; see `Translator` docstring) |
| Tests  | — |

## Research: OCR models

| Kind   | Path |
|--------|------|
| Base   | [`src/pipeline/OCRModels/BaseOCRModel.py`](../src/pipeline/OCRModels/BaseOCRModel.py) |
| Models | [`src/pipeline/OCRModels/MangaOCRModel.py`](../src/pipeline/OCRModels/MangaOCRModel.py) |
| Evaluators | [`src/pipeline/Evaluators/OCR/`](../src/pipeline/Evaluators/OCR) |
| Tests  | — |

## Research: segmentation models

| Kind   | Path |
|--------|------|
| Base   | [`src/pipeline/SegmentationModels/BaseSegmentationModel.py`](../src/pipeline/SegmentationModels/BaseSegmentationModel.py) |
| Models | [`src/pipeline/SegmentationModels/YoloSeg.py`](../src/pipeline/SegmentationModels/YoloSeg.py), [`BubbleSegmentationWithOrdering.py`](../src/pipeline/SegmentationModels/BubbleSegmentationWithOrdering.py), [`BubbleSegmenterWithSplit.py`](../src/pipeline/SegmentationModels/BubbleSegmenterWithSplit.py) |
| Training | [`src/bubble-detection/`](../src/bubble-detection) |
| Evaluators | [`src/pipeline/Evaluators/Segmentation/`](../src/pipeline/Evaluators/Segmentation) |
| Tests  | — |

## Research: end-to-end pipelines

| Kind   | Path |
|--------|------|
| Pipeline utility | [`src/pipeline/Utils/MangaPipeline.py`](../src/pipeline/Utils/MangaPipeline.py) |
| Notebooks | [`src/pipeline/Pipelines/pipeline1.ipynb`](../src/pipeline/Pipelines/pipeline1.ipynb) … `pipeline8.ipynb` |
| End-to-end evaluator | [`src/pipeline/Evaluators/Pipeline/OpenMantra/`](../src/pipeline/Evaluators/Pipeline/OpenMantra) |
| Tests  | — |

## Desktop app (amatra-app)

Qt/PySide6 app. Depends only on `amatra.translation` for translators;
segmentation/OCR/typesetter classes currently live inside the app package
(candidates for future migration behind the boundary).

| Kind   | Path |
|--------|------|
| Entry point | [`app/src/amatra_app/main.py`](../app/src/amatra_app/main.py) (`amatra` console script) |
| Bootstrap   | [`app/src/amatra_app/_amatra_bootstrap.py`](../app/src/amatra_app/_amatra_bootstrap.py) (walks up to find `<repo>/src/amatra`) |
| Segmentation wrapper | [`app/src/amatra_app/BubbleSegmenter.py`](../app/src/amatra_app/BubbleSegmenter.py) |
| OCR wrapper          | [`app/src/amatra_app/MangaOCRModel.py`](../app/src/amatra_app/MangaOCRModel.py) |
| Typesetter           | [`app/src/amatra_app/MangaTypesetter.py`](../app/src/amatra_app/MangaTypesetter.py) |
| Packaging   | [`app/pyproject.toml`](../app/pyproject.toml) |
| Licence     | [`app/LICENSE`](../app/LICENSE) (MIT; original Lucile attribution) |
| Docs        | [`app/README.md`](../app/README.md), [`README.md`](../README.md) §Quickstart |
| Tests       | — |

## Data and environments

| Kind   | Path |
|--------|------|
| Dataset setup | [`environments/set_up.sh`](../environments/set_up.sh), [`set_up_dataset.sh`](../environments/set_up_dataset.sh), [`get_hf_token.py`](../environments/get_hf_token.py), [`get_model_pth_file.py`](../environments/get_model_pth_file.py) |
| Conda env  | [`environments/py11.yml`](../environments/py11.yml) |
| Data root  | [`data/`](../data) (gitignored checkouts; see `data/DATA.md`) |
| Model cache | [`models/`](../models) (gitignored) |
| Outputs    | [`output/`](../output) (gitignored) |
| Docker (deprecated) | [`Dockerfile`](../Dockerfile) |

## Repo-level docs and rules

| Kind   | Path |
|--------|------|
| Cover page | [`README.md`](../README.md) |
| Agent rules | [`AGENTS.md`](../AGENTS.md) |
| Integration design | [`docs/INTEGRATION.md`](INTEGRATION.md) |
| Change log | [`JOURNAL.md`](../JOURNAL.md) |
| Feature map | [`docs/feature-map.md`](feature-map.md) (this file) |
