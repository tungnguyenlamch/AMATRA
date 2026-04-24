# AMATRA Integration Guide

This document describes how the `amatra_app` package (in `app/`) consumes
translator/model code from the research tree, and how to add a new
translation model without touching UI code.

## Architecture

```
AMATRA/
├── src/
│   ├── pipeline/                 # research code, source of truth for models
│   │   ├── BaseModel.py
│   │   └── TranslationModels/
│   │       ├── Translator.py
│   │       ├── ElanMtJaEnTranslator.py
│   │       ├── LLMTranslator.py
│   │       └── ...
│   └── amatra/                   # thin integration boundary (stable API)
│       └── translation/
│           ├── types.py          # TranslationRequest / TranslationResult / TranslatorConfig
│           ├── base.py           # BaseTranslator ABC
│           ├── registry.py       # @register_translator
│           ├── factory.py        # load_translator(config)
│           └── adapters/         # one file per concrete model family
└── app/
    └── src/amatra_app/           # Qt/Python app
        ├── _amatra_bootstrap.py  # makes `amatra` importable
        └── main.py               # uses amatra.translation only
```

### Dependency direction

```
app  -->  amatra.translation  -->  pipeline.TranslationModels
```

The research code never imports from `amatra`; the app never imports from
`pipeline`. If you find yourself writing `from pipeline...` inside the app,
that is a bug in the integration layer, not in the app.

## Runtime setup

The app locates the `amatra` package in one of two ways:

1. **Development (default).** Running the app out of the repo checkout
   works out of the box: `app/src/amatra_app/_amatra_bootstrap.py` walks up
   from the app source tree to find `<repo>/src/amatra/` and prepends it to
   `sys.path`. Nothing extra to install.

2. **Installed (optional upgrade path).** If the research tree becomes
   pip-installable (add a repo-root `pyproject.toml` shipping `src/amatra`
   and `src/pipeline`), the bootstrap sees `amatra` already importable and
   is a no-op. No app changes required.

## Adding a new translation model

### Case A: new research class that fits the existing `Translator` contract

1. Add a new module under `src/pipeline/TranslationModels/` that subclasses
   `pipeline.TranslationModels.Translator.Translator` and implements
   `load_model`, `_inference`, and `unload_model` (mirror the existing
   `ElanMtJaEnTranslator`/`LLMTranslator` patterns).
2. Create `src/amatra/translation/adapters/my_model.py`:

   ```python
   from ..base import BaseTranslator
   from ..registry import register_translator
   from ..types import TranslatorConfig
   from .research_adapter import ResearchTranslatorAdapter

   _VARIANTS = ("small", "large")

   @register_translator("my-model", variants=_VARIANTS,
                        description="One-line human description")
   def _build(cfg: TranslatorConfig) -> BaseTranslator:
       variant = cfg.variant or "small"

       def factory():
           from pipeline.TranslationModels.MyModelTranslator import (
               MyModelTranslator,
           )
           return MyModelTranslator(
               variant=variant,
               device=cfg.device,
               verbose=cfg.verbose,
               **cfg.extra,
           )

       return ResearchTranslatorAdapter(
           model_id=f"my-model:{variant}",
           factory=factory,
       )
   ```

3. Register it for import by adding `from . import my_model` to
   `src/amatra/translation/adapters/__init__.py`.

That's it. The app's "AI models → Translation Model" menu auto-populates from
`list_translators()`, so no UI edits are needed.

### Case B: model with a shape that doesn't match the research `Translator`

If your backend is something unusual (external HTTP API, async client, model
that returns structured output, ...), skip `ResearchTranslatorAdapter` and
subclass `BaseTranslator` directly inside the adapter file:

```python
from ..base import BaseTranslator
from ..registry import register_translator
from ..types import TranslationRequest, TranslationResult, TranslatorConfig


class _MyApiTranslator(BaseTranslator):
    def __init__(self, cfg: TranslatorConfig) -> None:
        self.model_id = f"my-api:{cfg.variant or 'default'}"
        self._client = None

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def load(self) -> None:
        ...

    def unload(self) -> None:
        self._client = None

    def translate(self, request: TranslationRequest) -> TranslationResult:
        translations = self._client.batch(list(request.source_texts))
        return TranslationResult(
            translations=translations,
            source_texts=request.source_texts,
            model_id=self.model_id,
        )


@register_translator("my-api", variants=("default",))
def _build(cfg: TranslatorConfig) -> BaseTranslator:
    return _MyApiTranslator(cfg)
```

The research tree is not touched and the app still only sees
`BaseTranslator`.

## Design rules

- The app imports only from `amatra.translation`. Do not add imports of
  `pipeline.*` anywhere in the app.
- Research classes stay unchanged; adapters wrap them. If you need app
  compatibility behavior, add it to the adapter, not the research class.
- Translator ids used by the UI are always `"type:variant"` strings. The UI
  is not allowed to inspect variant semantics - it emits the string and the
  registry/factory does the rest.
- Keep adapter files small: they should be declarative (decorator +
  factory closure) rather than contain inference logic.

## Testing the boundary quickly

From the repo root, a smoke check that does not load any model:

```bash
PYTHONPATH=src python -c "
from amatra.translation import list_translators, load_translator
print([s.type for s in list_translators()])
t = load_translator({'type': 'elan-mt', 'variant': 'base'})
print(t)  # not loaded yet; construction is cheap
"
```
