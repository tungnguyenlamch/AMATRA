# AMATRA

**A**utomated **M**anga **A**nnotation, **T**ranslation, and **R**endering **A**pp.

AMATRA is a manga translation research repository with an integrated Qt
desktop app. The research code in `src/pipeline/` is the source of truth for
segmentation, OCR, and translation models; a thin `amatra.*` boundary layer
exposes them to the app in `app/`.

This repository is a one-time fork of an upstream research project
(and of the [Lucile](https://github.com/nanachi-gh/Lucile) app, whose MIT
license is preserved at [`app/LICENSE`](app/LICENSE)); it has since diverged
and is maintained independently. No upstream synchronisation is intended.

## Table of contents

- [Repository layout](#repository-layout)
- [Quickstart: run the app](#quickstart-run-the-app)
- [Developer setup](#developer-setup)
  - [Clone](#clone)
  - [Research / training environment (conda)](#research--training-environment-conda)
  - [App environment (PySide6)](#app-environment-pyside6)
- [How the pieces fit together](#how-the-pieces-fit-together)
- [Adding a new translation model](#adding-a-new-translation-model)
- [Data](#data)
- [Conventions for contributors and agents](#conventions-for-contributors-and-agents)
- [License and credits](#license-and-credits)

## Repository layout

```
AMATRA/
├── src/
│   ├── pipeline/                 # research code; source of truth for models
│   │   ├── BaseModel.py
│   │   ├── OCRModels/
│   │   ├── SegmentationModels/
│   │   ├── TranslationModels/    # ElanMT, LLM, Google, ContextAware, ...
│   │   ├── Evaluators/           # per-dataset evaluation notebooks + scripts
│   │   └── Utils/
│   ├── amatra/                   # integration boundary (stable public API)
│   │   └── translation/          # BaseTranslator, registry, factory, adapters
│   ├── bubble-detection/         # training code for custom YOLO variants
│   ├── bubble-ocr/
│   ├── bubble-translation/       # fine-tuning utilities
│   └── utils/
├── app/                          # Qt/Python app (amatra-app, `amatra` CLI)
│   ├── pyproject.toml
│   ├── LICENSE                   # MIT, upstream attribution
│   └── src/amatra_app/
├── environments/                 # conda env, dataset bootstrap
├── data/                         # Manga109 + dialogue annotations (see data/DATA.md)
├── models/                       # local checkpoint cache (gitignored)
├── output/                       # experiment outputs (gitignored)
├── docs/
│   └── INTEGRATION.md            # how translation models are plugged in
├── AGENTS.md                     # rules for AI coding agents
└── README.md                     # ← this file
```

Dependency direction between the two Python trees is one-way:

```
app  ──▶  src/amatra/*  ──▶  src/pipeline/*
(UI)     (stable boundary)    (research code)
```

Research code must never import from `amatra` or the app; the app must only
import from `amatra.*`. This keeps research iteration and UI work from
breaking each other.

## Quickstart: run the app

Prereqs: Python 3.12, `pip`, a working torch install, and OpenCV's system
deps (on Debian/Ubuntu: `sudo apt install libgl1 libglib2.0-0`).

```bash
git clone <your-fork-url> AMATRA
cd AMATRA

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e app

amatra            # launches the Qt window
# equivalent: python -m amatra_app.main
```

On first launch the app downloads model weights from Hugging Face (YOLO
segmentation, manga-ocr, the selected translator). A progress dialog shows
status; the window becomes interactive once `"Ready"` appears in the status
bar.

> **How `amatra-app` finds `amatra`:** the app's entry point imports
> `amatra_app._amatra_bootstrap`, which locates `<repo>/src/` and prepends it
> to `sys.path` if the `amatra` package isn't already installed. That lets
> the app consume the research code in-tree without the main repo needing
> to be pip-installed.

## Developer setup

### Clone

```bash
git clone <your-fork-url> AMATRA
```

(No submodules — `app/` is a regular subdirectory of this repo.)

### Research / training environment (conda)

Used for notebooks under `src/pipeline/Evaluators/`, `src/bubble-*`, dataset
work, and running the research classes directly.

```bash
conda env create -f environments/py11.yml
conda activate py11
```

Pin the Hugging Face token used by `environments/set_up_dataset.sh` and some
evaluators:

```bash
chmod +x environments/set_up.sh
./environments/set_up.sh
# edit the generated ../.env and fill HF_TOKEN=...
```

Download the Manga109 + dialogue annotations:

```bash
chmod +x environments/set_up_dataset.sh
./environments/set_up_dataset.sh
```

The `Dockerfile`-based path from the ancestor research repo is deprecated
and not maintained here.

### App environment (PySide6)

The app in `app/` has its own `pyproject.toml`. Typically you don't need the
conda env to run the app:

```bash
python3.12 -m venv .venv-app
source .venv-app/bin/activate
pip install -e app
amatra
```

If you prefer a single environment, install both:

```bash
conda activate py11
pip install -e app
```

## How the pieces fit together

```
UI (amatra_app.main)
  │
  │  load_translator({"type": "elan-mt", "variant": "base"})
  ▼
amatra.translation          ← src/amatra/translation/*
  ├── BaseTranslator        (stable ABC the app depends on)
  ├── TranslationRequest / TranslationResult / TranslatorConfig
  ├── registry + factory    (@register_translator, load_translator)
  └── adapters/             (elan_mt, llm, google, research_adapter)
         │
         │  wraps research Translator subclasses, unchanged
         ▼
pipeline.TranslationModels  ← src/pipeline/TranslationModels/*
  (ElanMtJaEnTranslator, LLMTranslator, GoogleTranslator, ...)
```

- The UI never imports from `pipeline.*`.
- The research classes never import from `amatra.*`.
- Adapters are declarative (decorator + factory closure) and live under
  `src/amatra/translation/adapters/`. See
  [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for the full design doc.

### Sanity-check the boundary

```bash
PYTHONPATH=src python -c "
from amatra.translation import list_translators, load_translator
print([s.type for s in list_translators()])
t = load_translator({'type': 'elan-mt', 'variant': 'tiny'})
print(t)        # lazy: weights not loaded yet
"
```

## Adding a new translation model

Short version (full walkthrough in [`docs/INTEGRATION.md`](docs/INTEGRATION.md)):

1. Add the research class under `src/pipeline/TranslationModels/` as a
   `Translator` subclass (existing convention in that folder).
2. Create `src/amatra/translation/adapters/<my_model>.py`:

   ```python
   from ..registry import register_translator
   from ..types import TranslatorConfig
   from .research_adapter import ResearchTranslatorAdapter

   @register_translator("my-model", variants=("v1",),
                        description="One-line human description")
   def _build(cfg: TranslatorConfig):
       def factory():
           from pipeline.TranslationModels.MyModelTranslator import (
               MyModelTranslator,
           )
           return MyModelTranslator(device=cfg.device, verbose=cfg.verbose)
       return ResearchTranslatorAdapter(
           model_id=f"my-model:{cfg.variant or 'v1'}", factory=factory,
       )
   ```

3. Add `from . import my_model` to `src/amatra/translation/adapters/__init__.py`.

No UI edits required — the app's translation-model menu auto-populates from
`list_translators()`.

## Data

See `data/DATA.md` for dataset provenance, expected layout, and preprocessing
scripts. `environments/set_up_dataset.sh` bootstraps the Manga109-based data
used by the evaluators in `src/pipeline/Evaluators/`.

## Conventions for contributors and agents

- Read [`AGENTS.md`](AGENTS.md) before editing code. It defines the entry
  workflow (AGENTS.md → README.md → relevant code → tests → changes) and the
  working principles (Think Before Coding, Simplicity First, Surgical
  Changes, Goal-Driven Execution).
- Keep edits surgical — don't refactor unrelated code while implementing a
  request.
- Research classes in `src/pipeline/` must not depend on `amatra.*` or the
  app. Breaking this boundary re-couples the app to research internals and
  defeats the purpose of the integration layer.
- New translation models go through `src/amatra/translation/adapters/`, not
  UI code.
- Minimal inline comments only; prefer clear names over narrating code.

## License and credits

- Research code: inherited from the ancestor research repo. This repository
  has diverged and is maintained independently.
- App code in `app/`: derived from the original
  [Lucile](https://github.com/nanachi-gh/Lucile) project (MIT); the upstream
  license is preserved at [`app/LICENSE`](app/LICENSE).
- Downstream model weights (Mitsua ElanMT, `manga-ocr`, YOLO bubble
  segmenters published under the `TheBlindMaster/...-manga-bubble-seg`
  namespace) retain their own licenses; consult each Hugging Face repo.
