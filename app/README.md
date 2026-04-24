# amatra-app

Qt desktop app for AMATRA: manga bubble segmentation, OCR, translation, and
typesetting. Lives in this repo at `app/`; consumed only through the stable
`amatra.translation` boundary (research/model code lives in `../src/pipeline`).

## Install (editable, from the repo root)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e app
amatra
```

On first launch the app downloads model weights from Hugging Face; the
progress dialog shows status until `"Ready"` appears in the status bar.

## Package layout

```
app/
├── pyproject.toml
├── LICENSE              # MIT (attribution kept; see below)
├── README.md            # this file
└── src/
    └── amatra_app/
        ├── __init__.py
        ├── _amatra_bootstrap.py    # puts <repo>/src on sys.path so `amatra` imports resolve
        ├── main.py                 # Qt entry point (`amatra` console script)
        ├── BubbleSegmenter.py
        ├── MangaOCRModel.py
        └── MangaTypesetter.py
```

## Adding a new translation model

Don't edit this package. Register a new adapter under
`../src/amatra/translation/adapters/`; the UI menu auto-populates. See
[`../docs/INTEGRATION.md`](../docs/INTEGRATION.md).

## Attribution

Derived from the original **Lucile** app
(<https://github.com/nanachi-gh/Lucile>), MIT-licensed by its original
author. The upstream MIT license is preserved in [`LICENSE`](LICENSE). This
directory has diverged substantially from that upstream and is maintained
within the AMATRA repository; no synchronisation with the original is
intended.
