# Journal

Reverse-chronological log of material repository changes. Per
[`AGENTS.md`](AGENTS.md) step 9, update this file whenever you change
repository files. One entry per logical change, not per commit. Keep each
entry terse: what changed, why, and pointers to the relevant files/docs.

Format:

```
## YYYY-MM-DD — one-line title

- What changed (bullets).
- Why it was needed.
- Pointers: [file](path), [doc](path).
```

---

## 2026-04-24 — Absorb app submodule; rename to `amatra_app`

- Removed the `external/Lucile` git submodule and moved its working tree
  to `app/` as regular files. `.gitmodules` deleted; `.git/config`
  submodule section cleaned.
- Renamed Python package `Lucile_bigshaq9999` → `amatra_app`; directory
  `app/src/Lucile_bigshaq9999/` → `app/src/amatra_app/`. Updated all
  imports, wheel packaging, and CLI script (`lucile` → `amatra`).
- Dropped [`app/src/amatra_app/ElanMtJaEnTranslator.py`](app/src/amatra_app)
  (the deprecated re-implementation shim is unnecessary now that no
  upstream consumer exists).
- Dropped `app/.github/workflows/releases.yml` (upstream CI, broken for
  our new layout; re-add a proper workflow when we want releases).
- Upstream attribution preserved in [`app/LICENSE`](app/LICENSE) and in
  the README credits sections.
- Why: the repo is now a hard fork with no intent to sync upstream, so
  submodule ceremony and upstream-shaped identity were pure overhead.
- Pointers: [app/pyproject.toml](app/pyproject.toml),
  [app/src/amatra_app/main.py](app/src/amatra_app/main.py),
  [app/README.md](app/README.md), [README.md](README.md),
  [docs/INTEGRATION.md](docs/INTEGRATION.md).

## 2026-04-24 — Cover-page README

- Rewrote top-level [`README.md`](README.md) from the ancestor research
  repo's contents into an AMATRA cover page with repo layout, quickstart,
  developer setup (conda + app venv), architecture, "add a new
  translation model", data notes, contributor conventions, license.
- Why: the old README described a different project; developers and
  agents need an accurate entry point.

## 2026-04-24 — Translation adapter boundary (`amatra.translation`)

- Added stable integration package at `src/amatra/translation/`:
  `BaseTranslator` ABC, `TranslationRequest`/`TranslationResult`/
  `TranslatorConfig`, decorator-based registry, `load_translator(config)`
  factory. Adapters for `elan-mt`, `llm`, `google` register at import
  time via `ResearchTranslatorAdapter`, which wraps research translator
  classes without modifying them.
- Added empty `__init__.py` under `src/pipeline/` and
  `src/pipeline/TranslationModels/` so those paths are explicit packages
  for tooling purposes (no runtime behaviour change).
- Refactored app entry ([`app/src/amatra_app/main.py`](app/src/amatra_app/main.py))
  to use `load_translator(...)` and `TranslationRequest` instead of
  directly instantiating research classes. The translation-model menu is
  now built from `list_translators()` — adding a model requires no UI
  edits.
- Added [`app/src/amatra_app/_amatra_bootstrap.py`](app/src/amatra_app/_amatra_bootstrap.py)
  which locates `<repo>/src/amatra/` and prepends it to `sys.path` when
  `amatra` is not already installed.
- Added [`docs/INTEGRATION.md`](docs/INTEGRATION.md) explaining the
  architecture, dependency direction, and how to add new models
  (Case A: fits `Translator` contract; Case B: bespoke backend).
- Why: the app was coupled to concrete research classes (and shipped its
  own divergent `ElanMtJaEnTranslator` copy). A stable boundary decouples
  the UI from backend details and keeps the main repo as the source of
  truth for model logic.
- Pointers: [src/amatra/translation/](src/amatra/translation),
  [docs/INTEGRATION.md](docs/INTEGRATION.md).
