# Agent Workflows

## Fast local loop

```bash
uv pip install --python 3.12 -e ".[dev]"
make smoke
make test
```

## Add a translator

1. Add the research class under `src/pipeline/TranslationModels/`.
2. Add an adapter under `src/amatra/translation/adapters/`.
3. Register it with `@register_translator(...)`.
4. If it needs prompt knobs, add a preset to `src/amatra/config/translator_presets.v1.json`.
5. Add boundary tests in `tests/test_translation_boundary.py`.

## Run a mock smoke check

```bash
python -m amatra.cli smoke --mock --json
python -m amatra.cli process --input image.png --output-dir output/mock --mock --debug
```

## Process a bundled fixture

```bash
python -m amatra.cli process-fixture --fixture multiple-bubbles --debug
python -m amatra.cli evaluate-fixtures
```

The fixture outputs are deterministic and compare `result.json` against the
fixture golden file.

## Inspect debug artifacts

Each headless run writes a new directory under `output/runs/<run-id>/` with:

- `run.json`
- `input.json`
- `ocr_texts.json`
- `translations.json`
- `bubbles.json`
- `rendered.png`
- `error.txt` on failure

## Update prompts and presets

Edit `src/amatra/config/translator_presets.v1.json`, then rerun:

```bash
python -m amatra.cli translate --model llm:qwen2.5-0.5b --preset manga-dialogue --text こんにちは
python -m amatra.cli evaluate-fixtures
```
