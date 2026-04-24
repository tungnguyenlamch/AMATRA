from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from amatra.runtime.config import build_translator_config
from amatra.runtime.fixtures import list_fixture_names, load_fixture
from amatra.runtime.mock import bubbles_from_fixture, make_rect_mask, render_mock_output, MockTranslator
from amatra.runtime.runs import RunArtifacts
from amatra.translation import TranslationRequest, load_translator


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _result_summary(
    *,
    fixture_name: str | None,
    image_path: Path,
    output_image: Path,
    texts: list[str],
    translations: list[str],
    bboxes: list[list[int]],
) -> dict[str, Any]:
    return {
        "fixture": fixture_name,
        "input_image": str(image_path),
        "output_image": str(output_image),
        "bubble_count": len(bboxes),
        "ocr_texts": texts,
        "translations": translations,
        "bboxes": bboxes,
    }


def _write_process_outputs(
    *,
    output_dir: Path,
    image: np.ndarray,
    result: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_dir / "rendered.png")
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def _matches_golden(actual: dict[str, Any], golden: dict[str, Any]) -> bool:
    stable_keys = {"fixture", "bubble_count", "ocr_texts", "translations", "bboxes"}
    return {key: actual.get(key) for key in stable_keys} == {
        key: golden.get(key) for key in stable_keys
    }


def _process_mock_fixture(
    *,
    fixture_name: str | None,
    image_path: Path,
    spec: dict[str, Any],
    output_dir: Path,
    debug: bool,
) -> dict[str, Any]:
    run = RunArtifacts(command="process")
    try:
        with run.stage("input"):
            image = _load_rgb(image_path)
            run.write_json(
                "input.json",
                {"image": str(image_path), "fixture": fixture_name, "mock": True},
            )

        bubbles = bubbles_from_fixture(spec)
        mapping = {bubble.text: bubble.translation for bubble in bubbles}
        translator = MockTranslator(mapping=mapping)

        with run.stage("translate"):
            request = TranslationRequest(source_texts=[bubble.text for bubble in bubbles])
            translations = translator.translate(request).translations

        with run.stage("render"):
            rendered = render_mock_output(image, bubbles)

        width = int(image.shape[1])
        height = int(image.shape[0])
        bboxes = [bubble.bbox for bubble in bubbles]
        ocr_texts = [bubble.text for bubble in bubbles]
        result = _result_summary(
            fixture_name=fixture_name,
            image_path=image_path,
            output_image=output_dir / "rendered.png",
            texts=ocr_texts,
            translations=translations,
            bboxes=bboxes,
        )

        run.write_json("ocr_texts.json", {"ocr_texts": ocr_texts})
        run.write_json("translations.json", {"translations": translations})
        run.write_json(
            "bubbles.json",
            {
                "bubbles": [
                    {
                        "bbox": bubble.bbox,
                        "ocr_text": bubble.text,
                        "translation": translation,
                    }
                    for bubble, translation in zip(bubbles, translations)
                ]
            },
        )
        if debug:
            for index, bubble in enumerate(bubbles):
                run.write_image(
                    f"mask_{index}.png",
                    make_rect_mask(width, height, bubble.bbox),
                )
        run.write_image("rendered.png", rendered)
        _write_process_outputs(output_dir=output_dir, image=rendered, result=result)
        return run.finalize(result)
    except Exception as exc:
        run.record_error(exc)
        raise


def _process_mock_inferred(
    *,
    input_path: Path,
    output_dir: Path,
    debug: bool,
) -> dict[str, Any]:
    sidecar = input_path.with_suffix(input_path.suffix + ".fixture.json")
    spec: dict[str, Any] = {"bubbles": []}
    if sidecar.is_file():
        spec = json.loads(sidecar.read_text(encoding="utf-8"))
    return _process_mock_fixture(
        fixture_name=None,
        image_path=input_path,
        spec=spec,
        output_dir=output_dir,
        debug=debug,
    )


def _process_real(
    *,
    input_path: Path,
    output_dir: Path,
    translator_id: str,
    preset_name: str | None,
    debug: bool,
) -> dict[str, Any]:
    run = RunArtifacts(command="process")
    try:
        with run.stage("input"):
            run.write_json(
                "input.json",
                {"image": str(input_path), "translator": translator_id, "mock": False},
            )
            image = _load_rgb(input_path)

        with run.stage("segment"):
            from huggingface_hub import hf_hub_download
            from amatra_app.BubbleSegmenter import BubbleSegmenter

            model_path = hf_hub_download(
                repo_id="TheBlindMaster/yolov8s-manga-bubble-seg",
                filename="best.pt",
            )
            segmenter = BubbleSegmenter(model_path)
            _, _, refined = segmenter.detect_and_segment(str(input_path))

        bboxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in [item["bbox"] for item in refined]]
        masks = [item["mask"] for item in refined]
        original_masks = [item["original_mask"] for item in refined]

        with run.stage("ocr"):
            from amatra_app.MangaOCRModel import MangaOCRModel

            ocr = MangaOCRModel()
            ocr.load_model()
            try:
                ocr_texts = ocr.predict(Image.fromarray(image), bboxes)
            finally:
                ocr.unload_model()

        with run.stage("translate"):
            translator = load_translator(
                build_translator_config(
                    translator_id,
                    preset_name=preset_name,
                )
            )
            with translator:
                translations = translator.translate(
                    TranslationRequest(source_texts=ocr_texts)
                ).translations

        with run.stage("render"):
            from amatra_app.MangaTypesetter import MangaTypesetter

            typesetter = MangaTypesetter()
            rendered = typesetter.render(
                image,
                [
                    {"mask": mask, "original_mask": original_mask, "translated_text": text}
                    for mask, original_mask, text in zip(masks, original_masks, translations)
                ],
            )

        result = _result_summary(
            fixture_name=None,
            image_path=input_path,
            output_image=output_dir / "rendered.png",
            texts=ocr_texts,
            translations=translations,
            bboxes=bboxes,
        )
        run.write_json("ocr_texts.json", {"ocr_texts": ocr_texts})
        run.write_json("translations.json", {"translations": translations})
        run.write_json("bubbles.json", {"bboxes": bboxes})
        if debug:
            for index, mask in enumerate(masks):
                run.write_image(f"mask_{index}.png", mask)
        run.write_image("rendered.png", rendered)
        _write_process_outputs(output_dir=output_dir, image=rendered, result=result)
        return run.finalize(result)
    except Exception as exc:
        run.record_error(exc)
        raise


def run_smoke(*, mock: bool) -> dict[str, Any]:
    run = RunArtifacts(command="smoke")
    try:
        if mock:
            with run.stage("mock-translate"):
                translator = MockTranslator()
                result = translator.translate(
                    TranslationRequest(source_texts=["テスト"])
                ).translations
            payload = {
                "ok": result == ["EN::テスト"],
                "message": "mock smoke passed" if result == ["EN::テスト"] else "mock smoke failed",
                "mode": "mock",
            }
        else:
            with run.stage("real-import"):
                translator = load_translator(
                    build_translator_config("elan-mt:tiny", verbose=False)
                )
            payload = {
                "ok": translator.model_id == "elan-mt:tiny",
                "message": "real smoke constructed elan-mt:tiny",
                "mode": "real",
            }
        return run.finalize(payload)
    except Exception as exc:
        run.record_error(exc)
        return run.finalize(
            {
                "ok": False,
                "message": str(exc),
                "mode": "mock" if mock else "real",
            }
        )


def run_translate(
    *,
    model_id: str,
    texts: list[str],
    preset_name: str | None,
    mock: bool,
) -> dict[str, Any]:
    run = RunArtifacts(command="translate")
    try:
        with run.stage("translate"):
            if mock:
                translator = MockTranslator()
            else:
                translator = load_translator(
                    build_translator_config(model_id, preset_name=preset_name)
                )
            with translator:
                result = translator.translate(TranslationRequest(source_texts=texts))
        payload = {
            "model_id": result.model_id,
            "source_texts": list(result.source_texts),
            "translations": result.translations,
        }
        return run.finalize(payload)
    except Exception as exc:
        run.record_error(exc)
        raise


def process_image(
    *,
    input_path: Path,
    output_dir: Path,
    translator_id: str,
    preset_name: str | None,
    mock: bool,
    debug: bool,
) -> dict[str, Any]:
    if mock:
        return _process_mock_inferred(
            input_path=input_path,
            output_dir=output_dir,
            debug=debug,
        )
    return _process_real(
        input_path=input_path,
        output_dir=output_dir,
        translator_id=translator_id,
        preset_name=preset_name,
        debug=debug,
    )


def process_fixture(
    *,
    fixture_name: str,
    output_dir: Path | None,
    preset_name: str | None,
    debug: bool,
) -> dict[str, Any]:
    del preset_name
    fixture = load_fixture(fixture_name)
    destination = output_dir or Path("output/fixtures") / fixture_name
    result = _process_mock_fixture(
        fixture_name=fixture.name,
        image_path=fixture.image_path,
        spec=fixture.spec,
        output_dir=destination,
        debug=debug,
    )
    actual = json.loads((destination / "result.json").read_text(encoding="utf-8"))
    matched = _matches_golden(actual, fixture.golden)
    result["matched_golden"] = matched
    result["golden_path"] = str(fixture.image_path.parent / "golden.json")
    return result


def evaluate_fixtures(
    *,
    output_dir: Path | None,
    debug: bool,
) -> dict[str, Any]:
    passed = 0
    failed = 0
    details = []
    for name in list_fixture_names():
        destination = (output_dir or Path("output/eval")) / name
        result = process_fixture(
            fixture_name=name,
            output_dir=destination,
            preset_name=None,
            debug=debug,
        )
        details.append({"fixture": name, "matched_golden": result["matched_golden"]})
        if result["matched_golden"]:
            passed += 1
        else:
            failed += 1
    return {
        "passed": passed,
        "failed": failed,
        "details": details,
    }
