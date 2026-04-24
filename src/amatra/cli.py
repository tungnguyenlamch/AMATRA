from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runtime.headless import (
    evaluate_fixtures,
    process_fixture,
    process_image,
    run_smoke,
    run_translate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amatra")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a fast environment smoke check")
    smoke.add_argument("--mock", action="store_true", help="Use mock backends only")
    smoke.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    translate = subparsers.add_parser("translate", help="Translate text through a selected model")
    translate.add_argument("--model", required=True, help="Translator id in type[:variant] form")
    translate.add_argument("--text", nargs="+", required=True, help="Text to translate")
    translate.add_argument("--preset", help="Translator preset name")
    translate.add_argument("--mock", action="store_true", help="Use mock translation")

    process = subparsers.add_parser("process", help="Process an image end-to-end")
    process.add_argument("--input", required=True, help="Input image path")
    process.add_argument("--output-dir", required=True, help="Directory for outputs")
    process.add_argument("--translator", default="elan-mt:base", help="Translator id in type[:variant] form")
    process.add_argument("--preset", help="Translator preset name")
    process.add_argument("--mock", action="store_true", help="Use mock segmentation/OCR/translation")
    process.add_argument("--debug", action="store_true", help="Write detailed run artifacts")

    fixture = subparsers.add_parser("process-fixture", help="Run a bundled fixture and compare against golden output")
    fixture.add_argument("--fixture", required=True, help="Fixture name")
    fixture.add_argument("--output-dir", help="Directory for generated outputs")
    fixture.add_argument("--translator", default="mock", help="Translator id for non-mock runs")
    fixture.add_argument("--preset", help="Translator preset name")
    fixture.add_argument("--debug", action="store_true", help="Write detailed run artifacts")

    evaluate = subparsers.add_parser("evaluate-fixtures", help="Run all bundled fixtures and summarize results")
    evaluate.add_argument("--output-dir", help="Directory for generated outputs")
    evaluate.add_argument("--debug", action="store_true", help="Write detailed run artifacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        result = run_smoke(mock=args.mock)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(result["message"])
        return 0 if result["ok"] else 1

    if args.command == "translate":
        result = run_translate(
            model_id=args.model,
            texts=args.text,
            preset_name=args.preset,
            mock=args.mock,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "process":
        result = process_image(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            translator_id=args.translator,
            preset_name=args.preset,
            mock=args.mock,
            debug=args.debug,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "process-fixture":
        result = process_fixture(
            fixture_name=args.fixture,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            preset_name=args.preset,
            debug=args.debug,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["matched_golden"] else 1

    if args.command == "evaluate-fixtures":
        result = evaluate_fixtures(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            debug=args.debug,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["failed"] == 0 else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
