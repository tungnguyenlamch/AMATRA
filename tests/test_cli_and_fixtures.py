from __future__ import annotations

import json
from pathlib import Path

from amatra.cli import main
from amatra.runtime.fixtures import list_fixture_names


def test_mock_smoke(capsys):
    exit_code = main(["smoke", "--mock", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "mock"


def test_process_fixture_matches_golden(tmp_path, capsys):
    exit_code = main(
        [
            "process-fixture",
            "--fixture",
            "multiple-bubbles",
            "--output-dir",
            str(tmp_path),
            "--debug",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bubble_count"] == 3
    assert payload["matched_golden"] is True


def test_process_mock_with_sidecar(tmp_path, capsys):
    image_path = tmp_path / "page.pgm"
    image_path.write_text(
        "\n".join(
            [
                "P2",
                "4 4",
                "255",
                "255 255 255 255",
                "255 255 255 255",
                "255 255 255 255",
                "255 255 255 255",
            ]
        ),
        encoding="ascii",
    )
    sidecar = tmp_path / "page.pgm.fixture.json"
    sidecar.write_text(
        json.dumps(
            {
                "bubbles": [
                    {
                        "bbox": [0, 0, 2, 2],
                        "ocr_text": "テスト",
                        "translation": "Test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "process",
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--mock",
            "--debug",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bubble_count"] == 1
    assert (output_dir / "rendered.png").is_file()
    assert (output_dir / "result.json").is_file()


def test_evaluate_fixtures(tmp_path, capsys):
    exit_code = main(["evaluate-fixtures", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] + payload["failed"] == len(list_fixture_names())
    assert payload["failed"] == 0


def test_bootstrap_noop_when_amatra_importable():
    from amatra_app import _amatra_bootstrap

    before = list(__import__("sys").path)
    _amatra_bootstrap._ensure_amatra_on_path()
    after = list(__import__("sys").path)
    assert before == after
