from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FixtureResource:
    name: str
    image_path: Path
    spec: dict[str, Any]
    golden: dict[str, Any]


def _fixture_base() -> Path:
    return Path(resources.files("amatra.fixtures"))


def list_fixture_names() -> list[str]:
    base = _fixture_base()
    return sorted(
        path.name
        for path in base.iterdir()
        if path.is_dir() and (path / "fixture.json").is_file()
    )


def load_fixture(name: str) -> FixtureResource:
    base = _fixture_base() / name
    if not base.is_dir() or not (base / "fixture.json").is_file():
        raise ValueError(
            f"unknown fixture {name!r}. available: {list_fixture_names()}"
        )

    with (base / "fixture.json").open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    with (base / "golden.json").open("r", encoding="utf-8") as handle:
        golden = json.load(handle)

    return FixtureResource(
        name=name,
        image_path=base / spec["image"],
        spec=spec,
        golden=golden,
    )
