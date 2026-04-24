from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunArtifacts:
    command: str
    output_root: Path = Path("output/runs")
    run_id: str = field(default_factory=utc_now_stamp)

    def __post_init__(self) -> None:
        self.path = self.output_root / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.timings: dict[str, float] = {}

    def stage(self, name: str):
        return _StageTimer(self, name)

    def write_json(self, name: str, payload: Any) -> None:
        with (self.path / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def write_text(self, name: str, text: str) -> None:
        (self.path / name).write_text(text, encoding="utf-8")

    def write_image(self, name: str, array: np.ndarray) -> None:
        Image.fromarray(array).save(self.path / name)

    def record_error(self, exc: BaseException) -> None:
        self.write_text("error.txt", "".join(traceback.format_exception(exc)))

    def finalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched["run_dir"] = str(self.path)
        enriched["timings"] = self.timings
        self.write_json("run.json", enriched)
        return enriched


class _StageTimer:
    def __init__(self, owner: RunArtifacts, name: str) -> None:
        self.owner = owner
        self.name = name
        self.start = 0.0

    def __enter__(self) -> None:
        self.start = perf_counter()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.owner.timings[self.name] = round(perf_counter() - self.start, 6)
