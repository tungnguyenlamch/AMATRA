"""Locate and expose the ``amatra`` integration package from the main repo.

When the app runs as a git submodule inside the AMATRA research repo, the
``amatra`` package lives at ``<main_repo>/src/amatra``. This module walks up
the directory tree looking for that location and prepends it to ``sys.path``.

If ``amatra`` is already importable (for example, because the main repo was
pip-installed), this is a no-op. Import this module once, before any
``import amatra`` call.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys


def _ensure_amatra_on_path() -> None:
    if importlib.util.find_spec("amatra") is not None:
        return

    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src"
        if (candidate / "amatra" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return

    raise ImportError(
        "Could not locate the `amatra` integration package. "
        "Run the app from within the AMATRA main research repo "
        "(so `<main_repo>/src/amatra/__init__.py` exists), or "
        "pip-install amatra in the current environment."
    )


_ensure_amatra_on_path()
