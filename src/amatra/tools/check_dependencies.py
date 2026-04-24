from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path


PACKAGE_IMPORTS = {
    "PIL": "pillow",
    "PySide6": "pyside6",
    "cv2": "opencv-python",
    "googletrans": "googletrans",
    "huggingface_hub": "huggingface-hub",
    "manga_ocr": "manga-ocr",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pytest": "pytest",
    "torch": "torch",
    "transformers": "transformers",
    "ultralytics": "ultralytics",
}
LOCAL_PREFIXES = {"amatra", "amatra_app", "pipeline"}
STDLIB_MODULES = set(sys.stdlib_module_names)


def _load_declared() -> set[str]:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = set()
    for dependency in project.get("dependencies", []):
        declared.add(dependency.split("[", 1)[0].split(";", 1)[0].split("==", 1)[0].strip().lower())
    for values in project.get("optional-dependencies", {}).values():
        for dependency in values:
            declared.add(dependency.split("[", 1)[0].split(";", 1)[0].split("==", 1)[0].strip().lower())
    return declared


def _iter_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for file_path in path.rglob("*.py"):
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    return imports


def main() -> int:
    declared = _load_declared()
    imported = _iter_imports(Path("src/amatra")) | _iter_imports(Path("app/src/amatra_app"))
    required = set()
    for name in imported:
        if name in LOCAL_PREFIXES or name in STDLIB_MODULES:
            continue
        package = PACKAGE_IMPORTS.get(name)
        if package:
            required.add(package)

    missing = sorted(package for package in required if package not in declared)
    if missing:
        print("missing declared dependencies:", ", ".join(missing))
        return 1
    print("dependency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
