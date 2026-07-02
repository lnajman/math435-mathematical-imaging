#!/usr/bin/env python3
"""Execute student notebooks into a temporary validation directory."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "week01_image_formation.ipynb",
    ROOT / "notebooks" / "week02_convolution_blur.ipynb",
]
OUTPUT_DIR = Path("/private/tmp/math435-notebook-validation")


def validate(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
    )
    client.execute()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / path.name
    nbformat.write(notebook, output_path)
    print(f"Executed {path.relative_to(ROOT)} -> {output_path}")


def main() -> None:
    for path in NOTEBOOKS:
        validate(path)


if __name__ == "__main__":
    main()
