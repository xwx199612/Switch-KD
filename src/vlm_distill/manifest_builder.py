from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_schema import (
    PipelineConfig,
    remap_output_path,
    resolve_inference_image_dir,
    resolve_inference_manifest_path,
    resolve_training_image_dir,
    resolve_training_manifest_path,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_IMAGE_DIR = Path("data/images")
DEFAULT_OUTPUT_DIR = remap_output_path(Path("outputs"))

TASK_DEFAULTS = {
    "parsing": {
        "query": (
            "List all visible interactive UI elements on this screen."
        ),
    },
}


def infer_manifest_task_from_config_path(config_path: Path) -> str:
    stem = config_path.stem.casefold()
    if "parsing" in stem:
        return "parsing"
    raise ValueError(
        "Could not infer manifest task from config filename. "
        "Include 'parsing' in the config filename."
    )


def create_manifest_from_config(
    config: PipelineConfig,
    task: str,
    split: str,
    recursive: bool = False,
) -> Path:
    if split == "training":
        image_dir = resolve_training_image_dir(config.data) or DEFAULT_IMAGE_DIR
        output_path = resolve_training_manifest_path(config.data)
    elif split == "inference":
        image_dir = resolve_inference_image_dir(config.data) or DEFAULT_IMAGE_DIR
        output_path = resolve_inference_manifest_path(config.data)
    else:
        raise ValueError(f"Unsupported manifest split: {split}")

    if task == "parsing":
        return create_parsing_manifest(
            image_dir=image_dir,
            output_path=output_path,
            split=split,
            recursive=recursive,
        )

    raise ValueError(
        f"Unsupported task: {task}. "
        f"Available tasks: {sorted(TASK_DEFAULTS)}"
    )


def create_parsing_manifest(
    image_dir: Path,
    output_path: Path,
    split: str,
    recursive: bool = False,
) -> Path:
    query = TASK_DEFAULTS["parsing"]["query"]

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {image_dir}")

    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()

    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, image_path in enumerate(images, start=1):
            row = {
                "id": f"parsing-{index:06d}",
                "image": str(image_path).replace("\\", "/"),
                "task": "parsing",
                "query": query,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Selected split: {split}")
    print(f"Image dir: {image_dir}")
    print(f"Output manifest path: {output_path}")
    print(f"Created parsing manifest: {output_path}")
    print(f"Samples: {len(images)}")

    return output_path
