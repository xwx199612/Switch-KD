from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
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
    "parsing": {"query": "List all visible interactive UI elements on this screen."},
}


def infer_manifest_task_from_config_path(config_path: Path) -> str:
    stem = config_path.stem.casefold()
    if "parsing" in stem:
        return "parsing"
    raise ValueError(
        "Could not infer manifest task from config filename. Include 'parsing' in the config filename."
    )


def _image_paths(image_dir: Path, recursive: bool) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {image_dir}")
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix(),
    )


def _manifest_image_path(path: Path) -> str:
    """Use the existing builder convention for both training and validation rows."""
    return str(path).replace("\\", "/")


def _parsing_row(index: int, image_path: Path) -> dict[str, Any]:
    return {
        "id": f"parsing-{index:06d}",
        "image": _manifest_image_path(image_path),
        "task": "parsing",
        "query": TASK_DEFAULTS["parsing"]["query"],
    }


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return Path(temporary_name)
    except Exception:
        os.close(fd)
        Path(temporary_name).unlink(missing_ok=True)
        raise


def create_manifest_from_config(
    config: PipelineConfig,
    task: str,
    split: str,
    recursive: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Path:
    if task not in TASK_DEFAULTS:
        raise ValueError(f"Unsupported task: {task}. Available tasks: {sorted(TASK_DEFAULTS)}")
    if split == "inference":
        image_dir = resolve_inference_image_dir(config.data) or DEFAULT_IMAGE_DIR
        return create_parsing_manifest(image_dir, resolve_inference_manifest_path(config.data), split, recursive)
    if split != "training":
        raise ValueError(f"Unsupported manifest split: {split}")

    image_dir = resolve_training_image_dir(config.data) or DEFAULT_IMAGE_DIR
    output_path = resolve_training_manifest_path(config.data)
    images = _image_paths(image_dir, recursive)
    rows = [_parsing_row(index, path) for index, path in enumerate(images, start=1)]
    print("create-manifest: validation splitting is deferred until label completes")
    print(f"full_raw_manifest={output_path}")
    print(f"samples={len(rows)} dry_run={dry_run}")
    if dry_run:
        return output_path
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {output_path}; use --overwrite")
    return _write_standard_manifest(rows, output_path, split, image_dir)


def _write_standard_manifest(rows: list[dict[str, Any]], output_path: Path, split: str, image_dir: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Selected split: {split}")
    print(f"Image dir: {image_dir}")
    print(f"Output manifest path: {output_path}")
    print(f"Created parsing manifest: {output_path}")
    print(f"Samples: {len(rows)}")
    return output_path


def create_parsing_manifest(image_dir: Path, output_path: Path, split: str, recursive: bool = False) -> Path:
    images = _image_paths(image_dir, recursive)
    return _write_standard_manifest(
        [_parsing_row(index, path) for index, path in enumerate(images, start=1)],
        output_path, split, image_dir,
    )
