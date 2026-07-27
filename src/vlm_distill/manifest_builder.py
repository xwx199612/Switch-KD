from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import random
import shutil
import subprocess
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


def _commit_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _tool_version() -> str:
    try:
        return version("vlm-distill")
    except PackageNotFoundError:
        return "unknown"


def _split_metadata_path(validation_manifest: Path) -> Path:
    return validation_manifest.parent / "manifest_split_metadata.json"


def _validation_split(
    rows: list[dict[str, Any]],
    source_image_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    validation_image_dir: Path,
    ratio: float,
    seed: int,
    mode: str,
    dry_run: bool,
    overwrite: bool,
    task: str,
) -> Path:
    if not 0 < ratio < 1:
        raise ValueError("validation_ratio must satisfy 0 < validation_ratio < 1")
    if mode not in {"move", "copy"}:
        raise ValueError("validation_mode must be 'move' or 'copy'")
    if training_manifest == validation_manifest:
        raise ValueError("training and validation manifest paths must be different")

    total = len(rows)
    validation_count = max(1, round(total * ratio))
    if validation_count >= total and total > 1:
        validation_count = total - 1
    if total == 0:
        raise ValueError("Cannot create a validation split from zero images")

    shuffled = sorted(rows, key=lambda row: (str(row["id"]), str(row["image"])))
    random.Random(seed).shuffle(shuffled)
    validation_rows = shuffled[:validation_count]
    training_rows = shuffled[validation_count:]
    train_ids = {row["id"] for row in training_rows}
    validation_ids = {row["id"] for row in validation_rows}
    if train_ids & validation_ids or len(train_ids) != len(training_rows):
        raise ValueError("training/validation IDs overlap or are not unique")

    operations: list[tuple[Path, Path, dict[str, Any]]] = []
    for row in validation_rows:
        source = Path(row["image"])
        try:
            relative = source.relative_to(source_image_dir)
        except ValueError:
            relative = Path(source.name)
        destination = validation_image_dir / relative
        updated = dict(row)
        updated["image"] = _manifest_image_path(destination)
        operations.append((source, destination, updated))
    validation_rows = [updated for _, _, updated in operations]

    missing = sum(not source.is_file() for source, _, _ in operations)
    destination_conflicts = sum(destination.exists() for _, destination, _ in operations)
    output_conflicts = sum(path.exists() for path in (training_manifest, validation_manifest, _split_metadata_path(validation_manifest)))
    if missing:
        raise FileNotFoundError(f"{missing} validation source image(s) do not exist")
    if destination_conflicts and not overwrite:
        raise FileExistsError("validation destination contains existing image(s); use --overwrite")
    if output_conflicts and not overwrite:
        raise FileExistsError("manifest or split metadata already exists; use --overwrite")

    fingerprint = hashlib.sha256(
        "\n".join(sorted(str(Path(row["image"]).as_posix()) for row in rows)).encode("utf-8")
    ).hexdigest()
    metadata = {
        "source_image_directory": str(source_image_dir),
        "training_manifest_path": str(training_manifest),
        "validation_manifest_path": str(validation_manifest),
        "validation_image_directory": str(validation_image_dir),
        "ratio": ratio,
        "seed": seed,
        "mode": mode,
        "total_count": total,
        "training_count": len(training_rows),
        "validation_count": len(validation_rows),
        "validation_ids": [row["id"] for row in validation_rows],
        "source_image_fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit_sha": _commit_sha(),
        "tool_version": _tool_version(),
    }

    print("create-manifest summary:")
    for key, value in {
        "task": task, "source_split": "training", "source_image_dir": source_image_dir,
        "training_manifest": training_manifest, "validation_manifest": validation_manifest,
        "validation_image_dir": validation_image_dir, "validation_mode": mode,
        "validation_ratio": ratio, "validation_seed": seed, "total": total,
        "training": len(training_rows), "validation": len(validation_rows),
        "missing": missing, "destination_conflicts": destination_conflicts,
        "dry_run": dry_run, "overlap": len(train_ids & validation_ids),
    }.items():
        print(f"{key}={value}")
    print("validation_preview:")
    for source, _, _ in operations[:10]:
        print(f"  {source}")
    if dry_run:
        return training_manifest

    temporary_paths: list[Path] = []
    moved: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    output_paths = [training_manifest, validation_manifest, _split_metadata_path(validation_manifest)]
    backup_dir: Path | None = None
    try:
        validation_manifest.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and (destination_conflicts or output_conflicts):
            backup_dir = Path(tempfile.mkdtemp(prefix=".manifest-overwrite-", dir=validation_manifest.parent))
            for output_path in output_paths:
                if output_path.exists():
                    backup = backup_dir / f"output-{len(backups)}"
                    shutil.move(str(output_path), str(backup))
                    backups.append((output_path, backup))
        for source, destination, _ in operations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and overwrite:
                assert backup_dir is not None
                backup = backup_dir / str(len(backups))
                shutil.move(str(destination), str(backup))
                backups.append((destination, backup))
            if mode == "move":
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
            else:
                shutil.copy2(source, destination)
        temporary_paths = [_write_jsonl_atomic(training_manifest, training_rows), _write_jsonl_atomic(validation_manifest, validation_rows)]
        metadata_path = _split_metadata_path(validation_manifest)
        temporary_paths.append(_write_jsonl_atomic(metadata_path, [metadata]))
        os.replace(temporary_paths[0], training_manifest)
        os.replace(temporary_paths[1], validation_manifest)
        os.replace(temporary_paths[2], metadata_path)
        if backup_dir:
            shutil.rmtree(backup_dir)
    except Exception as exc:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        for output_path in output_paths:
            if output_path.exists() and not any(output_path == original for original, _ in backups):
                output_path.unlink()
        rollback_errors: list[str] = []
        for source, destination in reversed(moved):
            try:
                if destination.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        if mode == "copy":
            for _, destination, _ in operations:
                if destination.exists() and not any(destination == original for original, _ in backups):
                    destination.unlink()
        for destination, backup in reversed(backups):
            try:
                if backup.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(backup), str(destination))
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        detail = "; rollback errors: " + ", ".join(rollback_errors) if rollback_errors else "; rollback completed"
        raise RuntimeError(f"validation split failed: {exc}{detail}") from exc
    return training_manifest


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
