#!/usr/bin/env python3
"""Safely move every other testing image to a sibling backup directory."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
EXPECTED_IMAGE_COUNT = 504
DEFAULT_BACKUP_NAME = "testing_data_removed_half"
_NATURAL_PART = re.compile(r"(\d+)")


def natural_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Sort names naturally, while preserving deterministic case-insensitive order."""
    parts = _NATURAL_PART.split(path.name.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def image_files(image_dir: Path) -> list[Path]:
    return sorted(
        (entry for entry in image_dir.iterdir() if entry.is_file() and entry.suffix.casefold() in IMAGE_EXTENSIONS),
        key=natural_sort_key,
    )


@dataclass(frozen=True)
class Selection:
    images: list[Path]
    kept: list[Path]
    removed: list[Path]


def select_images(image_dir: Path) -> Selection:
    images = image_files(image_dir)
    return Selection(images, images[::2], images[1::2])


def print_report(selection: Selection, image_dir: Path, backup_dir: Path) -> None:
    print(f"Image directory: {image_dir}")
    print(f"Backup directory: {backup_dir}")
    print(f"Found images: {len(selection.images)}")
    if len(selection.images) != EXPECTED_IMAGE_COUNT:
        print(f"WARNING: expected {EXPECTED_IMAGE_COUNT} images, found {len(selection.images)}.", file=sys.stderr)
    print(f"Expected to keep: {len(selection.kept)}")
    print(f"Expected to remove: {len(selection.removed)}")
    print("First 20 kept filenames:")
    for path in selection.kept[:20]:
        print(f"  {path.name}")
    print("First 20 filenames to remove:")
    for path in selection.removed[:20]:
        print(f"  {path.name}")


def apply_selection(selection: Selection, image_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        if not backup_dir.is_dir():
            raise RuntimeError(f"Backup path exists but is not a directory: {backup_dir}")
        if any(backup_dir.iterdir()):
            raise RuntimeError(f"Backup directory is non-empty; refusing to continue: {backup_dir}")
    else:
        backup_dir.mkdir(parents=True)

    conflicts = [path.name for path in selection.removed if (backup_dir / path.name).exists()]
    if conflicts:
        raise RuntimeError(f"Backup filename conflicts; refusing to continue: {', '.join(conflicts)}")

    moved: list[tuple[Path, Path]] = []
    try:
        for source in selection.removed:
            destination = backup_dir / source.name
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        raise

    remaining = image_files(image_dir)
    backed_up = image_files(backup_dir)
    remaining_names = {path.name for path in remaining}
    backed_up_names = {path.name for path in backed_up}
    if len(remaining) != len(selection.kept) or len(backed_up) != len(selection.removed):
        raise RuntimeError("Post-move image counts do not match the planned counts")
    if remaining_names & backed_up_names:
        raise RuntimeError("Post-move verification found duplicate filenames")
    if len(remaining) + len(backed_up) != len(selection.images):
        raise RuntimeError("Post-move verification total does not match the original image count")
    print(f"Applied successfully: {len(remaining)} images remain; {len(backed_up)} moved to backup.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    backup_dir = (args.backup_dir or image_dir.parent / DEFAULT_BACKUP_NAME).resolve()
    if not image_dir.is_dir():
        print(f"Image directory does not exist or is not a directory: {image_dir}", file=sys.stderr)
        return 2
    selection = select_images(image_dir)
    print_report(selection, image_dir, backup_dir)
    if args.apply:
        try:
            apply_selection(selection, image_dir, backup_dir)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        manifest = Path("outputs/lora_ablation/parsing_inference_manifest.jsonl")
        if manifest.exists():
            print(f"Reminder: recreate inference manifest after moving images: {manifest}", file=sys.stderr)
    else:
        print("Dry-run only: no files were moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
