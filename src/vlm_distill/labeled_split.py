from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any

from .teacher_validation import validate_teacher_row


def _repository_commit_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _tool_version() -> str:
    try:
        from importlib.metadata import version
        return version("vlm-distill")
    except Exception:
        return "unknown"


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return Path(name)
    except Exception:
        os.close(fd)
        Path(name).unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _clean_split_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    cleaned.pop("source_image", None)
    return cleaned


def split_labeled_dataset(
    *,
    full_label_path: Path,
    training_label_path: Path,
    validation_label_path: Path,
    ratio: float,
    seed: int,
    overwrite: bool = False,
    dry_run: bool = False,
    repository_commit_sha: str | None = None,
    tool_version: str | None = None,
) -> dict[str, Any]:
    if not 0 < ratio < 1:
        raise ValueError("validation_ratio must satisfy 0 < validation_ratio < 1")
    if not full_label_path.is_file():
        raise FileNotFoundError(f"Full labeled dataset does not exist: {full_label_path}")
    if len({training_label_path, validation_label_path, full_label_path}) != 3:
        raise ValueError("full, training, and validation label paths must be different")

    rows: list[dict[str, Any]] = []
    with full_label_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{full_label_path}:{line_number} is not a JSON object")
                valid, reason = validate_teacher_row(row)
                if not valid:
                    raise ValueError(f"{full_label_path}:{line_number} invalid labeled row: {reason}")
                rows.append(row)
    if not rows:
        raise ValueError(f"No labeled rows found in {full_label_path}")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Labeled rows must have unique non-empty IDs")

    validation_count = max(1, round(len(rows) * ratio))
    if len(rows) > 1 and validation_count >= len(rows):
        validation_count = len(rows) - 1
    ordered = sorted(rows, key=lambda row: str(row.get("image", "")))
    random.Random(seed).shuffle(ordered)
    training_rows = [_clean_split_row(row) for row in ordered[validation_count:]]
    validation_rows = [_clean_split_row(row) for row in ordered[:validation_count]]

    metadata_path = validation_label_path.parent / "labeled_split_metadata.json"
    outputs = [training_label_path, validation_label_path, metadata_path]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("labeled split output already exists; use --overwrite")
    metadata = {
        "full_label_path": str(full_label_path), "full_label_sha256": _sha256_file(full_label_path),
        "training_label_path": str(training_label_path), "validation_label_path": str(validation_label_path),
        "ratio": ratio, "seed": seed, "total_count": len(rows),
        "training_count": len(training_rows), "validation_count": len(validation_rows),
        "training_ids_hash": _ids_hash(training_rows), "validation_ids": [str(row["id"]) for row in validation_rows],
        "validation_ids_hash": _ids_hash(validation_rows), "repository_commit_sha": repository_commit_sha or _repository_commit_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(), "tool_version": tool_version or _tool_version(),
    }
    if dry_run:
        return {"total_count": len(rows), "training_count": len(training_rows), "validation_count": len(validation_rows), "validation_ids": metadata["validation_ids"], "training_label_path": str(training_label_path), "validation_label_path": str(validation_label_path)}

    temporary: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    backup_dir: Path | None = None
    try:
        temporary = [_atomic_jsonl(training_label_path, training_rows), _atomic_jsonl(validation_label_path, validation_rows), _atomic_jsonl(metadata_path, [metadata])]
        if overwrite:
            backup_dir = Path(tempfile.mkdtemp(prefix=".labeled-split-backup-", dir=validation_label_path.parent))
            for index, target in enumerate(outputs):
                if target.exists():
                    backup = backup_dir / str(index)
                    os.replace(target, backup)
                    backups.append((target, backup))
        for temporary_path, target in zip(temporary, outputs):
            os.replace(temporary_path, target)
            installed.append(target)
        if backup_dir:
            for _, backup in backups:
                backup.unlink(missing_ok=True)
            backup_dir.rmdir()
        return metadata
    except Exception as exc:
        for path in temporary:
            path.unlink(missing_ok=True)
        for target in installed:
            if target.exists():
                target.unlink()
        for target, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        if backup_dir and backup_dir.exists():
            backup_dir.rmdir()
        raise RuntimeError(f"labeled split failed: {exc}; rollback completed") from exc
