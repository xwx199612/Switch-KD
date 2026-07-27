from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vlm_distill.data_manifest import validate_manifest
import vlm_distill.manifest_builder as manifest_builder
from vlm_distill.manifest_builder import create_manifest_from_config


def _config(tmp_path: Path, seed: int = 42, validation_manifest: Path | None = None, validation_dir: Path | None = None, enabled: bool = False, ratio: float | None = None, mode: str = "move", split_seed: int | None = None):
    image_dir = tmp_path / "training" / "image"
    manifest = tmp_path / "out" / "training.jsonl"
    return SimpleNamespace(
        seed=seed,
        data=SimpleNamespace(
            training_image_dir=image_dir,
            inference_image_dir=None,
            image_dir=None,
            training_manifest_path=manifest,
            inference_manifest_path=None,
        ),
        training=SimpleNamespace(
            validation_enabled=enabled,
            validation_ratio=ratio,
            validation_split_seed=split_seed,
            validation_split_mode=mode,
            validation_image_dir=validation_dir,
            validation_manifest_path=validation_manifest,
        ),
    )


def _images(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_without_ratio_keeps_original_single_manifest_behavior(tmp_path: Path):
    config = _config(tmp_path)
    _images(config.data.training_image_dir, ["b.png", "a.png"])
    create_manifest_from_config(config, "parsing", "training")
    rows = _rows(config.data.training_manifest_path)
    assert len(rows) == 2
    assert not (tmp_path / "validation").exists()


def test_deterministic_split_and_relative_structure_copy(tmp_path: Path):
    validation_dir = tmp_path / "validation" / "image"
    validation_manifest = tmp_path / "out" / "validation.jsonl"
    config = _config(tmp_path, validation_manifest=validation_manifest, validation_dir=validation_dir, enabled=True, ratio=.2, split_seed=42, mode="copy")
    _images(config.data.training_image_dir, ["netflix/home/001.png", "netflix/home/002.png", "other.png", "z.png", "a.png"])
    create_manifest_from_config(config, "parsing", "training", recursive=True)
    train, validation = _rows(config.data.training_manifest_path), _rows(validation_manifest)
    assert len(train) + len(validation) == 5
    assert not ({row["id"] for row in train} & {row["id"] for row in validation})
    assert all(Path(row["image"]).is_file() for row in validation)
    assert all(str(validation_dir) in row["image"] for row in validation)
    assert any((config.data.training_image_dir / name).is_file() for name in ["netflix/home/001.png", "netflix/home/002.png"])
    assert (validation_manifest.parent / "manifest_split_metadata.json").is_file()
    assert len(validate_manifest(config.data.training_manifest_path)) == len(train)
    assert len(validate_manifest(validation_manifest)) == len(validation)


def test_dry_run_does_not_modify_files(tmp_path: Path):
    validation_manifest = tmp_path / "out" / "validation.jsonl"
    validation_dir = tmp_path / "validation"
    config = _config(tmp_path, validation_manifest=validation_manifest, validation_dir=validation_dir, enabled=True, ratio=.2)
    _images(config.data.training_image_dir, ["a.png", "b.png", "c.png"])
    create_manifest_from_config(config, "parsing", "training", dry_run=True)
    assert not config.data.training_manifest_path.exists()
    assert not validation_manifest.exists()
    assert not validation_dir.exists()


def test_move_removes_source_and_preserves_nested_path(tmp_path: Path):
    validation_manifest = tmp_path / "out" / "validation.jsonl"
    validation_dir = tmp_path / "validation"
    config = _config(tmp_path, validation_manifest=validation_manifest, validation_dir=validation_dir, enabled=True, ratio=.5, mode="move")
    _images(config.data.training_image_dir, ["nested/a.png", "nested/b.png", "c.png"])
    create_manifest_from_config(config, "parsing", "training", recursive=True)
    for row in _rows(validation_manifest):
        assert Path(row["image"]).is_file()
        assert "nested" in row["image"] or row["image"].endswith("c.png")


def test_invalid_ratio_and_conflicts(tmp_path: Path):
    validation_manifest = tmp_path / "out" / "validation.jsonl"
    validation_dir = tmp_path / "validation"
    config = _config(tmp_path, validation_manifest=validation_manifest, validation_dir=validation_dir, enabled=True, ratio=1)
    _images(config.data.training_image_dir, ["a.png", "b.png"])
    with pytest.raises(ValueError, match="0 < validation_ratio"):
        create_manifest_from_config(config, "parsing", "training")
    config.training.validation_ratio = .5
    create_manifest_from_config(config, "parsing", "training")
    with pytest.raises(FileExistsError):
        create_manifest_from_config(config, "parsing", "training")


def test_inference_rejects_validation_options(tmp_path: Path):
    config = _config(tmp_path)
    _images(config.data.training_image_dir, ["a.png"])
    config.training.validation_enabled = True
    with pytest.raises(FileNotFoundError):
        create_manifest_from_config(config, "parsing", "inference")


def test_move_failure_rolls_back_images_and_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    validation_manifest = tmp_path / "out" / "validation.jsonl"
    validation_dir = tmp_path / "validation"
    config = _config(tmp_path, validation_manifest=validation_manifest, validation_dir=validation_dir, enabled=True, ratio=.5, mode="move")
    _images(config.data.training_image_dir, ["a.png", "b.png", "c.png"])
    original_move = manifest_builder.shutil.move
    calls = 0

    def fail_second_move(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated move failure")
        return original_move(source, destination)

    monkeypatch.setattr(manifest_builder.shutil, "move", fail_second_move)
    with pytest.raises(RuntimeError, match="rollback completed"):
        create_manifest_from_config(config, "parsing", "training")
    assert sorted(path.name for path in config.data.training_image_dir.glob("*.png")) == ["a.png", "b.png", "c.png"]
    assert not config.data.training_manifest_path.exists()
    assert not validation_manifest.exists()


def test_cli_no_longer_accepts_validation_dataset_options():
    result = subprocess.run(
        [sys.executable, "-m", "vlm_distill.cli", "create-manifest", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for option in ("--validation-seed", "--validation-mode", "--validation-image-dir", "--validation-manifest-out"):
        assert option not in result.stdout
