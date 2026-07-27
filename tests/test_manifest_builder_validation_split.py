from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlm_distill.labeled_split import split_labeled_dataset
from vlm_distill.manifest_builder import create_manifest_from_config


def _images(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _config(tmp_path: Path):
    image_dir = tmp_path / "training"
    return SimpleNamespace(
        seed=42,
        data=SimpleNamespace(training_image_dir=image_dir, image_dir=None, training_manifest_path=tmp_path / "raw.jsonl", inference_image_dir=None, inference_manifest_path=None),
        training=SimpleNamespace(validation_enabled=True),
    )


def _labeled(path: Path, image_dir: Path, names: list[str]) -> None:
    rows = []
    for i, name in enumerate(names):
        rows.append({"id": f"id-{i}", "image": str(image_dir / name), "task": "parsing", "query": "列出元件", "elements": [{"text": "按鈕", "bbox_norm": [0, 0, 10, 10], "focused": False}], "coordinate_system": "normalized_0_1000", "teacher_answer": "答案", "teacher_tokens": [1, 2], "teacher_logits": [[0.1]], "metadata": {"非ASCII": "保留"}})
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_create_manifest_only_writes_full_raw_manifest(tmp_path: Path):
    config = _config(tmp_path)
    _images(config.data.training_image_dir, ["app/home/a.png", "b.png"])
    create_manifest_from_config(config, "parsing", "training", recursive=True)
    rows = _rows(config.data.training_manifest_path)
    assert len(rows) == 2
    assert all(set(row) == {"id", "image", "task", "query"} for row in rows)
    assert not (tmp_path / "validation").exists()


def test_labeled_split_preserves_original_images_and_cleans_source_image(tmp_path: Path):
    source = tmp_path / "training"
    names = ["app/home/001.png", "app/home/002.png", "x.png", "y.png", "z.png"]
    _images(source, names)
    full = tmp_path / "full.jsonl"
    _labeled(full, source, names)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation" / "labels.jsonl"
    split_labeled_dataset(full_label_path=full, training_label_path=train, validation_label_path=validation, ratio=.2, seed=42)
    first = _rows(validation)
    assert len(_rows(train)) + len(first) == len(_rows(full))
    assert not ({r["id"] for r in _rows(train)} & {r["id"] for r in first})
    assert first[0]["metadata"]["非ASCII"] == "保留"
    assert first[0]["teacher_logits"] == [[0.1]]
    assert {row["image"] for row in _rows(train) + first} == {str(source / name) for name in names}
    assert all("source_image" not in row for row in _rows(train) + first)
    assert not (tmp_path / "validation-images").exists()


def test_split_overwrite_and_rollback_preserve_existing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "training"
    names = ["a.png", "b.png", "c.png"]
    _images(source, names)
    full = tmp_path / "full.jsonl"
    _labeled(full, source, names)
    train = tmp_path / "train.jsonl"; validation = tmp_path / "val.jsonl"
    split_labeled_dataset(full_label_path=full, training_label_path=train, validation_label_path=validation, ratio=.5, seed=42)
    metadata = validation.parent / "labeled_split_metadata.json"
    split_labeled_dataset(full_label_path=full, training_label_path=train, validation_label_path=validation, ratio=.5, seed=42, overwrite=True)
    assert len(_rows(train)) + len(_rows(validation)) == 3
    assert _rows(metadata)[0]["training_count"] == 1

    original = {path: path.read_text() for path in (train, validation, metadata)}
    monkeypatch.setattr("vlm_distill.labeled_split._atomic_jsonl", lambda *_args: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(RuntimeError, match="rollback completed"):
        split_labeled_dataset(full_label_path=full, training_label_path=train, validation_label_path=validation, ratio=.5, seed=1, overwrite=True)
    assert all(path.read_text() == original[path] for path in (train, validation, metadata))


def test_dry_run_does_not_write_or_copy(tmp_path: Path):
    source = tmp_path / "training"
    _images(source, ["a.png", "b.png"])
    full = tmp_path / "full.jsonl"
    _labeled(full, source, ["a.png", "b.png"])
    result = split_labeled_dataset(full_label_path=full, training_label_path=tmp_path / "train.jsonl", validation_label_path=tmp_path / "val.jsonl", ratio=.5, seed=42, dry_run=True)
    assert result["validation_count"] == 1
    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "validation").exists()
