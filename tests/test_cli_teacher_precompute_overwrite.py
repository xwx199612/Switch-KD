from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vlm_distill import cli
from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.data_manifest import VlmSample


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            training_manifest_path=tmp_path / "manifest.jsonl",
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            full_label_path=tmp_path / "full-labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher", backend="mock"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd"),
    )


def test_label_forwards_overwrite_to_label_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _config(tmp_path)
    samples = [VlmSample(id="sample-1", image="screen.png", task="qa", query="q")]
    calls: list[bool] = []

    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "validate_manifest", lambda *args, **kwargs: samples)
    monkeypatch.setattr(
        cli,
        "create_label_dataset",
        lambda _config, _samples, *, overwrite=False: calls.append(overwrite) or tmp_path / "labels.jsonl",
    )
    monkeypatch.setattr(sys, "argv", ["vlm-distill", "label", "--config", str(tmp_path / "config.yaml"), "--overwrite"])

    cli.main()

    assert calls == [True]


def test_label_forwards_overwrite_to_teacher_and_labeled_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _config(tmp_path)
    config.training.validation_enabled = True
    config.training.validation_ratio = 0.5
    config.training.validation_label_path = tmp_path / "validation.jsonl"
    config.training.validation_image_dir = tmp_path / "validation-images"
    samples = [VlmSample(id="sample-1", image="screen.png", task="qa", query="q")]
    teacher_calls: list[bool] = []
    split_calls: list[bool] = []

    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "validate_manifest", lambda *args, **kwargs: samples)
    monkeypatch.setattr(
        cli,
        "create_label_dataset",
        lambda _config, _samples, *, overwrite=False: teacher_calls.append(overwrite) or tmp_path / "full-labels.jsonl",
    )
    monkeypatch.setattr(
        cli,
        "split_labeled_dataset",
        lambda **kwargs: split_calls.append(kwargs["overwrite"]) or {"training_count": 1, "validation_count": 0},
    )
    monkeypatch.setattr(sys, "argv", ["vlm-distill", "label", "--config", str(tmp_path / "config.yaml"), "--overwrite"])

    cli.main()

    assert teacher_calls == [True]
    assert split_calls == [True]


def test_label_dry_run_does_not_call_teacher_or_modify_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _config(tmp_path)
    output_path = tmp_path / "labels.jsonl"
    original = '{"id":"old-row"}\n'
    output_path.write_text(original, encoding="utf-8")
    samples = [VlmSample(id="sample-1", image="screen.png", task="qa", query="q")]

    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "validate_manifest", lambda *args, **kwargs: samples)
    monkeypatch.setattr(cli, "create_label_dataset", lambda *args, **kwargs: pytest.fail("teacher called"))
    monkeypatch.setattr(sys, "argv", ["vlm-distill", "label", "--config", str(tmp_path / "config.yaml"), "--overwrite", "--dry-run"])

    cli.main()

    assert output_path.read_text(encoding="utf-8") == original


def test_removed_teacher_command_is_not_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    removed_command = "teacher" + "-precompute"
    monkeypatch.setattr(sys, "argv", ["vlm-distill", removed_command, "--config", str(tmp_path / "config.yaml")])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_cli_help_contains_label_but_not_teacher_precompute(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "argv", ["vlm-distill", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "label" in help_text
    assert ("teacher" + "-precompute") not in help_text
