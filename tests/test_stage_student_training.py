from __future__ import annotations

import types
from pathlib import Path

from vlm_distill.config_schema import DataConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.stage_student_training import _load_student_model


class _DummyModel:
    pass


def _make_config(tmp_path: Path, *, device_map="auto", quantization="none") -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "distill.jsonl",
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(
            model_name="mock-student",
            output_dir=tmp_path / "out",
            adapter_dir=tmp_path / "adapter",
            device_map=device_map,
            quantization=quantization,
        ),
    )


def test_load_student_model_omits_device_map_for_ddp(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(model_name_or_path, **kwargs):
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs
            return _DummyModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = DummyAutoModel
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")

    model, resolved_device_map = _load_student_model(_make_config(tmp_path, device_map=None), "student-model")

    assert isinstance(model, _DummyModel)
    assert resolved_device_map is None
    assert captured["model_name_or_path"] == "student-model"
    assert "device_map" not in captured["kwargs"]
    assert captured["kwargs"]["trust_remote_code"] is True
    assert captured["kwargs"]["local_files_only"] is True


def test_load_student_model_passes_resolved_device_map_without_ddp(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(model_name_or_path, **kwargs):
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs
            return _DummyModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = DummyAutoModel
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    model, resolved_device_map = _load_student_model(_make_config(tmp_path, device_map=None), "student-model")

    assert isinstance(model, _DummyModel)
    assert resolved_device_map == "auto"
    assert captured["kwargs"]["device_map"] == "auto"
