from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.data_manifest import VlmSample
import vlm_distill.stage_teacher_precompute as stage_teacher_precompute


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            training_manifest_path=tmp_path / "manifest.jsonl",
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher", backend="mock"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd"),
    )


def test_teacher_precompute_writes_elements_only_rows_and_json_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_config(tmp_path)
    sample = VlmSample(id="parsing-000001", image="screen.png", task="parsing", query="List UI elements")

    class _Teacher:
        def answer(self, _sample):
            return {
                "teacher_answer": json.dumps(
                    {
                        "elements": [{"text": "Home", "bbox_norm": [1, 2, 3, 4], "focused": False}],
                        "coordinate_system": "normalized_0_1000",
                    }
                )
            }

    monkeypatch.setattr(stage_teacher_precompute, "build_teacher", lambda _config: _Teacher())

    output_path = stage_teacher_precompute.create_teacher_precompute_dataset(config, [sample])
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert set(row.keys()) == {"id", "image", "task", "query", "elements", "coordinate_system"}
    assert "teacher_answer" not in row
    assert "teacher_tokens" not in row
    assert "type" not in row["elements"][0]
    assert not (tmp_path / "raw" / "teacher" / "parsing-000001.txt").exists()
    assert (tmp_path / "json" / "teacher" / "parsing-000001.json").exists()


def test_teacher_precompute_skips_invalid_parsing_rows_and_writes_sidecar_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_config(tmp_path)
    sample = VlmSample(id="parsing-000002", image="screen.png", task="parsing", query="List UI elements")

    class _Teacher:
        def answer(self, _sample):
            return {"teacher_answer": '{"elements":[{"text":"Home"}]}'}

    monkeypatch.setattr(stage_teacher_precompute, "build_teacher", lambda _config: _Teacher())

    output_path = stage_teacher_precompute.create_teacher_precompute_dataset(config, [sample])

    assert output_path.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "raw" / "teacher" / "parsing-000002.txt").exists()

    sidecar_path = tmp_path / "json" / "teacher" / "parsing-000002.json"
    failure_path = tmp_path / "json" / "teacher" / "parse_failures.jsonl"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8").splitlines()[0])

    assert sidecar["usable"] is False
    assert sidecar["elements"] == []
    assert failure["json_sidecar"] == "json/teacher/parsing-000002.json"


def test_format_prompt_is_canonical_source_of_strict_parsing_schema(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.distillation.prompt_template = "List visible UI elements."
    sample = VlmSample(id="parsing-000003", image="screen.png", task="parsing", query="List UI elements")

    prompt = stage_teacher_precompute._format_prompt(config, sample)

    assert "List visible UI elements." in prompt
    assert 'Use ASCII double quotes only: ".' in prompt
    assert 'Do not use smart quotes: “ ”.' in prompt
    assert 'Every element must use exactly these keys: "text", "bbox_norm", "focused".' in prompt
    assert "Do not include type." in prompt
    assert "Do not use alternative bbox key names: bbox, box_norm, bx_norm, bbox norm, bboxNorm, boxed_norm." in prompt
    assert "bbox_norm must be an array of exactly four integers, e.g. [80,120,140,180]." in prompt
    assert 'Include "coordinate_system": "normalized_0_1000".' in prompt
    assert "If unsure about an element bbox, omit that element." in prompt
    assert "Return at most" not in prompt


def test_retry_prompt_uses_same_strict_parsing_rules() -> None:
    sample = VlmSample(id="parsing-000004", image="screen.png", task="parsing", query="List UI elements")

    prompt = stage_teacher_precompute._build_parsing_retry_prompt(sample)

    assert "Prioritize valid JSON over recall." in prompt
    assert 'Every element must use exactly these keys: "text", "bbox_norm", "focused".' in prompt
    assert "bbox_norm must be an array of exactly four integers, e.g. [80,120,140,180]." in prompt
    assert "Do not write bbox_norm as comma-separated text." in prompt
    assert "Return at most" not in prompt
