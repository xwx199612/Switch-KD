from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig, load_config
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


def test_format_prompt_returns_exact_yaml_formatted_prompt_for_parsing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.distillation.prompt_template = (
        "Task:\n"
        "{query}\n\n"
        "Use this exact schema:\n"
        "{{\n"
        '  "elements": []\n'
        "}}"
    )
    sample = VlmSample(id="parsing-000003", image="screen.png", task="parsing", query="List UI elements")

    prompt = stage_teacher_precompute._format_prompt(config, sample)

    assert prompt == (
        "Task:\n"
        "List UI elements\n\n"
        "Use this exact schema:\n"
        "{\n"
        '  "elements": []\n'
        "}"
    )


def test_retry_prompt_reuses_original_prompt_with_generic_json_retry_prefix() -> None:
    prompt = stage_teacher_precompute._build_parsing_retry_prompt("Task:\nList UI elements")

    assert prompt == (
        "Previous response was not valid JSON. Retry. "
        "Follow the original instructions exactly. Return valid JSON only.\n\n"
        "Task:\nList UI elements"
    )


def test_qwen_parsing_prompt_template_formats_successfully_and_escapes_json_braces() -> None:
    config = load_config("configs/qwen3vl8b_r32_attn_mlp.yaml")
    sample = VlmSample(id="parsing-000004", image="screen.png", task="parsing", query="Find the focused tile.")

    prompt = stage_teacher_precompute._format_prompt(config, sample)

    assert "Task:\nFind the focused tile." in prompt
    assert '"elements": [' in prompt
    assert '"coordinate_system": "normalized_0_1000"' in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt


def test_parsing_output_instructions_function_is_removed() -> None:
    assert not hasattr(stage_teacher_precompute, "_parsing_output_instructions")
