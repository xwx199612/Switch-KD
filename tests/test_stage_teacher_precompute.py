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
    sample = VlmSample(id="parsing-000001", image="screen.png", query="List UI elements")

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

    output_path = stage_teacher_precompute.create_label_dataset(config, [sample])
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert set(row.keys()) == {"id", "image", "query", "elements", "coordinate_system"}
    assert "teacher_answer" not in row
    assert "teacher_tokens" not in row
    assert "type" not in row["elements"][0]
    assert not (tmp_path / "raw" / "teacher" / "parsing-000001.txt").exists()
    assert (tmp_path / "json" / "teacher" / "parsing-000001.json").exists()


def test_teacher_precompute_text_mode_preserves_generic_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_config(tmp_path)
    config.pipeline.output_mode = "text"
    sample = VlmSample(id="text-000001", image="screen.png", query="What is shown?")

    class _Teacher:
        def answer(self, _sample):
            return {"teacher_answer": "a white square", "teacher_confidence": 1.0}

    monkeypatch.setattr(stage_teacher_precompute, "build_teacher", lambda _config: _Teacher())

    output_path = stage_teacher_precompute.create_label_dataset(config, [sample])
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert row["teacher_answer"] == "a white square"
    assert "elements" not in row


def test_teacher_precompute_skips_invalid_parsing_rows_and_writes_sidecar_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_config(tmp_path)
    sample = VlmSample(id="parsing-000002", image="screen.png", query="List UI elements")

    class _Teacher:
        def answer(self, _sample):
            return {"teacher_answer": '{"elements":[{"text":"Home"}]}'}

    monkeypatch.setattr(stage_teacher_precompute, "build_teacher", lambda _config: _Teacher())

    output_path = stage_teacher_precompute.create_label_dataset(config, [sample])

    assert output_path.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "raw" / "teacher" / "parsing-000002.txt").exists()

    sidecar_path = tmp_path / "json" / "teacher" / "parsing-000002.json"
    failure_path = tmp_path / "json" / "teacher" / "parse_failures.jsonl"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8").splitlines()[0])

    assert sidecar["usable"] is False
    assert sidecar["elements"] == []
    assert failure["json_sidecar"] == "json/teacher/parsing-000002.json"


def test_teacher_precompute_resume_only_calls_teacher_for_pending_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_config(tmp_path)
    config.pipeline.output_mode = "text"
    samples = [VlmSample(id=f"sample-{index}", image="screen.png", query=f"q-{index}") for index in range(5)]
    output_path = tmp_path / "labels.jsonl"
    output_path.write_text(
        "".join(json.dumps({"id": sample.id, "old": True}) + "\n" for sample in samples[:3]),
        encoding="utf-8",
    )
    calls: list[str] = []

    class _Teacher:
        def answer(self, sample):
            calls.append(sample.id)
            return {"teacher_answer": f"new-{sample.id}"}

    monkeypatch.setattr(stage_teacher_precompute, "build_teacher", lambda _config: _Teacher())
    monkeypatch.setattr(
        stage_teacher_precompute,
        "_load_completed_teacher_rows",
        lambda path, *, config: stage_teacher_precompute.CompletedTeacherRows(
            ids={sample.id for sample in samples[:3]},
            valid_count=3,
            invalid_count=0,
            first_invalid_keys=None,
        ),
    )

    stage_teacher_precompute.create_label_dataset(config, samples)

    assert calls == ["sample-3", "sample-4"]
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [f"sample-{index}" for index in range(5)]


def test_teacher_precompute_overwrite_rebuilds_from_empty_and_removes_stale_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _make_config(tmp_path)
    config.pipeline.output_mode = "text"
    samples = [VlmSample(id=f"sample-{index}", image="screen.png", query=f"q-{index}") for index in range(5)]
    output_path = tmp_path / "labels.jsonl"
    output_path.write_text(
        "".join(json.dumps({"id": f"sample-{index}", "old": True}) + "\n" for index in range(3))
        + json.dumps({"id": "stale-id", "old": True})
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    class _Teacher:
        def answer(self, sample):
            calls.append(sample.id)
            return {"teacher_answer": f"new-{sample.id}"}

    monkeypatch.setattr(stage_teacher_precompute, "build_teacher", lambda _config: _Teacher())

    stage_teacher_precompute.create_label_dataset(config, samples, overwrite=True)

    assert calls == [sample.id for sample in samples]
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [sample.id for sample in samples]
    summary = capsys.readouterr().out
    assert "overwrite: true" in summary
    assert "resume_existing_labels: false" in summary
    assert "valid completed label rows: 0" in summary
    assert not (tmp_path / ".labels.jsonl.precompute.tmp").exists()


def test_teacher_precompute_overwrite_failure_preserves_formal_output_and_removes_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _make_config(tmp_path)
    config.pipeline.output_mode = "text"
    samples = [VlmSample(id=f"sample-{index}", image="screen.png", query=f"q-{index}") for index in range(3)]
    output_path = tmp_path / "labels.jsonl"
    original = json.dumps({"id": "old-row", "answer": "keep-me"}) + "\n"
    output_path.write_text(original, encoding="utf-8")
    calls = 0

    class _Teacher:
        def answer(self, sample):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("mock teacher failure")
            return {"teacher_answer": sample.id}

    monkeypatch.setattr(stage_teacher_precompute, "build_teacher", lambda _config: _Teacher())

    with pytest.raises(RuntimeError, match="mock teacher failure"):
        stage_teacher_precompute.create_label_dataset(config, samples, overwrite=True)

    assert output_path.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".labels.jsonl.precompute.tmp").exists()


def test_legacy_prompt_template_is_ignored_for_parsing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.distillation.prompt_template = (
        "Task:\n"
        "{query}\n\n"
        "Use this exact schema:\n"
        "{{\n"
        '  "elements": []\n'
        "}}"
    )
    sample = VlmSample(id="parsing-000003", image="screen.png", query="List UI elements")

    prompt = stage_teacher_precompute._format_prompt(config, sample)

    assert "List UI elements" in prompt
    assert '"elements"' in prompt
    assert "Task:" not in prompt


def test_huggingface_teacher_does_not_retry_parsing_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    sample = VlmSample(id="parsing-000004", image="screen.png", query="List UI elements")
    calls: list[str] = []

    teacher = object.__new__(stage_teacher_precompute.HuggingFaceTeacher)
    teacher.config = config
    monkeypatch.setattr(stage_teacher_precompute, "_load_teacher_image", lambda *_args, **_kwargs: object())

    def _generate(**kwargs):
        calls.append(kwargs["prompt"])
        return '{"elements":[{"text":"Home"}]}', []

    teacher._generate = _generate

    answer = teacher.answer(sample)

    assert answer["teacher_answer"] == '{"elements":[{"text":"Home"}]}'
    assert calls == [stage_teacher_precompute._format_prompt(config, sample)]


def test_retry_prompt_wraps_original_yaml_prompt_without_schema_duplication() -> None:
    prompt = stage_teacher_precompute._build_retry_prompt("Task:\nList UI elements")

    assert prompt == (
        "Previous response was not valid JSON. Retry using the exact same instructions. "
        "Return valid JSON only.\n\n"
        "Task:\nList UI elements"
    )


def test_qwen_legacy_parsing_prompt_is_ignored_and_fixed_schema_is_used() -> None:
    config = load_config("configs/qwen3vl8b_r32_attn_mlp.yaml")
    sample = VlmSample(id="parsing-000005", image="screen.png", query="Find the focused tile.")

    prompt = stage_teacher_precompute._format_prompt(config, sample)

    assert "User instruction:\nFind the focused tile." in prompt
    assert '"elements": [' in prompt
    assert '"coordinate_system": "normalized_0_1000"' in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt


def test_parsing_output_instructions_function_is_removed() -> None:
    assert not hasattr(stage_teacher_precompute, "_parsing_output_instructions")


def test_legacy_parsing_retry_prompt_function_is_removed() -> None:
    assert not hasattr(stage_teacher_precompute, "_build_parsing_retry_prompt")
