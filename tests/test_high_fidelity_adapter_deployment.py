from pathlib import Path

import pytest
import yaml

from vlm_distill.config_schema import DataConfig, PipelineConfig, StudentConfig, TeacherConfig, _build_student_config, load_config
from vlm_distill.stage_package_adapter_deployment import package_high_fidelity_adapter_deployment
from vlm_distill.deployment_loader import _processor_is_loadable


def _config(tmp_path: Path, **student_overrides):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    student = StudentConfig(
        model_name=str(base), output_dir=tmp_path / "out", adapter_dir=adapter,
        inference_adapter_path=adapter, quantization="4bit", use_lora=True,
        merged_artifact_mode="4bit_base_bf16_adapter",
        deployment_artifact_path=tmp_path / "deploy",
        **student_overrides,
    )
    return PipelineConfig(
        data=DataConfig(training_manifest_path=tmp_path / "manifest.jsonl", distill_path=tmp_path / "distill.jsonl"),
        teacher=TeacherConfig(model_name="mock-teacher"), student=student,
    )


def test_new_mode_and_deployment_schema_are_parsed(tmp_path):
    raw = {
        "data": {"training_manifest_path": str(tmp_path / "m"), "distill_path": str(tmp_path / "d")},
        "teacher": {"model_name": "mock-teacher"},
        "student": {"model_name": "mock-student", "output_dir": str(tmp_path / "o"),
                    "adapter_dir": str(tmp_path / "a"), "quantization": "4bit", "use_lora": True,
                    "merged_artifact_mode": "4bit_base_bf16_adapter"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).student.merged_artifact_mode == "4bit_base_bf16_adapter"


@pytest.mark.parametrize("updates", [{"quantization": "none"}, {"quantization": "8bit"}, {"use_lora": False}])
def test_high_fidelity_mode_rejects_invalid_quantization_or_lora(tmp_path, updates):
    with pytest.raises(ValueError):
        _build_student_config({"model_name": "m", "output_dir": str(tmp_path / "o"),
                               "adapter_dir": str(tmp_path / "a"),
                               "quantization": "4bit", "use_lora": True,
                               "merged_artifact_mode": "4bit_base_bf16_adapter", **updates})


def test_package_bundle_omits_incomplete_processor_and_records_base_fallback(tmp_path):
    bundle = package_high_fidelity_adapter_deployment(_config(tmp_path))
    assert (bundle / "deployment_config.json").exists()
    assert (bundle / "adapter" / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert not (bundle / "processor").exists()
    assert __import__("json").loads((bundle / "deployment_config.json").read_text())["processor_path"] is None
    assert not (bundle / "base").exists()


def test_package_keeps_processor_only_when_autoprocessor_loads(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", classmethod(lambda cls, *args, **kwargs: object()))
    bundle = package_high_fidelity_adapter_deployment(_config(tmp_path))
    assert (bundle / "processor" / "tokenizer_config.json").exists()
    assert __import__("json").loads((bundle / "deployment_config.json").read_text())["processor_path"] == "processor"


def test_empty_or_incomplete_processor_bundles_are_not_loadable(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert not _processor_is_loadable(empty)
    assert not _processor_is_loadable(incomplete)


@pytest.mark.parametrize("name", ["a0", "a1", "a2"])
def test_deployment_config_preserves_formal_evaluation_settings(name):
    formal_names = {
        "a0": "stage1_a0_r16_attn.yaml",
        "a1": "stage1_a1_r16_attn_projector.yaml",
        "a2": "stage1_a2_r16_attn_projector_lora.yaml",
    }
    deployment = load_config(Path("configs/lora_ablation/deploy") / f"stage1_{name}_4bit_base_bf16_adapter.yaml")
    formal = load_config(Path("configs/lora_ablation") / formal_names[name])
    assert deployment.distillation.prompt_template == formal.distillation.prompt_template
    assert deployment.training.image_resize == formal.training.image_resize == "1080p"
    assert deployment.teacher.image_resize == formal.teacher.image_resize == "1080p"
    assert deployment.teacher.max_new_tokens == formal.teacher.max_new_tokens == 1280
    assert deployment.student.attn_implementation == formal.student.attn_implementation
    assert deployment.teacher.attn_implementation == formal.teacher.attn_implementation
    assert deployment.data.inference_manifest_path == formal.data.inference_manifest_path
    assert deployment.data.label_path == formal.data.label_path
    assert deployment.evaluation == formal.evaluation
    assert deployment.student.merged_artifact_mode == "4bit_base_bf16_adapter"


@pytest.mark.parametrize("projector_mode,expected", [
    ("base", "base_bf16"), ("a1", "modules_to_save"), ("a2", "projector_lora")
])
def test_a0_a1_a2_metadata(tmp_path, projector_mode, expected):
    updates = {}
    if projector_mode == "a1":
        updates["train_multimodal_projector"] = True
    if projector_mode == "a2":
        updates["use_projector_lora"] = True
    bundle = package_high_fidelity_adapter_deployment(_config(tmp_path, **updates))
    import json
    metadata = json.loads((bundle / "deployment_config.json").read_text())
    assert metadata["projector_mode"] == expected
    assert metadata["adapter_merged"] is False
    assert metadata["base_model_copied"] is False
