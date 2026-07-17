from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from vlm_distill.config_schema import load_config
from vlm_distill.config_schema import DataConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.stage_package_adapter_deployment import package_high_fidelity_adapter_deployment
from vlm_distill.student_trainability import (
    QWEN3_VL_ATTENTION_TARGETS,
    QWEN3_VL_MLP_TARGETS,
    resolve_language_model_lora_targets,
    validate_a3_attn_mlp_full_projector_contract,
)


class _FakeQwen3VL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList()
        for _ in range(36):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.mlp = nn.Module()
            for target in QWEN3_VL_ATTENTION_TARGETS:
                setattr(layer.self_attn, target, nn.Linear(2, 2))
            for target in QWEN3_VL_MLP_TARGETS:
                setattr(layer.mlp, target, nn.Linear(2, 2))
            self.model.language_model.layers.append(layer)
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.modules_to_save = nn.ModuleDict({
            "default": nn.ModuleDict({
                "norm": nn.LayerNorm(2, dtype=torch.bfloat16),
                "linear_fc1": nn.Linear(2, 2, dtype=torch.bfloat16),
                "linear_fc2": nn.Linear(2, 2, dtype=torch.bfloat16),
            })
        })
        self.model.visual.merger.original_module = nn.Linear(2, 2)
        self.model.visual.encoder = nn.Linear(2, 2)
        self.model.visual.deepstack_merger_list = nn.ModuleList([nn.Linear(2, 2)])
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(False)
            if ".modules_to_save.default." in name:
                parameter.requires_grad_(True)
        for name, module in self.named_modules():
            if name.endswith((*QWEN3_VL_ATTENTION_TARGETS, *QWEN3_VL_MLP_TARGETS)):
                module.register_parameter("lora_A", nn.Parameter(torch.ones(1, 2, dtype=torch.float32)))
                module.register_parameter("lora_B", nn.Parameter(torch.ones(2, 1, dtype=torch.float32)))
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)


TARGETS = [*QWEN3_VL_ATTENTION_TARGETS, *QWEN3_VL_MLP_TARGETS]


def test_a3_config_and_exact_target_resolution():
    config = load_config("configs/lora_ablation/stage1_a3_r16_attn_mlp_projector.yaml")
    assert config.student.target_modules == TARGETS
    assert config.student.train_multimodal_projector is True
    assert config.student.use_projector_lora is False
    resolved = resolve_language_model_lora_targets(_FakeQwen3VL(), TARGETS)
    assert resolved["attention_module_count"] == 144
    assert resolved["mlp_module_count"] == 108
    assert resolved["total_module_count"] == 252
    assert all(layers == list(range(36)) for layers in resolved["layers"].values())


def test_a3_contract_allows_mlp_and_requires_full_projector():
    report = validate_a3_attn_mlp_full_projector_contract(_FakeQwen3VL())
    assert report["attention_module_count"] == 144
    assert report["mlp_module_count"] == 108
    assert report["projector_lora_tensor_count"] == 0


@pytest.mark.parametrize("bad_kind", [
    "blocks", "deepstack", "other_adapter", "original", "projector_lora",
])
def test_a3_contract_rejects_forbidden_trainables(bad_kind):
    model = _FakeQwen3VL()
    if bad_kind == "blocks":
        model.model.visual.blocks = nn.ModuleList([nn.Linear(2, 2)])
        parameter = model.model.visual.blocks[0].weight
    elif bad_kind == "deepstack":
        parameter = model.model.visual.deepstack_merger_list[0].weight
    elif bad_kind == "other_adapter":
        model.model.visual.merger.modules_to_save["other_adapter"] = nn.Linear(2, 2)
        parameter = model.model.visual.merger.modules_to_save["other_adapter"].weight
    elif bad_kind == "original":
        parameter = model.model.visual.merger.original_module.weight
    else:
        module = model.model.visual.merger.modules_to_save["default"]["linear_fc1"]
        module.lora_A = nn.Parameter(torch.ones(1, 2))
        parameter = module.lora_A
    parameter.requires_grad_(True)
    with pytest.raises(RuntimeError, match="A3 trainability contract failed"):
        validate_a3_attn_mlp_full_projector_contract(model)


def test_unknown_or_visual_targets_are_rejected(tmp_path: Path):
    raw = {
        "data": {"training_manifest_path": "m", "distill_path": "d"},
        "teacher": {"model_name": "mock"},
        "student": {"model_name": "mock", "output_dir": str(tmp_path / "o"),
                    "adapter_dir": str(tmp_path / "a"), "target_modules": ["linear_fc1"]},
    }
    path = tmp_path / "bad.yaml"
    import yaml
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="target_modules"):
        load_config(path)


def test_a3_deployment_config_preserves_a1_experiment_settings():
    config = load_config("configs/lora_ablation/deploy/stage1_a3_4bit_base_bf16_adapter.yaml")
    formal = load_config("configs/lora_ablation/stage1_a3_r16_attn_mlp_projector.yaml")
    assert config.student.merged_artifact_mode == "4bit_base_bf16_adapter"
    assert config.distillation.prompt_template == formal.distillation.prompt_template
    assert config.training.image_resize == formal.training.image_resize == "1080p"
    assert config.teacher.max_new_tokens == formal.teacher.max_new_tokens == 1280


def test_a3_package_metadata_contains_target_groups(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"target_modules": TARGETS}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"not-a-real-checkpoint")
    config = PipelineConfig(
        data=DataConfig(training_manifest_path=tmp_path / "m", distill_path=tmp_path / "d"),
        teacher=TeacherConfig(model_name="mock"),
        student=StudentConfig(
            model_name=str(base), output_dir=tmp_path / "out", adapter_dir=adapter,
            inference_adapter_path=adapter, quantization="4bit", use_lora=True,
            target_modules=TARGETS, train_multimodal_projector=True,
            merged_artifact_mode="4bit_base_bf16_adapter", deployment_artifact_path=tmp_path / "bundle",
        ),
    )
    bundle = package_high_fidelity_adapter_deployment(config)
    metadata = json.loads((bundle / "deployment_config.json").read_text(encoding="utf-8"))
    assert metadata["experiment_mode"] == "attention_mlp_lora_full_projector"
    assert metadata["lora_target_groups"]["attention"] == list(QWEN3_VL_ATTENTION_TARGETS)
    assert metadata["lora_target_groups"]["mlp"] == list(QWEN3_VL_MLP_TARGETS)
    assert metadata["adapter_merged"] is False
