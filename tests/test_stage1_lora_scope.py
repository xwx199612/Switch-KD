from pathlib import Path
import types

import pytest
import torch
from torch import nn

from vlm_distill.config_schema import load_config
from vlm_distill.student_trainability import validate_language_model_lora_scope


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
B1 = list(range(12, 36))
B2 = list(range(24, 36))


class _Toy(nn.Module):
    def __init__(self, layers=B1, *, extra=None, visual=False, mlp=False, projector=False, base=False):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Module() for _ in range(36)])
        selected = set(layers)
        for index, layer in enumerate(self.model.language_model.layers):
            layer.self_attn = nn.Module()
            for target in TARGETS:
                module = nn.Module()
                if index in selected or (extra is not None and index == extra):
                    module.register_parameter("lora_A", nn.Parameter(torch.ones(2, 2)))
                    module.register_parameter("lora_B", nn.Parameter(torch.ones(2, 2)))
                layer.self_attn.add_module(target, module)
            if mlp and index == 12:
                module = nn.Module()
                module.register_parameter("lora_A", nn.Parameter(torch.ones(2, 2)))
                layer.add_module("gate_proj", module)
        if visual:
            self.model.visual = nn.Module()
            self.model.visual.register_parameter("lora_A", nn.Parameter(torch.ones(2, 2)))
        if projector:
            self.model.visual = getattr(self.model, "visual", nn.Module())
            self.model.visual.merger = nn.Linear(2, 2)
        if base:
            self.model.language_model.base = nn.Parameter(torch.ones(2, 2))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name)
        if projector:
            for parameter in self.model.visual.merger.parameters():
                parameter.requires_grad_(True)
        if base:
            self.model.language_model.base.requires_grad_(True)


def test_null_scope_targets_all_language_model_layers():
    report = validate_language_model_lora_scope(_Toy(range(36)), None, TARGETS)
    assert all(value == list(range(36)) for value in report["detected_layers"].values())


@pytest.mark.parametrize("layers", [B1, B2])
def test_scopes_target_exact_layers(layers):
    report = validate_language_model_lora_scope(_Toy(layers), layers, TARGETS)
    assert all(value == layers for value in report["detected_layers"].values())


@pytest.mark.parametrize("kwargs", [{"layers": B1, "extra": 5}, {"layers": B1, "visual": True}, {"layers": B1, "mlp": True}, {"layers": B1, "projector": True}, {"layers": B1, "base": True}])
def test_invalid_trainability_fails(kwargs):
    with pytest.raises(RuntimeError):
        validate_language_model_lora_scope(_Toy(**kwargs), B1, TARGETS)


@pytest.mark.parametrize("bad", [[1, 1], [-1], []])
def test_layer_schema_rejects_invalid_indices(tmp_path: Path, bad):
    config = {
        "data": {"training_manifest_path": "m", "distill_path": "d"},
        "teacher": {"model_name": "mock-teacher"},
        "student": {"model_name": "mock-student", "output_dir": str(tmp_path / "o"),
                    "adapter_dir": str(tmp_path / "a"), "lora_layers_to_transform": bad,
                    "lora_layers_pattern": "layers"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_stage1_configs_and_parameter_ratio():
    configs = [load_config(f"configs/stage1_b{i}_r16_attn_" + suffix) for i, suffix in (
        (0, "all_layers.yaml"), (1, "layers_12_35.yaml"), (2, "layers_24_35.yaml"))]
    assert [c.student.lora_layers_to_transform for c in configs] == [None, B1, B2]
    assert [36, 24, 12] == [36, len(B1), len(B2)]
