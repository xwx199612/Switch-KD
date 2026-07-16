from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn

from vlm_distill.config_schema import load_config
from vlm_distill.student_trainability import (
    resolve_a2_lora_targets,
    validate_a2_projector_lora_contract,
)


class _ToyA2(nn.Module):
    def __init__(self, *, omit_fc2=False):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList()
        for _ in range(36):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(layer.self_attn, target, nn.Linear(2, 2))
            self.model.language_model.layers.append(layer)
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.linear_fc1 = nn.Linear(2, 2)
        if not omit_fc2:
            self.model.visual.merger.linear_fc2 = nn.Linear(2, 2)
        self.model.visual.deepstack_merger_list = nn.ModuleList([nn.Linear(2, 2)])


def _attach_a2_adapters(model):
    for name, module in model.named_modules():
        if ".self_attn." in name and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            module.lora_A = nn.Parameter(torch.ones(1, 2))
            module.lora_B = nn.Parameter(torch.ones(2, 1))
    for target in ("linear_fc1", "linear_fc2"):
        module = getattr(model.model.visual.merger, target)
        module.lora_A = nn.Parameter(torch.ones(1, 2))
        module.lora_B = nn.Parameter(torch.ones(2, 1))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_A" in name or "lora_B" in name)


def test_a2_resolves_only_main_merger_and_all_36_qkvo_layers():
    resolved = resolve_a2_lora_targets(_ToyA2())
    assert resolved["projector_targets"] == [
        "model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"
    ]
    assert len(resolved["attention_targets"]) == 144
    assert not any("deepstack" in name for name in resolved["all_targets"])


def test_a2_contract_rejects_mlp_and_unknown_trainables():
    model = _ToyA2()
    _attach_a2_adapters(model)
    model.model.language_model.layers[0].mlp = nn.Linear(2, 2)
    model.model.language_model.layers[0].mlp.lora_A = nn.Parameter(torch.ones(1, 2))
    with pytest.raises(RuntimeError, match="illegal parameters"):
        validate_a2_projector_lora_contract(model)


def test_a2_config_and_mutual_exclusion_validation():
    config = load_config("configs/lora_ablation/stage1_a2_r16_attn_projector_lora.yaml")
    assert config.student.use_projector_lora is True
    assert config.student.train_multimodal_projector is False
    assert config.student.projector_lora_rank == 16
    assert config.student.adapter_dir.as_posix().endswith("r16_attn_projector_lora/adapter")

    import yaml
    raw = yaml.safe_load(open("configs/lora_ablation/stage1_a2_r16_attn_projector_lora.yaml"))
    raw["student"]["train_multimodal_projector"] = True
    path = __import__("tempfile").NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(raw, path); path.close()
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(path.name)
