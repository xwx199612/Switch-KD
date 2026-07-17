from __future__ import annotations

import pytest
import torch
from torch import nn

from vlm_distill.deployment_loader import collect_runtime_lora_inventory, _summary


ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def _lora_target(target: str, *, components=("A", "B")) -> nn.Module:
    module = nn.Module()
    for component in components:
        setattr(module, f"lora_{component}", nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)}))
    return module


def _runtime_model(*, layers=range(36), components=("A", "B"), extra=None):
    root = nn.Module()
    root.base_model = nn.Module()
    root.base_model.model = nn.Module()
    root.base_model.model.model = nn.Module()
    lm = root.base_model.model.model.language_model = nn.Module()
    lm.layers = nn.ModuleList()
    for layer_index in layers:
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.mlp = nn.Module()
        for target in ATTENTION:
            setattr(layer.self_attn, target, _lora_target(target, components=components))
        for target in MLP:
            setattr(layer.mlp, target, _lora_target(target, components=components))
        lm.layers.append(layer)
    if extra is not None:
        root.base_model.model.model.visual = nn.Module()
        root.base_model.model.model.visual.deepstack_merger = nn.Module()
        root.base_model.model.model.visual.deepstack_merger.lora_A = nn.ModuleDict(
            {"default": nn.Linear(2, 2, bias=False)}
        )
    return root


def test_runtime_peft_names_parse_case_insensitively_and_count_tensors():
    inventory = collect_runtime_lora_inventory(_runtime_model(layers=[0]))
    assert inventory["attention_tensor_count"] == 8
    assert inventory["mlp_tensor_count"] == 6
    assert inventory["attention_module_count"] == 4
    assert inventory["mlp_module_count"] == 3
    assert inventory["detected"]["q_proj"] == {0}


def test_complete_36_layer_inventory_has_a3_counts():
    inventory = collect_runtime_lora_inventory(_runtime_model())
    assert inventory["attention_module_count"] == 144
    assert inventory["mlp_module_count"] == 108
    assert inventory["attention_tensor_count"] == 288
    assert inventory["mlp_tensor_count"] == 216
    assert all(inventory["detected"][target] == set(range(36)) for target in (*ATTENTION, *MLP))
    assert inventory["missing_components"] == {}


def test_missing_layer_and_missing_component_are_explicit():
    inventory = collect_runtime_lora_inventory(_runtime_model(layers=range(35), components=("A",)))
    assert inventory["detected"]["down_proj"] == set(range(35))
    assert "mlp:0:mlp:down_proj" in inventory["missing_components"]
    assert inventory["missing_components"]["mlp:0:mlp:down_proj"] == ["b"]


def test_projector_lora_is_separate_from_language_model_inventory():
    model = _runtime_model(layers=[0])
    merger = model.base_model.model.model.visual = nn.Module()
    merger.merger = nn.Module()
    merger.merger.linear_fc1 = _lora_target("linear_fc1")
    merger.merger.linear_fc2 = _lora_target("linear_fc2")
    inventory = collect_runtime_lora_inventory(model)
    assert inventory["projector_tensor_count"] == 4
    assert inventory["attention_tensor_count"] == 8
    assert inventory["mlp_tensor_count"] == 6


def test_deepstack_lora_is_unmatched():
    inventory = collect_runtime_lora_inventory(_runtime_model(layers=[0], extra=True))
    assert any("deepstack_merger" in name for name in inventory["unmatched_lora_names"])

