import json

import pytest
import torch
from safetensors.torch import save_file

from vlm_distill.train_online_align_dbild import (
    _validate_smoke_adapter_checkpoint,
    is_saved_full_projector_key,
    validate_smoke_gradient_contract,
    validate_smoke_losses,
)


class _NamedParameters:
    def __init__(self, entries):
        self.entries = entries

    def named_parameters(self):
        return iter(self.entries)


def _model(*, missing=None, extra=None):
    names = {
        "attention_lora": "model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "mlp_lora": "model.language_model.layers.0.mlp.gate_proj.lora_A.weight",
        "full_projector": "model.visual.merger.modules_to_save.default.linear_fc1.weight",
    }
    entries = []
    for group, name in names.items():
        parameter = torch.nn.Parameter(torch.ones(2))
        if group != missing:
            parameter.grad = torch.ones_like(parameter)
        entries.append((name, parameter))
    if extra == "vision_encoder":
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.ones_like(parameter)
        entries.append(("model.visual.blocks.0.weight", parameter))
    return _NamedParameters(entries)


def test_a3_gradient_contract_accepts_finite_attention_mlp_projector_gradients():
    groups = validate_smoke_gradient_contract(_model(), "model.visual.merger")
    assert groups["attention_lora"]["tensors_with_grad"] == 1
    assert groups["mlp_lora"]["gradient_norm"] > 0
    assert groups["full_projector"]["finite_gradient_tensors"] == 1
    assert groups["projector_lora"]["parameter_count"] == 0


@pytest.mark.parametrize("missing", ["attention_lora", "mlp_lora", "full_projector"])
def test_a3_gradient_contract_fails_when_required_group_has_no_gradient(missing):
    with pytest.raises(RuntimeError, match="has no gradient tensors"):
        validate_smoke_gradient_contract(_model(missing=missing), "model.visual.merger")


def test_a3_gradient_contract_fails_when_vision_becomes_trainable():
    with pytest.raises(RuntimeError, match="unexpected trainable group vision_encoder"):
        validate_smoke_gradient_contract(_model(extra="vision_encoder"), "model.visual.merger")


def test_smoke_loss_contract_rejects_nonfinite_and_accepts_finite_positive_loss():
    validate_smoke_losses(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.0), torch.tensor(3.0))
    with pytest.raises(FloatingPointError, match="non-finite"):
        validate_smoke_losses(torch.tensor(float("nan")), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0))


_PROJECTOR_KEYS = [
    "base_model.model.model.visual.merger.norm.weight",
    "base_model.model.model.visual.merger.norm.bias",
    "base_model.model.model.visual.merger.linear_fc1.weight",
    "base_model.model.model.visual.merger.linear_fc1.bias",
    "base_model.model.model.visual.merger.linear_fc2.weight",
    "base_model.model.model.visual.merger.linear_fc2.bias",
]
_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _write_adapter_fixture(tmp_path, *, projector_keys=None, modules_to_save=("model.visual.merger",)):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"modules_to_save": list(modules_to_save)}), encoding="utf-8"
    )
    keys = {
        f"base_model.model.model.language_model.layers.0.mlp.{target}.lora_A.weight": torch.ones(1)
        for target in _LORA_TARGETS
    }
    keys.update({key: torch.ones(2, 2) for key in (projector_keys or _PROJECTOR_KEYS)})
    save_file(keys, str(adapter / "adapter_model.safetensors"))
    return adapter


def test_saved_projector_key_accepts_peft_prefix_but_not_runtime_wrapper():
    saved = "base_model.model.model.visual.merger.linear_fc1.weight"
    runtime = "base_model.model.model.visual.merger.modules_to_save.default.linear_fc1.weight"
    assert is_saved_full_projector_key(saved, "model.visual.merger")
    assert not is_saved_full_projector_key(runtime, "model.visual.merger")


def test_smoke_adapter_validator_accepts_configured_saved_projector(tmp_path):
    _validate_smoke_adapter_checkpoint(_write_adapter_fixture(tmp_path))


@pytest.mark.parametrize("missing", ["linear_fc1.weight", "linear_fc2.weight"])
def test_smoke_adapter_validator_reports_missing_projector_tensor(tmp_path, missing):
    keys = [key for key in _PROJECTOR_KEYS if not key.endswith(missing)]
    with pytest.raises(RuntimeError, match="missing projector tensors"):
        _validate_smoke_adapter_checkpoint(_write_adapter_fixture(tmp_path, projector_keys=keys))


def test_smoke_adapter_validator_rejects_unconfigured_projector(tmp_path):
    adapter = _write_adapter_fixture(tmp_path, modules_to_save=())
    with pytest.raises(RuntimeError, match="modules_to_save"):
        _validate_smoke_adapter_checkpoint(adapter)


def test_smoke_adapter_validator_rejects_runtime_wrapper_keys(tmp_path):
    runtime_keys = [key.replace("visual.merger.", "visual.merger.modules_to_save.default.") for key in _PROJECTOR_KEYS]
    with pytest.raises(RuntimeError, match="missing projector tensors"):
        _validate_smoke_adapter_checkpoint(_write_adapter_fixture(tmp_path, projector_keys=runtime_keys))


def test_smoke_adapter_validator_rejects_deepstack_only_projector(tmp_path):
    deepstack = [key.replace("visual.merger", "visual.deepstack_merger_list.0") for key in _PROJECTOR_KEYS]
    with pytest.raises(RuntimeError, match="missing projector tensors"):
        _validate_smoke_adapter_checkpoint(_write_adapter_fixture(tmp_path, projector_keys=deepstack))


def test_smoke_adapter_validator_rejects_projector_lora(tmp_path):
    keys = _PROJECTOR_KEYS + [
        "base_model.model.model.visual.merger.linear_fc1.lora_A.weight",
    ]
    with pytest.raises(RuntimeError, match="projector LoRA"):
        _validate_smoke_adapter_checkpoint(_write_adapter_fixture(tmp_path, projector_keys=keys))
