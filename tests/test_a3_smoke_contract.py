import pytest
import torch

from vlm_distill.train_online_align_dbild import (
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
