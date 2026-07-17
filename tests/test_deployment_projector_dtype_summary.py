from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from vlm_distill import deployment_loader
from vlm_distill.deployment_loader import _summary, validate_high_fidelity_deployment


class _ToyDeployment(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.attention = nn.Module()
        self.model.language_model.attention.q_proj = nn.Module()
        self.model.language_model.attention.q_proj.lora_A = nn.Linear(2, 1)
        self.model.language_model.mlp = nn.Module()
        self.model.language_model.mlp.gate_proj = nn.Module()
        self.model.language_model.mlp.gate_proj.lora_A = nn.Linear(2, 1)
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Module()
        if mode in {"a1", "a3"}:
            full = nn.Module()
            full.linear_fc1 = nn.Linear(2, 2, dtype=torch.bfloat16)
            full.linear_fc2 = nn.Linear(2, 2, dtype=torch.bfloat16)
            self.model.visual.merger.modules_to_save = nn.ModuleDict({"default": full})
        else:
            self.model.visual.merger.linear_fc1 = nn.Linear(2, 2, dtype=torch.bfloat16)
            self.model.visual.merger.linear_fc2 = nn.Linear(2, 2, dtype=torch.bfloat16)
        if mode == "a2":
            for target in ("linear_fc1", "linear_fc2"):
                layer = getattr(self.model.visual.merger, target)
                layer.lora_A = nn.Linear(2, 1, dtype=torch.bfloat16)
                layer.lora_B = nn.Linear(1, 2, dtype=torch.bfloat16)


@pytest.mark.parametrize("mode", ["a0", "a1", "a2", "a3"])
def test_projector_dtype_summary_is_split_by_storage(mode):
    summary = _summary(_ToyDeployment(mode))
    assert summary["modules_to_save_projector_dtypes"] == (
        ["torch.bfloat16"] if mode in {"a1", "a3"} else []
    )
    assert summary["projector_lora_dtypes"] == (
        ["torch.bfloat16"] if mode == "a2" else []
    )
    assert summary["attention_dtypes"] == ["torch.float32"]
    assert summary["mlp_dtypes"] == ["torch.float32"]


def test_integer_active_modules_to_save_projector_fails(monkeypatch):
    class FakeLinear4bit(nn.Linear):
        pass

    import bitsandbytes
    monkeypatch.setattr(bitsandbytes.nn, "Linear4bit", FakeLinear4bit)
    model = _ToyDeployment("a1")
    model.model.language_model.quant = FakeLinear4bit(2, 2)
    model.model.visual.merger.modules_to_save.default.linear_fc1.bias = nn.Parameter(
        torch.ones(2, dtype=torch.int32), requires_grad=False
    )
    model.eval()
    model.active_adapter = "default"
    config = SimpleNamespace(student=SimpleNamespace(train_multimodal_projector=True))
    with pytest.raises(RuntimeError, match="modules_to_save projector tensors"):
        validate_high_fidelity_deployment(model, config)


def test_integer_projector_lora_fails(monkeypatch):
    class FakeLinear4bit(nn.Linear):
        pass

    import bitsandbytes
    monkeypatch.setattr(bitsandbytes.nn, "Linear4bit", FakeLinear4bit)
    model = _ToyDeployment("a2")
    model.model.language_model.quant = FakeLinear4bit(2, 2)
    layer = model.model.visual.merger.linear_fc1
    layer.lora_A.weight = nn.Parameter(torch.ones_like(layer.lora_A.weight, dtype=torch.int32), requires_grad=False)
    model.eval()
    model.active_adapter = "default"
    config = SimpleNamespace(student=SimpleNamespace(use_projector_lora=True))
    with pytest.raises(RuntimeError, match="adapter tensors must remain floating point"):
        validate_high_fidelity_deployment(model, config)
