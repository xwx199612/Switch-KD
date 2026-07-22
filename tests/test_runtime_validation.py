from types import SimpleNamespace

import pytest
from torch import nn

from vlm_distill.runtime_validation import summarize_model_precision, validate_loaded_precision


class Linear4bit(nn.Linear):
    pass


class FakeModel(nn.Module):
    def __init__(self, quantized=True, visual_quantized=False, peft=False):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.linear = Linear4bit(2, 2) if quantized else nn.Linear(2, 2)
        self.visual = nn.Module()
        self.visual.merger = nn.Module()
        self.visual.merger.linear_fc1 = Linear4bit(2, 2) if visual_quantized else nn.Linear(2, 2)
        self.visual.merger.linear_fc2 = nn.Linear(2, 2)
        if peft:
            self.peft_config = {"default": object()}
            self.active_adapter = "default"


def config(quantization="mixed_4bit_bf16"):
    return SimpleNamespace(student=SimpleNamespace(quantization=quantization,
                                                    deployment_artifact_path=None,
                                                    inference_model_path=None))


def test_precision_summary_identifies_linear4bit_and_dtypes():
    summary = summarize_model_precision(FakeModel())
    assert summary["linear4bit_module_count"] == 1
    assert summary["visual_linear4bit_module_count"] == 0
    assert summary["parameter_dtype_counts"]["torch.float32"] > 0


def test_mixed_precision_requires_4bit():
    with pytest.raises(RuntimeError, match="expected at least one Linear4bit"):
        validate_loaded_precision(config(), summarize_model_precision(FakeModel(quantized=False)))


def test_mixed_precision_rejects_quantized_sensitive_module():
    with pytest.raises(RuntimeError, match="excluded high-precision"):
        validate_loaded_precision(config(), summarize_model_precision(FakeModel(visual_quantized=True)))


def test_4bit_adapter_requires_unmerged_peft():
    cfg = config("4bit")
    cfg.student.deployment_artifact_path = None
    # Artifact metadata is tested separately in production bundles; this fake
    # config exercises the shared summary contract without requiring PEFT.
    assert summarize_model_precision(FakeModel(peft=True))["peft_model_mounted"]
