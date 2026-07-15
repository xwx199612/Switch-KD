from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts import vlm_compare_utils


class _Merger(nn.Module):
    def __init__(self, *, quantized: bool = False):
        super().__init__()
        if quantized:
            import bitsandbytes as bnb

            self.linear_fc1 = bnb.nn.Linear4bit(2, 2)
        else:
            self.linear_fc1 = nn.Linear(2, 2, dtype=torch.bfloat16)
        self.linear_fc2 = nn.Linear(2, 2, dtype=torch.bfloat16)


class _Model(nn.Module):
    def __init__(self, *, quantized_merger: bool = False, bf16_language_model: bool = False):
        super().__init__()
        linear = nn.Linear(2, 2, dtype=torch.bfloat16) if bf16_language_model else None
        if linear is None:
            import bitsandbytes as bnb

            linear = bnb.nn.Linear4bit(2, 2)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.linear = linear
        self.model.visual = nn.Module()
        self.model.visual.merger = _Merger(quantized=quantized_merger)


def _patch_transformers(monkeypatch, model):
    import transformers

    class Recorder:
        kwargs = None

    def fake_processor(*args, **kwargs):
        return object()

    def fake_model(*args, **kwargs):
        Recorder.kwargs = kwargs
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", fake_model)
    return Recorder


def test_generic_4bit_keeps_existing_configuration(monkeypatch):
    model = _Model()
    model_class = _patch_transformers(monkeypatch, model)
    _, loaded = vlm_compare_utils.load_processor_and_model(
        "model", "bfloat16", "cpu", "4bit"
    )
    config = model_class.kwargs["quantization_config"]
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert model_class.kwargs["device_map"] == "auto"
    assert loaded is model


def test_mixed_mode_uses_the_exact_a1_exclusions(monkeypatch):
    model = _Model()
    _patch_transformers(monkeypatch, model)
    captured = {}

    def fake_builder(*, quantization, excluded_module_paths):
        captured["quantization"] = quantization
        captured["excluded_module_paths"] = excluded_module_paths
        return object()

    monkeypatch.setattr(
        "vlm_distill.mixed_precision.build_mixed_precision_quantization_config", fake_builder
    )
    vlm_compare_utils.load_processor_and_model("model", "bfloat16", "cpu", "mixed_4bit_bf16")
    assert captured == {
        "quantization": "4bit",
        "excluded_module_paths": [
            "model.visual.merger.linear_fc1",
            "model.visual.merger.linear_fc2",
        ],
    }


def test_mixed_mode_rejects_requantized_merger(monkeypatch):
    _patch_transformers(monkeypatch, _Model(quantized_merger=True))
    with pytest.raises(RuntimeError, match="no bitsandbytes quantized layer"):
        vlm_compare_utils.load_processor_and_model("model", "bfloat16", "cpu", "mixed_4bit_bf16")


def test_mixed_mode_rejects_fully_bf16_language_model(monkeypatch):
    _patch_transformers(monkeypatch, _Model(bf16_language_model=True))
    with pytest.raises(RuntimeError, match="Linear4bit count must be > 0"):
        vlm_compare_utils.load_processor_and_model("model", "bfloat16", "cpu", "mixed_4bit_bf16")
