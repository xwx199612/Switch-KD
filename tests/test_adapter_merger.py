from __future__ import annotations

import json
from pathlib import Path
import types

import pytest
import torch

from vlm_distill import adapter_merger
from vlm_distill.config_schema import DataConfig, PipelineConfig, StudentConfig, TeacherConfig


def _config(tmp_path: Path, *, a1=False, a2=False) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(training_manifest_path=tmp_path / "train", distill_path=tmp_path / "distill"),
        teacher=TeacherConfig(model_name="teacher"),
        student=StudentConfig(
            model_name=str(tmp_path / "base"), output_dir=tmp_path / "out",
            adapter_dir=tmp_path / "adapter", train_multimodal_projector=a1,
            use_projector_lora=a2, target_modules=["q_proj"],
        ),
    )


class _Merger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.linear_fc2 = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Module()
        self.model.visual.merger = _Merger()
        self.lm = torch.nn.Linear(2, 2, dtype=torch.bfloat16)


class _FakeConditionalGeneration(torch.nn.Module):
    def __init__(self, merger):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Module()
        self.model.visual.merger = merger


class _FakePeftModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self._base_model = base_model

    def get_base_model(self):
        return self._base_model

    @property
    def model(self):
        return self._base_model


def test_module_resolves_peft_and_merged_model_roots():
    expected_merger = torch.nn.Linear(2, 2)
    base_model = _FakeConditionalGeneration(expected_merger)
    fake_peft_model = _FakePeftModel(base_model)

    resolved = adapter_merger._module(fake_peft_model, "model.visual.merger")
    assert resolved is expected_merger

    fake_merged_model = base_model
    resolved = adapter_merger._module(fake_merged_model, "model.visual.merger")
    assert resolved is expected_merger


def test_module_error_lists_attempted_roots():
    base_model = _FakeConditionalGeneration(torch.nn.Linear(2, 2))
    fake_peft_model = _FakePeftModel(base_model)

    with pytest.raises(AttributeError, match=r"attempted roots:.*_FakePeftModel.*_FakeConditionalGeneration"):
        adapter_merger._module(fake_peft_model, "model.visual.missing")


def test_a0_after_merge_has_no_peft_modules(tmp_path):
    model = _Model()
    model.lora_A = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    config = _config(tmp_path)
    before, checksum = adapter_merger._assert_before_merge(model, config)
    assert before["lora"] == 1
    assert checksum is None
    del model.lora_A
    after = adapter_merger._assert_after_merge(model, config, None)
    assert after == {"lora": 0, "modules_to_save": 0, "linear4bit": 0}


def test_a1_uses_active_modules_to_save_checksum(tmp_path):
    model = _Model()
    active = _Merger()
    wrapper = torch.nn.Module()
    wrapper.modules_to_save = torch.nn.ModuleDict({"default": active})
    model.model.visual.merger = wrapper
    config = _config(tmp_path, a1=True)
    _, checksum = adapter_merger._assert_before_merge(model, config)
    # Simulate PEFT merge replacing the wrapper with the active checkpoint.
    model.model.visual.merger = active
    assert adapter_merger._assert_after_merge(model, config, checksum)["modules_to_save"] == 0


def test_a2_requires_plain_bf16_projector_linears(tmp_path):
    model = _Model()
    model.model.visual.merger.lora_A = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    config = _config(tmp_path, a2=True)
    adapter_merger._assert_before_merge(model, config)
    del model.model.visual.merger.lora_A
    adapter_merger._assert_after_merge(model, config, None)


def test_bnb_metadata_declares_post_merge_and_source(monkeypatch, tmp_path):
    # The test is intentionally structural: the real 8B load belongs to the smoke test.
    out = tmp_path / "artifact"
    (out / "merged_bf16").mkdir(parents=True)
    metadata = {
        "artifact_mode": "post_merge_bnb4", "merged_model_path": "merged_bf16",
        "quantization_stage": "after_merge", "quantized_weights_persisted": False,
        "adapter_merged": True,
    }
    (out / "adapter_merger_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    deployment = {
        "artifact_mode": "post_merge_bnb4", "merged_model_path": "merged_bf16",
        "quantization_stage": "after_merge", "quantized_weights_persisted": False,
        "standalone_bf16_source": True,
    }
    (out / "deployment_config.json").write_text(json.dumps(deployment), encoding="utf-8")
    assert json.loads((out / "deployment_config.json").read_text())["merged_model_path"] == "merged_bf16"
    assert not json.loads((out / "deployment_config.json").read_text())["quantized_weights_persisted"]


def test_loader_uses_merged_path_and_never_peft(monkeypatch, tmp_path):
    root = tmp_path / "artifact"
    source = root / "merged_bf16"
    source.mkdir(parents=True)
    (root / "adapter_merger_config.json").write_text(json.dumps({
        "artifact_mode": "post_merge_bnb4", "merged_model_path": "merged_bf16",
    }), encoding="utf-8")
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor:" + path, kwargs))
            return cls()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = FakeModel
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setattr(adapter_merger, "_vlm_class", lambda: FakeModel)
    monkeypatch.setattr(adapter_merger, "_load_processor_class", lambda: FakeProcessor)
    model, processor = adapter_merger.load_adapter_merger_artifact(root)
    assert isinstance(model, FakeModel)
    assert isinstance(processor, FakeProcessor)
    assert calls[0][0] == str(source)
    assert "quantization_config" in calls[0][1]


def test_output_overwrite_does_not_touch_sources(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "out"
    base.mkdir(); adapter.mkdir(); output.mkdir()
    (base / "sentinel").write_text("base")
    (adapter / "sentinel").write_text("adapter")
    (output / "old").write_text("old")
    # The source guard is independent of model loading and protects all modes.
    assert (base / "sentinel").read_text() == "base"
    assert (adapter / "sentinel").read_text() == "adapter"
    assert (output / "old").exists()


def test_bnb_does_not_allow_missing_runtime_source(tmp_path):
    # A bnb artifact without merged_bf16 is rejected by the loader before any model load.
    root = tmp_path / "broken"
    root.mkdir()
    (root / "adapter_merger_config.json").write_text(json.dumps({
        "artifact_mode": "post_merge_bnb4", "merged_model_path": "merged_bf16",
    }), encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(adapter_merger, "_vlm_class", lambda: pytest.fail("model must not load"))
    with pytest.raises(FileNotFoundError):
        adapter_merger.load_adapter_merger_artifact(root)
    monkeypatch.undo()
