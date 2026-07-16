from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vlm_distill.config_schema import (
    DataConfig,
    PipelineConfig,
    StudentConfig,
    TeacherConfig,
)
from vlm_distill.stage_merge_adapter import (
    _build_merge_model_kwargs,
    _resolve_standalone_validation_image,
    _validate_standalone_merged_model,
)


def _config(
    tmp_path: Path,
    *,
    manifest: Path | None = None,
    image_root: Path | None = None,
    quantization: str = "none",
    merged_artifact_mode: str = "bf16_standalone",
):
    return PipelineConfig(
        data=DataConfig(
            training_manifest_path=manifest or tmp_path / "missing.jsonl",
            inference_manifest_path=manifest,
            distill_path=tmp_path / "distill.jsonl",
            image_root=image_root or tmp_path,
        ),
        teacher=TeacherConfig(model_name="teacher"),
        student=StudentConfig(
            model_name="student",
            output_dir=tmp_path / "output",
            adapter_dir=tmp_path / "adapter",
            quantization=quantization,
            merged_artifact_mode=merged_artifact_mode,
        ),
    )


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_bf16_standalone_ignores_student_quantization(tmp_path, quantization):
    kwargs = _build_merge_model_kwargs(_config(tmp_path, quantization=quantization))
    reload_kwargs = _build_merge_model_kwargs(
        _config(tmp_path, quantization=quantization), reload=True
    )

    assert "quantization_config" not in kwargs
    assert "quantization_config" not in reload_kwargs
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert reload_kwargs["torch_dtype"] is torch.bfloat16
    assert kwargs["device_map"] == reload_kwargs["device_map"] == "auto"


def test_mixed_4bit_bf16_builds_4bit_config_only_in_mixed_mode(tmp_path, monkeypatch):
    import vlm_distill.stage_merge_adapter as merge

    calls = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(merge, "build_mixed_precision_quantization_config", fake_builder)
    config = _config(tmp_path, quantization="4bit", merged_artifact_mode="mixed_4bit_bf16")
    kwargs = _build_merge_model_kwargs(config)
    reload_kwargs = _build_merge_model_kwargs(config, reload=True)

    assert "quantization_config" in kwargs
    assert "quantization_config" in reload_kwargs
    assert calls == [
        {"quantization": "4bit", "excluded_module_paths": []},
        {"quantization": "4bit", "excluded_module_paths": []},
    ]

    adapter_config = _config(tmp_path, quantization="4bit", merged_artifact_mode="adapter_plus_projector")
    assert "quantization_config" not in _build_merge_model_kwargs(adapter_config)


def test_partial_layer_adapter_config_is_accepted_by_peft(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model

    base = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    trained = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=4, target_modules=["0"], modules_to_save=None),
    )
    adapter_path = tmp_path / "adapter"
    trained.save_pretrained(adapter_path)

    reloaded_base = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    peft_model = PeftModel.from_pretrained(reloaded_base, str(adapter_path))
    merged = peft_model.merge_and_unload()

    assert not any("lora" in name.lower() for name, _ in merged.named_modules())


def _jpeg(path: Path) -> Path:
    Image.new("RGB", (2, 2), color="red").save(path, format="JPEG")
    return path


def test_missing_sample_image_returns_none(tmp_path, monkeypatch):
    import vlm_distill.stage_merge_adapter as merge

    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", tmp_path / "missing.jpg")
    assert _resolve_standalone_validation_image(_config(tmp_path)) is None


@pytest.mark.parametrize("writer", [lambda path: None, lambda path: path.write_bytes(b"not jpeg")])
def test_zero_byte_or_corrupt_image_returns_none(tmp_path, monkeypatch, writer):
    import vlm_distill.stage_merge_adapter as merge

    sample = tmp_path / "sample.jpg"
    writer(sample)
    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", sample)
    assert _resolve_standalone_validation_image(_config(tmp_path)) is None


def test_git_lfs_pointer_returns_none(tmp_path, monkeypatch):
    import vlm_distill.stage_merge_adapter as merge

    sample = tmp_path / "sample.jpg"
    sample.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\nsize 123\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", sample)
    assert _resolve_standalone_validation_image(_config(tmp_path)) is None


def test_valid_jpeg_is_selected(tmp_path, monkeypatch):
    import vlm_distill.stage_merge_adapter as merge

    sample = _jpeg(tmp_path / "sample.jpg")
    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", sample)
    assert _resolve_standalone_validation_image(_config(tmp_path)) == sample.resolve()


def test_invalid_manifest_image_is_skipped_for_later_valid_image(tmp_path, monkeypatch):
    import vlm_distill.stage_merge_adapter as merge

    valid = _jpeg(tmp_path / "valid.jpg")
    manifest = tmp_path / "inference.jsonl"
    manifest.write_text(
        "\n".join([
            json.dumps({"image": "missing.jpg"}),
            json.dumps({"image": valid.name}),
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", tmp_path / "unused.jpg")
    assert _resolve_standalone_validation_image(_config(tmp_path, manifest=manifest)) == valid.resolve()


class _Processor:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[1]]), "pixel_values": torch.zeros(1, 3, 2, 2)}


class _Model(torch.nn.Module):
    def __init__(self, logits=None, error=None):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.logits = logits
        self.error = error

    def forward(self, **kwargs):
        if self.error:
            raise self.error
        logits = self.logits if self.logits is not None else torch.zeros(1, 1, 2)
        return SimpleNamespace(logits=logits)


def test_no_valid_image_prints_skip_and_valid_image_runs_forward(tmp_path, monkeypatch, capsys):
    import vlm_distill.stage_merge_adapter as merge

    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", tmp_path / "missing.jpg")
    _validate_standalone_merged_model(_Model(), _Processor(), tmp_path, config=_config(tmp_path))
    assert "no valid validation image was found" in capsys.readouterr().out

    image = _jpeg(tmp_path / "valid.jpg")
    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", image)
    _validate_standalone_merged_model(_Model(), _Processor(), tmp_path, config=_config(tmp_path))
    output = capsys.readouterr().out
    assert "Standalone merged image smoke test:" in output
    assert f"image={image.resolve()}" in output


def test_processor_and_model_errors_propagate(tmp_path, monkeypatch):
    import vlm_distill.stage_merge_adapter as merge

    image = _jpeg(tmp_path / "valid.jpg")
    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", image)

    class FailingProcessor(_Processor):
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("processor failed")

    with pytest.raises(ValueError, match="processor failed"):
        _validate_standalone_merged_model(_Model(), FailingProcessor(), tmp_path, config=_config(tmp_path))
    with pytest.raises(RuntimeError, match="model failed"):
        _validate_standalone_merged_model(
            _Model(error=RuntimeError("model failed")), _Processor(), tmp_path, config=_config(tmp_path)
        )


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (torch.tensor([[[float("nan")]]]), "non-finite logits"),
        (torch.tensor([[[float("inf")]]]), "non-finite logits"),
        (torch.empty(0), "empty logits"),
    ],
)
def test_non_finite_or_empty_logits_propagate(tmp_path, monkeypatch, logits, message):
    import vlm_distill.stage_merge_adapter as merge

    image = _jpeg(tmp_path / "valid.jpg")
    monkeypatch.setattr(merge, "_REPOSITORY_SAMPLE_IMAGE", image)
    with pytest.raises(RuntimeError, match=message):
        _validate_standalone_merged_model(
            _Model(logits=logits),
            _Processor(),
            tmp_path,
            config=_config(tmp_path),
        )
