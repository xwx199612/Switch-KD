from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pytest

from vlm_distill.config_schema import load_config
from vlm_distill.student_trainability import (
    dequantize_trainable_projector,
    summarize_trainable_groups,
)
from vlm_distill.train_online_align_dbild import freeze_student_vision_keep_merger_lm_trainable


def _config_diff(left, right):
    differences = {}
    left_values = asdict(left)
    right_values = asdict(right)

    def walk(path, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(f"{path}.{key}" if path else key, a.get(key), b.get(key))
        elif a != b:
            differences[path] = (a, b)

    walk("", left_values, right_values)
    return differences


def test_a1_smoke_resolved_config_diff_is_only_explicitly_allowed_fields():
    formal = load_config("configs/qwen3vl8b_r16_attn_projector_trainable.yaml")
    smoke = load_config("configs/qwen3vl8b_r16_attn_projector_trainable_smoke.yaml")
    assert set(_config_diff(formal, smoke)) == {
        "data.max_samples",
        "student.output_dir",
        "student.adapter_dir",
        "student.merged_model_path",
        "training.epochs",
        "training.gradient_accumulation_steps",
        "training.max_steps",
        "training.log_every",
        "training.save_every",
    }


def test_a1_smoke_keeps_formal_prompt_and_kneedle_configuration():
    formal = load_config("configs/qwen3vl8b_r16_attn_projector_trainable.yaml")
    smoke = load_config("configs/qwen3vl8b_r16_attn_projector_trainable_smoke.yaml")
    assert smoke.distillation.prompt_template == formal.distillation.prompt_template
    for text in ("normalized 0-1000", "bbox_norm", "focused", "ASCII double quotes", "valid JSON only"):
        assert text in smoke.distillation.prompt_template
    assert smoke.distillation.dbild_top_k_mode == "kneedle"
    assert smoke.distillation.dbild_kneedle_candidate_k == formal.distillation.dbild_kneedle_candidate_k
    assert smoke.distillation.dbild_min_top_k == formal.distillation.dbild_min_top_k
    assert smoke.distillation.dbild_max_top_k == formal.distillation.dbild_max_top_k


def _a1_validation_model(*, extra_linear=False, dtype=None):
    torch = pytest.importorskip("torch")
    from torch import nn

    class Merger(nn.Module):
        def __init__(self):
            super().__init__()
            target_dtype = dtype or torch.bfloat16
            self.linear_fc1 = nn.Linear(2, 2).to(dtype=target_dtype)
            self.linear_fc2 = nn.Linear(2, 2).to(dtype=target_dtype)
            if extra_linear:
                self.extra = nn.Linear(2, 2).to(dtype=target_dtype)

    model = nn.Module()
    model.model = nn.Module()
    model.model.visual = nn.Module()
    model.model.visual.merger = Merger()
    return model


def test_precision_summary_returns_counts():
    from vlm_distill.stage_merge_adapter import _print_merged_precision_summary

    counts = _print_merged_precision_summary(_a1_validation_model())
    assert isinstance(counts, dict)
    assert counts["main merger BF16 linears"] == 2


def test_a1_merge_validation_rejects_zero_quantized_language_model_linears():
    from vlm_distill.stage_merge_adapter import _validate_a1_merged_precision

    counts = {"language_model quantized linears": 0, "main merger BF16 linears": 2,
              "remaining LoRA modules": 0, "remaining modules_to_save wrappers": 0}
    with pytest.raises(RuntimeError, match="quantized language-model linears > 0"):
        _validate_a1_merged_precision(_a1_validation_model(), counts, projector_path="model.visual.merger")


def test_a1_merge_validation_rejects_non_bf16_main_merger():
    from vlm_distill.stage_merge_adapter import _validate_a1_merged_precision

    model = _a1_validation_model(dtype=__import__("torch").float32)
    counts = {"language_model quantized linears": 1, "main merger BF16 linears": 0,
              "remaining LoRA modules": 0, "remaining modules_to_save wrappers": 0}
    with pytest.raises(RuntimeError, match="linear_fc1.*dtype"):
        _validate_a1_merged_precision(model, counts, projector_path="model.visual.merger")


@pytest.mark.parametrize("extra_linear, expected", [(False, 1), (True, 3)])
def test_a1_merge_validation_rejects_fewer_or_more_main_merger_linears(extra_linear, expected):
    from vlm_distill.stage_merge_adapter import _validate_a1_merged_precision

    model = _a1_validation_model(extra_linear=extra_linear)
    counts = {"language_model quantized linears": 1, "main merger BF16 linears": expected,
              "remaining LoRA modules": 0, "remaining modules_to_save wrappers": 0}
    if not extra_linear:
        del model.model.visual.merger.linear_fc2
    with pytest.raises(RuntimeError, match="main-merger|linear_fc2"):
        _validate_a1_merged_precision(model, counts, projector_path="model.visual.merger")


@pytest.mark.parametrize("field, message", [
    ("remaining LoRA modules", "LoRA"),
    ("remaining modules_to_save wrappers", "modules_to_save"),
])
def test_a1_merge_validation_rejects_remaining_peft_wrappers(field, message):
    from vlm_distill.stage_merge_adapter import _validate_a1_merged_precision

    counts = {"language_model quantized linears": 1, "main merger BF16 linears": 2,
              "remaining LoRA modules": 0, "remaining modules_to_save wrappers": 0}
    counts[field] = 1
    with pytest.raises(RuntimeError, match=message):
        _validate_a1_merged_precision(_a1_validation_model(), counts, projector_path="model.visual.merger")


def test_exact_main_merger_linear_children_pass_bf16_validation():
    from vlm_distill.stage_merge_adapter import _validate_a1_merged_precision

    counts = {"language_model quantized linears": 1, "main merger BF16 linears": 2,
              "remaining LoRA modules": 0, "remaining modules_to_save wrappers": 0}
    _validate_a1_merged_precision(_a1_validation_model(), counts, projector_path="model.visual.merger")


def test_a1_yaml_loads_and_has_controlled_lora_settings():
    config = load_config("configs/qwen3vl8b_r16_attn_projector_trainable.yaml")
    assert config.student.lora_rank == 16
    assert config.student.lora_alpha == 32
    assert config.student.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.student.train_multimodal_projector is True
    assert config.student.multimodal_projector_path == "model.visual.merger"
    assert "projector_trainable" in str(config.student.adapter_dir)


def test_a1_smoke_yaml_is_one_step_and_two_samples():
    config = load_config("configs/qwen3vl8b_r16_attn_projector_trainable_smoke.yaml")
    assert config.data.max_samples == 2
    assert config.training.max_steps == 1
    assert config.training.log_every == 1
    assert config.training.save_every == 1
    assert config.student.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.student.train_multimodal_projector is True


def test_exact_freezer_keeps_projector_and_vision_encoder_separate():
    torch = pytest.importorskip("torch")
    from torch import nn

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.q_proj.lora_A = nn.Parameter(torch.ones(1))
            self.k_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)
            self.o_proj = nn.Linear(4, 4)
            self.down_proj = nn.Linear(4, 4)

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])
            self.merger = nn.Linear(4, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = Visual()
            self.model.layer = Attention()

    model = Model()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    summary = freeze_student_vision_keep_merger_lm_trainable(
        model,
        use_lora=True,
        train_multimodal_projector=True,
        multimodal_projector_path="model.visual.merger",
    )
    assert summary.count > 0
    assert all(not p.requires_grad for n, p in model.named_parameters() if "visual.blocks" in n)
    assert all(p.requires_grad for n, p in model.named_parameters() if "model.visual.merger" in n)
    assert any(p.requires_grad for n, p in model.named_parameters() if "q_proj.lora_A" in n)
    assert all(not p.requires_grad for n, p in model.named_parameters() if "down_proj" in n)


def test_a0_default_projector_is_frozen():
    torch = pytest.importorskip("torch")
    from torch import nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = nn.Module()
            self.model.visual.merger = nn.Linear(2, 2)

    model = M()
    freeze_student_vision_keep_merger_lm_trainable(model, use_lora=False)
    assert not any(p.requires_grad for p in model.model.visual.merger.parameters())


def test_projector_lora_mode_trains_only_projector_adapter():
    torch = pytest.importorskip("torch")
    from torch import nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = nn.Module()
            self.model.visual.merger = nn.Module()
            self.model.visual.merger.lora_A = nn.Parameter(torch.ones(2))
            self.model.visual.merger.weight = nn.Parameter(torch.ones(2))

    model = M()
    freeze_student_vision_keep_merger_lm_trainable(
        model, use_lora=True, train_multimodal_projector=False,
        multimodal_projector_path="model.visual.merger",
    )
    assert model.model.visual.merger.lora_A.requires_grad
    assert not model.model.visual.merger.weight.requires_grad


def test_only_configured_quantized_projector_linears_are_rebuilt_as_bf16(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch import nn
    bnb = pytest.importorskip("bitsandbytes")
    import vlm_distill.student_trainability as trainability

    class QuantState:
        quant_type = "nf4"

    class Merger(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)
            self.linear_fc1 = bnb.nn.Linear4bit(4, 8)
            self.act_fn = nn.GELU()
            self.linear_fc2 = bnb.nn.Linear4bit(8, 4, bias=False)
            self.linear_fc1.weight.quant_state = QuantState()
            self.linear_fc2.weight.quant_state = QuantState()

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = Merger()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])
            self.deepstack_merger_list = nn.ModuleList([nn.Linear(4, 4)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = Visual()
            self.model.language_model = nn.Linear(4, 4)

    monkeypatch.setattr(trainability, "_dequantized_weight", lambda module: module.weight.data.float())
    model = Model()
    original_block = model.model.visual.blocks[0]
    original_deepstack = model.model.visual.deepstack_merger_list[0]
    before_classes = {name: type(module) for name, module in model.named_modules()}
    result = dequantize_trainable_projector(model, "model.visual.merger", validate_forward=False)
    merger = model.model.visual.merger

    assert result["converted_linears"] == 2
    assert isinstance(merger.norm, nn.LayerNorm)
    assert isinstance(merger.act_fn, nn.GELU)
    assert isinstance(merger.linear_fc1, nn.Linear)
    assert isinstance(merger.linear_fc2, nn.Linear)
    assert merger.linear_fc1.weight.dtype == torch.bfloat16
    assert merger.linear_fc2.weight.dtype == torch.bfloat16
    assert merger.linear_fc2.bias is None
    assert all(parameter.dtype == torch.bfloat16 for parameter in merger.linear_fc1.parameters())
    assert model.model.visual.blocks[0] is original_block
    assert model.model.visual.deepstack_merger_list[0] is original_deepstack
    assert type(model.model.visual.blocks[0]) is before_classes["model.visual.blocks.0"]
    assert type(model.model.visual.deepstack_merger_list[0]) is before_classes["model.visual.deepstack_merger_list.0"]
    assert model.model.language_model.weight.dtype == torch.float32


def test_params4bit_generic_dequantize_method_is_never_used(monkeypatch):
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    import bitsandbytes.functional as functional
    import vlm_distill.student_trainability as trainability

    module = bnb.nn.Linear4bit(2, 2)
    module.weight.quant_state = type("QuantState", (), {"quant_type": "nf4"})()
    called = {}
    def fake_dequantize_4bit(data, *, quant_state):
        called["data"] = data
        called["quant_state"] = quant_state
        return data.float()
    monkeypatch.setattr(functional, "dequantize_4bit", fake_dequantize_4bit)
    monkeypatch.setattr(module.weight, "dequantize", lambda: (_ for _ in ()).throw(AssertionError("generic dequantize used")))
    trainability._dequantized_weight(module)
    assert torch.equal(called["data"], module.weight.data)
    assert called["quant_state"] is module.weight.quant_state


def test_missing_quant_state_fails_before_conversion():
    bnb = pytest.importorskip("bitsandbytes")
    import vlm_distill.student_trainability as trainability
    module = bnb.nn.Linear4bit(2, 2)
    module.weight.quant_state = None
    with pytest.raises(RuntimeError, match="quant_state is missing"):
        trainability._dequantized_weight(module)


def test_linear8bit_projector_is_explicitly_rejected():
    bnb = pytest.importorskip("bitsandbytes")
    import vlm_distill.student_trainability as trainability
    from torch import nn
    module = bnb.nn.Linear8bitLt(2, 2)
    with pytest.raises(NotImplementedError, match="Linear8bitLt"):
        trainability._dequantized_weight(module)
    merger = nn.Module()
    merger.linear_fc1 = module
    model = nn.Module()
    model.model = nn.Module()
    model.model.visual = nn.Module()
    model.model.visual.merger = merger
    with pytest.raises(NotImplementedError, match="Linear8bitLt"):
        dequantize_trainable_projector(model, "model.visual.merger", validate_forward=False)


def test_maybe_enable_student_lora_passes_active_full_projector_path(monkeypatch):
    pytest.importorskip("peft")
    from torch import nn
    from transformers import PretrainedConfig
    import vlm_distill.train_online_align_dbild as online

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = nn.Linear(2, 2)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig(model_type="toy")
            self.model = nn.Module()
            self.model.visual = Visual()
            self.model.q_proj = nn.Linear(2, 2)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return kwargs

    student = type("S", (), {
        "use_lora": True, "quantization": "none", "target_modules": ["q_proj"],
        "lora_rank": 1, "lora_alpha": 2, "lora_dropout": 0.0,
        "train_multimodal_projector": True,
        "use_projector_lora": False,
        "multimodal_projector_path": "model.visual.merger",
        "lora_layers_to_transform": None,
        "lora_layers_pattern": None,
    })()
    config = type("C", (), {"student": student})()
    captured = []

    def capture(*args, **kwargs):
        captured.append(kwargs["allowed_full_projector_path"])

    monkeypatch.setattr(online, "validate_language_model_lora_scope", capture)
    online._maybe_enable_student_lora(config, Toy(), dry_run=True)
    assert captured == ["model.visual.merger.modules_to_save.default"]


def test_projector_survives_peft_save_reload_and_merge(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from torch import nn
    from transformers import PretrainedConfig
    from vlm_distill.train_online_align_dbild import _maybe_enable_student_lora, _build_optimizer
    from peft import PeftModel

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = nn.Linear(2, 2)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig(model_type="toy")
            self.model = nn.Module()
            self.model.visual = Visual()
            self.model.q_proj = nn.Linear(2, 2)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return kwargs

    config = type("C", (), {"student": type("S", (), {
        "use_lora": True, "quantization": "none", "target_modules": ["q_proj"],
        "lora_rank": 1, "lora_alpha": 2, "lora_dropout": 0.0,
        "train_multimodal_projector": True,
        "multimodal_projector_path": "model.visual.merger",
    })()})()
    config.training = type("T", (), {"learning_rate": 1e-4})()
    base = Toy()
    trained = _maybe_enable_student_lora(config, base)
    freeze_student_vision_keep_merger_lm_trainable(
        trained, use_lora=True, train_multimodal_projector=True,
        multimodal_projector_path="model.visual.merger",
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in trained.named_parameters()
        if ".original_module." in name and "visual.merger" in name
    )
    assert sum(
        parameter.requires_grad
        for name, parameter in trained.named_parameters()
        if ".modules_to_save.default." in name and "visual.merger" in name
    ) > 0
    optimizer = _build_optimizer(config, trained)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in trained.parameters() if parameter.requires_grad}
    expected = torch.full_like(trained.base_model.model.model.visual.merger.modules_to_save.default.weight, 3.0)
    with torch.no_grad():
        trained.base_model.model.model.visual.merger.modules_to_save.default.weight.copy_(expected)
    trained.save_pretrained(tmp_path)

    reloaded = PeftModel.from_pretrained(Toy(), tmp_path)
    actual = reloaded.base_model.model.model.visual.merger.modules_to_save.default.weight
    assert torch.equal(actual, expected)
    merged = reloaded.merge_and_unload()
    assert torch.equal(merged.model.visual.merger.weight, expected)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLM_DISTILL_RUN_REAL_4BIT_TESTS") != "1",
    reason="Set VLM_DISTILL_RUN_REAL_4BIT_TESTS=1 to run the local Qwen3-VL 4-bit test",
)
def test_qwen3vl_4bit_projector_gradient_smoke():
    """Real-model smoke test; opt-in because it needs a CUDA-capable Qwen3-VL install."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-VL 4-bit integration requires CUDA")
    if not Path("/mnt/nvme0/vlm_distill/models/Qwen3-VL-8B-Instruct").exists():
        pytest.skip("local Qwen3-VL-8B-Instruct weights are unavailable")
    import bitsandbytes as bnb
    from PIL import Image
    from vlm_distill.config_schema import load_config
    from vlm_distill.train_online_align_dbild import (
        _apply_student_train_setup,
        _build_optimizer,
        _load_student,
    )

    config = load_config("configs/qwen3vl8b_r16_attn_projector_trainable.yaml")
    model, processor, _, _ = _load_student(config)
    original_merger = model.model.visual.merger
    original_linears = (original_merger.linear_fc1, original_merger.linear_fc2)
    assert all(isinstance(module, bnb.nn.Linear4bit) for module in original_linears)
    assert all(module.weight.quant_state is not None for module in original_linears)
    torch.manual_seed(0)
    conversion_input = torch.randn(2, original_linears[0].in_features, device=original_linears[0].weight.device, dtype=torch.bfloat16)
    with torch.no_grad():
        original_output = original_linears[0](conversion_input)
    from vlm_distill.student_trainability import dequantize_trainable_projector
    dequantize_trainable_projector(model, config.student.multimodal_projector_path)
    with torch.no_grad():
        converted_output = model.model.visual.merger.linear_fc1(conversion_input)
    torch.testing.assert_close(converted_output.float(), original_output.float(), rtol=2e-2, atol=2e-2)
    model, _ = _apply_student_train_setup(config, model)
    optimizer = _build_optimizer(config, model)

    image = Image.open("examples/images/sample_001.jpg").convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image."}
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(parameter for parameter in model.parameters() if parameter.device.type == "cuda").device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    labels = inputs["input_ids"].clone()

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    def stats(predicate):
        norms = [parameter.grad.detach().float().norm().item() for name, parameter in model.named_parameters()
                 if parameter.grad is not None and predicate(name)]
        return (max(norms, default=0.0), sum(norms) / len(norms) if norms else 0.0)
    attention_stats = stats(lambda name: "lora" in name.lower() and any(x in name.lower() for x in ("q_proj", "k_proj", "v_proj", "o_proj")))
    projector_stats = stats(lambda name: ".modules_to_save.default." in name and "model.visual.merger" in name)
    vision_stats = stats(lambda name: "visual.blocks" in name or "visual.patch_embed" in name)
    lm_stats = stats(lambda name: "language_model" in name)
    print(f"gradient_norms attention_lora max/mean={attention_stats} projector max/mean={projector_stats} vision_encoder max/mean={vision_stats} base_llm max/mean={lm_stats}")
    assert attention_stats[0] > 0 and projector_stats[0] > 0
    assert vision_stats == (0.0, 0.0) and lm_stats == (0.0, 0.0)
    assert any("lora" in name.lower() and parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters())
    assert any(".modules_to_save.default." in name and "model.visual.merger" in name and parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if "visual.blocks" in name or "language_model" in name)
    optimizer.step()
    assert any("model.visual.merger" in name and not torch.equal(before[name], parameter)
               for name, parameter in model.named_parameters())
    assert any("lora" in name.lower() and not torch.equal(before[name], parameter)
               for name, parameter in model.named_parameters())
    assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
               if "visual.blocks" in name or "language_model" in name)


@pytest.mark.skipif(not Path("configs/qwen3vl8b_r16_attn_mlp.yaml").exists(), reason="A2 config not present")
def test_mlp_lora_is_not_part_of_attention_only_configs():
    config = load_config("configs/qwen3vl8b_r16_attn_mlp.yaml")
    assert {"gate_proj", "up_proj", "down_proj"}.issubset(set(config.student.target_modules))
