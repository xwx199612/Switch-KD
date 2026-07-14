from __future__ import annotations

from pathlib import Path

import pytest

from vlm_distill.config_schema import load_config
from vlm_distill.student_trainability import summarize_trainable_groups
from vlm_distill.train_online_align_dbild import freeze_student_vision_keep_merger_lm_trainable


def test_a1_yaml_loads_and_has_controlled_lora_settings():
    config = load_config("configs/qwen3vl8b_r16_attn_projector_trainable.yaml")
    assert config.student.lora_rank == 16
    assert config.student.lora_alpha == 32
    assert config.student.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.student.train_multimodal_projector is True
    assert config.student.multimodal_projector_path == "model.visual.merger"
    assert "projector_trainable" in str(config.student.adapter_dir)


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


def test_projector_survives_peft_save_reload_and_merge(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from torch import nn
    from transformers import PretrainedConfig
    from vlm_distill.train_online_align_dbild import _maybe_enable_student_lora
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
    base = Toy()
    trained = _maybe_enable_student_lora(config, base)
    freeze_student_vision_keep_merger_lm_trainable(
        trained, use_lora=True, train_multimodal_projector=True,
        multimodal_projector_path="model.visual.merger",
    )
    expected = torch.full_like(trained.base_model.model.model.visual.merger.modules_to_save.default.weight, 3.0)
    with torch.no_grad():
        trained.base_model.model.model.visual.merger.modules_to_save.default.weight.copy_(expected)
    trained.save_pretrained(tmp_path)

    reloaded = PeftModel.from_pretrained(Toy(), tmp_path)
    actual = reloaded.base_model.model.model.visual.merger.modules_to_save.default.weight
    assert torch.equal(actual, expected)
    merged = reloaded.merge_and_unload()
    assert torch.equal(merged.model.visual.merger.weight, expected)


@pytest.mark.skipif(not Path("configs/qwen3vl8b_r16_attn_mlp.yaml").exists(), reason="A2 config not present")
def test_mlp_lora_is_not_part_of_attention_only_configs():
    config = load_config("configs/qwen3vl8b_r16_attn_mlp.yaml")
    assert {"gate_proj", "up_proj", "down_proj"}.issubset(set(config.student.target_modules))
