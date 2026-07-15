from dataclasses import asdict, is_dataclass
import copy
from pathlib import Path
import types

import pytest
import torch
from torch import nn

from vlm_distill.config_schema import load_config
from vlm_distill.student_trainability import validate_language_model_lora_scope


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
B1 = list(range(12, 36))
B2 = list(range(24, 36))
FORMAL = {
    "b0": "configs/stage1_b0_r16_attn_all_layers.yaml",
    "b1": "configs/stage1_b1_r16_attn_layers_12_35.yaml",
    "b2": "configs/stage1_b2_r16_attn_layers_24_35.yaml",
}
SMOKE = {
    "b1": "configs/stage1_b1_r16_attn_layers_12_35_smoke.yaml",
    "b2": "configs/stage1_b2_r16_attn_layers_24_35_smoke.yaml",
}
STAGE1_A_R16_ATTN = "configs/qwen3vl8b_r16_attn.yaml"

FORMAL_ALLOWED_PATHS = {
    "student.output_dir", "student.adapter_dir", "student.merged_model_path",
    "data.eval_path", "data.prediction_path", "evaluation.output_path",
    "student.lora_layers_to_transform", "student.lora_layers_pattern",
}
SMOKE_ALLOWED_PATHS = FORMAL_ALLOWED_PATHS | {
    "data.max_samples", "training.epochs", "training.gradient_accumulation_steps",
    "training.max_steps", "training.log_every", "training.save_every",
}


def normalize_experiment_config(config, allowed_paths: set[str]) -> dict:
    """Return config values after removing only explicitly allowed paths."""
    values = asdict(config) if is_dataclass(config) else config

    def walk(value, path=""):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                if child in allowed_paths:
                    continue
                result[key] = walk(item, child)
            return result
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [walk(item, path) for item in value]
        return value

    return walk(values)


def _config_differences(left, right, *, allowed_paths: set[str]) -> dict:
    differences = {}

    def walk(path, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(f"{path}.{key}" if path else key, a.get(key), b.get(key))
        elif a != b:
            differences[path] = (a, b)

    walk(
        "",
        normalize_experiment_config(left, allowed_paths),
        normalize_experiment_config(right, allowed_paths),
    )
    return differences


class _Toy(nn.Module):
    def __init__(self, layers=B1, *, extra=None, visual=False, mlp=False, projector=False, base=False):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Module() for _ in range(36)])
        selected = set(layers)
        for index, layer in enumerate(self.model.language_model.layers):
            layer.self_attn = nn.Module()
            for target in TARGETS:
                module = nn.Module()
                if index in selected or (extra is not None and index == extra):
                    module.register_parameter("lora_A", nn.Parameter(torch.ones(2, 2)))
                    module.register_parameter("lora_B", nn.Parameter(torch.ones(2, 2)))
                layer.self_attn.add_module(target, module)
            if mlp and index == 12:
                module = nn.Module()
                module.register_parameter("lora_A", nn.Parameter(torch.ones(2, 2)))
                layer.add_module("gate_proj", module)
        if visual:
            self.model.visual = nn.Module()
            self.model.visual.register_parameter("lora_A", nn.Parameter(torch.ones(2, 2)))
        if projector:
            self.model.visual = getattr(self.model, "visual", nn.Module())
            self.model.visual.merger = nn.Linear(2, 2)
        if base:
            self.model.language_model.base = nn.Parameter(torch.ones(2, 2))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name)
        if projector:
            for parameter in self.model.visual.merger.parameters():
                parameter.requires_grad_(True)
        if base:
            self.model.language_model.base.requires_grad_(True)


def test_null_scope_targets_all_language_model_layers():
    report = validate_language_model_lora_scope(_Toy(range(36)), None, TARGETS)
    assert all(value == list(range(36)) for value in report["detected_layers"].values())


@pytest.mark.parametrize("layers", [B1, B2])
def test_scopes_target_exact_layers(layers):
    report = validate_language_model_lora_scope(_Toy(layers), layers, TARGETS)
    assert all(value == layers for value in report["detected_layers"].values())


@pytest.mark.parametrize("kwargs", [{"layers": B1, "extra": 5}, {"layers": B1, "visual": True}, {"layers": B1, "mlp": True}, {"layers": B1, "projector": True}, {"layers": B1, "base": True}])
def test_invalid_trainability_fails(kwargs):
    with pytest.raises(RuntimeError):
        validate_language_model_lora_scope(_Toy(**kwargs), B1, TARGETS)


@pytest.mark.parametrize("bad", [[1, 1], [-1], []])
def test_layer_schema_rejects_invalid_indices(tmp_path: Path, bad):
    config = {
        "data": {"training_manifest_path": "m", "distill_path": "d"},
        "teacher": {"model_name": "mock-teacher"},
        "student": {"model_name": "mock-student", "output_dir": str(tmp_path / "o"),
                    "adapter_dir": str(tmp_path / "a"), "lora_layers_to_transform": bad,
                    "lora_layers_pattern": "layers"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_b0_matches_completed_stage1_a_r16_attention():
    b0 = load_config(FORMAL["b0"])
    stage1_a = load_config(STAGE1_A_R16_ATTN)
    differences = _config_differences(b0, stage1_a, allowed_paths=FORMAL_ALLOWED_PATHS)
    assert not differences, f"B0 differs from {STAGE1_A_R16_ATTN}: {differences}"
    assert b0.student.lora_layers_to_transform is None
    assert b0.student.lora_layers_pattern is None


def test_stage1_b_configs_differ_only_by_scope_and_paths():
    configs = [load_config(FORMAL[key]) for key in ("b0", "b1", "b2")]
    for left, right in zip(configs, configs[1:]):
        differences = _config_differences(left, right, allowed_paths=FORMAL_ALLOWED_PATHS)
        assert not differences, f"unexpected controlled-experiment differences: {differences}"


def test_stage1_b_scope_prompt_and_dbild_settings_are_explicit_and_equal():
    configs = [load_config(FORMAL[key]) for key in ("b0", "b1", "b2")]
    assert [c.student.lora_layers_to_transform for c in configs] == [None, B1, B2]
    assert [c.student.lora_layers_pattern for c in configs] == [None, "layers", "layers"]
    prompts = [c.distillation.prompt_template for c in configs]
    assert prompts[0] == prompts[1] == prompts[2]
    assert prompts[0] != "Query: {query}\nAnswer:"
    for text in ("bbox_norm", "normalized 0-1000", "valid JSON only"):
        assert text in prompts[0]
    for config in configs:
        distill = config.distillation
        assert distill.method == "online_align_dbild"
        assert distill.kd_temperature == 2.0
        assert distill.dbild_top_k == 64
        assert distill.dbild_top_k_mode == "kneedle"
        assert distill.dbild_kneedle_candidate_k == 256
        assert distill.dbild_min_top_k == 4
        assert distill.dbild_max_top_k == 128
        assert distill.dbild_kl_mode == "reverse"
        assert config.student.train_multimodal_projector is False
        assert config.student.lora_rank == 16
        assert config.student.target_modules == TARGETS


def test_stage1_b_smoke_configs_differ_from_formal_only_by_smoke_controls_and_paths():
    for key in ("b1", "b2"):
        differences = _config_differences(
            load_config(FORMAL[key]),
            load_config(SMOKE[key]),
            allowed_paths=SMOKE_ALLOWED_PATHS,
        )
        assert not differences, f"{key} smoke has uncontrolled differences: {differences}"


def test_formal_equality_detects_epoch_difference():
    left = copy.deepcopy(asdict(load_config(FORMAL["b1"])))
    right = copy.deepcopy(left)
    right["training"]["epochs"] += 1
    differences = _config_differences(left, right, allowed_paths=FORMAL_ALLOWED_PATHS)
    assert set(differences) == {"training.epochs"}


def test_formal_equality_detects_gradient_accumulation_difference():
    left = copy.deepcopy(asdict(load_config(FORMAL["b1"])))
    right = copy.deepcopy(left)
    right["training"]["gradient_accumulation_steps"] += 1
    differences = _config_differences(left, right, allowed_paths=FORMAL_ALLOWED_PATHS)
    assert set(differences) == {"training.gradient_accumulation_steps"}


def test_formal_equality_detects_dbild_difference():
    left = copy.deepcopy(asdict(load_config(FORMAL["b1"])))
    right = copy.deepcopy(left)
    right["distillation"]["dbild_top_k_mode"] = "fixed"
    differences = _config_differences(left, right, allowed_paths=FORMAL_ALLOWED_PATHS)
    assert set(differences) == {"distillation.dbild_top_k_mode"}


def test_formal_equality_detects_prompt_difference():
    left = copy.deepcopy(asdict(load_config(FORMAL["b1"])))
    right = copy.deepcopy(left)
    right["distillation"]["prompt_template"] += "\nChanged."
    differences = _config_differences(left, right, allowed_paths=FORMAL_ALLOWED_PATHS)
    assert set(differences) == {"distillation.prompt_template"}


def test_smoke_equality_allows_only_smoke_controls():
    formal = copy.deepcopy(asdict(load_config(FORMAL["b1"])))
    smoke = copy.deepcopy(formal)
    for path, value in {
        "data.max_samples": 2,
        "training.epochs": 1,
        "training.gradient_accumulation_steps": 1,
        "training.max_steps": 1,
        "training.log_every": 1,
        "training.save_every": 1,
        "student.output_dir": "smoke/output",
    }.items():
        target = smoke
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    assert not _config_differences(formal, smoke, allowed_paths=SMOKE_ALLOWED_PATHS)

    smoke["training"]["learning_rate"] *= 2
    differences = _config_differences(formal, smoke, allowed_paths=SMOKE_ALLOWED_PATHS)
    assert set(differences) == {"training.learning_rate"}


class _PeftToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="toy")
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList()
        for _ in range(36):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(4096, 4096, bias=False, device="meta")
            layer.self_attn.k_proj = nn.Linear(4096, 1024, bias=False, device="meta")
            layer.self_attn.v_proj = nn.Linear(4096, 1024, bias=False, device="meta")
            layer.self_attn.o_proj = nn.Linear(4096, 4096, bias=False, device="meta")
            layer.mlp = nn.Linear(2, 2, bias=False, device="meta")
            self.model.language_model.layers.append(layer)
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Linear(2, 2, bias=False, device="meta")


def _actual_lora_counts(layers):
    from peft import LoraConfig, get_peft_model

    model = _PeftToy()
    kwargs = dict(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=TARGETS, task_type=None,
    )
    if layers is not None:
        kwargs.update(layers_to_transform=layers, layers_pattern="layers")
    wrapped = get_peft_model(model, LoraConfig(**kwargs))
    lora = [(name, parameter) for name, parameter in wrapped.named_parameters()
            if parameter.requires_grad and "lora_" in name]
    counts = {
        "tensors": len(lora),
        "params": sum(parameter.numel() for _, parameter in lora),
        "non_lora_trainable": sum(parameter.numel() for name, parameter in wrapped.named_parameters()
                                    if parameter.requires_grad and "lora_" not in name),
    }
    counts["visual_lora"] = sum(parameter.numel() for name, parameter in lora if ".visual." in name)
    counts["mlp_lora"] = sum(parameter.numel() for name, parameter in lora if ".mlp." in name)
    counts["base_lm_trainable"] = sum(parameter.numel() for name, parameter in wrapped.named_parameters()
                                      if parameter.requires_grad and "language_model.layers" not in name and "lora_" not in name)
    return counts


def test_stage1_b_actual_peft_lora_tensor_and_parameter_counts():
    counts = [_actual_lora_counts(layers) for layers in (None, B1, B2)]
    assert [item["tensors"] for item in counts] == [288, 192, 96]
    assert [item["params"] for item in counts] == [15335424, 10223616, 5111808]
    for item in counts:
        assert item["non_lora_trainable"] == 0
        assert item["visual_lora"] == 0
        assert item["mlp_lora"] == 0
        assert item["base_lm_trainable"] == 0
