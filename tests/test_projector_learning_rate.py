from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vlm_distill.train_online_align_dbild as online
from vlm_distill.config_schema import (
    DataConfig,
    PipelineConfig,
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
    _validate_projector_learning_rate_config,
    load_config,
)


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Linear(2, 2)
        self.lora_parameter = nn.Parameter(torch.ones(2))


def _optimizer_config(*, projector_lr=None, train_projector=True, projector_lora=False):
    return SimpleNamespace(
        student=SimpleNamespace(
            train_multimodal_projector=train_projector,
            use_projector_lora=projector_lora,
            multimodal_projector_path="model.visual.merger",
        ),
        training=SimpleNamespace(
            learning_rate=1e-4,
            projector_learning_rate=projector_lr,
        ),
    )


def _patch_legacy_contracts(monkeypatch):
    monkeypatch.setattr(
        online,
        "summarize_trainable_groups",
        lambda *_args: {
            "attention_lora": 1,
            "projector": 8,
            "vision_encoder": 0,
            "base_llm": 0,
            "other": 0,
        },
    )
    monkeypatch.setattr(online, "_validate_a1_trainable_contract", lambda *_args: None)


def test_projector_lr_none_puts_everything_in_default_group(monkeypatch):
    _patch_legacy_contracts(monkeypatch)
    optimizer = online._build_optimizer(_optimizer_config(), ToyStudent())
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["group_name"] == "default"
    assert optimizer.param_groups[0]["lr"] == 1e-4


def test_full_projector_uses_separate_identity_based_group(monkeypatch, capsys):
    _patch_legacy_contracts(monkeypatch)
    model = ToyStudent()
    optimizer = online._build_optimizer(_optimizer_config(projector_lr=1e-5), model)
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "default", "multimodal_projector"
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-5]
    default_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    projector_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    expected_projector_ids = {id(p) for p in model.model.visual.merger.parameters()}
    assert projector_ids == expected_projector_ids
    assert default_ids.isdisjoint(projector_ids)
    assert default_ids | projector_ids == {id(p) for p in model.parameters() if p.requires_grad}
    output = capsys.readouterr().out
    assert "default_lr=0.0001 projector_lr=0.00001" in output
    assert "trainable_parameters=" in output


def test_projector_group_uses_configured_module_path(monkeypatch):
    _patch_legacy_contracts(monkeypatch)
    model = ToyStudent()
    model.model.visual.projector = model.model.visual.merger
    del model.model.visual.merger
    config = _optimizer_config(projector_lr=1e-5)
    config.student.multimodal_projector_path = "model.visual.projector"
    optimizer = online._build_optimizer(config, model)
    projector_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert projector_ids == {id(p) for p in model.model.visual.projector.parameters()}


def test_frozen_parameters_are_not_in_optimizer(monkeypatch):
    _patch_legacy_contracts(monkeypatch)
    model = ToyStudent()
    model.lora_parameter.requires_grad = False
    optimizer = online._build_optimizer(_optimizer_config(projector_lr=1e-5), model)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.lora_parameter) not in ids


def test_no_trainable_projector_is_explicit_error(monkeypatch):
    _patch_legacy_contracts(monkeypatch)
    model = ToyStudent()
    for parameter in model.model.visual.merger.parameters():
        parameter.requires_grad = False
    with pytest.raises(RuntimeError, match="no trainable full-projector parameters"):
        online._build_optimizer(_optimizer_config(projector_lr=1e-5), model)


def test_invalid_projector_path_is_explicit_error(monkeypatch):
    _patch_legacy_contracts(monkeypatch)
    config = _optimizer_config(projector_lr=1e-5)
    config.student.multimodal_projector_path = "model.visual.missing"
    with pytest.raises(RuntimeError, match="invalid multimodal projector path"):
        online._build_optimizer(config, ToyStudent())


def test_projector_lora_mode_stays_in_default_group(monkeypatch):
    _patch_legacy_contracts(monkeypatch)
    optimizer = online._build_optimizer(
        _optimizer_config(projector_lr=None, projector_lora=True), ToyStudent()
    )
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["group_name"] == "default"


def test_scheduler_preserves_projector_lr_ratio(monkeypatch):
    _patch_legacy_contracts(monkeypatch)
    optimizer = online._build_optimizer(_optimizer_config(projector_lr=1e-5), ToyStudent())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"] == pytest.approx(10.0)


def _pipeline_config(*, projector_lr=None, train_projector=True, projector_lora=False):
    return PipelineConfig(
        data=DataConfig(training_manifest_path="train.jsonl", distill_path="labels.jsonl"),
        teacher=TeacherConfig(model_name="teacher"),
        student=StudentConfig(
            model_name="student", output_dir="out", adapter_dir="adapter",
            train_multimodal_projector=train_projector,
            use_projector_lora=projector_lora,
        ),
        training=TrainingConfig(projector_learning_rate=projector_lr),
    )


@pytest.mark.parametrize("value", [0, -1e-5])
def test_projector_lr_must_be_positive(value):
    with pytest.raises(ValueError, match="projector_learning_rate must be > 0"):
        _validate_projector_learning_rate_config(_pipeline_config(projector_lr=value))


@pytest.mark.parametrize("train_projector,projector_lora", [(False, False), (True, True)])
def test_projector_lr_requires_full_projector(train_projector, projector_lora):
    with pytest.raises(ValueError, match="requires student.train_multimodal_projector=true"):
        _validate_projector_learning_rate_config(
            _pipeline_config(
                projector_lr=1e-5,
                train_projector=train_projector,
                projector_lora=projector_lora,
            )
        )


def test_a4_configs_and_legacy_config_parse():
    formal = load_config("configs/lora_ablation/stage1_a4_r16_attn_mlp_projector.yaml")
    smoke = load_config("configs/lora_ablation/smoke_validation_a4_r16.yaml")
    legacy = load_config("configs/parsing_switch_kd_test.yaml")
    assert formal.training.projector_learning_rate == 1e-5
    assert smoke.training.projector_learning_rate == 1e-5
    assert legacy.training.projector_learning_rate is None
