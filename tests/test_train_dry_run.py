from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vlm_distill.train_online_align_dbild import (
    TrainableSummary,
    _print_dry_run_summary,
    run_training,
)

from .test_stage1_a3_attn_mlp_projector import _FakeQwen3VL


def _a3_config():
    return SimpleNamespace(
        distillation=SimpleNamespace(method="online_align_dbild", vsd_loss_weight=0.0,
                                     lm_loss_weight=1.0, dbild_loss_weight=1.0),
        training=SimpleNamespace(batch_size=1),
        student=SimpleNamespace(
            model_name="fake-student",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_lora=True,
            train_multimodal_projector=True,
            use_projector_lora=False,
            multimodal_projector_path="model.visual.merger",
            adapter_dir=None,
        ),
    )


def test_a3_dry_run_summary_reports_contract_counts_and_dtypes(capsys):
    _print_dry_run_summary(_a3_config(), _FakeQwen3VL())
    output = capsys.readouterr().out
    assert "attention target modules = 144" in output
    assert "MLP target modules = 108" in output
    assert "total LoRA target modules = 252" in output
    assert "projector mode = modules_to_save.default" in output
    assert "projector LoRA = 0" in output
    assert "vision trainable = 0" in output
    assert "base LM trainable = 0" in output
    assert "other trainable = 0" in output
    assert "attention dtype summary = float32" in output
    assert "MLP dtype summary = float32" in output
    assert "projector dtype summary = bfloat16" in output
    assert "GPU allocated/reserved memory = " in output


def test_dry_run_stops_before_teacher_optimizer_or_training(monkeypatch, capsys):
    config = _a3_config()
    model = _FakeQwen3VL()
    calls = {"setup": 0}

    def fake_load_student(_config):
        return model, object(), "fake-student", None

    def fake_setup(_config, loaded_model, *, dry_run=False):
        calls["setup"] += 1
        assert dry_run is True
        return loaded_model, TrainableSummary(1, 1, 1.0, 1, [])

    monkeypatch.setattr("vlm_distill.train_online_align_dbild._load_student", fake_load_student)
    monkeypatch.setattr("vlm_distill.train_online_align_dbild._apply_student_train_setup", fake_setup)
    monkeypatch.setattr(
        "vlm_distill.train_online_align_dbild._load_teacher",
        lambda *_args, **_kwargs: pytest.fail("dry-run loaded teacher"),
    )
    monkeypatch.setattr(
        "vlm_distill.train_online_align_dbild._build_optimizer",
        lambda *_args, **_kwargs: pytest.fail("dry-run built optimizer"),
    )
    monkeypatch.setattr(model, "forward", lambda *_args, **_kwargs: pytest.fail("dry-run forwarded"))
    result = run_training(config, dry_run=True)

    assert result is None
    assert calls["setup"] == 1
    assert "dry-run complete: optimizer=not_created forward=not_run backward=not_run checkpoint=not_written" in capsys.readouterr().out
