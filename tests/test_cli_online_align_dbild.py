from pathlib import Path
import sys

import pytest

from vlm_distill import cli
from vlm_distill.train_online_align_dbild import (
    _scale_partial_accumulation_gradients,
    _weighted_online_align_loss,
    run_training,
)
from tests.test_train_online_align_dbild import _config


def test_train_cli_routes_to_online_align_dbild(monkeypatch, tmp_path: Path, capsys):
    expected_config = object()
    called = []

    monkeypatch.setattr(cli, "load_config", lambda path: expected_config)
    monkeypatch.setattr(cli, "run_training", lambda config: called.append(config) or tmp_path / "adapter")
    monkeypatch.setattr(
        "vlm_distill.stage_student_training.train_student",
        lambda config: pytest.fail("legacy train_student path was called"),
    )
    monkeypatch.setattr(sys, "argv", ["vlm-distill", "train", "--config", str(tmp_path / "config.yaml")])

    cli.main()

    assert called == [expected_config]
    assert "Training backend: online_align_dbild" in capsys.readouterr().out


def test_online_align_method_guard_rejects_response(tmp_path: Path):
    config = _config(tmp_path)
    config.distillation.method = "response"

    with pytest.raises(ValueError, match="distillation.method='online_align_dbild'"):
        run_training(config)


def test_online_align_batch_size_guard_rejects_batch_two(tmp_path: Path):
    config = _config(tmp_path)
    config.distillation.method = "online_align_dbild"
    config.distillation.vsd_loss_weight = 0.0
    config.training.batch_size = 2

    with pytest.raises(ValueError, match="requires training.batch_size == 1"):
        run_training(config)


def test_online_align_vsd_guard_rejects_nonzero_weight(tmp_path: Path):
    config = _config(tmp_path)
    config.distillation.method = "online_align_dbild"
    config.distillation.vsd_loss_weight = 0.5

    with pytest.raises(ValueError, match="vsd_loss_weight must be 0.0"):
        run_training(config)


def test_online_align_loss_uses_configured_weights():
    import torch

    total = _weighted_online_align_loss(
        torch.tensor(2.0),
        torch.tensor(4.0),
        lm_loss_weight=1.25,
        dbild_loss_weight=0.5,
    )

    assert total.item() == pytest.approx(4.5)


def test_partial_accumulation_scaling_and_empty_window_are_safe():
    import torch

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    _scale_partial_accumulation_gradients(model, grad_accum_steps=4, micro_step=2)
    assert model.weight.grad.item() == pytest.approx(2.0)

    model.weight.grad = None
    _scale_partial_accumulation_gradients(model, grad_accum_steps=4, micro_step=0)
    assert model.weight.grad is None
