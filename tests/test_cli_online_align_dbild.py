from pathlib import Path
import sys

import pytest

from vlm_distill import cli
from vlm_distill.train_online_align_dbild import (
    _enable_student_gradient_checkpointing,
    _student_gradient_checkpointing_modules,
    _student_gradient_checkpointing_use_reentrant,
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


def test_partial_accumulation_scaling_uses_current_window_step():
    import torch

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[1.0]])

    _scale_partial_accumulation_gradients(model, grad_accum_steps=8, micro_step=10)

    assert model.weight.grad.item() == pytest.approx(4.0)


def test_online_align_student_checkpointing_is_explicitly_non_reentrant():
    import functools
    import torch

    class CheckpointToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._gradient_checkpointing_func = None

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs=None):
            self._gradient_checkpointing_func = functools.partial(
                object, **(gradient_checkpointing_kwargs or {})
            )

    model = CheckpointToy()

    assert _enable_student_gradient_checkpointing(model) is False
    assert model._gradient_checkpointing_func.keywords == {"use_reentrant": False}


def test_student_checkpointing_reads_non_reentrant_child_module():
    import functools
    import torch

    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._gradient_checkpointing_func = functools.partial(object, use_reentrant=False)

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()

    model = Parent()

    assert _student_gradient_checkpointing_use_reentrant(model) is False
    assert _student_gradient_checkpointing_modules(model) == [
        {"module": "child", "use_reentrant": False}
    ]


def test_student_checkpointing_all_child_modules_non_reentrant_pass():
    import functools
    import torch

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.children_with_checkpoint = torch.nn.ModuleList()
            for _ in range(2):
                child = torch.nn.Linear(1, 1)
                child._gradient_checkpointing_func = functools.partial(object, use_reentrant=False)
                self.children_with_checkpoint.append(child)

    assert _student_gradient_checkpointing_use_reentrant(Parent()) is False


def test_student_checkpointing_rejects_true_child_module():
    import functools
    import torch

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(1, 1)
            self.child._gradient_checkpointing_func = functools.partial(object, use_reentrant=True)

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs=None):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    with pytest.raises(RuntimeError, match="use_reentrant=False"):
        _enable_student_gradient_checkpointing(Parent())


def test_student_checkpointing_rejects_mixed_child_modules():
    import functools
    import torch

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.children_with_checkpoint = torch.nn.ModuleList()
            for use_reentrant in (False, True):
                child = torch.nn.Linear(1, 1)
                child._gradient_checkpointing_func = functools.partial(
                    object, use_reentrant=use_reentrant
                )
                self.children_with_checkpoint.append(child)

    with pytest.raises(RuntimeError, match="disagree about use_reentrant"):
        _student_gradient_checkpointing_use_reentrant(Parent())


def test_student_checkpointing_missing_function_returns_none_and_enable_fails():
    import torch

    class Parent(torch.nn.Module):
        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs=None):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    model = Parent()
    assert _student_gradient_checkpointing_use_reentrant(model) is None
    with pytest.raises(RuntimeError, match="did not activate use_reentrant=False"):
        _enable_student_gradient_checkpointing(model)


def test_online_align_checkpointing_refuses_models_without_explicit_kwarg_support():
    class LegacyCheckpointToy:
        def gradient_checkpointing_enable(self):
            pass

    with pytest.raises(RuntimeError, match="refusing to use"):
        _enable_student_gradient_checkpointing(LegacyCheckpointToy())
