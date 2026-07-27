from pathlib import Path
import pytest
import torch

from vlm_distill.config_schema import (
    DataConfig,
    PipelineConfig,
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
    _build_training_config,
    _migrate_legacy_validation_config,
    _validate_training_validation_config,
    load_config,
)
from vlm_distill.train_online_align_dbild import (
    _broadcast_early_stop,
    _early_stopping_update,
    _reduce_validation_totals,
    _restore_best_checkpoint,
)


def _config(tmp_path: Path, **training):
    return PipelineConfig(
        data=DataConfig(
            training_manifest_path=tmp_path / "train.jsonl",
            distill_path=tmp_path / "labels.jsonl",
        ),
        teacher=TeacherConfig(model_name="teacher"),
        student=StudentConfig(model_name="student", output_dir=tmp_path, adapter_dir=tmp_path / "adapter"),
        training=TrainingConfig(
            validation_manifest_path=training.pop("validation_manifest_path", None),
            validation_image_dir=training.pop("validation_image_dir", tmp_path / "validation-images"),
            validation_ratio=training.pop("validation_ratio", 0.2),
            **training,
        ),
    )


def test_validation_disabled_does_not_require_manifest(tmp_path):
    _validate_training_validation_config(_config(tmp_path))


def test_early_stopping_requires_validation(tmp_path):
    with pytest.raises(ValueError, match="requires training.validation_enabled=true"):
        _validate_training_validation_config(_config(tmp_path, early_stopping_enabled=True))


def test_validation_config_ranges_and_manifest(tmp_path):
    for field, value in (("validation_every_epochs", 0), ("early_stopping_patience", 0)):
        with pytest.raises(ValueError, match="must be > 0"):
            _validate_training_validation_config(_config(tmp_path, **{field: value}))
    with pytest.raises(ValueError, match="must be >= 0"):
        _validate_training_validation_config(_config(tmp_path, early_stopping_min_delta=-1))
    with pytest.raises(ValueError, match="validation_manifest_path"):
        _validate_training_validation_config(_config(tmp_path, validation_enabled=True))
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    _validate_training_validation_config(
        _config(tmp_path, validation_enabled=True, validation_manifest_path=manifest)
    )


def test_patience_and_min_delta_are_strict(tmp_path):
    best, bad, is_best, stop = _early_stopping_update(
        0.90, 1.0, 0, min_delta=0.1, patience=2
    )
    assert (best, bad, is_best, stop) == (1.0, 1, False, False)
    assert _early_stopping_update(0.9, 1.0, 1, min_delta=0.1, patience=2)[3]
    assert _early_stopping_update(0.89, 1.0, 0, min_delta=0.01, patience=2)[2]


def test_validation_split_config_fields_and_seed_fallback(tmp_path):
    training = _build_training_config({
        "validation_enabled": True,
        "validation_ratio": 0.2,
        "validation_split_seed": None,
        "validation_split_mode": "copy",
        "validation_image_dir": str(tmp_path / "images"),
        "validation_manifest_path": str(tmp_path / "validation.jsonl"),
    })
    assert training.validation_split_seed is None
    assert training.validation_split_mode == "copy"
    assert training.validation_image_dir == tmp_path / "images"
    assert training.validation_manifest_path == tmp_path / "validation.jsonl"


def test_invalid_validation_split_mode_is_rejected(tmp_path):
    config = _config(tmp_path, validation_enabled=False)
    config.training.validation_split_mode = "link"
    with pytest.raises(ValueError, match="validation_split_mode"):
        _validate_training_validation_config(config)


def test_legacy_validation_paths_migrate_and_conflicts_fail(tmp_path):
    migrated = _migrate_legacy_validation_config({
        "data": {"validation_manifest_path": str(tmp_path / "old.jsonl")},
        "training": {},
    })
    assert migrated["training"]["validation_manifest_path"].endswith("old.jsonl")
    assert "validation_manifest_path" not in migrated["data"]
    with pytest.raises(ValueError, match="Conflicting validation paths"):
        _migrate_legacy_validation_config({
            "data": {"validation_manifest_path": "old.jsonl"},
            "training": {"validation_manifest_path": "new.jsonl"},
        })


def test_old_config_keeps_validation_disabled():
    config = load_config("configs/parsing_switch_kd_test.yaml")
    assert config.training.validation_enabled is False
    assert config.training.early_stopping_enabled is False
    assert config.training.validation_manifest_path is None


def test_single_process_reduce_and_broadcast_are_identity():
    assert _reduce_validation_totals(3.5, 2) == (3.5, 2)
    assert _broadcast_early_stop(True) is True
    assert _broadcast_early_stop(False) is False


def test_ddp_loss_reduce_and_early_stop_broadcast(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(values, op=None):
        del op
        values[0] *= 2
        values[1] *= 2

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "broadcast", lambda values, src: values.fill_(1))
    assert _reduce_validation_totals(3.5, 2) == (7.0, 4)
    assert _broadcast_early_stop(False) is True


def test_best_checkpoint_restore_restores_model_optimizer_and_scheduler(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "best_checkpoint"
    checkpoint.mkdir()
    torch.save({
        "model_state_dict": expected,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": 2,
        "global_step": 4,
        "best_val_loss": 0.5,
        "epochs_without_improvement": 0,
    }, checkpoint / "training_state.pt")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    state = _restore_best_checkpoint(model, optimizer, scheduler, checkpoint)
    assert state["epoch"] == 2
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())
