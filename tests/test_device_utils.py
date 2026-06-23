from __future__ import annotations

import pytest

from vlm_distill.device_utils import (
    is_distributed_training_active,
    resolve_requested_device_map,
    resolve_training_device_map,
)


def test_resolve_requested_device_map_rejects_none_for_teacher():
    with pytest.raises(ValueError, match="teacher.device_map is required"):
        resolve_requested_device_map(None, quantization="none", role="teacher")


def test_resolve_training_device_map_defaults_to_auto_without_distributed(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    assert is_distributed_training_active() is False
    assert resolve_training_device_map(None, quantization="none", role="student") == "auto"


def test_resolve_training_device_map_returns_none_in_distributed_mode(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")

    assert is_distributed_training_active() is True
    assert resolve_training_device_map(None, quantization="none", role="student") is None


def test_resolve_training_device_map_allows_quantized_ddp_without_auto(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")

    assert resolve_training_device_map(None, quantization="4bit", role="student") is None


def test_resolve_training_device_map_keeps_quantized_non_ddp_strict(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    with pytest.raises(ValueError, match="must be 'auto'"):
        resolve_training_device_map("cuda:0", quantization="4bit", role="student")
