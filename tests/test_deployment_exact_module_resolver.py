from __future__ import annotations

import pytest
import torch
from torch import nn

from vlm_distill.deployment_loader import (
    _active_merger,
    _active_modules_to_save_projector,
    resolve_exact_module,
)


def _merger(*, wrapped: bool = False) -> nn.Module:
    merger = nn.Module()
    merger.linear_fc1 = nn.Linear(2, 2, dtype=torch.bfloat16)
    merger.linear_fc2 = nn.Linear(2, 2, dtype=torch.bfloat16)
    if wrapped:
        active = nn.Module()
        active.linear_fc1 = nn.Linear(2, 2, dtype=torch.bfloat16)
        active.linear_fc2 = nn.Linear(2, 2, dtype=torch.bfloat16)
        merger.modules_to_save = nn.ModuleDict({"default": active})
        merger.original_module = nn.Linear(2, 2)
    return merger


def _qwen_tree(*, wrapped: bool = False) -> nn.Module:
    qwen = nn.Module()
    qwen.model = nn.Module()
    qwen.model.visual = nn.Module()
    qwen.model.visual.merger = _merger(wrapped=wrapped)
    return qwen


def test_unwrapped_model_resolves_main_merger():
    root = _qwen_tree()
    assert resolve_exact_module(root, "model.visual.merger") is root.model.visual.merger


def test_single_peft_wrapper_resolves_main_merger():
    root = nn.Module()
    root.base_model = nn.Module()
    root.base_model.model = _qwen_tree()
    assert resolve_exact_module(root, "model.visual.merger") is root.base_model.model.model.visual.merger


def test_qwen3vl_peft_model_layout_does_not_assume_qwen_has_visual():
    # PeftModel.model -> Qwen3VLForConditionalGeneration; Qwen's visual is
    # under its own model attribute.
    root = nn.Module()
    root.model = _qwen_tree()
    assert resolve_exact_module(root, "model.visual.merger") is root.model.model.visual.merger


def test_modules_to_save_default_is_the_active_merger():
    root = _qwen_tree(wrapped=True)
    wrapper = root.model.visual.merger
    assert _active_merger(root) is wrapper.modules_to_save["default"]
    assert _active_modules_to_save_projector(root) is wrapper.modules_to_save["default"]
    assert _active_merger(root) is not wrapper.original_module


def test_deepstack_only_is_rejected_with_diagnostics():
    root = nn.Module()
    root.model = nn.Module()
    root.model.visual = nn.Module()
    root.model.visual.deepstack_merger_list = nn.ModuleList([_merger()])
    with pytest.raises(AttributeError, match="requested path=.*projector candidates.*deepstack_merger"):
        resolve_exact_module(root, "model.visual.merger")


def test_two_exact_main_merger_candidates_fail_fast():
    root = nn.Module()
    root.model = nn.Module()
    root.model.visual = nn.Module()
    root.model.visual.merger = _merger()
    root.base_model = nn.Module()
    root.base_model.model = nn.Module()
    root.base_model.model.model = nn.Module()
    root.base_model.model.model.visual = nn.Module()
    root.base_model.model.model.visual.merger = _merger()
    with pytest.raises(AttributeError, match="projector candidates"):
        resolve_exact_module(root, "model.visual.merger")


def test_missing_main_merger_reports_candidates():
    root = nn.Module()
    root.model = nn.Module()
    root.model.visual = nn.Module()
    root.model.visual.other_merger = nn.Linear(2, 2)
    with pytest.raises(AttributeError, match="other_merger"):
        resolve_exact_module(root, "model.visual.merger")
