from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn

from vlm_distill.deployment_loader import _tensor_digest
from vlm_distill.student_trainability import merger_base_checksum, merger_base_tensors, tensor_storage_bytes


class _MergerModel(nn.Module):
    def __init__(self, dtype=torch.bfloat16, reverse=False, *, with_bias=True):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Module()
        merger = self.model.visual.merger
        children = [
            ("norm", nn.LayerNorm(2, dtype=torch.float32)),
            ("linear_fc1", nn.Linear(2, 3, dtype=dtype, bias=with_bias)),
            ("linear_fc2", nn.Linear(3, 2, dtype=dtype, bias=with_bias)),
        ]
        if reverse:
            children.reverse()
        for name, child in children:
            setattr(merger, name, child)


def test_bf16_and_fp32_tensors_produce_checksums():
    assert merger_base_checksum(_MergerModel(torch.bfloat16))
    assert merger_base_checksum(_MergerModel(torch.float32))


def test_same_bf16_tensor_has_same_checksum_and_one_bit_change_does_not():
    model = _MergerModel(torch.bfloat16)
    before = merger_base_checksum(model)
    same = merger_base_checksum(model)
    model.model.visual.merger.linear_fc1.weight.data.view(torch.uint8)[0] ^= 1
    after = merger_base_checksum(model)
    assert before == same
    assert before != after


def test_dtype_shape_and_parameter_order_are_checksum_inputs():
    bf16 = _MergerModel(torch.bfloat16)
    fp32 = _MergerModel(torch.float32)
    reordered = _MergerModel(torch.bfloat16, reverse=True)
    different_shape = _MergerModel(torch.bfloat16)
    different_shape.model.visual.merger.linear_fc1 = nn.Linear(2, 4, dtype=torch.bfloat16)

    # Make values identical wherever shapes permit, isolating dtype/order here.
    left_parameters = dict(bf16.model.visual.merger.named_parameters())
    for name, right in fp32.model.visual.merger.named_parameters():
        right.data.copy_(left_parameters[name].data)
    for name, right in reordered.model.visual.merger.named_parameters():
        right.data.copy_(left_parameters[name].data)
    assert merger_base_checksum(bf16) == merger_base_checksum(reordered)
    assert merger_base_checksum(bf16) != merger_base_checksum(fp32)
    assert merger_base_checksum(bf16) != merger_base_checksum(different_shape)


def test_missing_bias_is_skipped_using_the_same_canonical_rule():
    assert merger_base_checksum(_MergerModel(with_bias=False))


def test_quantized_tensor_is_explicitly_rejected():
    quantized = torch.quantize_per_tensor(torch.ones(2), scale=0.1, zero_point=0, dtype=torch.qint8)
    with pytest.raises(TypeError, match="does not support PyTorch quantized tensors"):
        tensor_storage_bytes(quantized)

    model = _MergerModel()
    model.model.visual.merger.linear_fc1.weight = nn.Parameter(quantized, requires_grad=False)
    with pytest.raises(RuntimeError, match="requires floating tensors"):
        merger_base_checksum(model)


def test_storage_helper_preserves_bf16_storage_bytes():
    value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    assert tensor_storage_bytes(value) == value.view(torch.uint8).numpy().tobytes()


def test_training_and_deployment_checksum_match_for_same_bf16_merger():
    model = _MergerModel(torch.bfloat16)
    assert merger_base_checksum(model) == _tensor_digest(merger_base_tensors(model))
