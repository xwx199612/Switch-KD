import torch

from vlm_distill.loss_switch_kd import (
    _internal_pairwise_logits_differences,
    full_dynamic_bidirectional_logits_difference,
)


def test_internal_pairwise_logits_differences_values():
    selected = torch.tensor([[[5.0, 3.0, 1.0]]])

    diffs, active = _internal_pairwise_logits_differences(selected)

    expected = torch.tensor([[[2.0, 4.0, 2.0]]])
    assert torch.equal(diffs, expected)
    assert torch.equal(active, torch.ones_like(expected, dtype=torch.bool))


def test_internal_pairwise_logits_differences_active_mask():
    selected = torch.tensor([[[5.0, 3.0, 1.0]]])
    active_mask = torch.tensor([[[True, True, False]]])

    diffs, active = _internal_pairwise_logits_differences(selected, active_mask)

    expected_diffs = torch.tensor([[[2.0, 4.0, 2.0]]])
    expected_active = torch.tensor([[[True, False, False]]])
    assert torch.equal(diffs, expected_diffs)
    assert torch.equal(active, expected_active)


def test_full_dbild_backward_is_finite():
    reference_logits = torch.tensor(
        [[[4.0, 2.0, 1.0], [1.5, 0.5, -0.5]]],
        dtype=torch.float32,
    )
    target_logits = torch.tensor(
        [[[3.5, 2.5, 0.0], [1.0, 0.25, -0.75]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)

    loss = full_dynamic_bidirectional_logits_difference(
        reference_logits=reference_logits,
        target_logits=target_logits,
        attention_mask=attention_mask,
        temperature=2.0,
        top_k=3,
        top_k_mode="fixed",
        kl_mode="symmetric",
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert target_logits.grad is not None
    assert torch.isfinite(target_logits.grad).all()


def test_full_dbild_top_k_one_raises():
    reference_logits = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.float32)
    target_logits = torch.tensor([[[0.5, 0.0, -0.5]]], dtype=torch.float32)

    try:
        full_dynamic_bidirectional_logits_difference(
            reference_logits=reference_logits,
            target_logits=target_logits,
            top_k=1,
            top_k_mode="fixed",
        )
    except ValueError as exc:
        assert "top_k >= 2" in str(exc)
    else:
        raise AssertionError("Expected ValueError for top_k=1 in full DBiLD.")
