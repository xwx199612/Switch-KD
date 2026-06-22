import torch

from vlm_distill.loss_switch_kd import dynamic_bidirectional_logits_difference


def test_dynamic_bidirectional_logits_difference_runs():
    student_logits = torch.randn(1, 4, 8, dtype=torch.float32)
    reference_logits = torch.randn(1, 4, 8, dtype=torch.float32)
    attention_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

    loss = dynamic_bidirectional_logits_difference(
        student_logits=student_logits,
        reference_logits=reference_logits,
        attention_mask=attention_mask,
        temperature=2.0,
        top_k=4,
        min_prob=0.0,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
