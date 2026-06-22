from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwitchKDLossOutput:
    loss: object
    lm_loss: object
    dbild_loss: object
    vsd_loss: object


class SwitchKDLoss:
    """Core Switch-KD objective: LM + DBiLD + optional VSD reference loss.

    The implementation keeps the math framework model-agnostic. VSD logits can come
    from an online visual-switch forward pass or from a precomputed cache.
    """

    def __init__(
        self,
        lm_weight: float = 1.0,
        dbild_weight: float = 0.5,
        vsd_weight: float = 0.5,
        temperature: float = 2.0,
        top_k: int = 64,
        min_prob: float = 0.0,
    ) -> None:
        self.lm_weight = lm_weight
        self.dbild_weight = dbild_weight
        self.vsd_weight = vsd_weight
        self.temperature = temperature
        self.top_k = top_k
        self.min_prob = min_prob

    def __call__(
        self,
        student_logits,
        labels,
        teacher_logits=None,
        switch_logits=None,
        attention_mask=None,
        teacher_token_weight=None,
        switch_token_weight=None,
        sample_weight=None,
    ) -> SwitchKDLossOutput:
        lm_loss = _causal_lm_loss(student_logits, labels)
        zero = student_logits.new_zeros(())

        dbild_loss = zero
        if teacher_logits is not None:
            dbild_loss = dynamic_bidirectional_logits_difference(
                student_logits=student_logits,
                reference_logits=teacher_logits,
                attention_mask=attention_mask,
                temperature=self.temperature,
                top_k=self.top_k,
                min_prob=self.min_prob,
                token_weight=teacher_token_weight,
                sample_weight=sample_weight,
            )

        vsd_loss = zero
        if switch_logits is not None:
            vsd_loss = dynamic_bidirectional_logits_difference(
                student_logits=student_logits,
                reference_logits=switch_logits,
                attention_mask=attention_mask,
                temperature=self.temperature,
                top_k=self.top_k,
                min_prob=self.min_prob,
                token_weight=switch_token_weight,
                sample_weight=sample_weight,
            )

        loss = self.lm_weight * lm_loss + self.dbild_weight * dbild_loss + self.vsd_weight * vsd_loss
        return SwitchKDLossOutput(loss=loss, lm_loss=lm_loss, dbild_loss=dbild_loss, vsd_loss=vsd_loss)


def dynamic_bidirectional_logits_difference(
    student_logits,
    reference_logits,
    attention_mask=None,
    temperature: float = 2.0,
    top_k: int = 64,
    min_prob: float = 0.0,
    token_weight=None,
    sample_weight: float | None = None,
):
    """DBiLD approximation for Switch-KD.

    It dynamically selects informative vocabulary regions from the union of student
    and reference top-k probabilities, then applies bidirectional KL supervision.
    This preserves distribution shape in both directions while avoiding full-vocab
    KD memory pressure on consumer GPUs.
    """
    import torch
    import torch.nn.functional as F

    if student_logits.shape != reference_logits.shape:
        raise ValueError(
            "student_logits and reference_logits must have the same shape. "
            f"Got {student_logits.shape} and {reference_logits.shape}."
        )

    vocab_size = student_logits.shape[-1]
    effective_top_k = min(top_k, vocab_size)
    scaled_student = student_logits / temperature
    scaled_reference = reference_logits / temperature
    student_probs = F.softmax(scaled_student, dim=-1)
    reference_probs = F.softmax(scaled_reference, dim=-1)

    student_top = student_probs.topk(effective_top_k, dim=-1).indices
    reference_top = reference_probs.topk(effective_top_k, dim=-1).indices
    informative = torch.zeros_like(student_probs, dtype=torch.bool)
    informative.scatter_(-1, student_top, True)
    informative.scatter_(-1, reference_top, True)
    if min_prob > 0:
        informative |= (student_probs > min_prob) | (reference_probs > min_prob)

    fill_value = scaled_student.new_full((), -1.0e4)
    masked_student_logits = scaled_student.masked_fill(~informative, fill_value)
    masked_reference_logits = scaled_reference.masked_fill(~informative, fill_value)

    student_region_probs = F.softmax(masked_student_logits, dim=-1).clamp_min(1e-12)
    reference_region_probs = F.softmax(masked_reference_logits, dim=-1).clamp_min(1e-12)
    student_region_probs = student_region_probs / student_region_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    reference_region_probs = reference_region_probs / reference_region_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    student_log_probs = student_region_probs.log()
    reference_log_probs = reference_region_probs.log()

    forward_kl = (reference_region_probs * (reference_log_probs - student_log_probs)).sum(dim=-1)
    reverse_kl = (student_region_probs * (student_log_probs - reference_log_probs)).sum(dim=-1)
    token_loss = 0.5 * (forward_kl + reverse_kl) * (temperature**2)

    if token_weight is not None:
        token_loss = token_loss * token_weight.to(token_loss.dtype)
    if attention_mask is not None:
        token_loss = token_loss * attention_mask.to(token_loss.dtype)
        normalizer = attention_mask.to(token_loss.dtype)
        if token_weight is not None:
            normalizer = normalizer * token_weight.to(token_loss.dtype)
        loss = token_loss.sum() / normalizer.sum().clamp_min(1.0)
    else:
        loss = token_loss.mean()

    if sample_weight is not None:
        loss = loss * float(sample_weight)
    return loss


def _causal_lm_loss(logits, labels):
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
