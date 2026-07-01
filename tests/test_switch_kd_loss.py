from types import SimpleNamespace
from unittest.mock import patch

import torch

from vlm_distill.loss_switch_kd import SwitchKDLoss, _build_candidate_union, dynamic_bidirectional_logits_difference
from vlm_distill.stage_student_training import VocabAlignment, _prepare_reference_logits


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


def test_prepare_reference_logits_keeps_compact_cache_sparse(monkeypatch):
    def _fail_materialize(*args, **kwargs):
        raise AssertionError("materialize_cached_logits should not be used for compact cached logits")

    monkeypatch.setattr("vlm_distill.stage_student_training.materialize_cached_logits", _fail_materialize)

    compact_cached = {
        "indices": [[[0, 4, 7], [1, 6, 8], [2, 5, 9]]],
        "values": [[[9.0, 5.0, 1.0], [8.0, 4.0, 1.0], [7.0, 3.0, 1.0]]],
        "shape": [1, 3, 10],
        "vocab_size": 10,
        "token_k": [[2, 2, 2]],
        "entropy_weight": [[0.7, 0.8, 0.9]],
    }
    distill = SimpleNamespace(
        align_kd_logits_to_answer=True,
        skip_kd_on_vocab_mismatch=False,
    )

    compact = _prepare_reference_logits(
        cached=compact_cached,
        label="teacher",
        distill=distill,
        student_vocab_size=8,
        reference_vocab_size_meta=10,
        target_shape=(1, 5, 8),
        device=torch.device("cpu"),
        dtype=torch.float32,
        student_prompt_len=2,
        reference_prompt_len=1,
        warning_bucket=set(),
        vocab_alignment=VocabAlignment(shared_token_vocab_size=6),
    )

    assert isinstance(compact, dict)
    assert compact["is_compact"] is True
    assert compact["indices"].shape == (1, 5, 3)
    assert compact["logits"].shape == (1, 5, 3)
    assert compact["token_k"].shape == (1, 5)
    assert compact["token_weight"].shape == (1, 3)
    assert compact["vocab_size"] == 8
    assert compact["reference_prompt_len"] == 1
    assert compact["student_prompt_len"] == 2
    assert compact["token_k"][0, 0].item() == 0
    assert compact["token_k"][0, 1].item() == 0
    assert compact["token_k"][0, 2].item() == 1
    assert compact["token_k"][0, 3].item() == 2
    assert compact["indices"][0, 2, 2].item() == 5


def test_dynamic_bidirectional_logits_difference_compact_matches_manual_union_kl():
    import torch.nn.functional as F

    student_logits = torch.tensor([[[9.0, 8.0, -4.0, 3.0, 2.0, -5.0]]], dtype=torch.float32)
    reference_logits = {
        "indices": torch.tensor([[[3, 4]]], dtype=torch.long),
        "values": torch.tensor([[[7.0, 6.0]]], dtype=torch.float32),
        "logits": torch.tensor([[[7.0, 6.0]]], dtype=torch.float32),
        "token_k": torch.tensor([[2]], dtype=torch.long),
        "shape": (1, 1, 6),
        "vocab_size": 6,
        "is_compact": True,
    }
    attention_mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss = dynamic_bidirectional_logits_difference(
        student_logits=student_logits,
        reference_logits=reference_logits,
        attention_mask=attention_mask,
        temperature=1.0,
        top_k=2,
        min_prob=0.0,
    )

    student_top_indices = torch.topk(student_logits, k=2, dim=-1).indices
    candidate_indices, candidate_active, _ = _build_candidate_union(
        student_top_indices,
        reference_logits["indices"],
        reference_active=torch.tensor([[[True, True]]]),
    )
    candidate_list = candidate_indices[0, 0, candidate_active[0, 0]].tolist()

    assert set(candidate_list) == {0, 1, 3, 4}

    student_candidate_logits = torch.gather(student_logits, dim=-1, index=candidate_indices)
    reference_candidate_logits = torch.tensor([[[-24.0, -24.0, 7.0, 6.0]]], dtype=torch.float32)

    student_log_probs = F.log_softmax(student_candidate_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_candidate_logits, dim=-1)
    student_probs = student_log_probs.exp()
    reference_probs = reference_log_probs.exp()
    forward_kl = (reference_probs * (reference_log_probs - student_log_probs)).sum(dim=-1)
    reverse_kl = (student_probs * (student_log_probs - reference_log_probs)).sum(dim=-1)
    expected = 0.5 * (forward_kl + reverse_kl)

    assert torch.allclose(loss, expected.squeeze(0).squeeze(0), atol=1e-5)
    assert torch.isfinite(loss)


def test_switch_kd_loss_uses_raw_component_losses_without_balancing():
    student_logits = torch.tensor(
        [[[0.0, 5.0, -1.0], [0.0, -1.0, 5.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1]], dtype=torch.long)

    loss_fn = SwitchKDLoss(
        lm_weight=1.0,
        dbild_weight=1.0,
        vsd_weight=1.0,
    )

    original_dbild = torch.tensor(10000.0)
    original_vsd = torch.tensor(8000.0)

    def _fake_dbild(*args, **kwargs):
        del args, kwargs
        return original_dbild.clone()

    def _fake_vsd(*args, **kwargs):
        del args, kwargs
        return original_vsd.clone()

    with patch("vlm_distill.loss_switch_kd.dynamic_bidirectional_logits_difference", side_effect=_fake_dbild):
        with patch("vlm_distill.loss_switch_kd.visual_switch_divergence", side_effect=_fake_vsd):
            output = loss_fn(
                student_logits=student_logits,
                labels=labels,
                teacher_logits=torch.zeros_like(student_logits),
                switch_logits=torch.zeros_like(student_logits),
            )

    expected_lm = output.lm_loss.detach().float()
    assert torch.isclose(output.lm_loss.detach().float(), expected_lm, atol=1e-6)
    assert torch.isclose(output.dbild_loss.detach().float(), original_dbild, atol=1e-6)
    assert torch.isclose(output.vsd_loss.detach().float(), original_vsd, atol=1e-6)
    assert torch.isclose(output.loss.detach().float(), expected_lm + original_dbild + original_vsd, atol=1e-4)


def test_switch_kd_loss_keeps_missing_reference_losses_zero():
    student_logits = torch.randn(1, 3, 5, dtype=torch.float32)
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    loss_fn = SwitchKDLoss()

    output = loss_fn(student_logits=student_logits, labels=labels)

    assert output.dbild_loss.item() == 0.0
    assert output.vsd_loss.item() == 0.0


def test_switch_kd_loss_compact_reference_missing_candidate_uses_finite_floor():
    student_logits = torch.tensor([[[8.0, 7.0, -1.0, -2.0, -3.0, -4.0]]], dtype=torch.float32)
    reference_logits = {
        "indices": torch.tensor([[[3, 4]]], dtype=torch.long),
        "values": torch.tensor([[[7.0, 6.0]]], dtype=torch.float32),
        "logits": torch.tensor([[[7.0, 6.0]]], dtype=torch.float32),
        "token_k": torch.tensor([[2]], dtype=torch.long),
        "shape": (1, 1, 6),
        "vocab_size": 6,
        "is_compact": True,
    }

    loss = dynamic_bidirectional_logits_difference(
        student_logits=student_logits,
        reference_logits=reference_logits,
        attention_mask=torch.tensor([[1.0]], dtype=torch.float32),
        temperature=1.0,
        top_k=2,
        min_prob=0.0,
        inactive_logit_margin=30.0,
    )

    assert torch.isfinite(loss)
    student_top_indices = torch.topk(student_logits, k=2, dim=-1).indices
    candidate_indices, candidate_active, _ = _build_candidate_union(
        student_top_indices,
        reference_logits["indices"],
        reference_active=torch.tensor([[[True, True]]]),
    )
    candidate_list = candidate_indices[0, 0, candidate_active[0, 0]].tolist()
    assert set(candidate_list) == {0, 1, 3, 4}
