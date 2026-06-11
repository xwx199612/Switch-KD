import pytest
import torch

from vlm_distill.logits_cache_utils import compact_logits, is_compact_logits, materialize_cached_logits
from vlm_distill.stage_visual_switch_logits import _get_nested_attr


class _Leaf:
    value = "module"


class _EmbeddingGetter:
    def __init__(self):
        self.embed = _Leaf()

    def get_input_embeddings(self):
        return self.embed


class _Model:
    def __init__(self):
        self.language_model = _EmbeddingGetter()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()


def test_get_nested_attr_invokes_getter_methods():
    model = _Model()
    assert _get_nested_attr(model, "get_input_embeddings") is model.language_model.embed
    assert _get_nested_attr(model, "language_model.get_input_embeddings") is model.language_model.embed


def test_compact_logits_round_trip():
    logits = torch.randn(1, 4, 8)
    cached = compact_logits(logits, max_vocab=3)
    assert is_compact_logits(cached)
    restored = materialize_cached_logits(cached, device="cpu", dtype=torch.float32, vocab_size=8)
    top_values, top_indices = torch.topk(logits, k=3, dim=-1)
    for batch in range(logits.shape[0]):
        for position in range(logits.shape[1]):
            for slot, index in enumerate(top_indices[batch, position].tolist()):
                assert restored[batch, position, index] == pytest.approx(
                    top_values[batch, position, slot].item(),
                    rel=1e-5,
                    abs=1e-5,
                )
