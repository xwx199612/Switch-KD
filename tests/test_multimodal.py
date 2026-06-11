import torch
from PIL import Image

from vlm_distill.logits_cache import align_reference_logits_to_suffix, compact_logits
from vlm_distill.multimodal import VlmDataCollator, encode_vlm_training_sample


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})()

    def __call__(self, images=None, text="", return_tensors="pt", truncation=True, max_length=128):
        del images, truncation, max_length
        token_count = max(1, len(text.split()))
        input_ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(1, 3, 4, 4),
        }


def test_encode_vlm_training_sample_masks_prompt_and_image_prefix():
    image = Image.new("RGB", (8, 8))
    encoded = encode_vlm_training_sample(
        _FakeProcessor(),
        image=image,
        prompt="Question: cup?",
        target="a cup",
        max_length=64,
    )
    assert encoded.prompt_token_len == 2
    assert encoded.model_inputs["labels"][0].item() == -100
    assert encoded.model_inputs["labels"][-1].item() != -100
    assert "pixel_values" in encoded.model_inputs


def test_collator_keeps_logits_metadata():
    collator = VlmDataCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, -100, 3]),
                "pixel_values": torch.zeros(3, 4, 4),
                "prompt_token_len": 2,
                "teacher_logits": {"indices": [], "values": [], "shape": [1, 2, 4], "vocab_size": 4},
                "teacher_logits_prompt_len": 2,
                "teacher_logits_vocab_size": 4,
            }
        ]
    )
    assert batch["teacher_logits"]["vocab_size"] == 4
    assert batch["teacher_logits_prompt_len"] == 2


def test_align_reference_logits_to_suffix_places_answer_region():
    reference = torch.randn(1, 5, 4)
    aligned = align_reference_logits_to_suffix(
        reference,
        target_shape=(1, 6, 4),
        reference_prompt_len=2,
        student_prompt_len=3,
        dtype=torch.float32,
    )
    assert aligned.shape == (1, 6, 4)
    assert torch.isfinite(aligned[:, 3:, :]).all()
