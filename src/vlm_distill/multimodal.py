from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class EncodedVlmSample:
    model_inputs: dict[str, Any]
    prompt_token_len: int


def load_training_image(image_root: Path, image_path: str) -> Image.Image:
    path = image_root / image_path
    return Image.open(path).convert("RGB")


def encode_vlm_training_sample(
    processor,
    *,
    image: Image.Image,
    prompt: str,
    target: str,
    max_length: int,
    mask_prompt_labels: bool = True,
) -> EncodedVlmSample:
    """Encode one image+prompt+target sample for causal VLM fine-tuning."""
    prompt_text = prompt.strip()
    target_text = target.strip()
    full_text = f"{prompt_text} {target_text}".strip()

    common_kwargs = {"return_tensors": "pt", "truncation": True, "max_length": max_length}
    full_inputs = _processor_call(processor, image=image, text=full_text, **common_kwargs)
    prompt_inputs = _processor_call(processor, image=image, text=prompt_text, **common_kwargs)

    prompt_token_len = int(prompt_inputs["input_ids"].shape[1])
    model_inputs = {key: value.squeeze(0) for key, value in full_inputs.items()}
    labels = model_inputs["input_ids"].clone()
    if mask_prompt_labels:
        labels[:prompt_token_len] = -100
    model_inputs["labels"] = labels
    return EncodedVlmSample(model_inputs=model_inputs, prompt_token_len=prompt_token_len)


def build_vlm_data_collator(processor) -> "VlmDataCollator":
    pad_token_id = _resolve_pad_token_id(processor)
    return VlmDataCollator(pad_token_id=pad_token_id)


class VlmDataCollator:
    """Pad multimodal features; keep cached logits as per-sample payloads."""

    _LOGITS_FIELDS = ("teacher_logits", "switch_logits")
    _SKIP_KEYS = frozenset({"prompt_token_len", "image", "id", "question"})

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        logits_payload = {field: [feature.pop(field, None) for feature in features] for field in self._LOGITS_FIELDS}
        prompt_token_lens = [int(feature.pop("prompt_token_len", 0)) for feature in features]
        metadata: dict[str, Any] = {}
        for feature in features:
            for key in list(feature.keys()):
                if key.endswith("_prompt_len") or key.endswith("_vocab_size"):
                    metadata.setdefault(key, feature.pop(key))

        tensor_keys = sorted(
            {
                key
                for feature in features
                for key, value in feature.items()
                if key not in self._SKIP_KEYS and torch.is_tensor(value)
            }
        )
        batch: dict[str, Any] = {}
        for key in tensor_keys:
            values = [feature[key] for feature in features]
            if key == "labels":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=-100,
                )
                continue
            if key in {"input_ids", "attention_mask"}:
                padding_value = self.pad_token_id if key == "input_ids" else 0
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=padding_value,
                )
                continue
            if all(value.shape == values[0].shape for value in values):
                batch[key] = torch.stack(values, dim=0)
                continue
            raise ValueError(
                f"Cannot batch field '{key}' with variable tensor shapes. "
                "Use batch_size=1 or ensure images are resized to the same resolution."
            )

        for field, values in logits_payload.items():
            if any(value is not None for value in values):
                batch[field] = values[0] if len(values) == 1 else values

        batch["prompt_token_len"] = prompt_token_lens[0] if len(prompt_token_lens) == 1 else prompt_token_lens
        batch.update(metadata)
        return batch


def build_supervision_mask(labels):
    import torch

    return (labels != -100).to(dtype=torch.float32)


def _processor_call(processor, *, image: Image.Image, text: str, **kwargs):
    try:
        return processor(images=image, text=text, **kwargs)
    except TypeError:
        return processor(text=text, images=image, **kwargs)


def _resolve_pad_token_id(processor) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    return int(pad_token_id or 0)
