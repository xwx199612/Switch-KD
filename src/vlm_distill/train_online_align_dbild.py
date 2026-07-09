from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_schema import load_config, format_prompt, resolve_label_path
from .data_manifest import read_jsonl
from .device_utils import batch_to_device, resolve_requested_device_map, resolve_training_device_map, select_model_input_device
from .loss_switch_kd import _causal_lm_loss, full_dynamic_bidirectional_logits_difference
from .model_loading import apply_attn_implementation, resolve_model_path
from .parsing_output_parser import serialize_parsing_label
from .stage_student_training import VlmTrainingDataset
from .vlm_batching import build_vlm_data_collator, encode_vlm_training_sample, load_training_image


VISION_FREEZE_KEYWORDS = (
    "visual.blocks",
    "vision_model.encoder",
    "vision_tower",
    "patch_embed",
    "visual.patch_embed",
    "visual.rotary_pos_emb",
    "visual.window_index",
)


def _canonical_text_for_token_alignment(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _encode_answer_without_special_tokens(processor, answer: str) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(
        answer,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _decode_answer_tokens(processor, token_ids: list[int]) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(
        [int(token_id) for token_id in token_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _target_text_for_row(row: dict[str, Any]) -> str:
    if str(row.get("task") or "").strip() == "parsing":
        elements = row.get("elements")
        if not isinstance(elements, list) or not elements:
            raise ValueError(f"Parsing row {row.get('id')!r} is missing non-empty elements.")
        return serialize_parsing_label(row)
    return str(row.get("teacher_answer") or row.get("answer") or "")


def _first_mismatch_index(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _validate_teacher_token_identity(
    *,
    row: dict[str, Any],
    teacher_processor,
) -> None:
    sample_id = row.get("id")
    answer = _target_text_for_row(row)
    cached_tokens = [int(token_id) for token_id in row.get("teacher_tokens") or []]
    encoded_tokens = _encode_answer_without_special_tokens(teacher_processor, answer)

    if str(row.get("task") or "").strip() == "parsing":
        cached_tokens = encoded_tokens

    if cached_tokens != encoded_tokens:
        mismatch = _first_mismatch_index(cached_tokens, encoded_tokens)
        mismatch_index = mismatch or 0
        cached_window = cached_tokens[max(0, mismatch_index - 5):mismatch_index + 6]
        encoded_window = encoded_tokens[max(0, mismatch_index - 5):mismatch_index + 6]

        raise ValueError(
            "Teacher token identity validation failed. "
            f"id={sample_id!r}, "
            f"cached_len={len(cached_tokens)}, "
            f"encoded_len={len(encoded_tokens)}, "
            f"first_mismatch_index={mismatch}, "
            f"cached_token={cached_tokens[mismatch] if mismatch is not None and mismatch < len(cached_tokens) else None}, "
            f"encoded_token={encoded_tokens[mismatch] if mismatch is not None and mismatch < len(encoded_tokens) else None}, "
            f"cached_decoded_window={_decode_answer_tokens(teacher_processor, cached_window)!r}, "
            f"encoded_decoded_window={_decode_answer_tokens(teacher_processor, encoded_window)!r}, "
            f"target_text_preview={answer[:300]!r}"
        )


def _validate_teacher_student_tokenizer_identity(
    *,
    row: dict[str, Any],
    teacher_processor,
    student_processor,
) -> None:
    sample_id = row.get("id")
    answer = _target_text_for_row(row)
    teacher_ids = _encode_answer_without_special_tokens(teacher_processor, answer)
    student_ids = _encode_answer_without_special_tokens(student_processor, answer)

    if teacher_ids != student_ids:
        mismatch = _first_mismatch_index(teacher_ids, student_ids)
        mismatch_index = mismatch or 0
        teacher_window = teacher_ids[max(0, mismatch_index - 5):mismatch_index + 6]
        student_window = student_ids[max(0, mismatch_index - 5):mismatch_index + 6]

        raise ValueError(
            "Teacher/student tokenizer identity validation failed. "
            "Online token-position DBiLD requires identical answer token IDs. "
            f"id={sample_id!r}, "
            f"teacher_len={len(teacher_ids)}, "
            f"student_len={len(student_ids)}, "
            f"first_mismatch_index={mismatch}, "
            f"teacher_token={teacher_ids[mismatch] if mismatch is not None and mismatch < len(teacher_ids) else None}, "
            f"student_token={student_ids[mismatch] if mismatch is not None and mismatch < len(student_ids) else None}, "
            f"teacher_decoded_window={_decode_answer_tokens(teacher_processor, teacher_window)!r}, "
            f"student_decoded_window={_decode_answer_tokens(student_processor, student_window)!r}, "
            f"target_text_preview={answer[:300]!r}"
        )


def _validate_online_dbild_token_alignment_rows(
    *,
    rows: list[dict[str, Any]],
    teacher_processor,
    student_processor,
) -> None:
    checked = 0
    for row in rows:
        if row.get("task", "parsing") != "parsing":
            continue
        _validate_teacher_token_identity(
            row=row,
            teacher_processor=teacher_processor,
        )
        _validate_teacher_student_tokenizer_identity(
            row=row,
            teacher_processor=teacher_processor,
            student_processor=student_processor,
        )
        checked += 1

    print("Online DBiLD token alignment validation:")
    print(f"  checked_rows={checked}")
    print("  teacher_token_identity_ok=True")
    print("  teacher_student_tokenizer_identity_ok=True")


def _validate_student_label_answer_span(
    *,
    batch: dict[str, Any],
    student_processor,
) -> list[int]:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    target_texts = batch.get("target_text") or batch.get("target_texts")

    if target_texts is None:
        raise ValueError("Missing target_text metadata in batch; cannot validate supervised answer span.")

    supervised_counts: list[int] = []

    for batch_index, expected_answer in enumerate(target_texts):
        label_mask = labels[batch_index] != -100
        label_ids = input_ids[batch_index][label_mask].detach().cpu().tolist()
        supervised_counts.append(len(label_ids))

        decoded = _decode_answer_tokens(student_processor, label_ids)
        expected = str(expected_answer)

        if _canonical_text_for_token_alignment(decoded) != _canonical_text_for_token_alignment(expected):
            raise ValueError(
                "Student supervised answer span validation failed. "
                f"batch_index={batch_index}, "
                f"label_token_count={len(label_ids)}, "
                f"decoded_label_text={decoded[:500]!r}, "
                f"expected_target_text={expected[:500]!r}"
            )

    return supervised_counts


def _validate_answer_logits_alignment(
    *,
    teacher_answer_logits,
    student_answer_logits,
    supervised_label_counts: list[int],
    sample_ids: list[str] | None = None,
) -> None:
    teacher_len = int(teacher_answer_logits.shape[1])
    student_len = int(student_answer_logits.shape[1])

    if teacher_len != student_len:
        raise ValueError(
            "Teacher/student answer logits length mismatch. "
            f"teacher_answer_logits_len={teacher_len}, "
            f"student_answer_logits_len={student_len}, "
            f"sample_ids={sample_ids}"
        )

    if len(set(int(count) for count in supervised_label_counts)) != 1:
        raise ValueError(
            "Batch has non-uniform supervised answer lengths, but answer logits are represented as a single dense length. "
            f"supervised_label_counts={supervised_label_counts}, "
            f"sample_ids={sample_ids}"
        )

    label_len = int(supervised_label_counts[0])
    if teacher_len != label_len:
        raise ValueError(
            "Answer logits length does not match supervised answer span length. "
            f"teacher_answer_logits_len={teacher_len}, "
            f"student_answer_logits_len={student_len}, "
            f"supervised_label_len={label_len}, "
            f"sample_ids={sample_ids}"
        )


@dataclass(frozen=True)
class TrainableSummary:
    count: int
    total: int
    ratio: float
    tensor_count: int
    names: list[str]


@dataclass(frozen=True)
class LoraTargetStats:
    tensor_count: int
    param_count: int
    examples: list[str]


@dataclass(frozen=True)
class LoraTargetSummary:
    configured_targets: list[str]
    target_stats: dict[str, LoraTargetStats]
    missing_targets: list[str]
    total_lora_tensors: int
    total_lora_params: int
    total_non_lora_params: int


class OnlineAlignDataset(VlmTrainingDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        image = load_training_image(
            self.config.data.image_root,
            row["image"],
            resize_mode=self.config.training.image_resize,
        )
        prompt = format_prompt(
            self.config.distillation.prompt_template,
            query=row.get("query") or metadata.get("query"),
            task=row.get("task", "vqa"),
        )
        target = _target_text_for_row(row)
        encoded = encode_vlm_training_sample(
            self.processor,
            image=image,
            prompt=prompt,
            target=target,
            max_length=self.config.training.max_length,
            mask_prompt_labels=self.config.training.mask_prompt_labels,
            canonical_answer_span=True,
        )
        item = dict(encoded.model_inputs)
        item["prompt_token_len"] = encoded.prompt_token_len
        if not self._token_identity_debug_printed:
            student_supervised_label_count = int((item["labels"] != -100).sum().item())
            cached_teacher_tokens = row.get("teacher_tokens")
            cached_teacher_tokens_len = len(cached_teacher_tokens) if cached_teacher_tokens is not None else 0
            print("Online DBiLD first sample label debug:")
            print(f"  sample_id={row['id']}")
            print(f"  student_supervised_label_count={student_supervised_label_count}")
            print(f"  cached_teacher_tokens_len={cached_teacher_tokens_len}")
            print("  note=startup validation enforces target_text/tokenizer identity before training")
            self._token_identity_debug_printed = True
        item["sample_id"] = str(row["id"])
        item["image_path"] = str(row["image"])
        item["teacher_prompt"] = prompt
        item["target_text"] = target
        return item


class OnlineAlignCollator:
    def __init__(self, processor):
        self.base_collator = build_vlm_data_collator(processor, logits_fields=())
        self.metadata_keys = ("sample_id", "image_path", "teacher_prompt", "target_text")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        cloned = [dict(feature) for feature in features]
        metadata = {key: [feature.pop(key) for feature in cloned] for key in self.metadata_keys}
        batch = self.base_collator(cloned)
        batch.update(metadata)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online teacher-student full-logits DBiLD training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config.training.max_steps.")
    return parser.parse_args()


def _resolve_torch_dtype(name: str | None):
    import torch

    if name is None:
        return torch.bfloat16
    normalized = str(name).strip().lower()
    mapping = {
        "auto": None,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name!r}")
    return mapping[normalized]


def _build_model_kwargs(
    *,
    quantization: str,
    device_map: str | None,
    attn_implementation: str | None,
    torch_dtype_name: str | None,
    role: str,
) -> tuple[dict[str, Any], str | None]:
    import torch

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if role == "teacher":
        resolved_device_map = resolve_requested_device_map(device_map, quantization=quantization, role=role)
    else:
        resolved_device_map = resolve_training_device_map(device_map, quantization=quantization, role=role)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    apply_attn_implementation(model_kwargs, attn_implementation)

    if quantization == "none":
        dtype = _resolve_torch_dtype(torch_dtype_name) if torch_dtype_name is not None else torch.bfloat16
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
    elif quantization == "4bit":
        from transformers import BitsAndBytesConfig

        compute_dtype = _resolve_torch_dtype(torch_dtype_name)
        if compute_dtype is None:
            compute_dtype = torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization!r}")
    return model_kwargs, resolved_device_map


def _load_teacher(config):
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    from transformers import AutoProcessor

    teacher_model_path = resolve_model_path(config.teacher.model_name)
    processor = AutoProcessor.from_pretrained(
        teacher_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_kwargs, resolved_device_map = _build_model_kwargs(
        quantization=config.teacher.quantization,
        device_map=config.teacher.device_map,
        attn_implementation=config.teacher.attn_implementation,
        torch_dtype_name=config.teacher.torch_dtype,
        role="teacher",
    )
    model = AutoModelForVLM.from_pretrained(teacher_model_path, **model_kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_device = select_model_input_device(model, label="teacher")
    return model, processor, teacher_model_path, resolved_device_map, input_device


def _load_student(config):
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    from transformers import AutoProcessor

    student_model_path = resolve_model_path(config.student.model_name)
    processor = AutoProcessor.from_pretrained(
        student_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_kwargs, resolved_device_map = _build_model_kwargs(
        quantization=config.student.quantization,
        device_map=config.student.device_map,
        attn_implementation=config.student.attn_implementation,
        torch_dtype_name="bfloat16",
        role="student",
    )
    model = AutoModelForVLM.from_pretrained(student_model_path, **model_kwargs)
    return model, processor, student_model_path, resolved_device_map


def _can_require_grad(parameter) -> bool:
    return parameter.is_floating_point() or parameter.is_complex()


def freeze_student_vision_keep_merger_lm_trainable(model, *, use_lora: bool) -> TrainableSummary:
    trainable_names: list[str] = []
    trainable_tensor_count = 0

    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(keyword in lowered for keyword in VISION_FREEZE_KEYWORDS):
            parameter.requires_grad_(False)
            continue

        if use_lora:
            should_train = any(
                keyword in lowered
                for keyword in ("lora", "merger", "projector", "connector")
            )
        else:
            should_train = any(
                keyword in lowered
                for keyword in (
                    "merger",
                    "projector",
                    "connector",
                    "language_model",
                    "model.layers",
                    "lm_head",
                )
            )

        if should_train and _can_require_grad(parameter):
            parameter.requires_grad_(True)
            trainable_names.append(name)
            trainable_tensor_count += 1
        else:
            parameter.requires_grad_(False)

    trainable_count = 0
    total_count = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total_count += numel
        if parameter.requires_grad:
            trainable_count += numel

    ratio = float(trainable_count / max(total_count, 1))
    first_trainable_names = trainable_names[:20]
    print("Student trainable parameter summary:")
    print(f"trainable_param_count={trainable_count}")
    print(f"total_param_count={total_count}")
    print(f"trainable_param_ratio={ratio:.6f}")
    print(f"trainable_tensor_count={trainable_tensor_count}")
    print("first_trainable_parameter_names=", first_trainable_names)
    return TrainableSummary(
        count=int(trainable_count),
        total=int(total_count),
        ratio=ratio,
        tensor_count=int(trainable_tensor_count),
        names=first_trainable_names,
    )


def summarize_trainable_lora_targets(model, configured_targets: list[str]) -> LoraTargetSummary:
    normalized_targets = list(dict.fromkeys(configured_targets))
    target_stats: dict[str, dict[str, Any]] = {
        target: {
            "tensor_count": 0,
            "param_count": 0,
            "examples": [],
        }
        for target in normalized_targets
    }
    total_lora_tensors = 0
    total_lora_params = 0
    total_non_lora_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        numel = parameter.numel()
        lowered = name.lower()
        if "lora" in lowered:
            total_lora_tensors += 1
            total_lora_params += numel
            for target in normalized_targets:
                if target.lower() not in lowered:
                    continue
                stats = target_stats[target]
                stats["tensor_count"] += 1
                stats["param_count"] += numel
                if len(stats["examples"]) < 5:
                    stats["examples"].append(name)
        else:
            total_non_lora_params += numel

    frozen_target_stats = {
        target: LoraTargetStats(
            tensor_count=int(stats["tensor_count"]),
            param_count=int(stats["param_count"]),
            examples=list(stats["examples"]),
        )
        for target, stats in target_stats.items()
    }
    missing_targets = [
        target
        for target, stats in frozen_target_stats.items()
        if stats.tensor_count == 0
    ]
    return LoraTargetSummary(
        configured_targets=normalized_targets,
        target_stats=frozen_target_stats,
        missing_targets=missing_targets,
        total_lora_tensors=int(total_lora_tensors),
        total_lora_params=int(total_lora_params),
        total_non_lora_params=int(total_non_lora_params),
    )


def _print_lora_target_summary(summary: LoraTargetSummary) -> None:
    print("Configured LoRA target modules:")
    if summary.configured_targets:
        for target in summary.configured_targets:
            print(f"  - {target}")
    else:
        print("  - none")

    print("Detected trainable LoRA target modules:")
    if summary.configured_targets:
        for target in summary.configured_targets:
            stats = summary.target_stats[target]
            print(
                f"  - {target}: "
                f"trainable_param_tensors={stats.tensor_count}, "
                f"trainable_params={stats.param_count}"
            )
            if stats.examples:
                print(f"    examples={stats.examples}")
    else:
        print("  - none")

    print("Missing configured LoRA target modules:")
    if summary.missing_targets:
        for target in summary.missing_targets:
            print(f"  - {target}")
    else:
        print("  - none")

    print(f"Total trainable LoRA tensors: {summary.total_lora_tensors}")
    print(f"Total trainable LoRA params: {summary.total_lora_params}")
    print(f"Total trainable non-LoRA params: {summary.total_non_lora_params}")


def _autocast_context(mixed_precision: str):
    import torch

    if not torch.cuda.is_available() or mixed_precision == "no":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision!r}")


def _build_teacher_inputs(batch, teacher_processor, config):
    import torch

    if len(batch["target_text"]) != 1:
        raise ValueError("Online align training currently supports only batch_size == 1.")
    image = load_training_image(
        config.data.image_root,
        batch["image_path"][0],
        resize_mode=config.training.image_resize,
    )
    encoded = encode_vlm_training_sample(
        teacher_processor,
        image=image,
        prompt=batch["teacher_prompt"][0],
        target=batch["target_text"][0],
        max_length=config.training.max_length,
        mask_prompt_labels=True,
        canonical_answer_span=True,
    )
    teacher_labels = encoded.model_inputs.get("labels")
    if teacher_labels is None:
        raise ValueError("Teacher sample encoding did not produce labels for alignment.")
    if not torch.is_tensor(teacher_labels) or teacher_labels.ndim != 1:
        raise ValueError(
            "Teacher labels must be a rank-1 tensor before batching, "
            f"got {type(teacher_labels)!r} with shape {getattr(teacher_labels, 'shape', None)}."
        )

    teacher_inputs = {
        key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim >= 1 else value
        for key, value in encoded.model_inputs.items()
        if key != "labels"
    }
    return teacher_inputs, teacher_labels.unsqueeze(0)


def align_logits_to_supervised_positions(teacher_logits, student_logits, teacher_labels, student_labels):
    import torch

    if teacher_logits.ndim != 3 or student_logits.ndim != 3:
        raise ValueError("teacher_logits and student_logits must have shape [batch, seq, vocab].")
    if teacher_labels.ndim != 2 or student_labels.ndim != 2:
        raise ValueError("teacher_labels and student_labels must have shape [batch, seq].")
    if (
        teacher_logits.shape[0] != student_logits.shape[0]
        or teacher_logits.shape[0] != teacher_labels.shape[0]
        or student_logits.shape[0] != student_labels.shape[0]
    ):
        raise ValueError(
            "Batch size mismatch among teacher_logits, student_logits, teacher_labels, and student_labels: "
            f"{tuple(teacher_logits.shape)}, {tuple(student_logits.shape)}, "
            f"{tuple(teacher_labels.shape)}, {tuple(student_labels.shape)}."
        )
    if teacher_logits.shape[0] != 1:
        raise ValueError(
            "align_logits_to_supervised_positions currently requires batch_size == 1 "
            "for strict causal shifted teacher/student answer-logit alignment."
        )

    teacher_vocab = int(teacher_logits.shape[-1])
    student_vocab = int(student_logits.shape[-1])
    shared_vocab = min(teacher_vocab, student_vocab)
    vocab_prefix_alignment_used = teacher_vocab != student_vocab
    if vocab_prefix_alignment_used:
        print(
            "Online DBiLD vocab mismatch detected; using shared vocab prefix for alignment: "
            f"teacher_vocab={teacher_vocab}, student_vocab={student_vocab}, shared_vocab={shared_vocab}"
        )
    teacher_logits_for_align = teacher_logits[..., :shared_vocab]
    student_logits_for_align = student_logits[..., :shared_vocab]

    teacher_shift_logits = teacher_logits_for_align[:, :-1, :]
    teacher_shift_labels = teacher_labels[:, 1:]
    student_shift_logits = student_logits_for_align[:, :-1, :]
    student_shift_labels = student_labels[:, 1:]

    teacher_mask = teacher_shift_labels != -100
    student_mask = student_shift_labels != -100
    teacher_count = int(teacher_mask[0].sum().item())
    student_count = int(student_mask[0].sum().item())

    if teacher_count <= 0 or student_count <= 0:
        raise ValueError(
            "No supervised answer tokens found after causal shifted alignment: "
            f"teacher_count={teacher_count}, student_count={student_count}."
        )
    if teacher_count != student_count:
        raise ValueError(
            "Teacher/student supervised token count mismatch during causal shifted alignment: "
            f"teacher_count={teacher_count}, student_count={student_count}."
        )

    teacher_supervised_ids = teacher_shift_labels[teacher_mask]
    student_supervised_ids = student_shift_labels[student_mask]
    if int(teacher_supervised_ids.max().item()) >= teacher_vocab:
        raise ValueError("Teacher supervised label id exceeds teacher vocab size.")
    if int(student_supervised_ids.max().item()) >= student_vocab:
        raise ValueError("Student supervised label id exceeds student vocab size.")
    if (
        int(teacher_supervised_ids.max().item()) >= shared_vocab
        or int(student_supervised_ids.max().item()) >= shared_vocab
    ):
        print(
            "Warning: supervised answer labels include token ids outside shared_vocab. "
            "LM loss remains valid, but DBiLD compares only shared-vocab logits."
        )

    teacher_answer_logits = teacher_shift_logits[teacher_mask].view(1, teacher_count, shared_vocab)
    student_answer_logits = student_shift_logits[student_mask].view(1, student_count, shared_vocab)
    aligned_attention_mask = torch.ones(
        (1, teacher_count),
        device=student_logits.device,
        dtype=torch.float32,
    )
    return (
        teacher_answer_logits,
        student_answer_logits,
        aligned_attention_mask,
        teacher_count,
        student_count,
        shared_vocab,
        vocab_prefix_alignment_used,
    )


def _validate_rows(config) -> list[dict[str, Any]]:
    path = resolve_label_path(config.data)
    rows = read_jsonl(path, max_samples=config.data.max_samples)
    validated: list[dict[str, Any]] = []
    offline_fields = (
        "teacher_logits",
        "switch_logits",
        "teacher_logits_indices",
        "teacher_logits_values",
        "teacher_logits_token_k",
        "teacher_logits_answer_token_ids",
        "teacher_logits_source",
        "teacher_logits_aligned_to_answer",
    )
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} is not a JSON object.")
        for key in ("id", "image"):
            if row.get(key) in (None, ""):
                raise ValueError(f"{path}:{index} missing required field: {key}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if not (row.get("query") or metadata.get("query")):
            raise ValueError(f"{path}:{index} missing query and metadata.query")
        target_text = _target_text_for_row(row)
        if not target_text:
            raise ValueError(f"{path}:{index} missing required target text")
        if str(row.get("task") or "").strip() != "parsing" and row.get("teacher_tokens") is None:
            raise ValueError(f"{path}:{index} missing required teacher_tokens")
        if any(field in row for field in offline_fields):
            raise ValueError(
                f"{path}:{index} contains deprecated offline logits fields. "
                "Online DBiLD expects teacher label rows only."
            )
        validated.append(row)
    if not validated:
        raise ValueError(f"No training rows found in {path}.")
    return validated


def _maybe_enable_student_lora(config, model):
    if not config.student.use_lora:
        return model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.student.quantization in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)
    target_modules = config.student.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=config.student.lora_rank,
        lora_alpha=config.student.lora_alpha,
        lora_dropout=config.student.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _apply_student_train_setup(config, model):
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model = _maybe_enable_student_lora(config, model)
    summary = freeze_student_vision_keep_merger_lm_trainable(
        model,
        use_lora=config.student.use_lora,
    )
    model.train()
    return model, summary


def _build_optimizer(config, model):
    import torch

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(trainable_parameters, lr=config.training.learning_rate)


def _dataloader(dataset, processor, batch_size: int):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=OnlineAlignCollator(processor),
    )


def _gpu_mem_stats() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        return 0, 0
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


def _check_no_quantized_cpu_offload(
    model,
    *,
    role: str,
    quantization: str | None,
    resolved_device_map: str | None = None,
    input_device: Any | None = None,
):
    if quantization not in {"4bit", "8bit"}:
        return

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        bad_entries = {
            name: device
            for name, device in device_map.items()
            if str(device).lower() in {"cpu", "disk"}
        }
        if bad_entries:
            preview = list(bad_entries.items())[:20]
            raise RuntimeError(
                f"{role} model is quantized with bitsandbytes ({quantization}) but some modules were placed on CPU/disk "
                f"by device_map. This can trigger bitsandbytes CPU packing errors such as "
                f"'N must be divisible by block_n'. Move the quantized {role} fully onto CUDA devices, reduce model size, "
                f"free GPU memory, or use offline teacher logits. Offloaded modules preview: {preview}"
            )

    if input_device is not None and not str(input_device).lower().startswith("cuda"):
        raise RuntimeError(
            f"{role} model is quantized with bitsandbytes ({quantization}) but resolved to non-CUDA input device "
            f"{input_device!r}. This can trigger bitsandbytes CPU packing errors such as "
            f"'N must be divisible by block_n'. Resolved device_map={resolved_device_map!r}, "
            f"hf_device_map={_format_device_map_summary(device_map)}. Move the quantized {role} fully onto CUDA "
            f"devices, reduce model size, free GPU memory, or use offline teacher logits."
        )


def _format_device_map_summary(device_map: Any) -> str:
    if not device_map:
        return "None"
    items = list(device_map.items())
    preview = items[:20]
    suffix = "" if len(items) <= 20 else f" ... ({len(items)} entries total)"
    return f"{preview}{suffix}"


def run_training(config, *, max_steps_override: int | None = None) -> Path:
    import torch

    if config.training.batch_size != 1:
        raise ValueError(
            "This online full-logits DBiLD script currently requires training.batch_size == 1 "
            "for strict causal shifted teacher/student answer-logit alignment."
        )

    rows = _validate_rows(config)
    teacher_model, teacher_processor, teacher_model_path, _teacher_device_map, teacher_input_device = _load_teacher(config)
    _check_no_quantized_cpu_offload(
        teacher_model,
        role="teacher",
        quantization=config.teacher.quantization,
        resolved_device_map=_teacher_device_map,
        input_device=teacher_input_device,
    )
    student_model, student_processor, student_model_path, _student_device_map = _load_student(config)
    _validate_online_dbild_token_alignment_rows(
        rows=rows,
        teacher_processor=teacher_processor,
        student_processor=student_processor,
    )
    student_model, trainable_summary = _apply_student_train_setup(config, student_model)
    lora_target_summary = None
    if config.student.use_lora:
        configured_targets = config.student.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_target_summary = summarize_trainable_lora_targets(student_model, configured_targets)
        _print_lora_target_summary(lora_target_summary)
        if lora_target_summary.missing_targets:
            raise RuntimeError(
                "Configured LoRA target modules were not found as trainable LoRA parameters: "
                f"{lora_target_summary.missing_targets}"
            )
    student_input_device = select_model_input_device(student_model, label="student")

    dataset = OnlineAlignDataset(rows, config, student_processor)
    dataloader = _dataloader(dataset, student_processor, config.training.batch_size)
    optimizer = _build_optimizer(config, student_model)

    max_steps = max_steps_override if max_steps_override is not None else config.training.max_steps
    total_target_steps = int(max_steps) if max_steps is not None else None
    grad_accum_steps = int(config.training.gradient_accumulation_steps)
    global_step = 0
    first_batch_debug_printed = False
    token_alignment_batch_validated = False
    validate_every_batch = bool(getattr(config.training, "debug_token_alignment", False))
    last_step_log_values: dict[str, float | int | None] = {
        "lm_loss": None,
        "align_loss": None,
        "total_loss": None,
        "teacher_supervised_count": None,
        "student_supervised_count": None,
        "shared_vocab_size": None,
        "vocab_prefix_alignment_used": 0,
    }

    print("Online Align DBiLD training")
    print("offline_teacher_logits=disabled")
    print("teacher/student logits source=online forward pass during training")
    print(f"teacher_model_path={teacher_model_path}")
    print(f"student_model_path={student_model_path}")
    print(f"teacher_quantization={config.teacher.quantization}")
    print(f"student_quantization={config.student.quantization}")
    print(f"teacher_resolved_device_map={_teacher_device_map}")
    print(f"teacher_hf_device_map={_format_device_map_summary(getattr(teacher_model, 'hf_device_map', None))}")
    print(f"teacher_input_device={teacher_input_device}")
    print(f"mixed_precision={config.training.mixed_precision}")
    print("student vision frozen: true")
    print("VSD enabled: false")
    print(f"trainable_param_count={trainable_summary.count}")
    print(f"trainable_param_ratio={trainable_summary.ratio:.6f}")
    print(f"trainable_tensor_count={trainable_summary.tensor_count}")

    for epoch in range(int(config.training.epochs)):
        if total_target_steps is not None and global_step >= total_target_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(dataloader, start=1):
            if total_target_steps is not None and global_step >= total_target_steps:
                break

            teacher_inputs, teacher_labels = _build_teacher_inputs(batch, teacher_processor, config)
            teacher_inputs = batch_to_device(teacher_inputs, teacher_input_device)
            teacher_forward_inputs = {
                key: value
                for key, value in teacher_inputs.items()
                if key != "labels"
            }

            student_batch = {
                key: value
                for key, value in batch.items()
                if key not in {"sample_id", "image_path", "teacher_prompt", "target_text", "prompt_token_len"}
            }
            student_batch = batch_to_device(student_batch, student_input_device)
            labels = student_batch["labels"]
            if validate_every_batch or not token_alignment_batch_validated:
                validation_batch = dict(student_batch)
                validation_batch["target_text"] = batch.get("target_text")
                supervised_label_counts = _validate_student_label_answer_span(
                    batch=validation_batch,
                    student_processor=student_processor,
                )
                if not token_alignment_batch_validated:
                    print("Online DBiLD first-batch answer span validation:")
                    print(f"  sample_ids={batch.get('sample_id')}")
                    print(f"  supervised_label_counts={supervised_label_counts}")
                    print("  supervised_answer_span_decode_ok=True")
                    token_alignment_batch_validated = True
            else:
                supervised_label_counts = [
                    int((student_batch["labels"][i] != -100).sum().item())
                    for i in range(student_batch["labels"].shape[0])
                ]
            student_forward_inputs = {
                key: value
                for key, value in student_batch.items()
                if key != "labels"
            }

            with torch.no_grad():
                with _autocast_context(config.training.mixed_precision):
                    teacher_outputs = teacher_model(**teacher_forward_inputs)
                    teacher_logits = teacher_outputs.logits
            teacher_labels = teacher_labels.to(device=teacher_logits.device)

            with _autocast_context(config.training.mixed_precision):
                student_outputs = student_model(**student_forward_inputs)
                student_logits = student_outputs.logits
                lm_loss = _causal_lm_loss(student_logits, labels)
                (
                    aligned_teacher_logits,
                    aligned_student_logits,
                    aligned_attention_mask,
                    teacher_supervised_count,
                    student_supervised_count,
                    shared_vocab_size,
                    vocab_prefix_alignment_used,
                ) = align_logits_to_supervised_positions(teacher_logits, student_logits, teacher_labels, labels)
                dbild_device = aligned_student_logits.device
                if aligned_teacher_logits.device != dbild_device:
                    aligned_teacher_logits = aligned_teacher_logits.to(dbild_device)
                if aligned_attention_mask.device != dbild_device:
                    aligned_attention_mask = aligned_attention_mask.to(dbild_device)
                _validate_answer_logits_alignment(
                    teacher_answer_logits=aligned_teacher_logits,
                    student_answer_logits=aligned_student_logits,
                    supervised_label_counts=supervised_label_counts,
                    sample_ids=batch.get("sample_id"),
                )
                align_loss = full_dynamic_bidirectional_logits_difference(
                    reference_logits=aligned_teacher_logits,
                    target_logits=aligned_student_logits,
                    attention_mask=aligned_attention_mask,
                    temperature=config.distillation.kd_temperature,
                    top_k=config.distillation.dbild_top_k,
                    top_k_mode=config.distillation.dbild_top_k_mode,
                    kneedle_candidate_k=config.distillation.dbild_kneedle_candidate_k,
                    min_top_k=config.distillation.dbild_min_top_k,
                    max_top_k=config.distillation.dbild_max_top_k,
                    kl_mode=config.distillation.dbild_kl_mode,
                )
                # VSD is intentionally disabled because teacher and student share the same vision backbone.
                # This script targets online full-logits DBiLD L_Align reproduction, not full Switch-KD with VSD.
                total_loss = lm_loss + align_loss
                last_step_log_values = {
                    "lm_loss": float(lm_loss.detach().float().item()),
                    "align_loss": float(align_loss.detach().float().item()),
                    "total_loss": float(total_loss.detach().float().item()),
                    "teacher_supervised_count": teacher_supervised_count,
                    "student_supervised_count": student_supervised_count,
                    "shared_vocab_size": shared_vocab_size,
                    "vocab_prefix_alignment_used": int(vocab_prefix_alignment_used),
                }

            if not first_batch_debug_printed:
                print(f"teacher_logits.shape={tuple(teacher_logits.shape)}")
                print(f"student_logits.shape={tuple(student_logits.shape)}")
                print(f"teacher_labels.shape={tuple(teacher_labels.shape)}")
                print(f"student_labels.shape={tuple(labels.shape)}")
                print(f"teacher_shift_seq_len={int(teacher_logits.shape[1] - 1)}")
                print(f"student_shift_seq_len={int(student_logits.shape[1] - 1)}")
                print(f"teacher_supervised_count={teacher_supervised_count}")
                print(f"student_supervised_count={student_supervised_count}")
                print(f"aligned_teacher_logits.shape={tuple(aligned_teacher_logits.shape)}")
                print(f"aligned_student_logits.shape={tuple(aligned_student_logits.shape)}")
                print(f"aligned_attention_mask.shape={tuple(aligned_attention_mask.shape)}")
                print("answer_logits_length_ok=True")
                print("Online DBiLD tensor devices:")
                print(f"  teacher_logits_device={teacher_logits.device}")
                print(f"  student_logits_device={student_logits.device}")
                print(f"  aligned_teacher_logits_device={aligned_teacher_logits.device}")
                print(f"  aligned_student_logits_device={aligned_student_logits.device}")
                print(f"  aligned_attention_mask_device={aligned_attention_mask.device}")
                print(f"  dbild_loss_device={dbild_device}")
                first_batch_debug_printed = True

            loss_for_backward = total_loss / grad_accum_steps
            loss_for_backward.backward()

            if micro_step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_message = (
                    f"Online DBiLD step={global_step} "
                    f"lm_loss={last_step_log_values['lm_loss']:.6f} "
                    f"align_loss={last_step_log_values['align_loss']:.6f} "
                    f"total_loss={last_step_log_values['total_loss']:.6f}"
                )
                if last_step_log_values["teacher_supervised_count"] is not None:
                    step_message += (
                        f" teacher_supervised_count={last_step_log_values['teacher_supervised_count']}"
                        f" student_supervised_count={last_step_log_values['student_supervised_count']}"
                    )
                if last_step_log_values["vocab_prefix_alignment_used"]:
                    step_message += f" shared_vocab_size={last_step_log_values['shared_vocab_size']}"
                print(step_message)

                if total_target_steps is not None and global_step >= total_target_steps:
                    break

        if micro_step % grad_accum_steps != 0 and (total_target_steps is None or global_step < total_target_steps):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_message = (
                f"Online DBiLD step={global_step} "
                f"lm_loss={last_step_log_values['lm_loss']:.6f} "
                f"align_loss={last_step_log_values['align_loss']:.6f} "
                f"total_loss={last_step_log_values['total_loss']:.6f}"
            )
            if last_step_log_values["teacher_supervised_count"] is not None:
                step_message += (
                    f" teacher_supervised_count={last_step_log_values['teacher_supervised_count']}"
                    f" student_supervised_count={last_step_log_values['student_supervised_count']}"
                )
            if last_step_log_values["vocab_prefix_alignment_used"]:
                step_message += f" shared_vocab_size={last_step_log_values['shared_vocab_size']}"
            print(step_message)

    config.student.adapter_dir.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(config.student.adapter_dir)
    student_processor.save_pretrained(config.student.adapter_dir)
    print(f"OK online DBiLD training completed: optimizer_steps={global_step}")
    return config.student.adapter_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
