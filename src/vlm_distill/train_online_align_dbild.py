from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import time
from typing import Any

from .config_schema import load_config, format_prompt, resolve_label_path
from .data_manifest import read_jsonl
from .device_utils import batch_to_device, resolve_requested_device_map, resolve_training_device_map, select_model_input_device
from .loss_switch_kd import full_dynamic_bidirectional_logits_difference
from .model_loading import apply_attn_implementation, resolve_model_path
from .mixed_precision import build_mixed_precision_quantization_config
from .student_trainability import (
    QWEN3_VL_PROJECTOR_PATH,
    QWEN3_VL_ATTENTION_TARGETS,
    QWEN3_VL_MLP_TARGETS,
    parameter_matches_module_path,
    summarize_trainable_groups,
    dequantize_trainable_projector,
    full_projector_modules_to_save_path,
    prepare_projector_for_lora,
    build_a2_lora_scope,
    resolve_a2_lora_targets,
    resolve_language_model_lora_targets,
    validate_a2_projector_lora_contract,
    validate_a3_attn_mlp_full_projector_contract,
    validate_a0_attention_lora_contract,
    validate_language_model_lora_scope,
    validate_projector_trainable_parameters,
    validate_projector_path,
    validate_mixed_precision_merger,
    merger_base_checksum,
    merger_dtype_map,
)
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


def _answer_logits_request_from_labels(labels, *, label_name: str):
    """Return the suffix request needed to cover the causal answer logits.

    Qwen's integer ``logits_to_keep`` form requests the final N logits.  The
    requested suffix therefore starts at the logit immediately before the
    first supervised label and may include masked logits after the answer.
    """
    import torch

    if not torch.is_tensor(labels) or labels.ndim != 2:
        raise ValueError(f"{label_name} must have shape [batch, sequence], got {getattr(labels, 'shape', None)}")
    if labels.shape[0] != 1:
        raise ValueError(
            "Online Align DBiLD requires batch_size == 1 for exact answer-position logits; "
            f"{label_name}.shape={tuple(labels.shape)}"
        )

    supervised_mask = labels.ne(-100)
    supervised_positions = supervised_mask[0].nonzero(as_tuple=False).flatten()
    if supervised_positions.numel() == 0:
        raise ValueError(f"{label_name} contains no supervised answer tokens.")

    expected = torch.arange(
        int(supervised_positions[0]),
        int(supervised_positions[-1]) + 1,
        device=supervised_positions.device,
    )
    if not torch.equal(supervised_positions, expected):
        raise ValueError(
            f"{label_name} supervised answer span must be contiguous; "
            f"positions={supervised_positions.detach().cpu().tolist()}"
        )

    first_supervised = int(supervised_positions[0].item())
    last_supervised = int(supervised_positions[-1].item())
    if first_supervised <= 0:
        raise ValueError(
            f"{label_name} first supervised answer position must be > 0 for causal alignment; "
            f"first_supervised={first_supervised}"
        )

    answer_length = int(supervised_positions.numel())
    sequence_length = int(labels.shape[1])
    first_required_logit_position = first_supervised - 1
    logits_to_keep_count = int(sequence_length - first_required_logit_position)
    trailing_logit_count = int(logits_to_keep_count - answer_length)
    if first_required_logit_position < 0 or logits_to_keep_count <= 0:
        raise ValueError(
            f"{label_name} causal shifted logit range is invalid; "
            f"first_required_logit_position={first_required_logit_position}, "
            f"sequence_length={sequence_length}"
        )
    if logits_to_keep_count < answer_length or trailing_logit_count < 0:
        raise ValueError(
            f"{label_name} causal shifted logit range does not cover the supervised span; "
            f"logits_to_keep_count={logits_to_keep_count}, answer_length={answer_length}"
        )

    answer_labels = labels[:, first_supervised:last_supervised + 1]
    return logits_to_keep_count, answer_length, answer_labels, trailing_logit_count


def _answer_only_lm_loss(answer_logits, answer_labels):
    """Causal LM loss over already-extracted answer-position logits."""
    import torch.nn.functional as F

    if answer_logits.ndim != 3 or answer_labels.ndim != 2:
        raise ValueError("answer_logits must be [batch, answer_len, vocab] and answer_labels [batch, answer_len].")
    if tuple(answer_logits.shape[:2]) != tuple(answer_labels.shape):
        raise ValueError(
            "Answer logits/labels shape mismatch: "
            f"logits={tuple(answer_logits.shape)} labels={tuple(answer_labels.shape)}"
        )
    return F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_labels.reshape(-1),
        ignore_index=-100,
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
            runtime_target_token_count = len(
                _encode_answer_without_special_tokens(self.processor, target)
            )
            print("Online DBiLD first sample label debug:")
            print(f"  sample_id={row['id']}")
            print(f"  student_supervised_label_count={student_supervised_label_count}")
            print(f"  runtime_target_token_count={runtime_target_token_count}")
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
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run exactly one real training step and save an isolated smoke adapter.",
    )
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
    mixed_merger = bool(
        getattr(config.student, "train_multimodal_projector", False)
        or getattr(config.student, "use_projector_lora", False)
    )
    allow_fallback = bool(getattr(config.student, "allow_dequantized_projector_fallback", False))
    if mixed_merger and config.student.quantization in {"4bit", "8bit"}:
        model_kwargs, resolved_device_map = _build_model_kwargs(
            quantization=config.student.quantization,
            device_map=config.student.device_map,
            attn_implementation=config.student.attn_implementation,
            torch_dtype_name="bfloat16",
            role="student",
        )
        try:
            model_kwargs["quantization_config"] = build_mixed_precision_quantization_config(
                quantization=config.student.quantization,
                excluded_module_paths=[
                    "model.visual.merger.linear_fc1",
                    "model.visual.merger.linear_fc2",
                ],
            )
        except RuntimeError:
            if not allow_fallback:
                raise
            # Explicit compatibility mode: retain the old quantized load and let
            # the projector helper perform the documented dequantization fallback.
            model_kwargs, resolved_device_map = _build_model_kwargs(
                quantization=config.student.quantization,
                device_map=config.student.device_map,
                attn_implementation=config.student.attn_implementation,
                torch_dtype_name="bfloat16", role="student",
            )
    else:
        model_kwargs, resolved_device_map = _build_model_kwargs(
            quantization=config.student.quantization,
            device_map=config.student.device_map,
            attn_implementation=config.student.attn_implementation,
            torch_dtype_name="bfloat16",
            role="student",
        )
    model = AutoModelForVLM.from_pretrained(student_model_path, **model_kwargs)
    if mixed_merger and config.student.quantization in {"4bit", "8bit"} and "quantization_config" in model_kwargs and allow_fallback is False:
        validation = validate_mixed_precision_merger(model, config.student.multimodal_projector_path)
        model._mixed_precision_source = "load_time_exclusion"
        model._main_merger_base_checksum = merger_base_checksum(model, config.student.multimodal_projector_path)
        model._main_merger_dtype_map = merger_dtype_map(model, config.student.multimodal_projector_path)
        print(f"main_merger_base_checksum={model._main_merger_base_checksum}")
        print("mixed precision merger validation passed")
        print("training_mixed_precision_source=load_time_exclusion")
        print("main_merger_quantized_before_peft=false")
        print(f"language_model_linear4bit_count={sum(1 for _, m in model.named_modules() if type(m).__name__ == 'Linear4bit')}")
        print(f"main_merger_linear_count={validation['main_merger_linear_count']}")
        print(f"training_merger_norm_dtype_before_peft={validation['norm_dtype']}")
    setattr(model, "_allow_dequantized_projector_fallback",
            bool(getattr(config.student, "allow_dequantized_projector_fallback", False)))
    return model, processor, student_model_path, resolved_device_map


def _can_require_grad(parameter) -> bool:
    return parameter.is_floating_point() or parameter.is_complex()


def freeze_student_vision_keep_merger_lm_trainable(
    model,
    *,
    use_lora: bool,
    train_multimodal_projector: bool = False,
    use_projector_lora: bool = False,
    multimodal_projector_path: str = QWEN3_VL_PROJECTOR_PATH,
    configured_targets: list[str] | None = None,
) -> TrainableSummary:
    trainable_names: list[str] = []
    trainable_tensor_count = 0

    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(keyword in lowered for keyword in VISION_FREEZE_KEYWORDS):
            parameter.requires_grad_(False)
            continue

        lora_targets = configured_targets or list(QWEN3_VL_ATTENTION_TARGETS)
        is_attention_lora = "lora" in lowered and any(target in lowered for target in QWEN3_VL_ATTENTION_TARGETS)
        is_mlp_lora = "lora" in lowered and any(target in lowered for target in QWEN3_VL_MLP_TARGETS)
        is_projector_lora = ("lora_a" in lowered or "lora_b" in lowered) and (
            any(parameter_matches_module_path(name, f"{multimodal_projector_path}.{child}")
                for child in ("linear_fc1", "linear_fc2"))
            if use_projector_lora else parameter_matches_module_path(name, multimodal_projector_path)
        )
        is_peft_original_copy = ".original_module." in name
        should_train = (use_lora and (
            (is_attention_lora or is_mlp_lora) and any(target in lowered for target in lora_targets)
            or is_projector_lora
        )) or (
            train_multimodal_projector and not is_peft_original_copy
            and parameter_matches_module_path(name, multimodal_projector_path)
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
    groups = summarize_trainable_groups(model, multimodal_projector_path)
    print(f"Attention LoRA parameters: {groups['attention_lora']}")
    print(f"Projector LoRA parameters: {groups['projector_lora']}")
    print(f"Projector full-train parameters: {groups['projector_full_train']}")
    print(f"MLP LoRA parameters: {groups['llm_mlp_lora']}")
    print(f"Vision encoder parameters: {groups['vision_encoder']}")
    print(f"Base LM parameters: {groups['base_lm']}")
    print(f"Other parameters: {groups['other']}")
    if groups["vision_encoder"] != 0:
        raise RuntimeError("Trainability validation failed: vision encoder parameters are trainable.")
    if train_multimodal_projector and groups["projector"] == 0:
        raise RuntimeError("Trainability validation failed: configured projector has no trainable parameters.")
    if train_multimodal_projector:
        _validate_a1_trainable_contract(model, multimodal_projector_path)
    return TrainableSummary(
        count=int(trainable_count),
        total=int(total_count),
        ratio=ratio,
        tensor_count=int(trainable_tensor_count),
        names=first_trainable_names,
    )


def _validate_a1_trainable_contract(model, projector_path: str) -> dict[str, int]:
    """Validate the names and ownership of A1's trainable tensors."""
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    projector = [name for name, _ in trainable if parameter_matches_module_path(name, projector_path)]
    projector_original = [name for name in projector if ".original_module." in name]
    projector_saved = [name for name in projector if ".modules_to_save.default." in name]
    if projector_original:
        raise RuntimeError(f"Trainability validation failed: frozen original projector is trainable: {projector_original}")
    if projector_saved and len({name.split('.modules_to_save.default.', 1)[0] for name in projector_saved}) != 1:
        raise RuntimeError("Trainability validation failed: duplicate projector trainable copies detected.")
    groups = summarize_trainable_groups(model, projector_path)
    if groups["vision_encoder"] or groups["base_llm"] or groups["other"]:
        raise RuntimeError(
            "A1 trainability validation failed: "
            f"vision_encoder={groups['vision_encoder']} base_llm={groups['base_llm']} other={groups['other']}"
        )
    print("Trainable parameter representatives:")
    for label, predicate in (
        ("attention LoRA", lambda name: "lora" in name.lower() and any(x in name.lower() for x in ("q_proj", "k_proj", "v_proj", "o_proj"))),
        ("projector modules_to_save", lambda name: ".modules_to_save.default." in name and parameter_matches_module_path(name, projector_path)),
    ):
        examples = [name for name, _ in trainable if predicate(name)][:5]
        print(f"  {label}: {examples}")
    return groups


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

    # The forwards already received logits_to_keep=(supervised label position - 1),
    # so their sequence dimension is answer-only.  Do not shift or slice a full
    # sequence here: that would reintroduce the memory problem this path avoids.
    teacher_mask = teacher_labels != -100
    student_mask = student_labels != -100
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

    if int(teacher_logits_for_align.shape[1]) != teacher_count or int(student_logits_for_align.shape[1]) != student_count:
        raise ValueError(
            "Answer-only logits length does not match supervised label count: "
            f"teacher_logits_len={teacher_logits_for_align.shape[1]} teacher_count={teacher_count} "
            f"student_logits_len={student_logits_for_align.shape[1]} student_count={student_count}"
        )

    teacher_supervised_ids = teacher_labels[teacher_mask]
    student_supervised_ids = student_labels[student_mask]
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

    teacher_answer_logits = teacher_logits_for_align
    student_answer_logits = student_logits_for_align
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


def _validate_rows(config, *, path: Path | None = None) -> list[dict[str, Any]]:
    path = path or resolve_label_path(config.data)
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


def _maybe_enable_student_lora(config, model, *, dry_run: bool = False):
    if not config.student.use_lora:
        return model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.student.quantization in {"4bit", "8bit"}:
        model._allow_dequantized_projector_fallback = bool(
            getattr(config.student, "allow_dequantized_projector_fallback", False)
        )

    if config.student.quantization in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)
    projector_targets = []
    if getattr(config.student, "use_projector_lora", False):
        if config.student.train_multimodal_projector:
            raise ValueError("A1 and A2 projector modes cannot both be enabled.")
        if not config.student.use_lora:
            raise ValueError("use_projector_lora=true requires use_lora=true.")
        preparation = prepare_projector_for_lora(model, config.student.multimodal_projector_path)
        model._mixed_precision_source = preparation.get("source", "load_time_exclusion")
        projector_targets = list(preparation["projector_targets"])
        print(f"projector_lora_preparation={preparation}")
    if config.student.train_multimodal_projector:
        # prepare_model_for_kbit_training may cast floating parameters; do the
        # exact projector conversion after it and before PEFT copies it.
        conversion = dequantize_trainable_projector(
            model, config.student.multimodal_projector_path,
            validate_forward=not dry_run,
        )
        model._mixed_precision_source = conversion.get("source", "load_time_exclusion")
        print(f"projector_dequantization={conversion}")
        validate_projector_trainable_parameters(model, config.student.multimodal_projector_path)
    language_model_targets = config.student.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules = language_model_targets
    a2_scope = None
    if projector_targets:
        a2_scope = build_a2_lora_scope(
            model, language_model_targets, config.student.multimodal_projector_path
        )
        projector_targets = list(a2_scope["projector_targets"])
        target_modules = list(a2_scope["peft_target_modules"])
        print("A2 resolved LoRA target module names:")
        for target in target_modules:
            print(f"  - {target}")
    if set(language_model_targets) & set(QWEN3_VL_MLP_TARGETS):
        resolved_lm = resolve_language_model_lora_targets(model, language_model_targets)
        print(
            "A3 resolved LM LoRA targets: "
            f"attention_module_count={resolved_lm['attention_module_count']} "
            f"mlp_module_count={resolved_lm['mlp_module_count']} "
            f"total_module_count={resolved_lm['total_module_count']}"
        )
    rank_pattern = {}
    alpha_pattern = {}
    if projector_targets:
        rank = config.student.projector_lora_rank or config.student.lora_rank
        alpha = config.student.projector_lora_alpha or config.student.lora_alpha
        dropout = config.student.projector_lora_dropout
        if dropout is not None and dropout != config.student.lora_dropout:
            raise ValueError("PEFT uses one lora_dropout per adapter; projector_lora_dropout must equal lora_dropout for A2.")
        rank_pattern = {path: rank for path in projector_targets}
        alpha_pattern = {path: alpha for path in projector_targets}
    lora_kwargs = dict(
        r=config.student.lora_rank,
        lora_alpha=config.student.lora_alpha,
        lora_dropout=config.student.lora_dropout,
        target_modules=target_modules,
        modules_to_save=[config.student.multimodal_projector_path]
        if config.student.train_multimodal_projector else None,
        task_type="CAUSAL_LM",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    layers_to_transform = getattr(config.student, "lora_layers_to_transform", None)
    layers_pattern = getattr(config.student, "lora_layers_pattern", None)
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform
        lora_kwargs["layers_pattern"] = layers_pattern
    lora_config = LoraConfig(**lora_kwargs)
    wrapped = get_peft_model(model, lora_config)
    for attr in ("_main_merger_base_checksum", "_main_merger_dtype_map", "_mixed_precision_source",
                 "_allow_dequantized_projector_fallback"):
        if hasattr(model, attr):
            setattr(wrapped, attr, getattr(model, attr))
    validate_projector_trainable_parameters(wrapped, config.student.multimodal_projector_path)
    if hasattr(config.student, "lora_layers_to_transform"):
        allowed_full_projector_path = (
            full_projector_modules_to_save_path(config.student.multimodal_projector_path)
            if config.student.train_multimodal_projector and not getattr(config.student, "use_projector_lora", False)
            else None
        )
        validate_language_model_lora_scope(
            wrapped, layers_to_transform, language_model_targets,
            projector_path=config.student.multimodal_projector_path,
            allowed_projector_lora_paths=projector_targets,
            allowed_full_projector_path=allowed_full_projector_path,
        )
    return wrapped


def _apply_student_train_setup(config, model, *, dry_run: bool = False):
    attention_targets = config.student.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    checkpointing_enabled = bool(config.training.gradient_checkpointing)
    use_reentrant = None
    student_is_gradient_checkpointing = None
    checkpointing_modules = []
    if checkpointing_enabled:
        use_reentrant = _enable_student_gradient_checkpointing(model)
        student_is_gradient_checkpointing = getattr(model, "is_gradient_checkpointing", None)
        checkpointing_modules = _student_gradient_checkpointing_modules(model)
        if hasattr(model, "config"):
            model.config.use_cache = False
    print(f"student_gradient_checkpointing_enabled={str(checkpointing_enabled).lower()}")
    print(
        "student_is_gradient_checkpointing="
        f"{str(bool(student_is_gradient_checkpointing)).lower()}"
    )
    print(f"student_gradient_checkpointing_use_reentrant={use_reentrant}")
    print(f"student_gradient_checkpointing_module_count={len(checkpointing_modules)}")
    print(f"student_gradient_checkpointing_module_examples={checkpointing_modules[:20]}")
    validate_projector_path(model, config.student.multimodal_projector_path)
    mixed_merger = bool(
        getattr(config.student, "train_multimodal_projector", False)
        or getattr(config.student, "use_projector_lora", False)
    ) and config.student.quantization in {"4bit", "8bit"} and not bool(
        getattr(config.student, "allow_dequantized_projector_fallback", False)
    )
    if mixed_merger:
        # prepare_model_for_kbit_training may promote floating parameters to FP32;
        # restore the explicit deployment contract without quantize->dequantize.
        from .student_trainability import get_module_by_exact_path
        import torch
        prepared_merger = get_module_by_exact_path(model, config.student.multimodal_projector_path)
        for child_name in ("linear_fc1", "linear_fc2"):
            getattr(prepared_merger, child_name).to(dtype=torch.bfloat16)
        validation = validate_mixed_precision_merger(model, config.student.multimodal_projector_path)
        print(f"training_merger_norm_dtype_after_prepare_model_for_kbit_training={validation['norm_dtype']}")
    if config.student.train_multimodal_projector:
        # Do this before PEFT modules_to_save copies the module.
        if not config.student.use_lora:
            conversion = dequantize_trainable_projector(
                model, config.student.multimodal_projector_path,
                validate_forward=not dry_run,
            )
            model._mixed_precision_source = conversion.get("source", "load_time_exclusion")
            print(f"projector_dequantization={conversion}")
    model = _maybe_enable_student_lora(config, model, dry_run=dry_run)
    summary = freeze_student_vision_keep_merger_lm_trainable(
        model,
        use_lora=config.student.use_lora,
        train_multimodal_projector=config.student.train_multimodal_projector,
        use_projector_lora=getattr(config.student, "use_projector_lora", False),
        multimodal_projector_path=config.student.multimodal_projector_path,
        configured_targets=attention_targets,
    )
    if set(attention_targets) & set(QWEN3_VL_MLP_TARGETS):
        groups = summarize_trainable_groups(model, config.student.multimodal_projector_path)
        print("Experiment mode: A3 attention + MLP LoRA + full projector")
        print(f"Attention LoRA: targets={','.join(QWEN3_VL_ATTENTION_TARGETS)} module_count=144 parameter_count={groups['attention_lora']}")
        print(f"MLP LoRA: targets={','.join(QWEN3_VL_MLP_TARGETS)} module_count=108 parameter_count={groups['llm_mlp_lora']}")
        print("Full projector: path=model.visual.merger storage=modules_to_save.default dtype=bfloat16")
        print("projector LoRA parameter count = 0")
        print(f"vision encoder trainable count = {groups['vision_encoder']}")
        print(f"base LM trainable count = {groups['base_lm']}")
        print(f"other trainable count = {groups['other']}")
    if config.student.train_multimodal_projector:
        validate_projector_trainable_parameters(model, config.student.multimodal_projector_path)
    if (set(attention_targets) & set(QWEN3_VL_MLP_TARGETS)
            and config.student.train_multimodal_projector
            and not config.student.use_projector_lora):
        validate_a3_attn_mlp_full_projector_contract(
            model, projector_path=config.student.multimodal_projector_path
        )
    elif config.student.use_projector_lora:
        # Keep the explicit mode contract in the startup path so dry-run and
        # real training exercise the same checks.
        validate_a2_projector_lora_contract(
            model, projector_path=config.student.multimodal_projector_path
        )
    elif config.student.train_multimodal_projector:
        _validate_a1_trainable_contract(model, config.student.multimodal_projector_path)
    elif config.student.use_lora:
        validate_a0_attention_lora_contract(
            model, projector_path=config.student.multimodal_projector_path
        )
    model.train()
    if mixed_merger:
        # PEFT's modules_to_save copy must obey the same dtype contract.
        merger = model
        try:
            from .student_trainability import get_module_by_exact_path
            merger = get_module_by_exact_path(model, config.student.multimodal_projector_path)
            active = getattr(getattr(merger, "modules_to_save", None), "default", None)
            if active is not None:
                active.norm.to(dtype=__import__("torch").float32)
                print(f"training_merger_norm_dtype_modules_to_save={next(active.norm.parameters()).dtype}")
        except AttributeError:
            pass
    return model, summary


def _enable_student_gradient_checkpointing(model) -> bool:
    """Enable HF checkpointing explicitly with PyTorch's non-reentrant engine."""
    method = getattr(model, "gradient_checkpointing_enable", None)
    if method is None:
        raise RuntimeError(
            "Student gradient checkpointing is enabled, but the model does not expose "
            "gradient_checkpointing_enable()."
        )
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - unusual remote-code model
        signature = None
    if signature is not None and (
        "gradient_checkpointing_kwargs" not in signature.parameters
        and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    ):
        raise RuntimeError(
            "Student model cannot receive gradient_checkpointing_kwargs; refusing to use "
            "the PyTorch reentrant default."
        )

    method(gradient_checkpointing_kwargs={"use_reentrant": False})
    enabled = getattr(model, "is_gradient_checkpointing", None)
    if enabled is False:
        raise RuntimeError("Student gradient checkpointing did not activate.")

    checkpointing_modules = _student_gradient_checkpointing_modules(model)
    if not checkpointing_modules:
        raise RuntimeError(
            "Student gradient checkpointing did not activate use_reentrant=False; "
            "no checkpoint module was found."
        )

    actual = _student_gradient_checkpointing_use_reentrant(model)
    if actual is not False:
        raise RuntimeError(
            "Student gradient checkpointing did not activate use_reentrant=False; "
            f"observed {actual!r}."
        )
    return actual


def _student_gradient_checkpointing_use_reentrant(model) -> bool | None:
    observed = []

    for module_name, module in model.named_modules():
        checkpoint_func = getattr(module, "_gradient_checkpointing_func", None)
        keywords = getattr(checkpoint_func, "keywords", None)

        if isinstance(keywords, dict) and "use_reentrant" in keywords:
            observed.append(
                (module_name or "<root>", bool(keywords["use_reentrant"]))
            )

    if not observed:
        config = getattr(model, "config", None)
        configured = getattr(config, "gradient_checkpointing_kwargs", None)
        if isinstance(configured, dict) and "use_reentrant" in configured:
            return bool(configured["use_reentrant"])
        return None

    values = {value for _, value in observed}
    if len(values) > 1:
        raise RuntimeError(
            "Student gradient checkpoint modules disagree about use_reentrant: "
            f"{observed[:20]}"
        )

    return observed[0][1]


def _student_gradient_checkpointing_modules(model):
    observed = []

    for module_name, module in model.named_modules():
        checkpoint_func = getattr(module, "_gradient_checkpointing_func", None)
        keywords = getattr(checkpoint_func, "keywords", None)

        if isinstance(keywords, dict) and "use_reentrant" in keywords:
            observed.append(
                {
                    "module": module_name or "<root>",
                    "use_reentrant": bool(keywords["use_reentrant"]),
                }
            )

    return observed


def _build_optimizer(config, model):
    import torch

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_items = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not parameter.is_floating_point()
    ]
    if invalid:
        raise RuntimeError(
            "Optimizer received non-floating trainable parameters: "
            f"{invalid}"
        )
    if not trainable_parameters:
        raise RuntimeError("Optimizer validation failed: no trainable parameters.")
    if len({id(parameter) for parameter in trainable_parameters}) != len(trainable_parameters):
        raise RuntimeError("Optimizer validation failed: duplicate trainable parameter object identities.")
    groups = summarize_trainable_groups(model, config.student.multimodal_projector_path)
    print("Trainable parameter groups:")
    print(f"optimizer_unique_parameter_tensors={len(trainable_parameters)}")
    print(f"optimizer_total_numel={sum(parameter.numel() for parameter in trainable_parameters)}")
    for key in ("attention_lora", "projector", "vision_encoder", "base_llm", "other"):
        print(f"  {key}={groups.get(key, 0)}")
    if groups.get("attention_lora", 0) <= 0:
        raise RuntimeError("Optimizer validation failed: no attention LoRA parameters.")
    if config.student.train_multimodal_projector and groups.get("projector", 0) <= 0:
        raise RuntimeError("Optimizer validation failed: no projector parameters.")
    if config.student.train_multimodal_projector:
        _validate_a1_trainable_contract(model, config.student.multimodal_projector_path)
    if config.student.train_multimodal_projector and (
        groups.get("vision_encoder", 0) or groups.get("base_llm", 0) or groups.get("other", 0)
    ):
        raise RuntimeError(
            "Optimizer validation failed: unexpected trainable groups: "
            f"vision_encoder={groups['vision_encoder']} base_llm={groups['base_llm']} other={groups['other']}"
        )
    print(f"attention_lora_numel={groups['attention_lora']}")
    print(f"projector_numel={groups['projector']}")
    print(f"vision_encoder_numel={groups['vision_encoder']}")
    print(f"base_llm_numel={groups['base_llm']}")
    print(f"other_numel={groups['other']}")
    print(f"optimizer_trainable_names={len(trainable_items)}")
    return torch.optim.AdamW(trainable_parameters, lr=config.training.learning_rate)


def _weighted_online_align_loss(lm_loss, align_loss, *, lm_loss_weight: float, dbild_loss_weight: float):
    return lm_loss_weight * lm_loss + dbild_loss_weight * align_loss


def _scale_partial_accumulation_gradients(model, *, grad_accum_steps: int, micro_step: int) -> None:
    """Convert a partial window's full-window-scaled gradients to a true mean."""
    if micro_step <= 0 or micro_step % grad_accum_steps == 0:
        return
    partial_scale = grad_accum_steps / (micro_step % grad_accum_steps)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(partial_scale)


def _dataloader(dataset, processor, batch_size: int):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=OnlineAlignCollator(processor),
    )


def _distributed_state() -> tuple[bool, int, int]:
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return False, 0, 1
    return True, dist.get_rank(), dist.get_world_size()


def _reduce_validation_totals(total: Any, count: int) -> tuple[float, int]:
    """Reduce loss sum/count, making uneven validation shards mathematically correct."""
    import torch
    import torch.distributed as dist
    active, _rank, _world = _distributed_state()
    device = torch.device("cuda", torch.cuda.current_device()) if active and dist.get_backend() == "nccl" else torch.device("cpu")
    values = torch.tensor([float(total), float(count)], dtype=torch.float64, device=device)
    if active:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())


def _broadcast_early_stop(flag: bool) -> bool:
    import torch
    import torch.distributed as dist
    active, rank, _world = _distributed_state()
    if not active:
        return bool(flag)
    value = torch.tensor([int(flag) if rank == 0 else 0], dtype=torch.int64)
    dist.broadcast(value, src=0)
    return bool(value.item())


def _early_stopping_update(current_val_loss: float, best_val_loss: float,
                           epochs_without_improvement: int, *, min_delta: float,
                           patience: int) -> tuple[float, int, bool, bool]:
    """Pure state transition used by the loop and unit tests."""
    improved = current_val_loss < best_val_loss - min_delta
    next_best = current_val_loss if improved else best_val_loss
    next_bad = 0 if improved else epochs_without_improvement + 1
    return next_best, next_bad, improved, next_bad >= patience


def _jsonable_config(config) -> dict[str, Any]:
    from dataclasses import asdict
    value = asdict(config)
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, list):
            return [convert(val) for val in item]
        return item
    return convert(value)


def _save_best_checkpoint(model, processor, optimizer, scheduler, *, checkpoint_dir: Path,
                           epoch: int, global_step: int, best_val_loss: float,
                           epochs_without_improvement: int, config) -> None:
    import torch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "resolved_config": _jsonable_config(config),
    }, checkpoint_dir / "training_state.pt")


def _restore_best_checkpoint(model, optimizer, scheduler, checkpoint_dir: Path) -> dict[str, Any]:
    import torch
    state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def _validate_epoch(*, rows, config, student_model, teacher_model, student_processor,
                    teacher_processor, student_input_device, teacher_input_device,
                    batch_size: int, epoch: int, global_step: int,
                    best_val_loss: float, epochs_without_improvement: int) -> dict[str, Any]:
    """Evaluate the exact online Align DBiLD objective without touching gradients."""
    import torch
    dataset = OnlineAlignDataset(rows, config, student_processor)
    active, _rank, world_size = _distributed_state()
    sampler = None
    if active:
        import torch.utils.data.distributed
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=_rank, shuffle=False, drop_last=False
        )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False if sampler is not None else False,
        collate_fn=OnlineAlignCollator(student_processor),
    )
    teacher_was_training = teacher_model.training
    student_model.eval()
    teacher_model.eval()
    sums = {key: 0.0 for key in ("lm", "dbild", "total")}
    count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                teacher_inputs, teacher_labels = _build_teacher_inputs(batch, teacher_processor, config)
                teacher_inputs = batch_to_device(teacher_inputs, teacher_input_device)
                teacher_labels = teacher_labels.to(device=teacher_input_device)
                student_batch = batch_to_device(
                    {key: value for key, value in batch.items()
                     if key not in {"sample_id", "image_path", "teacher_prompt", "target_text", "prompt_token_len"}},
                    student_input_device,
                )
                labels = student_batch["labels"]
                t_keep, t_len, t_labels, _ = _answer_logits_request_from_labels(teacher_labels, label_name="teacher_labels")
                s_keep, s_len, s_labels, _ = _answer_logits_request_from_labels(labels, label_name="student_labels")
                teacher_inputs_no_labels = {key: value for key, value in teacher_inputs.items() if key != "labels"}
                student_inputs = {key: value for key, value in student_batch.items() if key != "labels"}
                with _autocast_context(config.training.mixed_precision):
                    teacher_logits = teacher_model(**teacher_inputs_no_labels, logits_to_keep=t_keep).logits[:, :t_len, :]
                    student_logits = student_model(**student_inputs, logits_to_keep=s_keep).logits[:, :s_len, :]
                    lm_loss = _answer_only_lm_loss(student_logits, s_labels)
                    (teacher_logits, student_logits, mask, *_rest) = align_logits_to_supervised_positions(
                        teacher_logits, student_logits, t_labels, s_labels
                    )
                    dbild_loss = full_dynamic_bidirectional_logits_difference(
                        reference_logits=teacher_logits, target_logits=student_logits,
                        attention_mask=mask, temperature=config.distillation.kd_temperature,
                        top_k=config.distillation.dbild_top_k, top_k_mode=config.distillation.dbild_top_k_mode,
                        kneedle_candidate_k=config.distillation.dbild_kneedle_candidate_k,
                        min_top_k=config.distillation.dbild_min_top_k, max_top_k=config.distillation.dbild_max_top_k,
                        kl_mode=config.distillation.dbild_kl_mode,
                    )
                    total_loss = _weighted_online_align_loss(
                        lm_loss, dbild_loss, lm_loss_weight=config.distillation.lm_loss_weight,
                        dbild_loss_weight=config.distillation.dbild_loss_weight,
                    )
                sums["lm"] += float(lm_loss.detach().float().item())
                sums["dbild"] += float(dbild_loss.detach().float().item())
                sums["total"] += float(total_loss.detach().float().item())
                count += 1
    finally:
        student_model.train()
        if teacher_was_training:
            teacher_model.train()
    reduced = {key: _reduce_validation_totals(value, count) for key, value in sums.items()}
    denominator = reduced["total"][1]
    if denominator <= 0:
        raise ValueError("Validation manifest produced no batches.")
    values = {f"val_{key}_loss": reduced[key][0] / denominator for key in sums}
    next_best, next_bad, improved, _should_stop = _early_stopping_update(
        values["val_total_loss"], best_val_loss, epochs_without_improvement,
        min_delta=config.training.early_stopping_min_delta,
        patience=config.training.early_stopping_patience,
    )
    return {
        "epoch": epoch, "global_step": global_step, **values,
        "best_val_loss": next_best, "epochs_without_improvement": next_bad,
        "is_best": improved, "early_stopped": False,
    }


def _gpu_mem_stats() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        return 0, 0
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


def _dtype_summary(model, predicate) -> str:
    dtypes = sorted({str(parameter.dtype).replace("torch.", "")
                     for name, parameter in model.named_parameters()
                     if parameter.requires_grad and predicate(name)})
    return ",".join(dtypes) if dtypes else "none"


def _print_dry_run_summary(config, model) -> None:
    groups = summarize_trainable_groups(model, config.student.multimodal_projector_path)
    attention_modules = {
        name.lower().rsplit(".lora_", 1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in name.lower()
        and any(f".{target}." in name.lower() for target in QWEN3_VL_ATTENTION_TARGETS)
    }
    mlp_modules = {
        name.lower().rsplit(".lora_", 1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in name.lower()
        and any(f".{target}." in name.lower() for target in QWEN3_VL_MLP_TARGETS)
    }
    projector_mode = "modules_to_save.default" if config.student.train_multimodal_projector else (
        "projector_lora" if config.student.use_projector_lora else "none"
    )
    print("Dry-run startup summary")
    print(f"attention target modules = {len(attention_modules)}")
    print(f"MLP target modules = {len(mlp_modules)}")
    print(f"total LoRA target modules = {len(attention_modules | mlp_modules)}")
    print(f"projector mode = {projector_mode}")
    print(f"projector LoRA = {groups['projector_lora']}")
    print(f"vision trainable = {groups['vision_encoder']}")
    print(f"base LM trainable = {groups['base_lm']}")
    print(f"other trainable = {groups['other']}")
    is_attention = lambda name: any(f".{target}." in name.lower() for target in QWEN3_VL_ATTENTION_TARGETS)
    is_mlp = lambda name: any(f".{target}." in name.lower() for target in QWEN3_VL_MLP_TARGETS)
    is_projector = lambda name: parameter_matches_module_path(name, config.student.multimodal_projector_path)
    print(f"attention dtype summary = {_dtype_summary(model, is_attention)}")
    print(f"MLP dtype summary = {_dtype_summary(model, is_mlp)}")
    print(f"projector dtype summary = {_dtype_summary(model, is_projector)}")
    allocated, reserved = _gpu_mem_stats()
    print(f"GPU allocated/reserved memory = {allocated}/{reserved} bytes")
    print("dry-run complete: optimizer=not_created forward=not_run backward=not_run checkpoint=not_written")


def _synchronize_for_timing(device: Any) -> None:
    import torch

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _gpu_peak_memory_allocated() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    return sum(
        int(torch.cuda.max_memory_allocated(index))
        for index in range(torch.cuda.device_count())
    )


SMOKE_ADAPTER_DIR = Path("outputs/lora_ablation/smoke/stage1_a3_one_step/adapter")


def _format_gib(byte_count: int) -> str:
    return f"{byte_count / (1024 ** 3):.3f} GiB"


def _print_gpu_memory_stage(stage: str) -> None:
    allocated, reserved = _gpu_mem_stats()
    peak = _gpu_peak_memory_allocated()
    print(
        f"GPU memory {stage}: allocated={allocated} bytes ({_format_gib(allocated)}), "
        f"reserved={reserved} bytes ({_format_gib(reserved)}), "
        f"peak={peak} bytes ({_format_gib(peak)})"
    )


def _gradient_group(name: str, projector_path: str) -> str:
    lowered = name.lower()
    if "lora_a" in lowered or "lora_b" in lowered:
        if any(target in lowered for target in QWEN3_VL_ATTENTION_TARGETS):
            return "attention_lora"
        if any(target in lowered for target in QWEN3_VL_MLP_TARGETS):
            return "mlp_lora"
        if parameter_matches_module_path(name, projector_path):
            return "projector_lora"
        return "other"
    if ".modules_to_save.default." in lowered and parameter_matches_module_path(name, projector_path):
        return "full_projector"
    if any(term in lowered for term in ("visual", "vision_tower", "vision_model", "patch_embed")):
        return "vision_encoder"
    if "model.language_model" in lowered or ".language_model." in lowered:
        return "base_lm"
    return "other"


def collect_gradient_contract(model, projector_path: str) -> dict[str, dict[str, float | int]]:
    """Collect the post-backward contract without changing parameter state."""
    groups = {
        key: {
            "parameter_count": 0,
            "tensors_with_grad": 0,
            "tensors_without_grad": 0,
            "finite_gradient_tensors": 0,
            "nonfinite_gradient_tensors": 0,
            "gradient_norm": 0.0,
        }
        for key in (
            "attention_lora", "mlp_lora", "full_projector", "projector_lora",
            "vision_encoder", "base_lm", "other",
        )
    }
    import torch

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = groups[_gradient_group(name, projector_path)]
        group["parameter_count"] += int(parameter.numel())
        if parameter.grad is None:
            group["tensors_without_grad"] += 1
            continue
        group["tensors_with_grad"] += 1
        gradient = parameter.grad.detach()
        if torch.isfinite(gradient).all().item():
            group["finite_gradient_tensors"] += 1
        else:
            group["nonfinite_gradient_tensors"] += 1
        group["gradient_norm"] += float(torch.linalg.vector_norm(gradient.float()).item() ** 2)

    for group in groups.values():
        group["gradient_norm"] = float(group["gradient_norm"] ** 0.5)
    return groups


def validate_smoke_gradient_contract(model, projector_path: str) -> dict[str, dict[str, float | int]]:
    groups = collect_gradient_contract(model, projector_path)
    for name, stats in groups.items():
        print(f"gradient_group={name} {stats}")
    for name in ("attention_lora", "mlp_lora", "full_projector"):
        stats = groups[name]
        if stats["parameter_count"] <= 0:
            raise RuntimeError(f"Smoke gradient contract failed: {name} has no trainable parameters.")
        if stats["tensors_with_grad"] <= 0:
            raise RuntimeError(f"Smoke gradient contract failed: {name} has no gradient tensors.")
        if stats["nonfinite_gradient_tensors"]:
            raise FloatingPointError(f"Smoke gradient contract failed: {name} has non-finite gradients.")
        if stats["gradient_norm"] <= 0.0:
            raise RuntimeError(f"Smoke gradient contract failed: {name} gradient norm is zero.")
    if groups["projector_lora"]["parameter_count"] != 0:
        raise RuntimeError("Smoke gradient contract failed: projector LoRA is trainable.")
    for name in ("vision_encoder", "base_lm", "other"):
        if groups[name]["parameter_count"] or groups[name]["tensors_with_grad"]:
            raise RuntimeError(f"Smoke gradient contract failed: unexpected trainable group {name}.")
    return groups


def validate_smoke_losses(lm_loss, dbild_loss, vsd_loss, total_loss) -> None:
    import torch

    values = torch.stack(tuple(value.detach().float() for value in (lm_loss, dbild_loss, vsd_loss, total_loss)))
    if not torch.isfinite(values).all().item():
        raise FloatingPointError("Smoke loss contract failed: loss is non-finite.")
    if float(total_loss.detach().float().item()) <= 0.0:
        raise FloatingPointError("Smoke loss contract failed: total_loss must be > 0.")
    if float(vsd_loss.detach().float().item()) != 0.0:
        raise RuntimeError("Smoke loss contract failed: vsd_loss must be 0.")


_FULL_PROJECTOR_TENSOR_SUFFIXES = (
    "norm.weight",
    "norm.bias",
    "linear_fc1.weight",
    "linear_fc1.bias",
    "linear_fc2.weight",
    "linear_fc2.bias",
)
_REQUIRED_FULL_PROJECTOR_TENSOR_SUFFIXES = {
    *_FULL_PROJECTOR_TENSOR_SUFFIXES,
}


def _normalize_peft_module_path(path: str) -> str:
    """Normalize PEFT's model prefixes while preserving exact module boundaries."""
    normalized = ".".join(part for part in str(path).strip(".").split(".") if part)
    while normalized.startswith("base_model.model."):
        normalized = normalized[len("base_model.model."):]
    if normalized.startswith("model."):
        normalized = normalized[len("model."):]
    return normalized


def is_saved_full_projector_key(key: str, projector_path: str) -> bool:
    """Return whether *key* is an exact, wrapper-free saved projector tensor key."""
    normalized_key = _normalize_peft_module_path(key)
    normalized_projector = _normalize_peft_module_path(projector_path)
    prefix = f"{normalized_projector}."
    if not normalized_key.startswith(prefix):
        return False
    suffix = normalized_key[len(prefix):]
    return suffix in _FULL_PROJECTOR_TENSOR_SUFFIXES


def _is_key_under_module(key: str, module_path: str) -> bool:
    return _normalize_peft_module_path(key).startswith(f"{_normalize_peft_module_path(module_path)}.")


def _validate_smoke_adapter_checkpoint(
    adapter_dir: Path,
    projector_path: str = QWEN3_VL_PROJECTOR_PATH,
    required_projector_tensors: set[str] | None = None,
) -> None:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"Smoke adapter is incomplete: expected {config_path} and {weights_path}.")
    with config_path.open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    modules_to_save = adapter_config.get("modules_to_save")
    if isinstance(modules_to_save, str):
        modules_to_save = [modules_to_save]
    if not isinstance(modules_to_save, list):
        modules_to_save = []
    normalized_projector = _normalize_peft_module_path(projector_path)
    normalized_modules_to_save = [_normalize_peft_module_path(value) for value in modules_to_save]
    if normalized_projector not in normalized_modules_to_save:
        raise RuntimeError(
            "Smoke adapter checkpoint does not declare the full projector in adapter_config modules_to_save. "
            f"adapter_config modules_to_save={modules_to_save!r}, expected path={projector_path!r}."
        )
    from safetensors.torch import load_file

    keys = list(load_file(str(weights_path), device="cpu"))
    for target in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if not any(
            target in key.split(".")
            and any(part.lower() in {"lora_a", "lora_b"} for part in key.split("."))
            for key in keys
        ):
            raise RuntimeError(f"Smoke adapter checkpoint is missing {target} LoRA keys.")
    projector_related_keys = [key for key in keys if _is_key_under_module(key, projector_path)]
    saved_projector_keys = [key for key in keys if is_saved_full_projector_key(key, projector_path)]
    saved_projector_suffixes = {
        _normalize_peft_module_path(key).split(f"{normalized_projector}.", 1)[1]
        for key in saved_projector_keys
    }
    required_projector_tensors = (
        _REQUIRED_FULL_PROJECTOR_TENSOR_SUFFIXES
        if required_projector_tensors is None
        else set(required_projector_tensors)
    )
    invalid_required_projector_tensors = sorted(
        set(required_projector_tensors) - set(_FULL_PROJECTOR_TENSOR_SUFFIXES)
    )
    if invalid_required_projector_tensors:
        raise ValueError(f"Unknown required projector tensors: {invalid_required_projector_tensors!r}")
    missing_projector_tensors = sorted(required_projector_tensors - saved_projector_suffixes)
    print(f"adapter_config modules_to_save={modules_to_save!r}")
    print(f"projector-related keys={sorted(projector_related_keys)!r}")
    if missing_projector_tensors:
        raise RuntimeError(
            "Smoke adapter checkpoint is missing full projector tensors. "
            f"adapter_config modules_to_save={modules_to_save!r}; "
            f"found projector-related keys={sorted(projector_related_keys)!r}; "
            f"missing projector tensors={missing_projector_tensors!r}."
        )
    if any(
        is_saved_full_projector_key(key, projector_path)
        and ("lora_a" in key.lower() or "lora_b" in key.lower())
        for key in keys
    ) or any(
        _is_key_under_module(key, projector_path)
        and ("lora_a" in key.lower() or "lora_b" in key.lower())
        for key in keys
    ):
        raise RuntimeError("Smoke adapter checkpoint unexpectedly contains projector LoRA keys.")
    unexpected_visual_keys = [
        key
        for key in keys
        if _normalize_peft_module_path(key).startswith("visual.")
        and not _is_key_under_module(key, projector_path)
    ]
    if unexpected_visual_keys:
        raise RuntimeError(
            "Smoke adapter checkpoint unexpectedly contains non-projector visual keys: "
            f"{sorted(unexpected_visual_keys)!r}."
        )
    print(f"Smoke adapter checkpoint validation passed: path={adapter_dir}")


def validate_adapter_checkpoint(adapter_dir: Path, config, projector_path: str | None = None) -> None:
    """Validate an adapter against the A0/A1/A2/A3 contract in *config*.

    The one-step smoke checkpoint is deliberately A3-shaped, but a normal
    ``validate-adapter`` invocation must validate the experiment selected by
    the supplied pipeline config.
    """
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"Adapter is incomplete: expected {config_path} and {weights_path}.")

    student = config.student
    projector_path = projector_path or student.multimodal_projector_path
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_lm_targets = set(student.target_modules or [])
    expected_projector_targets = (
        {f"{projector_path}.linear_fc1", f"{projector_path}.linear_fc2"}
        if student.use_projector_lora else set()
    )
    expected_targets = configured_lm_targets | expected_projector_targets
    actual_targets = set(adapter_config.get("target_modules", []))
    if actual_targets != expected_targets:
        missing = sorted(expected_targets - actual_targets)
        unexpected = sorted(actual_targets - expected_targets)
        raise RuntimeError(
            "Adapter target_modules do not match config contract: "
            f"missing={missing!r}, unexpected={unexpected!r}, "
            f"adapter={sorted(actual_targets)!r}, expected={sorted(expected_targets)!r}"
        )

    modules_to_save = adapter_config.get("modules_to_save") or []
    if isinstance(modules_to_save, str):
        modules_to_save = [modules_to_save]
    normalized_modules = {_normalize_peft_module_path(value) for value in modules_to_save}
    normalized_projector = _normalize_peft_module_path(projector_path)
    wants_full_projector = bool(student.train_multimodal_projector)
    if wants_full_projector and normalized_projector not in normalized_modules:
        raise RuntimeError(
            f"A1/A3 validation failed: modules_to_save must contain {projector_path!r}."
        )
    if not wants_full_projector and normalized_projector in normalized_modules:
        raise RuntimeError("A0/A2 validation failed: modules_to_save projector is not allowed")

    from safetensors.torch import load_file
    keys = list(load_file(str(weights_path), device="cpu"))
    projector_lora_keys = [
        key for key in keys
        if _is_key_under_module(key, projector_path)
        and ("lora_a" in key.lower() or "lora_b" in key.lower())
    ]
    if student.use_projector_lora:
        if not projector_lora_keys:
            raise RuntimeError("A2 validation failed: projector LoRA is missing")
        allowed = {f"{normalized_projector}.linear_fc1", f"{normalized_projector}.linear_fc2"}
        if any(
            not any(_normalize_peft_module_path(key).startswith(f"{path}.") for path in allowed)
            for key in projector_lora_keys
        ):
            raise RuntimeError("A2 validation failed: projector LoRA targets must be the two main merger linears")
    elif projector_lora_keys:
        mode = "A1" if wants_full_projector else "A0"
        raise RuntimeError(f"{mode} validation failed: projector LoRA is not allowed")

    if wants_full_projector:
        saved_projector_suffixes = {
            _normalize_peft_module_path(key).split(f"{normalized_projector}.", 1)[1]
            for key in keys if is_saved_full_projector_key(key, projector_path)
        }
        missing = sorted(_REQUIRED_FULL_PROJECTOR_TENSOR_SUFFIXES - saved_projector_suffixes)
        if missing:
            raise RuntimeError(f"A1/A3 validation failed: missing full projector tensors={missing!r}")
    elif normalized_modules:
        raise RuntimeError(f"A0/A2 validation failed: unexpected modules_to_save={sorted(normalized_modules)!r}")

    if not set(QWEN3_VL_MLP_TARGETS) & configured_lm_targets:
        illegal_mlp_keys = [
            key for key in keys
            if any(target in key.split(".") for target in QWEN3_VL_MLP_TARGETS)
            and ("lora_a" in key.lower() or "lora_b" in key.lower())
        ]
        if illegal_mlp_keys:
            raise RuntimeError(f"A0/A1/A2 validation failed: unexpected LM MLP LoRA keys={sorted(illegal_mlp_keys)!r}")
    if any("deepstack_merger" in key for key in keys):
        raise RuntimeError("Adapter validation failed: deepstack merger LoRA is not allowed")
    print(f"Adapter validation passed: path={adapter_dir}")


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


def run_training(
    config, *, max_steps_override: int | None = None, dry_run: bool = False,
    smoke_test: bool = False,
) -> Path | None:
    import torch

    if config.distillation.method != "online_align_dbild":
        raise ValueError(
            "The train command uses online Align DBiLD and requires "
            "distillation.method='online_align_dbild'."
        )
    if config.distillation.vsd_loss_weight != 0.0:
        raise ValueError(
            "Online Align DBiLD training does not implement a VSD forward path; "
            "distillation.vsd_loss_weight must be 0.0."
        )
    if config.training.batch_size != 1 and not smoke_test:
        raise ValueError(
            "This online full-logits DBiLD script currently requires training.batch_size == 1 "
            "for strict causal shifted teacher/student answer-logit alignment."
        )

    print("Online Align DBiLD training")
    print("offline_teacher_logits=disabled")
    print("teacher/student logits source=online forward pass during training")
    print("student vision frozen: true")
    print("VSD enabled: false")
    print(f"lm_loss_weight={config.distillation.lm_loss_weight}")
    print(f"dbild_loss_weight={config.distillation.dbild_loss_weight}")
    print(f"vsd_loss_weight={config.distillation.vsd_loss_weight}")

    if dry_run:
        print("dry_run=true")
        student_model, _student_processor, student_model_path, _student_device_map = _load_student(config)
        student_model, trainable_summary = _apply_student_train_setup(
            config, student_model, dry_run=True
        )
        _print_dry_run_summary(config, student_model)
        print(f"student_model_path={student_model_path}")
        print(f"trainable_param_count={trainable_summary.count}")
        return None

    rows = _validate_rows(config)
    if smoke_test:
        # These are deliberately local values: the production A3 config must remain unchanged.
        rows = rows[:1]
        effective_epochs = 1
        effective_batch_size = 1
        effective_grad_accum_steps = 1
        total_target_steps = 1
        adapter_dir = SMOKE_ADAPTER_DIR
        print("smoke_test=true (effective max_samples=1 epochs=1 batch_size=1 gradient_accumulation_steps=1 max_optimizer_steps=1)")
    else:
        effective_epochs = int(config.training.epochs)
        effective_batch_size = int(config.training.batch_size)
        effective_grad_accum_steps = int(config.training.gradient_accumulation_steps)
        adapter_dir = config.student.adapter_dir
        max_steps = max_steps_override if max_steps_override is not None else config.training.max_steps
        total_target_steps = int(max_steps) if max_steps is not None else None
    teacher_model, teacher_processor, teacher_model_path, _teacher_device_map, teacher_input_device = _load_teacher(config)
    _print_gpu_memory_stage("teacher_loaded")
    _check_no_quantized_cpu_offload(
        teacher_model,
        role="teacher",
        quantization=config.teacher.quantization,
        resolved_device_map=_teacher_device_map,
        input_device=teacher_input_device,
    )
    student_model, student_processor, student_model_path, _student_device_map = _load_student(config)
    print("student loaded")
    _print_gpu_memory_stage("student_loaded")
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
    dataloader = _dataloader(dataset, student_processor, effective_batch_size)
    optimizer = _build_optimizer(config, student_model)
    distributed, rank, world_size = _distributed_state()
    validation_enabled = bool(config.training.validation_enabled)
    validation_rows = None
    validation_history_path = config.student.output_dir / "validation_history.jsonl"
    best_checkpoint_dir = config.student.output_dir / "best_checkpoint"
    scheduler = None
    if validation_enabled:
        validation_rows = _validate_rows(
            config, path=config.data.validation_manifest_path,
        )
        # A constant scheduler preserves the historical learning-rate behavior while
        # making the best checkpoint restartable and explicit about scheduler state.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        if rank == 0:
            validation_history_path.parent.mkdir(parents=True, exist_ok=True)
            validation_history_path.unlink(missing_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    grad_accum_steps = effective_grad_accum_steps
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
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_requested = False

    print(f"teacher_model_path={teacher_model_path}")
    print(f"student_model_path={student_model_path}")
    print(f"teacher_quantization={config.teacher.quantization}")
    print(f"student_quantization={config.student.quantization}")
    print(f"teacher_resolved_device_map={_teacher_device_map}")
    print(f"teacher_hf_device_map={_format_device_map_summary(getattr(teacher_model, 'hf_device_map', None))}")
    print(f"teacher_input_device={teacher_input_device}")
    print(f"mixed_precision={config.training.mixed_precision}")
    print(f"trainable_param_count={trainable_summary.count}")
    print(f"trainable_param_ratio={trainable_summary.ratio:.6f}")
    print(f"trainable_tensor_count={trainable_summary.tensor_count}")

    for epoch in range(effective_epochs):
        if total_target_steps is not None and global_step >= total_target_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        for micro_step, batch in enumerate(dataloader, start=1):
            if total_target_steps is not None and global_step >= total_target_steps:
                break

            teacher_inputs, teacher_labels = _build_teacher_inputs(batch, teacher_processor, config)
            teacher_inputs = batch_to_device(teacher_inputs, teacher_input_device)
            teacher_labels = teacher_labels.to(device=teacher_input_device)
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
            (
                teacher_logits_to_keep_count,
                teacher_answer_length,
                teacher_answer_labels,
                teacher_trailing_logit_count,
            ) = _answer_logits_request_from_labels(
                teacher_labels, label_name="teacher_labels"
            )
            (
                student_logits_to_keep_count,
                student_answer_length,
                student_answer_labels,
                student_trailing_logit_count,
            ) = _answer_logits_request_from_labels(
                labels, label_name="student_labels"
            )
            assert isinstance(teacher_logits_to_keep_count, int)
            assert isinstance(student_logits_to_keep_count, int)
            if teacher_logits_to_keep_count < teacher_answer_length:
                raise ValueError("Teacher requested suffix is shorter than its answer span.")
            if student_logits_to_keep_count < student_answer_length:
                raise ValueError("Student requested suffix is shorter than its answer span.")
            if teacher_answer_length != student_answer_length:
                raise ValueError(
                    "Teacher/student answer length mismatch before forward: "
                    f"teacher={teacher_answer_length} student={student_answer_length}"
                )
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

            teacher_forward_started = time.perf_counter()
            _synchronize_for_timing(teacher_input_device)
            with torch.no_grad():
                with _autocast_context(config.training.mixed_precision):
                    teacher_outputs = teacher_model(
                        **teacher_forward_inputs,
                        logits_to_keep=teacher_logits_to_keep_count,
                    )
                    teacher_suffix_logits = teacher_outputs.logits
                    assert teacher_suffix_logits.shape[1] == teacher_logits_to_keep_count
                    teacher_logits = teacher_suffix_logits[:, :teacher_answer_length, :]
                    assert teacher_logits.shape[1] == teacher_answer_labels.shape[1]
                    del teacher_outputs
            _synchronize_for_timing(teacher_input_device)
            teacher_forward_seconds = time.perf_counter() - teacher_forward_started
            teacher_labels = teacher_labels.to(device=teacher_logits.device)
            teacher_answer_labels = teacher_answer_labels.to(device=teacher_logits.device)

            student_forward_started = time.perf_counter()
            _synchronize_for_timing(student_input_device)
            with _autocast_context(config.training.mixed_precision):
                student_outputs = student_model(
                    **student_forward_inputs,
                    logits_to_keep=student_logits_to_keep_count,
                )
                student_suffix_logits = student_outputs.logits
                assert student_suffix_logits.shape[1] == student_logits_to_keep_count
                student_logits = student_suffix_logits[:, :student_answer_length, :]
                assert student_logits.shape[1] == student_answer_labels.shape[1]
                lm_loss = _answer_only_lm_loss(student_logits, student_answer_labels)
                del student_outputs
            _synchronize_for_timing(student_input_device)
            student_forward_seconds = time.perf_counter() - student_forward_started

            dbild_loss_started = time.perf_counter()
            _synchronize_for_timing(student_input_device)
            with _autocast_context(config.training.mixed_precision):
                (
                    aligned_teacher_logits,
                    aligned_student_logits,
                    aligned_attention_mask,
                    teacher_supervised_count,
                    student_supervised_count,
                    shared_vocab_size,
                    vocab_prefix_alignment_used,
                ) = align_logits_to_supervised_positions(
                    teacher_logits,
                    student_logits,
                    teacher_answer_labels,
                    student_answer_labels,
                )
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
                # This script targets online answer-only DBiLD L_Align, not full Switch-KD with VSD.
                total_loss = _weighted_online_align_loss(
                    lm_loss,
                    align_loss,
                    lm_loss_weight=config.distillation.lm_loss_weight,
                    dbild_loss_weight=config.distillation.dbild_loss_weight,
                )
                vsd_loss = total_loss.new_zeros(())
            _synchronize_for_timing(student_input_device)
            dbild_loss_seconds = time.perf_counter() - dbild_loss_started
            if smoke_test:
                _print_gpu_memory_stage("after_forward")
                validate_smoke_losses(lm_loss, align_loss, vsd_loss, total_loss)
            elif not torch.isfinite(torch.stack((lm_loss.detach(), align_loss.detach(), vsd_loss.detach(), total_loss.detach()))).all():
                raise FloatingPointError(
                    f"Online DBiLD produced non-finite loss for sample_id={batch.get('sample_id')}"
                )
            print(
                f"lm_loss={lm_loss.detach().float().item():.6f} "
                f"dbild_loss={align_loss.detach().float().item():.6f} "
                f"vsd_loss={vsd_loss.detach().float().item():.6f} "
                f"total_loss={total_loss.detach().float().item():.6f}"
            )
            if not first_batch_debug_printed:
                print("teacher_forward_logits_scope=answer_only")
                print("student_forward_logits_scope=answer_only")
                print("teacher_logits_to_keep_mode=suffix_covering_answer")
                print("student_logits_to_keep_mode=suffix_covering_answer")
                print(
                    "teacher_first_supervised_label_position="
                    f"{int(teacher_labels.shape[1]) - teacher_logits_to_keep_count + 1}"
                )
                print(
                    "teacher_first_required_logit_position="
                    f"{int(teacher_labels.shape[1]) - teacher_logits_to_keep_count}"
                )
                print(f"teacher_requested_suffix_logit_count={teacher_logits_to_keep_count}")
                print(f"teacher_answer_logit_count={teacher_answer_length}")
                print(f"teacher_trailing_discarded_logit_count={teacher_trailing_logit_count}")
                print(
                    "student_first_supervised_label_position="
                    f"{int(labels.shape[1]) - student_logits_to_keep_count + 1}"
                )
                print(
                    "student_first_required_logit_position="
                    f"{int(labels.shape[1]) - student_logits_to_keep_count}"
                )
                print(f"student_requested_suffix_logit_count={student_logits_to_keep_count}")
                print(f"student_answer_logit_count={student_answer_length}")
                print(f"student_trailing_discarded_logit_count={student_trailing_logit_count}")
                print(f"full_sequence_length={int(labels.shape[1])}")
                print(
                    "avoided_full_logit_positions="
                    f"{int(labels.shape[1]) - student_logits_to_keep_count}"
                )
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
                print("teacher_logits_scope=answer_only")
                print("student_logits_scope=answer_only")
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
            _synchronize_for_timing(student_input_device)
            backward_started = time.perf_counter()
            loss_for_backward.backward()
            _synchronize_for_timing(student_input_device)
            backward_seconds = time.perf_counter() - backward_started
            if smoke_test and global_step == 0:
                validate_smoke_gradient_contract(student_model, config.student.multimodal_projector_path)
                _print_gpu_memory_stage("after_backward")
            print(
                "Online DBiLD micro_batch "
                f"sample_id={batch.get('sample_id')} "
                f"sequence_length={int(labels.shape[1])} "
                f"supervised_token_count={int(student_supervised_count)} "
                f"teacher_forward_seconds={teacher_forward_seconds:.6f} "
                f"student_forward_seconds={student_forward_seconds:.6f} "
                f"DBiLD_loss_seconds={dbild_loss_seconds:.6f} "
                f"backward_seconds={backward_seconds:.6f} "
                f"micro_step={micro_step} accumulation_steps={grad_accum_steps}"
            )

            if micro_step % grad_accum_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if smoke_test:
                    _print_gpu_memory_stage("after_optimizer_step")
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
                print(f"optimizer step {global_step} completed")

                if total_target_steps is not None and global_step >= total_target_steps:
                    break

        if micro_step > 0 and micro_step % grad_accum_steps != 0 and (
            total_target_steps is None or global_step < total_target_steps
        ):
            _scale_partial_accumulation_gradients(
                student_model,
                grad_accum_steps=grad_accum_steps,
                micro_step=micro_step,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
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
            print(f"optimizer step {global_step} completed")

        should_validate = validation_enabled and ((epoch + 1) % int(config.training.validation_every_epochs) == 0)
        if should_validate and validation_rows is not None:
            summary = _validate_epoch(
                rows=validation_rows, config=config, student_model=student_model,
                teacher_model=teacher_model, student_processor=student_processor,
                teacher_processor=teacher_processor, student_input_device=student_input_device,
                teacher_input_device=teacher_input_device, batch_size=effective_batch_size,
                epoch=epoch + 1, global_step=global_step, best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
            best_val_loss = float(summary["best_val_loss"])
            epochs_without_improvement = int(summary["epochs_without_improvement"])
            early_stop_requested = bool(
                config.training.early_stopping_enabled
                and epochs_without_improvement >= config.training.early_stopping_patience
            )
            summary["early_stopped"] = early_stop_requested
            if rank == 0:
                with validation_history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                print("validation " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
                if summary["is_best"]:
                    _save_best_checkpoint(
                        student_model, student_processor, optimizer, scheduler,
                        checkpoint_dir=best_checkpoint_dir, epoch=epoch + 1,
                        global_step=global_step, best_val_loss=best_val_loss,
                        epochs_without_improvement=epochs_without_improvement, config=config,
                    )
            early_stop_requested = _broadcast_early_stop(early_stop_requested)
            if distributed:
                import torch.distributed as dist
                dist.barrier()
            if early_stop_requested:
                break

    if smoke_test and global_step != 1:
        raise RuntimeError(f"Smoke test expected exactly one optimizer step, got {global_step}.")
    if validation_enabled and config.training.restore_best_model and best_checkpoint_dir.exists():
        if distributed:
            import torch.distributed as dist
            dist.barrier()
        _restore_best_checkpoint(student_model, optimizer, scheduler, best_checkpoint_dir)
        print(f"restored_best_checkpoint={best_checkpoint_dir}")
    if rank == 0:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(adapter_dir)
        student_processor.save_pretrained(adapter_dir)
        adapter_metadata = {
            "base_projector_checksum_before_lora": getattr(student_model, "_main_merger_base_checksum", None),
            "base_projector_dtype_map": getattr(student_model, "_main_merger_dtype_map", None),
            "mixed_precision_source": getattr(student_model, "_mixed_precision_source", "load_time_exclusion"),
            "main_merger_quantized_before_peft": False,
            "merger_norm_dtype": "torch.float32",
        }
        (adapter_dir / "adapter_metadata.json").write_text(
            json.dumps(adapter_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if smoke_test:
            _validate_smoke_adapter_checkpoint(adapter_dir)
    if distributed:
        import torch.distributed as dist
        dist.barrier()
    print(f"peak_vram_allocated_bytes={_gpu_peak_memory_allocated()}")
    print(f"OK online DBiLD training completed: optimizer_steps={global_step}")
    return adapter_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config, max_steps_override=args.max_steps, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
