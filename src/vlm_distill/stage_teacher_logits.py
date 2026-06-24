from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .config_schema import PipelineConfig, resolve_label_path, resolve_teacher_logits_path
from .data_manifest import VlmSample, read_jsonl, validate_manifest
from .device_utils import (
    batch_to_device,
    ensure_stage_uses_cuda,
    get_module_by_path,
    print_stage_model_debug,
    resolve_requested_device_map,
    select_model_input_device,
)
from .model_loading import apply_attn_implementation, resolve_model_path
from .label_validation import build_teacher_token_decoder, validate_teacher_row
from .stage_answer_labeling import (
    _canonicalize_teacher_answer,
    _load_teacher_image,
    _normalize_teacher_answer,
    _strip_special_tokens,
    build_teacher,
)
from .stage_visual_switch_logits import _compact_adaptive_sequence_logits


DistillationMode = Literal["response", "adaptive_topk", "switch_kd"]

INACTIVE_LOGIT = -1.0e4


@dataclass(frozen=True)
class CompletedLogitsRows:
    ids: set[str]
    valid_count: int
    invalid_count: int
    first_invalid_keys: list[str] | None


class TeacherLogitsGenerator:
    """
    Generate teacher distillation data in one pass.

    Supported modes:

    1. response
       - teacher.generate()
       - saves teacher_answer only
       - for normal response distillation / SFT

    2. adaptive_topk
       - teacher.generate(output_scores=True)
       - saves teacher_answer + adaptive top-k generation logits
       - for adaptive top-k logits distillation

    3. switch_kd
       - teacher.generate(output_scores=True)
       - saves teacher_answer + adaptive top-k logits + entropy + entropy weights
       - for Switch-KD style distillation data
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._input_device = None

    def load(self) -> None:
        if self.config.teacher.backend == "mock":
            return

        if self.config.teacher.backend != "hf":
            raise ValueError(
                "teacher-logits currently supports backend='hf' or backend='mock'. "
                f"Got backend={self.config.teacher.backend!r}."
            )

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM

        model_path = resolve_model_path(self.config.teacher.model_name)
        requested_device_map = resolve_requested_device_map(
            self.config.teacher.device_map,
            quantization=self.config.teacher.quantization,
            role="teacher",
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": requested_device_map,
            "trust_remote_code": True,
        }
        apply_attn_implementation(model_kwargs, self.config.teacher.attn_implementation)

        if self.config.teacher.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.teacher.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            if self.config.teacher.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif self.config.teacher.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif self.config.teacher.torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForVLM.from_pretrained(
            model_path,
            **model_kwargs,
            local_files_only=True,
        ).eval()
        self._input_device = select_model_input_device(
            self._model,
            preferred_modules=(
                get_module_by_path(self._model, "model.visual"),
                get_module_by_path(self._model, "visual"),
                get_module_by_path(self._model, "model.language_model.embed_tokens"),
                get_module_by_path(self._model, "model.language_model"),
            ),
            label="Teacher",
        )
        print_stage_model_debug(
            stage_label="Teacher logits",
            model_path=model_path,
            quantization_mode=self.config.teacher.quantization,
            requested_device_map=requested_device_map,
            model=self._model,
            selected_input_device=self._input_device,
        )
        ensure_stage_uses_cuda(
            stage_label="Teacher logits",
            requested_device_map=requested_device_map,
            model=self._model,
            selected_input_device=self._input_device,
        )

    def generate_for_sample(
        self,
        sample: VlmSample,
        *,
        mode: DistillationMode,
    ) -> dict[str, Any]:
        if self.config.teacher.backend == "mock":
            return self._mock_generate_for_sample(sample, mode=mode)

        if self._model is None or self._processor is None:
            self.load()

        import torch

        image_path = self.config.data.image_root / sample.image
        image = _load_teacher_image(
            image_path,
            self.config.teacher.image_resize,
        )
        prompt = _format_prompt(self.config, sample)

        with torch.no_grad():
            inputs = self._build_multimodal_inputs(image, prompt)
            prompt_len = int(inputs["input_ids"].shape[1])
            inputs = batch_to_device(inputs, self._input_device)

            include_scores = mode in {"adaptive_topk", "switch_kd"}
            generation = self._generate(inputs, include_scores=include_scores)

            generated_ids, scores = _extract_generated_ids_and_scores(
                generation,
                prompt_len=prompt_len,
                include_scores=include_scores,
            )

            answer = self._decode(generated_ids).strip()
            normalized_answer = _normalize_teacher_answer(sample, answer).strip()
            teacher_tokens = self.tokenize_teacher_answer(normalized_answer)

            result: dict[str, Any] = {
                "teacher_answer": answer,
                "teacher_confidence": 1.0,
                "teacher_rationale": f"Generated by Hugging Face teacher in {mode} mode.",
                "distillation_mode": mode,
                "teacher_generated_ids": generated_ids.detach().cpu().tolist(),
                "teacher_tokens": teacher_tokens,
            }
            result["teacher_answer"] = normalized_answer

            if mode == "response":
                return result

            if not scores:
                raise ValueError(
                    "Teacher generation did not return scores. "
                    "Cannot build logits distillation dataset."
                )

            raw_tokens = _flatten_generated_ids(generated_ids.detach().cpu().tolist())
            try:
                raw_matches = (
                    raw_tokens == teacher_tokens
                    and _canonicalize_teacher_answer(answer) == _canonicalize_teacher_answer(normalized_answer)
                )
            except ValueError:
                raw_matches = False
            if raw_matches and len(scores) == len(teacher_tokens):
                logits_payload = self._build_generation_logits_payload(
                    scores=scores,
                    mode=mode,
                    prompt_len=0,
                    source="generation_scores",
                )
            else:
                logits_payload = compute_teacher_forced_answer_logits(
                    image=image,
                    prompt=prompt,
                    teacher_answer=normalized_answer,
                    teacher_tokens=teacher_tokens,
                    model=self._model,
                    processor=self._processor,
                    config=self.config,
                )
            result.update(logits_payload)
            return result

    def tokenize_teacher_answer(self, answer: str) -> list[int]:
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            encoded = self._processor(text=[answer], return_tensors=None)
            input_ids = encoded["input_ids"][0]
        else:
            input_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        return [int(token_id) for token_id in input_ids]

    def _generate(self, inputs: dict[str, Any], *, include_scores: bool):
        temperature = float(self.config.teacher.temperature)
        do_sample = temperature > 0

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.teacher.max_new_tokens),
            "do_sample": do_sample,
        }

        if do_sample:
            generate_kwargs["temperature"] = temperature

        if include_scores:
            generate_kwargs["output_scores"] = True
            generate_kwargs["return_dict_in_generate"] = True

        return self._model.generate(
            **inputs,
            **generate_kwargs,
        )

    def _decode(self, generated_ids):
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _build_multimodal_inputs(self, image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

    def _build_generation_logits_payload(
        self,
        *,
        scores: list[Any],
        mode: DistillationMode,
        prompt_len: int,
        source: str = "generation_scores",
    ) -> dict[str, Any]:
        field = self.config.distillation.teacher_logits_field

        compact = _compact_adaptive_generation_scores(
            scores=scores,
            mode=mode,
            base_k=int(self.config.distillation.dbild_top_k),
            max_cached_logits_vocab=self.config.distillation.max_cached_logits_vocab,
            temperature=float(self.config.distillation.kd_temperature),
        )

        return {
            field: compact,
            f"{field}_format": mode,
            f"{field}_prompt_len": prompt_len,
            f"{field}_vocab_size": compact["vocab_size"],
            f"{field}_aligned_to_answer": True,
            f"{field}_source": source,
            f"{field}_temperature": float(self.config.distillation.kd_temperature),
        }

    def _mock_generate_for_sample(
        self,
        sample: VlmSample,
        *,
        mode: DistillationMode,
    ) -> dict[str, Any]:
        answer = _mock_answer(sample)
        answer = _normalize_teacher_answer(sample, answer).strip()
        teacher_tokens = [ord(char) for char in answer]

        result: dict[str, Any] = {
            **asdict(sample),
            "teacher_answer": answer,
            "teacher_confidence": 1.0,
            "teacher_rationale": f"Mock teacher used in {mode} mode.",
            "distillation_mode": mode,
            "teacher_generated_ids": [[1, 2, 3]],
            "teacher_tokens": teacher_tokens,
        }

        if mode == "response":
            return result

        field = (
            self.config.distillation.switch_logits_field
            if mode == "switch_kd"
            else self.config.distillation.teacher_logits_field
        )

        steps = len(teacher_tokens)
        vocab_size = 16
        base_k = int(self.config.distillation.dbild_top_k)
        max_k = min(vocab_size, max(2, min(base_k, 8)))

        indices = []
        values = []
        entropy = []
        token_k = []
        entropy_weight = []

        for _ in range(steps):
            step_indices = list(range(max_k))
            step_values = [5.0 - rank for rank in range(max_k)]
            step_entropy = 1.0
            step_k = min(max_k, max(2, base_k))

            indices.append(step_indices)
            values.append(step_values)
            entropy.append(step_entropy)
            token_k.append(step_k)
            entropy_weight.append(_entropy_to_weight(step_entropy))

        compact: dict[str, Any] = {
            "indices": [indices],
            "values": [values],
            "shape": [1, steps, vocab_size],
            "vocab_size": vocab_size,
            "token_k": [token_k],
            "entropy": [entropy],
            "adaptive": True,
        }

        if mode == "switch_kd":
            compact["entropy_weight"] = [entropy_weight]
            compact["switch_kd"] = True

        result.update(
            {
                field: compact,
                f"{field}_format": mode,
                f"{field}_prompt_len": 0,
                f"{field}_vocab_size": vocab_size,
                f"{field}_aligned_to_answer": True,
                f"{field}_source": "teacher_forcing_forward",
                f"{field}_temperature": float(self.config.distillation.kd_temperature),
            }
        )
        return result


def create_teacher_precompute_dataset(config: PipelineConfig, samples: list[VlmSample] | None = None) -> Path:
    require_logits = bool(getattr(config.distillation, "teacher_logits", True))
    samples = samples or validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )
    output_path = resolve_label_path(config.data)
    teacher_logits_path = resolve_teacher_logits_path(config.data)
    completed = _load_completed_teacher_rows(output_path, config=config, require_logits=require_logits)
    if completed.invalid_count:
        _rewrite_valid_teacher_rows(output_path, config=config, require_logits=require_logits)
    completed_ids = completed.ids
    pending_samples = [sample for sample in samples if sample.id not in completed_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if teacher_logits_path != output_path:
        teacher_logits_path.parent.mkdir(parents=True, exist_ok=True)

    print("Unified teacher precompute:")
    print(f"  distillation.method: {_resolve_distillation_mode(config)}")
    print(f"  distillation.teacher_logits: {require_logits}")
    print(f"  label_path: {output_path}")
    print(f"  teacher_logits_path: {teacher_logits_path}")
    print(f"  unified_output: {'duplicated' if teacher_logits_path != output_path else 'single_path'}")
    print(f"  total samples: {len(samples)}")
    print(f"  valid completed rows: {completed.valid_count}")
    print(f"  invalid stale rows: {completed.invalid_count}")
    print(f"  pending rows: {len(pending_samples)}")
    if completed.first_invalid_keys:
        print(f"  first invalid row id/reason: {completed.first_invalid_keys}")

    if not pending_samples:
        _mirror_unified_teacher_output(output_path, teacher_logits_path)
        return output_path

    if config.teacher.backend == "hf" and require_logits:
        generator: Any = TeacherLogitsGenerator(config)
        mode = _resolve_teacher_logits_mode(config)
    else:
        generator = build_teacher(config)
        mode = "response"

    completed_now = 0
    with output_path.open("a", encoding="utf-8") as label_handle:
        mirror_handle = (
            teacher_logits_path.open("a", encoding="utf-8")
            if teacher_logits_path != output_path
            else None
        )
        try:
            for sample in pending_samples:
                started = time.perf_counter()
                if isinstance(generator, TeacherLogitsGenerator):
                    generated = generator.generate_for_sample(sample, mode=mode)
                else:
                    generated = _generate_label_with_optional_mock_logits(
                        config,
                        generator,
                        sample,
                        include_logits=require_logits,
                    )
                row = {**asdict(sample), **generated}
                if require_logits:
                    _assert_teacher_logits_answer_length(row, config.distillation.teacher_logits_field)
                encoded = json.dumps(row, ensure_ascii=False) + "\n"
                label_handle.write(encoded)
                label_handle.flush()
                if mirror_handle is not None:
                    mirror_handle.write(encoded)
                    mirror_handle.flush()
                completed_now += 1
                elapsed = time.perf_counter() - started
                print(
                    "[teacher-precompute] "
                    f"total={len(samples)} completed={len(completed_ids) + completed_now} "
                    f"pending={len(pending_samples) - completed_now} id={sample.id} "
                    f"elapsed_seconds_per_sample={elapsed:.2f}"
                )
        finally:
            if mirror_handle is not None:
                mirror_handle.close()
    return output_path


def create_teacher_logits_compat_dataset(config: PipelineConfig) -> Path:
    """Compatibility path: fill teacher_logits for existing valid teacher labels."""
    label_path = resolve_label_path(config.data)
    output_path = resolve_teacher_logits_path(config.data)
    decoder = build_teacher_token_decoder(config)
    rows = read_jsonl(label_path)
    valid_rows = []
    for row in rows:
        valid, reason = validate_teacher_row(row, require_logits=False, decode_tokens=decoder)
        if not valid:
            raise ValueError(f"Cannot fill teacher_logits for invalid teacher label id={row.get('id')}: {reason}")
        if validate_teacher_row(row, require_logits=True, decode_tokens=decoder)[0]:
            valid_rows.append(row)
            continue
        sample = VlmSample(**{key: row.get(key) for key in VlmSample.__dataclass_fields__ if key in row})
        valid_rows.append(_fill_logits_for_existing_row(config, sample, row))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if output_path != label_path:
        with label_path.open("w", encoding="utf-8") as handle:
            for row in valid_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def _generate_label_with_optional_mock_logits(
    config: PipelineConfig,
    teacher: Any,
    sample: VlmSample,
    *,
    include_logits: bool,
) -> dict[str, Any]:
    label = teacher.answer(sample)
    answer = _normalize_teacher_answer(sample, str(label["teacher_answer"])).strip()
    tokenizer = getattr(teacher, "tokenize_teacher_answer", None)
    if callable(tokenizer):
        tokens = [int(token_id) for token_id in tokenizer(answer)]
    else:
        tokens = [ord(char) for char in answer]
    row = {
        "teacher_answer": answer,
        "teacher_tokens": tokens,
        "teacher_confidence": float(label.get("teacher_confidence", 1.0)),
        "teacher_rationale": label.get("teacher_rationale", "Generated by teacher backend."),
        "distillation_mode": _resolve_teacher_logits_mode(config),
    }
    if include_logits:
        row.update(_mock_answer_only_logits_payload(config, len(tokens), source="teacher_forcing_forward"))
    return row


def _fill_logits_for_existing_row(
    config: PipelineConfig,
    sample: VlmSample,
    row: dict[str, Any],
) -> dict[str, Any]:
    if config.teacher.backend == "hf":
        generator = TeacherLogitsGenerator(config)
        generator.load()
        image_path = config.data.image_root / sample.image
        image = _load_teacher_image(image_path, config.teacher.image_resize)
        prompt = _format_prompt(config, sample)
        logits_payload = compute_teacher_forced_answer_logits(
            image=image,
            prompt=prompt,
            teacher_answer=str(row["teacher_answer"]),
            teacher_tokens=[int(token_id) for token_id in row["teacher_tokens"]],
            model=generator._model,
            processor=generator._processor,
            config=config,
        )
    else:
        logits_payload = _mock_answer_only_logits_payload(
            config,
            len([int(token_id) for token_id in row["teacher_tokens"]]),
            source="teacher_forcing_forward",
        )
    updated = dict(row)
    updated.update(logits_payload)
    return updated


def _mock_answer_only_logits_payload(config: PipelineConfig, answer_len: int, *, source: str) -> dict[str, Any]:
    field = config.distillation.teacher_logits_field
    vocab_size = 16
    max_k = min(vocab_size, max(1, min(int(config.distillation.dbild_top_k), 8)))
    indices = []
    values = []
    token_k = []
    entropy = []
    for step_index in range(answer_len):
        step_indices = [(step_index + offset) % vocab_size for offset in range(max_k)]
        indices.append(step_indices)
        values.append([5.0 - rank for rank in range(max_k)])
        token_k.append(max_k)
        entropy.append(1.0)
    compact = {
        "indices": [indices],
        "values": [values],
        "shape": [1, answer_len, vocab_size],
        "vocab_size": vocab_size,
        "token_k": [token_k],
        "entropy": [entropy],
        "adaptive": True,
    }
    return {
        field: compact,
        f"{field}_format": _resolve_teacher_logits_mode(config),
        f"{field}_prompt_len": 0,
        f"{field}_vocab_size": vocab_size,
        f"{field}_aligned_to_answer": True,
        f"{field}_source": source,
        f"{field}_temperature": float(config.distillation.kd_temperature),
    }


def compute_teacher_forced_answer_logits(
    *,
    image,
    prompt: str,
    teacher_answer: str,
    teacher_tokens: list[int],
    model,
    processor,
    config: PipelineConfig,
) -> dict[str, Any]:
    import torch

    prompt_inputs = _build_multimodal_inputs_for_processor(processor, image, prompt)
    full_inputs = _build_multimodal_inputs_for_processor(
        processor,
        image,
        _join_prompt_and_answer(prompt, teacher_answer),
    )
    input_device = select_model_input_device(model, label="Teacher forcing")
    prompt_inputs = batch_to_device(prompt_inputs, input_device)
    full_inputs = batch_to_device(full_inputs, input_device)
    prefix_len = int(prompt_inputs["input_ids"].shape[1])
    answer_len = len(teacher_tokens)
    with torch.no_grad():
        outputs = model(**full_inputs)
    full_logits = outputs.logits
    answer_logits = full_logits[:, prefix_len - 1 : prefix_len - 1 + answer_len, :]
    if int(answer_logits.shape[1]) != answer_len:
        raise ValueError(
            "Teacher-forced logits length mismatch: "
            f"answer_logits_len={int(answer_logits.shape[1])}, teacher_tokens_len={answer_len}"
        )
    compact = _compact_adaptive_sequence_logits(
        answer_logits,
        base_k=int(config.distillation.dbild_top_k),
        max_cached_logits_vocab=config.distillation.max_cached_logits_vocab,
        temperature=float(config.distillation.kd_temperature),
    )
    field = config.distillation.teacher_logits_field
    if int(compact["shape"][1]) != answer_len:
        raise ValueError("Compacted teacher logits are not aligned to teacher_tokens.")
    return {
        field: compact,
        f"{field}_format": _resolve_teacher_logits_mode(config),
        f"{field}_prompt_len": 0,
        f"{field}_vocab_size": int(compact["vocab_size"]),
        f"{field}_aligned_to_answer": True,
        f"{field}_source": "teacher_forcing_forward",
        f"{field}_temperature": float(config.distillation.kd_temperature),
    }


def _build_multimodal_inputs_for_processor(processor, image, text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    templated = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(text=[templated], images=[image], return_tensors="pt")


def _join_prompt_and_answer(prompt: str, answer: str) -> str:
    separator = "" if prompt.endswith((" ", "\n")) else " "
    return f"{prompt}{separator}{answer}".strip()


def _assert_teacher_logits_answer_length(row: dict[str, Any], field_name: str) -> None:
    valid, reason = validate_teacher_row(row, require_logits=True, logits_field=field_name)
    if not valid:
        raise ValueError(f"Unified teacher row failed validation id={row.get('id')}: {reason}")


def _load_completed_teacher_rows(
    path: Path,
    *,
    config: PipelineConfig,
    require_logits: bool,
) -> CompletedLogitsRows:
    if not path.exists():
        return CompletedLogitsRows(ids=set(), valid_count=0, invalid_count=0, first_invalid_keys=None)
    completed_ids: set[str] = set()
    valid_count = 0
    invalid_count = 0
    first_invalid: list[str] | None = None
    decoder = build_teacher_token_decoder(config)
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is None:
            continue
        valid, reason = validate_teacher_row(row, require_logits=require_logits, decode_tokens=decoder)
        if valid:
            completed_ids.add(str(sample_id))
            valid_count += 1
        else:
            invalid_count += 1
            if first_invalid is None:
                first_invalid = [str(sample_id), str(reason)]
    return CompletedLogitsRows(
        ids=completed_ids,
        valid_count=valid_count,
        invalid_count=invalid_count,
        first_invalid_keys=first_invalid,
    )


def _rewrite_valid_teacher_rows(path: Path, *, config: PipelineConfig, require_logits: bool) -> None:
    decoder = build_teacher_token_decoder(config)
    valid_rows = [
        row for row in read_jsonl(path)
        if validate_teacher_row(row, require_logits=require_logits, decode_tokens=decoder)[0]
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[teacher-precompute] pruned invalid existing rows from {path}; remaining_valid_rows={len(valid_rows)}")


def _mirror_unified_teacher_output(label_path: Path, teacher_logits_path: Path) -> None:
    if label_path == teacher_logits_path:
        return
    if not label_path.exists():
        return
    rows = read_jsonl(label_path)
    with teacher_logits_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_teacher_logits_dataset(config: PipelineConfig) -> Path:
    """
    Compatibility alias.

    Fill or regenerate teacher_logits for existing valid teacher labels without
    independently generating a new teacher_answer.
    """
    print("[teacher-logits] compatibility mode: filling logits for existing teacher labels.")
    return create_teacher_logits_compat_dataset(config)

    method = _resolve_distillation_mode(config)
    mode = _resolve_teacher_logits_mode(config)
    include_scores = mode in {"adaptive_topk", "switch_kd"}
    teacher_logits_field = config.distillation.teacher_logits_field
    samples = validate_manifest(
        config.data.manifest_path,
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )

    output_path = resolve_teacher_logits_path(config.data)
    completed = _load_completed_ids(
        output_path,
        field_name=teacher_logits_field,
        require_logits=mode in {"adaptive_topk", "switch_kd"},
    )
    if completed.invalid_count and mode in {"adaptive_topk", "switch_kd"}:
        _rewrite_valid_completed_rows(output_path, field_name=teacher_logits_field)
    completed_ids = completed.ids
    total = len(samples)
    pending_samples = [sample for sample in samples if sample.id not in completed_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Teacher logits debug:")
    print(f"  distillation.method: {method}")
    print(f"  teacher_logits_mode: {mode}")
    if method == "switch_kd" and mode == "adaptive_topk":
        print("  switch_kd teacher logits mode resolved to adaptive_topk")
    print(f"  include_scores: {include_scores}")
    print(f"  manifest_path: {config.data.manifest_path}")
    print(f"  label_path: {resolve_label_path(config.data)}")
    print(f"  teacher_logits_path: {output_path}")
    print(f"  teacher_logits_field: {teacher_logits_field}")
    print(f"  input_samples: {total}")
    print(f"  completed_rows_loaded: {completed.valid_count + completed.invalid_count}")
    print(f"  completed_valid_count: {completed.valid_count}")
    print(f"  completed_invalid_count: {completed.invalid_count}")
    print(f"  first_invalid_row_keys: {completed.first_invalid_keys}")
    print(f"  pending_rows: {len(pending_samples)}")
    print(
        f"Teacher logits samples: total={total}, completed={len(completed_ids)}, "
        f"pending={len(pending_samples)}, output={output_path}"
    )

    if not pending_samples:
        print("No pending samples. Existing teacher logits output is already complete for this manifest.")
        return output_path

    generator = TeacherLogitsGenerator(config)
    completed_now = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for sample in pending_samples:
            started = time.perf_counter()
            row = {
                **asdict(sample),
                **generator.generate_for_sample(sample, mode=mode),
            }
            if completed_now == 0 and include_scores:
                _validate_first_teacher_logits_row(
                    row,
                    field_name=teacher_logits_field,
                    method=method,
                    mode=mode,
                    include_scores=include_scores,
                    output_path=output_path,
                )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed_now += 1
            elapsed = time.perf_counter() - started
            total_done = len(completed_ids) + completed_now
            pending = total - total_done
            print(
                "[teacher-logits] "
                f"total={total} completed={total_done} pending={pending} "
                f"current_sample_id={sample.id} elapsed_seconds_per_sample={elapsed:.2f} "
                f"output_path={output_path}"
            )
    return output_path


def _load_completed_ids(
    path: Path,
    *,
    field_name: str | None = None,
    require_logits: bool = False,
) -> CompletedLogitsRows:
    if not path.exists():
        return CompletedLogitsRows(ids=set(), valid_count=0, invalid_count=0, first_invalid_keys=None)

    completed_ids: set[str] = set()
    valid_count = 0
    invalid_count = 0
    first_invalid_keys: list[str] | None = None
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is None:
            continue
        if require_logits and field_name is not None and not _is_valid_logits_row(row, field_name):
            invalid_count += 1
            if first_invalid_keys is None:
                first_invalid_keys = sorted(str(key) for key in row.keys())
            continue
        completed_ids.add(str(sample_id))
        valid_count += 1
    return CompletedLogitsRows(
        ids=completed_ids,
        valid_count=valid_count,
        invalid_count=invalid_count,
        first_invalid_keys=first_invalid_keys,
    )


def _rewrite_valid_completed_rows(path: Path, *, field_name: str) -> None:
    valid_rows = [row for row in read_jsonl(path) if _is_valid_logits_row(row, field_name)]
    with path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[teacher-logits] pruned invalid existing rows from {path}; "
        f"remaining_valid_rows={len(valid_rows)}"
    )


def _is_valid_logits_row(row: dict[str, Any], field_name: str) -> bool:
    payload = row.get(field_name)
    if not isinstance(payload, dict):
        return False
    if not all(key in payload for key in ("indices", "values", "vocab_size")):
        return False
    indices_shape = _nested_shape(payload.get("indices"))
    values_shape = _nested_shape(payload.get("values"))
    if not indices_shape or not values_shape:
        return False
    return indices_shape == values_shape


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return ()
    first_shape = _nested_shape(value[0])
    for item in value[1:]:
        if _nested_shape(item) != first_shape:
            return ()
    return (len(value), *first_shape)


def _validate_first_teacher_logits_row(
    row: dict[str, Any],
    *,
    field_name: str,
    method: DistillationMode,
    mode: DistillationMode,
    include_scores: bool,
    output_path: Path,
) -> None:
    if _is_valid_logits_row(row, field_name):
        return
    raise ValueError(
        "Teacher logits output row is missing a valid logits payload. "
        f"method={method}, resolved_mode={mode}, include_scores={include_scores}, "
        f"output_path={output_path}, first_row_keys={sorted(row.keys())}"
    )


def _flatten_generated_ids(generated_ids: Any) -> list[int]:
    if isinstance(generated_ids, list) and len(generated_ids) == 1 and isinstance(generated_ids[0], list):
        return [int(value) for value in generated_ids[0]]
    if isinstance(generated_ids, list):
        return [int(value) for value in generated_ids]
    return []


def _resolve_teacher_logits_mode(config: PipelineConfig) -> DistillationMode:
    mode = _resolve_distillation_mode(config)
    if mode == "switch_kd":
        return "adaptive_topk"
    return mode


def _resolve_distillation_mode(config: PipelineConfig) -> DistillationMode:
    raw_mode = str(
        getattr(config.distillation, "mode", None)
        or getattr(config.distillation, "method", "response")
        or "response"
    ).strip().lower()

    aliases = {
        "sft": "response",
        "response": "response",
        "response_distillation": "response",
        "response-distillation": "response",
        "topk": "adaptive_topk",
        "topk_logits": "adaptive_topk",
        "top-k": "adaptive_topk",
        "top-k-logits": "adaptive_topk",
        "adaptive_topk": "adaptive_topk",
        "adaptive-topk": "adaptive_topk",
        "adaptive_topk_logits": "adaptive_topk",
        "adaptive-topk-logits": "adaptive_topk",
        "dbild": "adaptive_topk",
        "switch": "switch_kd",
        "switch_kd": "switch_kd",
        "switch-kd": "switch_kd",
    }

    if raw_mode not in aliases:
        raise ValueError(
            f"Unsupported distillation method: {raw_mode!r}. "
            "Expected one of: response, adaptive_topk, switch_kd."
        )

    return aliases[raw_mode]  # type: ignore[return-value]


def _format_prompt(config: PipelineConfig, sample: VlmSample) -> str:
    template = config.distillation.prompt_template

    try:
        return template.format(
            query=sample.query or "",
            question=sample.query or "",
            target_label=sample.target_label or "",
            target_type=sample.target_type or "",
            task=sample.task,
        )
    except KeyError as exc:
        raise KeyError(
            f"Prompt template references unsupported placeholder: {exc}. "
            "Supported placeholders are: query, question, target_label, target_type, task."
        ) from exc


def _compact_adaptive_generation_scores(
    *,
    scores: list[Any],
    mode: DistillationMode,
    base_k: int,
    max_cached_logits_vocab: int | None,
    temperature: float,
) -> dict[str, Any]:
    """
    Convert generation scores into a compact adaptive top-k cache.

    Input:
      scores: list of tensors, each shaped [batch, vocab]

    Output:
      {
        "indices": [batch, generated_steps, max_k],
        "values": [batch, generated_steps, max_k],
        "shape": [batch, generated_steps, vocab],
        "vocab_size": vocab,
        "token_k": [batch, generated_steps],
        "entropy": [batch, generated_steps],
        "entropy_weight": [batch, generated_steps],  # switch_kd only
      }

    Notes:
      - We keep a rectangular [B, T, max_k] structure so existing cache
        materialization code can still scatter indices/values.
      - For each token, only the first token_k entries are active.
      - Inactive entries are filled with INACTIVE_LOGIT.
    """

    import torch

    if not scores:
        raise ValueError("Cannot compact empty generation scores.")

    first = scores[0].detach().float().cpu()
    if first.ndim != 2:
        raise ValueError(
            f"Expected each generation score to have shape [batch, vocab], got {tuple(first.shape)}"
        )

    batch_size, vocab_size = first.shape

    base_k = max(1, int(base_k))
    low_k = max(1, base_k // 4)
    mid_k = base_k
    high_k = base_k * 2

    if max_cached_logits_vocab is not None:
        high_k = min(high_k, int(max_cached_logits_vocab))

    max_k = min(vocab_size, max(low_k, mid_k, high_k))

    low_entropy_threshold = 1.0
    high_entropy_threshold = 2.5

    step_indices: list[Any] = []
    step_values: list[Any] = []
    step_entropy: list[Any] = []
    step_token_k: list[Any] = []
    step_entropy_weight: list[Any] = []

    safe_temperature = max(float(temperature), 1e-6)

    for score in scores:
        logits = score.detach().float().cpu()
        if logits.ndim != 2:
            raise ValueError(
                f"Expected each generation score to have shape [batch, vocab], got {tuple(logits.shape)}"
            )

        if logits.shape[0] != batch_size or logits.shape[1] != vocab_size:
            raise ValueError(
                "All generation scores must share the same [batch, vocab] shape. "
                f"Expected {(batch_size, vocab_size)}, got {tuple(logits.shape)}"
            )

        scaled_logits = logits / safe_temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        top_values, top_indices = torch.topk(
            logits,
            k=max_k,
            dim=-1,
        )

        token_k = torch.empty((batch_size,), dtype=torch.long)
        entropy_weight = torch.empty((batch_size,), dtype=torch.float32)

        for batch_index in range(batch_size):
            entropy_value = float(entropy[batch_index].item())
            active_k = _adaptive_k(
                entropy_value,
                low_entropy_threshold=low_entropy_threshold,
                high_entropy_threshold=high_entropy_threshold,
                low_k=low_k,
                mid_k=mid_k,
                high_k=high_k,
                max_k=max_k,
            )

            token_k[batch_index] = active_k
            entropy_weight[batch_index] = _entropy_to_weight(entropy_value)

            if active_k < max_k:
                top_values[batch_index, active_k:] = INACTIVE_LOGIT

        step_indices.append(top_indices)
        step_values.append(top_values)
        step_entropy.append(entropy)
        step_token_k.append(token_k)
        step_entropy_weight.append(entropy_weight)

    indices_tensor = torch.stack(step_indices, dim=1)
    values_tensor = torch.stack(step_values, dim=1)
    entropy_tensor = torch.stack(step_entropy, dim=1)
    token_k_tensor = torch.stack(step_token_k, dim=1)

    compact: dict[str, Any] = {
        "indices": indices_tensor.tolist(),
        "values": values_tensor.tolist(),
        "shape": [batch_size, len(scores), vocab_size],
        "vocab_size": int(vocab_size),
        "adaptive": True,
        "token_k": token_k_tensor.tolist(),
        "entropy": entropy_tensor.tolist(),
        "k_policy": {
            "low_entropy_threshold": low_entropy_threshold,
            "high_entropy_threshold": high_entropy_threshold,
            "low_k": int(low_k),
            "mid_k": int(mid_k),
            "high_k": int(high_k),
            "max_k": int(max_k),
        },
    }

    if mode == "switch_kd":
        entropy_weight_tensor = torch.stack(step_entropy_weight, dim=1)
        compact["entropy_weight"] = entropy_weight_tensor.tolist()
        compact["switch_kd"] = True

    return compact


def _adaptive_k(
    entropy: float,
    *,
    low_entropy_threshold: float,
    high_entropy_threshold: float,
    low_k: int,
    mid_k: int,
    high_k: int,
    max_k: int,
) -> int:
    if entropy < low_entropy_threshold:
        return min(max_k, low_k)

    if entropy < high_entropy_threshold:
        return min(max_k, mid_k)

    return min(max_k, high_k)


def _entropy_to_weight(entropy: float) -> float:
    return 1.0 / (1.0 + max(float(entropy), 0.0))


def _extract_generated_ids_and_scores(
    generation: Any,
    *,
    prompt_len: int,
    include_scores: bool,
):
    if include_scores:
        sequences = generation.sequences
        scores = list(generation.scores or [])
        generated_steps = len(scores)

        if generated_steps > 0:
            if sequences.shape[1] >= prompt_len + generated_steps:
                generated_ids = sequences[:, prompt_len : prompt_len + generated_steps]
            else:
                generated_ids = sequences[:, -generated_steps:]
        else:
            generated_ids = sequences[:, prompt_len:] if sequences.shape[1] > prompt_len else sequences

        return generated_ids, scores

    sequences = generation
    if sequences.shape[1] > prompt_len:
        generated_ids = sequences[:, prompt_len:]
    else:
        generated_ids = sequences

    return generated_ids, []


def _mock_answer(sample: VlmSample) -> str:
    if sample.answer:
        return sample.answer

    if sample.task == "parsing":
        return json.dumps(
            {
                "focused_element": "mock settings",
                "elements": [
                    {
                        "label": "mock icon",
                        "type": "app_icon",
                        "bbox": [0, 0, 100, 100],
                    },
                    {
                        "label": "mock settings",
                        "type": "button",
                        "bbox": [120, 0, 220, 100],
                    },
                ],
            },
            ensure_ascii=False,
        )

    if sample.task == "grounding":
        return json.dumps(
            {
                "label": sample.target_label or "target",
                "type": sample.target_type or "object",
                "bbox": [0, 0, 100, 100],
            },
            ensure_ascii=False,
        )

    return f"mock answer for {sample.task}"
