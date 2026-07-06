#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_distill.data_manifest import read_jsonl, write_jsonl
from vlm_distill.device_utils import (
    ensure_stage_uses_cuda,
    print_stage_model_debug,
    resolve_requested_device_map,
    select_model_input_device,
)
from vlm_distill.model_loading import apply_attn_implementation, resolve_model_path


def build_prompt(max_elements: int) -> str:
    return f"""List visible clickable/focusable UI elements in this TV screenshot.

Return ONLY one complete minified JSON object.
No markdown. No explanation.

Schema:
{{"e":[["name",[x1,y1,x2,y2]]]}}

Rules:
- Use short names.
- Use original image pixel coordinates.
- Include at most {max_elements} elements.
- Each element name may appear only once.
- Do not repeat elements.
- Do not include background, decorative graphics, long descriptions, or non-clickable text.
- After the JSON object is complete, stop immediately.
- If unsure, omit the element.
"""


MODEL_ROLE_ORDER = ("teacher32b", "base8b", "distill32to8")
COMPARISON_ROLES = ("base8b", "distill32to8")


@dataclass(frozen=True)
class ModelRunSpec:
    role: str
    model_path: str
    adapter_path: str | None = None


@dataclass
class LoadedInferenceModel:
    role: str
    model_path: str
    adapter_path: str | None
    processor: Any
    model: Any
    input_device: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare teacher/base/distilled Qwen-VL models on UI element listing."
    )
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--comparison_output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_output_json", required=True, type=Path)
    parser.add_argument("--teacher32b_path", required=True)
    parser.add_argument("--reuse_teacher_raw_jsonl", type=Path)
    parser.add_argument("--base8b_path", required=True)
    parser.add_argument("--distill_model_path", required=True)
    parser.add_argument("--distill_adapter_path")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_elements", type=int, default=30)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--match_threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _prepare_output_paths(
        args.output_jsonl,
        args.comparison_output_jsonl,
        args.summary_output_json,
    )

    samples = _load_input_rows(args.input_jsonl, max_samples=args.max_samples)
    model_specs = _build_model_specs(args)
    teacher_reuse = _load_reused_teacher_rows(
        args.reuse_teacher_raw_jsonl,
        samples=samples,
    )

    raw_rows_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in MODEL_ROLE_ORDER}
    raw_rows_by_role["teacher32b"] = teacher_reuse["rows"]
    prompt = build_prompt(args.max_elements)
    for spec in model_specs:
        loaded = _load_model(spec, args)
        try:
            rows = _run_model_on_samples(
                loaded=loaded,
                samples=samples,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            raw_rows_by_role[spec.role] = rows
        finally:
            _release_model(loaded)

    comparison_rows = build_comparison_rows(
        raw_rows_by_role=raw_rows_by_role,
        match_threshold=args.match_threshold,
    )
    _attach_bbox_eval_to_raw_rows(
        raw_rows_by_role=raw_rows_by_role,
        comparison_rows=comparison_rows,
    )
    write_jsonl(args.output_jsonl, _flatten_raw_rows(raw_rows_by_role))
    write_jsonl(args.comparison_output_jsonl, comparison_rows)

    summary = build_summary(
        raw_rows_by_role=raw_rows_by_role,
        comparison_rows=comparison_rows,
        match_threshold=args.match_threshold,
        teacher_reuse=teacher_reuse["summary"],
        generation_settings={
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_elements": args.max_elements,
        },
    )
    args.summary_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_output_paths(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()


def _load_input_rows(path: Path, *, max_samples: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path, max_samples=max_samples)
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} is not a JSON object.")
        sample_id = row.get("id")
        image = row.get("image")
        if sample_id is None or image is None:
            raise ValueError(f"{path}:{index} must contain both 'id' and 'image'.")
        validated.append({"id": str(sample_id), "image": str(image)})
    return validated


def _build_model_specs(args: argparse.Namespace) -> list[ModelRunSpec]:
    specs = [
        ModelRunSpec(role="base8b", model_path=args.base8b_path),
        ModelRunSpec(
            role="distill32to8",
            model_path=args.distill_model_path,
            adapter_path=args.distill_adapter_path,
        ),
    ]
    if args.reuse_teacher_raw_jsonl is None:
        specs.insert(0, ModelRunSpec(role="teacher32b", model_path=args.teacher32b_path))
    return specs


def _load_reused_teacher_rows(
    path: Path | None,
    *,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if path is None:
        return {
            "rows": [],
            "summary": {
                "enabled": False,
                "source_path": None,
                "num_reused_rows": 0,
            },
        }

    rows = read_jsonl(path)
    teacher_rows_by_id: dict[str, dict[str, Any]] = {}
    selected_ids = [sample["id"] for sample in samples]
    selected_id_set = set(selected_ids)
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} is not a JSON object.")
        if row.get("model_role") != "teacher32b":
            continue
        sample_id = row.get("id")
        if sample_id is None:
            raise ValueError(f"{path}:{index} teacher32b row is missing 'id'.")
        sample_id = str(sample_id)
        if sample_id not in selected_id_set:
            continue
        if sample_id in teacher_rows_by_id:
            raise ValueError(f"{path} contains duplicate teacher32b rows for id={sample_id}.")
        teacher_rows_by_id[sample_id] = row

    selected_rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for sample in samples:
        sample_id = sample["id"]
        row = teacher_rows_by_id.get(sample_id)
        if row is None:
            missing_ids.append(sample_id)
            continue
        if str(row.get("image")) != sample["image"]:
            raise ValueError(
                f"Reused teacher row image mismatch for id={sample_id}: "
                f"{row.get('image')} != {sample['image']}"
            )
        parsed = row.get("parsed")
        if not isinstance(parsed, dict):
            raise ValueError(f"Reused teacher row for id={sample_id} is missing parsed object.")
        elements = parsed.get("elements")
        num_elements = (
            sum(1 for element in elements if isinstance(element, dict))
            if isinstance(elements, list)
            else 0
        )
        if num_elements <= 0:
            raise ValueError(
                f"Reused teacher row for id={sample_id} has empty parsed.elements."
            )
        selected_rows.append(row)

    if missing_ids:
        raise ValueError(
            f"Reused teacher rows do not cover all selected samples; missing ids: {missing_ids}"
        )
    if len(selected_rows) != len(samples):
        raise ValueError(
            "Number of reused teacher rows does not match selected input samples: "
            f"{len(selected_rows)} != {len(samples)}"
        )

    return {
        "rows": selected_rows,
        "summary": {
            "enabled": True,
            "source_path": str(path),
            "num_reused_rows": len(selected_rows),
        },
    }


def _load_model(spec: ModelRunSpec, args: argparse.Namespace) -> LoadedInferenceModel:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover - fallback for older transformers
        from transformers import AutoModelForVision2Seq as AutoModelForVLM

    model_path = resolve_model_path(spec.model_path)
    adapter_path = _resolve_adapter_path(spec.adapter_path)
    requested_device_map = resolve_requested_device_map(
        "auto",
        quantization="4bit" if args.load_in_4bit else "none",
        role=spec.role,
    )
    torch_dtype = _resolve_torch_dtype(args.torch_dtype)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model_kwargs: dict[str, Any] = {
        "device_map": requested_device_map,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    apply_attn_implementation(model_kwargs, "sdpa")
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    print(f"Loading {spec.role} model from {model_path} with 4bit={args.load_in_4bit}")
    model = AutoModelForVLM.from_pretrained(model_path, **model_kwargs)
    if adapter_path is not None:
        _validate_adapter_path(adapter_path)
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install peft to evaluate a LoRA-distilled model.") from exc
        print(f"Loading {spec.role} adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            local_files_only=True,
        )
    model.eval()

    input_device = select_model_input_device(
        model,
        preferred_modules=(getattr(model, "visual", None),),
        label=spec.role,
    )
    print_stage_model_debug(
        stage_label=spec.role,
        model_path=model_path,
        quantization_mode="4bit" if args.load_in_4bit else args.torch_dtype,
        requested_device_map=requested_device_map,
        model=model,
        selected_input_device=input_device,
    )
    ensure_stage_uses_cuda(
        stage_label=spec.role,
        requested_device_map=requested_device_map,
        model=model,
        selected_input_device=input_device,
    )
    return LoadedInferenceModel(
        role=spec.role,
        model_path=model_path,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
        processor=processor,
        model=model,
        input_device=input_device,
    )


def _resolve_torch_dtype(value: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[value]


def _resolve_adapter_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {candidate}")
    return candidate.resolve()


def _validate_adapter_path(adapter_path: Path) -> None:
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Adapter path is missing adapter_config.json: {adapter_config}"
        )


def _run_model_on_samples(
    *,
    loaded: LoadedInferenceModel,
    samples: list[dict[str, Any]],
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(
            f"[{loaded.role}] sample {index}/{len(samples)} "
            f"id={sample['id']} image={sample['image']}"
        )
        row = _infer_single_sample(
            loaded=loaded,
            sample=sample,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        rows.append(row)
    return rows


def _infer_single_sample(
    *,
    loaded: LoadedInferenceModel,
    sample: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    raw_output = ""
    error: str | None = None
    try:
        raw_output = _generate_for_image(
            processor=loaded.processor,
            model=loaded.model,
            input_device=loaded.input_device,
            image_path=Path(sample["image"]),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        parsed = parse_model_output(raw_output)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        parsed = {"elements": []}
    row: dict[str, Any] = {
        "id": sample["id"],
        "image": sample["image"],
        "model_role": loaded.role,
        "model_path": loaded.model_path,
        "adapter_path": loaded.adapter_path,
        "raw_output": raw_output,
        "parsed": parsed,
        "bbox_eval_against_teacher32b": None,
    }
    if error is not None:
        row["error"] = error
    return row


def _generate_for_image(
    *,
    processor,
    model,
    input_device,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = _apply_chat_template(processor, messages)
    inputs = _build_model_inputs(
        processor=processor,
        messages=messages,
        chat_text=chat_text,
        image_path=image_path,
    )
    inputs = inputs.to(input_device)
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    if no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    output_ids = model.generate(**generate_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer.strip()


def _apply_chat_template(processor, messages: list[dict[str, Any]]) -> str:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return str(messages[0]["content"][1]["text"])
    return apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_model_inputs(*, processor, messages: list[dict[str, Any]], chat_text: str, image_path: Path):
    qwen_images = None
    qwen_videos = None
    try:
        from qwen_vl_utils import process_vision_info

        qwen_images, qwen_videos = process_vision_info(messages)
    except ImportError:
        qwen_images, qwen_videos = None, None

    if qwen_images is not None or qwen_videos is not None:
        processor_kwargs: dict[str, Any] = {
            "text": [chat_text],
            "return_tensors": "pt",
        }
        if qwen_images is not None:
            processor_kwargs["images"] = qwen_images
        if qwen_videos is not None:
            processor_kwargs["videos"] = qwen_videos
        try:
            return processor(**processor_kwargs)
        except TypeError:
            processor_kwargs.pop("videos", None)
            return processor(**processor_kwargs)

    image = _load_image(image_path)
    try:
        return processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        )
    except TypeError:
        return processor(
            images=[image],
            text=[chat_text],
            return_tensors="pt",
        )


def _load_image(image_path: Path):
    from vlm_distill.stage_teacher_precompute import _load_teacher_image

    return _load_teacher_image(image_path, "original")


def parse_model_output(raw_output: str) -> dict[str, Any]:
    text = (raw_output or "").strip()
    if not text:
        return {"elements": [], "parse_error": "empty_output"}

    candidate, candidate_error = _extract_json_candidate(text)
    if candidate_error is not None:
        return {"elements": [], "parse_error": candidate_error}
    if candidate is None:
        return {"elements": [], "parse_error": "could_not_find_json_payload"}

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"elements": [], "parse_error": f"json_decode_error: {exc}"}

    has_top_level_named_elements_list = False
    if isinstance(payload, list):
        elements = payload
        schema = "legacy_list"
    elif isinstance(payload, dict):
        if isinstance(payload.get("e"), list):
            elements = payload.get("e")
            schema = "compact"
            has_top_level_named_elements_list = True
        else:
            elements = payload.get("elements")
            schema = "legacy_object"
            has_top_level_named_elements_list = isinstance(elements, list)
    else:
        return {"elements": [], "parse_error": "top_level_json_is_not_object_or_array"}

    if not isinstance(elements, list):
        return {"elements": [], "parse_error": "missing_or_invalid_elements_list"}

    normalized_elements: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        normalized = (
            _normalize_compact_element(index, element)
            if schema == "compact"
            else _normalize_legacy_element(index, element)
        )
        if normalized is not None:
            normalized_elements.append(normalized)
    result: dict[str, Any] = {"elements": normalized_elements}
    if has_top_level_named_elements_list and elements and not normalized_elements:
        result["parse_error"] = "all_elements_invalid_after_normalization"
    elif not normalized_elements:
        result["parse_error"] = "empty_elements_from_nonempty_output"
    return result


def _normalize_compact_element(index: int, element: Any) -> dict[str, Any] | None:
    if not isinstance(element, (list, tuple)):
        return None
    if len(element) == 2:
        name = _safe_string(element[0])
        bbox = element[1]
    elif len(element) == 5:
        name = _safe_string(element[0])
        bbox = element[1:5]
    else:
        return None
    return {
        "element_index": index,
        "name": name,
        "name_norm": normalize_text(name),
        "type": None,
        "bbox": _normalize_bbox(bbox),
        "confidence": None,
    }


def _normalize_legacy_element(index: int, element: Any) -> dict[str, Any] | None:
    if not isinstance(element, dict):
        return None
    name = _safe_string(element.get("name"))
    return {
        "element_index": index,
        "name": name,
        "name_norm": normalize_text(name),
        "type": _safe_string(element.get("type")),
        "bbox": _normalize_bbox(element.get("bbox")),
        "confidence": _normalize_confidence(element.get("confidence")),
    }


def _extract_json_candidate(text: str) -> tuple[str | None, str | None]:
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        candidate, error = _extract_json_candidate_from_source(block)
        if candidate is not None or error is not None:
            return candidate, error
    return _extract_json_candidate_from_source(text)


def _extract_json_candidate_from_source(text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, None
    if _looks_like_named_top_level_object_payload(stripped):
        return _decode_json_value_at_start(stripped, "top_level")
    if stripped.startswith("["):
        return _decode_json_value_at_start(stripped, "top_level")
    return _extract_first_json_value(stripped), None


def _extract_first_json_value(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            obj, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, (dict, list)):
            return text[match.start(): match.start() + end]
    return None


def _looks_like_named_top_level_object_payload(text: str) -> bool:
    return bool(re.match(r'^\{\s*"(?:e|elements)"\s*:', text))


def _decode_json_value_at_start(text: str, context: str) -> tuple[str | None, str | None]:
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        if context == "top_level" and text.startswith("{"):
            message = f"json_decode_error_top_level: {exc}"
            if re.match(r'^\{\s*"(?:e|elements)"\s*:', text):
                return None, message or "truncated_or_malformed_top_level_json"
            return None, message
        return None, f"json_decode_error: {exc}"
    if not isinstance(obj, (dict, list)):
        return None, "top_level_json_is_not_object_or_array"
    return text[:end], None


def _safe_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        parts = [part.strip() for part in value.strip().split(",")]
        if len(parts) != 4:
            return None
        raw_items: list[Any] = parts
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        raw_items = list(value)
    else:
        return None
    bbox: list[float] = []
    for item in raw_items:
        numeric = _normalize_float(item)
        if numeric is None:
            return None
        bbox.append(numeric)
    x1, y1, x2, y2 = bbox
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return [x1, y1, x2, y2]


def _normalize_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower().strip()
    text = re.sub(r"[\s_-]+", " ", text)
    kept_chars: list[str] = []
    for char in text:
        if char.isalnum() or _is_cjk(char) or char.isspace():
            kept_chars.append(char)
            continue
        if unicodedata.category(char).startswith(("P", "S")):
            continue
    normalized = "".join(kept_chars)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def name_similarity(a: str | None, b: str | None) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.9
    return max(_token_jaccard(left, right), _character_f1(left, right))


def _token_jaccard(a: str, b: str) -> float:
    left = set(token for token in a.split() if token)
    right = set(token for token in b.split() if token)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _character_f1(a: str, b: str) -> float:
    left_chars = [char for char in a if not char.isspace()]
    right_chars = [char for char in b if not char.isspace()]
    if not left_chars or not right_chars:
        return 0.0
    left_counter = Counter(left_chars)
    right_counter = Counter(right_chars)
    common = sum((left_counter & right_counter).values())
    if common <= 0:
        return 0.0
    precision = common / len(left_chars)
    recall = common / len(right_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_comparison_rows(
    *,
    raw_rows_by_role: dict[str, list[dict[str, Any]]],
    match_threshold: float,
) -> list[dict[str, Any]]:
    rows_by_role_and_id: dict[str, dict[str, dict[str, Any]]] = {
        role: {row["id"]: row for row in rows}
        for role, rows in raw_rows_by_role.items()
    }
    comparison_rows: list[dict[str, Any]] = []
    teacher_rows = raw_rows_by_role.get("teacher32b", [])

    for teacher_row in teacher_rows:
        reference_elements = _extract_elements(teacher_row)
        for candidate_role in COMPARISON_ROLES:
            candidate_row = rows_by_role_and_id.get(candidate_role, {}).get(teacher_row["id"])
            candidate_elements = _extract_elements(candidate_row)
            comparison_rows.extend(
                _compare_single_sample(
                    image=teacher_row["image"],
                    sample_id=teacher_row["id"],
                    reference_elements=reference_elements,
                    candidate_elements=candidate_elements,
                    candidate_role=candidate_role,
                    match_threshold=match_threshold,
                )
            )
    return comparison_rows


def _extract_elements(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not row:
        return []
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        return []
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        return []
    return [element for element in elements if isinstance(element, dict)]


def _compare_single_sample(
    *,
    image: str,
    sample_id: str,
    reference_elements: list[dict[str, Any]],
    candidate_elements: list[dict[str, Any]],
    candidate_role: str,
    match_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unused_candidate_indices = set(range(len(candidate_elements)))

    for reference_element in reference_elements:
        best_index = None
        best_similarity = -1.0
        for candidate_index in unused_candidate_indices:
            similarity = name_similarity(
                reference_element.get("name"),
                candidate_elements[candidate_index].get("name"),
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = candidate_index

        best_candidate = candidate_elements[best_index] if best_index is not None else None
        matched = best_index is not None and best_similarity >= match_threshold
        candidate_element = best_candidate if matched else None
        candidate_name = candidate_element.get("name") if candidate_element is not None else None
        if matched and best_index is not None:
            unused_candidate_indices.remove(best_index)

        rows.append(
            {
                "id": sample_id,
                "image": image,
                "reference_role": "teacher32b",
                "candidate_role": candidate_role,
                "reference_element": reference_element,
                "reference_name": reference_element.get("name"),
                "matched": matched,
                "candidate_element": candidate_element,
                "candidate_name": candidate_name,
                "best_candidate_even_if_unmatched": best_candidate,
                "best_candidate_name_even_if_unmatched": (
                    best_candidate.get("name") if best_candidate is not None else None
                ),
                "name_similarity": max(best_similarity, 0.0),
                "match_threshold": match_threshold,
                "bbox_metrics": _build_bbox_metrics(
                    reference_element=reference_element,
                    candidate_element=candidate_element,
                    matched=matched,
                ),
            }
        )
    return rows


def build_summary(
    *,
    raw_rows_by_role: dict[str, list[dict[str, Any]]],
    comparison_rows: list[dict[str, Any]],
    match_threshold: float,
    teacher_reuse: dict[str, Any],
    generation_settings: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    role_stats: dict[str, dict[str, Any]] = {}

    for role in COMPARISON_ROLES:
        rows = [row for row in comparison_rows if row["candidate_role"] == role]
        num_reference_elements = len(rows)
        num_matched_elements = sum(1 for row in rows if row["matched"])
        best_scores = [float(row["name_similarity"]) for row in rows]
        matched_scores = [float(row["name_similarity"]) for row in rows if row["matched"]]
        role_stats[role] = {
            "num_reference_elements": num_reference_elements,
            "num_matched_elements": num_matched_elements,
            "element_recall_against_32b": (
                num_matched_elements / num_reference_elements if num_reference_elements else 0.0
            ),
            "mean_best_name_similarity_all_reference": (
                sum(best_scores) / len(best_scores) if best_scores else 0.0
            ),
            "mean_name_similarity_matched_only": (
                sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
            ),
            "bbox_metrics": _aggregate_bbox_metrics(rows),
        }
        summary[role] = role_stats[role]

    summary["parse_stats"] = _build_parse_stats(raw_rows_by_role)
    summary["bbox_parse_stats"] = _build_bbox_parse_stats(raw_rows_by_role)
    summary["teacher_reuse"] = teacher_reuse
    summary["generation_settings"] = generation_settings

    base = role_stats.get("base8b", {})
    distill = role_stats.get("distill32to8", {})
    base_bbox = base.get("bbox_metrics") if isinstance(base, dict) else {}
    distill_bbox = distill.get("bbox_metrics") if isinstance(distill, dict) else {}
    summary["distill_vs_base8b"] = {
        "recall_delta": float(distill.get("element_recall_against_32b", 0.0))
        - float(base.get("element_recall_against_32b", 0.0)),
        "mean_best_name_similarity_delta": float(
            distill.get("mean_best_name_similarity_all_reference", 0.0)
        )
        - float(base.get("mean_best_name_similarity_all_reference", 0.0)),
        "mean_matched_name_similarity_delta": float(
            distill.get("mean_name_similarity_matched_only", 0.0)
        )
        - float(base.get("mean_name_similarity_matched_only", 0.0)),
        "bbox_metric_delta": {
            "mean_center_error_px_delta": _metric_delta(
                distill_bbox, base_bbox, "mean_center_error_px"
            ),
            "mean_area_error_ratio_delta": _metric_delta(
                distill_bbox, base_bbox, "mean_area_error_ratio"
            ),
            "mean_bbox_l1_error_delta": _metric_delta(
                distill_bbox, base_bbox, "mean_bbox_l1_error"
            ),
            "mean_bbox_iou_delta": _metric_delta(
                distill_bbox, base_bbox, "mean_bbox_iou"
            ),
        },
    }
    summary["notes"] = {
        "reference": "teacher32b element names are used as reference.",
        "matched": "A candidate element is considered matched if name_similarity >= match_threshold.",
        "name_similarity": (
            "Exact match = 1.0, substring match = 0.9, otherwise "
            "max(token_jaccard, character_f1)."
        ),
        "fallback": "Only name fallback is used. Type, bbox, IoU, and center distance are not used.",
        "bbox_metrics": (
            "BBox metrics are computed only after a name-based match is made and never affect "
            "matched=true/false."
        ),
        "bbox_metric_delta": (
            "For center_error_px, area_error_ratio, and bbox_l1_error, lower is better so a "
            "negative distill_vs_base8b delta means improvement. For bbox_iou, higher is better "
            "so a positive delta means improvement."
        ),
        "match_threshold": match_threshold,
    }
    warnings: list[str] = []
    teacher_parse_stats = summary["parse_stats"].get("teacher32b", {})
    if int(teacher_parse_stats.get("total_parsed_elements", 0)) == 0:
        warnings.append(
            "teacher32b parsed zero reference elements, so comparison metrics are invalid."
        )
    if warnings:
        summary["warnings"] = warnings
    return summary


def _build_parse_stats(raw_rows_by_role: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for role in MODEL_ROLE_ORDER:
        rows = raw_rows_by_role.get(role, [])
        parse_errors = Counter()
        total_parsed_elements = 0
        for row in rows:
            parsed = row.get("parsed")
            if not isinstance(parsed, dict):
                parse_errors["missing_parsed_object"] += 1
                continue
            parse_error = parsed.get("parse_error")
            if parse_error:
                parse_errors[str(parse_error)] += 1
            elements = parsed.get("elements")
            if isinstance(elements, list):
                total_parsed_elements += sum(1 for element in elements if isinstance(element, dict))

        num_rows = len(rows)
        num_parse_errors = sum(parse_errors.values())
        stats[role] = {
            "num_rows": num_rows,
            "num_parse_errors": num_parse_errors,
            "parse_error_rate": (num_parse_errors / num_rows if num_rows else 0.0),
            "total_parsed_elements": total_parsed_elements,
            "mean_parsed_elements_per_row": (
                total_parsed_elements / num_rows if num_rows else 0.0
            ),
            "parse_error_counts": dict(parse_errors),
        }
    return stats


def _build_bbox_parse_stats(
    raw_rows_by_role: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for role in MODEL_ROLE_ORDER:
        rows = raw_rows_by_role.get(role, [])
        total_elements = 0
        elements_with_valid_bbox = 0
        for row in rows:
            parsed = row.get("parsed")
            if not isinstance(parsed, dict):
                continue
            elements = parsed.get("elements")
            if not isinstance(elements, list):
                continue
            for element in elements:
                if not isinstance(element, dict):
                    continue
                total_elements += 1
                if _normalize_bbox(element.get("bbox")) is not None:
                    elements_with_valid_bbox += 1
        elements_with_missing_or_invalid_bbox = total_elements - elements_with_valid_bbox
        stats[role] = {
            "total_elements": total_elements,
            "elements_with_valid_bbox": elements_with_valid_bbox,
            "valid_bbox_rate": (
                elements_with_valid_bbox / total_elements if total_elements else 0.0
            ),
            "elements_with_missing_or_invalid_bbox": elements_with_missing_or_invalid_bbox,
        }
    return stats


def _build_bbox_metrics(
    *,
    reference_element: dict[str, Any],
    candidate_element: dict[str, Any] | None,
    matched: bool,
) -> dict[str, Any]:
    reference_bbox = _normalize_bbox(reference_element.get("bbox"))
    candidate_bbox = _normalize_bbox(candidate_element.get("bbox")) if candidate_element else None
    if not matched:
        return _empty_bbox_metrics(
            reason="unmatched",
            reference_bbox=reference_bbox,
            candidate_bbox=candidate_bbox,
        )
    if reference_bbox is None:
        return _empty_bbox_metrics(
            reason="missing_or_invalid_reference_bbox",
            reference_bbox=None,
            candidate_bbox=candidate_bbox,
        )
    if candidate_bbox is None:
        return _empty_bbox_metrics(
            reason="missing_or_invalid_candidate_bbox",
            reference_bbox=reference_bbox,
            candidate_bbox=None,
        )

    ref_x1, ref_y1, ref_x2, ref_y2 = reference_bbox
    cand_x1, cand_y1, cand_x2, cand_y2 = candidate_bbox
    ref_cx = (ref_x1 + ref_x2) / 2.0
    ref_cy = (ref_y1 + ref_y2) / 2.0
    cand_cx = (cand_x1 + cand_x2) / 2.0
    cand_cy = (cand_y1 + cand_y2) / 2.0
    dx = cand_cx - ref_cx
    dy = cand_cy - ref_cy
    reference_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
    candidate_area = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)
    area_delta = candidate_area - reference_area
    return {
        "valid": True,
        "reference_bbox": reference_bbox,
        "candidate_bbox": candidate_bbox,
        "reference_center": [ref_cx, ref_cy],
        "candidate_center": [cand_cx, cand_cy],
        "center_offset_dx": dx,
        "center_offset_dy": dy,
        "center_error_px": (dx * dx + dy * dy) ** 0.5,
        "reference_area": reference_area,
        "candidate_area": candidate_area,
        "area_error_abs": abs(area_delta),
        "area_error_ratio": abs(area_delta) / reference_area,
        "signed_area_error_ratio": area_delta / reference_area,
        "bbox_l1_error": (
            abs(cand_x1 - ref_x1)
            + abs(cand_y1 - ref_y1)
            + abs(cand_x2 - ref_x2)
            + abs(cand_y2 - ref_y2)
        )
        / 4.0,
        "bbox_iou": _compute_bbox_iou(reference_bbox, candidate_bbox),
    }


def _empty_bbox_metrics(
    *,
    reason: str,
    reference_bbox: list[float] | None,
    candidate_bbox: list[float] | None,
) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "reference_bbox": reference_bbox,
        "candidate_bbox": candidate_bbox,
        "reference_center": None,
        "candidate_center": None,
        "center_offset_dx": None,
        "center_offset_dy": None,
        "center_error_px": None,
        "reference_area": None,
        "candidate_area": None,
        "area_error_abs": None,
        "area_error_ratio": None,
        "signed_area_error_ratio": None,
        "bbox_l1_error": None,
        "bbox_iou": None,
    }


def _compute_bbox_iou(reference_bbox: list[float], candidate_bbox: list[float]) -> float:
    ref_x1, ref_y1, ref_x2, ref_y2 = reference_bbox
    cand_x1, cand_y1, cand_x2, cand_y2 = candidate_bbox
    inter_x1 = max(ref_x1, cand_x1)
    inter_y1 = max(ref_y1, cand_y1)
    inter_x2 = min(ref_x2, cand_x2)
    inter_y2 = min(ref_y2, cand_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    reference_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
    candidate_area = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)
    union = reference_area + candidate_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _aggregate_bbox_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows = [row for row in rows if row.get("matched")]
    valid_metrics = [
        metrics
        for row in matched_rows
        for metrics in [row.get("bbox_metrics")]
        if isinstance(metrics, dict) and metrics.get("valid") is True
    ]
    return {
        "num_matched_pairs": len(matched_rows),
        "num_valid_bbox_pairs": len(valid_metrics),
        "valid_bbox_rate_among_matched": (
            len(valid_metrics) / len(matched_rows) if matched_rows else 0.0
        ),
        "mean_center_error_px": _mean_metric(valid_metrics, "center_error_px"),
        "mean_abs_center_offset_dx": _mean_abs_metric(valid_metrics, "center_offset_dx"),
        "mean_abs_center_offset_dy": _mean_abs_metric(valid_metrics, "center_offset_dy"),
        "mean_area_error_abs": _mean_metric(valid_metrics, "area_error_abs"),
        "mean_area_error_ratio": _mean_metric(valid_metrics, "area_error_ratio"),
        "mean_signed_area_error_ratio": _mean_metric(valid_metrics, "signed_area_error_ratio"),
        "mean_bbox_l1_error": _mean_metric(valid_metrics, "bbox_l1_error"),
        "mean_bbox_iou": _mean_metric(valid_metrics, "bbox_iou"),
    }


def _mean_metric(metrics_list: list[dict[str, Any]], key: str) -> float | None:
    values = [float(metrics[key]) for metrics in metrics_list if metrics.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _mean_abs_metric(metrics_list: list[dict[str, Any]], key: str) -> float | None:
    values = [abs(float(metrics[key])) for metrics in metrics_list if metrics.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _attach_bbox_eval_to_raw_rows(
    *,
    raw_rows_by_role: dict[str, list[dict[str, Any]]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    comparison_by_role_and_id: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {} for role in COMPARISON_ROLES
    }
    for row in comparison_rows:
        role = row.get("candidate_role")
        sample_id = row.get("id")
        if role not in comparison_by_role_and_id or not isinstance(sample_id, str):
            continue
        comparison_by_role_and_id[role].setdefault(sample_id, []).append(row)

    for row in raw_rows_by_role.get("teacher32b", []):
        row["bbox_eval_against_teacher32b"] = None

    for role in COMPARISON_ROLES:
        for row in raw_rows_by_role.get(role, []):
            sample_rows = comparison_by_role_and_id.get(role, {}).get(row["id"], [])
            row["bbox_eval_against_teacher32b"] = _aggregate_bbox_metrics(sample_rows)


def _flatten_raw_rows(raw_rows_by_role: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in MODEL_ROLE_ORDER:
        rows.extend(raw_rows_by_role.get(role, []))
    return rows


def _metric_delta(left_metrics: Any, right_metrics: Any, key: str) -> float | None:
    if not isinstance(left_metrics, dict) or not isinstance(right_metrics, dict):
        return None
    left_value = left_metrics.get(key)
    right_value = right_metrics.get(key)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _release_model(loaded: LoadedInferenceModel | None) -> None:
    if loaded is None:
        return
    try:
        import torch
    except ImportError:
        torch = None

    model = loaded.model
    processor = loaded.processor
    del model
    del processor
    del loaded.model
    del loaded.processor
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()

# Usage examples
#
# Merged distilled model:
# python scripts/eval_three_model_element_listing.py \
#   --input_jsonl data/eval_images.jsonl \
#   --output_jsonl outputs/eval_list_elements_raw.jsonl \
#   --comparison_output_jsonl outputs/eval_list_elements_comparison.jsonl \
#   --summary_output_json outputs/eval_list_elements_summary.json \
#   --teacher32b_path /mnt/nvme0/vlm_distill/models/Qwen3-VL-32B-Instruct \
#   --base8b_path /mnt/nvme0/vlm_distill/models/Qwen3-VL-8B-Instruct \
#   --distill_model_path /mnt/nvme0/vlm_distill/outputs/qwen3vl-32to8-merged \
#   --torch_dtype bfloat16 \
#   --load_in_4bit \
#   --match_threshold 0.70
#
# LoRA adapter:
# python scripts/eval_three_model_element_listing.py \
#   --input_jsonl data/eval_images.jsonl \
#   --output_jsonl outputs/eval_list_elements_raw_lora.jsonl \
#   --comparison_output_jsonl outputs/eval_list_elements_comparison_lora.jsonl \
#   --summary_output_json outputs/eval_list_elements_summary_lora.json \
#   --teacher32b_path /mnt/nvme0/vlm_distill/models/Qwen3-VL-32B-Instruct \
#   --base8b_path /mnt/nvme0/vlm_distill/models/Qwen3-VL-8B-Instruct \
#   --distill_model_path /mnt/nvme0/vlm_distill/models/Qwen3-VL-8B-Instruct \
#   --distill_adapter_path /mnt/nvme0/vlm_distill/outputs/student/adapter \
#   --torch_dtype bfloat16 \
#   --load_in_4bit \
#   --match_threshold 0.70
