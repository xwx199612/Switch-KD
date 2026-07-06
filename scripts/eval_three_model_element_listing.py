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


PROMPT = """You are evaluating Android TV / Smart TV UI understanding.

Task:
List all visible interactive UI elements on this screen.

Requirements:
Return ONLY valid JSON.
Do not include markdown.
Do not include explanation.

JSON schema:
{
  "elements": [
    {
      "name": "visible text or short visual description",
      "type": "app_icon | button | menu_item | tab | input | setting | navigation | other",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }
  ]
}

Coordinate rules:
- Use pixel coordinates relative to the original image.
- x1,y1 is top-left. x2,y2 is bottom-right.
- If you are not sure about an element bbox, still estimate it.
- Include all visible clickable or focusable UI elements.
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
    parser.add_argument("--base8b_path", required=True)
    parser.add_argument("--distill_model_path", required=True)
    parser.add_argument("--distill_adapter_path")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
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

    raw_rows_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in MODEL_ROLE_ORDER}
    for spec in model_specs:
        loaded = _load_model(spec, args)
        try:
            rows = _run_model_on_samples(
                loaded=loaded,
                samples=samples,
                output_jsonl=args.output_jsonl,
                max_new_tokens=args.max_new_tokens,
            )
            raw_rows_by_role[spec.role] = rows
        finally:
            _release_model(loaded)

    comparison_rows = build_comparison_rows(
        raw_rows_by_role=raw_rows_by_role,
        match_threshold=args.match_threshold,
    )
    write_jsonl(args.comparison_output_jsonl, comparison_rows)

    summary = build_summary(
        comparison_rows=comparison_rows,
        match_threshold=args.match_threshold,
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
    return [
        ModelRunSpec(role="teacher32b", model_path=args.teacher32b_path),
        ModelRunSpec(role="base8b", model_path=args.base8b_path),
        ModelRunSpec(
            role="distill32to8",
            model_path=args.distill_model_path,
            adapter_path=args.distill_adapter_path,
        ),
    ]


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
        compute_dtype = _resolve_torch_dtype(args.torch_dtype)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = _resolve_torch_dtype(args.torch_dtype)

    print(f"Loading {spec.role} model from {model_path}")
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
    output_jsonl: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            print(
                f"[{loaded.role}] sample {index}/{len(samples)} "
                f"id={sample['id']} image={sample['image']}"
            )
            row = _infer_single_sample(
                loaded=loaded,
                sample=sample,
                max_new_tokens=max_new_tokens,
            )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _infer_single_sample(
    *,
    loaded: LoadedInferenceModel,
    sample: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    raw_output = ""
    error: str | None = None
    try:
        raw_output = _generate_for_image(
            processor=loaded.processor,
            model=loaded.model,
            input_device=loaded.input_device,
            image_path=Path(sample["image"]),
            prompt=PROMPT,
            max_new_tokens=max_new_tokens,
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
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
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

    candidate = _extract_json_candidate(text)
    if candidate is None:
        return {"elements": [], "parse_error": "could_not_find_json_object"}

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"elements": [], "parse_error": f"json_decode_error: {exc}"}

    if not isinstance(payload, dict):
        return {"elements": [], "parse_error": "top_level_json_is_not_object"}

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return {"elements": [], "parse_error": "missing_or_invalid_elements_list"}

    normalized_elements: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        name = _safe_string(element.get("name"))
        normalized_elements.append(
            {
                "element_index": index,
                "name": name,
                "name_norm": normalize_text(name),
                "type": _safe_string(element.get("type")),
                "bbox": _normalize_bbox(element.get("bbox")),
                "confidence": _normalize_confidence(element.get("confidence")),
            }
        )
    result: dict[str, Any] = {"elements": normalized_elements}
    return result


def _extract_json_candidate(text: str) -> str | None:
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        candidate = _extract_first_json_object(block)
        if candidate is not None:
            return candidate
    return _extract_first_json_object(text)


def _extract_first_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return text[match.start(): match.start() + end]
    return None


def _safe_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    bbox: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        bbox.append(float(item))
    return bbox


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
            }
        )
    return rows


def build_summary(*, comparison_rows: list[dict[str, Any]], match_threshold: float) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    role_stats: dict[str, dict[str, float | int]] = {}

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
        }
        summary[role] = role_stats[role]

    base = role_stats.get("base8b", {})
    distill = role_stats.get("distill32to8", {})
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
    }
    summary["notes"] = {
        "reference": "teacher32b element names are used as reference.",
        "matched": "A candidate element is considered matched if name_similarity >= match_threshold.",
        "name_similarity": (
            "Exact match = 1.0, substring match = 0.9, otherwise "
            "max(token_jaccard, character_f1)."
        ),
        "fallback": "Only name fallback is used. Type, bbox, IoU, and center distance are not used.",
        "match_threshold": match_threshold,
    }
    return summary


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
#   --teacher32b_path /home/phison/vlm_distill/models/Qwen3-VL-32B-Instruct \
#   --base8b_path /home/phison/vlm_distill/models/Qwen3-VL-8B-Instruct \
#   --distill_model_path /home/phison/vlm_distill/outputs/qwen3vl-32to8-merged \
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
#   --teacher32b_path /home/phison/vlm_distill/models/Qwen3-VL-32B-Instruct \
#   --base8b_path /home/phison/vlm_distill/models/Qwen3-VL-8B-Instruct \
#   --distill_model_path /home/phison/vlm_distill/models/Qwen3-VL-8B-Instruct \
#   --distill_adapter_path /home/phison/vlm_distill/outputs/student/adapter \
#   --torch_dtype bfloat16 \
#   --load_in_4bit \
#   --match_threshold 0.70
