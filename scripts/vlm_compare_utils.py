from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


MODEL_SPECS = (
    ("qwen3vl_32b", "model_32b"),
    ("qwen3vl_8b", "model_8b"),
    ("distilled_32to8b", "model_distilled"),
)
QUANTIZATION_CHOICES = ("none", "4bit", "8bit", "mixed_4bit_bf16")
A1_MIXED_MERGER_PATHS = [
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
]

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|[A-Za-z][.)])\s*")
_OBJECT_EXPLANATION_PREFIXES = (
    "here are",
    "objects:",
    "object names:",
    "visible objects:",
    "visible ui elements:",
    "ui elements:",
    "elements:",
    "output:",
    "answer:",
    "explanation:",
    "note:",
)
_BOOL_TRUE_VALUES = {"true", "yes", "y", "focused", "1"}
_BOOL_FALSE_VALUES = {"false", "no", "n", "unfocused", "0"}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_model_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_")
    return normalized or "model"


def list_images(image_dir: str | Path, extensions: list[str]) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")

    normalized_exts = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (item.strip().lower() for item in extensions)
        if ext
    }
    images = [
        path for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in normalized_exts
    ]
    if not images:
        raise FileNotFoundError(
            f"No images found in {root} with extensions: {sorted(normalized_exts)}"
        )
    return images


def resolve_torch_dtype(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_processor_and_model(
    model_path: str,
    torch_dtype: str,
    device_map: str,
    quantization: str,
):
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

    resolved_torch_dtype = resolve_torch_dtype(torch_dtype)
    quantization_config = None
    effective_device_map = device_map
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=resolved_torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        effective_device_map = "auto"
    elif quantization == "mixed_4bit_bf16":
        from vlm_distill.mixed_precision import (
            build_mixed_precision_quantization_config,
        )

        quantization_config = build_mixed_precision_quantization_config(
            quantization="4bit",
            excluded_module_paths=A1_MIXED_MERGER_PATHS,
        )
        effective_device_map = "auto"
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        effective_device_map = "auto"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map=effective_device_map,
        torch_dtype=resolved_torch_dtype,
        quantization_config=quantization_config,
    )
    model.eval()
    if quantization == "mixed_4bit_bf16":
        _validate_and_print_mixed_model(model)
    return processor, model


def _validate_and_print_mixed_model(model) -> None:
    """Validate and summarize the A1 4-bit language/BF16 merger contract."""
    import torch

    try:
        import bitsandbytes as bnb
        linear4bit_type = bnb.nn.Linear4bit
        quantized_types = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "mixed_4bit_bf16 requires bitsandbytes so the loaded module classes can be validated."
        ) from exc

    language_model_linear4bit_count = sum(
        isinstance(module, linear4bit_type)
        for name, module in model.named_modules()
        if "language_model" in name
    )
    try:
        merger = model.get_submodule("model.visual.merger")
        fc1 = model.get_submodule("model.visual.merger.linear_fc1")
        fc2 = model.get_submodule("model.visual.merger.linear_fc2")
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            "mixed_4bit_bf16 validation failed: expected model.visual.merger.linear_fc1 "
            "and model.visual.merger.linear_fc2."
        ) from exc

    merger_quantized = [
        name or "model.visual.merger"
        for name, module in merger.named_modules()
        if isinstance(module, quantized_types)
    ]
    errors = []
    if language_model_linear4bit_count <= 0:
        errors.append("language-model Linear4bit count must be > 0")
    for name, layer in (("linear_fc1", fc1), ("linear_fc2", fc2)):
        if not isinstance(layer, torch.nn.Linear) or layer.weight.dtype != torch.bfloat16:
            errors.append(
                f"model.visual.merger.{name} must be torch.nn.Linear with torch.bfloat16 "
                f"weights (got {type(layer).__module__}.{type(layer).__name__}/{layer.weight.dtype})"
            )
    if merger_quantized:
        errors.append(
            "no bitsandbytes quantized layer may exist under model.visual.merger "
            f"(found {', '.join(merger_quantized)})"
        )
    if errors:
        raise RuntimeError("mixed_4bit_bf16 validation failed: " + "; ".join(errors))

    def module_label(layer) -> str:
        return f"{type(layer).__module__}.{type(layer).__name__}/{layer.weight.dtype}"

    print("quantization_mode=mixed_4bit_bf16")
    print(f"language_model_linear4bit_count={language_model_linear4bit_count}")
    print(f"merger_linear_fc1={module_label(fc1)}")
    print(f"merger_linear_fc2={module_label(fc2)}")


def select_input_device(model):
    import torch

    device = getattr(model, "device", None)
    if device is not None:
        return device

    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return torch.device("cpu")


def build_qwen_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _move_inputs_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_vlm_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    # Kept as a compatibility API for comparison scripts; the implementation
    # lives in the package so src code never imports from scripts.
    from vlm_distill.bbox_grounding_inference import BBoxGroundingInferenceEngine
    return BBoxGroundingInferenceEngine(model, processor).generate_raw(
        image=image, prompt=prompt, max_new_tokens=max_new_tokens
    )


EXPECTED_TOP_LEVEL_KEYS = (
    "elements",
    "objects",
    "bounding_boxes",
    "detections",
    "predictions",
)


_BBOX_FIELD_NAMES = (
    "bbox",
    "box",
    "bbox_2d",
    "bounding_box",
    "coordinates",
)
_JSON_NUMBER_RE = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_BBOX_VALUE_RE = (
    rf"{_JSON_NUMBER_RE}\s*,\s*"
    rf"{_JSON_NUMBER_RE}\s*,\s*"
    rf"{_JSON_NUMBER_RE}\s*,\s*"
    rf"{_JSON_NUMBER_RE}"
)
_BBOX_FIELD_RE = "|".join(re.escape(field) for field in _BBOX_FIELD_NAMES)
_MISSING_BOTH_BBOX_BRACKETS_RE = re.compile(
    rf'("(?P<field>{_BBOX_FIELD_RE})"\s*:\s*)'
    rf'(?P<coords>{_BBOX_VALUE_RE})'
    rf'(?P<tail>\s*,\s*"[^"]+"\s*:)'
)
_MALFORMED_BBOX_KEY_RE = re.compile(
    rf'"(?P<field>{_BBOX_FIELD_RE})=(?P<quote>\")(?P<coords>{_BBOX_VALUE_RE})(?P<closing>\s*\])'
)
_MISSING_OPEN_BBOX_BRACKET_RE = re.compile(
    rf'("(?P<field>{_BBOX_FIELD_RE})"\s*:\s*)'
    rf'(?P<coords>{_BBOX_VALUE_RE})'
    rf'(?P<closing>\s*\])'
)
_MISSING_CLOSE_BBOX_BRACKET_RE = re.compile(
    rf'("(?P<field>{_BBOX_FIELD_RE})"\s*:\s*\[\s*)'
    rf'(?P<coords>{_BBOX_VALUE_RE})'
    rf'(?P<tail>\s*,\s*"[^"]+"\s*:)'
)


def repair_common_json_errors(text: str) -> tuple[str, list[str]]:
    repaired = text
    notes: list[str] = []

    repaired, count = _MALFORMED_BBOX_KEY_RE.subn(
        lambda match: (
            f'"{match.group("field")}":[{match.group("coords")}{match.group("closing")}'
        ),
        repaired,
    )
    if count:
        notes.append(f"repaired malformed bbox key/value ({count} occurrence(s))")

    repaired, count = _MISSING_BOTH_BBOX_BRACKETS_RE.subn(
        lambda match: (
            f'{match.group(1)}[{match.group("coords")}]{match.group("tail")}'
        ),
        repaired,
    )
    if count:
        notes.append(f"repaired missing bbox brackets ({count} occurrence(s))")

    repaired, count = _MISSING_OPEN_BBOX_BRACKET_RE.subn(
        lambda match: (
            f'{match.group(1)}[{match.group("coords")}{match.group("closing")}'
        ),
        repaired,
    )
    if count:
        notes.append(f"repaired missing bbox opening bracket ({count} occurrence(s))")

    repaired, count = _MISSING_CLOSE_BBOX_BRACKET_RE.subn(
        lambda match: (
            f'{match.group(1)}{match.group("coords")}]{match.group("tail")}'
        ),
        repaired,
    )
    if count:
        notes.append(f"repaired missing bbox closing bracket ({count} occurrence(s))")

    return repaired, notes


def _iter_top_level_json_starts(text: str):
    in_string = False
    escaped = False
    depth = 0

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in "{[":
            if depth == 0:
                yield index
            depth += 1
            continue
        if char in "}]":
            if depth > 0:
                depth -= 1


def _is_truncated_top_level_json(text: str, start_index: int) -> bool:
    opening = text[start_index]
    if opening not in "{[":
        return False

    expected_closing = "}" if opening == "{" else "]"
    in_string = False
    escaped = False
    depth = 0

    for char in text[start_index:]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
            continue
        if char == expected_closing:
            depth -= 1
            if depth == 0:
                return False

    return depth > 0


def _extract_json_from_text_once(
    text: str,
    *,
    json_repaired: bool = False,
    repair_notes: list[str] | None = None,
) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    found_candidate = False

    for start_index in _iter_top_level_json_starts(text):
        found_candidate = True
        try:
            parsed, _ = decoder.raw_decode(text[start_index:])
        except json.JSONDecodeError:
            if _is_truncated_top_level_json(text, start_index):
                raise ValueError(
                    "No complete top-level JSON object found. The model output may be truncated."
                ) from None
            continue

        if isinstance(parsed, list):
            result: dict[str, Any] = {"elements": parsed}
            if json_repaired:
                result["json_repaired"] = True
            if repair_notes:
                result["repair_notes"] = list(repair_notes)
            return result
        if isinstance(parsed, dict):
            if any(key in parsed for key in EXPECTED_TOP_LEVEL_KEYS):
                result = dict(parsed)
                if json_repaired:
                    result["json_repaired"] = True
                if repair_notes:
                    result["repair_notes"] = list(repair_notes)
                return result
            continue

    if found_candidate:
        raise ValueError(
            "No valid top-level JSON object found in model output. "
            "The output may be malformed or truncated."
        )
    raise ValueError(
        "No valid top-level JSON object found in model output. "
        "The output may be malformed or truncated."
    )


def extract_json_from_text(text: str) -> dict[str, Any]:
    try:
        return _extract_json_from_text_once(text)
    except ValueError as original_error:
        repaired_text, repair_notes = repair_common_json_errors(text)
        if repaired_text == text:
            raise original_error

        try:
            return _extract_json_from_text_once(
                repaired_text,
                json_repaired=True,
                repair_notes=repair_notes,
            )
        except ValueError as repaired_error:
            raise repaired_error


def _extract_text_like(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def normalize_object_list(parsed_json: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    objects = parsed_json.get("objects")
    if isinstance(objects, list):
        for item in objects:
            text = _extract_text_like(item)
            if text is not None:
                candidates.append(text)

    elements = parsed_json.get("elements")
    if isinstance(elements, list):
        for element in elements:
            if not isinstance(element, dict):
                continue
            for field in ("text", "name", "label"):
                text = _extract_text_like(element.get(field))
                if text is not None:
                    candidates.append(text)
                    break

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _strip_common_line_prefix(line: str) -> str:
    return _BULLET_PREFIX_RE.sub("", line, count=1).strip()


def _looks_like_non_object_line(line: str) -> bool:
    lowered = line.strip().casefold()
    if not lowered:
        return True
    if lowered in _OBJECT_EXPLANATION_PREFIXES:
        return True
    if any(lowered.startswith(prefix) for prefix in _OBJECT_EXPLANATION_PREFIXES):
        return True
    if lowered.startswith(("json", "{", "[", "```")):
        return True
    if "|" in line:
        return True
    if len(line.split()) > 12:
        return True
    return False


def parse_line_object_names(raw_text: str) -> list[str]:
    objects: list[str] = []
    seen: set[str] = set()

    for raw_line in raw_text.splitlines():
        line = _strip_common_line_prefix(raw_line)
        if not line or _looks_like_non_object_line(line):
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        objects.append(line)
    return objects


def _parse_focused_value(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized in _BOOL_TRUE_VALUES:
        return True
    if normalized in _BOOL_FALSE_VALUES:
        return False
    return None


def parse_line_bbox_elements(raw_text: str) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    lines = raw_text.splitlines()
    begin_index = next((index for index, line in enumerate(lines) if line.strip() == "BEGIN_ELEMENTS"), None)
    end_index = next((index for index, line in enumerate(lines) if line.strip() == "END_ELEMENTS"), None)
    if begin_index is None or end_index is None or end_index <= begin_index:
        return elements

    saw_header = False
    for raw_line in lines[begin_index + 1 : end_index]:
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split("|")]
        if not saw_header and tuple(part.casefold() for part in parts) == (
            "text",
            "type",
            "x1",
            "y1",
            "x2",
            "y2",
            "focused",
        ):
            saw_header = True
            continue
        if len(parts) != 7:
            continue

        text, element_type, raw_x1, raw_y1, raw_x2, raw_y2, focused_text = parts
        text = _strip_common_line_prefix(text)
        if not text or _looks_like_non_object_line(text):
            continue
        if element_type not in {"button", "tab", "app_icon", "card", "menu_item", "input", "unknown"}:
            continue

        try:
            bbox = [int(raw_x1), int(raw_y1), int(raw_x2), int(raw_y2)]
        except (TypeError, ValueError):
            continue
        if not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000):
            continue

        focused = _parse_focused_value(focused_text)
        if focused is None:
            continue

        elements.append(
            {
                "text": text,
                "type": element_type,
                "bbox_norm": bbox,
                "focused": focused,
            }
        )

    return elements


def cleanup_model(model, processor) -> None:
    del model
    del processor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
