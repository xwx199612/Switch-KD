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
):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

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
        device_map=device_map,
        torch_dtype=resolve_torch_dtype(torch_dtype),
    )
    model.eval()
    return processor, model


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
    messages = build_qwen_messages(image, prompt)
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        chat_text = prompt

    qwen_images = None
    qwen_videos = None
    try:
        from qwen_vl_utils import process_vision_info

        qwen_images, qwen_videos = process_vision_info(messages)
    except ImportError:
        qwen_images, qwen_videos = None, None

    processor_kwargs: dict[str, Any] = {
        "text": [chat_text],
        "return_tensors": "pt",
    }
    if qwen_images is not None:
        processor_kwargs["images"] = qwen_images
    else:
        processor_kwargs["images"] = [image]
    if qwen_videos is not None:
        processor_kwargs["videos"] = qwen_videos

    try:
        inputs = processor(**processor_kwargs)
    except TypeError:
        processor_kwargs.pop("videos", None)
        inputs = processor(
            images=processor_kwargs["images"],
            text=processor_kwargs["text"],
            return_tensors="pt",
        )
    input_device = select_input_device(model)
    inputs = _move_inputs_to_device(inputs, input_device)

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
    )
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        generated_ids = output_ids[:, input_ids.shape[1]:]
    else:
        generated_ids = output_ids
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def extract_json_from_text(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in model output.")


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
