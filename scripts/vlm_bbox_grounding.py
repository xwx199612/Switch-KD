#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlm_compare_utils import (
    cleanup_model,
    ensure_dir,
    extract_json_from_text,
    list_images,
    load_processor_and_model,
    run_vlm_inference,
)
from tools.draw_lm_bboxes import (
    COORD_SYSTEM_AUTO,
    COORD_SYSTEM_NORMALIZED_1000,
    COORD_SYSTEM_PIXEL,
    draw_bboxes,
)


TRAINING_JSON_PROMPT_TEMPLATE = """You are labeling Android TV screenshots for UI grounding.

Task:
{query}

Find all visible interactive UI elements.

Return valid JSON only.
Use ASCII double quotes only: "
Do not use smart quotes: “ ”
Do not use markdown.
Do not explain.

Use this exact schema:
{{
  "elements": [
    {{
      "text": "visible UI element name",
      "bbox_norm": [80, 120, 140, 180],
      "focused": false
    }}
  ],
  "coordinate_system": "normalized_0_1000"
}}

Rules:
- Every element must use exactly these keys: "text", "bbox_norm", "focused".
- Do not include type.
- Do not use alternative bbox key names: bbox, box_norm, bx_norm, bbox norm, bboxNorm, boxed_norm.
- bbox_norm must be an array of exactly four integers, e.g. [80, 120, 140, 180].
- Do not write bbox_norm as comma-separated text.
- Use normalized 0-1000 coordinates, not pixel coordinates.
- Coordinates must satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
- text must be the visible UI label or a short visual name of the element.
- Do not use schema words such as text, button, tab, app_icon, menu_item, icon, bbox_norm, coordinate_system, or elements as text unless that exact word is visibly displayed on screen.
- focused must be exactly true or false.
- If focused is not visually clear, use false.
- Include "coordinate_system": "normalized_0_1000".
- Prioritize valid JSON over recall.
- If unsure about an element bbox, omit that element.
- Keep text short. If visible text is very long, summarize it into a short label."""

DEFAULT_QUERY = "List all visible interactive UI elements on this screen."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one VLM model on a folder of images for bbox grounding."
    )
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--coord-system",
        choices=(COORD_SYSTEM_PIXEL, COORD_SYSTEM_NORMALIZED_1000, "normalized_1000", COORD_SYSTEM_AUTO),
        default=COORD_SYSTEM_NORMALIZED_1000,
    )
    parser.add_argument("--image-extensions", default=".jpg,.jpeg,.png,.webp")
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--font")
    return parser.parse_args()


def normalize_elements(parsed_json: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    elements = parsed_json.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Parsed JSON does not contain an 'elements' list.")

    normalized: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, element in enumerate(elements, start=1):
        if not isinstance(element, dict):
            skipped.append(f"element_{index}: not an object")
            continue
        text_value = element.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            skipped.append(f"element_{index}: missing text")
            continue
        bbox = element.get("bbox_norm")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped.append(f"element_{index}: malformed bbox_norm")
            continue
        normalized_bbox: list[int] = []
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
            skipped.append(f"element_{index}: bbox_norm must contain numeric values")
            continue
        if any(isinstance(value, float) and not value.is_integer() for value in bbox):
            skipped.append(f"element_{index}: bbox_norm must contain integers")
            continue
        normalized_bbox = [int(value) for value in bbox]
        x1, y1, x2, y2 = normalized_bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            skipped.append(f"element_{index}: invalid bbox_norm coordinates")
            continue
        focused = element.get("focused")
        if not isinstance(focused, bool):
            skipped.append(f"element_{index}: focused must be boolean")
            continue
        normalized.append({"text": text_value.strip(), "bbox_norm": normalized_bbox, "focused": focused})
    return normalized, skipped


def annotated_name(image_path: Path) -> str:
    return f"{image_path.stem}_annotated{image_path.suffix or '.jpg'}"


def debug_stem(image_path: Path) -> str:
    return image_path.stem


def main() -> None:
    args = parse_args()
    images = list_images(args.image_dir, args.image_extensions.split(","))
    prompt = TRAINING_JSON_PROMPT_TEMPLATE.format(
        query=args.query,
        question=args.query,
        task="parsing",
    )
    output_dir = ensure_dir(args.output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    json_dir = ensure_dir(output_dir / "json")
    print(f"[load] model_path={args.model}")
    print(f"[load] query={args.query}")
    print(f"[load] quantization={args.quantization} torch_dtype={args.torch_dtype} device_map={args.device_map}")

    processor, model = load_processor_and_model(
        model_path=args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        quantization=args.quantization,
    )
    successful_images = 0
    parse_failed_images = 0
    runtime_failed_images = 0
    total_elements = 0
    try:
        for index, image_path in enumerate(images, start=1):
            print(f"[infer] image={index}/{len(images)} filename={image_path.name}")
            raw_output = ""
            try:
                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")
                raw_output = run_vlm_inference(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                (raw_dir / f"{debug_stem(image_path)}.txt").write_text(raw_output + ("\n" if raw_output else ""), encoding="utf-8")
                skipped: list[str] = []
                schema_warnings: list[str] = []
                if not raw_output.strip():
                    normalized_elements = []
                else:
                    parsed_json = extract_json_from_text(raw_output)
                    normalized_elements, skipped = normalize_elements(parsed_json)
                    if parsed_json.get("coordinate_system") != "normalized_0_1000":
                        schema_warnings.append("missing_or_invalid_coordinate_system")
                debug_payload: dict[str, Any] = {
                    "image": str(image_path.absolute()),
                    "model": str(args.model),
                    "elements": normalized_elements,
                    "parse_format": "json",
                    "coordinate_system": "normalized_0_1000",
                }
                if schema_warnings:
                    debug_payload["schema_warnings"] = schema_warnings
                if not normalized_elements:
                    if raw_output.strip():
                        debug_payload["parse_error"] = "no_valid_elements"
                        debug_payload["hint"] = (
                            "The model output may be malformed. Inspect raw output."
                        )
                    else:
                        debug_payload["parse_error"] = "empty_output"
                        debug_payload["hint"] = (
                            "The model returned no output. Inspect generation settings and model behavior."
                        )
                if skipped:
                    debug_payload["skipped_elements"] = skipped
                (json_dir / f"{debug_stem(image_path)}.json").write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                draw_bboxes(image_path=image_path, lm_data=debug_payload, output_path=output_dir / annotated_name(image_path), coord_system=(COORD_SYSTEM_NORMALIZED_1000 if args.coord_system == "normalized_1000" else args.coord_system), font_size=args.font_size, line_width=args.line_width, font=args.font, include_focused_suffix=False)
                if normalized_elements:
                    successful_images += 1
                    total_elements += len(normalized_elements)
                    print(f"[done] image={image_path.name} elements={len(normalized_elements)}")
                else:
                    parse_failed_images += 1
                    parse_error = debug_payload["parse_error"]
                    print(f"[parse-failed] image={image_path.name} error={parse_error}")
            except Exception as exc:  # noqa: BLE001
                runtime_failed_images += 1
                print(f"[error] image={image_path.name} error={type(exc).__name__}: {exc}")
                raw_text = raw_output or f"{type(exc).__name__}: {exc}\n{traceback.format_exc().strip()}"
                (raw_dir / f"{debug_stem(image_path)}.txt").write_text(raw_text + "\n", encoding="utf-8")
                error_payload = {"image": str(image_path.absolute()), "model": str(args.model), "elements": [], "coordinate_system": "normalized_0_1000", "parse_format": "json", "parse_error": f"{type(exc).__name__}: {exc}", "hint": "The model output may be malformed or truncated. Inspect raw output."}
                (json_dir / f"{debug_stem(image_path)}.json").write_text(json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                try:
                    draw_bboxes(image_path=image_path, lm_data={"elements": []}, output_path=output_dir / annotated_name(image_path), coord_system=(COORD_SYSTEM_NORMALIZED_1000 if args.coord_system == "normalized_1000" else args.coord_system), font_size=args.font_size, line_width=args.line_width, font=args.font, include_focused_suffix=False)
                except Exception as draw_exc:  # pragma: no cover - only for unreadable input/output failures
                    print(f"[error] image={image_path.name} error={type(draw_exc).__name__}: {draw_exc}")
        failed_images = parse_failed_images + runtime_failed_images
        print(f"[complete] images={len(images)} success={successful_images} parse_failed={parse_failed_images} runtime_failed={runtime_failed_images} failed={failed_images} total_elements={total_elements} output_dir={output_dir}")
    finally:
        print("[cleanup] model")
        cleanup_model(model, processor)


if __name__ == "__main__":
    main()
