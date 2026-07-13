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
    parse_line_bbox_elements,
    run_vlm_inference,
)
from tools.draw_lm_bboxes import (
    COORD_SYSTEM_AUTO,
    COORD_SYSTEM_NORMALIZED_1000,
    COORD_SYSTEM_PIXEL,
    draw_bboxes,
)


LINE_PROMPT = """You are an Android TV visual grounding assistant.

Analyze the screenshot and list all visible interactive UI elements.

Return only the table below. No JSON, no markdown, no explanation.
BEGIN_ELEMENTS
text | type | x1 | y1 | x2 | y2 | focused
...
END_ELEMENTS

Rules:
- Each line must contain exactly 7 fields separated by " | ".
- type must be one of: button, tab, app_icon, card, menu_item, input, unknown.
- Use normalized 0-1000 coordinates, not pixel coordinates.
- x1, y1, x2, y2 must be integers only.
- Coordinates must satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
- Do not combine coordinates into one field.
- Do not use commas in coordinates.
- focused must be exactly true or false.
- If a valid bbox cannot be provided, omit that element."""

JSON_PROMPT = """You are an Android TV visual grounding assistant.

Analyze the screenshot and list all visible interactive UI elements and visible object names.

For each element, return:
- text: the visible label text, or a short semantic name if it is icon-only
- bbox: [x_min, y_min, x_max, y_max]
- focused: true or false

Coordinate rule:
Return bbox coordinates in normalized 0-1000 coordinate system.
The origin is the top-left corner of the image.
x_min and y_min are the top-left corner.
x_max and y_max are the bottom-right corner.

Return valid JSON only.

Output format:
{
  "elements": [
    {
      "text": "Picture",
      "bbox": [145, 238, 276, 292],
      "focused": false
    }
  ]
}"""


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
    parser.add_argument("--output-format", choices=("line", "json"), default="line")
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
            text_value = element.get("name")
        if not isinstance(text_value, str) or not text_value.strip():
            text_value = element.get("label")
        if not isinstance(text_value, str) or not text_value.strip():
            skipped.append(f"element_{index}: missing text")
            continue
        bbox = element.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped.append(f"element_{index}: malformed bbox")
            continue
        try:
            normalized_bbox = [float(value) for value in bbox]
        except (TypeError, ValueError):
            skipped.append(f"element_{index}: non-numeric bbox")
            continue
        normalized.append({"text": text_value.strip(), "bbox_norm": normalized_bbox, "focused": bool(element.get("focused", False))})
    return normalized, skipped


def annotated_name(image_path: Path) -> str:
    return f"{image_path.stem}_annotated{image_path.suffix or '.jpg'}"


def debug_stem(image_path: Path) -> str:
    return image_path.stem


def main() -> None:
    args = parse_args()
    images = list_images(args.image_dir, args.image_extensions.split(","))
    output_dir = ensure_dir(args.output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    json_dir = ensure_dir(output_dir / "json")
    print(f"[load] model_path={args.model}")
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
                    prompt=JSON_PROMPT if args.output_format == "json" else LINE_PROMPT,
                    max_new_tokens=args.max_new_tokens,
                )
                (raw_dir / f"{debug_stem(image_path)}.txt").write_text(raw_output + ("\n" if raw_output else ""), encoding="utf-8")
                skipped: list[str] = []
                if args.output_format == "json":
                    parsed_json = extract_json_from_text(raw_output)
                    normalized_elements, skipped = normalize_elements(parsed_json)
                else:
                    normalized_elements = parse_line_bbox_elements(raw_output)
                debug_payload: dict[str, Any] = {
                    "image": str(image_path.absolute()),
                    "model": str(args.model),
                    "elements": normalized_elements,
                    "parse_format": args.output_format,
                }
                if not normalized_elements and raw_output.strip():
                    debug_payload["parse_error"] = "no_valid_lines"
                    debug_payload["hint"] = "The model output may be malformed. Inspect raw output."
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
                    print(f"[parse-failed] image={image_path.name} error=no_valid_lines")
            except Exception as exc:  # noqa: BLE001
                runtime_failed_images += 1
                print(f"[error] image={image_path.name} error={type(exc).__name__}: {exc}")
                raw_text = raw_output or f"{type(exc).__name__}: {exc}\n{traceback.format_exc().strip()}"
                (raw_dir / f"{debug_stem(image_path)}.txt").write_text(raw_text + "\n", encoding="utf-8")
                error_payload = {"image": str(image_path.absolute()), "model": str(args.model), "elements": [], "parse_format": args.output_format, "parse_error": f"{type(exc).__name__}: {exc}", "hint": "The model output may be malformed or truncated. Inspect raw output."}
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
