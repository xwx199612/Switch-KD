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
    MODEL_SPECS,
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

Return line-based TSV-like output only.
Each line must be:
text | x_min,y_min,x_max,y_max | focused

Example:
Picture | 145,238,276,292 | false
General | 145,348,276,404 | true
Network Settings | 705,396,807,432 | false

Rules:
Do not return JSON.
Do not return markdown.
Do not return explanations.
One visible interactive UI element per line.
Use normalized 0-1000 coordinates.
Include all visible interactive UI elements.
Do not impose a maximum number of elements.
Do not include decorative background graphics.
Do not include duplicate elements.
Do not include long helper descriptions unless the text itself is a distinct clickable UI element.
Use concise labels only."""

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
        description="Compare three VLM models on bbox grounding for a folder of images."
    )
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-32b", required=True)
    parser.add_argument("--model-8b", required=True)
    parser.add_argument("--model-distilled", required=True)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for non-quantized loading. Quantized loading always uses auto.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Torch dtype for model loading and 4-bit compute.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit", "8bit"),
        default="none",
        help="Quantized model loading mode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--output-format",
        choices=("line", "json"),
        default="line",
        help="Model output format. 'line' is the parse-resistant default; 'json' keeps legacy parsing.",
    )
    parser.add_argument(
        "--coord-system",
        choices=(COORD_SYSTEM_PIXEL, COORD_SYSTEM_NORMALIZED_1000, COORD_SYSTEM_AUTO),
        default=COORD_SYSTEM_NORMALIZED_1000,
    )
    parser.add_argument(
        "--image-extensions",
        default=".jpg,.jpeg,.png,.webp",
    )
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
        normalized.append(
            {
                "text": text_value.strip(),
                "bbox": normalized_bbox,
                "focused": bool(element.get("focused", False)),
            }
        )
    return normalized, skipped


def annotated_name(image_path: Path) -> str:
    return f"{image_path.stem}_annotated{image_path.suffix or '.jpg'}"


def debug_stem(image_path: Path) -> str:
    return image_path.stem


def main() -> None:
    args = parse_args()
    images = list_images(args.image_dir, args.image_extensions.split(","))
    output_dir = ensure_dir(args.output_dir)
    model_paths = {
        "model_32b": args.model_32b,
        "model_8b": args.model_8b,
        "model_distilled": args.model_distilled,
    }

    for role, path_key in MODEL_SPECS:
        model_path = model_paths[path_key]
        model_dir = ensure_dir(output_dir / role)
        raw_dir = ensure_dir(model_dir / "raw")
        json_dir = ensure_dir(model_dir / "json")
        print(f"[load] role={role} model_path={model_path}")
        processor, model = load_processor_and_model(
            model_path=model_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            quantization=args.quantization,
        )
        try:
            for index, image_path in enumerate(images, start=1):
                print(
                    f"[infer] role={role} image={index}/{len(images)} "
                    f"filename={image_path.name}"
                )
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
                    raw_path = raw_dir / f"{debug_stem(image_path)}.txt"
                    raw_path.write_text(raw_output + ("\n" if raw_output else ""), encoding="utf-8")

                    if args.output_format == "json":
                        parsed_json = extract_json_from_text(raw_output)
                        normalized_elements, skipped = normalize_elements(parsed_json)
                    else:
                        normalized_elements = parse_line_bbox_elements(raw_output)
                        skipped = []
                    debug_payload: dict[str, Any] = {
                        "elements": normalized_elements,
                        "parse_format": args.output_format,
                    }
                    if not normalized_elements and raw_output.strip():
                        debug_payload["parse_error"] = "no_valid_lines"
                        debug_payload["hint"] = (
                            "The model output may be malformed. Inspect raw output."
                        )
                    if skipped:
                        debug_payload["skipped_elements"] = skipped
                    json_path = json_dir / f"{debug_stem(image_path)}.json"
                    json_path.write_text(
                        json.dumps(debug_payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    draw_bboxes(
                        image_path=image_path,
                        lm_data=debug_payload,
                        output_path=model_dir / annotated_name(image_path),
                        coord_system=args.coord_system,
                        font_size=args.font_size,
                        line_width=args.line_width,
                        font=args.font,
                        include_focused_suffix=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[error] role={role} image={image_path.name} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    raw_path = raw_dir / f"{debug_stem(image_path)}.txt"
                    json_path = json_dir / f"{debug_stem(image_path)}.json"
                    annotated_path = model_dir / annotated_name(image_path)
                    if raw_output:
                        error_text = raw_output
                    else:
                        error_text = (
                            f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc().strip()}"
                        )
                    raw_path.write_text(error_text + "\n", encoding="utf-8")
                    json_path.write_text(
                        json.dumps(
                            {
                                "elements": [],
                                "parse_format": args.output_format,
                                "parse_error": f"{type(exc).__name__}: {exc}",
                                "hint": "The model output may be malformed or truncated. Inspect raw output.",
                            },
                            ensure_ascii=False,
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    draw_bboxes(
                        image_path=image_path,
                        lm_data={"elements": []},
                        output_path=annotated_path,
                        coord_system=args.coord_system,
                        font_size=args.font_size,
                        line_width=args.line_width,
                        font=args.font,
                        include_focused_suffix=False,
                    )
            print(f"[done] role={role} output_dir={model_dir}")
        finally:
            print(f"[cleanup] role={role}")
            cleanup_model(model, processor)


if __name__ == "__main__":
    main()
