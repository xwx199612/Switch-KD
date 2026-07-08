#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlm_compare_utils import (
    MODEL_SPECS,
    cleanup_model,
    ensure_dir,
    list_images,
    load_processor_and_model,
    parse_line_object_names,
    extract_json_from_text,
    normalize_object_list,
    run_vlm_inference,
)


LINE_PROMPT = """You are analyzing an Android TV / GUI screenshot.
List all visible interactive UI elements and visible object names in the image.
Do not return JSON.
Do not return markdown.
Do not return explanations.
Return one visible object / UI element name per line.
Use concise names only.
Include all visible interactive UI elements.
Do not include bbox.
Do not include confidence.
Do not include numbering or bullets.
Do not include duplicate elements unless they are visually distinct repeated items."""

JSON_PROMPT = """You are analyzing an Android TV / GUI screenshot.
List all visible interactive UI elements and visible object names in the image.
Return JSON only.

Output format:
{
  "objects": [
    "Picture",
    "Sound",
    "General"
  ]
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three VLM models on object listing for a folder of images."
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output-format",
        choices=("line", "json"),
        default="line",
        help="Model output format. 'line' is the parse-resistant default; 'json' keeps legacy parsing.",
    )
    parser.add_argument(
        "--image-extensions",
        default=".jpg,.jpeg,.png,.webp",
    )
    return parser.parse_args()


def debug_stem(image_path: Path) -> str:
    return image_path.stem


def build_parsed_text(objects: list[str]) -> str:
    lines = [f"Object count: {len(objects)}", "Objects:"]
    if objects:
        for name in objects:
            lines.append(f"- {name}")
    else:
        lines.append("(none)")
    return "\n".join(lines) + "\n"


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
        parsed_dir = ensure_dir(model_dir / "parsed")
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
                objects: list[str] = []
                parse_error: str | None = None
                parse_format = args.output_format
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
                    if args.output_format == "json":
                        parsed = extract_json_from_text(raw_output)
                        objects = normalize_object_list(parsed)
                    else:
                        objects = parse_line_object_names(raw_output)
                        if raw_output.strip() and not objects:
                            parse_error = "no_valid_lines"
                except Exception as exc:  # noqa: BLE001
                    if not raw_output:
                        raw_output = (
                            f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc().strip()}"
                        )
                    parse_error = f"{type(exc).__name__}: {exc}"

                stem = debug_stem(image_path)
                (raw_dir / f"{stem}.txt").write_text(
                    raw_output + ("\n" if raw_output else ""),
                    encoding="utf-8",
                )
                (parsed_dir / f"{stem}.txt").write_text(
                    build_parsed_text(objects),
                    encoding="utf-8",
                )
                payload = {
                    "objects": objects,
                    "count": len(objects),
                    "parse_format": parse_format,
                }
                if parse_error is not None:
                    payload["parse_error"] = parse_error
                (json_dir / f"{stem}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(f"[done] role={role} output_dir={model_dir}")
        finally:
            print(f"[cleanup] role={role}")
            cleanup_model(model, processor)


if __name__ == "__main__":
    main()
