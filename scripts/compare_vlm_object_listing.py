#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    extract_json_from_text,
    list_images,
    load_processor_and_model,
    normalize_object_list,
    run_vlm_inference,
)


PROMPT = """You are analyzing an Android TV / GUI screenshot.
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
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--image-extensions",
        default=".jpg,.jpeg,.png,.webp",
    )
    return parser.parse_args()


def output_filename_for_role(role: str) -> str:
    return f"{role}_objects.txt"


def write_result_block(
    handle,
    image_name: str,
    objects: list[str],
    *,
    parse_failed: bool,
    raw_output: str,
) -> None:
    handle.write("=" * 60 + "\n")
    handle.write(f"Image: {image_name}\n")
    handle.write(f"Object count: {len(objects)}\n")
    if parse_failed:
        handle.write("Status: parse_failed\n")
    handle.write("Objects:\n")
    if objects:
        for index, name in enumerate(objects, start=1):
            handle.write(f"{index}. {name}\n")
    else:
        handle.write("(none)\n")
    if parse_failed:
        handle.write("Raw output:\n")
        handle.write(raw_output.strip() + "\n" if raw_output.strip() else "(empty)\n")
    handle.write("\n")


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
        output_path = output_dir / output_filename_for_role(role)
        print(f"[load] role={role} model_path={model_path}")
        processor, model = load_processor_and_model(
            model_path=model_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                for index, image_path in enumerate(images, start=1):
                    print(
                        f"[infer] role={role} image={index}/{len(images)} "
                        f"filename={image_path.name}"
                    )
                    raw_output = ""
                    parse_failed = False
                    objects: list[str] = []
                    try:
                        with Image.open(image_path) as image_file:
                            image = image_file.convert("RGB")
                        raw_output = run_vlm_inference(
                            model=model,
                            processor=processor,
                            image=image,
                            prompt=PROMPT,
                            max_new_tokens=args.max_new_tokens,
                        )
                        parsed = extract_json_from_text(raw_output)
                        objects = normalize_object_list(parsed)
                    except Exception as exc:  # noqa: BLE001
                        parse_failed = True
                        if not raw_output:
                            raw_output = (
                                f"{type(exc).__name__}: {exc}\n"
                                f"{traceback.format_exc().strip()}"
                            )
                    write_result_block(
                        handle,
                        image_path.name,
                        objects,
                        parse_failed=parse_failed,
                        raw_output=raw_output,
                    )
            print(f"[done] role={role} output={output_path}")
        finally:
            print(f"[cleanup] role={role}")
            cleanup_model(model, processor)


if __name__ == "__main__":
    main()
