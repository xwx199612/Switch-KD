from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COORD_SYSTEM_PIXEL = "pixel"
COORD_SYSTEM_NORMALIZED_1000 = "normalized-1000"
COORD_SYSTEM_AUTO = "auto"

NORMAL_BBOX_COLOR = "#00B7FF"
FOCUSED_BBOX_COLOR = "#FF3B30"
LABEL_TEXT_COLOR = "#FFFFFF"


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

    raise ValueError("No JSON object found in LM output.")


def load_lm_output(path: str | Path) -> dict[str, Any]:
    lm_output_path = Path(path)
    if not lm_output_path.exists():
        raise FileNotFoundError(f"LM output file not found: {lm_output_path}")

    text = lm_output_path.read_text(encoding="utf-8")
    data = extract_json_from_text(text)
    elements = data.get("elements")
    if not isinstance(elements, list):
        raise ValueError("LM output must contain an 'elements' list.")
    return data


def infer_coord_system(elements: list[Any], width: int, height: int) -> str:
    max_coord: float | None = None

    for element in elements:
        if not isinstance(element, dict):
            continue

        bbox = element.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            coords = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue

        if max_coord is None:
            max_coord = max(coords)
        else:
            max_coord = max(max_coord, *coords)

    if max_coord is not None and max_coord <= 1000 and (width > 1000 or height > 1000):
        return COORD_SYSTEM_NORMALIZED_1000
    return COORD_SYSTEM_PIXEL


def convert_bbox_to_pixels(
    bbox: Any,
    width: int,
    height: int,
    coord_system: str,
) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x_min, y_min, x_max, y_max = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    if coord_system == COORD_SYSTEM_NORMALIZED_1000:
        x_scale = width / 1000.0
        y_scale = height / 1000.0
        return x_min * x_scale, y_min * y_scale, x_max * x_scale, y_max * y_scale

    return x_min, y_min, x_max, y_max


def clamp_bbox(bbox: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x_min, y_min, x_max, y_max = (int(round(float(value))) for value in bbox)
    except (TypeError, ValueError):
        return None

    x_min = min(max(x_min, 0), width)
    y_min = min(max(y_min, 0), height)
    x_max = min(max(x_max, 0), width)
    y_max = min(max(y_max, 0), height)

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def _load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _label_text(element: dict[str, Any]) -> str | None:
    text = element.get("text")
    if not isinstance(text, str):
        return None

    label = text.strip()
    if not label:
        return None

    if element.get("focused") is True:
        return f"{label} FOCUSED"
    return label


def label_text(
    element: dict[str, Any],
    *,
    include_focused_suffix: bool = True,
) -> str | None:
    label = _label_text(element)
    if label is None:
        return None
    if include_focused_suffix:
        return label
    if element.get("focused") is True:
        return label.removesuffix(" FOCUSED")
    return label


def draw_bboxes(
    image_path: str | Path,
    lm_data: dict[str, Any],
    output_path: str | Path,
    coord_system: str = COORD_SYSTEM_AUTO,
    font_size: int = 18,
    line_width: int = 3,
    font: str | None = None,
    include_focused_suffix: bool = True,
) -> None:
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file}")

    elements = lm_data.get("elements")
    if not isinstance(elements, list):
        raise ValueError("LM output must contain an 'elements' list.")

    with Image.open(image_file) as source_image:
        annotated = source_image.convert("RGB").copy()

    width, height = annotated.size
    draw = ImageDraw.Draw(annotated)
    image_font = _load_font(font, font_size)
    resolved_coord_system = coord_system
    if coord_system == COORD_SYSTEM_AUTO:
        resolved_coord_system = infer_coord_system(elements, width, height)

    for element in elements:
        if not isinstance(element, dict):
            continue

        pixel_bbox = convert_bbox_to_pixels(element.get("bbox"), width, height, resolved_coord_system)
        clamped_bbox = clamp_bbox(pixel_bbox, width, height)
        label = label_text(
            element,
            include_focused_suffix=include_focused_suffix,
        )
        if clamped_bbox is None or label is None:
            continue

        x_min, y_min, x_max, y_max = clamped_bbox
        focused = element.get("focused") is True
        bbox_color = FOCUSED_BBOX_COLOR if focused else NORMAL_BBOX_COLOR

        draw.rectangle(clamped_bbox, outline=bbox_color, width=line_width)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=image_font)
        text_width = right - left
        text_height = bottom - top
        padding_x = 6
        padding_y = 4
        label_height = text_height + (padding_y * 2)

        label_x = x_min
        if y_min >= label_height:
            label_y = y_min - label_height
        else:
            label_y = min(y_max, max(0, height - label_height))

        background_box = (
            label_x,
            label_y,
            min(width, label_x + text_width + (padding_x * 2)),
            min(height, label_y + label_height),
        )
        draw.rectangle(background_box, fill=bbox_color)
        draw.text(
            (label_x + padding_x, label_y + padding_y),
            label,
            font=image_font,
            fill=LABEL_TEXT_COLOR,
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw LM bbox annotations onto a copy of an image.")
    parser.add_argument("--image", required=True, help="Path to the original image.")
    parser.add_argument("--lm-output", required=True, help="Path to the LM output text or JSON file.")
    parser.add_argument("--output", required=True, help="Path to the annotated output image.")
    parser.add_argument(
        "--coord-system",
        choices=(COORD_SYSTEM_PIXEL, COORD_SYSTEM_NORMALIZED_1000, COORD_SYSTEM_AUTO),
        default=COORD_SYSTEM_AUTO,
        help="BBox coordinate system: original pixels, normalized 0-1000, or auto-infer.",
    )
    parser.add_argument("--font-size", type=int, default=18, help="Label font size.")
    parser.add_argument("--line-width", type=int, default=3, help="Bounding box line width.")
    parser.add_argument("--font", help="Optional custom font path.")
    args = parser.parse_args()

    lm_data = load_lm_output(args.lm_output)
    draw_bboxes(
        image_path=args.image,
        lm_data=lm_data,
        output_path=args.output,
        coord_system=args.coord_system,
        font_size=args.font_size,
        line_width=args.line_width,
        font=args.font,
    )


if __name__ == "__main__":
    main()
