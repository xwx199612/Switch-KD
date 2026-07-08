from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from tools.draw_lm_bboxes import (
    COORD_SYSTEM_NORMALIZED_1000,
    COORD_SYSTEM_PIXEL,
    clamp_bbox,
    convert_bbox_to_pixels,
    draw_bboxes,
    extract_json_from_text,
    infer_coord_system,
    label_text,
    load_lm_output,
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extract_json_from_text_skips_prefix() -> None:
    text = """qwen3-vl-32b-instruct
Some markdown text.
{"elements":[{"text":"Picture","bbox":[1,2,3,4],"focused":false,"confidence":0.5}]}
"""

    parsed = extract_json_from_text(text)

    assert parsed["elements"][0]["text"] == "Picture"


def test_load_lm_output_requires_elements_list(tmp_path: Path) -> None:
    output_path = tmp_path / "lm_output.txt"
    output_path.write_text(json.dumps({"elements": "bad"}), encoding="utf-8")

    try:
        load_lm_output(output_path)
    except ValueError as exc:
        assert "elements" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-list elements.")


def test_clamp_bbox_clamps_to_image_bounds() -> None:
    assert clamp_bbox([-5, 10, 120, 90], width=100, height=80) == (0, 10, 100, 80)
    assert clamp_bbox([50, 50, 40, 70], width=100, height=100) is None
    assert clamp_bbox(["bad", 0, 10, 10], width=100, height=100) is None


def test_infer_coord_system_detects_normalized_1000_for_large_images() -> None:
    elements = [{"bbox": [283, 700, 518, 754]}]

    assert infer_coord_system(elements, width=1920, height=1080) == COORD_SYSTEM_NORMALIZED_1000
    assert infer_coord_system(elements, width=800, height=600) == COORD_SYSTEM_PIXEL


def test_convert_bbox_to_pixels_scales_normalized_1000() -> None:
    converted = convert_bbox_to_pixels([283, 700, 518, 754], 1920, 1080, COORD_SYSTEM_NORMALIZED_1000)

    assert converted == (543.36, 756.0, 994.56, 814.32)
    assert clamp_bbox(converted, 1920, 1080) == (543, 756, 995, 814)


def test_label_text_can_hide_focused_suffix() -> None:
    element = {"text": "General", "focused": True}

    assert label_text(element) == "General FOCUSED"
    assert label_text(element, include_focused_suffix=False) == "General"


def test_draw_bboxes_creates_output_without_modifying_original(tmp_path: Path) -> None:
    image_path = tmp_path / "original.png"
    output_path = tmp_path / "annotated.png"
    Image.new("RGB", (120, 120), color="white").save(image_path)

    original_hash = _file_hash(image_path)
    lm_data = {
        "elements": [
            {
                "text": "Picture",
                "bbox": [10, 10, 60, 60],
                "focused": False,
                "confidence": 0.95,
                "type": "menu_item",
            },
            {
                "text": "General",
                "bbox": [20, 70, 100, 110],
                "focused": True,
                "confidence": 0.98,
                "type": "menu_item",
            },
            {
                "text": "Skip",
                "bbox": [25, 25],
                "focused": False,
            },
        ]
    }

    draw_bboxes(image_path, lm_data, output_path, font_size=14, line_width=2)

    assert output_path.exists()
    assert _file_hash(image_path) == original_hash
    assert _file_hash(output_path) != original_hash


def test_draw_bboxes_supports_normalized_1000_bboxes(tmp_path: Path) -> None:
    image_path = tmp_path / "original_1080p.png"
    output_path = tmp_path / "annotated_1080p.png"
    Image.new("RGB", (1920, 1080), color="white").save(image_path)

    lm_data = {
        "elements": [
            {
                "text": "External Devices",
                "bbox": [283, 700, 518, 754],
                "focused": True,
                "confidence": 0.98,
                "type": "menu_item",
            }
        ]
    }

    draw_bboxes(
        image_path,
        lm_data,
        output_path,
        coord_system=COORD_SYSTEM_NORMALIZED_1000,
        font_size=18,
        line_width=3,
    )

    output = Image.open(output_path)
    try:
        assert output.getpixel((543, 756)) == (255, 59, 48)
    finally:
        output.close()
