from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from scripts import compare_vlm_bbox_grounding, compare_vlm_object_listing


def _make_images(image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for name in ("sample_001.jpg", "sample_002.jpg"):
        path = image_dir / name
        Image.new("RGB", (100, 100), color="white").save(path)
        image_paths.append(path)
    return image_paths


def test_compare_vlm_object_listing_writes_debug_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir)
    output_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        compare_vlm_object_listing,
        "load_processor_and_model",
        lambda model_path, torch_dtype, device_map, quantization=None: (
            {"model_path": model_path},
            {"model_path": model_path},
        ),
    )
    monkeypatch.setattr(compare_vlm_object_listing, "cleanup_model", lambda model, processor: None)

    def fake_run_vlm_inference(model, processor, image, prompt, max_new_tokens):
        name = Path(model["model_path"]).name
        if name == "distilled" and image.size == (100, 100):
            fake_run_vlm_inference.calls += 1
            if fake_run_vlm_inference.calls == 1:
                return "not json"
        if name == "qwen32b":
            return json.dumps({"objects": ["Picture", "Sound", "Picture"]})
        if name == "qwen8b":
            return json.dumps(
                {
                    "elements": [
                        {"text": "Picture"},
                        {"name": "Sound"},
                        {"label": "General"},
                    ]
                }
            )
        return json.dumps({"objects": ["Network"]})

    fake_run_vlm_inference.calls = 0
    monkeypatch.setattr(compare_vlm_object_listing, "run_vlm_inference", fake_run_vlm_inference)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_vlm_object_listing.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--model-32b",
            "qwen32b",
                "--model-8b",
                "qwen8b",
                "--model-distilled",
                "distilled",
                "--output-format",
                "json",
            ],
        )

    compare_vlm_object_listing.main()

    for model_dir_name in ("qwen3vl_32b", "qwen3vl_8b", "distilled_32to8b"):
        model_dir = output_dir / model_dir_name
        assert model_dir.is_dir()
        assert (model_dir / "raw" / "sample_001.txt").exists()
        assert (model_dir / "parsed" / "sample_001.txt").exists()
        assert (model_dir / "json" / "sample_001.json").exists()

    qwen8b_payload = json.loads((output_dir / "qwen3vl_8b" / "json" / "sample_001.json").read_text(encoding="utf-8"))
    assert qwen8b_payload["objects"] == ["Picture", "Sound", "General"]

    distilled_payload = json.loads(
        (output_dir / "distilled_32to8b" / "json" / "sample_001.json").read_text(encoding="utf-8")
    )
    assert distilled_payload["parse_error"].startswith("ValueError:")


def test_compare_vlm_bbox_grounding_writes_annotations_and_debug_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir)
    output_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        compare_vlm_bbox_grounding,
        "load_processor_and_model",
        lambda model_path, torch_dtype, device_map, quantization=None: (
            {"model_path": model_path},
            {"model_path": model_path},
        ),
    )
    monkeypatch.setattr(compare_vlm_bbox_grounding, "cleanup_model", lambda model, processor: None)

    def fake_run_vlm_inference(model, processor, image, prompt, max_new_tokens):
        name = Path(model["model_path"]).name
        fake_run_vlm_inference.calls.setdefault(name, 0)
        fake_run_vlm_inference.calls[name] += 1
        if name == "distilled" and fake_run_vlm_inference.calls[name] == 1:
            return "invalid json"
        return "\n".join(
            [
                "BEGIN_ELEMENTS",
                "text | type | x1 | y1 | x2 | y2 | focused",
                f"Picture | card | 100 | 100 | 300 | 300 | {'true' if name == 'qwen32b' else 'false'}",
                "SkipBad | card | 1 | 2 | 1 | 2 | false",
                "END_ELEMENTS",
            ]
        )

    fake_run_vlm_inference.calls = {}
    monkeypatch.setattr(compare_vlm_bbox_grounding, "run_vlm_inference", fake_run_vlm_inference)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_vlm_bbox_grounding.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--model-32b",
            "qwen32b",
            "--model-8b",
            "qwen8b",
            "--model-distilled",
            "distilled",
            "--coord-system",
            "normalized-1000",
        ],
    )

    compare_vlm_bbox_grounding.main()

    for model_dir_name in ("qwen3vl_32b", "qwen3vl_8b", "distilled_32to8b"):
        model_dir = output_dir / model_dir_name
        assert model_dir.is_dir()
        assert (model_dir / "sample_001_annotated.jpg").exists()
        assert (model_dir / "sample_002_annotated.jpg").exists()
        assert (model_dir / "raw" / "sample_001.txt").exists()
        assert (model_dir / "json" / "sample_001.json").exists()

    parsed = json.loads((output_dir / "qwen3vl_32b" / "json" / "sample_001.json").read_text(encoding="utf-8"))
    assert parsed["elements"][0]["text"] == "Picture"
    assert parsed["elements"][0]["bbox_norm"] == [100, 100, 300, 300]

    distilled_debug = json.loads(
        (output_dir / "distilled_32to8b" / "json" / "sample_001.json").read_text(encoding="utf-8")
    )
    assert distilled_debug["parse_error"] == "no_valid_lines"

    annotated = Image.open(output_dir / "qwen3vl_32b" / "sample_001_annotated.jpg")
    try:
        assert annotated.getpixel((10, 10))[0] > 200
    finally:
        annotated.close()
