from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from vlm_distill.visualize_predictions import normalized_bbox_to_pixel, run_visualization


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_bbox_conversion_orders_and_clamps() -> None:
    assert normalized_bbox_to_pixel([900, 800, 100, 100], 100, 50) == [10, 5, 90, 40]
    assert normalized_bbox_to_pixel([-100, -1, 1100, 1200], 100, 50) == [0, 0, 99, 49]


def test_json_string_dict_and_missing_bbox_are_rendered(tmp_path: Path) -> None:
    source = tmp_path / "images" / "screen.png"
    source.parent.mkdir(); Image.new("RGB", (100, 50), "white").save(source)
    before = _sha(source)
    predictions = tmp_path / "predictions.jsonl"
    _write_rows(predictions, [
        {"id": "one", "image": str(source), "student_answer": 'prefix ```json {"elements":[{"text":"A","bbox_norm":"10,20,100,200"},{"text":"missing"}]} ``` suffix'},
        {"id": "two", "image": str(source), "elements": {"elements": [{"text": "B", "bbox_norm": [200, 200, 300, 300], "type": "button"}]}},
    ])
    result = run_visualization(predictions, tmp_path / "out", write_sidecar=True)
    assert result["images_written"] == 2
    assert result["elements_drawn"] == 2
    assert result["elements_skipped"] == 1
    assert _sha(source) == before
    assert len(list((tmp_path / "out").glob("*.png"))) == 2
    sidecar = next((tmp_path / "out").glob("one__*.json"))
    assert json.loads(sidecar.read_text())["skipped_element_count"] == 1


def test_malformed_row_does_not_stop_next_row_and_names_are_unique(tmp_path: Path) -> None:
    image_a = tmp_path / "a" / "same.png"; image_b = tmp_path / "b" / "same.png"
    image_a.parent.mkdir(); image_b.parent.mkdir()
    Image.new("RGB", (20, 20), "white").save(image_a); Image.new("RGB", (20, 20), "black").save(image_b)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("not json\n" + json.dumps({"id": "ok", "image": str(image_a), "elements": []}) + "\n" + json.dumps({"id": "ok", "image": str(image_b), "elements": []}) + "\n", encoding="utf-8")
    result = run_visualization(predictions, tmp_path / "out", write_sidecar=False)
    assert result["rows_parse_failed"] == 1
    assert result["images_written"] == 2
    assert len(list((tmp_path / "out").glob("*.png"))) == 2


def test_existing_output_requires_overwrite_and_never_targets_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"; Image.new("RGB", (10, 10), "white").save(source)
    predictions = tmp_path / "p.jsonl"; _write_rows(predictions, [{"id": "x", "image": str(source), "elements": []}])
    output = tmp_path / "out"; run_visualization(predictions, output, write_sidecar=False)
    before = _sha(source)
    result = run_visualization(predictions, output, write_sidecar=False)
    assert result["rows_parse_failed"] == 1
    run_visualization(predictions, output, overwrite=True, write_sidecar=False)
    assert _sha(source) == before
