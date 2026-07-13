from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts import vlm_bbox_grounding


def _make_images(image_dir: Path, names=("first.jpg", "second.png")) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (100, 80), color="white").save(image_dir / name)


def test_cli_uses_single_model_argument(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vlm_bbox_grounding.py", "--image-dir", "images", "--output-dir", "out", "--model", "model"])
    args = vlm_bbox_grounding.parse_args()
    assert args.model == "model"
    assert not hasattr(args, "model_32b")
    assert not hasattr(args, "model_8b")
    assert not hasattr(args, "model_distilled")


def test_annotated_name_preserves_extension() -> None:
    assert vlm_bbox_grounding.annotated_name(Path("screen.jpg")) == "screen_annotated.jpg"
    assert vlm_bbox_grounding.annotated_name(Path("screen.png")) == "screen_annotated.png"


@pytest.mark.parametrize(
    ("element", "expected", "skipped"),
    [
        ({"text": "Picture", "bbox": [1, 2, 30, 40]}, "Picture", None),
        ({"bbox": [1, 2, 30, 40]}, None, "missing text"),
        ({"text": "Bad", "bbox": [1, 2, 3]}, None, "malformed bbox"),
        ({"text": "Bad", "bbox": [1, "x", 3, 4]}, None, "non-numeric bbox"),
    ],
)
def test_normalize_elements(element, expected, skipped) -> None:
    normalized, skipped_items = vlm_bbox_grounding.normalize_elements({"elements": [element]})
    if expected:
        assert normalized[0]["text"] == expected
        assert normalized[0]["bbox_norm"] == [1.0, 2.0, 30.0, 40.0]
        assert skipped_items == []
    else:
        assert normalized == []
        assert skipped in skipped_items[0]


def _run_cli(
    monkeypatch,
    tmp_path: Path,
    output_format: str = "line",
    fail_first: bool = False,
    responses=None,
    image_names=("first.jpg", "second.png"),
):
    image_dir = tmp_path / "images"
    _make_images(image_dir, image_names)
    output_dir = tmp_path / "outputs"
    calls = {"load": 0, "cleanup": 0, "inference": 0}
    shared_model = object()
    shared_processor = object()

    def fake_load(**kwargs):
        calls["load"] += 1
        assert kwargs["model_path"] == "selected-model"
        return shared_processor, shared_model

    def fake_run(**kwargs):
        calls["inference"] += 1
        assert kwargs["model"] is shared_model
        assert kwargs["processor"] is shared_processor
        if fail_first and calls["inference"] == 1:
            raise ValueError("bad image output")
        if responses is not None:
            response = responses[calls["inference"] - 1]
            if isinstance(response, Exception):
                raise response
            return response
        if output_format == "json":
            return json.dumps({"elements": [{"text": "Picture", "bbox": [10, 20, 50, 60], "focused": False}]})
        return "BEGIN_ELEMENTS\ntext | type | x1 | y1 | x2 | y2 | focused\nPicture | card | 10 | 20 | 50 | 60 | false\nEND_ELEMENTS"

    monkeypatch.setattr(vlm_bbox_grounding, "load_processor_and_model", fake_load)
    monkeypatch.setattr(vlm_bbox_grounding, "run_vlm_inference", fake_run)
    monkeypatch.setattr(vlm_bbox_grounding, "cleanup_model", lambda model, processor: calls.__setitem__("cleanup", calls["cleanup"] + 1))
    monkeypatch.setattr(sys, "argv", ["vlm_bbox_grounding.py", "--image-dir", str(image_dir), "--output-dir", str(output_dir), "--model", "selected-model", "--output-format", output_format])
    vlm_bbox_grounding.main()
    return output_dir, calls


def test_one_model_is_loaded_once_and_reused_across_images(monkeypatch, tmp_path: Path) -> None:
    output_dir, calls = _run_cli(monkeypatch, tmp_path)
    assert calls == {"load": 1, "cleanup": 1, "inference": 2}
    assert (output_dir / "raw").is_dir()
    assert (output_dir / "json").is_dir()
    assert (output_dir / "first_annotated.jpg").exists()
    assert (output_dir / "second_annotated.png").exists()
    assert not [path for path in output_dir.iterdir() if path.name not in {"raw", "json", "first_annotated.jpg", "second_annotated.png"}]
    payload = json.loads((output_dir / "json" / "first.json").read_text(encoding="utf-8"))
    assert payload["model"] == "selected-model"
    assert payload["parse_format"] == "line"


def test_failed_image_does_not_stop_batch(monkeypatch, tmp_path: Path) -> None:
    output_dir, calls = _run_cli(monkeypatch, tmp_path, fail_first=True)
    assert calls == {"load": 1, "cleanup": 1, "inference": 2}
    failed = json.loads((output_dir / "json" / "first.json").read_text(encoding="utf-8"))
    succeeded = json.loads((output_dir / "json" / "second.json").read_text(encoding="utf-8"))
    assert failed["parse_error"].startswith("ValueError:")
    assert len(succeeded["elements"]) == 1
    assert (output_dir / "first_annotated.jpg").exists()
    assert (output_dir / "second_annotated.png").exists()


def test_non_empty_malformed_output_is_parse_failed_and_batch_continues(monkeypatch, tmp_path: Path, capsys) -> None:
    output_dir, calls = _run_cli(
        monkeypatch,
        tmp_path,
        responses=[
            "BEGIN_ELEMENTS\nmalformed output\nEND_ELEMENTS",
            "BEGIN_ELEMENTS\ntext | type | x1 | y1 | x2 | y2 | focused\nPicture | card | 10 | 20 | 50 | 60 | false\nEND_ELEMENTS",
        ],
    )

    summary = capsys.readouterr().out
    assert "[parse-failed] image=first.jpg error=no_valid_lines" in summary
    assert "[complete] images=2 success=1 parse_failed=1 runtime_failed=0 failed=1 total_elements=1" in summary
    assert calls["inference"] == 2
    assert (output_dir / "raw" / "first.txt").read_text(encoding="utf-8").strip()
    failed_payload = json.loads((output_dir / "json" / "first.json").read_text(encoding="utf-8"))
    assert failed_payload["elements"] == []
    assert failed_payload["parse_error"] == "no_valid_lines"
    assert (output_dir / "first_annotated.jpg").exists()


def test_empty_output_is_parse_failed_and_batch_continues(monkeypatch, tmp_path: Path, capsys) -> None:
    output_dir, calls = _run_cli(
        monkeypatch,
        tmp_path,
        responses=[
            " \t\n",
            "BEGIN_ELEMENTS\ntext | type | x1 | y1 | x2 | y2 | focused\nPicture | card | 10 | 20 | 50 | 60 | false\nEND_ELEMENTS",
        ],
    )

    summary = capsys.readouterr().out
    assert "[parse-failed] image=first.jpg error=empty_output" in summary
    assert "[complete] images=2 success=1 parse_failed=1 runtime_failed=0 failed=1 total_elements=1" in summary
    assert calls["inference"] == 2
    payload = json.loads((output_dir / "json" / "first.json").read_text(encoding="utf-8"))
    assert payload["parse_error"] == "empty_output"
    assert payload["hint"] == "The model returned no output. Inspect generation settings and model behavior."
    assert (output_dir / "raw" / "first.txt").exists()
    assert (output_dir / "first_annotated.jpg").exists()
    assert len(json.loads((output_dir / "json" / "second.json").read_text(encoding="utf-8"))["elements"]) == 1


def test_json_empty_elements_is_parse_failed(monkeypatch, tmp_path: Path, capsys) -> None:
    output_dir, _ = _run_cli(
        monkeypatch,
        tmp_path,
        output_format="json",
        responses=[json.dumps({"elements": []}), json.dumps({"elements": [{"text": "Picture", "bbox": [10, 20, 50, 60]}]})],
    )

    summary = capsys.readouterr().out
    assert "[complete] images=2 success=1 parse_failed=1 runtime_failed=0 failed=1 total_elements=1" in summary
    payload = json.loads((output_dir / "json" / "first.json").read_text(encoding="utf-8"))
    assert payload["parse_error"] == "no_valid_lines"
    assert (output_dir / "raw" / "first.txt").exists()
    assert (output_dir / "first_annotated.jpg").exists()


def test_summary_counters_add_up_with_runtime_and_parse_failures(monkeypatch, tmp_path: Path, capsys) -> None:
    _run_cli(
        monkeypatch,
        tmp_path,
        responses=[
            "BEGIN_ELEMENTS\ntext | type | x1 | y1 | x2 | y2 | focused\nPicture | card | 10 | 20 | 50 | 60 | false\nEND_ELEMENTS",
            "not parseable",
            RuntimeError("inference failed"),
        ],
        image_names=("first.jpg", "second.png", "third.jpg"),
    )

    summary = capsys.readouterr().out
    assert "[complete] images=3 success=1 parse_failed=1 runtime_failed=1 failed=2 total_elements=1" in summary


def test_json_parsing_branch(monkeypatch, tmp_path: Path) -> None:
    output_dir, _ = _run_cli(monkeypatch, tmp_path, output_format="json")
    payload = json.loads((output_dir / "json" / "first.json").read_text(encoding="utf-8"))
    assert payload["elements"][0]["bbox_norm"] == [10.0, 20.0, 50.0, 60.0]
    assert payload["parse_format"] == "json"
