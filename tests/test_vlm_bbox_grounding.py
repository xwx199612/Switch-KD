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


def test_cli_uses_single_model_and_query_arguments(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vlm_bbox_grounding.py", "--image-dir", "images", "--output-dir", "out", "--model", "model"])
    args = vlm_bbox_grounding.parse_args()
    assert args.query == vlm_bbox_grounding.DEFAULT_QUERY
    assert not hasattr(args, "output_format")


def test_cli_does_not_support_output_format(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vlm_bbox_grounding.py", "--image-dir", "images", "--output-dir", "out", "--model", "model", "--output-format", "line"])
    with pytest.raises(SystemExit):
        vlm_bbox_grounding.parse_args()


def test_prompt_matches_training_schema_and_query() -> None:
    prompt = vlm_bbox_grounding.TRAINING_JSON_PROMPT_TEMPLATE.format(
        query="Find the focused Settings tile.", question="ignored", task="parsing"
    )
    assert "Return valid JSON only." in prompt
    assert '"bbox_norm"' in prompt
    assert '"coordinate_system": "normalized_0_1000"' in prompt
    assert "Do not include type." in prompt
    assert "Find the focused Settings tile." in prompt


@pytest.mark.parametrize(
    ("element", "valid", "reason"),
    [
        ({"text": "Settings", "bbox_norm": [1, 2, 30, 40], "focused": False}, True, None),
        ({"text": "", "bbox_norm": [1, 2, 30, 40], "focused": False}, False, "missing text"),
        ({"text": "Bad", "bbox_norm": [1, 2, 3], "focused": False}, False, "malformed bbox_norm"),
        ({"text": "Bad", "bbox": [1, 2, 30, 40], "focused": False}, False, "malformed bbox_norm"),
        ({"text": "Bad", "bbox_norm": [1, 2.5, 30, 40], "focused": False}, False, "must contain integers"),
        ({"text": "Bad", "bbox_norm": [2, 2, 1, 40], "focused": False}, False, "invalid bbox_norm coordinates"),
        ({"text": "Bad", "bbox_norm": [-1, 2, 30, 40], "focused": False}, False, "invalid bbox_norm coordinates"),
        ({"text": "Bad", "bbox_norm": [1, 2, 30, 40], "focused": "false"}, False, "focused must be boolean"),
    ],
)
def test_normalize_elements_strict_training_schema(element, valid, reason) -> None:
    normalized, skipped = vlm_bbox_grounding.normalize_elements({"elements": [element]})
    if valid:
        assert normalized == [{"text": "Settings", "bbox_norm": [1, 2, 30, 40], "focused": False}]
        assert skipped == []
    else:
        assert normalized == []
        assert reason in skipped[0]


def _run_cli(monkeypatch, tmp_path: Path, responses, image_names=("first.jpg", "second.png"), query=None):
    image_dir = tmp_path / "images"
    _make_images(image_dir, image_names)
    output_dir = tmp_path / "outputs"
    calls = {"load": 0, "cleanup": 0, "inference": 0, "prompts": []}
    shared_model = object()
    shared_processor = object()

    def fake_load(**kwargs):
        calls["load"] += 1
        return shared_processor, shared_model

    def fake_run(**kwargs):
        calls["inference"] += 1
        calls["prompts"].append(kwargs["prompt"])
        response = responses[calls["inference"] - 1]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(vlm_bbox_grounding, "load_processor_and_model", fake_load)
    monkeypatch.setattr(vlm_bbox_grounding, "run_vlm_inference", fake_run)
    monkeypatch.setattr(vlm_bbox_grounding, "cleanup_model", lambda model, processor: calls.__setitem__("cleanup", calls["cleanup"] + 1))
    argv = ["vlm_bbox_grounding.py", "--image-dir", str(image_dir), "--output-dir", str(output_dir), "--model", "selected-model"]
    if query:
        argv += ["--query", query]
    monkeypatch.setattr(sys, "argv", argv)
    vlm_bbox_grounding.main()
    return output_dir, calls


def _valid_response(**extra) -> str:
    payload = {"elements": [{"text": "Settings", "bbox_norm": [80, 120, 140, 180], "focused": False}], "coordinate_system": "normalized_0_1000"}
    payload.update(extra)
    return json.dumps(payload)


def test_valid_json_artifacts_prompt_and_model_reuse(monkeypatch, tmp_path: Path) -> None:
    output_dir, calls = _run_cli(monkeypatch, tmp_path, [_valid_response(), _valid_response()], query="Find Settings.")
    assert calls["load"] == calls["cleanup"] == 1
    assert calls["inference"] == 2
    assert all("Find Settings." in prompt for prompt in calls["prompts"])
    payload = json.loads((output_dir / "json" / "first.json").read_text())
    assert payload["elements"][0] == {"text": "Settings", "bbox_norm": [80, 120, 140, 180], "focused": False}
    assert payload["coordinate_system"] == "normalized_0_1000"
    assert payload["parse_format"] == "json"
    assert (output_dir / "raw" / "first.txt").read_text().strip() == _valid_response()


def test_invalid_coordinate_system_warns_but_valid_element_succeeds(monkeypatch, tmp_path: Path) -> None:
    output_dir, _ = _run_cli(monkeypatch, tmp_path, [_valid_response(coordinate_system="pixels"), _valid_response()])
    payload = json.loads((output_dir / "json" / "first.json").read_text())
    assert payload["schema_warnings"] == ["missing_or_invalid_coordinate_system"]
    assert "parse_error" not in payload


def test_empty_and_no_valid_elements_are_parse_failures(monkeypatch, tmp_path: Path, capsys) -> None:
    output_dir, _ = _run_cli(monkeypatch, tmp_path, ["", json.dumps({"elements": []})])
    output = capsys.readouterr().out
    assert "[parse-failed] image=first.jpg error=empty_output" in output
    assert "[parse-failed] image=second.png error=no_valid_elements" in output
    first = json.loads((output_dir / "json" / "first.json").read_text())
    second = json.loads((output_dir / "json" / "second.json").read_text())
    assert first["parse_error"] == "empty_output"
    assert second["parse_error"] == "no_valid_elements"


def test_json_extraction_exception_is_runtime_failed(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(vlm_bbox_grounding, "extract_json_from_text", lambda raw: (_ for _ in ()).throw(ValueError("bad JSON")))
    output_dir, _ = _run_cli(monkeypatch, tmp_path, ["not JSON", _valid_response()])
    assert "runtime_failed=2" in capsys.readouterr().out
    payload = json.loads((output_dir / "json" / "first.json").read_text())
    assert payload["parse_error"] == "ValueError: bad JSON"


def test_inference_failure_does_not_stop_batch(monkeypatch, tmp_path: Path, capsys) -> None:
    output_dir, calls = _run_cli(monkeypatch, tmp_path, [RuntimeError("inference failed"), _valid_response()])
    assert calls["inference"] == 2
    assert "runtime_failed=1" in capsys.readouterr().out
    assert json.loads((output_dir / "json" / "second.json").read_text())["elements"]
