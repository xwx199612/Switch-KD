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


def test_cli_accepts_mixed_4bit_bf16(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vlm_bbox_grounding.py",
            "--image-dir", "images",
            "--output-dir", "out",
            "--model", "model",
            "--quantization", "mixed_4bit_bf16",
        ],
    )
    assert vlm_bbox_grounding.parse_args().quantization == "mixed_4bit_bf16"


def test_cli_does_not_support_output_format(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vlm_bbox_grounding.py", "--image-dir", "images", "--output-dir", "out", "--model", "model", "--output-format", "line"])
    with pytest.raises(SystemExit):
        vlm_bbox_grounding.parse_args()


def test_prompt_matches_training_schema_and_query() -> None:
    prompt = vlm_bbox_grounding.TRAINING_JSON_PROMPT_TEMPLATE.format(
        query="Find the focused Settings tile.", question="ignored", task="parsing"
    )
    assert """Important:
Because Python .format() is used on prompt_template:
- Keep placeholders as single braces: {query}, {question}, {task}
- Escape literal JSON braces as double braces: {{ and }}
- Do not use unescaped JSON braces in YAML.""" in prompt
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
    assert "parse_recovered" not in payload
    assert (output_dir / "raw" / "first.txt").read_text().strip() == _valid_response()


def test_complete_json_uses_strict_path_and_is_not_recovered() -> None:
    raw = _valid_response()
    parsed = vlm_bbox_grounding.extract_json_from_text(raw)
    assert parsed["elements"][0]["text"] == "Settings"
    assert "parse_recovered" not in parsed


def test_recover_truncated_elements_and_discard_incomplete_object() -> None:
    truncated = '''{
  "elements": [
    {"text": "Search", "bbox_norm": [175, 180, 240, 220], "focused": false},
    {"text": "Home", "bbox_norm": [255, 180, 295, 220], "focused": false},
    {"text": "Incomplete", "bbox_norm": [300, 300,
'''
    recovered = vlm_bbox_grounding.recover_truncated_elements_json(truncated)
    assert recovered["elements"] == [
        {"text": "Search", "bbox_norm": [175, 180, 240, 220], "focused": False},
        {"text": "Home", "bbox_norm": [255, 180, 295, 220], "focused": False},
    ]
    assert recovered["coordinate_system"] is None


def test_recovery_ends_immediately_after_completed_object_and_keeps_coordinate_system() -> None:
    raw = '{"elements":[{"text":"Home","bbox_norm":[1,2,3,4],"focused":false}], "coordinate_system":"normalized_0_1000"'
    recovered = vlm_bbox_grounding.recover_truncated_elements_json(raw)
    assert recovered["elements"] == [{"text": "Home", "bbox_norm": [1, 2, 3, 4], "focused": False}]
    assert recovered["coordinate_system"] == "normalized_0_1000"


def test_recovery_balances_braces_and_escaped_quotes_in_strings() -> None:
    raw = r'''{"elements":[
      {"text":"Use {device} and \"name\"","bbox_norm":[10,20,30,40],"focused":false}'''
    recovered = vlm_bbox_grounding.recover_truncated_elements_json(raw)
    assert recovered["elements"][0]["text"] == 'Use {device} and "name"'


def test_recovered_schema_invalid_element_is_skipped() -> None:
    raw = '''{"elements":[
      {"text":"Bad","bbox_norm":[10, 20, 30],"focused":false},
      {"text":"Good","bbox_norm":[10,20,30,40],"focused":false}'''
    recovered = vlm_bbox_grounding.recover_truncated_elements_json(raw)
    normalized, skipped = vlm_bbox_grounding.normalize_elements(recovered)
    assert normalized == [{"text": "Good", "bbox_norm": [10, 20, 30, 40], "focused": False}]
    assert "malformed bbox_norm" in skipped[0]


def test_recovery_succeeds_and_preserves_raw_artifact(monkeypatch, tmp_path: Path, capsys) -> None:
    raw = '''{
  "elements": [
    {"text": "Search", "bbox_norm": [175, 180, 240, 220], "focused": false},
    {"text": "Home", "bbox_norm": [255, 180, 295, 220], "focused": false},
'''
    output_dir, _ = _run_cli(monkeypatch, tmp_path, [raw, _valid_response()])
    output = capsys.readouterr().out
    assert "[recovered] image=first.jpg elements=2 warning=truncated_json_recovered" in output
    assert "[done] image=first.jpg" not in output
    assert "success=2 recovered=1 parse_failed=0 runtime_failed=0 failed=0" in output
    assert (output_dir / "raw" / "first.txt").read_text() == raw
    payload = json.loads((output_dir / "json" / "first.json").read_text())
    assert payload["parse_recovered"] is True
    assert payload["schema_warnings"] == ["truncated_json_recovered", "missing_or_invalid_coordinate_system"]


def test_nontruncation_and_no_completed_elements_remain_runtime_failed(monkeypatch, tmp_path: Path, capsys) -> None:
    output_dir, _ = _run_cli(monkeypatch, tmp_path, ["{\"elements\": [", "prose {not json}"])
    output = capsys.readouterr().out
    assert "success=0 recovered=0 parse_failed=0 runtime_failed=2 failed=2" in output
    assert json.loads((output_dir / "json" / "first.json").read_text())["parse_error"].startswith("ValueError:")
    assert json.loads((output_dir / "json" / "second.json").read_text())["parse_error"].startswith("ValueError:")


def test_random_prose_with_braces_is_not_recovered() -> None:
    with pytest.raises(ValueError):
        vlm_bbox_grounding.recover_truncated_elements_json("Here is {not JSON} with braces")


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
