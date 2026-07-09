from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import vlm_distill.cli as cli
from vlm_distill.config_schema import load_config, resolve_label_path
from vlm_distill.teacher_validation import validate_teacher_output_file


def _row() -> dict:
    return {
        "id": "sample-1",
        "image": "screen.png",
        "task": "parsing",
        "query": "List the visible UI elements.",
        "elements": [{"text": "Home", "bbox_norm": [1, 2, 3, 4], "focused": True}],
        "coordinate_system": "normalized_0_1000",
    }


def test_validate_teacher_output_file_accepts_elements_only_rows(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    summary = validate_teacher_output_file(path)

    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 0


def test_validate_teacher_cli_works(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config_path = Path("configs/parsing_switch_kd.yaml")
    monkeypatch.setenv("VLM_DISTILL_OUTPUT_ROOT", str(tmp_path))

    config = load_config(config_path)
    label_path = resolve_label_path(config.data)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["vlm-distill", "validate-teacher", "--config", str(config_path)])

    cli.main()

    output = capsys.readouterr().out
    assert "OK validated teacher output path=" in output
    assert "valid_rows=1" in output
