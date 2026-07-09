from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.teacher_validation import validate_teacher_output_file, validate_teacher_row


def _row(*, elements=None, coordinate_system="normalized_0_1000") -> dict:
    return {
        "id": "sample-1",
        "image": "screen.png",
        "task": "parsing",
        "query": "List the visible UI elements.",
        "elements": elements if elements is not None else [{"text": "Home", "bbox_norm": [10, 20, 30, 40], "focused": True}],
        "coordinate_system": coordinate_system,
    }


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_validate_teacher_row_accepts_elements_only_parsing_row() -> None:
    valid, reason = validate_teacher_row(_row())
    assert valid is True
    assert reason is None


def test_validate_teacher_row_rejects_missing_elements() -> None:
    valid, reason = validate_teacher_row(_row(elements=[]))
    assert valid is False
    assert "elements is missing or empty" in str(reason)


def test_validate_teacher_row_rejects_invalid_coordinate_system() -> None:
    valid, reason = validate_teacher_row(_row(coordinate_system="pixels"))
    assert valid is False
    assert "coordinate_system must be normalized_0_1000" in str(reason)


def test_validate_teacher_output_file_reports_invalid_rows(tmp_path: Path) -> None:
    summary = validate_teacher_output_file(_write_rows(tmp_path, [_row(elements=[])]))
    assert summary["valid_rows"] == 0
    assert summary["invalid_rows"] == 1


def test_validate_teacher_output_file_accepts_valid_rows(tmp_path: Path) -> None:
    summary = validate_teacher_output_file(_write_rows(tmp_path, [_row()]))
    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 0
