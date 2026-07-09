from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.teacher_validation import validate_teacher_output_file, validate_teacher_row


def _valid_answer() -> str:
    return "\n".join(
        [
            "BEGIN_ELEMENTS",
            "text | type | x1 | y1 | x2 | y2 | focused",
            "Home | tab | 10 | 20 | 30 | 40 | true",
            "END_ELEMENTS",
        ]
    )


def _row(*, answer: str | None = None, tokens: list[int] | None = None) -> dict:
    final_answer = answer or _valid_answer()
    final_tokens = tokens or [ord(char) for char in final_answer]
    return {
        "id": "sample-1",
        "image": "screen.png",
        "task": "parsing",
        "query": "List the visible UI elements.",
        "teacher_answer": final_answer,
        "teacher_tokens": final_tokens,
    }


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_validate_teacher_row_accepts_valid_table_answer() -> None:
    valid, reason = validate_teacher_row(_row())
    assert valid is True
    assert reason is None


def test_validate_teacher_row_rejects_invalid_table_answer() -> None:
    broken = "\n".join(
        [
            "BEGIN_ELEMENTS",
            "text | type | x1 | y1 | x2 | y2 | focused",
            "Home | tab | 10 | 20 | 10 | 40 | true",
            "END_ELEMENTS",
        ]
    )
    valid, reason = validate_teacher_row(_row(answer=broken))
    assert valid is False
    assert "canonical table format" in str(reason)


def test_validate_teacher_output_file_reports_token_mismatch(tmp_path: Path) -> None:
    summary = validate_teacher_output_file(
        _write_rows(tmp_path, [_row(tokens=[1, 2, 3])]),
        decode_tokens=lambda _tokens: "broken",
    )
    assert summary["valid_rows"] == 0
    assert summary["invalid_rows"] == 1
    assert summary["answer_token_mismatch_rows"] == 1


def test_validate_teacher_output_file_accepts_matching_tokens(tmp_path: Path) -> None:
    answer = _valid_answer()
    summary = validate_teacher_output_file(
        _write_rows(tmp_path, [_row(answer=answer, tokens=[ord(char) for char in answer])]),
        decode_tokens=lambda tokens: "".join(chr(token) for token in tokens),
    )
    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 0
