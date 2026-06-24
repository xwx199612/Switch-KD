from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.label_validation import validate_label_rows, validate_teacher_row


def _answer() -> str:
    return '{"elements":[{"text":"Home","type":"tab","focused":true}]}'


def _logits(length: int) -> dict:
    return {
        "indices": [[[0] for _ in range(length)]],
        "values": [[[1.0] for _ in range(length)]],
        "shape": [1, length, 8],
        "vocab_size": 8,
    }


def test_stale_row_without_teacher_logits_is_invalid_when_required():
    row = {
        "id": "sample-1",
        "teacher_answer": _answer(),
        "teacher_tokens": [1, 2, 3],
    }

    valid, reason = validate_teacher_row(row, require_logits=True)

    assert valid is False
    assert "teacher_logits" in str(reason)


def test_valid_unified_teacher_row_passes(tmp_path: Path):
    row = {
        "id": "sample-1",
        "teacher_answer": _answer(),
        "teacher_tokens": [1, 2, 3],
        "teacher_logits": _logits(3),
        "teacher_logits_aligned_to_answer": True,
    }
    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = validate_label_rows(label_path, require_logits=True)

    assert summary["schema_valid_rows"] == 1
    assert summary["valid_teacher_logits_rows"] == 1
    assert summary["answer_logits_length_mismatch_rows"] == 0
