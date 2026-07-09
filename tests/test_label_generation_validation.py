from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.data_manifest import VlmSample
from vlm_distill.teacher_validation import validate_teacher_output_file
from vlm_distill.stage_answer_labeling import _label_sample, _normalize_teacher_answer


class _TokenizingTeacher:
    def __init__(self, answer: str):
        self._answer = answer

    def answer(self, sample: VlmSample) -> dict:
        return {
            "teacher_answer": self._answer,
            "teacher_tokens": [999],
            "teacher_confidence": 1.0,
            "teacher_rationale": "test",
        }

    def tokenize_teacher_answer(self, answer: str) -> list[int]:
        return [ord(char) for char in answer]

    def decode_teacher_tokens(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _Config:
    class distillation:
        min_teacher_confidence = 0.0


def _sample() -> VlmSample:
    return VlmSample(
        id="screen-1",
        image="screen.jpg",
        task="parsing",
        query="List all visible UI elements.",
    )


def _table_answer() -> str:
    return "\n".join(
        [
            "BEGIN_ELEMENTS",
            "text | type | x1 | y1 | x2 | y2 | focused",
            "Search | input | 1 | 2 | 3 | 4 | false",
            "Home | tab | 10 | 20 | 30 | 40 | true",
            "END_ELEMENTS",
        ]
    )


def test_table_teacher_answer_is_preserved_as_canonical_text() -> None:
    normalized = _normalize_teacher_answer(_sample(), _table_answer())
    assert normalized == _table_answer()


def test_json_teacher_answer_with_valid_bbox_is_canonicalized_to_table() -> None:
    raw_answer = json.dumps(
        {
            "elements": [
                {"text": "Search", "type": "input", "bbox_norm": [1, 2, 3, 4], "focused": False},
                {"text": "Home", "type": "tab", "bbox_norm": [10, 20, 30, 40], "focused": True},
            ]
        }
    )

    normalized = _normalize_teacher_answer(_sample(), raw_answer)
    assert normalized == _table_answer()


def test_teacher_tokens_are_recomputed_after_normalization() -> None:
    row = _label_sample(_Config(), _TokenizingTeacher(_table_answer()), _sample())

    assert row is not None
    assert row["teacher_answer"] == _table_answer()
    assert row["teacher_tokens"] == [ord(char) for char in row["teacher_answer"]]
    assert row["teacher_tokens"] != [999]
    assert row["usable"] is True
    assert row["parse_ok"] is True
    assert row["coordinate_system"] == "normalized_0_1000"


def test_validate_teacher_output_file_checks_decoded_tokens_against_final_answer(tmp_path: Path) -> None:
    answer = _table_answer()
    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(
        json.dumps(
            {
                "id": "screen-1",
                "image": "screen.jpg",
                "query": "List all visible UI elements.",
                "task": "parsing",
                "teacher_answer": answer,
                "teacher_tokens": [ord(char) for char in answer],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_teacher_output_file(
        label_path,
        decode_tokens=lambda tokens: "".join(chr(token) for token in tokens),
    )

    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 0
    assert summary["answer_token_mismatch_rows"] == 0
