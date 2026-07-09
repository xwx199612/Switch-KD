from __future__ import annotations

import json

from vlm_distill.data_manifest import VlmSample
from vlm_distill.stage_answer_labeling import _label_sample, _normalize_teacher_answer


class _Teacher:
    def __init__(self, answer: str):
        self._answer = answer

    def answer(self, sample: VlmSample) -> dict:
        del sample
        return {
            "teacher_answer": self._answer,
            "teacher_confidence": 1.0,
            "teacher_rationale": "test",
        }


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


def test_invalid_parsing_teacher_answer_is_not_canonicalized() -> None:
    raw_answer = "Search | input | 1 | 2 | 3 | 4 | false"
    assert _normalize_teacher_answer(_sample(), raw_answer) == raw_answer


def test_json_parsing_teacher_answer_is_canonicalized_to_json_schema() -> None:
    raw_answer = json.dumps(
        {
            "elements": [
                {"text": "Search", "type": "input", "bbox_norm": [1, 2, 3, 4], "focused": False},
                {"text": "Home", "bbox_norm": [10, 20, 30, 40], "focused": True},
            ]
        }
    )

    normalized = _normalize_teacher_answer(_sample(), raw_answer)
    assert json.loads(normalized) == {
        "coordinate_system": "normalized_0_1000",
        "elements": [
            {"bbox_norm": [1, 2, 3, 4], "focused": False, "text": "Search"},
            {"bbox_norm": [10, 20, 30, 40], "focused": True, "text": "Home"},
        ],
    }


def test_label_sample_returns_elements_only_parsing_row() -> None:
    answer = json.dumps(
        {
            "elements": [{"text": "Search", "bbox_norm": [1, 2, 3, 4], "focused": False}],
            "coordinate_system": "normalized_0_1000",
        }
    )

    row = _label_sample(_Config(), _Teacher(answer), _sample())

    assert row is not None
    assert set(row.keys()) == {"id", "image", "task", "query", "elements", "coordinate_system"}
    assert row["elements"] == [{"text": "Search", "bbox_norm": [1, 2, 3, 4], "focused": False}]
