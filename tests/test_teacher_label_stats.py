from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.teacher_label_stats import format_teacher_label_summary, summarize_teacher_label_file


def test_summarize_teacher_label_file_reports_elements_only_counts(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "".join(
            [
                json.dumps(
                    {
                        "id": "row-1",
                        "image": "a.png",
                        "task": "parsing",
                        "query": "q",
                        "elements": [
                            {"text": "focused", "bbox_norm": [1, 2, 3, 4], "focused": False},
                            {"text": "Home", "bbox_norm": [10, 20, 30, 40], "focused": True},
                            {"text": "Home", "bbox_norm": [50, 60, 70, 80], "focused": False},
                            {"text": "Bad", "bbox_norm": [0, 0, 0, 5], "focused": False},
                        ],
                        "coordinate_system": "normalized_0_1000",
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "id": "row-2",
                        "image": "b.png",
                        "task": "parsing",
                        "query": "q",
                        "elements": [],
                        "coordinate_system": "normalized_0_1000",
                    }
                )
                + "\n",
                json.dumps({"id": "row-3", "image": "c.png", "task": "parsing", "query": "q"}) + "\n",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_teacher_label_file(path)

    assert summary["total_samples"] == 3
    assert summary["rows_with_elements"] == 1
    assert summary["empty_element_rows"] == 2
    assert summary["total_elements"] == 3
    assert summary["avg_elements_per_row"] == 1.0
    assert summary["invalid_bbox_count"] == 1
    assert summary["focused_true_count"] == 1
    assert summary["schema_word_element_count"] == 1
    assert summary["duplicate_text_count"] == 1


def test_format_teacher_label_summary_emits_expected_fields():
    rendered = format_teacher_label_summary(
        {
            "path": "labels.jsonl",
            "total_samples": 5,
            "rows_with_elements": 4,
            "empty_element_rows": 1,
            "total_elements": 20,
            "avg_elements_per_row": 4.0,
            "invalid_bbox_count": 2,
            "focused_true_count": 3,
            "schema_word_element_count": 1,
            "duplicate_text_count": 5,
        }
    )

    assert "total_samples=5" in rendered
    assert "rows_with_elements=4" in rendered
    assert "empty_element_rows=1" in rendered
    assert "total_elements=20" in rendered
    assert "avg_elements_per_row=4.0000" in rendered
    assert "invalid_bbox_count=2" in rendered
    assert "focused_true_count=3" in rendered
    assert "schema_word_element_count=1" in rendered
    assert "duplicate_text_count=5" in rendered
