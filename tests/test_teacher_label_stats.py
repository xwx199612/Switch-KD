from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.teacher_label_stats import format_teacher_label_summary, summarize_teacher_label_file


def test_summarize_teacher_label_file_reports_unknown_empty_and_schema_counts(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "".join(
            [
                json.dumps(
                    {
                        "id": "row-1",
                        "teacher_answer": "\n".join(
                            [
                                "BEGIN_ELEMENTS",
                                "text | type | x1 | y1 | x2 | y2 | focused",
                                "focused | unknown | 1 | 2 | 3 | 4 | false",
                                "Blank | unknown | 10 | 20 | 30 | 40 | false",
                                "END_ELEMENTS",
                            ]
                        ),
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "id": "row-2",
                        "teacher_answer": "\n".join(
                            [
                                "BEGIN_ELEMENTS",
                                "text | type | x1 | y1 | x2 | y2 | focused",
                                "Home | tab | 10 | 20 | 30 | 40 | true",
                                "END_ELEMENTS",
                            ]
                        ),
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_teacher_label_file(path)

    assert summary["total_samples"] == 2
    assert summary["total_elements"] == 3
    assert summary["unknown_type_ratio"] == 2 / 3
    assert summary["empty_elements_ratio"] == 0.0
    assert summary["schema_word_element_count"] == 1


def test_format_teacher_label_summary_emits_expected_fields():
    rendered = format_teacher_label_summary(
        {
            "path": "labels.jsonl",
            "total_samples": 5,
            "total_elements": 20,
            "unknown_type_ratio": 0.25,
            "empty_elements_ratio": 0.10,
            "schema_word_element_count": 3,
        }
    )

    assert "total_samples=5" in rendered
    assert "total_elements=20" in rendered
    assert "unknown_type_ratio=0.2500" in rendered
    assert "empty_elements_ratio=0.1000" in rendered
    assert "schema_word_element_count=3" in rendered
