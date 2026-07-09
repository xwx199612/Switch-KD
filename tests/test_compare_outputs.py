from __future__ import annotations

from vlm_distill.compare_outputs import build_teacher_student_unique_rows


def test_build_teacher_student_unique_rows_reports_counts_and_contents():
    teacher_rows = [
        {
            "id": "parsing-000001",
            "image": "a.png",
            "task": "parsing",
            "elements": [
                {"text": "Home", "bbox_norm": [0, 0, 10, 10], "focused": True},
                {"text": "Search", "bbox_norm": [10, 0, 20, 10], "focused": False},
                {"text": "YouTube", "bbox_norm": [20, 0, 30, 10], "focused": False},
            ],
        }
    ]
    student_rows = [
        {
            "id": "parsing-000001",
            "image": "a.png",
            "task": "parsing",
            "elements": [
                {"text": "Home", "bbox_norm": [0, 0, 10, 10], "focused": True},
                {"text": "Netflix", "bbox_norm": [30, 0, 40, 10], "focused": False},
            ],
        }
    ]

    rows = build_teacher_student_unique_rows(
        teacher_rows=teacher_rows,
        student_rows=student_rows,
        teacher_name="teacher",
        student_name="student",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["teacher_element_count"] == 3
    assert row["student_element_count"] == 2
    assert row["shared_element_count"] == 1
    assert row["teacher_unique_count"] == 2
    assert row["student_unique_count"] == 1
    assert row["teacher_unique_elements"] == ["Search", "YouTube"]
    assert row["student_unique_elements"] == ["Netflix"]
    assert row["shared_elements"] == ["Home"]
