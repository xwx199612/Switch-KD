from __future__ import annotations

from vlm_distill.compare_outputs import build_teacher_student_unique_rows


def test_build_teacher_student_unique_rows_reports_counts_and_contents():
    teacher_rows = [
        {
            "id": "parsing-000001",
            "image": "a.png",
            "task": "parsing",
            "teacher_answer": '{"elements":["Home","Search","YouTube"]}',
        }
    ]
    student_rows = [
        {
            "id": "parsing-000001",
            "image": "a.png",
            "task": "parsing",
            "student_answer": '{"elements":["Home","Netflix"]}',
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
