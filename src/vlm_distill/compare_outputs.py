from __future__ import annotations

from typing import Any


def build_teacher_student_unique_rows(
    *,
    teacher_rows: list[dict[str, Any]],
    student_rows: list[dict[str, Any]],
    teacher_name: str,
    student_name: str,
    keep_empty: bool = True,
) -> list[dict[str, Any]]:
    teacher_by_key = {_row_key(row): row for row in teacher_rows}
    student_by_key = {_row_key(row): row for row in student_rows}
    ordered_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in teacher_rows + student_rows:
        key = _row_key(row)
        if key not in seen:
            ordered_keys.append(key)
            seen.add(key)

    output_rows: list[dict[str, Any]] = []
    for key in ordered_keys:
        teacher_row = teacher_by_key.get(key)
        student_row = student_by_key.get(key)
        if teacher_row is None or student_row is None:
            continue

        teacher_labels = _ordered_labels(teacher_row.get("elements"))
        student_labels = _ordered_labels(student_row.get("elements"))
        teacher_unique = [
            original
            for normalized, original in teacher_labels.items()
            if normalized not in student_labels
        ]
        student_unique = [
            original
            for normalized, original in student_labels.items()
            if normalized not in teacher_labels
        ]
        shared = [
            teacher_labels[normalized]
            for normalized in teacher_labels
            if normalized in student_labels
        ]

        if not keep_empty and not teacher_unique and not student_unique:
            continue

        base_row = teacher_row
        output_rows.append(
            {
                "id": base_row.get("id"),
                "image": base_row.get("image"),
                "task": base_row.get("task"),
                "teacher_name": teacher_name,
                "student_name": student_name,
                "teacher_element_count": len(teacher_labels),
                "student_element_count": len(student_labels),
                "shared_element_count": len(shared),
                "teacher_unique_count": len(teacher_unique),
                "student_unique_count": len(student_unique),
                "shared_elements": shared,
                "teacher_unique_elements": teacher_unique,
                "student_unique_elements": student_unique,
            }
        )

    return output_rows


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id", "")).strip(), str(row.get("image", "")).strip()


def _ordered_labels(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    elements = value

    labels: dict[str, str] = {}
    for element in elements:
        label = _element_label(element)
        if label is None:
            continue
        normalized = _normalize_label(label)
        if normalized not in labels:
            labels[normalized] = label
    return labels


def _element_label(element: Any) -> str | None:
    if isinstance(element, dict):
        value = element.get("text")
        if value is None:
            return None
        label = str(value).strip()
        return label or None
    return None


def _normalize_label(label: str) -> str:
    return "".join(char for char in label.casefold() if char.isalnum())
