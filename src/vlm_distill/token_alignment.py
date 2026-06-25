from __future__ import annotations

from typing import Any


def coerce_token_ids(value: Any) -> list[int]:
    if isinstance(value, list) and (not value or not isinstance(value[0], list)):
        return [int(item) for item in value]
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [int(item) for item in value[0]]
    return []


def first_token_mismatch(expected: list[int], actual: list[int]) -> tuple[int | None, int | None, int | None]:
    for index, (expected_token, actual_token) in enumerate(zip(expected, actual)):
        if int(expected_token) != int(actual_token):
            return index, int(expected_token), int(actual_token)
    if len(expected) != len(actual):
        index = min(len(expected), len(actual))
        expected_token = int(expected[index]) if index < len(expected) else None
        actual_token = int(actual[index]) if index < len(actual) else None
        return index, expected_token, actual_token
    return None, None, None


def build_token_mismatch_details(
    *,
    expected: list[int],
    actual: list[int],
    actual_field_name: str,
    extra: dict[str, Any] | None = None,
) -> str:
    mismatch_index, expected_token, actual_token = first_token_mismatch(expected, actual)
    details: list[str] = [
        f"mismatch_index={mismatch_index}",
        f"expected_teacher_token_id={expected_token}",
        f"{actual_field_name}={actual_token}",
        f"expected_len={len(expected)}",
        f"actual_len={len(actual)}",
    ]
    if extra:
        details.extend(f"{key}={value}" for key, value in extra.items())
    return ", ".join(details)
