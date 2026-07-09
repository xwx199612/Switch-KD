from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .data_manifest import read_jsonl


DecodeTokens = Callable[[list[int]], str]


def validate_teacher_output_file(
    path: Path,
    *,
    max_samples: int | None = None,
    decode_tokens: DecodeTokens | None = None,
    require_teacher_logits: bool = False,
    bad_limit: int = 5,
    logits_field: str = "teacher_logits",
) -> dict[str, Any]:
    del decode_tokens, require_teacher_logits, logits_field
    rows = read_jsonl(path, max_samples=max_samples)
    summary: dict[str, Any] = {
        "path": str(path),
        "total_rows": len(rows),
        "valid_rows": 0,
        "invalid_rows": 0,
        "answer_token_match_rows": 0,
        "answer_token_mismatch_rows": 0,
        "bad_rows": [],
    }

    bad_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or index)
        valid, reason = validate_teacher_row(row)
        if valid:
            summary["valid_rows"] += 1
        else:
            summary["invalid_rows"] += 1
            if len(bad_rows) < bad_limit:
                bad_rows.append({"id": row_id, "reason": str(reason or "invalid teacher row")})

    summary["bad_rows"] = bad_rows
    return summary


def validate_teacher_row(
    row: dict[str, Any],
    *,
    require_teacher_logits: bool = False,
    decode_tokens: DecodeTokens | None = None,
    logits_field: str = "teacher_logits",
) -> tuple[bool, str | None]:
    del require_teacher_logits, decode_tokens, logits_field

    row_id = row.get("id")
    if row_id is None or str(row_id).strip() == "":
        return False, "id is missing"
    if not _has_text(row.get("image")):
        return False, "image is missing"
    if "query" not in row:
        return False, "query is missing"
    if str(row.get("task") or "").strip() == "parsing":
        elements = row.get("elements")
        if not isinstance(elements, list) or not elements:
            return False, "elements is missing or empty"
        if row.get("coordinate_system") != "normalized_0_1000":
            return False, "coordinate_system must be normalized_0_1000"
    return True, None


def build_teacher_token_decoder(config) -> DecodeTokens | None:
    del config
    return None


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
