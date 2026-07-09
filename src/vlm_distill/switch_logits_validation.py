from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_manifest import read_jsonl
from .token_alignment import build_token_mismatch_details, coerce_token_ids


@dataclass(frozen=True)
class _PayloadReport:
    valid: bool
    reason: str | None
    token_identity_match: bool
    length_match: bool
    vocab_mismatch: bool


@dataclass(frozen=True)
class _RowReport:
    valid: bool
    reason: str | None
    switch_logits_present: bool
    valid_switch_logits: bool
    token_identity_match: bool
    length_match: bool
    vocab_mismatch: bool


def validate_switch_logits_file(
    path: Path,
    *,
    max_samples: int | None = None,
    switch_logits_field: str = "switch_logits",
    teacher_logits_field: str = "teacher_logits",
    bad_limit: int = 5,
) -> dict[str, Any]:
    rows = read_jsonl(path, max_samples=max_samples)
    summary: dict[str, Any] = {
        "path": str(path),
        "total_rows": len(rows),
        "rows_with_switch_logits": 0,
        "valid_switch_logits_rows": 0,
        "token_identity_match_rows": 0,
        "token_identity_mismatch_rows": 0,
        "length_match_rows": 0,
        "length_mismatch_rows": 0,
        "vocab_mismatch_rows": 0,
        "invalid_rows": 0,
        "bad_rows": [],
    }

    bad_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") if isinstance(row, dict) else index)
        report = _validate_row(
            row,
            switch_logits_field=switch_logits_field,
            teacher_logits_field=teacher_logits_field,
        )
        if report.switch_logits_present:
            summary["rows_with_switch_logits"] += 1
        if report.valid_switch_logits:
            summary["valid_switch_logits_rows"] += 1
        if report.token_identity_match:
            summary["token_identity_match_rows"] += 1
        elif report.switch_logits_present or report.reason is not None:
            summary["token_identity_mismatch_rows"] += 1
        if report.length_match:
            summary["length_match_rows"] += 1
        elif report.switch_logits_present or report.reason is not None:
            summary["length_mismatch_rows"] += 1
        if report.vocab_mismatch:
            summary["vocab_mismatch_rows"] += 1
        if not report.valid:
            summary["invalid_rows"] += 1
            if len(bad_rows) < bad_limit:
                bad_rows.append({"id": row_id, "reason": report.reason or "invalid switch logits row"})

    summary["bad_rows"] = bad_rows
    return summary


def _validate_row(
    row: Any,
    *,
    switch_logits_field: str,
    teacher_logits_field: str,
) -> _RowReport:
    if not isinstance(row, dict):
        return _row_report(False, "row is not a JSON object")

    row_id = row.get("id")
    if row_id is None or str(row_id).strip() == "":
        return _row_report(False, "id is missing")

    teacher_tokens = _extract_teacher_tokens(row)
    if not teacher_tokens:
        answer_token_ids = row.get(f"{switch_logits_field}_answer_token_ids")
        if answer_token_ids is not None:
            teacher_tokens = coerce_token_ids(answer_token_ids)
    if not teacher_tokens:
        return _row_report(False, "runtime target token ids are missing")

    switch_report = _validate_compact_logits_payload(
        row,
        field_name=switch_logits_field,
        answer_len=len(teacher_tokens),
        required=True,
    )
    if not switch_report.valid:
        return _row_report(
            False,
            switch_report.reason,
            switch_logits_present=row.get(switch_logits_field) is not None,
            valid_switch_logits=False,
            token_identity_match=switch_report.token_identity_match,
            length_match=switch_report.length_match,
            vocab_mismatch=switch_report.vocab_mismatch,
        )

    if row.get(teacher_logits_field) is not None:
        teacher_report = _validate_compact_logits_payload(
            row,
            field_name=teacher_logits_field,
            answer_len=len(teacher_tokens),
            required=False,
        )
        if not teacher_report.valid:
            return _row_report(
                False,
                teacher_report.reason,
                switch_logits_present=True,
                valid_switch_logits=True,
                token_identity_match=switch_report.token_identity_match,
                length_match=switch_report.length_match,
                vocab_mismatch=switch_report.vocab_mismatch or teacher_report.vocab_mismatch,
            )

    return _row_report(
        True,
        None,
        switch_logits_present=True,
        valid_switch_logits=True,
        token_identity_match=switch_report.token_identity_match,
        length_match=switch_report.length_match,
        vocab_mismatch=switch_report.vocab_mismatch,
    )


def _validate_compact_logits_payload(
    row: dict[str, Any],
    *,
    field_name: str,
    answer_len: int,
    required: bool,
) -> _PayloadReport:
    payload = row.get(field_name)
    if payload is None:
        if required:
            return _payload_report(False, f"{field_name} missing")
        return _payload_report(True, None)
    if not isinstance(payload, dict):
        return _payload_report(False, f"{field_name} missing or not a dict")

    if row.get(f"{field_name}_aligned_to_answer") is not True:
        return _payload_report(False, f"{field_name}_aligned_to_answer is not true")
    if row.get(f"{field_name}_token_identity_match") is not True:
        return _payload_report(False, f"{field_name}_token_identity_match is not true")

    row_vocab_size = row.get(f"{field_name}_vocab_size")
    try:
        row_vocab_size_int = int(row_vocab_size)
    except (TypeError, ValueError):
        return _payload_report(False, f"{field_name}_vocab_size is missing or invalid")
    if row_vocab_size_int <= 0:
        return _payload_report(False, f"{field_name}_vocab_size must be positive")

    answer_token_ids = row.get(f"{field_name}_answer_token_ids")
    if answer_token_ids is None:
        return _payload_report(False, f"{field_name}_answer_token_ids is missing")
    answer_token_ids = coerce_token_ids(answer_token_ids)
    teacher_tokens = _extract_teacher_tokens(row)
    if answer_token_ids != teacher_tokens:
        return _payload_report(
            False,
            (
                f"{field_name} token identity mismatch: "
                f"{build_token_mismatch_details(expected=teacher_tokens, actual=answer_token_ids, actual_field_name='actual_answer_token_id')}"
            ),
        )

    if "shape" not in payload:
        return _payload_report(False, f"{field_name}.shape is missing", token_identity_match=True)
    if "indices" not in payload:
        return _payload_report(False, f"{field_name}.indices is missing", token_identity_match=True)
    values_key = "values" if "values" in payload else "logits" if "logits" in payload else None
    if values_key is None:
        return _payload_report(False, f"{field_name} missing values or logits", token_identity_match=True)

    shape = payload.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        return _payload_report(False, f"{field_name}.shape must be rank 3 [batch, seq, k]", token_identity_match=True)
    try:
        batch_size = int(shape[0])
        seq_len = int(shape[1])
        shape_width = int(shape[2])
    except (TypeError, ValueError):
        return _payload_report(False, f"{field_name}.shape must contain integers", token_identity_match=True)
    if batch_size <= 0 or seq_len <= 0 or shape_width <= 0:
        return _payload_report(False, f"{field_name}.shape dimensions must be positive", token_identity_match=True)
    if seq_len != answer_len:
        return _payload_report(
            False,
            f"{field_name} length mismatch with teacher_tokens",
            token_identity_match=True,
        )

    payload_vocab_size = payload.get("vocab_size", row_vocab_size_int)
    try:
        payload_vocab_size_int = int(payload_vocab_size)
    except (TypeError, ValueError):
        return _payload_report(False, f"{field_name}.vocab_size is invalid", token_identity_match=True, length_match=True)
    if payload_vocab_size_int <= 0:
        return _payload_report(False, f"{field_name}.vocab_size must be positive", token_identity_match=True, length_match=True)
    if payload_vocab_size_int != row_vocab_size_int:
        return _payload_report(
            False,
            f"{field_name} vocab_size mismatch",
            token_identity_match=True,
            length_match=True,
            vocab_mismatch=True,
        )

    indices = payload.get("indices")
    values = payload.get(values_key)
    sequence_error = _validate_compact_sequence(
        indices,
        values,
        field_name=field_name,
        answer_len=answer_len,
        batch_size=batch_size,
        vocab_size=payload_vocab_size_int,
        values_label=values_key,
    )
    if sequence_error is not None:
        return _payload_report(False, sequence_error, token_identity_match=True, length_match=seq_len == answer_len)

    for optional_key in ("token_k", "entropy", "entropy_weight"):
        optional_error = _validate_optional_per_token_matrix(
            payload.get(optional_key),
            field_name=f"{field_name}.{optional_key}",
            answer_len=answer_len,
            batch_size=batch_size,
        )
        if optional_error is not None:
            return _payload_report(False, optional_error, token_identity_match=True, length_match=True)

    return _payload_report(True, None, token_identity_match=True, length_match=True)


def _validate_compact_sequence(
    indices: Any,
    values: Any,
    *,
    field_name: str,
    answer_len: int,
    batch_size: int,
    vocab_size: int,
    values_label: str,
) -> str | None:
    if not isinstance(indices, list) or not isinstance(values, list):
        return f"{field_name}.indices/{values_label} must be lists"
    if len(indices) != batch_size or len(values) != batch_size:
        return f"{field_name}.indices/{values_label} batch shape mismatch"

    for batch_index, (seq_indices, seq_values) in enumerate(zip(indices, values), start=0):
        if not isinstance(seq_indices, list) or not isinstance(seq_values, list):
            return f"{field_name}.indices/{values_label} batch {batch_index} is not a list"
        if len(seq_indices) != answer_len or len(seq_values) != answer_len:
            return f"{field_name} length mismatch with teacher_tokens"

        for position, (position_indices, position_values) in enumerate(zip(seq_indices, seq_values), start=0):
            if not isinstance(position_indices, list) or not isinstance(position_values, list):
                return f"{field_name} position {position} is not a list"
            if len(position_indices) != len(position_values):
                return f"{field_name} indices/{values_label} top-k length mismatch at position {position}"
            if len(position_indices) <= 0:
                return f"{field_name} top-k length is zero at position {position}"

            for token_index in position_indices:
                try:
                    token_index_int = int(token_index)
                except (TypeError, ValueError):
                    return f"{field_name} contains a non-integer token index"
                if token_index_int < 0 or token_index_int >= vocab_size:
                    return f"{field_name} token index out of range"

    return None


def _validate_optional_per_token_matrix(
    value: Any,
    *,
    field_name: str,
    answer_len: int,
    batch_size: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != batch_size:
        return f"{field_name} batch shape mismatch"
    for batch_index, seq_values in enumerate(value, start=0):
        if not isinstance(seq_values, list):
            return f"{field_name} batch {batch_index} is not a list"
        if len(seq_values) != answer_len:
            return f"{field_name} length mismatch"
    return None


def _extract_teacher_tokens(row: dict[str, Any]) -> list[int]:
    if str(row.get("task") or "").strip() == "parsing":
        for field_name in ("teacher_logits", "switch_logits"):
            token_ids = row.get(f"{field_name}_answer_token_ids")
            if token_ids is not None:
                return coerce_token_ids(token_ids)
    tokens = row.get("teacher_tokens")
    if not isinstance(tokens, list):
        return []
    extracted: list[int] = []
    for token in tokens:
        if isinstance(token, bool):
            return []
        try:
            extracted.append(int(token))
        except (TypeError, ValueError):
            return []
    return extracted


def _row_report(
    valid: bool,
    reason: str | None,
    *,
    switch_logits_present: bool = False,
    valid_switch_logits: bool = False,
    token_identity_match: bool = False,
    length_match: bool = False,
    vocab_mismatch: bool = False,
) -> _RowReport:
    return _RowReport(
        valid=valid,
        reason=reason,
        switch_logits_present=switch_logits_present,
        valid_switch_logits=valid_switch_logits,
        token_identity_match=token_identity_match,
        length_match=length_match,
        vocab_mismatch=vocab_mismatch,
    )


def _payload_report(
    valid: bool,
    reason: str | None,
    *,
    token_identity_match: bool = False,
    length_match: bool = False,
    vocab_mismatch: bool = False,
) -> _PayloadReport:
    return _PayloadReport(
        valid=valid,
        reason=reason,
        token_identity_match=token_identity_match,
        length_match=length_match,
        vocab_mismatch=vocab_mismatch,
    )
