from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .data_manifest import read_jsonl
from .stage_teacher_precompute import (
    _canonicalize_teacher_answer,
    _parse_json_object,
    _strip_special_tokens,
)


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
        valid, reason = validate_teacher_row(
            row,
            require_teacher_logits=require_teacher_logits,
            decode_tokens=decode_tokens,
            logits_field=logits_field,
        )
        if valid:
            summary["valid_rows"] += 1
            if decode_tokens is not None:
                summary["answer_token_match_rows"] += 1
        else:
            summary["invalid_rows"] += 1
            if decode_tokens is not None:
                summary["answer_token_mismatch_rows"] += 1
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
    del require_teacher_logits
    del logits_field

    row_id = row.get("id")
    if row_id is None or str(row_id).strip() == "":
        return False, "id is missing"
    if not _has_text(row.get("image")):
        return False, "image is missing"
    if not _has_text(row.get("query")):
        return False, "query is missing"

    answer = row.get("teacher_answer")
    if not isinstance(answer, str) or not answer.strip():
        return False, "teacher_answer is missing or not a string"

    tokens = _extract_teacher_tokens(row)
    if not tokens:
        return False, "teacher_tokens missing or empty"

    raw_answer = str(answer)
    if str(row.get("task") or "").strip() == "parsing":
        try:
            _canonicalize_teacher_answer(_strip_special_tokens(raw_answer))
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
    else:
        parsed = _parse_json_object(raw_answer)
        if parsed is None:
            return False, "teacher_answer is not valid JSON"

    if decode_tokens is not None:
        try:
            decoded = _strip_special_tokens(decode_tokens(tokens))
            canonical_answer = _canonicalize_teacher_answer(_strip_special_tokens(raw_answer))
            canonical_decoded = _canonicalize_teacher_answer(decoded)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        if canonical_answer != canonical_decoded:
            return False, "decoded teacher_tokens do not match teacher_answer"

    return True, None


def build_teacher_token_decoder(config) -> DecodeTokens | None:
    try:
        from transformers import AutoProcessor

        from .model_loading import resolve_model_path
    except ImportError:
        return None

    try:
        processor = AutoProcessor.from_pretrained(
            resolve_model_path(config.teacher.model_name),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        return None

    tokenizer = getattr(processor, "tokenizer", None)
    decoder = tokenizer if tokenizer is not None else processor

    def decode(token_ids: list[int]) -> str:
        return decoder.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    return decode


def _extract_teacher_tokens(row: dict[str, Any]) -> list[int]:
    tokens = row.get("teacher_tokens")
    if not isinstance(tokens, list):
        return []
    extracted: list[int] = []
    for token in tokens:
        try:
            extracted.append(int(token))
        except (TypeError, ValueError):
            return []
    return extracted


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
