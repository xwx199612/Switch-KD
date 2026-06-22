from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from vlm_distill.data_manifest import VlmSample, read_jsonl, write_jsonl
from vlm_distill.stage_answer_labeling import _normalize_teacher_answer


def convert_legacy_labeling(
    input_path: Path,
    output_path: Path,
    *,
    drop_student_target: bool = True,
) -> tuple[int, int, int]:
    rows = read_jsonl(input_path)
    converted_rows: list[dict] = []
    copied_from_student_target = 0
    kept_existing_teacher_answer = 0

    for index, row in enumerate(rows, start=1):
        teacher_answer = row.get("teacher_answer")
        student_target = row.get("student_target")

        if teacher_answer is None:
            if student_target is None:
                raise ValueError(
                    f"{input_path}:{index} missing both teacher_answer and student_target"
                )
            teacher_answer = student_target
            copied_from_student_target += 1
        else:
            kept_existing_teacher_answer += 1

        sample = _sample_from_row(row)
        updated_row = dict(row)
        updated_row["teacher_answer"] = _normalize_teacher_answer(sample, str(teacher_answer))

        if drop_student_target:
            updated_row.pop("student_target", None)

        converted_rows.append(updated_row)

    write_jsonl(output_path, converted_rows)
    return len(converted_rows), copied_from_student_target, kept_existing_teacher_answer


def _sample_from_row(row: dict) -> VlmSample:
    known_fields = {
        "id",
        "image",
        "task",
        "query",
        "target_label",
        "target_type",
        "answer",
        "metadata",
    }
    metadata = row.get("metadata")
    merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    for key, value in row.items():
        if key not in known_fields:
            merged_metadata.setdefault(key, value)

    sample = VlmSample(
        id=str(row.get("id", "")),
        image=str(row.get("image", "")),
        task=str(row.get("task", "vqa")),
        query=str(row["query"]) if row.get("query") is not None else None,
        target_label=str(row["target_label"]) if row.get("target_label") is not None else None,
        target_type=str(row["target_type"]) if row.get("target_type") is not None else None,
        answer=row.get("answer"),
        metadata=merged_metadata,
    )
    return VlmSample(**asdict(sample))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy labeling JSONL rows to the teacher_answer-only schema."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the old labeling JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the converted JSONL.")
    parser.add_argument(
        "--keep-student-target",
        action="store_true",
        help="Keep the legacy student_target field instead of removing it.",
    )
    args = parser.parse_args()

    total, copied, kept = convert_legacy_labeling(
        args.input,
        args.output,
        drop_student_target=not args.keep_student_target,
    )
    print(f"Converted: {total}")
    print(f"Copied student_target -> teacher_answer: {copied}")
    print(f"Kept existing teacher_answer: {kept}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
