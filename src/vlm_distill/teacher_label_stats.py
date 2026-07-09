from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_manifest import read_jsonl
from .parsing_output_parser import normalize_element


SCHEMA_WORD_ELEMENTS = {
    "text",
    "focused",
    "true",
    "false",
    "elements",
    "bbox_norm",
    "coordinate_system",
}


def summarize_teacher_label_file(path: Path, *, max_samples: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(path, max_samples=max_samples)
    rows_with_elements = 0
    empty_element_rows = 0
    total_elements = 0
    invalid_bbox_count = 0
    focused_true_count = 0
    schema_word_elements = 0
    duplicate_text_count = 0

    for row in rows:
        elements = row.get("elements")
        if not isinstance(elements, list):
            empty_element_rows += 1
            continue
        if elements:
            rows_with_elements += 1
        else:
            empty_element_rows += 1
        seen_texts: set[str] = set()
        for raw_element in elements:
            normalized, error = normalize_element(raw_element)
            if normalized is None:
                if error and "bbox_norm" in error:
                    invalid_bbox_count += 1
                continue
            total_elements += 1
            text = normalized["text"]
            normalized_text = text.casefold()
            if normalized_text in seen_texts:
                duplicate_text_count += 1
            else:
                seen_texts.add(normalized_text)
            if normalized["focused"]:
                focused_true_count += 1
            if text.lower() in SCHEMA_WORD_ELEMENTS:
                schema_word_elements += 1

    return {
        "path": str(path),
        "total_samples": len(rows),
        "rows_with_elements": rows_with_elements,
        "empty_element_rows": empty_element_rows,
        "total_elements": total_elements,
        "avg_elements_per_row": _safe_ratio(total_elements, len(rows)),
        "invalid_bbox_count": invalid_bbox_count,
        "focused_true_count": focused_true_count,
        "schema_word_element_count": schema_word_elements,
        "duplicate_text_count": duplicate_text_count,
    }


def _safe_ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def format_teacher_label_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"path={summary['path']}",
            f"total_samples={summary['total_samples']}",
            f"rows_with_elements={summary['rows_with_elements']}",
            f"empty_element_rows={summary['empty_element_rows']}",
            f"total_elements={summary['total_elements']}",
            f"avg_elements_per_row={summary['avg_elements_per_row']:.4f}",
            f"invalid_bbox_count={summary['invalid_bbox_count']}",
            f"focused_true_count={summary['focused_true_count']}",
            f"schema_word_element_count={summary['schema_word_element_count']}",
            f"duplicate_text_count={summary['duplicate_text_count']}",
        ]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python -m vlm_distill.teacher_label_stats")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    print(format_teacher_label_summary(summarize_teacher_label_file(args.path, max_samples=args.max_samples)))
