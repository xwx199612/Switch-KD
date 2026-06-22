from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from vlm_distill.data_manifest import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple teacher_labels JSONL files and keep only the "
            "elements each source recognized that the other sources did not."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Two or more teacher_labels JSONL files to compare.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output comparison JSONL file.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop rows whose unique element lists are empty across all compared profiles.",
    )
    args = parser.parse_args()

    input_paths = [Path(value) for value in args.inputs]
    if len(input_paths) < 2:
        raise ValueError("Provide at least two input JSONL files.")

    rows = build_comparison_rows(
        input_paths=input_paths,
        keep_empty=not args.drop_empty,
    )
    output_path = Path(args.output)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} comparison rows to {output_path}")


def build_comparison_rows(
    *,
    input_paths: list[Path],
    keep_empty: bool = False,
) -> list[dict[str, Any]]:
    rows_by_file: dict[Path, dict[tuple[str, str], dict[str, Any]]] = {}
    ordered_keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    for input_path in input_paths:
        file_rows = read_jsonl(input_path)
        keyed_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in file_rows:
            key = _sample_key(row)
            keyed_rows[key] = row
            if key not in seen_keys:
                ordered_keys.append(key)
                seen_keys.add(key)
        rows_by_file[input_path] = keyed_rows

    output_rows: list[dict[str, Any]] = []
    for key in ordered_keys:
        rows_for_key = {
            path: rows_by_file[path][key]
            for path in input_paths
            if key in rows_by_file[path]
        }
        if len(rows_for_key) < 2:
            continue

        normalized_labels_by_file = {
            path: _ordered_labels(row)
            for path, row in rows_for_key.items()
        }
        unique_labels_by_profile: dict[str, list[str]] = {}
        source_file_by_profile: dict[str, str] = {}
        source_resolution_by_profile: dict[str, str] = {}
        source_quantization_by_profile: dict[str, str] = {}

        for current_path, row in rows_for_key.items():
            descriptor = _describe_source_file(current_path)
            other_labels = {
                label
                for path, labels in normalized_labels_by_file.items()
                if path != current_path
                for label in labels.keys()
            }
            unique_labels = [
                original
                for normalized, original in normalized_labels_by_file[current_path].items()
                if normalized not in other_labels
            ]
            profile = descriptor["profile"]
            unique_labels_by_profile[profile] = unique_labels
            source_file_by_profile[profile] = current_path.name
            source_resolution_by_profile[profile] = descriptor["resolution"]
            source_quantization_by_profile[profile] = descriptor["quantization"]

        if not keep_empty and not any(unique_labels_by_profile.values()):
            continue

        base_row = next(iter(rows_for_key.values()))
        output_rows.append(
            _build_output_row(
                row=base_row,
                compared_paths=input_paths,
                unique_labels_by_profile=unique_labels_by_profile,
                source_file_by_profile=source_file_by_profile,
                source_resolution_by_profile=source_resolution_by_profile,
                source_quantization_by_profile=source_quantization_by_profile,
            )
        )

    return output_rows


def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
    sample_id = str(row.get("id", "")).strip()
    image = str(row.get("image", "")).strip()
    return sample_id, image


def _ordered_labels(row: dict[str, Any]) -> dict[str, str]:
    payload = _parse_json_like(row.get("teacher_answer"))
    if not isinstance(payload, dict):
        return {}

    elements = payload.get("elements")
    if elements is None:
        elements = payload.get("selectable_elements")
    if not isinstance(elements, list):
        return {}

    labels: dict[str, str] = {}
    for element in elements:
        label = _element_label(element)
        if label is None:
            continue
        normalized = _normalize_label(label)
        if normalized not in labels:
            labels[normalized] = label
    return labels


def _build_output_row(
    *,
    row: dict[str, Any],
    compared_paths: list[Path],
    unique_labels_by_profile: dict[str, list[str]],
    source_file_by_profile: dict[str, str],
    source_resolution_by_profile: dict[str, str],
    source_quantization_by_profile: dict[str, str],
) -> dict[str, Any]:
    output_row = {
        "image": row.get("image"),
        "task": row.get("task"),
        "unique_element_count_by_profile": {
            profile: len(labels)
            for profile, labels in unique_labels_by_profile.items()
        },
    }
    for profile, labels in unique_labels_by_profile.items():
        output_row[profile] = labels
    return output_row


def _describe_source_file(path: Path) -> dict[str, str]:
    stem = path.stem
    tokens = [token for token in stem.split("_") if token]

    resolution = "unknown"
    quantization = "unknown"
    variant_tokens: list[str] = []

    for token in tokens:
        lower = token.casefold()
        if _is_resolution_token(lower):
            resolution = token
            continue
        if _is_quantization_token(lower):
            quantization = token
            continue
        variant_tokens.append(token)

    if resolution != "unknown":
        variant_tokens.append(resolution)
    if quantization != "unknown":
        variant_tokens.append(quantization)

    variant = "_".join(variant_tokens) if variant_tokens else stem
    profile = _extract_profile(stem, resolution=resolution, quantization=quantization)
    return {
        "variant": variant,
        "profile": profile,
        "resolution": resolution,
        "quantization": quantization,
    }


def _extract_profile(stem: str, *, resolution: str, quantization: str) -> str:
    marker = "teacher_labels_"
    lower_stem = stem.casefold()
    marker_index = lower_stem.find(marker)
    if marker_index >= 0:
        suffix = stem[marker_index + len(marker) :]
        if suffix:
            return suffix

    profile_tokens = [token for token in (resolution, quantization) if token != "unknown"]
    if profile_tokens:
        return "_".join(profile_tokens)
    return stem


def _is_resolution_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d{3,4}p", token))


def _is_quantization_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+bit", token))


def _parse_json_like(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _element_label(element: Any) -> str | None:
    if isinstance(element, str):
        label = element.strip()
        return label or None
    if isinstance(element, dict):
        value = (
            element.get("label")
            or element.get("text")
            or element.get("name")
            or element.get("title")
        )
        if value is None:
            return None
        label = str(value).strip()
        return label or None
    return None


def _normalize_label(label: str) -> str:
    return "".join(char for char in label.casefold() if char.isalnum())


if __name__ == "__main__":
    main()
