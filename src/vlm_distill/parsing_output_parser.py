from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


BEGIN_ELEMENTS_MARKER = "BEGIN_ELEMENTS"
END_ELEMENTS_MARKER = "END_ELEMENTS"
COORDINATE_SYSTEM_NORMALIZED_0_1000 = "normalized_0_1000"
HEADER_FIELDS = ("text", "type", "x1", "y1", "x2", "y2", "focused")
ALLOWED_ELEMENT_TYPES = {
    "button",
    "tab",
    "app_icon",
    "card",
    "menu_item",
    "input",
    "unknown",
}
TEXT_KEYS = ("text", "label", "name", "title")
_TRUE_VALUES = {"1", "true", "yes"}
_FALSE_VALUES = {"0", "false", "no"}


def parse_parsing_answer(raw_text: str) -> dict[str, Any]:
    line_parsed = parse_line_format(raw_text)
    if line_parsed is not None:
        return line_parsed

    try:
        parsed = parse_json_like(raw_text)
    except ValueError as exc:
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error=str(exc),
        )

    elements_raw = _extract_elements(parsed)
    if elements_raw is None:
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error="Parsed JSON did not contain an elements/selectable_elements list.",
        )

    elements: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for index, element in enumerate(elements_raw, start=1):
        normalized, error = normalize_element(element)
        if normalized is not None:
            elements.append(normalized)
            continue
        if error is not None:
            parse_errors.append(
                {
                    "row": index,
                    "line": None,
                    "raw_line": None,
                    "error": error,
                }
            )

    return _parsed_payload(
        elements=elements,
        parse_errors=parse_errors,
        parse_error=str(parse_errors[0]["error"]) if parse_errors else None,
    )


def parse_line_format(raw_text: str) -> dict[str, Any] | None:
    lines = raw_text.splitlines()
    begin_index = next((index for index, line in enumerate(lines) if line.strip() == BEGIN_ELEMENTS_MARKER), None)
    end_index = next((index for index, line in enumerate(lines) if line.strip() == END_ELEMENTS_MARKER), None)
    if begin_index is None and end_index is None:
        return None

    if begin_index is None or end_index is None or end_index <= begin_index:
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error=f"Expected {BEGIN_ELEMENTS_MARKER} ... {END_ELEMENTS_MARKER} block.",
        )

    block_lines = lines[begin_index + 1 : end_index]
    elements: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    saw_header = False

    for offset, raw_line in enumerate(block_lines, start=begin_index + 2):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not saw_header and _is_header_line(stripped):
            saw_header = True
            continue

        element, error = _parse_table_line(stripped)
        if element is not None:
            elements.append(element)
            continue
        if error is not None:
            parse_errors.append(
                {
                    "row": len(elements) + len(parse_errors) + 1,
                    "line": offset,
                    "raw_line": raw_line,
                    "error": error,
                }
            )

    return _parsed_payload(
        elements=elements,
        parse_errors=parse_errors,
        parse_error=str(parse_errors[0]["error"]) if parse_errors else None,
    )


def elements_to_line_format(elements: list[dict[str, Any]]) -> str:
    lines = [BEGIN_ELEMENTS_MARKER, "text | type | x1 | y1 | x2 | y2 | focused"]
    for element in elements:
        serialized = _serialize_element_line(element)
        if serialized is not None:
            lines.append(serialized)
    lines.append(END_ELEMENTS_MARKER)
    return "\n".join(lines)


def parse_json_like(raw_text: str) -> object | None:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty raw text.")

    attempts: list[tuple[str, str]] = []
    attempts.append(("direct JSON parse", text))

    unfenced = _strip_markdown_fences(text)
    if unfenced != text:
        attempts.append(("markdown-fence-stripped JSON parse", unfenced))

    object_block = _extract_first_balanced_block(text, "{", "}")
    if object_block is not None:
        attempts.append(("first object-block JSON parse", object_block))

    list_block = _extract_first_balanced_block(text, "[", "]")
    if list_block is not None:
        attempts.append(("first list-block JSON parse", list_block))

    errors: list[str] = []
    for label, candidate in attempts:
        parsed, repaired = _try_json_candidate(candidate)
        if parsed is not None:
            return parsed
        if repaired is not None:
            return repaired
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{label}: {exc.msg} at line {exc.lineno} column {exc.colno}")

    raise ValueError("; ".join(errors) if errors else "Unable to parse JSON-like content.")


def normalize_element(element: object) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(element, dict):
        return None, "element is not an object"

    text_value: str | None = None
    for key in TEXT_KEYS:
        candidate = element.get(key)
        if candidate is None:
            continue
        candidate_text = str(candidate).strip()
        if candidate_text:
            text_value = candidate_text
            break
    if not text_value:
        return None, "element is missing text"

    normalized_type = _normalize_type_value(element.get("type"))
    bbox = _normalize_bbox_value(element.get("bbox_norm"))
    if bbox is None:
        bbox = _normalize_bbox_value(element.get("bbox"))
    if bbox is None:
        return None, "element is missing a valid normalized bbox"

    focused_value = _normalize_focused_value(element.get("focused", element.get("focus")))
    if focused_value is None:
        return None, "element has invalid focused value"

    return (
        {
            "text": text_value,
            "type": normalized_type,
            "bbox_norm": bbox,
            "focused": focused_value,
        },
        None,
    )


def convert_parsing_output_dir(
    *,
    raw_dir: Path,
    json_dir: Path,
    overwrite: bool = False,
) -> dict[str, int]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir not found: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"raw-dir is not a directory: {raw_dir}")
    if json_dir.exists() and not json_dir.is_dir():
        raise NotADirectoryError(f"json-dir is not a directory: {json_dir}")

    json_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*.txt"))
    failures: list[dict[str, str]] = []
    total_elements = 0
    parse_ok = 0

    for source_path in raw_files:
        output_path = json_dir / f"{source_path.stem}.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Use --overwrite to replace existing JSON files."
            )

        raw_text = source_path.read_text(encoding="utf-8")
        parsed = parse_parsing_answer(raw_text)
        source_file = str(Path(raw_dir.name) / source_path.name).replace("\\", "/")
        output_payload = {
            "source_file": source_file,
            **parsed,
        }

        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if parsed["parse_ok"]:
            parse_ok += 1
            total_elements += int(parsed["element_count"])
        else:
            failures.append(
                {
                    "source_file": source_file,
                    "parse_error": str(parsed["parse_error"]),
                    "raw_preview": _build_raw_preview(raw_text),
                }
            )

    report = {
        "total_files": len(raw_files),
        "parse_ok": parse_ok,
        "parse_failed": len(raw_files) - parse_ok,
        "total_elements": total_elements,
    }

    report_path = json_dir / "parse_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failures_path = json_dir / "parse_failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return report


def _extract_elements(parsed: object) -> list[object] | None:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return None

    elements = parsed.get("elements")
    if isinstance(elements, list):
        return elements

    selectable_elements = parsed.get("selectable_elements")
    if isinstance(selectable_elements, list):
        return selectable_elements

    return None


def _parsed_payload(
    *,
    elements: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    parse_error: str | None,
) -> dict[str, Any]:
    usable = bool(elements)
    parse_ok = usable and not parse_errors
    return {
        "parse_ok": parse_ok,
        "usable": usable,
        "parse_error": parse_error,
        "parse_errors": parse_errors,
        "elements": elements,
        "element_count": len(elements),
        "coordinate_system": COORDINATE_SYSTEM_NORMALIZED_0_1000 if usable else None,
    }


def _parse_table_line(line: str) -> tuple[dict[str, Any] | None, str | None]:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 7:
        return None, "expected exactly 7 fields: text | type | x1 | y1 | x2 | y2 | focused"

    text, raw_type, raw_x1, raw_y1, raw_x2, raw_y2, raw_focused = parts
    if not text:
        return None, "text field is empty"
    if raw_type not in ALLOWED_ELEMENT_TYPES:
        return None, f"invalid type: {raw_type!r}"

    x1 = _parse_coord(raw_x1)
    y1 = _parse_coord(raw_y1)
    x2 = _parse_coord(raw_x2)
    y2 = _parse_coord(raw_y2)
    if None in {x1, y1, x2, y2}:
        return None, "coordinates must be integers only"
    bbox = [x1, y1, x2, y2]
    if not _bbox_in_range(bbox):
        return None, "coordinates must satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000"

    focused = _normalize_focused_value(raw_focused)
    if focused is None:
        return None, "focused must be exactly true or false"

    return {
        "text": text,
        "type": raw_type,
        "bbox_norm": bbox,
        "focused": focused,
    }, None


def _serialize_element_line(element: dict[str, Any]) -> str | None:
    text = str(element.get("text", "")).strip()
    normalized_type = _normalize_type_value(element.get("type"))
    bbox = _normalize_bbox_value(element.get("bbox_norm"))
    focused = element.get("focused")
    if not text or bbox is None or not isinstance(focused, bool):
        return None
    return (
        f"{text} | {normalized_type} | {bbox[0]} | {bbox[1]} | {bbox[2]} | {bbox[3]} | "
        f"{'true' if focused else 'false'}"
    )


def _is_header_line(line: str) -> bool:
    return tuple(part.strip().casefold() for part in line.split("|")) == HEADER_FIELDS


def _try_json_candidate(candidate: str) -> tuple[object | None, object | None]:
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError:
        repaired = _repair_simple_trailing_commas(candidate)
        if repaired == candidate:
            return None, None
        try:
            return None, json.loads(repaired)
        except json.JSONDecodeError:
            return None, None


def _strip_markdown_fences(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    first_line = lines[0].strip()
    last_line = lines[-1].strip()
    if not first_line.startswith("```") or last_line != "```":
        return text

    return "\n".join(lines[1:-1]).strip()


def _extract_first_balanced_block(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _repair_simple_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _normalize_focused_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        return None
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _normalize_type_value(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in ALLOWED_ELEMENT_TYPES:
        return normalized
    return "unknown"


def _normalize_bbox_value(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    coords: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            coord = item
        elif isinstance(item, float) and item.is_integer():
            coord = int(item)
        elif isinstance(item, str) and re.fullmatch(r"-?\d+", item.strip()):
            coord = int(item.strip())
        else:
            return None
        coords.append(coord)
    if not _bbox_in_range(coords):
        return None
    return coords


def _bbox_in_range(bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def _parse_coord(value: str) -> int | None:
    if not re.fullmatch(r"\d+", value):
        return None
    return int(value)


def _build_raw_preview(raw_text: str, limit: int = 200) -> str:
    compact = " ".join(raw_text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
