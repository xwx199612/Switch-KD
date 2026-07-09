from __future__ import annotations

import json
import re
from typing import Any


COORDINATE_SYSTEM_NORMALIZED_0_1000 = "normalized_0_1000"
_SCHEMA_TOKEN_TEXTS = {
    "text",
    "label",
    "name",
    "title",
    "type",
    "button",
    "tab",
    "icon",
    "app_icon",
    "app-icon",
    "app icon",
    "menu_item",
    "menu-item",
    "menu item",
    "input",
    "unknown",
    "focused",
    "true",
    "false",
    "bbox",
    "bbox_norm",
    "coordinate_system",
    "elements",
}


def parse_parsing_answer(raw_text: str) -> dict[str, Any]:
    try:
        parsed = parse_json_like(raw_text)
    except ValueError as exc:
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error=str(exc),
            coordinate_system=None,
        )

    if not isinstance(parsed, dict):
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error="Parsed JSON must be an object.",
            coordinate_system=None,
        )

    raw_elements = parsed.get("elements")
    if not isinstance(raw_elements, list):
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error="Parsed JSON did not contain an elements list.",
            coordinate_system=None,
        )

    coordinate_system = parsed.get("coordinate_system")
    if coordinate_system is None:
        normalized_coordinate_system = COORDINATE_SYSTEM_NORMALIZED_0_1000
        coordinate_system_error = None
    elif coordinate_system == COORDINATE_SYSTEM_NORMALIZED_0_1000:
        normalized_coordinate_system = COORDINATE_SYSTEM_NORMALIZED_0_1000
        coordinate_system_error = None
    else:
        normalized_coordinate_system = None
        coordinate_system_error = "coordinate_system must be 'normalized_0_1000'"

    elements: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for index, element in enumerate(raw_elements, start=1):
        normalized, error = normalize_element(element)
        if normalized is None:
            parse_errors.append(
                {
                    "row": index,
                    "line": None,
                    "raw_line": None,
                    "error": error,
                }
            )
            continue
        if _should_drop_schema_token_text(normalized["text"]):
            continue
        elements.append(normalized)

    parse_error = None
    if coordinate_system_error is not None:
        parse_error = coordinate_system_error
    elif parse_errors:
        parse_error = str(parse_errors[0]["error"])
    elif not elements:
        parse_error = "No usable elements remained after normalization."

    return _parsed_payload(
        elements=elements,
        parse_errors=parse_errors if coordinate_system_error is None else [
            {
                "row": None,
                "line": None,
                "raw_line": None,
                "error": coordinate_system_error,
            },
            *parse_errors,
        ],
        parse_error=parse_error,
        coordinate_system=normalized_coordinate_system if elements else None,
    )


def serialize_parsing_label(row: dict[str, Any]) -> str:
    elements = row.get("elements")
    if not isinstance(elements, list):
        raise ValueError("serialize_parsing_label requires row['elements'] to be a list.")
    payload = {
        "coordinate_system": COORDINATE_SYSTEM_NORMALIZED_0_1000,
        "elements": [
            {
                "bbox_norm": element["bbox_norm"],
                "focused": element["focused"],
                "text": element["text"],
            }
            for element in elements
            if isinstance(element, dict)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_json_like(raw_text: str) -> object:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty raw text.")

    attempts: list[tuple[str, str]] = [("direct JSON parse", text)]

    unfenced = _strip_markdown_fences(text)
    if unfenced != text:
        attempts.append(("markdown-fence-stripped JSON parse", unfenced))

    object_block = _extract_first_balanced_block(unfenced, "{", "}")
    if object_block is not None and object_block != unfenced:
        attempts.append(("first object-block JSON parse", object_block))

    errors: list[str] = []
    for label, candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{label}: {exc.msg} at line {exc.lineno} column {exc.colno}")

    raise ValueError("; ".join(errors) if errors else "Unable to parse JSON content.")


def normalize_element(element: object) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(element, dict):
        return None, "element is not an object"

    text_value = element.get("text")
    if not isinstance(text_value, str) or not text_value.strip():
        return None, "text is required and must be non-empty"
    text = text_value.strip()

    if "bbox_norm" not in element:
        return None, "bbox_norm is required"
    bbox = _normalize_bbox_value(element.get("bbox_norm"))
    if bbox is None:
        return None, "bbox_norm must be a list of exactly four integers satisfying 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000"

    focused_raw = element.get("focused", False)
    if not isinstance(focused_raw, bool):
        return None, "focused must be boolean"

    return {
        "text": text,
        "bbox_norm": bbox,
        "focused": focused_raw,
    }, None


def _parsed_payload(
    *,
    elements: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    parse_error: str | None,
    coordinate_system: str | None,
) -> dict[str, Any]:
    usable = bool(elements)
    parse_ok = usable and not parse_errors and coordinate_system == COORDINATE_SYSTEM_NORMALIZED_0_1000
    return {
        "parse_ok": parse_ok,
        "usable": usable,
        "parse_error": parse_error,
        "parse_errors": parse_errors,
        "elements": elements,
        "element_count": len(elements),
        "coordinate_system": coordinate_system,
    }


def _strip_markdown_fences(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    if not lines[0].strip().startswith("```") or lines[-1].strip() != "```":
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


def _normalize_bbox_value(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    coords: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        coords.append(item)
    return coords if _bbox_in_range(coords) else None


def _bbox_in_range(bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def _should_drop_schema_token_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized in _SCHEMA_TOKEN_TEXTS


def _build_raw_preview(raw_text: str, limit: int = 200) -> str:
    compact = " ".join(raw_text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
