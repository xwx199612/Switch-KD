from __future__ import annotations

import json
import re
from typing import Any


COORDINATE_SYSTEM_NORMALIZED_0_1000 = "normalized_0_1000"


def is_truncation_error(exc: ValueError) -> bool:
    message = str(exc).casefold()
    return "no complete top-level json object found" in message or "truncated" in message or "unterminated" in message


def recover_truncated_elements_json(raw_text: str) -> dict[str, Any]:
    """Recover complete element objects from a truncated top-level elements array."""
    text = raw_text.lstrip()
    if not text.startswith("{"):
        raise ValueError("No completed elements found in truncated JSON output.")
    match = re.search(r'"elements"\s*:\s*\[', text)
    if not match:
        raise ValueError("No completed elements found in truncated JSON output.")
    start = match.end() - 1
    depth = 0
    in_string = escaped = False
    object_start = None
    elements = []
    array_depth = 1
    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': in_string = False
            continue
        if char == '"': in_string = True
        elif char == "[": array_depth += 1
        elif char == "]":
            array_depth -= 1
            if array_depth == 0: break
        elif char == "{" and depth == 0: object_start = index; depth = 1
        elif char == "{" and depth > 0: depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and object_start is not None:
                try: value = json.loads(text[object_start:index + 1])
                except json.JSONDecodeError: value = None
                if isinstance(value, dict): elements.append(value)
                object_start = None
    if not elements:
        raise ValueError("No completed elements found in truncated JSON output.")
    coordinate_system = (COORDINATE_SYSTEM_NORMALIZED_0_1000
                         if re.search(r'"coordinate_system"\s*:\s*"normalized_0_1000"', text)
                         else None)
    return {"elements": elements, "coordinate_system": coordinate_system}


def normalize_elements(parsed_json: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    elements = parsed_json.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Parsed JSON does not contain an 'elements' list.")
    normalized, skipped = [], []
    for index, element in enumerate(elements, start=1):
        if not isinstance(element, dict): skipped.append(f"element_{index}: not an object"); continue
        text = element.get("text")
        bbox = element.get("bbox_norm")
        focused = element.get("focused")
        if not isinstance(text, str) or not text.strip(): skipped.append(f"element_{index}: missing text"); continue
        if bbox is None: skipped.append(f"element_{index}: malformed bbox_norm"); continue
        if not isinstance(bbox, list) or len(bbox) != 4: skipped.append(f"element_{index}: malformed bbox_norm"); continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox): skipped.append(f"element_{index}: bbox_norm must contain numeric values"); continue
        if any(isinstance(v, float) and not v.is_integer() for v in bbox): skipped.append(f"element_{index}: bbox_norm must contain integers"); continue
        bbox = [int(v) for v in bbox]
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000): skipped.append(f"element_{index}: invalid bbox_norm coordinates"); continue
        if focused is None: focused = False
        if not isinstance(focused, bool): skipped.append(f"element_{index}: focused must be boolean"); continue
        normalized.append({"text": text.strip(), "bbox_norm": bbox, "focused": focused})
    return normalized, skipped
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
        parsed, salvage_metadata = _parse_json_like_with_metadata(raw_text)
    except ValueError as exc:
        try:
            recovered = recover_truncated_elements_json(raw_text)
        except ValueError:
            recovered = None
        if recovered is not None:
            elements, parse_errors = normalize_elements(recovered)
            return _parsed_payload(
                elements=elements,
                parse_errors=[{"row": i, "line": None, "raw_line": None, "error": e}
                              for i, e in enumerate(parse_errors, 1)],
                parse_error=None if elements else "No usable elements remained after normalization.",
                coordinate_system=COORDINATE_SYSTEM_NORMALIZED_0_1000,
                salvaged=True, salvage_reason="truncated_elements_recovered",
                dropped_tail_element=True,
            )
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error=str(exc),
            coordinate_system=None,
            salvaged=False,
            salvage_reason=None,
            dropped_tail_element=False,
        )

    if not isinstance(parsed, dict):
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error="Parsed JSON must be an object.",
            coordinate_system=None,
            salvaged=False,
            salvage_reason=None,
            dropped_tail_element=False,
        )

    raw_elements = parsed.get("elements")
    if not isinstance(raw_elements, list):
        return _parsed_payload(
            elements=[],
            parse_errors=[],
            parse_error="Parsed JSON did not contain an elements list.",
            coordinate_system=None,
            salvaged=False,
            salvage_reason=None,
            dropped_tail_element=False,
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

    normalized_elements, skipped = normalize_elements({"elements": raw_elements})
    elements: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = [
        {"row": index, "line": None, "raw_line": None,
         "error": ("bbox_norm is required" if index <= len(raw_elements)
                    and isinstance(raw_elements[index - 1], dict)
                    and "bbox_norm" not in raw_elements[index - 1] else error)}
        for index, error in enumerate(skipped, 1)
    ]
    for normalized in normalized_elements:
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
        salvaged=salvage_metadata["salvaged"],
        salvage_reason=salvage_metadata["salvage_reason"],
        dropped_tail_element=salvage_metadata["dropped_tail_element"],
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
    parsed, _salvage_metadata = _parse_json_like_with_metadata(raw_text)
    return parsed


def _parse_json_like_with_metadata(raw_text: str) -> tuple[object, dict[str, Any]]:
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
            return json.loads(_normalize_json_typography(candidate)), {
                "salvaged": False,
                "salvage_reason": None,
                "dropped_tail_element": False,
            }
        except json.JSONDecodeError as exc:
            errors.append(f"{label}: {exc.msg} at line {exc.lineno} column {exc.colno}")

    salvaged_json = _salvage_truncated_tail_elements(_normalize_json_typography(unfenced))
    if salvaged_json is not None:
        return json.loads(salvaged_json), {
            "salvaged": True,
            "salvage_reason": "truncated_tail_element_dropped",
            "dropped_tail_element": True,
        }

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
    salvaged: bool,
    salvage_reason: str | None,
    dropped_tail_element: bool,
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
        "salvaged": salvaged,
        "salvage_reason": salvage_reason,
        "dropped_tail_element": dropped_tail_element,
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


def _normalize_json_typography(text: str) -> str:
    return text.replace("“", '"').replace("”", '"').replace("：", ":")


def _salvage_truncated_tail_elements(text: str) -> str | None:
    array_start = _find_elements_array_start(text)
    if array_start is None:
        return None

    complete_objects: list[str] = []
    index = array_start
    expect_element = True

    while True:
        index = _skip_json_whitespace(text, index)
        if expect_element:
            if index >= len(text):
                return None
            if text[index] == "]":
                return None
            if text[index] != "{":
                return None
            object_block, next_index = _extract_balanced_object_at(text, index)
            if object_block is None:
                break
            try:
                parsed_object = json.loads(object_block)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed_object, dict):
                return None
            complete_objects.append(object_block)
            index = next_index
            expect_element = False
            continue

        if index >= len(text):
            return None
        if text[index] == ",":
            index += 1
            expect_element = True
            continue
        if text[index] == "]":
            return None
        return None

    if not complete_objects:
        return None

    elements_payload = ",".join(complete_objects)
    return (
        "{"
        f'"elements":[{elements_payload}],'
        f'"coordinate_system":"{COORDINATE_SYSTEM_NORMALIZED_0_1000}"'
        "}"
    )


def _find_elements_array_start(text: str) -> int | None:
    match = re.search(r'"elements"\s*:', text)
    if match is None:
        return None
    index = _skip_json_whitespace(text, match.end())
    if index >= len(text) or text[index] != "[":
        return None
    return index + 1


def _skip_json_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _extract_balanced_object_at(text: str, start: int) -> tuple[str | None, int]:
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
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1
    return None, start


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
