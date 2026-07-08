from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


ALLOWED_ELEMENT_TYPES = {
    "tab",
    "button",
    "app_icon",
    "app_tile",
    "menu_item",
    "tile",
    "toggle",
    "input",
    "icon",
    "link",
    "other",
    "unknown",
}

TEXT_KEYS = ("text", "label", "name", "title")


def parse_parsing_answer(raw_text: str) -> dict[str, Any]:
    try:
        parsed = parse_json_like(raw_text)
    except ValueError as exc:
        return {
            "parse_ok": False,
            "parse_error": str(exc),
            "elements": [],
            "element_count": 0,
        }

    elements_raw = _extract_elements(parsed)
    if elements_raw is None:
        return {
            "parse_ok": False,
            "parse_error": "Parsed JSON did not contain an elements/selectable_elements list.",
            "elements": [],
            "element_count": 0,
        }

    elements: list[dict[str, Any]] = []
    for element in elements_raw:
        normalized = normalize_element(element)
        if normalized is not None:
            elements.append(normalized)

    return {
        "parse_ok": True,
        "parse_error": None,
        "elements": elements,
        "element_count": len(elements),
    }


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


def normalize_element(element: object) -> dict[str, Any] | None:
    if not isinstance(element, dict):
        return None

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
        return None

    focused = _normalize_focused_value(element.get("focused", False))

    raw_type = element.get("type")
    normalized_type = str(raw_type).strip() if raw_type is not None else ""
    if normalized_type not in ALLOWED_ELEMENT_TYPES:
        normalized_type = "other"

    return {
        "text": text_value,
        "type": normalized_type,
        "focused": focused,
    }


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


def _normalize_focused_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return bool(value) if isinstance(value, int) and value in (0, 1) else False


def _build_raw_preview(raw_text: str, limit: int = 200) -> str:
    compact = " ".join(raw_text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
