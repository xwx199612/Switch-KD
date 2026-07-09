from __future__ import annotations

from vlm_distill.parsing_output_parser import parse_parsing_answer


def test_parse_table_format_exact_example() -> None:
    raw_text = "\n".join(
        [
            "BEGIN_ELEMENTS",
            "text | type | x1 | y1 | x2 | y2 | focused",
            "Picture | card | 145 | 238 | 276 | 292 | false",
            "General | button | 145 | 348 | 276 | 404 | true",
            "Network Settings | menu_item | 705 | 396 | 807 | 432 | false",
            "END_ELEMENTS",
        ]
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed == {
        "parse_ok": True,
        "usable": True,
        "parse_error": None,
        "parse_errors": [],
        "elements": [
            {"text": "Picture", "type": "card", "bbox_norm": [145, 238, 276, 292], "focused": False},
            {"text": "General", "type": "button", "bbox_norm": [145, 348, 276, 404], "focused": True},
            {"text": "Network Settings", "type": "menu_item", "bbox_norm": [705, 396, 807, 432], "focused": False},
        ],
        "element_count": 3,
        "coordinate_system": "normalized_0_1000",
    }


def test_parse_table_format_keeps_valid_rows_when_one_row_fails() -> None:
    raw_text = "\n".join(
        [
            "BEGIN_ELEMENTS",
            "text | type | x1 | y1 | x2 | y2 | focused",
            "Picture | card | 145 | 238 | 276 | 292 | false",
            "Broken | card | 145 | 238 | 145 | 292 | false",
            "General | button | 145 | 348 | 276 | 404 | true",
            "END_ELEMENTS",
        ]
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed["parse_ok"] is False
    assert parsed["usable"] is True
    assert parsed["element_count"] == 2
    assert [element["text"] for element in parsed["elements"]] == ["Picture", "General"]
    assert len(parsed["parse_errors"]) == 1
    assert "0 <= x1 < x2 <= 1000" in str(parsed["parse_errors"][0]["error"])


def test_parse_table_format_skips_optional_header_line() -> None:
    raw_text = "\n".join(
        [
            "BEGIN_ELEMENTS",
            "text | type | x1 | y1 | x2 | y2 | focused",
            "A | unknown | 1 | 2 | 3 | 4 | true",
            "END_ELEMENTS",
        ]
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed["parse_ok"] is True
    assert parsed["elements"][0]["focused"] is True


def test_parse_table_format_requires_markers() -> None:
    parsed = parse_parsing_answer("Picture | card | 1 | 2 | 3 | 4 | false\n")

    assert parsed["parse_ok"] is False
    assert parsed["usable"] is False
