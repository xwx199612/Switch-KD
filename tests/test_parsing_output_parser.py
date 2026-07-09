from __future__ import annotations

from vlm_distill.parsing_output_parser import parse_parsing_answer


def test_parse_line_format_exact_example() -> None:
    raw_text = (
        "Picture | 145,238,276,292 | false\n"
        "General | 145,348,276,404 | true\n"
        "Network Settings | 705,396,807,432 | false\n"
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed == {
        "parse_ok": True,
        "parse_error": None,
        "elements": [
            {"text": "Picture", "bbox": [145, 238, 276, 292], "focused": False},
            {"text": "General", "bbox": [145, 348, 276, 404], "focused": True},
            {"text": "Network Settings", "bbox": [705, 396, 807, 432], "focused": False},
        ],
        "element_count": 3,
    }


def test_parse_line_format_reports_invalid_bbox_with_line_number() -> None:
    parsed = parse_parsing_answer("Picture | 145,238,276 | false\n")

    assert parsed["parse_ok"] is False
    assert "Line 1" in str(parsed["parse_error"])
    assert "invalid bbox" in str(parsed["parse_error"])


def test_parse_line_format_reports_invalid_focused_value_with_line_number() -> None:
    parsed = parse_parsing_answer("Picture | 145,238,276,292 | maybe\n")

    assert parsed["parse_ok"] is False
    assert "Line 1" in str(parsed["parse_error"])
    assert "invalid focused value" in str(parsed["parse_error"])


def test_parse_line_format_normalizes_focused_variants() -> None:
    raw_text = "\n".join(
        [
            "A | 1,2,3,4 | true",
            "B | 1,2,3,4 | false",
            "C | 1,2,3,4 | True",
            "D | 1,2,3,4 | False",
            "E | 1,2,3,4 | TRUE",
            "F | 1,2,3,4 | FALSE",
            "G | 1,2,3,4 | 1",
            "H | 1,2,3,4 | 0",
            "I | 1,2,3,4 | yes",
            "J | 1,2,3,4 | no",
        ]
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed["parse_ok"] is True
    assert [element["focused"] for element in parsed["elements"]] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
