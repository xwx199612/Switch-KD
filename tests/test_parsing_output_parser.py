from __future__ import annotations

import json

import vlm_distill.parsing_output_parser as parsing_output_parser
from vlm_distill.parsing_output_parser import parse_parsing_answer


def test_parser_accepts_valid_json_schema() -> None:
    raw_text = json.dumps(
        {
            "elements": [
                {"text": "Picture", "bbox_norm": [145, 238, 276, 292], "focused": False},
                {"text": "General", "bbox_norm": [145, 348, 276, 404], "focused": True},
            ],
            "coordinate_system": "normalized_0_1000",
        }
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed == {
        "parse_ok": True,
        "usable": True,
        "parse_error": None,
        "parse_errors": [],
        "elements": [
            {"text": "Picture", "bbox_norm": [145, 238, 276, 292], "focused": False},
            {"text": "General", "bbox_norm": [145, 348, 276, 404], "focused": True},
        ],
        "element_count": 2,
        "coordinate_system": "normalized_0_1000",
        "salvaged": False,
        "salvage_reason": None,
        "dropped_tail_element": False,
    }


def test_parser_rejects_pipe_table_format() -> None:
    parsed = parse_parsing_answer("Picture | card | 1 | 2 | 3 | 4 | false\n")

    assert parsed["parse_ok"] is False
    assert parsed["usable"] is False
    assert parsed["salvaged"] is False
    assert "JSON" in str(parsed["parse_error"])


def test_raw_txt_converter_is_removed() -> None:
    assert not hasattr(parsing_output_parser, "convert_parsing_output_dir")


def test_parser_ignores_type_if_json_includes_it() -> None:
    parsed = parse_parsing_answer(
        json.dumps(
            {
                "elements": [
                    {"text": "Search", "type": "input", "bbox_norm": [1, 2, 3, 4], "focused": False}
                ]
            }
        )
    )

    assert parsed["parse_ok"] is True
    assert parsed["salvaged"] is False
    assert parsed["elements"] == [{"text": "Search", "bbox_norm": [1, 2, 3, 4], "focused": False}]


def test_parser_drops_schema_label_text_rows() -> None:
    parsed = parse_parsing_answer(
        json.dumps(
            {
                "elements": [
                    {"text": "button", "bbox_norm": [1, 2, 3, 4], "focused": False},
                    {"text": "Search", "bbox_norm": [10, 20, 30, 40], "focused": False},
                ]
            }
        )
    )

    assert parsed["parse_ok"] is True
    assert parsed["salvaged"] is False
    assert parsed["element_count"] == 1
    assert parsed["elements"][0]["text"] == "Search"


def test_invalid_bbox_norm_is_rejected() -> None:
    parsed = parse_parsing_answer(
        json.dumps(
            {
                "elements": [
                    {"text": "Search", "bbox_norm": [1, 2, 1, 4], "focused": False}
                ]
            }
        )
    )

    assert parsed["parse_ok"] is False
    assert parsed["usable"] is False
    assert parsed["salvaged"] is False


def test_missing_bbox_norm_is_rejected() -> None:
    parsed = parse_parsing_answer(
        json.dumps(
            {
                "elements": [
                    {"text": "Search", "focused": False}
                ]
            }
        )
    )

    assert parsed["parse_ok"] is False
    assert parsed["usable"] is False
    assert parsed["salvaged"] is False


def test_missing_focused_defaults_to_false() -> None:
    parsed = parse_parsing_answer(
        json.dumps(
            {
                "elements": [
                    {"text": "Search", "bbox_norm": [1, 2, 3, 4]}
                ]
            }
        )
    )

    assert parsed["parse_ok"] is True
    assert parsed["salvaged"] is False
    assert parsed["elements"][0]["focused"] is False


def test_parser_normalizes_safe_json_typography_only() -> None:
    parsed = parse_parsing_answer(
        '{“elements”：[{"text":"Search","bbox_norm":[1,2,3,4],"focused":false}],'
        '“coordinate_system”："normalized_0_1000"}'
    )

    assert parsed["parse_ok"] is True
    assert parsed["salvaged"] is False
    assert parsed["elements"] == [{"text": "Search", "bbox_norm": [1, 2, 3, 4], "focused": False}]


def test_parser_salvages_complete_elements_from_truncated_tail() -> None:
    raw_text = (
        '{\n'
        '  "elements": [\n'
        '    {"text": "Home", "bbox_norm": [250, 180, 300, 225], "focused": true},\n'
        '    {"text": "Broken", "bbox_norm": [200, 745, 3\n'
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed["parse_ok"] is True
    assert parsed["usable"] is True
    assert parsed["salvaged"] is True
    assert parsed["salvage_reason"] == "truncated_tail_element_dropped"
    assert parsed["dropped_tail_element"] is True
    assert parsed["elements"] == [
        {"text": "Home", "bbox_norm": [250, 180, 300, 225], "focused": True}
    ]


def test_salvaged_elements_still_go_through_normalize_element() -> None:
    raw_text = (
        '{\n'
        '  "elements": [\n'
        '    {"text": "  Home  ", "bbox_norm": [250, 180, 300, 225]},\n'
        '    {"text": "Broken", "bbox_norm": [200, 745, 3\n'
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed["parse_ok"] is True
    assert parsed["salvaged"] is True
    assert parsed["elements"] == [
        {"text": "Home", "bbox_norm": [250, 180, 300, 225], "focused": False}
    ]


def test_salvage_does_not_accept_malformed_key_variants() -> None:
    raw_text = (
        '{\n'
        '  "elements": [\n'
        '    {"text": "Home", "box_norm": [250, 180, 300, 225], "focused": true},\n'
        '    {"text": "Broken", "bbox_norm": [200, 745, 3\n'
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed["salvaged"] is True
    assert parsed["parse_ok"] is False
    assert parsed["usable"] is False
    assert parsed["elements"] == []
    assert "bbox_norm is required" in str(parsed["parse_error"])


def test_salvage_fails_when_no_complete_elements_exist() -> None:
    raw_text = (
        '{\n'
        '  "elements": [\n'
        '    {"text": "Broken", "bbox_norm": [200, 745, 3\n'
    )

    parsed = parse_parsing_answer(raw_text)

    assert parsed["parse_ok"] is False
    assert parsed["usable"] is False
    assert parsed["salvaged"] is False
    assert parsed["elements"] == []
