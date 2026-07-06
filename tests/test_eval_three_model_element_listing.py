from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_three_model_element_listing.py"
SPEC = importlib.util.spec_from_file_location("eval_three_model_element_listing", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_model_output_accepts_top_level_array() -> None:
    parsed = MODULE.parse_model_output(
        """
        [
          {"name": "Netflix", "type": "app_icon", "bbox": [1, 2, 3, 4], "confidence": 0.9},
          {"name": "YouTube"}
        ]
        """
    )

    assert "parse_error" not in parsed
    assert parsed["elements"] == [
        {
            "element_index": 0,
            "name": "Netflix",
            "name_norm": "netflix",
            "type": "app_icon",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "confidence": 0.9,
        },
        {
            "element_index": 1,
            "name": "YouTube",
            "name_norm": "youtube",
            "type": None,
            "bbox": None,
            "confidence": None,
        },
    ]


def test_compare_single_sample_uses_name_only_matching() -> None:
    rows = MODULE._compare_single_sample(
        image="sample.png",
        sample_id="sample-1",
        reference_elements=[
            {"name": "Settings", "type": "tab", "bbox": [0, 0, 10, 10]},
        ],
        candidate_elements=[
            {"name": "Settings", "type": "button", "bbox": [100, 100, 120, 120]},
        ],
        candidate_role="base8b",
        match_threshold=0.7,
    )

    assert len(rows) == 1
    assert rows[0]["matched"] is True
    assert rows[0]["name_similarity"] == 1.0
