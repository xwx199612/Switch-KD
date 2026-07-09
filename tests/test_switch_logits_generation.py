from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.stage_visual_switch_logits import _load_switch_base_rows, _target_text


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            training_manifest_path=tmp_path / "manifest.jsonl",
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd"),
    )


def test_target_text_serializes_parsing_elements():
    row = {
        "task": "parsing",
        "elements": [{"text": "Home", "bbox_norm": [1, 2, 3, 4], "focused": False}],
        "coordinate_system": "normalized_0_1000",
    }

    payload = json.loads(_target_text(row))
    assert payload["elements"][0]["text"] == "Home"
    assert payload["elements"][0]["bbox_norm"] == [1, 2, 3, 4]


def test_load_switch_base_rows_reads_label_rows(tmp_path: Path):
    config = _config(tmp_path)
    config.data.distill_path.write_text(
        '{"id":"row-1","image":"screen.png","task":"parsing","query":"q","elements":[{"text":"Home","bbox_norm":[1,2,3,4],"focused":false}],"coordinate_system":"normalized_0_1000"}\n',
        encoding="utf-8",
    )

    rows = _load_switch_base_rows(config)

    assert len(rows) == 1
    assert rows[0]["id"] == "row-1"
