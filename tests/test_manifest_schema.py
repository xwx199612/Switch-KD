from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.data_manifest import VlmSample, validate_manifest
from vlm_distill.stage_answer_labeling import _format_prompt


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_screen_parsing_manifest_validates_without_question(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "screen.jpg")
    manifest = tmp_path / "screen.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "screen-1",
                "image": "screen.jpg",
                "task": "screen_parsing",
                "query": "List all visible UI elements.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = validate_manifest(manifest, image_root=image_root)
    assert len(samples) == 1
    assert samples[0].query == "List all visible UI elements."
    assert samples[0].target_label is None


def test_grounding_manifest_requires_target_label(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "ground.jpg")
    manifest = tmp_path / "ground.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "ground-1",
                "image": "ground.jpg",
                "task": "grounding",
                "target_label": "YouTube",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = validate_manifest(manifest, image_root=image_root)
    assert len(samples) == 1
    assert samples[0].target_label == "YouTube"


def test_grounding_prompt_formats_target_label(tmp_path: Path):
    config = PipelineConfig(
        data=DataConfig(manifest_path=tmp_path / "manifest.jsonl", distill_path=tmp_path / "distill.jsonl"),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(prompt_template="Target label: {target_label}\nTask: {task}"),
    )
    prompt = _format_prompt(
        config,
        VlmSample(
            id="ground-1",
            image="ground.jpg",
            task="grounding",
            target_label="YouTube",
            target_type="app_icon",
        ),
    )

    assert prompt == "Target label: YouTube\nTask: grounding"
