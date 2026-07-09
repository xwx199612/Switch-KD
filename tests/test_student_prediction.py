from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vlm_distill.config_schema import (
    DataConfig,
    EvaluationConfig,
    PipelineConfig,
    StudentConfig,
    TeacherConfig,
)
from vlm_distill.data_manifest import read_jsonl, validate_manifest
from vlm_distill.stage_prediction_evaluation import evaluate_predictions
from vlm_distill.stage_student_prediction import create_student_predictions


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_create_student_predictions_writes_mock_predictions(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "sample.jpg")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "image": "sample.jpg",
                "task": "parsing",
                "query": "List all visible UI elements.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = PipelineConfig(
        data=DataConfig(
            training_manifest_path=manifest,
            manifest_path=manifest,
            distill_path=tmp_path / "distill.jsonl",
            prediction_path=tmp_path / "predictions.jsonl",
            image_root=image_root,
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
    )

    samples = validate_manifest(manifest, image_root=image_root)
    output_path = create_student_predictions(config, samples)
    rows = read_jsonl(output_path)

    assert rows[0]["elements"]
    assert rows[0]["coordinate_system"] == "normalized_0_1000"


def test_evaluate_predictions_scores_against_eval_labels(tmp_path: Path):
    prediction_path = tmp_path / "predictions.jsonl"
    eval_path = tmp_path / "labels.jsonl"
    report_path = tmp_path / "eval_report.json"

    prediction_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "image": "sample.jpg",
                "task": "vqa",
                "student_answer": "a white square",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "image": "sample.jpg",
                "task": "vqa",
                "teacher_answer": "a white square",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = PipelineConfig(
        data=DataConfig(
            training_manifest_path=tmp_path / "manifest.jsonl",
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "distill.jsonl",
            prediction_path=prediction_path,
            eval_path=eval_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        evaluation=EvaluationConfig(output_path=report_path),
    )

    output_path = evaluate_predictions(config)
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["metrics"]["num_predictions"] == 1
    assert report["metrics"]["num_scored_samples"] == 1
    assert report["metrics"]["exact_match"] == 1.0


def test_create_student_predictions_writes_parsing_sidecars(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "sample.jpg")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "parsing-000001",
                "image": "sample.jpg",
                "task": "parsing",
                "query": "List all visible UI elements.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = PipelineConfig(
        data=DataConfig(
            training_manifest_path=manifest,
            manifest_path=manifest,
            distill_path=tmp_path / "distill.jsonl",
            prediction_path=tmp_path / "predictions.jsonl",
            image_root=image_root,
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
    )

    samples = validate_manifest(manifest, image_root=image_root)
    output_path = create_student_predictions(config, samples)
    rows = read_jsonl(output_path)

    assert rows[0]["elements"]
    assert (tmp_path / "json" / "student" / "parsing-000001.json").exists()
    assert not (tmp_path / "raw" / "student" / "parsing-000001.txt").exists()
