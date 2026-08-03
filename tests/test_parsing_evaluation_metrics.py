from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_distill.config_schema import DataConfig, EvaluationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.parsing_metrics import aggregate_samples, score_sample
from vlm_distill.stage_evaluation import evaluate
from vlm_distill.stage_experiment_evaluation import evaluate_experiments
from vlm_distill.stage_prediction_evaluation import evaluate_prediction_paths


def element(text: str, bbox: list[int]) -> dict:
    return {"text": text, "bbox_norm": bbox}


def test_exact_match_counts_and_iou() -> None:
    result = score_sample(
        prediction_raw=[element("Settings", [0, 0, 100, 100]), element("Home", [100, 0, 200, 100])],
        reference_raw=[element("settings", [0, 0, 100, 100]), element(" home ", [100, 0, 200, 100])],
        sample_id="1", image="1.png",
    )
    assert result["element_tp"] == 2
    assert result["element_fp"] == result["element_fn"] == 0
    assert result["element_precision"] == result["element_recall"] == 1.0
    assert result["bbox_iou"] == 1.0


def test_missing_and_extra_elements() -> None:
    missing = score_sample(
        prediction_raw=[element("a", [0, 0, 1, 1]), element("b", [0, 0, 1, 1])],
        reference_raw=[element("a", [0, 0, 1, 1]), element("b", [0, 0, 1, 1]), element("c", [0, 0, 1, 1])],
        sample_id="1", image="1.png",
    )
    extra = score_sample(
        prediction_raw=[element("a", [0, 0, 1, 1]), element("b", [0, 0, 1, 1]), element("c", [0, 0, 1, 1])],
        reference_raw=[element("a", [0, 0, 1, 1]), element("b", [0, 0, 1, 1])],
        sample_id="1", image="1.png",
    )
    assert (missing["element_tp"], missing["element_fp"], missing["element_fn"]) == (2, 0, 1)
    assert missing["element_recall"] == 2 / 3
    assert (extra["element_tp"], extra["element_fp"], extra["element_fn"]) == (2, 1, 0)
    assert extra["element_precision"] == 2 / 3


def test_text_match_without_bbox_and_text_mismatch() -> None:
    same_text = score_sample(
        prediction_raw=[element("a", [100, 100, 200, 200])],
        reference_raw=[element("a", [0, 0, 100, 100])], sample_id="1", image="1.png",
    )
    different_text = score_sample(
        prediction_raw=[element("a", [0, 0, 100, 100])],
        reference_raw=[element("b", [0, 0, 100, 100])], sample_id="1", image="1.png",
    )
    assert same_text["element_tp"] == 1 and same_text["bbox_iou"] == 0.0
    assert (different_text["element_tp"], different_text["element_fp"], different_text["element_fn"]) == (0, 1, 1)
    assert different_text["bbox_iou"] == 0.0


def test_duplicate_text_is_one_to_one_and_uses_highest_iou() -> None:
    result = score_sample(
        prediction_raw=[element("x", [0, 0, 10, 10]), element("x", [50, 50, 100, 100])],
        reference_raw=[element("x", [50, 50, 100, 100]), element("x", [0, 0, 10, 10])],
        sample_id="1", image="1.png",
    )
    assert result["element_tp"] == 2
    assert result["matched_bbox_count"] == 2
    assert result["bbox_iou"] == 1.0


def test_empty_and_micro_aggregation() -> None:
    empty = score_sample(prediction_raw=[], reference_raw=[], sample_id="1", image="1.png")
    first = score_sample(
        prediction_raw=[element("a", [0, 0, 1, 1])], reference_raw=[element("a", [0, 0, 1, 1])], sample_id="2", image="2.png"
    )
    second = score_sample(
        prediction_raw=[], reference_raw=[element("b", [0, 0, 1, 1]), element("c", [0, 0, 1, 1])], sample_id="3", image="3.png"
    )
    metrics = aggregate_samples([empty, first, second])
    assert (metrics["element_tp"], metrics["element_fp"], metrics["element_fn"]) == (1, 0, 2)
    assert metrics["element_precision"] == 1.0
    assert metrics["element_recall"] == 1 / 3


def test_same_path_and_missing_samples_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "same.jsonl"
    path.write_text(json.dumps({"id": "1", "image": "a.png", "task": "parsing", "elements": []}) + "\n")
    with pytest.raises(ValueError, match="different files"):
        evaluate_prediction_paths(prediction_path=path, reference_path=path, output_path=tmp_path / "report.json")

    prediction = tmp_path / "prediction.jsonl"
    reference = tmp_path / "reference.jsonl"
    prediction.write_text(json.dumps({"id": "1", "image": "a.png", "task": "parsing", "elements": []}) + "\n")
    reference.write_text(
        "\n".join([
            json.dumps({"id": "1", "image": "a.png", "task": "parsing", "elements": []}),
            json.dumps({"id": "2", "image": "b.png", "task": "parsing", "elements": []}),
        ]) + "\n"
    )
    report_path = evaluate_prediction_paths(prediction_path=prediction, reference_path=reference, output_path=tmp_path / "report.json")
    report = json.loads(report_path.read_text())
    assert report["metrics"]["missing_prediction_samples"] == 1
    assert report["metrics"]["missing_reference_samples"] == 0


def test_batch_evaluation_writes_reports_and_csv(tmp_path: Path) -> None:
    reference = tmp_path / "teacher.jsonl"
    student = tmp_path / "student.jsonl"
    row = {"id": "1", "image": "a.png", "task": "parsing", "elements": [element("a", [0, 0, 10, 10])]}
    reference.write_text(json.dumps(row) + "\n")
    student.write_text(json.dumps(row) + "\n")
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        "reference_path: teacher.jsonl\n"
        "output_dir: evaluation\n"
        "experiments:\n"
        "  - name: A0_R16\n"
        "    prediction_path: student.jsonl\n"
    )
    output_dir, csv_path = evaluate_experiments(config)
    assert (output_dir / "32B_eval_report.json").exists()
    assert (output_dir / "A0_R16_eval_report.json").exists()
    assert "element_tp" in csv_path.read_text()


def test_legacy_evaluate_rejects_parsing_self_comparison(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": "1", "task": "parsing", "elements": []}) + "\n")
    config = PipelineConfig(
        data=DataConfig(training_manifest_path=labels, distill_path=labels, label_path=labels),
        teacher=TeacherConfig(model_name="mock"),
        student=StudentConfig(model_name="mock", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        evaluation=EvaluationConfig(output_path=tmp_path / "report.json"),
    )
    with pytest.raises(ValueError, match="separate prediction and reference"):
        evaluate(config)
