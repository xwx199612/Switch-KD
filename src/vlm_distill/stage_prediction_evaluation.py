from __future__ import annotations

import json
from pathlib import Path

from .config_schema import PipelineConfig, resolve_label_path, resolve_prediction_path
from .data_manifest import read_jsonl
from .stage_evaluation import (
    _aggregate_prediction_metrics,
    _build_parsing_eval_item,
    exact_match,
    token_f1,
)


def evaluate_predictions(config: PipelineConfig) -> Path:
    prediction_path = resolve_prediction_path(config.data)
    target_path = resolve_label_path(config.data) if config.data.eval_path is None else config.data.eval_path

    prediction_rows = read_jsonl(prediction_path, max_samples=config.data.max_samples)
    target_rows = read_jsonl(target_path, max_samples=config.data.max_samples)
    targets_by_key = {_row_key(row): row for row in target_rows}

    predictions = []
    missing_targets = 0

    for row in prediction_rows:
        target_row = targets_by_key.get(_row_key(row))
        if target_row is None:
            missing_targets += 1
            continue

        prediction = str(row.get("student_answer") or row.get("teacher_answer") or "")
        target = str(target_row.get("teacher_answer") or "")

        item = {
            "id": row["id"],
            "task": row.get("task", target_row.get("task", "parsing")),
            "prediction": prediction,
            "target": target,
            "exact_match": exact_match(prediction, target),
            "token_f1": token_f1(prediction, target),
        }

        if item["task"] == "parsing":
            item.update(_build_parsing_eval_item(prediction=prediction, target=target))

        predictions.append(item)

    metrics = _aggregate_prediction_metrics(predictions, sample_key="num_scored_samples")
    metrics["num_predictions"] = len(prediction_rows)
    metrics["missing_targets"] = missing_targets

    report = {
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "metrics": metrics,
        "predictions": predictions,
    }
    config.evaluation.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.evaluation.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return config.evaluation.output_path


def _row_key(row: dict) -> tuple[str, str]:
    return str(row.get("id", "")).strip(), str(row.get("image", "")).strip()
