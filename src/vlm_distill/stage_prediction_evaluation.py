from __future__ import annotations

import json
from pathlib import Path

from .config_schema import PipelineConfig, resolve_label_path, resolve_prediction_path
from .data_manifest import read_jsonl
from .stage_evaluation import (
    _build_target,
    _extract_bbox,
    _mean,
    _parsing_eval_payload,
    _parsing_eval_target_payload,
    _parse_json,
    bbox_iou,
    element_f1,
    exact_match,
    label_match,
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

        prediction = row.get("student_answer") or row.get("teacher_answer") or ""
        target = _build_target(target_row)

        item = {
            "id": row["id"],
            "task": row.get("task", target_row.get("task", "vqa")),
            "prediction": prediction,
            "target": target,
            "exact_match": exact_match(prediction, target),
            "token_f1": token_f1(prediction, target),
        }

        if item["task"] == "grounding":
            pred_json = _parse_json(prediction)
            target_json = _parse_json(target)
            pred_bbox = _extract_bbox(pred_json)
            target_bbox = _extract_bbox(target_json)
            iou = bbox_iou(pred_bbox, target_bbox) if pred_bbox and target_bbox else 0.0
            item.update(
                {
                    "valid_json": float(pred_json is not None),
                    "bbox_iou": iou,
                    "iou_50": float(iou >= 0.5),
                    "label_match": label_match(pred_json, target_json),
                }
            )

        if item["task"] == "parsing":
            pred_json = _parsing_eval_payload(row, prefix="student", answer_field="student_answer")
            target_json = _parsing_eval_target_payload(target_row)
            precision, recall, f1 = element_f1(pred_json, target_json)
            parse_ok = float(pred_json is not None)
            item.update(
                {
                    "valid_json": parse_ok,
                    "parse_ok": parse_ok,
                    "element_precision": precision,
                    "element_recall": recall,
                    "element_f1": f1,
                }
            )

        predictions.append(item)

    metrics = {
        "num_predictions": len(prediction_rows),
        "num_scored_samples": len(predictions),
        "missing_targets": missing_targets,
        "exact_match": _mean(item["exact_match"] for item in predictions),
        "token_f1": _mean(item["token_f1"] for item in predictions),
        "valid_json_rate": _mean(item.get("valid_json", 1.0) for item in predictions),
        "parse_ok_rate": _mean(item.get("parse_ok", 1.0) for item in predictions),
        "mean_iou": _mean(item.get("bbox_iou", 0.0) for item in predictions if item.get("task") == "grounding"),
        "accuracy_iou_50": _mean(item.get("iou_50", 0.0) for item in predictions if item.get("task") == "grounding"),
        "label_match_rate": _mean(item.get("label_match", 0.0) for item in predictions if item.get("task") == "grounding"),
        "element_f1": _mean(item.get("element_f1", 0.0) for item in predictions if item.get("task") == "parsing"),
    }

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
