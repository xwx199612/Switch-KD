from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config_schema import PipelineConfig
from .data_manifest import read_jsonl


def normalize(text: str | None) -> str:
    return " ".join((text or "").lower().strip().split())


def exact_match(prediction: str, target: str) -> float:
    return float(normalize(prediction) == normalize(target))


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize(prediction).split()
    target_tokens = normalize(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(target_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(config: PipelineConfig) -> Path:
    eval_path = config.data.distill_path if config.data.eval_path is None else config.data.eval_path
    rows = read_jsonl(eval_path, max_samples=config.data.max_samples)
    predictions = []

    for row in rows:
        prediction = row.get(config.distillation.target_field) or row.get("teacher_answer") or ""
        target = _build_target(row, config.distillation.target_field)

        item = {
            "id": row["id"],
            "task": row.get("task", "vqa"),
            "prediction": prediction,
            "target": target,
            "exact_match": exact_match(prediction, target),
            "token_f1": token_f1(prediction, target),
        }

        if row.get("task") == "grounding":
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

        if row.get("task") == "screen_parsing":
            pred_json = _parse_json(prediction)
            target_json = _parse_json(target)
            precision, recall, f1 = element_f1(pred_json, target_json)
            item.update(
                {
                    "valid_json": float(pred_json is not None),
                    "element_precision": precision,
                    "element_recall": recall,
                    "element_f1": f1,
                }
            )

        predictions.append(item)

    metrics = {
        "num_samples": len(predictions),
        "exact_match": _mean(item["exact_match"] for item in predictions),
        "token_f1": _mean(item["token_f1"] for item in predictions),
        "valid_json_rate": _mean(item.get("valid_json", 1.0) for item in predictions),
        "mean_iou": _mean(item.get("bbox_iou", 0.0) for item in predictions if item.get("task") == "grounding"),
        "accuracy_iou_50": _mean(item.get("iou_50", 0.0) for item in predictions if item.get("task") == "grounding"),
        "label_match_rate": _mean(item.get("label_match", 0.0) for item in predictions if item.get("task") == "grounding"),
        "element_f1": _mean(item.get("element_f1", 0.0) for item in predictions if item.get("task") == "screen_parsing"),
    }

    report = {"metrics": metrics, "predictions": predictions}
    config.evaluation.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.evaluation.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return config.evaluation.output_path


def _build_target(row: dict[str, Any], target_field: str) -> str:
    if row.get("task") == "grounding" and row.get("bbox"):
        return json.dumps(
            {
                "label": row.get("target_label", "target object"),
                "bbox": row["bbox"],
            },
            ensure_ascii=False,
        )

    if row.get("task") == "screen_parsing" and row.get("elements"):
        return json.dumps(
            {
                "screen_type": row.get("screen_type", "unknown_gui_screen"),
                "elements": row["elements"],
            },
            ensure_ascii=False,
        )

    return row.get("answer") or row.get(target_field) or row.get("teacher_answer") or ""


def _parse_json(text: str | dict | None) -> dict | None:
    if isinstance(text, dict):
        return text
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _extract_bbox(data: dict | None) -> list[float] | None:
    if not data:
        return None
    bbox = data.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(v) for v in bbox]
    target = data.get("target")
    if isinstance(target, dict):
        bbox = target.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return [float(v) for v in bbox]
    return None


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def label_match(pred: dict | None, target: dict | None) -> float:
    if not pred or not target:
        return 0.0
    return float(normalize(pred.get("label")) == normalize(target.get("label")))


def element_f1(pred: dict | None, target: dict | None) -> tuple[float, float, float]:
    pred_labels = _element_labels(pred)
    target_labels = _element_labels(target)

    if not pred_labels and not target_labels:
        return 1.0, 1.0, 1.0
    if not pred_labels or not target_labels:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_labels)
    target_counter = Counter(target_labels)
    common = sum((pred_counter & target_counter).values())

    precision = common / len(pred_labels)
    recall = common / len(target_labels)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _element_labels(data: dict | None) -> list[str]:
    if not data:
        return []
    elements = data.get("elements") or []
    labels = []
    for element in elements:
        if isinstance(element, dict) and element.get("label"):
            labels.append(normalize(element["label"]))
        elif isinstance(element, str):
            labels.append(normalize(element))
    return labels


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0