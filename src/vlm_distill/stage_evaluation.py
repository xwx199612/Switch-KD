from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config_schema import PipelineConfig, resolve_label_path
from .data_manifest import read_jsonl
from .parsing_output_parser import normalize_element


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
    eval_path = resolve_label_path(config.data) if config.data.eval_path is None else config.data.eval_path
    rows = read_jsonl(eval_path, max_samples=config.data.max_samples)
    predictions = []

    for row in rows:
        task = row.get("task", "parsing")
        item = {
            "id": row["id"],
            "task": task,
            "prediction": row.get("elements", []) if task == "parsing" else str(row.get("teacher_answer") or ""),
            "target": row.get("elements", []) if task == "parsing" else str(row.get("teacher_answer") or ""),
            "exact_match": None if task == "parsing" else 1.0,
            "token_f1": None if task == "parsing" else 1.0,
        }
        if task == "parsing":
            item.update(
                _build_parsing_eval_item(
                    prediction_elements=row.get("elements", []),
                    target_elements=row.get("elements", []),
                )
            )
        predictions.append(item)

    metrics = _aggregate_prediction_metrics(predictions, sample_key="num_samples")
    report = {"metrics": metrics, "predictions": predictions}
    config.evaluation.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.evaluation.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return config.evaluation.output_path


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


def bbox_center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    return (dx * dx + dy * dy) ** 0.5


def element_f1(pred: dict[str, Any], target: dict[str, Any]) -> tuple[float, float, float]:
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


def _element_labels(parsed: dict[str, Any]) -> list[str]:
    elements = parsed.get("elements") or []
    labels = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        label = element.get("text") or element.get("label") or element.get("name") or element.get("title")
        if label:
            labels.append(normalize(str(label)))
    return labels


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _build_parsing_eval_item(*, prediction_elements: Any, target_elements: Any) -> dict[str, Any]:
    pred_parsed = _parsed_from_elements(prediction_elements)
    target_parsed = _parsed_from_elements(target_elements)
    precision, recall, f1 = element_f1(pred_parsed, target_parsed)
    element_count_abs_diff = abs(
        int(pred_parsed["element_count"]) - int(target_parsed["element_count"])
    )
    focused_accuracy = _focused_accuracy(pred_parsed, target_parsed)
    bbox_iou = _mean(_matching_label_ious(pred_parsed, target_parsed))
    bbox_center_distance = _mean(_matching_label_center_distances(pred_parsed, target_parsed))

    return {
        "parse_ok": float(pred_parsed["parse_ok"]),
        "teacher_parse_ok": float(target_parsed["parse_ok"]),
        "element_precision": precision,
        "element_recall": recall,
        "element_f1": f1,
        "element_count_abs_diff": float(element_count_abs_diff),
        "focused_accuracy": focused_accuracy,
        "bbox_iou": bbox_iou,
        "bbox_center_distance": bbox_center_distance,
        "prediction_element_count": int(pred_parsed["element_count"]),
        "target_element_count": int(target_parsed["element_count"]),
    }


def _aggregate_prediction_metrics(predictions: list[dict[str, Any]], *, sample_key: str) -> dict[str, float]:
    parsing_predictions = [item for item in predictions if item.get("task") == "parsing"]
    return {
        sample_key: len(predictions),
        "exact_match": _mean(item["exact_match"] for item in predictions if item.get("exact_match") is not None),
        "token_f1": _mean(item["token_f1"] for item in predictions if item.get("token_f1") is not None),
        "parse_ok_rate": _mean(item.get("parse_ok", 1.0) for item in parsing_predictions),
        "teacher_parse_ok_rate": _mean(item.get("teacher_parse_ok", 1.0) for item in parsing_predictions),
        "element_f1": _mean(item.get("element_f1", 0.0) for item in parsing_predictions),
        "element_count_abs_diff": _mean(item.get("element_count_abs_diff", 0.0) for item in parsing_predictions),
        "focused_accuracy": _mean(item.get("focused_accuracy", 0.0) for item in parsing_predictions),
        "bbox_iou": _mean(item.get("bbox_iou", 0.0) for item in parsing_predictions),
        "bbox_center_distance": _mean(item.get("bbox_center_distance", 0.0) for item in parsing_predictions),
    }


def _elements_by_label(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    elements_by_label: dict[str, dict[str, Any]] = {}
    for element in parsed.get("elements") or []:
        if not isinstance(element, dict):
            continue
        label = element.get("text") or element.get("label") or element.get("name") or element.get("title")
        if not label:
            continue
        normalized = normalize(str(label))
        elements_by_label.setdefault(normalized, element)
    return elements_by_label


def _focused_accuracy(pred: dict[str, Any], target: dict[str, Any]) -> float:
    pred_by_label = _elements_by_label(pred)
    target_by_label = _elements_by_label(target)
    shared = [label for label in pred_by_label if label in target_by_label]
    if not shared:
        return 0.0
    matches = sum(
        bool(pred_by_label[label].get("focused", False)) == bool(target_by_label[label].get("focused", False))
        for label in shared
    )
    return matches / len(shared)


def _matching_label_ious(pred: dict[str, Any], target: dict[str, Any]) -> list[float]:
    pred_by_label = _elements_by_label(pred)
    target_by_label = _elements_by_label(target)
    ious: list[float] = []
    for label, pred_element in pred_by_label.items():
        target_element = target_by_label.get(label)
        if target_element is None:
            continue
        pred_bbox = _extract_bbox(pred_element)
        target_bbox = _extract_bbox(target_element)
        if pred_bbox is None or target_bbox is None:
            continue
        ious.append(bbox_iou(pred_bbox, target_bbox))
    return ious


def _matching_label_center_distances(pred: dict[str, Any], target: dict[str, Any]) -> list[float]:
    pred_by_label = _elements_by_label(pred)
    target_by_label = _elements_by_label(target)
    distances: list[float] = []
    for label, pred_element in pred_by_label.items():
        target_element = target_by_label.get(label)
        if target_element is None:
            continue
        pred_bbox = _extract_bbox(pred_element)
        target_bbox = _extract_bbox(target_element)
        if pred_bbox is None or target_bbox is None:
            continue
        distances.append(bbox_center_distance(pred_bbox, target_bbox))
    return distances


def _extract_bbox(data: dict[str, Any]) -> list[float] | None:
    bbox = data.get("bbox_norm")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(value) for value in bbox]
    return None


def _parsed_from_elements(raw_elements: Any) -> dict[str, Any]:
    if not isinstance(raw_elements, list):
        return {
            "parse_ok": False,
            "element_count": 0,
            "elements": [],
        }
    elements: list[dict[str, Any]] = []
    for element in raw_elements:
        normalized, _error = normalize_element(element)
        if normalized is not None:
            elements.append(normalized)
    return {
        "parse_ok": bool(elements),
        "element_count": len(elements),
        "elements": elements,
    }
