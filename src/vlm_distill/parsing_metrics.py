"""Shared parsing prediction/reference alignment and metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


def normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _read_bbox(element: dict[str, Any]) -> list[float] | None:
    value = element.get("bbox_norm", element.get("normalized_bbox"))
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return bbox


def _elements(raw: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(raw, list):
        return [], False
    elements: list[dict[str, Any]] = []
    parse_ok = True
    for item in raw:
        if not isinstance(item, dict):
            parse_ok = False
            continue
        text = normalize_text(item.get("text"))
        if not text:
            parse_ok = False
            continue
        elements.append({"text": text, "bbox": _read_bbox(item)})
    return elements, parse_ok


@dataclass(frozen=True)
class ElementPair:
    prediction_index: int
    reference_index: int
    iou: float
    has_bbox: bool


def match_elements(
    prediction_elements: list[dict[str, Any]], reference_elements: list[dict[str, Any]]
) -> list[ElementPair]:
    prediction_by_text: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    reference_by_text: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, element in enumerate(prediction_elements):
        prediction_by_text[element["text"]].append((index, element))
    for index, element in enumerate(reference_elements):
        reference_by_text[element["text"]].append((index, element))

    pairs: list[ElementPair] = []
    for text in prediction_by_text.keys() & reference_by_text.keys():
        candidates = []
        for prediction_index, prediction in prediction_by_text[text]:
            for reference_index, reference in reference_by_text[text]:
                prediction_bbox = prediction["bbox"]
                reference_bbox = reference["bbox"]
                has_bbox = prediction_bbox is not None and reference_bbox is not None
                candidates.append(
                    (bbox_iou(prediction_bbox, reference_bbox), prediction_index, reference_index, has_bbox)
                )
        used_prediction: set[int] = set()
        used_reference: set[int] = set()
        for iou, prediction_index, reference_index, has_bbox in sorted(
            candidates, key=lambda item: item[0], reverse=True
        ):
            if prediction_index in used_prediction or reference_index in used_reference:
                continue
            used_prediction.add(prediction_index)
            used_reference.add(reference_index)
            pairs.append(ElementPair(prediction_index, reference_index, iou, has_bbox))
    return pairs


def score_sample(
    *, prediction_raw: Any, reference_raw: Any, sample_id: str, image: str
) -> dict[str, Any]:
    prediction, parse_ok = _elements(prediction_raw)
    reference, reference_parse_ok = _elements(reference_raw)
    pairs = match_elements(prediction, reference)
    tp = len(pairs)
    fp = len(prediction) - tp
    fn = len(reference) - tp
    precision = 1.0 if not prediction and not reference else (tp / len(prediction) if prediction else 0.0)
    recall = 1.0 if not prediction and not reference else (tp / len(reference) if reference else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Every text match contributes to the pair-weighted mean; invalid/missing
    # boxes contribute IoU=0, while matched_bbox_count records valid pairs.
    bbox_ious = [pair.iou for pair in pairs]
    return {
        "id": sample_id,
        "image": image,
        "reference_element_count": len(reference),
        "prediction_element_count": len(prediction),
        "element_tp": tp,
        "element_fp": fp,
        "element_fn": fn,
        "element_precision": precision,
        "element_recall": recall,
        "element_f1": f1,
        "bbox_iou": sum(bbox_ious) / len(bbox_ious) if bbox_ious else 0.0,
        "matched_bbox_count": sum(pair.has_bbox for pair in pairs),
        "parse_ok": parse_ok,
        "reference_parse_ok": reference_parse_ok,
    }


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(int(sample["element_tp"]) for sample in samples)
    total_fp = sum(int(sample["element_fp"]) for sample in samples)
    total_fn = sum(int(sample["element_fn"]) for sample in samples)
    prediction_count = sum(int(sample["prediction_element_count"]) for sample in samples)
    reference_count = sum(int(sample["reference_element_count"]) for sample in samples)
    matched_pair_count = total_tp
    bbox_sum = sum(
        float(sample["bbox_iou"]) * int(sample["element_tp"]) for sample in samples
    )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 1.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_precision = sum(float(sample["element_precision"]) for sample in samples) / len(samples) if samples else 0.0
    macro_recall = sum(float(sample["element_recall"]) for sample in samples) / len(samples) if samples else 0.0
    return {
        "image_count": len(samples),
        "reference_element_count": reference_count,
        "prediction_element_count": prediction_count,
        "element_tp": total_tp,
        "element_fp": total_fp,
        "element_fn": total_fn,
        "element_precision": precision,
        "element_recall": recall,
        "element_f1": f1,
        "macro_element_precision": macro_precision,
        "macro_element_recall": macro_recall,
        "bbox_iou": bbox_sum / matched_pair_count if matched_pair_count else 0.0,
        "matched_bbox_count": sum(int(sample["matched_bbox_count"]) for sample in samples),
    }
