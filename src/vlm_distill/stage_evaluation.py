from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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
        target = row.get("answer") or row.get(config.distillation.target_field) or ""
        predictions.append(
            {
                "id": row["id"],
                "prediction": prediction,
                "target": target,
                "exact_match": exact_match(prediction, target),
                "token_f1": token_f1(prediction, target),
            }
        )

    metrics = {
        "num_samples": len(predictions),
        "exact_match": _mean(item["exact_match"] for item in predictions),
        "token_f1": _mean(item["token_f1"] for item in predictions),
    }
    report = {"metrics": metrics, "predictions": predictions}
    config.evaluation.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.evaluation.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return config.evaluation.output_path


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
