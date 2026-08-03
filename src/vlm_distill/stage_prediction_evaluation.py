from __future__ import annotations

import json
from pathlib import Path

from .config_schema import PipelineConfig, resolve_label_path, resolve_prediction_path
from .data_manifest import read_jsonl
from .parsing_metrics import aggregate_samples, score_sample
from .stage_evaluation import exact_match, token_f1


def evaluate_predictions(config: PipelineConfig) -> Path:
    prediction_path = resolve_prediction_path(config.data)
    reference_path = (
        config.data.reference_path
        or config.data.eval_path
        or resolve_label_path(config.data)
    )
    return evaluate_prediction_paths(
        prediction_path=prediction_path,
        reference_path=reference_path,
        output_path=config.evaluation.output_path,
        max_samples=config.data.max_samples,
        output_mode=config.pipeline.output_mode,
    )


def evaluate_prediction_paths(
    *, prediction_path: Path, reference_path: Path, output_path: Path,
    max_samples: int | None = None, experiment_name: str | None = None,
    is_reference: bool = False, output_mode: str = "parsing",
) -> Path:
    if not is_reference:
        _ensure_distinct_paths(prediction_path, reference_path)
    prediction_rows = read_jsonl(prediction_path, max_samples=max_samples)
    reference_rows = read_jsonl(reference_path, max_samples=max_samples)
    prediction_index, duplicate_prediction_keys = _index_rows(prediction_rows)
    reference_index, duplicate_reference_keys = _index_rows(reference_rows)
    all_keys = list(dict.fromkeys([*prediction_index, *reference_index]))

    samples: list[dict] = []
    for key in all_keys:
        prediction_row = prediction_index.get(key, {})
        reference_row = reference_index.get(key, {})
        sample_id = str(prediction_row.get("id") or reference_row.get("id") or "")
        image = _normalized_image(prediction_row.get("image") or reference_row.get("image"))
        if output_mode == "parsing":
            sample = score_sample(
                prediction_raw=prediction_row.get("elements", []),
                reference_raw=reference_row.get("elements", []),
                sample_id=sample_id,
                image=image,
            )
        else:
            prediction = str(prediction_row.get("student_answer") or prediction_row.get("teacher_answer") or "")
            target = str(reference_row.get("teacher_answer") or "")
            sample = {
                "id": sample_id,
                "image": image,
                "prediction": prediction,
                "target": target,
                "exact_match": exact_match(prediction, target),
                "token_f1": token_f1(prediction, target),
            }
        sample["missing_prediction"] = key not in prediction_index
        sample["missing_reference"] = key not in reference_index
        samples.append(sample)

    parsing_samples = samples if output_mode == "parsing" else []
    metrics = aggregate_samples(parsing_samples) if parsing_samples else {
        "image_count": 0, "reference_element_count": 0, "prediction_element_count": 0,
        "element_tp": 0, "element_fp": 0, "element_fn": 0,
        "element_precision": 0.0, "element_recall": 0.0, "element_f1": 0.0,
        "bbox_iou": 0.0, "matched_bbox_count": 0,
    }
    metrics.update({
        "num_predictions": len(prediction_rows),
        "num_scored_samples": len(samples),
        "exact_match": _mean_optional(samples, "exact_match"),
        "token_f1": _mean_optional(samples, "token_f1"),
        "matched_samples": sum(key in prediction_index and key in reference_index for key in all_keys),
        "missing_prediction_samples": sum(key not in prediction_index for key in all_keys),
        "missing_reference_samples": sum(key not in reference_index for key in all_keys),
        "duplicate_prediction_keys": len(duplicate_prediction_keys),
        "duplicate_reference_keys": len(duplicate_reference_keys),
    })
    report = {
        "reference": {"type": "32B_teacher", "path": str(reference_path)},
        "prediction": {"path": str(prediction_path)},
        "is_reference": is_reference,
        "metrics": metrics,
        "samples": samples,
    }
    if experiment_name is not None:
        report["experiment"] = experiment_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def _normalized_image(value: object) -> str:
    return "/".join(
        part for part in str(value or "").replace("\\", "/").split("/")
        if part and part != "."
    )


def _row_key(row: dict) -> tuple[str, str]:
    row_id = str(row.get("id") or "").strip()
    return ("id", row_id) if row_id else ("image", _normalized_image(row.get("image")))


def _index_rows(rows: list[dict]) -> tuple[dict[tuple[str, str], dict], list[tuple[str, str]]]:
    indexed: dict[tuple[str, str], dict] = {}
    duplicates: list[tuple[str, str]] = []
    for row in rows:
        key = _row_key(row)
        if key in indexed:
            duplicates.append(key)
        else:
            indexed[key] = row
    return indexed, duplicates


def _ensure_distinct_paths(prediction_path: Path, reference_path: Path) -> None:
    if prediction_path.resolve() == reference_path.resolve():
        raise ValueError("Prediction and reference paths must be different files.")


def _mean_optional(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0
