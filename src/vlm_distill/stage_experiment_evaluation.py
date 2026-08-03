from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .stage_prediction_evaluation import evaluate_prediction_paths


SUMMARY_FIELDS = [
    "experiment", "image_count", "reference_element_count", "prediction_element_count",
    "element_tp", "element_fp", "element_fn", "element_precision", "element_recall",
    "bbox_iou", "matched_bbox_count",
]


def evaluate_experiments(config_path: Path) -> tuple[Path, Path]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    reference_path = _resolve_path(config_path.parent, raw["reference_path"])
    output_dir = Path(raw.get("output_dir", "outputs/evaluation"))
    if not output_dir.is_absolute() and config_path.parent != Path.cwd():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = list(raw.get("experiments", []))
    rows: list[dict[str, Any]] = []

    if raw.get("include_reference", True):
        report_path = output_dir / "32B_eval_report.json"
        rows.append(_evaluate_one("32B", reference_path, reference_path, report_path, True))
    for experiment in experiments:
        name = str(experiment["name"])
        prediction_path = _resolve_path(config_path.parent, experiment["prediction_path"])
        report_path = output_dir / f"{_safe_name(name)}_eval_report.json"
        rows.append(_evaluate_one(name, prediction_path, reference_path, report_path, False))

    csv_path = output_dir / "experiment_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in SUMMARY_FIELDS} for row in rows)
    return output_dir, csv_path


def _evaluate_one(
    name: str, prediction_path: Path, reference_path: Path, report_path: Path, is_reference: bool
) -> dict[str, Any]:
    output = evaluate_prediction_paths(
        prediction_path=prediction_path,
        reference_path=reference_path,
        output_path=report_path,
        experiment_name=name,
        is_reference=is_reference,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    return {"experiment": name, **report["metrics"]}


def _safe_name(name: str) -> str:
    return "_".join(name.strip().split()) or "experiment"


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path
