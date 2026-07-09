from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import yaml

from . import teacher_validation
from .config_schema import (
    load_config,
    resolve_inference_manifest_path,
    resolve_label_path,
    resolve_prediction_path,
    resolve_switch_logits_path,
    resolve_training_manifest_path,
)
from .data_manifest import validate_manifest
from .hf_runtime import configure_hf_offline_mode
from .manifest_builder import create_manifest_from_config, infer_manifest_task_from_config_path
from .model_output_artifacts import refresh_parsing_sidecar_reports
from .stage_evaluation import evaluate
from .stage_merge_adapter import merge_student_adapter
from .stage_prediction_evaluation import evaluate_predictions
from .stage_student_prediction import create_student_predictions
from .stage_teacher_precompute import create_teacher_precompute_dataset
from .stage_student_training import train_student
from .stage_visual_switch_logits import create_visual_switch_dataset
from .switch_logits_validation import validate_switch_logits_file
from .teacher_label_stats import format_teacher_label_summary, summarize_teacher_label_file


def main() -> None:
    configure_hf_offline_mode()

    parser = argparse.ArgumentParser(prog="vlm-distill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_manifest_parser = subparsers.add_parser("create-manifest")
    create_manifest_parser.add_argument("--config", type=Path, required=True)
    create_manifest_parser.add_argument(
        "--split",
        choices=("training", "inference"),
        required=True,
    )
    create_manifest_parser.add_argument(
        "--task",
        choices=("parsing",),
    )
    create_manifest_parser.add_argument("--recursive", action="store_true")

    parse_outputs_parser = subparsers.add_parser("parse-parsing-outputs")
    parse_outputs_parser.add_argument("--output-root", type=Path, required=True)
    parse_outputs_parser.add_argument("--role", choices=("teacher", "student"), required=True)

    for command in (
        "validate-manifest",
        "label",
        "teacher-precompute",
        "predict",
        "switch-logits",
        "validate-switch-logits",
        "train",
        "merge-adapter",
        "evaluate",
        "evaluate-predictions",
    ):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)
        if command == "validate-manifest":
            command_parser.add_argument(
                "--split",
                choices=("training", "inference"),
                default="training",
            )
    validate_teacher_parser = subparsers.add_parser("validate-teacher")
    validate_teacher_parser.add_argument("--config", type=Path, required=True)
    teacher_stats_parser = subparsers.add_parser("teacher-label-stats")
    teacher_stats_parser.add_argument("--config", type=Path, required=True)

    validate_labels_parser = subparsers.add_parser(
        "validate-labels",
        help=argparse.SUPPRESS,
    )
    validate_labels_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "create-manifest":
        config = load_config(args.config)
        task = _resolve_create_manifest_task(
            config_path=args.config,
            cli_task=args.task,
        )

        output_path = create_manifest_from_config(
            config=config,
            task=task,
            split=args.split,
            recursive=args.recursive,
        )
        print(f"OK manifest written: {output_path}")
        return

    if args.command == "parse-parsing-outputs":
        report = refresh_parsing_sidecar_reports(
            output_root=args.output_root,
            role=args.role,
        )
        print(
            "OK refreshed parsing sidecar reports "
            f"total_files={report['total_files']} parse_ok={report['parse_ok']} "
            f"parse_failed={report['parse_failed']} total_elements={report['total_elements']} "
            f"json_dir={args.output_root / 'json' / args.role}"
        )
        return

    config = load_config(args.config)

    if args.command == "validate-manifest":
        if args.split == "inference":
            manifest_path = resolve_inference_manifest_path(config.data)
        else:
            manifest_path = resolve_training_manifest_path(config.data)
        samples = validate_manifest(
            manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        print(
            "OK validated manifest "
            f"split={args.split} samples={len(samples)} path={manifest_path}"
        )
        return

    if args.command == "validate-teacher":
        decoder = teacher_validation.build_teacher_token_decoder(config)
        summary = teacher_validation.validate_teacher_output_file(
            resolve_label_path(config.data),
            max_samples=config.data.max_samples,
            decode_tokens=decoder,
        )
        _print_teacher_validation_summary(summary)
        if summary["invalid_rows"]:
            raise SystemExit(1)
        return

    if args.command == "teacher-label-stats":
        summary = summarize_teacher_label_file(
            resolve_label_path(config.data),
            max_samples=config.data.max_samples,
        )
        print(format_teacher_label_summary(summary))
        return

    if args.command == "validate-labels":
        raise SystemExit("validate-labels is deprecated. Use validate-teacher.")

    if args.command == "label":
        manifest_path = resolve_training_manifest_path(config.data)
        samples = validate_manifest(
            manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        output_path = create_teacher_precompute_dataset(config, samples)
        print(f"OK teacher precompute dataset written: {output_path}")
        return

    if args.command == "teacher-precompute":
        manifest_path = resolve_training_manifest_path(config.data)
        samples = validate_manifest(
            manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        output_path = create_teacher_precompute_dataset(config, samples)
        print(f"OK teacher precompute dataset written: {output_path}")
        return

    if args.command == "predict":
        manifest_path = resolve_inference_manifest_path(config.data)
        samples = validate_manifest(
            manifest_path,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        output_path = create_student_predictions(config, samples)
        print(f"OK student predictions written: {output_path}")
        return

    if args.command == "switch-logits":
        output_path = create_visual_switch_dataset(config)
        print(f"OK visual-switch logits written: {output_path}")
        return

    if args.command == "validate-switch-logits":
        summary = validate_switch_logits_file(
            resolve_switch_logits_path(config.data),
            max_samples=config.data.max_samples,
            switch_logits_field=config.distillation.switch_logits_field,
            teacher_logits_field=config.distillation.teacher_logits_field,
        )
        _print_switch_logits_validation_summary(summary)
        if summary["invalid_rows"]:
            raise SystemExit(1)
        return

    if args.command == "train":
        artifact = train_student(config)
        print(f"OK student artifact written: {artifact}")
        return

    if args.command == "merge-adapter":
        merge_student_adapter(config)
        return

    if args.command == "evaluate":
        report_path = evaluate(config)
        print(f"OK eval report written: {report_path}")
        return

    if args.command == "evaluate-predictions":
        report_path = evaluate_predictions(config)
        print(
            "OK prediction eval report written: "
            f"{report_path} predictions={resolve_prediction_path(config.data)} "
            f"targets={resolve_label_path(config.data) if config.data.eval_path is None else config.data.eval_path}"
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


def _print_teacher_validation_summary(summary: dict[str, object]) -> None:
    print(f"OK validated teacher output path={summary['path']}")
    print(f"total_rows={summary['total_rows']}")
    print(f"valid_rows={summary['valid_rows']}")
    print(f"invalid_rows={summary['invalid_rows']}")
    print(f"answer_token_match_rows={summary['answer_token_match_rows']}")
    print(f"answer_token_mismatch_rows={summary['answer_token_mismatch_rows']}")
    bad_rows = summary.get("bad_rows") or []
    if bad_rows:
        print("first_bad_rows:")
        for bad_row in bad_rows:
            print(f"  id={bad_row['id']} reason={bad_row['reason']}")


def _resolve_create_manifest_task(
    *,
    config_path: Path,
    cli_task: str | None,
) -> str:
    raw_config_task = _read_raw_config_task_name(config_path)
    task = cli_task or raw_config_task
    if task is None:
        task = infer_manifest_task_from_config_path(config_path)

    allowed_tasks = ("parsing",)
    normalized_task = str(task).strip().casefold()
    if normalized_task not in allowed_tasks:
        raise ValueError(
            "Invalid create-manifest task resolution. "
            f"resolved task={task!r} config path={config_path} "
            f"allowed tasks={list(allowed_tasks)}"
        )
    return normalized_task


def _read_raw_config_task_name(config_path: Path) -> str | None:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded: Any = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        return None
    options = loaded.get("options")
    if not isinstance(options, dict):
        return None
    task_name = options.get("task_name")
    if task_name is None:
        return None
    task_text = str(task_name).strip()
    return task_text or None


def _print_switch_logits_validation_summary(summary: dict[str, object]) -> None:
    print(f"OK validated switch logits path={summary['path']}")
    print(f"total_rows={summary['total_rows']}")
    print(f"rows_with_switch_logits={summary['rows_with_switch_logits']}")
    print(f"valid_switch_logits_rows={summary['valid_switch_logits_rows']}")
    print(f"token_identity_match_rows={summary['token_identity_match_rows']}")
    print(f"token_identity_mismatch_rows={summary['token_identity_mismatch_rows']}")
    print(f"length_match_rows={summary['length_match_rows']}")
    print(f"length_mismatch_rows={summary['length_mismatch_rows']}")
    print(f"vocab_mismatch_rows={summary['vocab_mismatch_rows']}")
    print(f"invalid_rows={summary['invalid_rows']}")
    bad_rows = summary.get("bad_rows") or []
    if bad_rows:
        print("first_bad_rows:")
        for bad_row in bad_rows:
            print(f"  id={bad_row['id']} reason={bad_row['reason']}")


if __name__ == "__main__":
    main()
