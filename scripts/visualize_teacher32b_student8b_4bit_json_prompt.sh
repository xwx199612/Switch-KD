#!/usr/bin/env bash
set -euo pipefail

declare -a names=(
  "Qwen3-VL-32B Teacher 4bit"
  "Qwen3-VL-8B Student 4bit"
)
declare -a predictions=(
  "outputs/baselines/qwen3_vl_32b_teacher_4bit_json_prompt_1280/teacher_predictions.jsonl"
  "outputs/baselines/qwen3_vl_8b_student_4bit_json_prompt_1280/student_predictions.jsonl"
)
declare -a outputs=(
  "outputs/baselines/qwen3_vl_32b_teacher_4bit_json_prompt_1280/visualizations"
  "outputs/baselines/qwen3_vl_8b_student_4bit_json_prompt_1280/visualizations"
)

for index in "${!names[@]}"; do
  name="${names[$index]}"
  prediction_path="${predictions[$index]}"
  output_path="${outputs[$index]}"
  echo "Visualizing $name"
  if [[ ! -f "$prediction_path" ]]; then
    echo "Warning: prediction file not found, skipping: $prediction_path" >&2
    continue
  fi
  vlm-distill annotate-predictions --predictions "$prediction_path" --output-dir "$output_path"
done
