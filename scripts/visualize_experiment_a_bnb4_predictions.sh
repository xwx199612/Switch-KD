#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

log_dir="/tmp/visualize_experiment_a_bnb4_predictions_logs"
mkdir -p "$log_dir"

entries=(
  "a0_r16|outputs/lora_ablation/stage1_a0_r16_attn/post_merge_bnb4_predictions/student_predictions.jsonl|outputs/lora_ablation/visualizations_post_merge_bnb4/a0_r16"
  "a0_r32|outputs/lora_ablation/stage1_a0_r32_attn/post_merge_bnb4_predictions/student_predictions.jsonl|outputs/lora_ablation/visualizations_post_merge_bnb4/a0_r32"
  "a1_r16|outputs/lora_ablation/stage1_a1_r16_attn_projector/post_merge_bnb4_predictions/student_predictions.jsonl|outputs/lora_ablation/visualizations_post_merge_bnb4/a1_r16"
  "a1_r32|outputs/lora_ablation/stage1_a1_r32_attn_projector/post_merge_bnb4_predictions/student_predictions.jsonl|outputs/lora_ablation/visualizations_post_merge_bnb4/a1_r32"
  "a2_r16|outputs/lora_ablation/stage1_a2_r16_attn_projector_lora/post_merge_bnb4_predictions/student_predictions.jsonl|outputs/lora_ablation/visualizations_post_merge_bnb4/a2_r16"
  "a2_r32|outputs/lora_ablation/stage1_a2_r32_attn_projector_lora/post_merge_bnb4_predictions/student_predictions.jsonl|outputs/lora_ablation/visualizations_post_merge_bnb4/a2_r32"
)

count_images() {
  local directory="$1"
  if [[ ! -d "$directory" ]]; then
    echo 0
    return
  fi
  find "$directory" -maxdepth 1 -type f \( \
    -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \
  \) -print | wc -l
}

validate_predictions() {
  local prediction_path="$1"
  python - "$prediction_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)

rows = 0
required_fields = ("student_answer", "prediction", "elements", "raw_output")
with path.open("r", encoding="utf-8-sig") as handle:
    for line in handle:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(1)
        if not isinstance(row, dict) or not row.get("id") or not any(
            field in row for field in required_fields
        ):
            raise SystemExit(1)
        image = row.get("image")
        if not image:
            raise SystemExit(1)
        image_path = Path(str(image))
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        if not image_path.is_file():
            raise SystemExit(1)
        rows += 1

if rows == 0:
    raise SystemExit(1)
print(rows)
PY
}

declare -a summary_lines=()

for entry in "${entries[@]}"; do
  IFS='|' read -r name prediction_rel output_rel <<< "$entry"
  prediction_path="$repo_root/$prediction_rel"
  output_dir="$repo_root/$output_rel"
  log_path="$log_dir/$name.log"

  if ! prediction_rows="$(validate_predictions "$prediction_path")"; then
    echo "ERROR $name: invalid or incomplete prediction file" >&2
    exit 1
  fi

  echo "START $name"
  existing_images="$(count_images "$output_dir")"
  if [[ -d "$output_dir" && "$existing_images" -eq "$prediction_rows" ]]; then
    echo "SKIP $name: images=$existing_images predictions=$prediction_rows"
  else
    if [[ -d "$output_dir" ]]; then
      timestamp="$(date +%Y%m%d_%H%M%S)"
      backup_dir="${output_dir}.incomplete.${timestamp}"
      while [[ -e "$backup_dir" ]]; do
        timestamp="$(date +%Y%m%d_%H%M%S)"
        backup_dir="${output_dir}.incomplete.${timestamp}"
      done
      mv -- "$output_dir" "$backup_dir"
      echo "BACKUP $name: $backup_dir"
    fi
    mkdir -p "$output_dir"
    PYTHONUNBUFFERED=1 vlm-distill annotate-predictions \
      --predictions "$prediction_path" \
      --output-dir "$output_dir" 2>&1 | tee "$log_path"
  fi

  images_created="$(count_images "$output_dir")"
  errors=0
  if [[ -s "$output_dir/visualization_errors.jsonl" ]]; then
    errors="$(wc -l < "$output_dir/visualization_errors.jsonl")"
  fi
  if [[ "$errors" -gt 0 ]]; then
    status="complete_with_errors"
  elif [[ "$images_created" -eq "$prediction_rows" ]]; then
    status="complete"
  else
    status="incomplete"
  fi
  summary_lines+=("$name: $status, predictions=$prediction_rows, images=$images_created, errors=$errors")
  echo "DONE $name"
done

echo "Experiment A PostMerge-BNB4 bbox visualization summary"
printf '%s\n' "${summary_lines[@]}"
