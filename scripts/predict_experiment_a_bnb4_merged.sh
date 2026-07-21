#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/predict_experiment_a_bnb4_merged_logs"
mkdir -p "$LOG_DIR"
RUN_EVAL="${RUN_EVAL:-0}"

ENTRIES=(
  "a0_r16|configs/lora_ablation/predict_bnb4_merged/stage1_a0_r16_bnb4_merged.yaml|outputs/lora_ablation/stage1_a0_r16_attn/adapter_merger/bnb4|outputs/lora_ablation/stage1_a0_r16_attn/post_merge_bnb4_predictions/student_predictions.jsonl"
  "a0_r32|configs/lora_ablation/predict_bnb4_merged/stage1_a0_r32_bnb4_merged.yaml|outputs/lora_ablation/stage1_a0_r32_attn/adapter_merger/bnb4|outputs/lora_ablation/stage1_a0_r32_attn/post_merge_bnb4_predictions/student_predictions.jsonl"
  "a1_r16|configs/lora_ablation/predict_bnb4_merged/stage1_a1_r16_bnb4_merged.yaml|outputs/lora_ablation/stage1_a1_r16_attn_projector/adapter_merger/bnb4|outputs/lora_ablation/stage1_a1_r16_attn_projector/post_merge_bnb4_predictions/student_predictions.jsonl"
  "a1_r32|configs/lora_ablation/predict_bnb4_merged/stage1_a1_r32_bnb4_merged.yaml|outputs/lora_ablation/stage1_a1_r32_attn_projector/adapter_merger/bnb4|outputs/lora_ablation/stage1_a1_r32_attn_projector/post_merge_bnb4_predictions/student_predictions.jsonl"
  "a2_r16|configs/lora_ablation/predict_bnb4_merged/stage1_a2_r16_bnb4_merged.yaml|outputs/lora_ablation/stage1_a2_r16_attn_projector_lora/adapter_merger/bnb4|outputs/lora_ablation/stage1_a2_r16_attn_projector_lora/post_merge_bnb4_predictions/student_predictions.jsonl"
  "a2_r32|configs/lora_ablation/predict_bnb4_merged/stage1_a2_r32_bnb4_merged.yaml|outputs/lora_ablation/stage1_a2_r32_attn_projector_lora/adapter_merger/bnb4|outputs/lora_ablation/stage1_a2_r32_attn_projector_lora/post_merge_bnb4_predictions/student_predictions.jsonl"
)

check_artifact() {
  local name="$1"
  local artifact="$2"
  if ! python - "$artifact" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
deployment = json.loads((root / "deployment_config.json").read_text(encoding="utf-8"))
merger = json.loads((root / "adapter_merger_config.json").read_text(encoding="utf-8"))
assert deployment["artifact_mode"] == "post_merge_bnb4"
assert deployment["adapter_merged"] is True
assert deployment["quantization_stage"] == "after_merge"
assert deployment["quantized_weights_persisted"] is False
assert deployment["merged_model_path"] == "merged_bf16"
assert merger["artifact_mode"] == "post_merge_bnb4"
assert (root / "merged_bf16" / "config.json").is_file()
PY
  then
    echo "ERROR $name: invalid post_merge_bnb4 artifact" >&2
    exit 1
  fi
}

prediction_is_complete() {
  local config="$1"
  local prediction="$2"
  python - "$config" "$prediction" <<'PY'
import json
import sys
from pathlib import Path

from vlm_distill.config_schema import load_config, resolve_inference_manifest_path
from vlm_distill.data_manifest import validate_manifest

config = load_config(sys.argv[1])
prediction = Path(sys.argv[2])
if not prediction.is_file() or prediction.stat().st_size == 0:
    raise SystemExit(1)
expected = validate_manifest(
    resolve_inference_manifest_path(config.data),
    image_root=config.data.image_root,
    max_samples=config.data.max_samples,
)
rows = []
with prediction.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            raise SystemExit(1)
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(1)
        if not isinstance(row, dict) or not row.get("id"):
            raise SystemExit(1)
        if not any(key in row for key in ("student_answer", "prediction", "elements", "raw_output")):
            raise SystemExit(1)
        rows.append(row)
if len(rows) != len(expected):
    raise SystemExit(1)
print(len(rows))
PY
}

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r name config artifact prediction <<< "$entry"
  check_artifact "$name" "$artifact"

  if rows="$(prediction_is_complete "$config" "$prediction" 2>/dev/null)"; then
    echo "SKIP $name: prediction already complete, rows=$rows"
    continue
  fi

  if [[ -s "$prediction" ]]; then
    backup="${prediction}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv -- "$prediction" "$backup"
    echo "BACKUP $name: incomplete prediction moved to $backup"
  fi

  log_path="$LOG_DIR/$name.log"
  echo "START $name"
  PYTHONUNBUFFERED=1 vlm-distill predict \
    --config "$config" \
    2>&1 | tee "$log_path"
  if [[ "$RUN_EVAL" == "1" ]]; then
    PYTHONUNBUFFERED=1 vlm-distill evaluate-predictions \
      --config "$config" \
      2>&1 | tee -a "$log_path"
  fi
  echo "DONE $name"
done

echo "Experiment A PostMerge-BNB4 prediction summary"
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r name config artifact prediction <<< "$entry"
  if rows="$(prediction_is_complete "$config" "$prediction" 2>/dev/null)"; then
    echo "$name: complete, rows=$rows"
  else
    echo "$name: incomplete, rows=$(wc -l < "$prediction" 2>/dev/null || true)"
  fi
done
