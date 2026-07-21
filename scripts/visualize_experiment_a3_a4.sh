#!/usr/bin/env bash
set -euo pipefail

root="outputs/lora_ablation"
declare -a runs=(
  "stage1_a3_r16_attn_mlp stage1_a3_r16_attn_mlp"
  "stage1_a3_r32_attn_mlp stage1_a3_r32_attn_mlp"
  "stage1_a4_r16_attn_mlp_projector stage1_a4_r16_attn_mlp_projector"
  "stage1_a4_r32_attn_mlp_projector stage1_a4_r32_attn_mlp_projector"
)

for run in "${runs[@]}"; do
  read -r name directory <<<"$run"
  predictions="$root/$directory/mixed_precision_predictions/student_predictions.jsonl"
  output="$root/visualizations/$name"
  if [[ ! -f "$predictions" ]]; then
    echo "Warning: prediction file not found, skipping: $predictions" >&2
    continue
  fi
  vlm-distill annotate-predictions --predictions "$predictions" --output-dir "$output"
done
