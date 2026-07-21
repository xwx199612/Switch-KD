#!/usr/bin/env bash
set -euo pipefail

root="outputs/lora_ablation"
declare -a runs=(
  "a0_r16 stage1_a0_r16_attn"
  "a0_r32 stage1_a0_r32_attn"
  "a1_r16 stage1_a1_r16_attn_projector"
  "a1_r32 stage1_a1_r32_attn_projector"
  "a2_r16 stage1_a2_r16_attn_projector_lora"
  "a2_r32 stage1_a2_r32_attn_projector_lora"
)

for run in "${runs[@]}"; do
  read -r name directory <<<"$run"
  predictions="$root/$directory/parsing_online_dbild_1080p_4bit_student_4bit/student_predictions.jsonl"
  output="$root/visualizations/$name"
  if [[ ! -f "$predictions" ]]; then
    echo "Warning: prediction file not found, skipping: $predictions" >&2
    continue
  fi
  vlm-distill annotate-predictions --predictions "$predictions" --output-dir "$output"
done
