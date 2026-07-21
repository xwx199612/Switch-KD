#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIXED_CONFIG="configs/lora_ablation/predict/stage1_a3_r32_attn_mlp_mixed_precision.yaml"
BNB4_CONFIG="configs/lora_ablation/predict_bnb4_merged/stage1_a3_r32_bnb4_merged.yaml"
RUN_EVAL="${RUN_EVAL:-0}"

python scripts/check_stage1_a3_config_consistency.py "$MIXED_CONFIG" "$BNB4_CONFIG"

echo "START a3_r32_mixed_precision"
PYTHONUNBUFFERED=1 vlm-distill predict --config "$MIXED_CONFIG"
if [[ "$RUN_EVAL" == "1" ]]; then
  PYTHONUNBUFFERED=1 vlm-distill evaluate-predictions --config "$MIXED_CONFIG"
fi
echo "DONE a3_r32_mixed_precision"

echo "START a3_r32_post_merge_bnb4"
PYTHONUNBUFFERED=1 vlm-distill predict --config "$BNB4_CONFIG"
if [[ "$RUN_EVAL" == "1" ]]; then
  PYTHONUNBUFFERED=1 vlm-distill evaluate-predictions --config "$BNB4_CONFIG"
fi
echo "DONE a3_r32_post_merge_bnb4"
