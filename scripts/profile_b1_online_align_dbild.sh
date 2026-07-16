#!/usr/bin/env bash
set -euo pipefail

# One-step B1 profiling only. The config is derived from the formal B1 config
# and changes only the existing smoke controls (including accumulation=1).
python -m vlm_distill.train_online_align_dbild \
  --config configs/stage1_b1_r16_attn_layers_12_35_smoke.yaml \
  --max-steps 1
