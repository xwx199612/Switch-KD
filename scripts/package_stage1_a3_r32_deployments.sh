#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIXED_CONFIG="configs/lora_ablation/deploy/stage1_a3_r32_attn_mlp_deploy.yaml"
TRAIN_CONFIG="configs/lora_ablation/stage1_a3_r32_attn_mlp.yaml"
MIXED_ARTIFACT="outputs/lora_ablation/stage1_a3_r32_attn_mlp/deploy_4bit_bf16_adapter"
BNB4_ARTIFACT="outputs/lora_ablation/stage1_a3_r32_attn_mlp/adapter_merger/bnb4"

valid_mixed() {
  python - "$MIXED_ARTIFACT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / "deployment_config.json").read_text())
assert m["artifact_mode"] == "4bit_base_bf16_adapter"
assert m["adapter_merged"] is False
assert (p / m["adapter_path"]).is_dir()
PY
}

valid_bnb4() {
  python - "$BNB4_ARTIFACT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads((p / "deployment_config.json").read_text())
m = json.loads((p / "adapter_merger_config.json").read_text())
assert d["artifact_mode"] == "post_merge_bnb4"
assert d["adapter_merged"] is True
assert d["quantization_stage"] == "after_merge"
assert d["quantized_weights_persisted"] is False
assert m["artifact_mode"] == "post_merge_bnb4"
assert (p / "merged_bf16" / "config.json").is_file()
PY
}

if [[ -e "$MIXED_ARTIFACT" ]]; then
  if valid_mixed; then
    echo "SKIP mixed precision: valid artifact already exists at $MIXED_ARTIFACT"
  else
    echo "ERROR mixed precision artifact is incomplete: $MIXED_ARTIFACT" >&2
    echo "Run the package command manually after resolving it; do not delete automatically." >&2
    exit 1
  fi
else
  vlm-distill package-adapter --config "$MIXED_CONFIG"
fi

if [[ -e "$BNB4_ARTIFACT" ]]; then
  if valid_bnb4; then
    echo "SKIP post-merge BNB4: valid artifact already exists at $BNB4_ARTIFACT"
  else
    echo "ERROR post-merge BNB4 artifact is incomplete: $BNB4_ARTIFACT" >&2
    echo "Re-run vlm-distill adapter-merger with --overwrite only after manual review." >&2
    exit 1
  fi
else
  vlm-distill adapter-merger \
    --config "$TRAIN_CONFIG" \
    --output-dir "$BNB4_ARTIFACT" \
    --quantization bnb4
fi

valid_mixed
valid_bnb4
echo "A3 deployment artifacts validated"
