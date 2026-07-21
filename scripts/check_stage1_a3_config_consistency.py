#!/usr/bin/env python3
"""Compare the non-artifact inference contract for the two A3 predictions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from vlm_distill.config_schema import load_config, resolve_inference_manifest_path


def _contract(path: str) -> dict:
    config = load_config(path)
    response_profile = (
        f"{config.teacher.image_resize}_{config.teacher.quantization}"
        f"_student_{config.student.quantization}"
    )
    return {
        "manifest_path": str(resolve_inference_manifest_path(config.data)),
        "image_root": str(config.data.image_root),
        "max_samples": config.data.max_samples,
        "generation_parameters": {
            "max_new_tokens": config.teacher.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": None,
            "generation_config": None,
        },
        "response_profile": response_profile,
        "prompt_template": config.distillation.prompt_template,
        "evaluation": {"metrics": list(config.evaluation.metrics)},
    }


def main() -> None:
    paths = sys.argv[1:] or [
        "configs/lora_ablation/predict/stage1_a3_r32_attn_mlp_mixed_precision.yaml",
        "configs/lora_ablation/predict_bnb4_merged/stage1_a3_r32_bnb4_merged.yaml",
    ]
    if len(paths) != 2:
        raise SystemExit("usage: check_stage1_a3_config_consistency.py MIXED_CONFIG BNB4_CONFIG")
    contracts = [_contract(path) for path in paths]
    print(json.dumps({"mixed_precision": contracts[0], "post_merge_bnb4": contracts[1]}, indent=2))
    if contracts[0] != contracts[1]:
        differences = [key for key in contracts[0] if contracts[0][key] != contracts[1][key]]
        raise SystemExit(f"A3 config consistency check failed: {differences}")
    print("A3 config consistency: PASS")


if __name__ == "__main__":
    main()
