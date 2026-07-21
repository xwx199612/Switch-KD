from __future__ import annotations

import subprocess
from pathlib import Path

from vlm_distill.config_schema import load_config, resolve_inference_manifest_path


ROOT = Path(__file__).parents[1]
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_a3_training_contract():
    config = load_config(ROOT / "configs/lora_ablation/stage1_a3_r32_attn_mlp.yaml")
    assert set(config.student.target_modules) == TARGETS
    assert config.student.train_multimodal_projector is False
    assert config.student.use_projector_lora is False
    assert config.student.lora_rank == 32
    assert config.student.output_dir.as_posix().endswith("outputs/lora_ablation/stage1_a3_r32_attn_mlp")


def test_a3_prediction_contracts_are_isolated_and_consistent():
    mixed = load_config(ROOT / "configs/lora_ablation/predict/stage1_a3_r32_attn_mlp_mixed_precision.yaml")
    bnb4 = load_config(ROOT / "configs/lora_ablation/predict_bnb4_merged/stage1_a3_r32_bnb4_merged.yaml")
    assert resolve_inference_manifest_path(mixed.data) == resolve_inference_manifest_path(bnb4.data)
    assert mixed.teacher.max_new_tokens == bnb4.teacher.max_new_tokens
    assert mixed.distillation.prompt_template == bnb4.distillation.prompt_template
    assert mixed.evaluation.metrics == bnb4.evaluation.metrics
    assert mixed.data.prediction_path != bnb4.data.prediction_path
    assert mixed.student.deployment_artifact_path != bnb4.student.deployment_artifact_path
    assert mixed.student.merged_artifact_mode == "4bit_base_bf16_adapter"
    assert bnb4.student.merged_artifact_mode == "post_merge_bnb4"
    assert mixed.student.load_adapter is False
    assert bnb4.student.load_adapter is False
    assert bnb4.student.inference_adapter_path is None


def test_consistency_check_passes():
    result = subprocess.run(
        ["python", "scripts/check_stage1_a3_config_consistency.py"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "A3 config consistency: PASS" in result.stdout


def test_one_shot_scripts_have_valid_shell_syntax():
    for name in ("package_stage1_a3_r32_deployments.sh", "predict_stage1_a3_r32_comparison.sh"):
        result = subprocess.run(["bash", "-n", f"scripts/{name}"], cwd=ROOT, check=False)
        assert result.returncode == 0


def test_a3_adapter_merger_loader_contract_is_not_active_adapter():
    source = (ROOT / "src/vlm_distill/adapter_merger.py").read_text(encoding="utf-8")
    loader = (ROOT / "src/vlm_distill/bbox_grounding_inference.py").read_text(encoding="utf-8")
    assert "adapter_path=none" in source
    assert "load_adapter_merger_artifact" in loader
    assert "PeftModel.from_pretrained" not in source[source.index("def load_adapter_merger_artifact"):]
