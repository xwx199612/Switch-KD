import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlm_distill.adapter_merger import _validate_adapter_target_contract
from vlm_distill.config_schema import load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimpleNamespace(
    a3=REPOSITORY_ROOT / "configs/lora_ablation/stage1_a3_r32_attn_mlp.yaml",
    deploy=REPOSITORY_ROOT
    / "configs/lora_ablation/deploy/stage1_a3_r32_attn_mlp_deploy.yaml",
    mixed=REPOSITORY_ROOT
    / "configs/lora_ablation/predict/stage1_a3_r32_attn_mlp_mixed_precision.yaml",
    bnb4=REPOSITORY_ROOT
    / "configs/lora_ablation/predict_bnb4_merged/stage1_a3_r32_bnb4_merged.yaml",
)

A3_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def test_a3_training_config_loads_and_freezes_projector():
    config = load_config(CONFIG.a3)

    assert A3_TARGET_MODULES <= set(config.student.target_modules)
    assert config.student.train_multimodal_projector is False
    assert config.student.use_projector_lora is False


def test_a3_deploy_config_loads_and_points_to_mixed_artifact():
    config = load_config(CONFIG.deploy)
    expected_artifact = Path(
        "outputs/lora_ablation/stage1_a3_r32_attn_mlp/deploy_4bit_bf16_adapter"
    )

    assert config.student.inference_model_path == str(expected_artifact)
    assert config.student.deployment_artifact_path == expected_artifact
    assert config.student.merged_artifact_mode == "4bit_base_bf16_adapter"


def test_a3_bnb4_prediction_config_loads_and_points_to_merger_artifact():
    config = load_config(CONFIG.bnb4)
    expected_artifact = Path(
        "outputs/lora_ablation/stage1_a3_r32_attn_mlp/adapter_merger/bnb4"
    )

    assert config.student.inference_model_path == str(expected_artifact)
    assert config.student.deployment_artifact_path == expected_artifact
    assert config.student.inference_adapter_path is None
    assert config.student.merged_artifact_mode == "post_merge_bnb4"


def test_a3_prediction_paths_are_distinct():
    mixed = load_config(CONFIG.mixed)
    bnb4 = load_config(CONFIG.bnb4)

    assert mixed.data.prediction_path == Path(
        "outputs/lora_ablation/stage1_a3_r32_attn_mlp/"
        "mixed_precision_predictions/student_predictions.jsonl"
    )
    assert bnb4.data.prediction_path == Path(
        "outputs/lora_ablation/stage1_a3_r32_attn_mlp/"
        "post_merge_bnb4_predictions/student_predictions.jsonl"
    )
    assert mixed.data.prediction_path != bnb4.data.prediction_path


def test_a3_prediction_configs_share_schema_fields():
    mixed = load_config(CONFIG.mixed)
    bnb4 = load_config(CONFIG.bnb4)

    # response_profile is an interpolation option, not a PipelineConfig field;
    # comparing the interpolated distill path verifies the effective profile.
    assert mixed.data.inference_manifest_path == bnb4.data.inference_manifest_path
    assert mixed.data.max_samples == bnb4.data.max_samples
    assert mixed.teacher.temperature == bnb4.teacher.temperature
    assert mixed.teacher.max_new_tokens == bnb4.teacher.max_new_tokens
    assert mixed.data.distill_path == bnb4.data.distill_path
    assert mixed.distillation.prompt_template == bnb4.distillation.prompt_template


def _write_adapter_config(adapter_dir: Path, *, modules_to_save):
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": sorted(A3_TARGET_MODULES),
                "modules_to_save": modules_to_save,
            }
        ),
        encoding="utf-8",
    )


def test_a3_adapter_contract_accepts_null_modules_to_save(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_adapter_config(adapter_dir, modules_to_save=None)

    config = load_config(CONFIG.a3)
    _validate_adapter_target_contract(config, adapter_dir)


def test_a3_adapter_contract_rejects_nonempty_modules_to_save(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_adapter_config(
        adapter_dir, modules_to_save=["model.visual.merger"]
    )

    config = load_config(CONFIG.a3)
    with pytest.raises(RuntimeError, match="modules_to_save"):
        _validate_adapter_target_contract(config, adapter_dir)


def test_a3_adapter_contract_rejects_missing_mlp_target(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_adapter_config(adapter_dir, modules_to_save=None)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": sorted(A3_TARGET_MODULES - {"gate_proj"}),
                "modules_to_save": None,
            }
        ),
        encoding="utf-8",
    )

    config = load_config(CONFIG.a3)
    with pytest.raises(RuntimeError, match="target contract|gate_proj"):
        _validate_adapter_target_contract(config, adapter_dir)
