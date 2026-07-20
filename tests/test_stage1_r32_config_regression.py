from dataclasses import asdict
from pathlib import Path

from vlm_distill.config_schema import load_config


TRAINING = {
    "a0": Path("configs/lora_ablation/stage1_a0_r32_attn.yaml"),
    "a1": Path("configs/lora_ablation/stage1_a1_r32_attn_projector.yaml"),
    "a2": Path("configs/lora_ablation/stage1_a2_r32_attn_projector_lora.yaml"),
    "a3": Path("configs/lora_ablation/stage1_a3_r32_attn_mlp_projector.yaml"),
}
R16 = {
    "a0": Path("configs/lora_ablation/stage1_a0_r16_attn.yaml"),
    "a1": Path("configs/lora_ablation/stage1_a1_r16_attn_projector.yaml"),
    "a2": Path("configs/lora_ablation/stage1_a2_r16_attn_projector_lora.yaml"),
    "a3": Path("configs/lora_ablation/stage1_a3_r16_attn_mlp_projector.yaml"),
}
DEPLOY = {
    key: Path(f"configs/lora_ablation/deploy/stage1_{key}_r32_4bit_base_bf16_adapter.yaml")
    for key in TRAINING
}
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def _paths(config):
    values = asdict(config)
    return str(values)


def test_all_stage1_r32_configs_load_and_are_disjoint_from_r16_outputs():
    for key, path in TRAINING.items():
        config = load_config(path)
        r16 = load_config(R16[key])
        assert config.student.lora_rank == 32
        assert config.student.lora_alpha / config.student.lora_rank == r16.student.lora_alpha / r16.student.lora_rank
        assert "r16" not in _paths(config).lower()
        for value in (
            config.student.output_dir,
            config.student.adapter_dir,
            config.student.merged_model_path,
            config.student.inference_adapter_path,
            config.student.inference_model_path,
            config.student.deployment_artifact_path,
            config.data.eval_path,
            config.data.prediction_path,
            config.evaluation.output_path,
        ):
            assert value is not None
            assert "stage1_" + key + "_r32" in str(value)
            assert "stage1_" + key + "_r16" not in str(value)


def test_stage1_r32_projector_contracts():
    a0 = load_config(TRAINING["a0"]).student
    a1 = load_config(TRAINING["a1"]).student
    a2 = load_config(TRAINING["a2"]).student
    a3 = load_config(TRAINING["a3"]).student

    assert a0.target_modules == ATTENTION
    assert not a0.train_multimodal_projector and not a0.use_projector_lora
    assert a1.target_modules == ATTENTION
    assert a1.train_multimodal_projector and not a1.use_projector_lora
    assert a2.target_modules == ATTENTION
    assert not a2.train_multimodal_projector and a2.use_projector_lora
    assert a2.projector_lora_rank == 32
    assert a2.projector_lora_alpha / a2.projector_lora_rank == 2
    assert a3.target_modules == ATTENTION + MLP
    assert a3.train_multimodal_projector and not a3.use_projector_lora


def test_stage1_r32_deployments_are_nonmerged_high_fidelity_and_match_adapters():
    expected_modes = {"a0": "base", "a1": "full", "a2": "lora", "a3": "full"}
    for key, path in DEPLOY.items():
        config = load_config(path)
        assert config.student.merged_artifact_mode == "4bit_base_bf16_adapter"
        assert config.student.quantization == "4bit"
        assert config.student.merge_adapter is False
        assert config.student.load_adapter is False
        assert config.student.inference_adapter_path == config.student.adapter_dir
        assert f"stage1_{key}_r32" in str(config.student.inference_adapter_path)
        assert expected_modes[key] == (
            "lora" if config.student.use_projector_lora else
            "full" if config.student.train_multimodal_projector else "base"
        )


def test_stage1_r32_lora_parameter_counts_from_qwen3_vl_8b_shapes():
    # Qwen3-VL-8B local model: 36 layers, hidden=4096, KV=1024,
    # MLP=12288, projector fc1=(4608,4608), fc2=(4096,4608).
    attention_r16 = 36 * (2 * 16 * (4096 + 4096) + 2 * 16 * (4096 + 1024))
    attention_r32 = 2 * attention_r16
    mlp_r16 = 36 * 3 * (2 * 16 * (4096 + 12288))
    mlp_r32 = 2 * mlp_r16
    projector_lora_r16 = 2 * 16 * (4608 + 4608) + 2 * 16 * (4608 + 4096)
    projector_lora_r32 = 2 * projector_lora_r16
    full_projector = 4608 * 4608 + 4608 + 4096 * 4608 + 4096 + 1152 + 1152

    assert (attention_r16, attention_r32) == (15_335_424, 30_670_848)
    assert (mlp_r16, mlp_r32) == (56_623_104, 113_246_208)
    assert (projector_lora_r16, projector_lora_r32) == (573_440, 1_146_880)
    assert full_projector == 40_119_040
    assert {
        "a0": (attention_r16, attention_r32),
        "a1": (attention_r16 + full_projector, attention_r32 + full_projector),
        "a2": (attention_r16 + projector_lora_r16, attention_r32 + projector_lora_r32),
        "a3": (attention_r16 + mlp_r16 + full_projector, attention_r32 + mlp_r32 + full_projector),
    } == {
        "a0": (15_335_424, 30_670_848),
        "a1": (55_454_464, 70_789_888),
        "a2": (15_908_864, 31_817_728),
        "a3": (112_077_568, 184_036_096),
    }
