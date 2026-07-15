from __future__ import annotations

import pytest

from vlm_distill.mixed_precision import (
    build_mixed_precision_exclusion_paths,
    build_mixed_precision_quantization_config,
    mixed_precision_capabilities,
)


def test_exact_merger_paths_are_excluded_without_loose_terms():
    paths = build_mixed_precision_exclusion_paths([
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    ])
    assert paths == [
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    ]
    assert "merger" not in paths
    assert "visual" not in paths
    assert "linear_fc1" not in paths
    assert "model.visual.deepstack_merger_list.0" not in paths

    from transformers.quantizers.quantizers_utils import should_convert_module

    assert not should_convert_module("model.visual.merger.linear_fc1", paths)
    assert not should_convert_module("wrapped.model.visual.merger.linear_fc2", paths)
    assert should_convert_module("model.visual.deepstack_merger_list.0.linear_fc1", paths)
    assert should_convert_module("model.visual.linear_fc1", paths)


def test_non_exact_exclusion_is_rejected():
    with pytest.raises(ValueError, match="exact merger linears"):
        build_mixed_precision_exclusion_paths(["model.visual.merger"])


def test_supported_exclusion_api_is_detected_from_installed_implementation():
    capabilities = mixed_precision_capabilities()
    assert capabilities["transformers_version"]
    assert capabilities["torch_version"]
    assert capabilities["4bit_module_exclusion_supported"] is True
    assert capabilities["artifact_mode_supported"] is True
    assert capabilities["exclusion_api"] == "BitsAndBytesConfig.llm_int8_skip_modules"


def test_quantization_config_contains_exact_exclusions():
    config = build_mixed_precision_quantization_config(
        quantization="4bit",
        excluded_module_paths=[
            "model.visual.merger.linear_fc1",
            "model.visual.merger.linear_fc2",
        ],
    )
    assert config.load_in_4bit is True
    assert config.llm_int8_skip_modules == [
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    ]


def test_unsupported_stack_fails_actionably(monkeypatch):
    import vlm_distill.mixed_precision as mixed_precision

    monkeypatch.setattr(
        mixed_precision,
        "mixed_precision_capabilities",
        lambda: {"4bit_module_exclusion_supported": False},
    )
    with pytest.raises(RuntimeError, match="merged_artifact_mode=bf16_standalone"):
        mixed_precision.build_mixed_precision_quantization_config(
            quantization="4bit",
            excluded_module_paths=["model.visual.merger.linear_fc1"],
        )


def test_cpu_capability_report_does_not_claim_cuda_success():
    capabilities = mixed_precision_capabilities()
    if not capabilities["cuda_available"]:
        assert capabilities["cuda_available"] is False
