"""Capability-aware mixed 4-bit/BF16 loading helpers."""

from __future__ import annotations

import inspect
from typing import Any


def mixed_precision_capabilities() -> dict[str, Any]:
    """Report the installed stack and whether BNB 4-bit exclusions are reliable."""
    import torch

    result: dict[str, Any] = {
        "transformers_version": None,
        "bitsandbytes_version": None,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "4bit_module_exclusion_supported": False,
        "artifact_mode_supported": False,
        "exclusion_api": None,
    }
    try:
        import transformers
        from transformers import BitsAndBytesConfig
        from transformers.integrations import replace_with_bnb_linear
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

        result["transformers_version"] = transformers.__version__
        config_signature = inspect.signature(BitsAndBytesConfig)
        quantizer_source = inspect.getsource(Bnb4BitHfQuantizer._process_model_before_weight_loading)
        replacement_signature = inspect.signature(replace_with_bnb_linear)
        reliable = (
            "llm_int8_skip_modules" in config_signature.parameters
            and "llm_int8_skip_modules" in quantizer_source
            and "modules_to_not_convert" in replacement_signature.parameters
            and "should_convert_module" in inspect.getsource(replace_with_bnb_linear)
        )
        result["4bit_module_exclusion_supported"] = reliable
        result["artifact_mode_supported"] = reliable
        if reliable:
            result["exclusion_api"] = "BitsAndBytesConfig.llm_int8_skip_modules"
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        pass
    try:
        import bitsandbytes

        result["bitsandbytes_version"] = bitsandbytes.__version__
    except (ImportError, AttributeError):
        pass
    return result


def build_mixed_precision_exclusion_paths(excluded_module_paths: list[str]) -> list[str]:
    """Return exact paths, which are also safe suffix patterns in HF's matcher."""
    paths = []
    for path in excluded_module_paths:
        normalized = path.strip().strip(".")
        if not normalized or normalized in paths:
            continue
        if normalized.endswith(("linear_fc1", "linear_fc2")):
            paths.append(normalized)
        else:
            raise ValueError(
                "Mixed-precision exclusions must name exact merger linears; "
                f"got {path!r}."
            )
    return paths


def build_mixed_precision_quantization_config(
    *, quantization: str, excluded_module_paths: list[str]
):
    """Build a quantization config with exact module exclusions.

    The installed BNB 4-bit implementation exposes the exclusion list through
    the historically named ``llm_int8_skip_modules`` field.
    """
    if quantization not in {"4bit", "8bit"}:
        raise ValueError(f"Mixed-precision quantization requires 4bit or 8bit, got {quantization!r}.")
    capabilities = mixed_precision_capabilities()
    if quantization == "4bit" and not capabilities["4bit_module_exclusion_supported"]:
        raise RuntimeError(
            "A1 mixed standalone artifact is unsupported by the installed "
            "Transformers/bitsandbytes versions because exact module exclusion "
            "during 4-bit reload is unavailable.\n\n"
            "Use merged_artifact_mode=bf16_standalone or "
            "merged_artifact_mode=adapter_plus_projector."
        )
    import torch
    from transformers import BitsAndBytesConfig

    exclusions = build_mixed_precision_exclusion_paths(excluded_module_paths)
    kwargs: dict[str, Any] = {
        "load_in_4bit": quantization == "4bit",
        "load_in_8bit": quantization == "8bit",
        "llm_int8_skip_modules": exclusions,
    }
    if quantization == "4bit":
        kwargs.update(
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(**kwargs)
