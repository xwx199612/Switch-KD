"""Fail-fast checks for the precision contract used by inference deployments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def _linear4bit_type() -> type | tuple[()]:
    try:
        import bitsandbytes as bnb
        return bnb.nn.Linear4bit
    except (ImportError, AttributeError):
        return ()


def _is_linear4bit(module: Any, linear_type: Any) -> bool:
    return bool(linear_type and isinstance(module, linear_type)) or type(module).__name__ == "Linear4bit"


def _active_adapter(model: Any) -> str | None:
    value = getattr(model, "active_adapter", None)
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    if value is not None:
        return str(value)
    configs = getattr(model, "peft_config", None)
    if isinstance(configs, dict) and configs:
        return next(iter(configs))
    return None


def summarize_model_precision(model: Any) -> dict[str, Any]:
    linear_type = _linear4bit_type()
    linear4bit_names: list[str] = []
    visual_names: list[str] = []
    module_names: list[str] = []
    try:
        modules = model.named_modules()
    except AttributeError:
        modules = ()
    for name, module in modules:
        module_names.append(name)
        if _is_linear4bit(module, linear_type):
            linear4bit_names.append(name)
            if "visual" in name:
                visual_names.append(name)

    dtypes: Counter[str] = Counter()
    for parameter in model.parameters():
        dtypes[str(parameter.dtype)] += 1
    adapter_name = _active_adapter(model)
    peft_config = getattr(model, "peft_config", None)
    has_lora = any("lora_" in name.lower() for name, _ in getattr(model, "named_parameters", lambda: ())())
    peft_mounted = isinstance(peft_config, dict) or adapter_name is not None or has_lora
    return {
        "linear4bit_module_count": len(linear4bit_names),
        "visual_linear4bit_module_count": len(visual_names),
        "linear4bit_module_names": linear4bit_names,
        "module_names": module_names,
        "parameter_dtype_counts": dict(sorted(dtypes.items())),
        "model_class": type(model).__name__,
        "peft_model_mounted": bool(peft_mounted),
        "active_adapter_name": adapter_name,
        "adapter_merged": bool(getattr(model, "merged_adapters", None)),
    }


def _deployment_metadata(config: Any) -> dict[str, Any]:
    student = config.student
    candidates = [getattr(student, "deployment_artifact_path", None),
                  getattr(student, "inference_model_path", None)]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            metadata_path = path / "deployment_config.json"
            if metadata_path.exists():
                import json
                return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def validate_loaded_precision(config: Any, summary: dict[str, Any]) -> None:
    """Validate only contracts implied by the selected config/artifact."""
    student = config.student
    metadata = _deployment_metadata(config)
    mode = str(metadata.get("artifact_mode") or getattr(student, "quantization", "none"))
    is_4bit = mode in {"mixed_4bit_bf16", "4bit", "4bit_base_bf16_adapter", "post_merge_bnb4"}
    if is_4bit and summary["linear4bit_module_count"] <= 0:
        raise RuntimeError("Precision validation failed: expected at least one Linear4bit module")

    exclusions = metadata.get("excluded_from_quantization")
    if exclusions is None and mode == "mixed_4bit_bf16":
        exclusions = ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
    if exclusions:
        if mode == "mixed_4bit_bf16":
            missing = [path for path in exclusions if not any(
                name == path or name.endswith("." + path) or path.endswith("." + name)
                for name in summary.get("module_names", [])
            )]
            if missing:
                raise RuntimeError(
                    "Precision validation failed: configured mixed-precision exclusions are missing: "
                    f"{missing!r}"
                )
        bad = [name for name in summary.get("linear4bit_module_names", [])
               if any(name == path or name.endswith("." + path) or path.endswith("." + name)
                      for path in exclusions)]
        if bad:
            raise RuntimeError(
                "Precision validation failed: excluded high-precision modules are Linear4bit: "
                f"{bad!r}"
            )

    artifact_mode = metadata.get("artifact_mode")
    if artifact_mode == "4bit_base_bf16_adapter":
        if not summary["peft_model_mounted"]:
            raise RuntimeError("Precision validation failed: 4bit base deployment has no PEFT adapter")
        if summary.get("adapter_merged"):
            raise RuntimeError("Precision validation failed: 4bit base adapter must not be merged")
