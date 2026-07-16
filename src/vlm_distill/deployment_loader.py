"""Loader and runtime checks for the non-merged 4-bit/BF16 deployment bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import torch

from .mixed_precision import build_mixed_precision_quantization_config
from .model_loading import apply_attn_implementation, resolve_model_path

MAIN_MERGER_PATHS = [
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
]


def _module(model: Any, path: str) -> Any:
    current = model
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _is_lora(name: str) -> bool:
    lower = name.lower()
    return "lora_a" in lower or "lora_b" in lower


def _active_merger(model: Any) -> Any:
    merger = _module(model, "model.visual.merger")
    # PEFT's ModulesToSaveWrapper keeps the trained copy under this exact path.
    modules_to_save = getattr(merger, "modules_to_save", None)
    if modules_to_save is not None and hasattr(modules_to_save, "default"):
        return modules_to_save.default
    return merger


def _summary(model: Any) -> dict[str, Any]:
    try:
        import bitsandbytes as bnb
        linear4bit = bnb.nn.Linear4bit
    except ImportError:
        linear4bit = ()
    counts = {"linear4bit": 0, "attention_lora": 0, "projector_lora": 0,
              "modules_to_save": 0, "vision_trainable": 0, "projector_lora_names": []}
    attention_dtypes: set[str] = set()
    projector_dtypes: set[str] = set()
    for name, module in model.named_modules():
        if linear4bit and isinstance(module, linear4bit) and "language_model" in name:
            counts["linear4bit"] += 1
        if _is_lora(name):
            is_projector = "visual.merger.linear_fc" in name
            counts["projector_lora" if is_projector else "attention_lora"] += 1
            if is_projector:
                counts["projector_lora_names"].append(name)
            for parameter in module.parameters(recurse=False):
                (projector_dtypes if is_projector else attention_dtypes).add(str(parameter.dtype))
        if "modules_to_save" in name and "default" in name:
            counts["modules_to_save"] += sum(1 for _ in module.parameters(recurse=False))
    for name, parameter in model.named_parameters():
        if "visual" in name and "visual.merger" not in name and parameter.requires_grad:
            counts["vision_trainable"] += parameter.numel()
    counts["attention_dtypes"] = sorted(attention_dtypes)
    counts["projector_dtypes"] = sorted(projector_dtypes)
    return counts


def validate_high_fidelity_deployment(model: Any, config: Any = None, *, smoke_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the deployment contract and return a machine-readable summary."""
    merger = _active_merger(model)
    for child in ("linear_fc1", "linear_fc2"):
        layer = getattr(merger, child, None)
        if type(layer) is not torch.nn.Linear or layer.weight.dtype != torch.bfloat16:
            raise RuntimeError(f"main merger {child} must be torch.nn.Linear with BF16 weights")
    summary = _summary(model)
    if summary["linear4bit"] <= 0:
        raise RuntimeError("deployment validation failed: language model has no Linear4bit modules")
    if summary["attention_lora"] <= 0:
        raise RuntimeError("deployment validation failed: no attention LoRA modules found")
    if summary["vision_trainable"]:
        raise RuntimeError("deployment validation failed: vision encoder has trainable parameters")
    if not model.training is False:
        raise RuntimeError("deployment validation failed: model must be eval mode")
    if hasattr(model, "active_adapter") and not getattr(model, "active_adapter"):
        raise RuntimeError("deployment validation failed: PEFT adapter is not active")
    if getattr(model, "merged_adapters", None):
        raise RuntimeError("deployment validation failed: adapter is merged")
    if any("torch.int" in dtype or "torch.uint" in dtype for dtype in summary["attention_dtypes"] + summary["projector_dtypes"]):
        raise RuntimeError("deployment validation failed: adapter tensors must remain floating point")

    mode = getattr(getattr(config, "student", config), "train_multimodal_projector", False) if config else False
    use_projector_lora = getattr(getattr(config, "student", config), "use_projector_lora", False) if config else False
    if mode and (summary["modules_to_save"] <= 0 or summary["projector_lora"]):
        raise RuntimeError("A1 validation failed: active modules_to_save projector is missing or projector LoRA exists")
    if mode:
        wrapper = _module(model, "model.visual.merger")
        active_copy = getattr(getattr(wrapper, "modules_to_save", None), "default", None)
        if active_copy is None:
            raise RuntimeError("A1 validation failed: model.visual.merger.modules_to_save.default is missing")
        active_adapter = getattr(wrapper, "active_adapter", "default")
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0] if active_adapter else "default"
        if str(active_adapter) != "default":
            raise RuntimeError("A1 validation failed: trained projector copy is not the active adapter copy")
    if use_projector_lora:
        if summary["projector_lora"] <= 0:
            raise RuntimeError("A2 validation failed: projector LoRA is missing")
        if any("deepstack_merger" in n or ("mlp" in n.lower() and "lora" in n.lower()) for n, _ in model.named_modules()):
            raise RuntimeError("A2 validation failed: deepstack merger or LLM MLP LoRA is not allowed")
        if any(not ("model.visual.merger.linear_fc1" in n or "model.visual.merger.linear_fc2" in n)
               for n in summary["projector_lora_names"]):
            raise RuntimeError("A2 validation failed: projector LoRA targets must be the two main merger linears")
    if not mode and not use_projector_lora and (summary["projector_lora"] or summary["modules_to_save"]):
        raise RuntimeError("A0 validation failed: projector must be base BF16 without projector adapter weights")
    if use_projector_lora and summary["modules_to_save"]:
        raise RuntimeError("A2 validation failed: modules_to_save projector is not allowed")
    if mode and summary["projector_lora"]:
        raise RuntimeError("A1 validation failed: projector LoRA is not allowed")
    if smoke_inputs is not None:
        with torch.no_grad():
            outputs = model(**smoke_inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None or logits.numel() == 0 or not torch.isfinite(logits).all():
            raise RuntimeError("deployment validation failed: logits smoke test was not finite")
    print(f"language_model Linear4bit count: {summary['linear4bit']}")
    print("main merger BF16 linear count: 2")
    print(f"attention LoRA tensor count: {summary['attention_lora']}")
    print(f"projector LoRA tensor count: {summary['projector_lora']}")
    print(f"modules_to_save tensor count: {summary['modules_to_save']}")
    print(f"Attention LoRA dtype summary: {summary['attention_dtypes']}")
    print(f"Projector LoRA dtype summary: {summary['projector_dtypes']}")
    return summary


def load_high_fidelity_adapter_deployment(
    deployment_path: str | Path, *, base_model_path: str | Path | None = None,
):
    deployment_path = Path(deployment_path)
    metadata_path = deployment_path / "deployment_config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Not a high-fidelity deployment bundle: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_mode") != "4bit_base_bf16_adapter":
        raise ValueError("deployment_config.json is not a 4bit_base_bf16_adapter bundle")
    base = str(base_model_path or metadata["base_model_path"])
    base = resolve_model_path(base)
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    processor_path = deployment_path / "processor"
    processor = AutoProcessor.from_pretrained(str(processor_path if processor_path.exists() else base), trust_remote_code=True, use_fast=False, local_files_only=True)
    kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True, "local_files_only": True}
    apply_attn_implementation(kwargs, metadata.get("attn_implementation", "sdpa"))
    kwargs["quantization_config"] = build_mixed_precision_quantization_config(
        quantization="4bit", excluded_module_paths=MAIN_MERGER_PATHS
    )
    model = AutoModelForVLM.from_pretrained(base, **kwargs)
    adapter_path = deployment_path / metadata.get("adapter_path", "adapter")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.eval()
    projector_mode = metadata.get("projector_mode")
    validation_config = SimpleNamespace(student=SimpleNamespace(
        train_multimodal_projector=projector_mode == "modules_to_save",
        use_projector_lora=projector_mode == "projector_lora",
    ))
    validate_high_fidelity_deployment(model, validation_config)
    print("High-fidelity quantized adapter deployment loaded; adapter_merged=false")
    return model, processor
