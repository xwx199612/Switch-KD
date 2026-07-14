"""Exact trainability and checkpoint helpers for multimodal students."""

from __future__ import annotations

QWEN3_VL_PROJECTOR_PATH = "model.visual.merger"


def get_module_by_exact_path(model, path: str):
    """Return a module by its exact dotted path, without keyword matching."""
    roots = [model]
    if hasattr(model, "base_model"):
        roots.append(model.base_model)
        if hasattr(model.base_model, "model"):
            roots.append(model.base_model.model)
    parts = path.split(".")
    for root in roots:
        current = root
        for part in parts:
            if not hasattr(current, part):
                break
            current = getattr(current, part)
        else:
            if not hasattr(current, "parameters"):
                raise TypeError(f"Resolved exact path {path!r} is not a module.")
            return current
    raise AttributeError(f"Model has no module at exact path {path!r}.")


def _is_bitsandbytes_linear(module) -> bool:
    """Return whether *module* is a bitsandbytes quantized linear layer."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))


def _dequantized_weight(module):
    import torch

    weight = module.weight
    # Params4bit exposes dequantize(), while the functional fallback also
    # works with versions of bitsandbytes that do not expose it there.
    if hasattr(weight, "dequantize"):
        return weight.dequantize()
    import bitsandbytes.functional as bnb_functional

    return bnb_functional.dequantize_4bit(weight, quant_state=weight.quant_state)


def _replace_quantized_linears(module, *, dtype) -> int:
    """Replace quantized linear descendants in-place, preserving the tree."""
    import torch
    from torch import nn

    converted = 0
    for child_name, child in list(module.named_children()):
        if _is_bitsandbytes_linear(child):
            device = child.weight.device
            linear = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(_dequantized_weight(child).to(device=device, dtype=dtype))
                if child.bias is not None:
                    linear.bias.copy_(child.bias.detach().to(device=device, dtype=dtype))
            setattr(module, child_name, linear)
            converted += 1
        else:
            converted += _replace_quantized_linears(child, dtype=dtype)
    return converted


def dequantize_trainable_projector(model, projector_path: str, *, dtype=None) -> dict[str, str | int]:
    """Convert only a configured projector's bitsandbytes linears to BF16."""
    import torch

    dtype = torch.bfloat16 if dtype is None else dtype
    projector = get_module_by_exact_path(model, projector_path)
    before = {name: str(parameter.dtype) for name, parameter in projector.named_parameters()}
    converted = _replace_quantized_linears(projector, dtype=dtype)
    # A quantized parameter can also be attached directly to a custom module;
    # reject it here rather than allowing PEFT to fail deep in modules_to_save.
    remaining = [name for name, parameter in projector.named_parameters()
                 if not parameter.is_floating_point()]
    if remaining:
        raise RuntimeError(
            "Configured fully trainable projector still contains non-floating parameters "
            f"after dequantization: {remaining}"
        )
    after = {name: str(parameter.dtype) for name, parameter in projector.named_parameters()}
    return {"converted_linears": converted, "before": before, "after": after}


def validate_projector_trainable_parameters(model, projector_path: str) -> None:
    projector = get_module_by_exact_path(model, projector_path)
    bad = [
        f"{projector_path}.{name}" if name else projector_path
        for name, parameter in projector.named_parameters()
        if parameter.requires_grad and not parameter.is_floating_point()
    ]
    if bad:
        raise RuntimeError(
            "Configured fully trainable projector still contains non-floating parameters "
            f"after dequantization: {bad}"
        )


def parameter_matches_module_path(name: str, path: str) -> bool:
    """Match a module path, including PEFT's deterministic wrapper prefixes."""
    module_name = name.rsplit(".", 1)[0]
    dotted = "." + module_name + "."
    return module_name == path or module_name.endswith("." + path) or f".{path}." in dotted


def find_relevant_module_names(model) -> list[str]:
    terms = ("merger", "projector", "visual", "multi_modal", "mm_projector", "mlp")
    return [name for name, module in model.named_modules() if any(term in name.lower() for term in terms)]


def summarize_trainable_groups(model, projector_path: str) -> dict[str, int]:
    groups = {
        "attention_lora": 0, "projector": 0, "vision_encoder": 0,
        "base_llm": 0, "other": 0, "total": 0, "trainable": 0,
    }
    for name, parameter in model.named_parameters():
        groups["total"] += parameter.numel()
        if not parameter.requires_grad:
            continue
        groups["trainable"] += parameter.numel()
        lowered = name.lower()
        if parameter_matches_module_path(name, projector_path):
            groups["projector"] += parameter.numel()
        elif "lora_a" in lowered or "lora_b" in lowered:
            if any(target in lowered for target in ("q_proj", "k_proj", "v_proj", "o_proj")):
                groups["attention_lora"] += parameter.numel()
            else:
                groups["other"] += parameter.numel()
        elif any(term in lowered for term in ("visual", "vision_tower", "vision_model", "patch_embed")):
            groups["vision_encoder"] += parameter.numel()
        elif "model.language_model" in lowered or ".language_model." in lowered:
            groups["base_llm"] += parameter.numel()
        else:
            groups["other"] += parameter.numel()
    return groups


def validate_projector_path(model, projector_path: str) -> None:
    """Validate the configured path and print nearby names for reproducibility."""
    get_module_by_exact_path(model, projector_path)
    print("Relevant loaded student module names:")
    for name in find_relevant_module_names(model):
        print(f"  {name}")
    print(f"Qwen3-VL multimodal projector/merger path: {projector_path}")
