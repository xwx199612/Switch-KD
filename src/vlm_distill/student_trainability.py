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


def _is_bnb_4bit_linear(module) -> bool:
    """Return whether *module* is specifically a bitsandbytes 4-bit linear."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return isinstance(module, bnb.nn.Linear4bit)


def _is_bnb_8bit_linear(module) -> bool:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return isinstance(module, bnb.nn.Linear8bitLt)


def _quant_state_type_name(quant_state) -> str:
    value = getattr(quant_state, "quant_type", None)
    if value is None and isinstance(quant_state, dict):
        value = quant_state.get("quant_type")
    return str(value or "<unknown>")


def _dequantized_weight(module):
    """Reconstruct a bnb Linear4bit weight using its actual NF4 state."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional

    if isinstance(module, bnb.nn.Linear4bit):
        weight = module.weight
        quant_state = getattr(weight, "quant_state", None)
        if quant_state is None:
            raise RuntimeError(
                "Cannot dequantize Linear4bit projector weight because quant_state is missing. "
                "Ensure the layer has been materialized on a supported CUDA device before conversion."
            )
        return bnb_functional.dequantize_4bit(weight.data, quant_state=quant_state)
    if isinstance(module, bnb.nn.Linear8bitLt):
        raise NotImplementedError(
            "Fully trainable projector conversion for bitsandbytes Linear8bitLt is not currently supported."
        )
    raise TypeError(f"Unsupported quantized projector layer type: {type(module)!r}")


def _projector_linear_metadata(module, *, target_dtype) -> dict[str, object]:
    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    return {
        "module_type": f"{type(module).__module__}.{type(module).__name__}",
        "weight_type": f"{type(weight).__module__}.{type(weight).__name__}",
        "weight_dtype": str(weight.dtype),
        "device": str(weight.device),
        "quant_state_present": quant_state is not None,
        "quant_type": _quant_state_type_name(quant_state) if quant_state is not None else "<missing>",
        "compute_dtype": str(getattr(module, "compute_dtype", "<unknown>")),
        "target_dtype": str(target_dtype),
    }


def _replace_quantized_linears(module, *, dtype, prefix="", validate_forward=True) -> tuple[int, list[dict[str, object]]]:
    """Replace quantized linear descendants in-place, preserving the tree."""
    import torch
    from torch import nn

    converted = 0
    metadata = []
    for child_name, child in list(module.named_children()):
        child_path = f"{prefix}.{child_name}" if prefix else child_name
        if _is_bnb_8bit_linear(child):
            raise NotImplementedError(
                "Fully trainable projector conversion for bitsandbytes Linear8bitLt is not currently supported "
                f"(at {child_path})."
            )
        if _is_bnb_4bit_linear(child):
            info = _projector_linear_metadata(child, target_dtype=dtype)
            print("Projector conversion:")
            print(f"  path={child_path}")
            for key in ("module_type", "weight_type", "weight_dtype", "device", "quant_state_present", "quant_type", "compute_dtype", "target_dtype"):
                print(f"  {key}={info[key]}")
            if not info["quant_state_present"]:
                raise RuntimeError(
                    "Cannot dequantize Linear4bit projector weight because quant_state is missing. "
                    f"module path={child_path}. Ensure the layer has been materialized on a supported CUDA device before conversion."
                )
            original_training = child.training
            original_weight_grad = child.weight.requires_grad
            original_bias_grad = child.bias.requires_grad if child.bias is not None else None
            original_output = None
            validation_input = None
            if validate_forward:
                import torch
                generator = torch.Generator(device=child.weight.device)
                generator.manual_seed(0)
                validation_input = torch.randn(
                    (2, child.in_features), device=child.weight.device, dtype=dtype, generator=generator
                )
                with torch.no_grad():
                    original_output = child(validation_input)
            dequantized = _dequantized_weight(child)
            device = child.weight.device
            linear = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(dequantized.to(device=device, dtype=dtype))
                if child.bias is not None:
                    linear.bias.copy_(child.bias.detach().to(device=device, dtype=dtype))
            linear.train(original_training)
            linear.weight.requires_grad_(original_weight_grad)
            if linear.bias is not None:
                linear.bias.requires_grad_(bool(original_bias_grad))
            if validate_forward:
                with torch.no_grad():
                    replacement_output = linear(validation_input)
                torch.testing.assert_close(
                    replacement_output.float(), original_output.float(), rtol=2e-2, atol=2e-2
                )
                print(f"  forward_equivalence=ok rtol=0.02 atol=0.02 path={child_path}")
            setattr(module, child_name, linear)
            converted += 1
            metadata.append({"path": child_path, **info, "replacement_type": "torch.nn.Linear"})
        else:
            nested_count, nested_metadata = _replace_quantized_linears(
                child, dtype=dtype, prefix=child_path, validate_forward=validate_forward
            )
            converted += nested_count
            metadata.extend(nested_metadata)
    return converted, metadata


def dequantize_trainable_projector(model, projector_path: str, *, dtype=None, validate_forward=True) -> dict[str, object]:
    """Convert only a configured projector's bitsandbytes linears to BF16."""
    import torch

    dtype = torch.bfloat16 if dtype is None else dtype
    projector = get_module_by_exact_path(model, projector_path)
    before = {name: str(parameter.dtype) for name, parameter in projector.named_parameters()}
    converted, metadata = _replace_quantized_linears(
        projector, dtype=dtype, prefix=projector_path, validate_forward=validate_forward
    )
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
    for item in metadata:
        relative_path = str(item["path"])
        prefix = projector_path + "."
        if relative_path.startswith(prefix):
            relative_path = relative_path[len(prefix):]
        module = projector.get_submodule(relative_path)
        if module.weight.dtype != dtype or not module.weight.is_floating_point():
            raise RuntimeError(f"Converted projector linear is not floating {dtype}: {item['path']}")
    return {"converted_linears": converted, "before": before, "after": after, "metadata": metadata}


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
    return [
        name for name, module in model.named_modules()
        if name == QWEN3_VL_PROJECTOR_PATH
        or name.startswith(QWEN3_VL_PROJECTOR_PATH + ".")
        or name == "model.visual.deepstack_merger_list"
    ]


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
    print("Projector-focused loaded student module names:")
    for name in find_relevant_module_names(model):
        print(f"  {name}")
    print(f"Qwen3-VL multimodal projector/merger path: {projector_path}")
