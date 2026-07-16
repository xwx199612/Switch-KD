"""Exact trainability and checkpoint helpers for multimodal students."""

from __future__ import annotations

import re

QWEN3_VL_PROJECTOR_PATH = "model.visual.merger"
QWEN3_VL_LANGUAGE_LAYER_COUNT = 36
QWEN3_VL_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
A2_PROJECTOR_LINEAR_NAMES = ("linear_fc1", "linear_fc2")
_LM_LORA_RE = re.compile(
    r"(?:^|.*\.)model\.language_model\.layers\.(\d+)\."
    r"(?:[^.]+\.)*([A-Za-z][A-Za-z0-9_]*)\.lora_[ab](?:\.|$)",
    re.IGNORECASE,
)
_LM_LAYER_RE = re.compile(r"(?:^|.*\.)model\.language_model\.layers\.(\d+)(?:\.|$)")


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


def resolve_a2_lora_targets(model, projector_path: str = QWEN3_VL_PROJECTOR_PATH,
                             *, expected_layer_count: int = QWEN3_VL_LANGUAGE_LAYER_COUNT) -> dict[str, object]:
    """Resolve only exact Qwen3-VL main-merger and LM attention module names."""
    if projector_path != QWEN3_VL_PROJECTOR_PATH:
        raise ValueError(f"A2 requires projector path {QWEN3_VL_PROJECTOR_PATH!r}, got {projector_path!r}.")
    merger = get_module_by_exact_path(model, projector_path)
    projector_targets = []
    for child in A2_PROJECTOR_LINEAR_NAMES:
        path = f"{projector_path}.{child}"
        try:
            module = merger.get_submodule(child)
        except AttributeError as exc:
            raise RuntimeError(f"A2 requires exact main-merger module {path}.") from exc
        if not hasattr(module, "weight"):
            raise RuntimeError(f"A2 main-merger target {path} is not a linear module.")
        projector_targets.append(path)
    attention_targets = []
    expected_layers = set(range(expected_layer_count))
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$", name)
        if match and hasattr(module, "weight"):
            attention_targets.append(
                f"model.language_model.layers.{match.group(1)}.self_attn.{match.group(2)}"
            )
    attention_targets = list(dict.fromkeys(attention_targets))
    found = {(name.rsplit(".", 1)[-1], int(name.split(".layers.", 1)[1].split(".", 1)[0]))
             for name in attention_targets}
    missing = [f"model.language_model.layers.{layer}.self_attn.{target}"
               for target in QWEN3_VL_ATTENTION_TARGETS for layer in sorted(expected_layers)
               if (target, layer) not in found]
    extra = sorted({layer for target, layer in found if layer not in expected_layers})
    if missing or extra:
        raise RuntimeError(
            f"A2 attention target resolution failed; missing {missing[:20]}, extra_layers={extra}"
        )
    if len(projector_targets) != 2:
        raise RuntimeError("A2 must resolve exactly two main-merger targets.")
    return {"attention_targets": attention_targets, "projector_targets": projector_targets,
            "all_targets": attention_targets + projector_targets}


def _is_projector_lora_parameter(name: str, allowed_paths: set[str]) -> bool:
    lowered = name.lower()
    if "lora_a" not in lowered and "lora_b" not in lowered:
        return False
    return any(parameter_matches_module_path(name, path) for path in allowed_paths)


def validate_language_model_lora_scope(
    model,
    configured_layers: list[int] | None,
    configured_targets: list[str],
    *,
    expected_layer_count: int = QWEN3_VL_LANGUAGE_LAYER_COUNT,
    projector_path: str = QWEN3_VL_PROJECTOR_PATH,
    allowed_projector_lora_paths: list[str] | None = None,
) -> dict[str, object]:
    """Validate PEFT trainability using only exact Qwen3-VL LM paths."""
    targets = list(dict.fromkeys(configured_targets))
    expected = set(range(expected_layer_count)) if configured_layers is None else set(configured_layers)
    architecture_layers = {
        int(match.group(1))
        for name, _ in model.named_modules()
        if (match := _LM_LAYER_RE.fullmatch(name))
    }
    if not architecture_layers:
        architecture_layers = {
            int(match.group(1))
            for name, _ in model.named_parameters()
            if (match := _LM_LAYER_RE.match(name))
        }
    if architecture_layers != set(range(expected_layer_count)):
        raise RuntimeError(
            "Language-model layer validation failed: detected layers "
            f"{sorted(architecture_layers)}; expected exactly 0-{expected_layer_count - 1}."
        )

    detected: dict[str, set[int]] = {target: set() for target in targets}
    unexpected_lora_targets: list[str] = []
    allowed_targets = {item.lower() for item in targets}
    trainable_lora: list[tuple[str, object]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if "lora_" not in lowered:
            continue
        trainable_lora.append((name, parameter))
        match = _LM_LORA_RE.search(name)
        if match is not None:
            target = match.group(2).lower()
            if target in allowed_targets:
                detected[next(item for item in targets if item.lower() == target)].add(int(match.group(1)))
            elif target not in unexpected_lora_targets:
                unexpected_lora_targets.append(target)

    missing = {target: sorted(expected - detected[target]) for target in targets}
    unexpected = {target: sorted(detected[target] - expected) for target in targets}
    allowed_projector = set(allowed_projector_lora_paths or [])
    visual_lora = [name for name, _ in trainable_lora
                   if _LM_LORA_RE.search(name) is None and not _is_projector_lora_parameter(name, allowed_projector)]
    mlp_lora = [name for name, _ in trainable_lora
                if any(f".{target}.lora_" in name.lower() for target in ("gate_proj", "up_proj", "down_proj"))]
    projector = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter_matches_module_path(name, projector_path)
                 and not _is_projector_lora_parameter(name, allowed_projector)]
    base_model = [name for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "lora_" not in name.lower()
                  and ("model.language_model." in name or ".language_model." in name)]
    vision = [name for name, parameter in model.named_parameters()
              if parameter.requires_grad and not _is_projector_lora_parameter(name, allowed_projector)
              and any(term in name.lower() for term in
                      ("visual", "vision_tower", "vision_model", "patch_embed"))]
    other = [name for name, parameter in model.named_parameters()
             if parameter.requires_grad and "lora_" not in name.lower()
             and name not in projector and name not in base_model and name not in vision]
    if (set(targets) - set(QWEN3_VL_ATTENTION_TARGETS) or any(missing.values())
            or any(unexpected.values()) or unexpected_lora_targets or visual_lora or mlp_lora
            or projector or base_model or vision or other):
        raise RuntimeError(
            "Language-model LoRA trainability validation failed: "
            f"missing={missing}, unexpected={unexpected}, visual_lora={visual_lora[:5]}, "
            f"unexpected_lora_targets={unexpected_lora_targets}, mlp_lora={mlp_lora[:5]}, "
            f"projector={projector[:5]}, base_model={base_model[:5]}, vision={vision[:5]}, other={other[:5]}"
        )

    report = {
        "configured_layers": sorted(expected),
        "detected_layers": {target: sorted(values) for target, values in detected.items()},
        "missing_layers": missing,
        "unexpected_layers": unexpected,
        "trainable_tensor_count": len(trainable_lora),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable_lora),
    }
    print(f"Configured LoRA layers: {report['configured_layers']}")
    print(f"Detected trainable LoRA layers: {report['detected_layers']}")
    print(f"Missing selected layers: {report['missing_layers']}")
    print(f"Unexpected trainable layers: {report['unexpected_layers']}")
    print(f"Trainable tensor count: {report['trainable_tensor_count']}")
    print(f"Trainable parameter count: {report['trainable_parameter_count']}")
    return report


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


def prepare_projector_for_lora(model, projector_path: str = QWEN3_VL_PROJECTOR_PATH, *, dtype=None) -> dict[str, object]:
    """Prepare only A2's two main-merger linears; their base weights remain frozen."""
    import torch
    dtype = torch.bfloat16 if dtype is None else dtype
    resolved = resolve_a2_lora_targets(model, projector_path)
    projector = get_module_by_exact_path(model, projector_path)
    converted, metadata = _replace_quantized_linears(
        projector, dtype=dtype, prefix=projector_path, validate_forward=False
    )
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    for path in resolved["projector_targets"]:
        module = get_module_by_exact_path(model, path)
        if not module.weight.is_floating_point():
            raise RuntimeError(f"A2 projector LoRA requires floating-point base weight at {path}.")
    return {"converted_linears": converted, "metadata": metadata,
            "projector_targets": resolved["projector_targets"]}


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
        "attention_lora": 0, "projector_lora": 0, "projector_full_train": 0,
        "llm_mlp_lora": 0, "vision_encoder": 0, "base_llm": 0,
        "other": 0, "projector": 0, "total": 0, "trainable": 0,
    }
    for name, parameter in model.named_parameters():
        groups["total"] += parameter.numel()
        if not parameter.requires_grad:
            continue
        groups["trainable"] += parameter.numel()
        lowered = name.lower()
        if "lora_a" in lowered or "lora_b" in lowered:
            if any(target in lowered for target in ("q_proj", "k_proj", "v_proj", "o_proj")):
                groups["attention_lora"] += parameter.numel()
            elif parameter_matches_module_path(name, projector_path):
                groups["projector_lora"] += parameter.numel()
            elif any(f".{target}." in lowered for target in ("gate_proj", "up_proj", "down_proj")):
                groups["llm_mlp_lora"] += parameter.numel()
            else:
                groups["other"] += parameter.numel()
        elif parameter_matches_module_path(name, projector_path):
            groups["projector_full_train"] += parameter.numel()
        elif any(term in lowered for term in ("visual", "vision_tower", "vision_model", "patch_embed")):
            groups["vision_encoder"] += parameter.numel()
        elif "model.language_model" in lowered or ".language_model." in lowered:
            groups["base_llm"] += parameter.numel()
        else:
            groups["other"] += parameter.numel()
    groups["projector"] = groups["projector_lora"] + groups["projector_full_train"]
    return groups


def validate_a2_projector_lora_contract(
    model, *, projector_path: str = QWEN3_VL_PROJECTOR_PATH,
    expected_layer_count: int = QWEN3_VL_LANGUAGE_LAYER_COUNT,
) -> dict[str, object]:
    """Fail-fast A2 contract: all LM QKVO plus only main-merger LoRA A/B train."""
    resolved = resolve_a2_lora_targets(model, projector_path, expected_layer_count=expected_layer_count)
    allowed = set(resolved["projector_targets"])
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    illegal = []
    attention = {target: set() for target in QWEN3_VL_ATTENTION_TARGETS}
    projector_lora = []
    for name, parameter in trainable:
        lowered = name.lower()
        lm_match = _LM_LORA_RE.search(name)
        if lm_match and lm_match.group(2).lower() in {x.lower() for x in QWEN3_VL_ATTENTION_TARGETS}:
            attention[lm_match.group(2).lower()].add(int(lm_match.group(1)))
        elif _is_projector_lora_parameter(name, allowed):
            projector_lora.append(name)
        else:
            illegal.append(name)
    expected = set(range(expected_layer_count))
    missing = {target: sorted(expected - layers) for target, layers in attention.items()}
    if any(missing.values()) or len(projector_lora) != 4 or illegal:
        raise RuntimeError(
            "A2 projector LoRA trainability validation failed; illegal parameters (first 20): "
            f"{illegal[:20]}; missing attention={missing}; projector_lora={projector_lora}"
        )
    report = {
        "attention_lora_parameters": sum(p.numel() for n, p in trainable if _LM_LORA_RE.search(n)),
        "projector_lora_parameters": sum(p.numel() for n, p in trainable if n in projector_lora),
        "attention_targets": resolved["attention_targets"],
        "projector_targets": resolved["projector_targets"],
    }
    print("Experiment mode: A2 attention LoRA + projector LoRA")
    print(f"Projector path: {projector_path}")
    print("Projector LoRA targets:")
    for path in resolved["projector_targets"]:
        print(f"  - {path}")
    print(f"Attention LoRA layers: 0-{expected_layer_count - 1}")
    print("Attention LoRA targets: q_proj,k_proj,v_proj,o_proj")
    print("Projector fully trainable: false")
    print("Projector modules_to_save: false")
    print("Deepstack merger LoRA count: 0")
    print("MLP LoRA count: 0")
    return report


def validate_projector_path(model, projector_path: str) -> None:
    """Validate the configured path and print nearby names for reproducibility."""
    get_module_by_exact_path(model, projector_path)
    print("Projector-focused loaded student module names:")
    for name in find_relevant_module_names(model):
        print(f"  {name}")
    print(f"Qwen3-VL multimodal projector/merger path: {projector_path}")
