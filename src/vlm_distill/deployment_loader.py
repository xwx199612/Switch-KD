"""Loader and runtime checks for the non-merged 4-bit/BF16 deployment bundle."""

from __future__ import annotations

import json
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import torch

from .mixed_precision import build_mixed_precision_quantization_config
from .model_loading import apply_attn_implementation, resolve_model_path
from .student_trainability import (
    QWEN3_VL_ATTENTION_TARGETS, QWEN3_VL_MLP_TARGETS,
    merger_base_checksum, merger_dtype_map, tensor_storage_bytes,
)

MAIN_MERGER_PATHS = [
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
]

_RUNTIME_LORA_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.language_model\.layers\.(\d+)\."
    r"(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_([ab])(?:\.[^.]+)?\.weight$",
    re.IGNORECASE,
)
_PROJECTOR_LORA_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.visual\.merger\."
    r"(linear_fc1|linear_fc2)\.lora_([ab])(?:\.[^.]+)?\.weight$",
    re.IGNORECASE,
)
_LEGACY_SUMMARY_LORA_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.language_model\.(attention\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.lora_[ab]\.weight$",
    re.IGNORECASE,
)
_LEGACY_SUMMARY_PROJECTOR_LORA_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.visual\.merger\.(linear_fc1|linear_fc2)\.lora_[ab]\.weight$",
    re.IGNORECASE,
)
_LEGACY_SUMMARY_LORA_MODULE_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.language_model\.(?:attention\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.lora_[ab]\.(?:weight|bias)$",
    re.IGNORECASE,
)
_LEGACY_SUMMARY_PROJECTOR_LORA_MODULE_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.visual\.merger\.(?:linear_fc1|linear_fc2)\.lora_[ab]\.(?:weight|bias)$",
    re.IGNORECASE,
)
_ANY_LORA_PARAMETER_RE = re.compile(r"(?:^|\.)lora_[ab](?:\.|$)", re.IGNORECASE)


def _processor_is_loadable(path: Path) -> bool:
    if not path.is_dir() or not any(path.iterdir()):
        return False
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(
            str(path), trust_remote_code=True, use_fast=False, local_files_only=True,
        )
    except Exception:
        return False
    return True


def _tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor_storage_bytes(tensor))
    return digest.hexdigest()


def _projector_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    marker = "model.visual.merger."
    for name, tensor in state.items():
        if marker in name and "modules_to_save.default." in name:
            suffix = name.split("modules_to_save.default.", 1)[1]
            result[suffix] = tensor
    return result


def projector_checksum_from_adapter_checkpoint(path: str | Path) -> str | None:
    path = Path(path)
    checkpoint = path / "adapter_model.safetensors"
    if not checkpoint.exists():
        checkpoint = path / "adapter_model.bin"
    if not checkpoint.exists():
        return None
    try:
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(checkpoint), device="cpu")
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tensors = _projector_tensors(state)
        return _tensor_digest(tensors) if tensors else None
    except Exception:
        return None


def _walk_module(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _deployment_roots(model: Any) -> list[tuple[str, Any]]:
    """Return the model roots used by the known PEFT wrapper layouts."""
    base_model = getattr(model, "base_model", None)
    return [
        ("model", model),
        ("base_model", base_model),
        ("base_model.model", getattr(base_model, "model", None)),
        ("model.model", getattr(model, "model", None)),
    ]


def _resolve_exact_module_with_path(model: Any, path: str) -> tuple[Any, str]:
    """Resolve the main module path across unwrapped and PEFT model layouts.

    The suffix is exact: a module is accepted only when its name ends in the
    requested path, never merely because it contains ``merger``.  The only
    accepted prefixes are the deterministic PEFT prefixes observed in saved
    Qwen-VL models.
    """
    roots = _deployment_roots(model)
    root_descriptions = [name for name, root in roots if root is not None]
    attempts: list[str] = []
    candidates: dict[str, Any] = {}
    allowed_prefixes = {"", "model", "base_model", "base_model.model", "base_model.model.model"}

    # named_modules gives the actual runtime path and avoids accidentally
    # treating an arbitrary attribute called ``merger`` as the main merger.
    try:
        named_modules = dict(model.named_modules())
    except (AttributeError, TypeError):
        named_modules = {}
    suffix = "." + path
    for name, module in named_modules.items():
        if name == path:
            candidates[name] = module
        elif name.endswith(suffix):
            prefix = name[: -len(suffix)].rstrip(".")
            if prefix in allowed_prefixes:
                candidates[name] = module

    # Keep the explicit roots as a fallback for lightweight model doubles that
    # do not implement named_modules(), while recording every attempted root.
    direct_candidates: dict[str, Any] = {}
    for root_name, root in roots:
        if root is None:
            attempts.append(f"{root_name}: <missing>")
            continue
        try:
            resolved = _walk_module(root, path)
        except AttributeError as exc:
            attempts.append(f"{root_name}: {exc}")
        else:
            full_name = path if root_name == "model" else f"{root_name}.{path}"
            direct_candidates.setdefault(full_name, resolved)
            attempts.append(f"{root_name}: {full_name}")

    # Prefer the authoritative named_modules result.  This also prevents a
    # successful traversal from a shorter alias root from being counted as a
    # second match for the same PEFT-wrapped module.
    if not candidates:
        candidates = direct_candidates

    if len(candidates) == 1:
        return next(iter(candidates.values())), next(iter(candidates))

    projector_candidates = [
        name for name in named_modules
        if ".visual." in name and "merger" in name
    ]
    details = ", ".join(projector_candidates) if projector_candidates else "<none>"
    raise AttributeError(
        f"unable to resolve exact module {path!r}; requested path={path!r}; "
        f"tried roots={root_descriptions!r}; attempts={attempts!r}; "
        f"projector candidates named_modules=[{details}]"
    )


def resolve_exact_module(model: Any, path: str) -> Any:
    """Resolve an exact deployment module path through PEFT wrappers."""
    module, _ = _resolve_exact_module_with_path(model, path)
    return module


def _projector_wrapper(model: Any) -> Any:
    wrapper, resolved_path = _resolve_exact_module_with_path(model, "model.visual.merger")
    print(f"resolved_projector_wrapper_path={resolved_path}")
    print(f"resolved_projector_wrapper_type={type(wrapper)}")
    active_copy = _modules_to_save_default(wrapper)
    if active_copy is not None:
        print("active_projector_source=modules_to_save.default")
        print(f"active_projector_type={type(active_copy)}")
    return wrapper


def _modules_to_save_default(wrapper: Any) -> Any | None:
    modules_to_save = getattr(wrapper, "modules_to_save", None)
    if modules_to_save is None:
        return None
    try:
        return modules_to_save["default"]
    except (KeyError, IndexError, TypeError):
        return getattr(modules_to_save, "default", None)


def _is_lora(name: str) -> bool:
    lower = name.lower()
    return "lora_a" in lower or "lora_b" in lower


def collect_runtime_lora_inventory(model: Any) -> dict[str, Any]:
    """Collect runtime PEFT LoRA tensors from parameter names.

    PEFT exposes adapters as parameters such as ``lora_A.default.weight``.
    This inventory deliberately accepts only the exact Qwen3-VL language-model
    layout and the two main projector linears; everything else is reported as
    unmatched instead of being guessed into a target group.
    """
    attention_names: list[str] = []
    mlp_names: list[str] = []
    projector_names: list[str] = []
    attention_dtypes: set[str] = set()
    mlp_dtypes: set[str] = set()
    projector_dtypes: set[str] = set()
    attention_modules: set[tuple[int, str, str]] = set()
    mlp_modules: set[tuple[int, str, str]] = set()
    components: dict[tuple[str, int, str, str], set[str]] = {}
    unmatched: list[str] = []
    detected: dict[str, set[int]] = {target: set() for target in (*QWEN3_VL_ATTENTION_TARGETS, *QWEN3_VL_MLP_TARGETS)}

    for name, parameter in model.named_parameters():
        match = _RUNTIME_LORA_PARAMETER_RE.fullmatch(name)
        if match:
            layer, branch, target, component = int(match.group(1)), match.group(2).lower(), match.group(3).lower(), match.group(4).lower()
            if ((branch == "self_attn" and target not in QWEN3_VL_ATTENTION_TARGETS)
                    or (branch == "mlp" and target not in QWEN3_VL_MLP_TARGETS)):
                unmatched.append(name)
                continue
            key = (layer, branch, target)
            group = "attention" if branch == "self_attn" else "mlp"
            (attention_modules if group == "attention" else mlp_modules).add(key)
            (attention_names if group == "attention" else mlp_names).append(name)
            (attention_dtypes if group == "attention" else mlp_dtypes).add(str(parameter.dtype))
            components.setdefault((group, *key), set()).add(component)
            detected[target].add(layer)
            continue
        match = _PROJECTOR_LORA_PARAMETER_RE.fullmatch(name)
        if match:
            projector_names.append(name)
            projector_dtypes.add(str(parameter.dtype))
            continue
        # Keep the small pre-runtime test doubles useful without allowing
        # their non-layer paths to satisfy the A3 exact layer contract.
        legacy = _LEGACY_SUMMARY_LORA_PARAMETER_RE.fullmatch(name)
        if legacy:
            (mlp_dtypes if ".mlp." in name.lower() else attention_dtypes).add(str(parameter.dtype))
            continue
        if _LEGACY_SUMMARY_PROJECTOR_LORA_PARAMETER_RE.fullmatch(name):
            projector_dtypes.add(str(parameter.dtype))
            continue
        if (_LEGACY_SUMMARY_LORA_MODULE_PARAMETER_RE.fullmatch(name)
                or _LEGACY_SUMMARY_PROJECTOR_LORA_MODULE_PARAMETER_RE.fullmatch(name)):
            continue
        if _ANY_LORA_PARAMETER_RE.search(name):
            unmatched.append(name)

    missing_components = {
        f"{group}:{layer}:{branch}:{target}": sorted({"a", "b"} - parts)
        for (group, layer, branch, target), parts in sorted(components.items())
        if parts != {"a", "b"}
    }
    return {
        "attention_lora_names": sorted(attention_names),
        "mlp_lora_names": sorted(mlp_names),
        "projector_lora_names": sorted(projector_names),
        "attention_tensor_count": len(attention_names),
        "mlp_tensor_count": len(mlp_names),
        "projector_tensor_count": len(projector_names),
        "attention_logical_modules": attention_modules,
        "mlp_logical_modules": mlp_modules,
        "attention_module_count": len(attention_modules),
        "mlp_module_count": len(mlp_modules),
        "attention_dtypes": sorted(attention_dtypes),
        "mlp_dtypes": sorted(mlp_dtypes),
        "projector_dtypes": sorted(projector_dtypes),
        "detected": detected,
        "missing_components": missing_components,
        "unmatched_lora_names": sorted(unmatched),
    }


def _active_merger(model: Any) -> Any:
    merger = _projector_wrapper(model)
    # PEFT's ModulesToSaveWrapper keeps the trained copy under this exact path.
    active_copy = _modules_to_save_default(merger)
    if active_copy is not None:
        return active_copy
    return merger


def _active_modules_to_save_projector(model: Any) -> Any | None:
    """Return the active full projector copy, but only for the default adapter copy."""
    wrapper = _projector_wrapper(model)
    active_copy = _modules_to_save_default(wrapper)
    active_adapter = getattr(wrapper, "active_adapter", "default")
    if isinstance(active_adapter, (list, tuple)):
        active_adapter = active_adapter[0] if active_adapter else "default"
    return active_copy if active_copy is not None and str(active_adapter) == "default" else None


def _summary(model: Any) -> dict[str, Any]:
    try:
        import bitsandbytes as bnb
        linear4bit = bnb.nn.Linear4bit
    except ImportError:
        linear4bit = ()
    inventory = collect_runtime_lora_inventory(model)
    counts = {"linear4bit": 0, "modules_to_save": 0, "vision_trainable": 0}
    for name, module in model.named_modules():
        if linear4bit and isinstance(module, linear4bit) and "language_model" in name:
            counts["linear4bit"] += 1
        if "modules_to_save" in name and "default" in name:
            counts["modules_to_save"] += sum(1 for _ in module.parameters())
    for name, parameter in model.named_parameters():
        if "visual" in name and "visual.merger" not in name and parameter.requires_grad:
            counts["vision_trainable"] += parameter.numel()
    counts.update({
        "attention_lora": inventory["attention_tensor_count"],
        "mlp_lora": inventory["mlp_tensor_count"],
        "projector_lora": inventory["projector_tensor_count"],
        "attention_lora_tensor_count": inventory["attention_tensor_count"],
        "mlp_lora_tensor_count": inventory["mlp_tensor_count"],
        "attention_lora_module_count": inventory["attention_module_count"],
        "mlp_lora_module_count": inventory["mlp_module_count"],
        "attention_lora_names": inventory["attention_lora_names"],
        "mlp_lora_names": inventory["mlp_lora_names"],
        "projector_lora_names": inventory["projector_lora_names"],
        "attention_dtypes": inventory["attention_dtypes"],
        "mlp_dtypes": inventory["mlp_dtypes"],
        "projector_lora_dtypes": inventory["projector_dtypes"],
        "runtime_lora_parameter_count": inventory["attention_tensor_count"] + inventory["mlp_tensor_count"] + inventory["projector_tensor_count"],
        "runtime_lora_parameter_examples": (inventory["attention_lora_names"] + inventory["mlp_lora_names"] + inventory["projector_lora_names"])[:20],
        "detected": inventory["detected"],
        "missing_lora_components": inventory["missing_components"],
        "unmatched_lora_parameter_names": inventory["unmatched_lora_names"],
    })
    active_projector = _active_modules_to_save_projector(model)
    modules_to_save_projector_dtypes = set()
    if active_projector is not None:
        modules_to_save_projector_dtypes = {
            str(parameter.dtype) for parameter in active_projector.parameters()
        }
    counts["modules_to_save_projector_dtypes"] = sorted(modules_to_save_projector_dtypes)
    # Compatibility for callers written before the split. New code must use the
    # explicit fields above; this alias intentionally represents LoRA only.
    counts["projector_dtypes"] = counts["projector_lora_dtypes"]
    return counts


def validate_high_fidelity_deployment(model: Any, config: Any = None, *, smoke_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the deployment contract and return a machine-readable summary."""
    merger = _active_merger(model)
    require_bf16_merger = getattr(getattr(config, "student", config), "main_merger_bf16", True) if config else True
    for child in ("linear_fc1", "linear_fc2"):
        layer = getattr(merger, child, None)

        # A2 wraps the projector Linear with a PEFT LoRA layer.
        base_layer = getattr(layer, "base_layer", layer)

        if require_bf16_merger and (
            type(base_layer) is not torch.nn.Linear
            or base_layer.weight.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                f"main merger {child} base layer must be "
                "torch.nn.Linear with BF16 weights; "
                f"runtime_type={type(layer)!r}, "
                f"base_type={type(base_layer)!r}, "
                f"dtype={getattr(getattr(base_layer, 'weight', None), 'dtype', None)}"
            )
    norm = getattr(merger, "norm", None)
    if require_bf16_merger and norm is not None and any(parameter.dtype != torch.bfloat16 for parameter in norm.parameters()):
        raise RuntimeError(
            "deployment validation failed: active merger.norm "
            "must be torch.bfloat16 for inference"
        )
    summary = _summary(model)
    print(f"runtime LoRA parameter count: {summary['runtime_lora_parameter_count']}")
    print(f"runtime LoRA parameter examples: {summary['runtime_lora_parameter_examples']}")
    print(f"detected attention modules: {summary['attention_lora_module_count']}")
    print(f"detected MLP modules: {summary['mlp_lora_module_count']}")
    print(f"detected A/B tensor counts: attention={summary['attention_lora_tensor_count']} MLP={summary['mlp_lora_tensor_count']}")
    print(f"unmatched LoRA parameter names: {summary['unmatched_lora_parameter_names'][:20]}")
    if summary["unmatched_lora_parameter_names"]:
        raise RuntimeError(
            "deployment validation failed: unmatched runtime LoRA parameter names "
            f"(first 20)={summary['unmatched_lora_parameter_names'][:20]!r}"
        )
    if any("torch.int" in dtype or "torch.uint" in dtype for dtype in summary["attention_dtypes"] + summary["projector_lora_dtypes"]):
        raise RuntimeError("deployment validation failed: adapter tensors must remain floating point")
    if any("torch.int" in dtype or "torch.uint" in dtype for dtype in summary["mlp_dtypes"]):
        raise RuntimeError("deployment validation failed: MLP adapter tensors must remain floating point")
    if any(not dtype.startswith("torch.") or not getattr(torch, dtype.removeprefix("torch."), torch.float32).is_floating_point
           for dtype in summary["modules_to_save_projector_dtypes"]):
        raise RuntimeError("deployment validation failed: modules_to_save projector tensors must remain floating point")
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

    mode = getattr(getattr(config, "student", config), "train_multimodal_projector", False) if config else False
    use_projector_lora = getattr(getattr(config, "student", config), "use_projector_lora", False) if config else False
    if mode and (summary["modules_to_save"] <= 0 or summary["projector_lora"]):
        raise RuntimeError("A1 validation failed: active modules_to_save projector is missing or projector LoRA exists")
    if mode:
        wrapper = _projector_wrapper(model)
        active_copy = _modules_to_save_default(wrapper)
        if active_copy is None:
            raise RuntimeError("A1 validation failed: model.visual.merger.modules_to_save.default is missing")
        active_adapter = getattr(wrapper, "active_adapter", "default")
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0] if active_adapter else "default"
        if str(active_adapter) != "default":
            raise RuntimeError("A1 validation failed: trained projector copy is not the active adapter copy")
        expected_checksum = getattr(getattr(config, "student", config), "projector_checksum", None) if config else None
        if expected_checksum:
            actual_checksum = _tensor_digest({
                name.rsplit(".", 1)[-2] + "." + name.rsplit(".", 1)[-1]: parameter
                for name, parameter in active_copy.named_parameters()
            })
            if actual_checksum != expected_checksum:
                raise RuntimeError(
                    "A1 validation failed: active modules_to_save projector checksum does not match adapter checkpoint"
                )
    if use_projector_lora:
        if any(not getattr(torch, dtype.removeprefix("torch."), torch.float32).is_floating_point
               for dtype in summary["projector_lora_dtypes"]):
            raise RuntimeError("A2 validation failed: projector LoRA tensors must remain floating point")
        if summary["projector_lora"] <= 0:
            raise RuntimeError("A2 validation failed: projector LoRA is missing")
        deepstack_lora_names = [
            name
            for name, _ in model.named_parameters()
            if "deepstack_merger" in name.lower()
            and _is_lora(name)
        ]
        if deepstack_lora_names:
            raise RuntimeError(
                "A2 validation failed: deepstack merger LoRA is not allowed; "
                f"found={deepstack_lora_names[:20]!r}"
            )

        if summary["mlp_lora"] > 0:
            raise RuntimeError(
                "A2 validation failed: LLM MLP LoRA is not allowed; "
                f"found={summary['mlp_lora_names'][:20]!r}"
            )
        if any(not ("model.visual.merger.linear_fc1" in n or "model.visual.merger.linear_fc2" in n)
               for n in summary["projector_lora_names"]):
            raise RuntimeError("A2 validation failed: projector LoRA targets must be the two main merger linears")
    if not mode and not use_projector_lora and (summary["projector_lora"] or summary["modules_to_save"]):
        raise RuntimeError("A0 validation failed: projector must be base BF16 without projector adapter weights")
    if use_projector_lora and summary["modules_to_save"]:
        raise RuntimeError("A2 validation failed: modules_to_save projector is not allowed")
    if mode and summary["projector_lora"]:
        raise RuntimeError("A1 validation failed: projector LoRA is not allowed")
    configured_targets = list(getattr(getattr(config, "student", config), "target_modules", []) or []) if config else []
    if mode and set(configured_targets) & set(QWEN3_VL_MLP_TARGETS):
        expected_layers = set(range(36))
        expected_attn = set(QWEN3_VL_ATTENTION_TARGETS)
        expected_mlp = set(QWEN3_VL_MLP_TARGETS)
        detected = summary["detected"]
        missing_layers = {
            target: sorted(expected_layers - detected[target])
            for target in (*QWEN3_VL_ATTENTION_TARGETS, *QWEN3_VL_MLP_TARGETS)
            if expected_layers - detected[target]
        }
        extra_layers = {
            target: sorted(detected[target] - expected_layers)
            for target in (*QWEN3_VL_ATTENTION_TARGETS, *QWEN3_VL_MLP_TARGETS)
            if detected[target] - expected_layers
        }
        missing_components = summary["missing_lora_components"]
        if (summary["attention_lora_module_count"] != 36 * len(expected_attn)
                or summary["mlp_lora_module_count"] != 36 * len(expected_mlp)
                or missing_layers or extra_layers or missing_components):
            raise RuntimeError(
                "A3 deployment validation failed: expected 144 attention and 108 MLP LoRA modules, "
                f"got {summary['attention_lora_module_count']} and {summary['mlp_lora_module_count']}; "
                f"missing_layers={missing_layers}; extra_layers={extra_layers}; "
                f"missing_A_or_B={missing_components}"
            )
        if summary["projector_lora"] or summary["modules_to_save"] <= 0:
            raise RuntimeError("A3 deployment validation failed: full modules_to_save projector is required")
    if smoke_inputs is not None:
        with torch.no_grad():
            outputs = model(**smoke_inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None or logits.numel() == 0 or not torch.isfinite(logits).all():
            raise RuntimeError("deployment validation failed: logits smoke test was not finite")
    print(f"language_model Linear4bit count: {summary['linear4bit']}")
    print("main merger BF16 linear count: 2")
    print(f"attention LoRA tensor count: {summary['attention_lora']}")
    print(f"MLP LoRA tensor count: {summary['mlp_lora']}")
    print(f"projector LoRA tensor count: {summary['projector_lora']}")
    print(f"modules_to_save tensor count: {summary['modules_to_save']}")
    print(f"Attention LoRA dtype summary: {summary['attention_dtypes']}")
    print(f"MLP LoRA dtype summary: {summary['mlp_dtypes']}")
    print(f"Modules-to-save projector dtype summary: {summary['modules_to_save_projector_dtypes']}")
    print(f"Projector LoRA dtype summary: {summary['projector_lora_dtypes']}")
    if norm is not None:
        print(f"deployment_merger_norm_dtype_after_peft={next(norm.parameters()).dtype}")
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
    processor_path = deployment_path / metadata.get("processor_path", "processor") if metadata.get("processor_path") else None
    processor_source = processor_path if processor_path and _processor_is_loadable(processor_path) else Path(base)
    if processor_path is not None and processor_path.exists() and processor_source != processor_path:
        shutil.rmtree(processor_path)
    processor = AutoProcessor.from_pretrained(str(processor_source), trust_remote_code=True, use_fast=False, local_files_only=True)
    kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True, "local_files_only": True}
    apply_attn_implementation(kwargs, metadata.get("attn_implementation", "sdpa"))
    excluded = metadata.get("excluded_from_quantization", MAIN_MERGER_PATHS)
    kwargs["quantization_config"] = build_mixed_precision_quantization_config(
        quantization="4bit", excluded_module_paths=excluded
    )
    model = AutoModelForVLM.from_pretrained(base, **kwargs)
    # The checksum is intentionally computed before PEFT attaches any adapter.
    # A2 stores only deltas, so this is the base-weight identity contract.
    merger = resolve_exact_module(model, "model.visual.merger")
    if metadata.get("projector_mode") in {"projector_lora", "modules_to_save"}:
        for child in ("linear_fc1", "linear_fc2"):
            getattr(merger, child).to(dtype=torch.bfloat16)
        merger.norm.to(dtype=torch.float32)
        expected_dtype_map = metadata.get("base_projector_dtype_map")
        actual_dtype_map = merger_dtype_map(model)
        if expected_dtype_map and actual_dtype_map != expected_dtype_map:
            raise RuntimeError("A2 deployment base projector dtype map does not match the projector used during training")
        expected_checksum = metadata.get("base_projector_checksum_before_lora")
        if expected_checksum:
            actual_checksum = merger_base_checksum(model)
            if actual_checksum != expected_checksum:
                raise RuntimeError("A2 deployment base projector does not match the projector used during training")
        print(f"deployment_merger_norm_dtype_before_peft={next(merger.norm.parameters()).dtype}")
    adapter_path = deployment_path / metadata.get("adapter_path", "adapter")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.eval()

    # Generation runs outside training autocast. Keep the active projector
    # executable with BF16 visual hidden states.
    active_merger = _active_merger(model)
    for child in ("linear_fc1", "linear_fc2"):
        getattr(active_merger, child).to(dtype=torch.bfloat16)
    active_merger.norm.to(dtype=torch.bfloat16)
    print(
        "deployment_active_merger_norm_dtype_after_peft="
        f"{next(active_merger.norm.parameters()).dtype}"
    )

    projector_mode = metadata.get("projector_mode")
    validation_config = SimpleNamespace(student=SimpleNamespace(
        train_multimodal_projector=projector_mode == "modules_to_save",
        use_projector_lora=projector_mode == "projector_lora",
        projector_checksum=(metadata.get("projector_checksum") or projector_checksum_from_adapter_checkpoint(adapter_path)),
        target_modules=[*metadata.get("lora_target_groups", {}).get("attention", []),
                        *metadata.get("lora_target_groups", {}).get("mlp", [])],
        main_merger_bf16=bool(metadata.get("main_merger_bf16", True)),
    ))
    validate_high_fidelity_deployment(model, validation_config)
    print(f"prediction_model_source={metadata.get('artifact_mode')}")
    print(f"experiment_mode={metadata.get('experiment_mode', '<unspecified>')}")
    print("adapter_merged=false")
    print(f"language_model_quantization={metadata.get('quantization', '4bit_nf4')}")
    print("main_merger_dtype=bfloat16")
    print(f"attention_lora_active={bool(metadata.get('lora_target_groups', {}).get('attention'))}")
    print(f"mlp_lora_active={bool(metadata.get('lora_target_groups', {}).get('mlp'))}")
    print(f"modules_to_save_projector_active={metadata.get('projector_mode') == 'modules_to_save'}")
    print("High-fidelity quantized adapter deployment loaded; adapter_merged=false")
    return model, processor
