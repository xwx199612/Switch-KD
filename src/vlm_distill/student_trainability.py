"""Exact trainability and checkpoint helpers for multimodal students."""

from __future__ import annotations

QWEN3_VL_PROJECTOR_PATH = "model.visual.merger"


def get_module_by_exact_path(model, path: str):
    """Return a module by its exact dotted path, without keyword matching."""
    current = model
    # PEFT wraps the original model as base_model.model.  Resolve the
    # configured path against the original model when possible.
    if not hasattr(current, path.split(".")[0]) and hasattr(current, "base_model"):
        current = current.base_model
        if hasattr(current, "model"):
            current = current.model
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"Model has no module at exact path {path!r} (missing {part!r}).")
        current = getattr(current, part)
    if not hasattr(current, "parameters"):
        raise TypeError(f"Resolved exact path {path!r} is not a module.")
    return current


def parameter_matches_module_path(name: str, path: str) -> bool:
    """Match a module path, including PEFT's deterministic wrapper prefixes."""
    module_name = name.rsplit(".", 1)[0]
    dotted = "." + module_name + "."
    return module_name == path or module_name.endswith("." + path) or f".{path}." in dotted


def find_relevant_module_names(model) -> list[str]:
    terms = ("merger", "projector", "visual", "multi_modal", "mm_projector", "mlp")
    return [name for name, module in model.named_modules() if any(term in name.lower() for term in terms)]


def summarize_trainable_groups(model, projector_path: str) -> dict[str, int]:
    groups = {"llm_lora": 0, "projector": 0, "vision_encoder": 0, "other": 0, "total": 0, "trainable": 0}
    for name, parameter in model.named_parameters():
        groups["total"] += parameter.numel()
        if not parameter.requires_grad:
            continue
        groups["trainable"] += parameter.numel()
        lowered = name.lower()
        if parameter_matches_module_path(name, projector_path):
            groups["projector"] += parameter.numel()
        elif "lora_a" in lowered or "lora_b" in lowered:
            groups["llm_lora"] += parameter.numel()
        elif any(term in lowered for term in ("visual", "vision_tower", "vision_model", "patch_embed")):
            groups["vision_encoder"] += parameter.numel()
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
