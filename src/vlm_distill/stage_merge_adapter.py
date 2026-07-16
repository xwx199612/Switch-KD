from __future__ import annotations

import json
from pathlib import Path

from .config_schema import PipelineConfig
from .config_schema import resolve_inference_manifest_path
from .mixed_precision import build_mixed_precision_quantization_config, mixed_precision_capabilities
from .model_loading import resolve_model_path
from .student_trainability import dequantize_trainable_projector, get_module_by_exact_path

A1_MAIN_MERGER_LINEAR_PATHS = [
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
]


def _print_merged_precision_summary(model, *, projector_path: str = "model.visual.merger") -> dict[str, int]:
    import torch
    counts = {
        "language_model quantized linears": 0,
        "language_model Linear4bit linears": 0,
        "language_model floating-point linears": 0,
        "visual encoder quantized linears": 0,
        "main merger BF16 linears": 0,
        "remaining LoRA modules": 0,
        "remaining modules_to_save wrappers": 0,
    }
    try:
        import bitsandbytes as bnb
    except ImportError:  # pragma: no cover - merge requires bnb for quantized configs
        bnb = None
    for name, module in model.named_modules():
        if "lora" in name.lower() and ("lora_a" in name.lower() or "lora_b" in name.lower()):
            counts["remaining LoRA modules"] += 1
        if "modules_to_save.default" in name:
            counts["remaining modules_to_save wrappers"] += 1
        if bnb is not None and isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            if "language_model" in name:
                counts["language_model quantized linears"] += 1
                if isinstance(module, bnb.nn.Linear4bit):
                    counts["language_model Linear4bit linears"] += 1
            elif "visual.merger" not in name:
                counts["visual encoder quantized linears"] += 1
        if "language_model" in name and type(module) is torch.nn.Linear:
            counts["language_model floating-point linears"] += 1
    try:
        merger = get_module_by_exact_path(model, projector_path)
    except (AttributeError, KeyError):
        merger = None
    if merger is not None:
        for module in merger.modules():
            if isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.bfloat16:
                counts["main merger BF16 linears"] += 1
    print("Merged model precision/module summary:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    return counts


def _validate_a1_merged_precision(model, counts: dict[str, int], *, projector_path: str) -> None:
    """Enforce the mixed 4-bit LLM/BF16-main-merger A1 post-merge contract."""
    import torch

    if counts["remaining LoRA modules"]:
        raise RuntimeError(
            "Merged A1 precision validation failed: "
            f"expected no remaining LoRA modules, observed {counts['remaining LoRA modules']}."
        )

    if counts["remaining modules_to_save wrappers"]:
        raise RuntimeError(
            "Merged A1 precision validation failed: "
            "expected no remaining modules_to_save wrappers, "
            f"observed {counts['remaining modules_to_save wrappers']}."
        )
    if counts.get("language_model Linear4bit linears", counts["language_model quantized linears"]) <= 0:
        raise RuntimeError(
            "Merged A1 precision validation failed: expected quantized language-model "
            "linears > 0, observed 0."
        )
    if counts.get("language_model floating-point linears", 0):
        raise RuntimeError(
            "Merged A1 precision validation failed: unexpected floating-point "
            "language-model linears, observed "
            f"{counts['language_model floating-point linears']}."
        )

    merger = get_module_by_exact_path(model, projector_path)
    merger_quantized = []
    try:
        import bitsandbytes as bnb
        quantized_types = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:  # pragma: no cover - A1 requires bitsandbytes at runtime
        quantized_types = ()
    if quantized_types:
        merger_quantized.extend(
            f"{projector_path}.{name}" if name else projector_path
            for name, layer in merger.named_modules()
            if isinstance(layer, quantized_types)
        )
    for relative_name, expected_dtype in (("linear_fc1", torch.bfloat16), ("linear_fc2", torch.bfloat16)):
        path = f"{projector_path}.{relative_name}"
        try:
            layer = merger.get_submodule(relative_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Merged A1 precision validation failed: missing exact main-merger module {path}."
            ) from exc
        if not isinstance(layer, torch.nn.Linear):
            raise RuntimeError(
                f"Merged A1 precision validation failed: {path} has unexpected type "
                f"{type(layer).__name__}; expected torch.nn.Linear."
            )
        if layer.weight.dtype != expected_dtype or not layer.weight.is_floating_point():
            raise RuntimeError(
                f"Merged A1 precision validation failed: {path} has dtype "
                f"{layer.weight.dtype}; expected torch.bfloat16 floating-point weights."
            )
    if merger_quantized:
        raise RuntimeError(
            "Merged A1 precision validation failed: bitsandbytes quantized linear remains "
            f"inside exact main merger path: {merger_quantized[0]}."
        )
    non_floating = [
        f"{projector_path}.{name}" if name else projector_path
        for name, parameter in merger.named_parameters()
        if not parameter.is_floating_point()
    ]
    if non_floating:
        raise RuntimeError(
            "Merged A1 precision validation failed: non-floating parameter remains under "
            f"{projector_path}: {non_floating[0]}."
        )
    if counts["main merger BF16 linears"] != 2:
        raise RuntimeError(
            "Merged A1 precision validation failed: expected exactly 2 BF16 main-merger "
            f"linears, observed {counts['main merger BF16 linears']}."
        )


def _validate_bf16_standalone_precision(model, counts: dict[str, int], *, projector_path: str) -> None:
    """Enforce the fully floating-point contract of a BF16 standalone artifact."""
    import torch

    if counts["remaining LoRA modules"] or counts["remaining modules_to_save wrappers"]:
        raise RuntimeError("BF16 standalone validation failed: PEFT wrappers remain after merge.")
    if counts["language_model quantized linears"]:
        raise RuntimeError(
            "BF16 standalone validation failed: language-model quantized linears remain."
        )
    if counts["language_model floating-point linears"] <= 0:
        raise RuntimeError(
            "BF16 standalone validation failed: expected floating-point language-model linears."
        )

    language_model = get_module_by_exact_path(model, "model.language_model")
    non_bf16 = [
        name for name, parameter in language_model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
    ]
    if non_bf16:
        raise RuntimeError(
            "BF16 standalone validation failed: language-model floating weights are not BF16; "
            f"first={non_bf16[0]}."
        )
    get_module_by_exact_path(model, "model.visual")
    get_module_by_exact_path(model, projector_path)


def _is_a1_projector_4bit(config: PipelineConfig) -> bool:
    return (
        config.student.quantization == "4bit"
        and config.student.train_multimodal_projector
        and config.student.multimodal_projector_path == "model.visual.merger"
    )


def _validate_artifact_mode(config: PipelineConfig) -> None:
    if _is_a1_projector_4bit(config) and config.student.merged_artifact_mode != "mixed_4bit_bf16":
        raise RuntimeError(
            "A1 requires student.merged_artifact_mode=mixed_4bit_bf16; refusing to silently "
            f"change the configured mode {config.student.merged_artifact_mode!r}."
        )


def _build_merge_model_kwargs(config: PipelineConfig, *, reload: bool = False) -> dict:
    """Build the model kwargs for both merge loading and artifact reloading.

    ``student.quantization`` describes the training/inference model.  A
    standalone BF16 artifact is deliberately reloaded without quantization,
    regardless of that setting.
    """
    import torch

    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": True,
        "attn_implementation": config.student.attn_implementation,
    }
    artifact_mode = config.student.merged_artifact_mode
    if artifact_mode == "bf16_standalone":
        return kwargs

    if artifact_mode == "mixed_4bit_bf16" and config.student.quantization == "4bit":
        kwargs["quantization_config"] = build_mixed_precision_quantization_config(
            quantization="4bit",
            excluded_module_paths=(
                A1_MAIN_MERGER_LINEAR_PATHS
                if _is_a1_projector_4bit(config)
                else []
            ),
        )
    elif config.student.quantization == "8bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return kwargs


_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_REPOSITORY_SAMPLE_IMAGE = Path("examples/images/sample_001.jpg")


def _usable_validation_image(path: Path) -> Path | None:
    """Return a validated image path, or None when the candidate is unusable."""
    from PIL import Image, UnidentifiedImageError

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        with path.open("rb") as handle:
            if handle.read(len(_GIT_LFS_POINTER_PREFIX)) == _GIT_LFS_POINTER_PREFIX:
                return None
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            rgb_image.load()
            rgb_image.close()
        return path.resolve()
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None


def _resolve_standalone_validation_image(config: PipelineConfig) -> Path | None:
    """Find the first usable configured inference image, with a repository fallback."""
    manifest_path = resolve_inference_manifest_path(config.data)
    image_root = config.data.image_root
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_value = row.get("image") if isinstance(row, dict) else None
                if not isinstance(image_value, str) or not image_value:
                    continue
                image_path = _usable_validation_image(image_root / image_value)
                if image_path is not None:
                    return image_path
    except (FileNotFoundError, OSError):
        pass

    return _usable_validation_image(_REPOSITORY_SAMPLE_IMAGE)


def _validate_standalone_merged_model(
    model, processor, output_path: Path, *, config: PipelineConfig
) -> None:
    """Smoke-test the saved standalone model when a usable image exists."""
    import torch
    from PIL import Image

    image_path = _resolve_standalone_validation_image(config)
    if image_path is None:
        print("Standalone merged image smoke test skipped: no valid validation image was found.")
        return

    print("Standalone merged image smoke test:")
    print(f"  image={image_path}")
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if outputs.logits.numel() == 0:
        raise RuntimeError("Standalone merged image inference produced empty logits.")
    if not torch.isfinite(outputs.logits).all():
        raise RuntimeError("Standalone merged image inference produced non-finite logits.")
    print("Standalone merged image inference: ok (finite logits)")


def merge_student_adapter(config: PipelineConfig) -> Path:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:  # pragma: no cover - fallback for older transformers
            from transformers import AutoModelForVision2Seq as AutoModelForVLM
    except ImportError as exc:
        raise RuntimeError(
            "Install torch, transformers, and peft to merge a student adapter."
        ) from exc

    _validate_artifact_mode(config)
    capabilities = mixed_precision_capabilities()
    print("Mixed-precision environment:")
    for key, value in capabilities.items():
        print(f"  {key}={value}")
    if _is_a1_projector_4bit(config) and not capabilities["artifact_mode_supported"]:
        raise RuntimeError(
            "A1 mixed standalone artifact is unsupported by the installed "
            "Transformers/bitsandbytes versions because exact module exclusion "
            "during 4-bit reload is unavailable.\n\n"
            "Use merged_artifact_mode=bf16_standalone or "
            "merged_artifact_mode=adapter_plus_projector."
        )

    base_model_path = resolve_model_path(config.student.model_name)
    adapter_path = config.student.inference_adapter_path or config.student.adapter_dir
    output_path = config.student.merged_model_path or config.student.output_dir / "merged_model"

    resolved_base_path = Path(base_model_path).resolve()
    resolved_output_path = output_path.resolve()
    if resolved_output_path == resolved_base_path:
        raise ValueError(
            "Refusing to overwrite the base model directory while merging the adapter. "
            "Set student.merged_model_path to a different output directory."
        )

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter path is missing adapter_config.json: {adapter_path / 'adapter_config.json'}"
        )

    print(f"base_model_path={base_model_path}")
    print(f"adapter_path={adapter_path}")
    print(f"merged_model_path={output_path}")

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model_kwargs = _build_merge_model_kwargs(config)
    model = AutoModelForVLM.from_pretrained(base_model_path, **model_kwargs)
    if config.student.train_multimodal_projector:
        conversion = dequantize_trainable_projector(
            model, config.student.multimodal_projector_path
        )
        print(f"projector_dequantization={conversion}")
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model = model.merge_and_unload()
    expected_projector_state = {
        key: value.detach().cpu().clone()
        for key, value in get_module_by_exact_path(model, config.student.multimodal_projector_path).state_dict().items()
    }

    counts = _print_merged_precision_summary(model, projector_path=config.student.multimodal_projector_path)
    if counts["remaining LoRA modules"] or counts["remaining modules_to_save wrappers"]:
        raise RuntimeError("Merged model still contains PEFT wrappers.")
    if _is_a1_projector_4bit(config):
        _validate_a1_merged_precision(model, counts, projector_path=config.student.multimodal_projector_path)
    elif config.student.merged_artifact_mode == "bf16_standalone":
        _validate_bf16_standalone_precision(
            model, counts, projector_path=config.student.multimodal_projector_path
        )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(output_path)
    # Reload the artifact to ensure the saved directory is standalone and the
    # trained merger survives serialization, rather than only validating the
    # in-memory PEFT merge.
    reload_kwargs = _build_merge_model_kwargs(config, reload=True)
    if config.student.merged_artifact_mode == "mixed_4bit_bf16":
        if _is_a1_projector_4bit(config):
            print("Standalone mixed reload:")
            print("  quantization=4bit")
            print("  excluded_from_quantization:")
            for path in A1_MAIN_MERGER_LINEAR_PATHS:
                print(f"    - {path}")
    try:
        reloaded = AutoModelForVLM.from_pretrained(str(output_path), **reload_kwargs)
    except Exception as exc:
        if _is_a1_projector_4bit(config):
            raise RuntimeError(
                "Standalone merged artifact format unsupported: Transformers could not reload "
                "the intended 4-bit base language model plus BF16 main merger."
            ) from exc
        raise
    reloaded_processor = AutoProcessor.from_pretrained(
        str(output_path), trust_remote_code=True, use_fast=False, local_files_only=True
    )
    reloaded_counts = _print_merged_precision_summary(
        reloaded, projector_path=config.student.multimodal_projector_path
    )
    if reloaded_counts["remaining LoRA modules"] or reloaded_counts["remaining modules_to_save wrappers"]:
        raise RuntimeError("Reloaded merged model still contains PEFT wrappers.")
    if _is_a1_projector_4bit(config):
        _validate_a1_merged_precision(
            reloaded, reloaded_counts, projector_path=config.student.multimodal_projector_path
        )
        metadata = getattr(getattr(reloaded, "config", None), "quantization_config", None)
        print(f"configured_student_quantization={config.student.quantization}")
        print(f"reloaded_language_model_quantized_linears={reloaded_counts['language_model quantized linears']}")
        reloaded_merger = get_module_by_exact_path(reloaded, config.student.multimodal_projector_path)
        print(
            "  merger_linear_fc1="
            f"{type(reloaded_merger.linear_fc1).__module__}.{type(reloaded_merger.linear_fc1).__name__}/"
            f"{reloaded_merger.linear_fc1.weight.dtype}"
        )
        print(
            "  merger_linear_fc2="
            f"{type(reloaded_merger.linear_fc2).__module__}.{type(reloaded_merger.linear_fc2).__name__}/"
            f"{reloaded_merger.linear_fc2.weight.dtype}"
        )
        if metadata is None:
            raise RuntimeError(
                "Standalone merged artifact format unsupported: reloaded model has no "
                "quantization metadata for its quantized language model."
            )
    elif config.student.merged_artifact_mode == "bf16_standalone":
        _validate_bf16_standalone_precision(
            reloaded, reloaded_counts, projector_path=config.student.multimodal_projector_path
        )
    reloaded_projector = get_module_by_exact_path(reloaded, config.student.multimodal_projector_path)
    max_abs_diff = 0.0
    for key, expected in expected_projector_state.items():
        reloaded_value = reloaded_projector.state_dict()[key]
        difference = (reloaded_value.float().cpu() - expected.float().cpu()).abs()
        max_abs_diff = max(max_abs_diff, float(difference.max().item()))
        try:
            torch.testing.assert_close(reloaded_value.float().cpu(), expected.float().cpu())
        except AssertionError as exc:
            raise RuntimeError(
                f"Saved/reloaded projector state mismatch at {key}; "
                f"maximum absolute difference={max_abs_diff}."
            ) from exc
    print(f"Saved/reloaded merged projector weights: exact/near-exact (max_abs_diff={max_abs_diff})")
    _validate_standalone_merged_model(
        reloaded, reloaded_processor, output_path, config=config
    )
    print(f"OK merged model written: {output_path}")
    return output_path
