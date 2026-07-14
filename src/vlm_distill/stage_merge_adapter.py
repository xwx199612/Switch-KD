from __future__ import annotations

from pathlib import Path

from .config_schema import PipelineConfig
from .model_loading import resolve_model_path
from .student_trainability import dequantize_trainable_projector, get_module_by_exact_path


def _print_merged_precision_summary(model) -> None:
    import torch
    counts = {
        "language_model quantized linears": 0,
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
            elif "visual.merger" not in name:
                counts["visual encoder quantized linears"] += 1
        if name.startswith("model.visual.merger") and type(module).__name__ == "Linear":
            if getattr(module, "weight", None) is not None and module.weight.dtype == torch.bfloat16:
                counts["main merger BF16 linears"] += 1
    print("Merged model precision/module summary:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    if counts["remaining LoRA modules"] or counts["remaining modules_to_save wrappers"]:
        raise RuntimeError("Merged model still contains PEFT wrappers.")


def _validate_standalone_merged_model(model, processor, output_path: Path) -> None:
    """Smoke-test the saved standalone model when the repository sample exists."""
    import torch
    from PIL import Image

    if not Path("examples/images/sample_001.jpg").exists():
        print("Standalone merged inference validation skipped: sample image is unavailable.")
        return
    image_path = Path("examples/images/sample_001.jpg")
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
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
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": True,
        "attn_implementation": config.student.attn_implementation,
    }
    if config.student.quantization == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.student.quantization == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForVLM.from_pretrained(base_model_path, **model_kwargs)
    if config.student.train_multimodal_projector:
        conversion = dequantize_trainable_projector(
            model, config.student.multimodal_projector_path
        )
        print(f"projector_dequantization={conversion}")
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model = model.merge_and_unload()
    merged_projector_state = {
        key: value.detach().cpu().clone()
        for key, value in get_module_by_exact_path(model, config.student.multimodal_projector_path).state_dict().items()
    }

    _print_merged_precision_summary(model)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(output_path)
    # Reload the artifact to ensure the saved directory is standalone and the
    # trained merger survives serialization, rather than only validating the
    # in-memory PEFT merge.
    reloaded = AutoModelForVLM.from_pretrained(
        str(output_path),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=config.student.attn_implementation,
    )
    reloaded_processor = AutoProcessor.from_pretrained(
        str(output_path), trust_remote_code=True, use_fast=False, local_files_only=True
    )
    _print_merged_precision_summary(reloaded)
    reloaded_projector = get_module_by_exact_path(reloaded, config.student.multimodal_projector_path)
    for key, expected in merged_projector_state.items():
        torch.testing.assert_close(reloaded_projector.state_dict()[key].float().cpu(), expected.float())
    print("Saved/reloaded merged projector weights: exact/near-exact")
    _validate_standalone_merged_model(reloaded, reloaded_processor, output_path)
    print(f"OK merged model written: {output_path}")
    return output_path
