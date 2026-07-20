"""Package a base reference, floating PEFT adapter and processor metadata."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config_schema import PipelineConfig
from .deployment_loader import MAIN_MERGER_PATHS, projector_checksum_from_adapter_checkpoint
from .model_loading import resolve_model_path
from .student_trainability import QWEN3_VL_ATTENTION_TARGETS, QWEN3_VL_MLP_TARGETS


def package_high_fidelity_adapter_deployment(config: PipelineConfig) -> Path:
    student = config.student
    if student.merged_artifact_mode != "4bit_base_bf16_adapter":
        raise ValueError("package-adapter requires merged_artifact_mode=4bit_base_bf16_adapter")
    if student.quantization != "4bit" or not student.use_lora:
        raise ValueError("High-fidelity deployment requires quantization=4bit and use_lora=true")
    if student.copy_base_model_into_deployment:
        raise RuntimeError("Cannot copy base model: standalone bitsandbytes 4-bit serialization is unsupported")
    base_model = Path(resolve_model_path(student.model_name)).resolve()
    adapter = (student.inference_adapter_path or student.adapter_dir).resolve()
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"PEFT adapter is missing adapter_config.json: {adapter}")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_targets = adapter_config.get("target_modules", [])
    configured_targets = list(student.target_modules or adapter_targets)
    if set(adapter_targets) and set(adapter_targets) != set(configured_targets):
        raise ValueError(
            "Adapter target_modules do not match pipeline config: "
            f"adapter={sorted(adapter_targets)!r}, config={sorted(configured_targets)!r}"
        )
    unknown = set(configured_targets) - set(QWEN3_VL_ATTENTION_TARGETS) - set(QWEN3_VL_MLP_TARGETS)
    if unknown:
        raise ValueError(f"Unsupported deployment LoRA targets: {sorted(unknown)}")
    output = (student.deployment_artifact_path or student.merged_model_path or student.output_dir / "deploy_4bit_bf16_adapter").resolve()
    if output == base_model:
        raise ValueError("Refusing to package into the base model directory")
    output.mkdir(parents=True, exist_ok=True)
    destination_adapter = output / "adapter"
    if destination_adapter.exists():
        shutil.rmtree(destination_adapter)
    shutil.copytree(adapter, destination_adapter)

    processor_dir = output / "processor"
    if processor_dir.exists():
        shutil.rmtree(processor_dir)
    processor_dir.mkdir()
    # Processor/tokenizer files are small and are copied without touching model weights.
    processor_names = {"preprocessor_config.json", "processor_config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json", "chat_template.json", "chat_template.jinja", "added_tokens.json", "vocab.json", "merges.txt"}
    for source in base_model.iterdir():
        if source.is_file() and source.name in processor_names:
            shutil.copy2(source, processor_dir / source.name)
    # A directory containing a single tokenizer file is not a processor bundle.
    # Validate the copied files in isolation before advertising the path.
    from transformers import AutoProcessor
    try:
        AutoProcessor.from_pretrained(
            str(processor_dir), trust_remote_code=True, use_fast=False,
            local_files_only=True,
        )
    except Exception as exc:
        shutil.rmtree(processor_dir)
        print(f"Processor bundle omitted; base-model fallback will be used ({exc})")
    projector_mode = "projector_lora" if student.use_projector_lora else ("modules_to_save" if student.train_multimodal_projector else "base_bf16")
    excluded_from_quantization = MAIN_MERGER_PATHS if projector_mode != "base_bf16" else []
    metadata = {
        "artifact_mode": "4bit_base_bf16_adapter",
        "description": "High-fidelity quantized adapter deployment",
        "base_model_path": str(base_model),
        "adapter_path": "adapter",
        "processor_path": "processor" if processor_dir.exists() else None,
        "projector_mode": projector_mode,
        "experiment_mode": (
            "attention_mlp_lora_full_projector"
            if set(configured_targets) & set(QWEN3_VL_MLP_TARGETS) and student.train_multimodal_projector
            else ("attention_lora_full_projector" if student.train_multimodal_projector
                  else ("attention_projector_lora" if student.use_projector_lora else "attention_lora"))
        ),
        "projector_path": "model.visual.merger",
        "projector_source": "adapter" if projector_mode == "modules_to_save" else None,
        "projector_checksum": (
            projector_checksum_from_adapter_checkpoint(destination_adapter)
            if projector_mode == "modules_to_save" else None
        ),
        "base_projector_checksum_before_lora": None,
        "base_projector_dtype_map": None,
        "mixed_precision_source": "load_time_exclusion",
        "merger_norm_dtype": "torch.float32",
        "projector_lora_targets": MAIN_MERGER_PATHS if projector_mode == "projector_lora" else [],
        "lora_target_groups": {
            "attention": [target for target in QWEN3_VL_ATTENTION_TARGETS if target in configured_targets],
            "mlp": [target for target in QWEN3_VL_MLP_TARGETS if target in configured_targets],
        },
        "excluded_from_quantization": excluded_from_quantization,
        "main_merger_bf16": projector_mode != "base_bf16",
        "quantization": "4bit_nf4",
        "attn_implementation": student.attn_implementation,
        "adapter_merged": False,
        "base_model_copied": False,
    }
    adapter_metadata_path = destination_adapter / "adapter_metadata.json"
    if adapter_metadata_path.exists():
        adapter_metadata = json.loads(adapter_metadata_path.read_text(encoding="utf-8"))
        metadata.update({key: adapter_metadata.get(key) for key in (
            "base_projector_checksum_before_lora", "base_projector_dtype_map",
            "mixed_precision_source", "merger_norm_dtype",
        )})
    (output / "deployment_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "High-fidelity quantized adapter deployment\n\n"
        "This is a composition artifact: load the referenced 4-bit base with the deployment loader, "
        "then attach the BF16 PEFT adapter. The adapter is intentionally not merged.\n",
        encoding="utf-8",
    )
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    adapter_size = sum(path.stat().st_size for path in destination_adapter.rglob("*") if path.is_file())
    print("Artifact mode: 4bit_base_bf16_adapter")
    print("High-fidelity quantized adapter deployment")
    print("Base model copied: false")
    print("Adapter merged: false")
    print("Language model: NF4 4-bit")
    print("Main merger: BF16")
    print("Attention LoRA: floating, active")
    print(f"Projector mode: {projector_mode}")
    print(f"deployment bundle size: {size} bytes")
    print(f"adapter size: {adapter_size} bytes")
    return output
