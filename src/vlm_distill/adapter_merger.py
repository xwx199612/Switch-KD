"""Standalone post-merge adapter artifacts.

This module intentionally does not use the legacy merge/package stages.  In
particular, the model used for PEFT attachment is always a floating point
BF16 model; bitsandbytes is only used while reloading the saved merge.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .config_schema import PipelineConfig
from .model_loading import resolve_model_path


def _vlm_class():
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:  # pragma: no cover - older Transformers
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    return AutoModelForVLM


def _load_processor_class():
    from transformers import AutoProcessor
    return AutoProcessor


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _candidate_module_roots(model: Any) -> list[Any]:
    roots: list[Any] = []

    def add(candidate: Any) -> None:
        if candidate is None:
            return
        if any(candidate is existing for existing in roots):
            return
        roots.append(candidate)

    add(model)

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            add(get_base_model())
        except Exception:
            pass

    base_model = getattr(model, "base_model", None)
    add(base_model)
    add(getattr(base_model, "model", None))

    return roots


def _module(model: Any, path: str) -> Any:
    failures: list[str] = []

    for root in _candidate_module_roots(model):
        try:
            get_submodule = getattr(root, "get_submodule", None)
            if callable(get_submodule):
                return get_submodule(path)

            current = root
            for part in path.split("."):
                current = getattr(current, part)
            return current
        except (AttributeError, KeyError) as exc:
            failures.append(f"{type(root).__name__}: {exc}")

    raise AttributeError(
        f"Unable to resolve module path {path!r} from "
        f"{type(model).__name__}; attempted roots: {'; '.join(failures)}"
    )


def _checksum_module(module: Any) -> str:
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        # NumPy does not support torch.bfloat16 on all repository runtimes.
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _counts(model: Any) -> dict[str, int]:
    counts = {"lora": 0, "modules_to_save": 0, "linear4bit": 0}
    for name, child in model.named_modules():
        lowered = name.lower()
        if "lora_a" in lowered or "lora_b" in lowered:
            counts["lora"] += 1
        if "modules_to_save" in lowered or type(child).__name__ == "ModulesToSaveWrapper":
            counts["modules_to_save"] += 1
        if type(child).__name__ == "Linear4bit":
            counts["linear4bit"] += 1
    return counts


def _assert_before_merge(model: Any, config: PipelineConfig) -> tuple[dict[str, int], str | None]:
    counts = _counts(model)
    if not counts["lora"] and not counts["modules_to_save"]:
        raise RuntimeError("Adapter merge validation failed: PEFT adapter was not attached.")
    projector_checksum = None
    if config.student.train_multimodal_projector:
        projector = _module(model, config.student.multimodal_projector_path)
        modules_to_save = getattr(projector, "modules_to_save", {})
        active = modules_to_save["default"] if "default" in modules_to_save else None
        if active is None:
            raise RuntimeError("A1 validation failed: active projector modules_to_save.default is missing.")
        projector_checksum = _checksum_module(active)
    return counts, projector_checksum


def _assert_after_merge(model: Any, config: PipelineConfig, before_projector_checksum: str | None) -> dict[str, int]:
    counts = _counts(model)
    if counts["lora"] or counts["modules_to_save"]:
        raise RuntimeError(
            "Merged model still contains PEFT wrappers: "
            f"lora={counts['lora']} modules_to_save={counts['modules_to_save']}"
        )
    if type(model).__name__ == "PeftModel":
        raise RuntimeError("Merged model is still a PeftModel.")

    projector = _module(model, config.student.multimodal_projector_path)
    if before_projector_checksum is not None:
        after = _checksum_module(projector)
        if after != before_projector_checksum:
            raise RuntimeError(
                "A1 projector checksum mismatch: active modules_to_save.default was not preserved."
            )
    if config.student.use_projector_lora:
        for child_name in ("linear_fc1", "linear_fc2"):
            child = getattr(projector, child_name)
            import torch
            if type(child) is not torch.nn.Linear or child.weight.dtype != torch.bfloat16:
                raise RuntimeError(f"A2 validation failed: {child_name} is not a BF16 torch.nn.Linear.")
        names = [name.lower() for name, _ in model.named_modules()]
        forbidden = ("deepstack", "mlp")
        if any("lora" in name and any(token in name for token in forbidden) for name in names):
            raise RuntimeError("A2 validation failed: unexpected deepstack or LLM MLP LoRA remains.")
    return counts


def _adapter_metadata(adapter_path: Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter path is missing {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "adapter_rank": raw.get("r"),
        "adapter_alpha": raw.get("lora_alpha"),
        "target_modules": raw.get("target_modules", []),
        "source_adapter_sha256": _sha256_directory(adapter_path),
    }


def _validate_adapter_target_contract(config: PipelineConfig, adapter_path: Path) -> None:
    """Check configured LM targets without applying A2 projector restrictions to A3."""
    raw = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    actual = {str(item) for item in raw.get("target_modules", [])}
    expected = {str(item) for item in (config.student.target_modules or [])}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Adapter target contract failed: missing={missing!r}")
    if not config.student.train_multimodal_projector and not config.student.use_projector_lora:
        modules_to_save = {str(item) for item in raw.get("modules_to_save", [])}
        if modules_to_save:
            raise RuntimeError(
                "Frozen-projector adapter target contract failed: modules_to_save must be empty; "
                f"found={sorted(modules_to_save)!r}"
            )


def _write_readme(path: Path, mode: str) -> None:
    text = (
        "Standalone BF16 adapter-merger artifact.\n"
        if mode == "bf16_merged"
        else "Post-merge bitsandbytes NF4 load-time artifact. The merged BF16 source is required at runtime; quantized weights are not persisted.\n"
    )
    (path / "README.txt").write_text(text, encoding="utf-8")


def _load_kwargs(*, device_map: str, quantization_config: Any = None) -> dict[str, Any]:
    import torch
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return kwargs


def merge_adapter_artifact(
    config: PipelineConfig,
    output_dir: str | Path,
    *,
    quantization: str = "none",
    overwrite: bool = False,
    keep_bf16_intermediate: bool = False,
    max_shard_size: str = "5GB",
    device_map: str = "auto",
    smoke_test: bool = False,
) -> Path:
    """Create a standalone BF16 or post-merge bnb4 artifact."""
    if quantization not in {"none", "bnb4"}:
        raise ValueError("quantization must be one of: none, bnb4")
    import torch
    from peft import PeftModel

    base_path = Path(resolve_model_path(config.student.model_name)).resolve()
    adapter_path = Path(config.student.inference_adapter_path or config.student.adapter_dir).resolve()
    output = Path(output_dir).resolve()
    if output == base_path or output == adapter_path or base_path in output.parents or adapter_path in output.parents:
        raise ValueError("Refusing to write inside or over the source base model or adapter.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}; use --overwrite")
        shutil.rmtree(output)
    if not base_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    _validate_adapter_target_contract(config, adapter_path)

    model_class = _vlm_class()
    processor_class = _load_processor_class()
    # Deliberately no quantization_config here: this is the merge input.
    base_model = model_class.from_pretrained(str(base_path), **_load_kwargs(device_map=device_map))
    if any(p.dtype != torch.bfloat16 for p in base_model.parameters() if p.is_floating_point()):
        raise RuntimeError("Merge input model is not BF16; refusing to attach adapter.")
    processor = processor_class.from_pretrained(str(base_path), trust_remote_code=True, use_fast=False, local_files_only=True)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), local_files_only=True)
    peft_model.eval()
    before_counts, before_checksum = _assert_before_merge(peft_model, config)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    after_counts = _assert_after_merge(merged_model, config, before_checksum)
    projector_after_checksum = _checksum_module(
        _module(merged_model, config.student.multimodal_projector_path)
    )

    merged_dir = output if quantization == "none" else output / "merged_bf16"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size=max_shard_size)
    processor.save_pretrained(merged_dir)
    # The bnb artifact must keep this source; the flag is intentionally ignored.
    if quantization == "none":
        reloaded = model_class.from_pretrained(str(merged_dir), **_load_kwargs(device_map=device_map))
        _assert_after_merge(reloaded, config, None)
    else:
        del merged_model, peft_model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        quantized_model = model_class.from_pretrained(
            str(merged_dir), **_load_kwargs(device_map=device_map, quantization_config=quant_config)
        )
        qcounts = _counts(quantized_model)
        if qcounts["lora"] or qcounts["modules_to_save"] or qcounts["linear4bit"] <= 0:
            raise RuntimeError("Post-merge bnb4 validation failed.")
        print(f"Linear4bit count={qcounts['linear4bit']}")
        print(f"LoRA count={qcounts['lora']}")
        del quantized_model

    mode = "bf16_merged" if quantization == "none" else "post_merge_bnb4"
    adapter_info = _adapter_metadata(adapter_path)
    projector_mode = "projector_lora" if config.student.use_projector_lora else ("modules_to_save" if config.student.train_multimodal_projector else "base")
    configured_targets = set(config.student.target_modules or [])
    has_mlp_lora = bool(configured_targets & {"gate_proj", "up_proj", "down_proj"})
    metadata: dict[str, Any] = {
        "artifact_mode": mode, "base_model_path": str(base_path), "source_adapter_path": str(adapter_path),
        **adapter_info, "experiment_mode": (
            "attention_mlp_lora" if has_mlp_lora and not config.student.train_multimodal_projector
            else ("attention_mlp_lora_full_projector" if has_mlp_lora else
                  ("attention_projector_lora" if config.student.use_projector_lora else "attention_lora"))
        ),
        "projector_mode": projector_mode, "adapter_merged": True, "merge_input_dtype": "bfloat16",
        "experiment": "a3_r32_attn_mlp" if has_mlp_lora and config.student.lora_rank == 32 and not config.student.train_multimodal_projector and not config.student.use_projector_lora else None,
        "merge_input_quantization": "none", "quantization": "none" if quantization == "none" else "bnb_nf4_4bit",
        "quantization_stage": None if quantization == "none" else "after_merge",
        "quantized_weights_persisted": False if quantization == "bnb4" else None,
        "created_at": datetime.now(timezone.utc).isoformat(), "validation_before_merge": before_counts,
        "validation_after_merge": after_counts,
        "active_projector_before_merge_checksum": before_checksum,
        "projector_after_merge_checksum": projector_after_checksum,
    }
    (output / "adapter_merger_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if quantization == "bnb4":
        deployment = {
            "artifact_mode": mode,
            "description": "BF16 adapter merge followed by bitsandbytes NF4 load-time quantization",
            "merge_order": ["load_base_bf16", "attach_adapter", "merge_and_unload", "save_bf16_merged", "reload_bnb4"],
            "merged_model_path": "merged_bf16", "adapter_merged": True,
            "source_adapter_required_at_runtime": False, "quantization": "bnb_nf4_4bit",
            "quantization_stage": "after_merge", "quantized_weights_persisted": False,
            "standalone_bf16_source": True,
        }
        (output / "deployment_config.json").write_text(json.dumps(deployment, indent=2) + "\n", encoding="utf-8")
    _write_readme(output, mode)
    print(f"prediction_model_source=adapter_merger")
    print(f"artifact_mode={mode}")
    print("adapter_merged=true")
    print(f"quantization={'none' if quantization == 'none' else 'bnb_nf4_4bit'}")
    print("merge_input_quantization=none")
    print("merge_input_dtype=bfloat16")
    if quantization == "bnb4":
        print("quantization_stage=after_merge")
        print("quantized_weights_persisted=false")
    if smoke_test:
        print(f"smoke_test=structural_ok merged_source={merged_dir}")
    return output


def load_adapter_merger_artifact(path: str | Path, *, device_map: str = "auto"):
    """Load a merged artifact without ever attaching a PEFT adapter."""
    path = Path(path)
    metadata_path = path / "adapter_merger_config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Not an adapter-merger artifact: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    merged_model_path = metadata.get("merged_model_path")
    if metadata.get("artifact_mode") == "post_merge_bnb4" and not merged_model_path:
        deployment_path = path / "deployment_config.json"
        if deployment_path.exists():
            deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
            merged_model_path = deployment.get("merged_model_path")
    source = path / merged_model_path if metadata["artifact_mode"] == "post_merge_bnb4" else path
    if not source.exists():
        raise FileNotFoundError(f"Adapter-merger model source does not exist: {source}")
    model_class = _vlm_class()
    processor_class = _load_processor_class()
    kwargs = _load_kwargs(device_map=device_map)
    if metadata["artifact_mode"] == "post_merge_bnb4":
        import torch
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = model_class.from_pretrained(str(source), **kwargs)
    processor = processor_class.from_pretrained(str(source), trust_remote_code=True, use_fast=False, local_files_only=True)
    named_modules = model.named_modules() if hasattr(model, "named_modules") else ()
    modules = model.modules() if hasattr(model, "modules") else ()
    lora_count = sum("lora_" in name.lower() for name, _ in named_modules)
    linear4bit_count = sum(type(module).__name__ == "Linear4bit" for module in modules)
    if metadata.get("experiment"):
        print(f"experiment={metadata['experiment']}")
    print("artifact_mode=post_merge_bnb4")
    print("adapter_path=none")
    print("runtime_adapter_required=false")
    print(f"merged_model_source={source}")
    print(f"LoRA count={lora_count}")
    print(f"Linear4bit count={linear4bit_count}")
    return model, processor
