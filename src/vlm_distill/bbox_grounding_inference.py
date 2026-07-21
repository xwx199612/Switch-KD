from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .deployment_loader import load_high_fidelity_adapter_deployment
from .adapter_merger import load_adapter_merger_artifact
from .model_loading import apply_attn_implementation, resolve_model_path


QUANTIZATION_CHOICES = ("none", "4bit", "8bit", "mixed_4bit_bf16")
_A1_MIXED_MERGER_PATHS = [
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
]


def build_qwen_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]


def _select_input_device(model):
    import torch
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_inputs_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()}


def _digest(value: Any) -> str:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous()
        payload = f"{value.dtype}:{tuple(value.shape)}:".encode() + value.numpy().tobytes()
    else:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _debug_value(value: Any):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_debug_value(item) for item in value]
    return value


class BBoxGroundingInferenceEngine:
    def __init__(self, model, processor, *, model_path: str = "", debug_inference_parity: bool = False):
        self.model = model
        self.processor = processor
        self.model_path = model_path
        self.debug_inference_parity = debug_inference_parity
        self.last_debug: dict[str, Any] = {}

    @classmethod
    def from_pipeline_config(cls, config):
        student = config.student
        merger_artifact = None
        for candidate in (student.deployment_artifact_path,
                          Path(student.inference_model_path) if student.inference_model_path else None):
            if candidate is None:
                continue
            candidate = Path(candidate)
            metadata_path = candidate / "adapter_merger_config.json"
            if metadata_path.exists():
                import json
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("artifact_mode") == "post_merge_bnb4":
                    merger_artifact = candidate
                    break
        if merger_artifact is not None:
            model, processor = load_adapter_merger_artifact(
                merger_artifact, device_map=getattr(student, "device_map", None) or "auto"
            )
            return cls(model, processor, model_path=str(merger_artifact),
                       debug_inference_parity=bool(getattr(getattr(config, "prediction", None), "debug_inference_parity", False)))

        deployment = None
        for candidate in (student.deployment_artifact_path, student.merged_model_path,
                          Path(student.inference_model_path) if student.inference_model_path else None):
            if candidate is not None and (Path(candidate) / "deployment_config.json").exists():
                deployment = Path(candidate)
                break
        if deployment is not None:
            model, processor = load_high_fidelity_adapter_deployment(deployment)
            return cls(model, processor, model_path=str(deployment),
                       debug_inference_parity=bool(getattr(getattr(config, "prediction", None), "debug_inference_parity", False)))

        model_path = resolve_model_path(student.inference_model_path or student.merged_model_path or student.model_name)
        return cls._load(
            model_path=model_path,
            torch_dtype=getattr(student, "torch_dtype", None) or "bfloat16",
            device_map=getattr(student, "device_map", None) or "auto",
            quantization=student.quantization,
            adapter_path=(student.inference_adapter_path or student.adapter_dir)
            if (student.load_adapter or student.merge_adapter) else None,
            merge_adapter=student.merge_adapter,
            attn_implementation=student.attn_implementation,
            debug_inference_parity=bool(getattr(getattr(config, "prediction", None), "debug_inference_parity", False)),
        )

    @classmethod
    def from_cli_args(cls, *, model_path, torch_dtype="bfloat16", device_map="auto",
                      quantization="none", adapter_path=None, merge_adapter=False,
                      attn_implementation="sdpa", debug_inference_parity=False):
        path = Path(model_path)
        if (path / "deployment_config.json").exists():
            model, processor = load_high_fidelity_adapter_deployment(path)
            return cls(model, processor, model_path=str(path), debug_inference_parity=debug_inference_parity)
        return cls._load(model_path=resolve_model_path(str(model_path)), torch_dtype=torch_dtype,
                         device_map=device_map, quantization=quantization, adapter_path=adapter_path,
                         merge_adapter=merge_adapter, attn_implementation=attn_implementation,
                         debug_inference_parity=debug_inference_parity)

    @classmethod
    def _load(cls, *, model_path, torch_dtype, device_map, quantization, adapter_path,
              merge_adapter, attn_implementation, debug_inference_parity):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import AutoModelForImageTextToText as AutoModel
        except ImportError:  # pragma: no cover
            from transformers import AutoModelForVision2Seq as AutoModel
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
        kwargs = {"trust_remote_code": True, "local_files_only": True, "device_map": device_map,
                  "torch_dtype": dtype}
        apply_attn_implementation(kwargs, attn_implementation)
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
        elif quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        elif quantization == "mixed_4bit_bf16":
            from .mixed_precision import build_mixed_precision_quantization_config
            kwargs["quantization_config"] = build_mixed_precision_quantization_config(
                quantization="4bit", excluded_module_paths=_A1_MIXED_MERGER_PATHS)
            kwargs["device_map"] = "auto"
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)
        model = AutoModel.from_pretrained(model_path, **kwargs)
        if adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
            if merge_adapter:
                model = model.merge_and_unload()
        model.eval()
        return cls(model, processor, model_path=str(model_path), debug_inference_parity=debug_inference_parity)

    def generate_raw(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = build_qwen_messages(image, prompt)
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) \
            if hasattr(self.processor, "apply_chat_template") else prompt
        images = videos = None
        try:
            from qwen_vl_utils import process_vision_info
            images, videos = process_vision_info(messages)
        except ImportError:
            images = [image]
        kwargs = {"text": [chat_text], "images": images, "return_tensors": "pt"}
        if videos is not None:
            kwargs["videos"] = videos
        try:
            inputs = self.processor(**kwargs)
        except TypeError:
            kwargs.pop("videos", None)
            inputs = self.processor(images=kwargs["images"], text=kwargs["text"], return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, _select_input_device(self.model))
        generation_kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}
        output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_ids = inputs.get("input_ids")
        generated_ids = output_ids[:, input_ids.shape[1]:] if input_ids is not None else output_ids
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        raw = decoded[0].strip() if decoded else ""
        if self.debug_inference_parity:
            self.last_debug = {"prompt_sha256": _digest(prompt), "chat_text_sha256": _digest(chat_text),
                "input_ids_shape": list(input_ids.shape) if input_ids is not None else None,
                "input_ids_hash": _digest(input_ids) if input_ids is not None else None,
                "pixel_values_shape": list(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
                "pixel_values_dtype": str(inputs["pixel_values"].dtype) if "pixel_values" in inputs else None,
                "pixel_values_hash": _digest(inputs["pixel_values"]) if "pixel_values" in inputs else None,
                "image_grid_thw": _debug_value(inputs.get("image_grid_thw")), "generation_kwargs": generation_kwargs,
                "generated_token_ids_hash": _digest(generated_ids), "raw_output_hash": _digest(raw)}
        return raw
