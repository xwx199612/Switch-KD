from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image


GIB = 1024 ** 3
_TRUST_REMOTE_CODE = True


def get_nvidia_smi_memory() -> dict:
    record: dict[str, Any] = {
        "gpu_name": "",
        "gpu_count": 0,
        "nvidia_smi_used_mib": 0,
        "nvidia_smi_total_mib": 0,
        "nvidia_smi_per_gpu": [],
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        record["nvidia_smi_error"] = str(exc)
        return record

    used_total = 0
    total_total = 0
    per_gpu: list[dict[str, Any]] = []
    gpu_names: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, used_text, total_text = parts
        try:
            used_mib = int(used_text)
            total_mib = int(total_text)
        except ValueError:
            continue
        gpu_names.append(name)
        used_total += used_mib
        total_total += total_mib
        per_gpu.append(
            {
                "gpu_name": name,
                "used_mib": used_mib,
                "total_mib": total_mib,
            }
        )

    record["gpu_name"] = ", ".join(dict.fromkeys(gpu_names))
    record["gpu_count"] = len(per_gpu)
    record["nvidia_smi_used_mib"] = used_total
    record["nvidia_smi_total_mib"] = total_total
    record["nvidia_smi_per_gpu"] = per_gpu
    return record


def log_memory(label: str, model_path: str, stage: str, output_path: Path) -> None:
    torch_stats = _get_torch_cuda_memory()
    smi_stats = get_nvidia_smi_memory()
    payload = {
        "label": label,
        "model_path": model_path,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cuda_available": torch.cuda.is_available(),
        **torch_stats,
        **smi_stats,
    }
    _append_jsonl(output_path, payload)


def cleanup_model(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
            torch.cuda.synchronize(device_index)


def load_processor(model_path: str):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=args_trust_remote_code(),
        )
        return processor, getattr(processor, "tokenizer", None)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=args_trust_remote_code(),
        )
        return None, tokenizer


def load_model(model_path: str, args):
    from transformers import BitsAndBytesConfig

    model_classes: list[type] = []
    try:
        from transformers import AutoModelForImageTextToText

        model_classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass

    try:
        from transformers import AutoModelForVision2Seq

        model_classes.append(AutoModelForVision2Seq)
    except ImportError:
        pass

    from transformers import AutoModelForCausalLM

    model_classes.append(AutoModelForCausalLM)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map

    torch_dtype = _parse_torch_dtype(args.torch_dtype)
    if not args.load_in_4bit and not args.load_in_8bit:
        model_kwargs["torch_dtype"] = torch_dtype

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    last_error: Exception | None = None
    attempted_classes: list[str] = []
    for model_class in model_classes:
        attempted_classes.append(model_class.__name__)
        try:
            model = model_class.from_pretrained(model_path, **model_kwargs)
            return model
        except Exception as exc:
            last_error = exc

    attempted = ", ".join(attempted_classes)
    raise RuntimeError(
        f"Failed to load model {model_path!r} with classes: {attempted}. "
        f"Last error: {last_error}"
    )


def profile_one_model(label: str, model_path: str, args, output_path: Path) -> None:
    resolved_model_path = _normalize_model_path(model_path)
    print(f"[profile] {label}: starting {resolved_model_path}")

    _prepare_for_next_model(args.sleep_seconds)
    log_memory(label, resolved_model_path, "before_model_load", output_path)

    model = None
    processor = None
    tokenizer = None
    image = None
    generated_text = None

    try:
        processor, tokenizer = load_processor(resolved_model_path)
        model = load_model(resolved_model_path, args)
        log_memory(label, resolved_model_path, "after_model_load", output_path)

        if args.run_smoke_test and args.image is not None:
            image = Image.open(args.image).convert("RGB")
            log_memory(label, resolved_model_path, "before_generate", output_path)
            try:
                generated_text = _run_smoke_test(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    image=image,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                log_memory(label, resolved_model_path, "after_generate", output_path)
                _append_jsonl(
                    output_path,
                    {
                        "label": label,
                        "model_path": resolved_model_path,
                        "stage": "generate_result",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "generated_text": generated_text,
                    },
                )
            except Exception as exc:
                _append_jsonl(
                    output_path,
                    {
                        "label": label,
                        "model_path": resolved_model_path,
                        "stage": "generate_error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": repr(exc),
                    },
                )
    except Exception as exc:
        _append_jsonl(
            output_path,
            {
                "label": label,
                "model_path": resolved_model_path,
                "stage": "load_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            },
        )
        print(f"[profile] {label}: load failed: {exc}")
    finally:
        cleanup_model(generated_text, image, tokenizer, processor, model)
        log_memory(label, resolved_model_path, "after_cleanup", output_path)
        _prepare_for_next_model(args.sleep_seconds)
        print(f"[profile] {label}: finished {resolved_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile VRAM usage for teacher/student/distilled VLMs.")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--distilled-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--run-smoke-test", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("Choose at most one of --load-in-4bit or --load-in-8bit.")

    if args.image is not None and not args.image.exists():
        parser.error(f"Image path does not exist: {args.image}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    _set_trust_remote_code(args.trust_remote_code)

    models = [
        ("teacher_32b", args.teacher_model),
        ("student_8b", args.student_model),
        ("distilled", args.distilled_model),
    ]
    for label, model_path in models:
        profile_one_model(label, model_path, args, args.output)


def _append_jsonl(output_path: Path, payload: dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _get_torch_cuda_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "torch_allocated_gib": 0.0,
            "torch_reserved_gib": 0.0,
            "torch_max_allocated_gib": 0.0,
        }

    allocated = 0
    reserved = 0
    max_allocated = 0
    for device_index in range(torch.cuda.device_count()):
        allocated += torch.cuda.memory_allocated(device_index)
        reserved += torch.cuda.memory_reserved(device_index)
        max_allocated += torch.cuda.max_memory_allocated(device_index)

    return {
        "torch_allocated_gib": round(allocated / GIB, 4),
        "torch_reserved_gib": round(reserved / GIB, 4),
        "torch_max_allocated_gib": round(max_allocated / GIB, 4),
    }


def _prepare_for_next_model(sleep_seconds: float) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def _normalize_model_path(model_path: str) -> str:
    candidate = Path(model_path).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        if (resolved / "config.json").exists():
            return str(resolved)
        snapshots_dir = resolved / "snapshots"
        refs_main = resolved / "refs" / "main"
        if snapshots_dir.exists():
            if refs_main.exists():
                snapshot_name = refs_main.read_text(encoding="utf-8").strip()
                snapshot_path = snapshots_dir / snapshot_name
                if (snapshot_path / "config.json").exists():
                    return str(snapshot_path)
            snapshots = sorted(
                path
                for path in snapshots_dir.iterdir()
                if path.is_dir() and (path / "config.json").exists()
            )
            if len(snapshots) == 1:
                return str(snapshots[0])
        return str(resolved)
    return model_path


def _parse_torch_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported --torch-dtype={value!r}.")
    dtype = mapping[normalized]
    if dtype is None:
        raise ValueError("--torch-dtype=auto is not supported by this script.")
    return dtype


def _set_trust_remote_code(value: bool) -> None:
    global _TRUST_REMOTE_CODE
    _TRUST_REMOTE_CODE = value


def args_trust_remote_code() -> bool:
    return _TRUST_REMOTE_CODE


def _select_input_device(model) -> torch.device | None:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for location in device_map.values():
            if isinstance(location, int):
                return torch.device(f"cuda:{location}")
            if isinstance(location, str) and location.startswith("cuda"):
                return torch.device(location)

    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _move_batch_to_device(batch: Any, device: torch.device | None) -> Any:
    if device is None:
        return batch
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _run_smoke_test(
    *,
    model,
    processor,
    tokenizer,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    if processor is None:
        raise RuntimeError("Smoke test requires an AutoProcessor-compatible processor.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    model_inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )
    input_device = _select_input_device(model)
    model_inputs = _move_batch_to_device(model_inputs, input_device)

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_length = model_inputs["input_ids"].shape[-1]
    new_tokens = generated[0][prompt_length:]
    decode_tokenizer = tokenizer or getattr(processor, "tokenizer", None)
    if decode_tokenizer is None:
        raise RuntimeError("No tokenizer available to decode generated tokens.")
    return decode_tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


if __name__ == "__main__":
    main()
