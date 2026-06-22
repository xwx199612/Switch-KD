from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoProcessor


def load_base_model(model_name_or_path: str, *, device_map: str):
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception:
        pass

    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception:
        pass

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        trust_remote_code=True,
    )


def _try_resolve_local_path(path_like: str) -> Path | None:
    candidate = Path(path_like)
    if not candidate.exists():
        return None
    return candidate.resolve()


def _ensure_safe_output_dir(base_model: str, adapter_path: str, output_dir: Path, allow_overwrite: bool) -> Path:
    output_dir = output_dir.resolve()
    base_model_path = _try_resolve_local_path(base_model)
    adapter_path_resolved = _try_resolve_local_path(adapter_path)

    if base_model_path is not None and output_dir == base_model_path:
        raise ValueError("Refusing to write merged weights into the base model directory.")
    if adapter_path_resolved is not None and output_dir == adapter_path_resolved:
        raise ValueError("Refusing to write merged weights into the adapter directory.")
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise ValueError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Choose a new directory or pass --overwrite-output."
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="HF repo id or local base student model path.")
    parser.add_argument("--adapter-path", required=True, help="Local or HF Hub LoRA adapter path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for merged model.")
    parser.add_argument(
        "--device-map",
        default="cpu",
        help=(
            "Device map used while loading the base model for merge. "
            "Defaults to 'cpu' because offloaded 'auto' loading can break save_pretrained on merged Qwen2.5-VL models."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    args = parser.parse_args()

    output_dir = _ensure_safe_output_dir(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        allow_overwrite=args.overwrite_output,
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    base_model = load_base_model(args.base_model, device_map=args.device_map)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    print(f"Base model kept unchanged: {args.base_model}")
    print(f"Adapter source: {args.adapter_path}")
    print(f"Merged model written: {output_dir}")


if __name__ == "__main__":
    main()
