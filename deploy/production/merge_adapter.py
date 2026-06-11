from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoProcessor


def load_base_model(model_name_or_path: str):
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        pass

    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        pass

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="HF repo id or local base student model path.")
    parser.add_argument("--adapter-path", required=True, help="Local or HF Hub LoRA adapter path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for merged model.")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = load_base_model(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Merged model written: {args.output_dir}")


if __name__ == "__main__":
    main()
