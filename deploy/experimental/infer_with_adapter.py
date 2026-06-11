from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor


def load_base_model(model_name_or_path: str):
    """Load common VLM architectures without forcing one exact HF class."""
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


def build_inputs(processor, image_path: Path, question: str):
    image = Image.open(image_path).convert("RGB")
    prompt = f"Question: {question}\nAnswer:"
    try:
        return processor(images=image, text=prompt, return_tensors="pt")
    except TypeError:
        return processor(text=prompt, images=image, return_tensors="pt")


def move_to_model_device(inputs: dict, model):
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="HF repo id or local base student model path.")
    parser.add_argument("--adapter-path", required=True, help="Local or HF Hub LoRA adapter path.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = load_base_model(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    inputs = build_inputs(processor, args.image, args.question)
    inputs = move_to_model_device(inputs, model)
    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(answer.strip())


if __name__ == "__main__":
    main()
