from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForVLM


BASE_MODEL = Path("/mnt/nvme0/vlm_distill/models/Qwen3-VL-8B-Instruct")
ADAPTER_DIR = Path(
    "/mnt/nvme0/vlm_distill/Switch-KD/outputs/switch-kd/parsing_switch_kd_1080p_4bit_student_4bit/adapter"
)
OUTPUT_DIR = Path(
    "/mnt/nvme0/vlm_distill/Switch-KD/outputs/switch-kd/parsing_switch_kd_1080p_4bit_student_4bit/merged_hf_bf16"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading base model: {BASE_MODEL}")
model = AutoModelForVLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

print(f"Loading adapter: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(
    model,
    ADAPTER_DIR,
    local_files_only=True,
)

print("Merging LoRA adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_DIR}")
model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,
    max_shard_size="4GB",
)

print("Saving processor/tokenizer...")
processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False,
)
processor.save_pretrained(OUTPUT_DIR)

print("OK merged HF model saved.")