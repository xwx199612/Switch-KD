from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)

import torch

model_path = r"D:\Models\Qwen2.5-VL-7B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

print("Creating 4bit config...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model...")

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loaded successfully")
print(type(model))