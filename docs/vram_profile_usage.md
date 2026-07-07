# VRAM Profiling Script

`scripts/profile_vram_models.py` profiles VRAM usage for three models in sequence without keeping multiple models resident at the same time. It records PyTorch allocator memory and `nvidia-smi` GPU memory into JSONL so the results are easy to diff across runs.

## Profile Teacher, Student, and Distilled Models

```bash
python scripts/profile_vram_models.py \
  --teacher-model /home/phison/vlm_distill/models/Qwen3-VL-32B-Instruct \
  --student-model /home/phison/vlm_distill/models/Qwen3-VL-8B-Instruct \
  --distilled-model /mnt/nvme0/vlm_distill/outputs/qwen-vl-switch-kd-1080p \
  --output outputs/vram_profile.jsonl
```

You can also pass Hugging Face repo ids such as `Qwen/Qwen3-VL-32B-Instruct` instead of local paths.

The script writes records for the labels:

- `teacher_32b`
- `student_8b`
- `distilled`

Typical stages are:

- `before_model_load`
- `after_model_load`
- `before_generate`
- `after_generate`
- `after_cleanup`

`after_cleanup` is recorded only after the script clears the Python references for the loaded objects, runs `gc.collect()`, and empties the CUDA cache.

If load or generate fails, the script keeps going and writes `load_error` or `generate_error` records.

## Run the Smoke Test

The smoke test is optional. If you do not pass `--image`, the script only measures load-time VRAM.

```bash
python scripts/profile_vram_models.py \
  --teacher-model /home/phison/vlm_distill/models/Qwen3-VL-32B-Instruct \
  --student-model /home/phison/vlm_distill/models/Qwen3-VL-8B-Instruct \
  --distilled-model /mnt/nvme0/vlm_distill/outputs/qwen-vl-switch-kd-1080p \
  --image examples/images/sample_001.jpg \
  --prompt "Describe this image briefly." \
  --run-smoke-test \
  --output outputs/vram_profile_smoke.jsonl
```

Useful loading options:

- `--torch-dtype bfloat16`
- `--device-map auto`
- `--load-in-4bit`
- `--load-in-8bit`
- `--trust-remote-code`

## Interpreting Torch vs `nvidia-smi`

`torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` only cover memory tracked by the PyTorch CUDA allocator. They do not include every source of GPU memory usage.

`nvidia-smi` reports actual GPU memory usage from the driver view. This is usually the number to trust for total VRAM pressure.

In this script:

- `torch_allocated_gib` is the sum of live PyTorch allocations across visible CUDA devices.
- `torch_reserved_gib` is the sum reserved by the PyTorch allocator.
- `torch_max_allocated_gib` is the summed PyTorch peak allocation since the last reset.
- `nvidia_smi_used_mib` is the summed used memory reported by `nvidia-smi`.

If `nvidia_smi_used_mib` is much larger than `torch_allocated_gib`, the gap is usually allocator caching, non-PyTorch CUDA memory, kernels, or other processes.

## Notes

The 32B teacher may not fit on a single GPU. In practice it may require `device_map=auto`, multi-GPU placement, CPU offload, 4-bit or 8-bit quantization, or a smaller smoke-test target environment.
