# Experimental Deployment

實驗階段建議不要把 LoRA adapter 合併進 base model。這樣可以快速切換不同蒸餾實驗產出的 adapter，也最能確保原始 student model 不被污染。

## Usage

```powershell
cd C:\Users\GT13-1365xt\Documents\Codex\2026-06-09\vlm-pipeline\outputs\Switch-KD

$env:PYTHONPATH="src"
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe deploy\experimental\infer_with_adapter.py `
  --base-model "Qwen/Qwen2.5-VL-3B-Instruct" `
  --adapter-path "outputs/student/adapter" `
  --image "examples/images/sample_001.jpg" `
  --question "What object is on the table?"
```

本地 base model 也一樣，把 `--base-model` 換成本地路徑：

```powershell
--base-model "D:\models\student\Qwen2.5-VL-3B-Instruct"
```

## When To Use

- 仍在比較不同 adapter。
- 需要保留 base model 不動。
- 想快速 A/B test 多個蒸餾版本。
