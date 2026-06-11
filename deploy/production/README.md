# Production Deployment

正式部署建議先把 LoRA adapter merge 回 base model，輸出成一個完整模型資料夾。部署服務只需要載入 merged model，不需要另外處理 PEFT adapter。

## Step 1: Merge Adapter

```powershell
cd C:\Users\GT13-1365xt\Documents\Codex\2026-06-09\vlm-pipeline\outputs\vlm-distillation-pipeline

C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe deploy\production\merge_adapter.py `
  --base-model "Qwen/Qwen2.5-VL-3B-Instruct" `
  --adapter-path "outputs/student/adapter" `
  --output-dir "outputs/student/merged"
```

本地 base model：

```powershell
--base-model "D:\models\student\Qwen2.5-VL-3B-Instruct"
```

## Step 2: Inference With Merged Model

```powershell
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe deploy\production\infer_merged.py `
  --model-path "outputs/student/merged" `
  --image "examples/images/sample_001.jpg" `
  --question "What object is on the table?"
```

## When To Use

- adapter 已經選定，不需要頻繁切換。
- 要部署到 API server、batch inference service 或離線推論節點。
- 希望推論端只管理一個模型資料夾。
