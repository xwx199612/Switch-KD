# Production Deployment

正式部署建議先把 LoRA adapter merge 回 base model，輸出成一個完整模型資料夾。部署服務只需要載入 merged model，不需要另外處理 PEFT adapter。

## Safety Rules

- `--base-model` 指向原始 student base model，這個目錄不會被改寫。
- `--adapter-path` 保留訓練產出的 adapter，之後仍可重新 merge 到同一個 base model。
- `--output-dir` 一律使用新的 merged 目錄，例如 `outputs/student/merged-v1`。
- merge script 預設拒絕把 merged 權重寫回 base model 目錄或 adapter 目錄。
- 如果 `--output-dir` 已存在且非空，script 也會拒絕，避免不小心覆蓋既有 merged 模型。

## Step 1: Merge Adapter

```powershell
cd C:\Users\GT13-1365xt\Documents\Codex\2026-06-09\vlm-pipeline\outputs\Switch-KD

C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe deploy\production\merge_adapter.py `
  --base-model "Qwen/Qwen2.5-VL-3B-Instruct" `
  --adapter-path "outputs/student/adapter" `
  --output-dir "outputs/student/merged"
```

本地 base model：

```powershell
--base-model "D:\models\student\Qwen2.5-VL-3B-Instruct"
```

如果你要保留多個蒸餾版本，建議分開輸出：

```powershell
--output-dir "outputs/student/merged-screen-parsing-v1"
--output-dir "outputs/student/merged-grounding-v2"
```

只有在你確定要覆蓋既有 merged 輸出時，才額外加：

```powershell
--overwrite-output
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
