# Switch-KD on RTX 4060 Ti

目標是復現 Switch-KD 的核心訓練框架：

1. Visual-Switch Distillation (VSD)
2. Dynamic Bi-directional Logits Difference (DBiLD) loss
3. Standard autoregressive language modeling loss

## Paper Summary

Switch-KD 指出 VLM 蒸餾不應只把 vision/text 分開監督，而是要把多模態知識統一到 shared text-probability space。它的兩個核心元件是：

- VSD: 將 student 的 visual outputs switch 到 teacher 的 language pathway，取得跨模態的 probability reference。
- DBiLD: 對 teacher/student 的 informative probability regions 做動態、雙向 logits distribution alignment。

論文頁面：<https://arxiv.org/abs/2604.14629>

## 4060 Ti Constraints

4060 Ti 常見有 8GB 與 16GB 版本。若是 8GB，不能期待同時在線載入 7B teacher + 2B/3B student 做完整 VSD。建議：

- Teacher logits / switch logits 先離線快取。
- Student 用 0.5B-2B 級 VLM。
- Student 使用 4-bit + LoRA。
- `batch_size: 1`，靠 `gradient_accumulation_steps` 增加有效 batch。
- `max_length` 先設 384 或 512。
- freeze vision tower，只訓練 language/projector LoRA。
- DBiLD 只對 top-k logits 做，避免 full-vocab KD 記憶體爆掉。

## Code Changes Added

- `src/vlm_distill/stage_teacher_logits.py`
  - `TeacherLogitsGenerator`
  - teacher 正常 VLM forward -> cached teacher logits

- `src/vlm_distill/stage_visual_switch_logits.py`
  - `VisualSwitchDistiller`
  - student vision encoder -> student projector -> teacher LLM -> switch logits
  - `create_visual_switch_dataset`
  - component path 若指向 `get_input_embeddings` 等方法會自動呼叫

- `src/vlm_distill/logits_cache_utils.py`
  - top-k logits 快取與訓練時還原

- `src/vlm_distill/loss_switch_kd.py`
  - `SwitchKDLoss`
  - `dynamic_bidirectional_logits_difference`
  - standard causal LM loss

- `src/vlm_distill/stage_student_training.py`
  - `distillation.method: switch_kd` 時啟用 SwitchKDTrainer。
  - 支援 `teacher_logits` 與 `switch_logits` cached fields。
  - 支援 gradient checkpointing、freeze vision tower、max length。

- `configs/switch_kd_4060ti.yaml`
  - 4060 Ti 友善設定。

## Dataset Needed For Full Switch-KD

目前 core trainer 期待蒸餾資料可以包含：

```json
{
  "id": "sample-001",
  "image": "data/images/001.jpg",
  "query": "What is in the image?",
  "student_target": "a cup",
  "teacher_logits": [[[...]]],
  "switch_logits": [[[...]]]
}
```

欄位意思：

- `teacher_logits`: teacher 正常 VLM forward 的 token logits。
- `switch_logits`: VSD 產生的 logits，也就是 student visual outputs 接到 teacher language pathway 後得到的 reference logits。

若缺少 logits 欄位，訓練器會退化成標準 LM loss，並印出 warning。

## Multimodal Student Training

`train.py` 現在會對每筆樣本做完整 VLM forward：

1. 讀取 `image` 路徑並載入 RGB 影像。
2. 用 student `processor(images=..., text=prompt+target)` 編碼，保留 `pixel_values` / `image_grid_thw` 等模型所需欄位。
3. 另外編碼 `prompt`，把 prompt（含影像 token）之前的 `labels` 設為 `-100`，只在 answer 區域計算 LM loss。
4. `VlmDataCollator` 會把可堆疊的 tensor 組 batch，並保留快取的 `teacher_logits` / `switch_logits` 與其 metadata。

4060 Ti 建議維持 `batch_size: 1`；若影像解析度不一致，請先統一 resize，否則 collator 無法堆疊 `pixel_values`。

## Teacher Logits 離線產生細節 (`teacher-label`)

命令：

```powershell
python -m vlm_distill.cli teacher-label --config configs\switch_kd_4060ti.yaml
```

前置條件：

- 已執行 `label`，`outputs/switch_kd_dataset.jsonl` 內每筆樣本都有 `student_target`。
- `data/manifest.jsonl` 的 `image` 路徑在 `image_root` 下可讀。

每筆樣本內部流程（`TeacherLogitsGenerator.generate_for_sample`）：

1. 載入 teacher：`AutoProcessor` + `AutoModelForVision2Seq`，`device_map=auto`。
2. 讀取影像，組出：
   - `prompt = "Question: {query}\nAnswer:"`
   - `full_text = prompt + " " + student_target`
3. **Prompt-only forward 編碼**：`processor(images, prompt)`，記錄 `teacher_logits_prompt_len`。
4. **Full multimodal forward**：`processor(images, full_text)` → `model(**inputs)` → `outputs.logits`。
5. 以 `compact_logits` 壓成 top-k（預設 4096）寫入 JSONL：
   - `teacher_logits`
   - `teacher_logits_format`
   - `teacher_logits_prompt_len`
   - `teacher_logits_vocab_size`

訓練時還原方式：

- `materialize_cached_logits` 把 top-k 還原成 dense tensor。
- 若 `align_kd_logits_to_answer: true`，會依 `teacher_logits_prompt_len` 與 student `prompt_token_len` 對齊 answer 區段。
- 若 teacher / student 詞表大小不同且 `skip_kd_on_vocab_mismatch: true`，DBiLD 會跳過，只保留 LM loss。

若要完整 DBiLD，請使用**相同詞表**的 teacher / student（例如 Qwen2.5-VL-7B teacher + Qwen2.5-VL-3B student）。

## Student 訓練細節 (`train`)

命令：

```powershell
python -m vlm_distill.cli train --config configs\switch_kd_4060ti.yaml
```

主要設定（`configs/switch_kd_4060ti.yaml`）：

| 區塊 | 重點 |
|------|------|
| `student.quantization: 4bit` | 以 BitsAndBytes 4-bit 載入 student，搭配 LoRA |
| `student.use_lora: true` | 只訓練 `q/k/v/o_proj` 等 language adapter |
| `training.freeze_vision_tower: true` | 凍結 vision encoder |
| `training.mask_prompt_labels: true` | prompt + 影像 token 不算 LM loss |
| `distillation.method: switch_kd` | 啟用 `SwitchKDTrainer` |
| `distillation.lm/dbild/vsd_loss_weight` | 三項 loss 權重 |

單步 `compute_loss` 流程：

1. Student 多模態 forward：`model(pixel_values=..., input_ids=..., ...)`。
2. 還原並對齊 `teacher_logits`、`switch_logits` 至 student logits 形狀。
3. 以 `labels != -100` 建立 supervision mask，只在 answer token 上計算 DBiLD / VSD。
4. 合成 loss：`lm + dbild + vsd`。

輸出：

- LoRA adapter：`outputs/student_switch_kd/adapter`
- checkpoint：`outputs/student_switch_kd`

## Recommended Call Flow

```powershell
cd C:\Users\GT13-1365xt\Documents\Codex\2026-06-09\vlm-pipeline\outputs\vlm-distillation-pipeline
$env:PYTHONPATH="src"

# 1. 產生 teacher answer / distillation target
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe -m vlm_distill.cli label --config configs\switch_kd_4060ti.yaml

# 2. 離線快取 teacher logits（DBiLD 用）
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe -m vlm_distill.cli teacher-label --config configs\switch_kd_4060ti.yaml

# 3. 產生 VSD switch logits
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe -m vlm_distill.cli switch-label --config configs\switch_kd_4060ti.yaml

# 4. 使用 LM + DBiLD + VSD loss 訓練 student
C:\Users\GT13-1365xt\miniconda3\envs\vl_distill\python.exe -m vlm_distill.cli train --config configs\switch_kd_4060ti.yaml
```

## VSD Implementation Note

VSD 已新增模型無關的離線產生器，但不同 VLM 的 component path 不一定相同。若自動解析失敗，請在 config 中指定：

```yaml
distillation:
  # SmolVLM2 student
  student_vision_path: model.vision_model
  student_projector_path: model.connector
  # Qwen2.5-VL teacher
  teacher_lm_path: language_model
  teacher_token_embedding_path: get_input_embeddings
  teacher_lm_head_path: lm_head
  visual_token_placeholder: "<|image_pad|>"
```

實際 path 需要依你選定的 teacher/student 模型列印 `model.named_modules()` 後確認。
