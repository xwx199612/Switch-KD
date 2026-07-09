# Online DBiLD on RTX 4060 Ti 16GB

This repo no longer stores offline teacher logits.

Current flow:

1. `teacher-precompute` writes teacher labels only.
2. `train_online_align_dbild.py` computes teacher/student logits online during training.

## Teacher Precompute Output

Teacher precompute writes rows like:

```json
{
  "id": "sample-001",
  "image": "data/images/001.jpg",
  "task": "parsing",
  "query": "List the visible UI elements.",
  "teacher_answer": "Picture | 145,238,276,292 | false",
  "teacher_tokens": [1, 2, 3],
  "teacher_element_count": 1
}
```

No `teacher_logits` or `switch_logits` JSONL is written.

## Online Training

Use:

```bash
python -m vlm_distill.train_online_align_dbild \
  --config configs/lora_ablation/qwen3vl8b_r32_attn_mlp.yaml \
  --max-steps 1
```

The training path keeps DBiLD online:

- `teacher_outputs = teacher_model(...)`
- `teacher_logits = teacher_outputs.logits`
- `student_outputs = student_model(...)`
- `student_logits = student_outputs.logits`

`align_loss` still uses DBiLD, but logits stay in memory and are never saved to disk.
