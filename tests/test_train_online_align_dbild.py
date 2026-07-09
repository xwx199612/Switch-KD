from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.parsing_output_parser import serialize_parsing_label
from vlm_distill.train_online_align_dbild import (
    OnlineAlignCollator,
    OnlineAlignDataset,
    _target_text_for_row,
    _validate_rows,
)
from vlm_distill.vlm_batching import EncodedVlmSample


class _Processor:
    def __init__(self):
        self.tokenizer = self
        self.pad_token_id = 0
        self.eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        texts: list[str] = []
        for message in messages:
            for content in message["content"]:
                text = content.get("text")
                if text:
                    texts.append(text)
        return f"<chat>{' '.join(texts)}</chat>"

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False, **kwargs):
        del add_special_tokens, return_attention_mask, kwargs
        if isinstance(text, list):
            text = text[0]
        return {"input_ids": [len(str(text)), len(str(text)) + 1]}

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del token_ids, skip_special_tokens, clean_up_tokenization_spaces
        return ""


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            training_manifest_path=tmp_path / "manifest.jsonl",
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd", prompt_template="{query}"),
    )


def test_target_text_for_row_serializes_parsing_elements():
    row = {
        "id": "row-1",
        "task": "parsing",
        "elements": [{"text": "Home", "bbox_norm": [1, 2, 3, 4], "focused": False}],
        "coordinate_system": "normalized_0_1000",
    }

    assert _target_text_for_row(row) == serialize_parsing_label(row)


def test_target_text_for_row_rejects_missing_parsing_elements():
    with pytest.raises(ValueError, match="missing non-empty elements"):
        _target_text_for_row({"id": "row-1", "task": "parsing", "elements": []})


def test_online_align_dataset_uses_serialized_parsing_target_without_teacher_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    row = {
        "id": "row-1",
        "image": "screen.png",
        "task": "parsing",
        "query": "List UI elements",
        "elements": [{"text": "Home", "bbox_norm": [1, 2, 3, 4], "focused": False}],
        "coordinate_system": "normalized_0_1000",
    }
    target_text = serialize_parsing_label(row)

    monkeypatch.setattr("vlm_distill.train_online_align_dbild.load_training_image", lambda *args, **kwargs: object())

    def fake_encode(processor, image, prompt, target, max_length, mask_prompt_labels, canonical_answer_span):
        del processor, image, max_length, mask_prompt_labels, canonical_answer_span
        assert prompt == "List UI elements"
        assert target == target_text
        return EncodedVlmSample(
            model_inputs={
                "input_ids": torch.tensor([101, 102], dtype=torch.long),
                "labels": torch.tensor([-100, 102], dtype=torch.long),
            },
            prompt_token_len=1,
        )

    monkeypatch.setattr("vlm_distill.train_online_align_dbild.encode_vlm_training_sample", fake_encode)

    dataset = OnlineAlignDataset([row], _config(tmp_path), _Processor())
    item = dataset[0]

    assert item["target_text"] == target_text
    assert "teacher_answer" not in item


def test_online_align_collator_uses_target_text_metadata():
    collator = OnlineAlignCollator(_Processor())
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, 2], dtype=torch.long),
                "pixel_values": torch.zeros(3, 2, 2),
                "sample_id": "row-1",
                "image_path": "screen.png",
                "teacher_prompt": "prompt",
                "target_text": "target",
            }
        ]
    )

    assert batch["target_text"] == ["target"]
    assert "teacher_answer" not in batch


def test_validate_rows_accepts_parsing_rows_without_teacher_fields(tmp_path: Path):
    config = _config(tmp_path)
    config.data.distill_path.write_text(
        '{"id":"row-1","image":"screen.png","task":"parsing","query":"q","elements":[{"text":"Home","bbox_norm":[1,2,3,4],"focused":false}],"coordinate_system":"normalized_0_1000"}\n',
        encoding="utf-8",
    )

    rows = _validate_rows(config)

    assert len(rows) == 1
