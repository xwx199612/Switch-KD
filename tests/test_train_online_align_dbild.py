from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig
from vlm_distill.parsing_output_parser import serialize_parsing_label
from vlm_distill.train_online_align_dbild import (
    OnlineAlignCollator,
    OnlineAlignDataset,
    _answer_logits_request_from_labels,
    _answer_only_lm_loss,
    align_logits_to_supervised_positions,
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


def test_answer_only_forward_positions_and_lm_loss_match_full_logits_slice():
    labels = torch.tensor([[-100, -100, 2, 3, 4, -100]], dtype=torch.long)
    positions, answer_labels = _answer_logits_request_from_labels(labels, label_name="labels")

    assert positions.tolist() == [1, 2, 3]
    assert answer_labels.tolist() == [[2, 3, 4]]

    torch.manual_seed(7)
    hidden = torch.randn(1, 6, 5, requires_grad=True)
    lm_head = torch.nn.Linear(5, 11, bias=False)
    full_logits = lm_head(hidden)
    answer_logits = full_logits[:, positions, :]

    expected = torch.nn.functional.cross_entropy(
        full_logits[:, 1:4, :].reshape(-1, 11),
        answer_labels.reshape(-1),
    )
    actual = _answer_only_lm_loss(answer_logits, answer_labels)
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)

    actual.backward()
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad).item() > 0
    assert lm_head.weight.grad is not None and torch.count_nonzero(lm_head.weight.grad).item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device placement coverage")
def test_cuda_labels_are_converted_to_cpu_long_logits_indices():
    labels = torch.tensor([[-100, -100, 2, 3, 4, -100]], dtype=torch.long, device="cuda")
    positions, _ = _answer_logits_request_from_labels(labels, label_name="labels")

    assert positions.device.type == "cuda"
    positions = positions.to(device="cpu", dtype=torch.long)

    assert positions.device.type == "cpu"
    assert positions.dtype == torch.long


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device placement coverage")
def test_cpu_long_logits_indices_can_index_cuda_hidden_states():
    hidden_states = torch.randn(1, 6, 4, device="cuda")
    indices = torch.tensor([1, 2, 3], device="cpu", dtype=torch.long)

    selected = hidden_states[:, indices, :]

    assert selected.shape == (1, 3, 4)
    assert torch.equal(selected, hidden_states[:, [1, 2, 3], :])


def test_answer_only_logits_equal_full_logits_slice():
    torch.manual_seed(19)
    hidden_states = torch.randn(1, 6, 4)
    lm_head = torch.nn.Linear(4, 13, bias=True)
    indices = torch.tensor([1, 2, 3], device="cpu", dtype=torch.long)

    full_logits = lm_head(hidden_states)
    answer_only_logits = lm_head(hidden_states[:, indices, :])

    assert torch.equal(answer_only_logits, full_logits[:, indices, :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device placement coverage")
def test_cpu_logits_indices_work_when_input_and_lm_head_devices_differ():
    class SplitDeviceCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4, device="cuda")
            self.lm_head = torch.nn.Linear(4, 16, bias=False, device="cpu")

        def forward(self, input_ids, *, logits_to_keep):
            hidden_states = self.embedding(input_ids)
            selected = hidden_states[:, logits_to_keep, :]
            return self.lm_head(selected.to(self.lm_head.weight.device))

    model = SplitDeviceCausalModel()
    input_ids = torch.tensor([[5, 6, 2, 3, 4, 0]], dtype=torch.long, device="cuda")
    indices = torch.tensor([1, 2, 3], device="cpu", dtype=torch.long)

    logits = model(input_ids, logits_to_keep=indices)

    assert logits.shape == (1, 3, 16)
    assert logits.device.type == "cpu"


def test_answer_only_model_forward_returns_requested_length_and_teacher_has_no_graph():
    class TinyCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 16, bias=False)
            self.requested = None

        def forward(self, input_ids, *, logits_to_keep=0):
            self.requested = logits_to_keep.detach().clone()
            hidden = self.embedding(input_ids)
            return self.lm_head(hidden[:, logits_to_keep, :])

    labels = torch.tensor([[-100, -100, 2, 3, 4, -100]], dtype=torch.long)
    positions, answer_labels = _answer_logits_request_from_labels(labels, label_name="labels")
    teacher = TinyCausalModel()
    student = TinyCausalModel()
    input_ids = torch.tensor([[5, 6, 2, 3, 4, 0]], dtype=torch.long)

    with torch.no_grad():
        teacher_logits = teacher(input_ids, logits_to_keep=positions)
    student_logits = student(input_ids, logits_to_keep=positions)
    loss = _answer_only_lm_loss(student_logits, answer_labels)
    loss.backward()

    assert teacher_logits.shape == (1, 3, 16)
    assert student_logits.shape == (1, 3, 16)
    assert teacher.requested.tolist() == positions.tolist()
    assert student.requested.tolist() == positions.tolist()
    assert not teacher_logits.requires_grad
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in student.parameters())


def test_answer_only_alignment_preserves_compact_logits_and_length():
    torch.manual_seed(11)
    teacher_logits = torch.randn(1, 3, 9)
    student_logits = torch.randn(1, 3, 9, requires_grad=True)
    answer_labels = torch.tensor([[2, 3, 4]], dtype=torch.long)

    aligned = align_logits_to_supervised_positions(
        teacher_logits, student_logits, answer_labels, answer_labels
    )
    assert aligned[0].shape == (1, 3, 9)
    assert aligned[1].shape == (1, 3, 9)
    assert aligned[3:5] == (3, 3)
