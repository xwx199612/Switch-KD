from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
import pytest
import yaml

from vlm_distill.config_schema import DataConfig, DistillationConfig, PipelineConfig, StudentConfig, TeacherConfig, load_config
from vlm_distill.data_manifest import VlmSample
from vlm_distill.stage_visual_switch_logits import (
    VisualSwitchDistiller,
    _load_switch_base_rows,
    _infer_module_input_dim,
    _validate_switch_logits_row,
    extract_student_vision_hidden_states,
    get_teacher_visual_projector_or_merger,
)


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "distill.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher", backend="mock"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd", prompt_template="Query: {query}"),
    )


class _FakeProjector(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.last_input_device = None
        self.last_input_dtype = None

    def forward(self, x=None, hidden_states=None, inputs_embeds=None):
        tensor = x if x is not None else hidden_states if hidden_states is not None else inputs_embeds
        if tensor is None:
            raise ValueError("expected tensor input")
        self.last_input_device = tensor.device
        self.last_input_dtype = tensor.dtype
        return self.linear(tensor)


class _FakeVisualTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([])

    def forward(self, *args, **kwargs):
        return torch.zeros(1, 1, 8)


class _FakeQwenVisual(torch.nn.Module):
    def __init__(self, *, return_last_hidden_state: bool = True):
        super().__init__()
        self.patch_embed = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([])
        self.merger = _FakeProjector(8, 12)
        self._return_last_hidden_state = return_last_hidden_state

    def forward(self, hidden_states=None, grid_thw=None, **kwargs):
        if self._return_last_hidden_state:
            return SimpleNamespace(
                last_hidden_state=torch.full((4, 8), 3.0),
                pooler_output=torch.full((1, 12), 9.0),
            )
        return torch.full((1, 12), 9.0)


class _FakeQwenMerger(torch.nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.ln_q = torch.nn.LayerNorm(normalized_shape)


class _FakeOpaqueMerger(torch.nn.Module):
    pass


class _FakeTeacherConfig(SimpleNamespace):
    pass


def test_paper_mode_config_loads_correctly():
    config = load_config(Path("configs/parsing_switch_kd.yaml"))

    assert config.distillation.switch_kd.enabled is True
    assert config.distillation.switch_kd.visual_switch.mode == "paper"
    assert config.distillation.switch_kd.visual_switch.teacher_projector == "native"
    assert config.distillation.switch_kd.visual_switch.allow_fallback_adapter is False


def test_switch_logits_are_answer_only(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    row = distiller.generate_for_sample(
        VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world"),
        base_row={"teacher_tokens": [10, 11, 12], "visual_token_count": 7},
    )

    assert row["switch_logits_prompt_len"] == 0
    assert row["switch_logits_aligned_to_answer"] is True
    assert row["switch_logits_token_identity_match"] is True
    assert row["switch_logits_answer_token_ids"] == row["teacher_tokens"]
    assert row["switch_logits"]["shape"][1] == len(row["teacher_tokens"])


def test_switch_logits_generation_enforces_teacher_token_identity_match(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    sample = VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world")

    def _teacher_text_inputs(text: str):
        token_ids = [101, 102, 103] if text == "prompt" else [101, 102, 103, 10, 11]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    def _splice_visual_embeds(*, teacher_inputs, projected_visual):
        seq_len = int(teacher_inputs["input_ids"].shape[1])
        return torch.zeros(1, seq_len, 4), torch.ones(1, seq_len, dtype=torch.long)

    distiller._teacher_text_inputs = _teacher_text_inputs
    distiller._splice_visual_embeds = _splice_visual_embeds
    distiller._apply_visual_switch_projection = lambda student_visual, student_inputs: student_visual
    distiller._teacher_lm_forward = lambda *, inputs_embeds, attention_mask: torch.zeros(1, 5, 8)

    row = distiller._generate_for_sample_from_student_visual(
        sample=sample,
        prompt="prompt",
        student_visual=torch.zeros(1, 2, 4),
        base_row={"teacher_tokens": [10, 11], "teacher_answer": "{}"},
        student_inputs={},
    )

    assert row["switch_logits_answer_token_ids"] == [10, 11]
    assert row["switch_logits_token_identity_match"] is True
    assert row["switch_logits_debug"]["token_identity_validation_passed"] is True


def test_switch_logits_generation_fails_on_same_length_wrong_token_ids(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    sample = VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world")

    def _teacher_text_inputs(text: str):
        token_ids = [101, 102, 103] if text == "prompt" else [101, 102, 103, 10, 99]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    def _splice_visual_embeds(*, teacher_inputs, projected_visual):
        seq_len = int(teacher_inputs["input_ids"].shape[1])
        return torch.zeros(1, seq_len, 4), torch.ones(1, seq_len, dtype=torch.long)

    distiller._teacher_text_inputs = _teacher_text_inputs
    distiller._splice_visual_embeds = _splice_visual_embeds
    distiller._apply_visual_switch_projection = lambda student_visual, student_inputs: student_visual
    distiller._teacher_lm_forward = lambda *, inputs_embeds, attention_mask: torch.zeros(1, 5, 8)

    with pytest.raises(ValueError, match="Switch logits token identity mismatch"):
        distiller._generate_for_sample_from_student_visual(
            sample=sample,
            prompt="prompt",
            student_visual=torch.zeros(1, 2, 4),
            base_row={"teacher_tokens": [10, 11], "teacher_answer": "{}"},
            student_inputs={},
        )


def test_switch_logits_generation_slices_in_embedding_space_after_visual_splice(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    sample = VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world")

    def _teacher_text_inputs(text: str):
        token_ids = [101, 102, 103] if text == "prompt" else [101, 102, 103, 10, 11]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    def _splice_visual_embeds(*, teacher_inputs, projected_visual):
        seq_len = int(teacher_inputs["input_ids"].shape[1])
        embed_len = seq_len + 4
        return torch.zeros(1, embed_len, 4), torch.ones(1, embed_len, dtype=torch.long)

    def _teacher_lm_forward(*, inputs_embeds, attention_mask):
        seq_len = int(inputs_embeds.shape[1])
        logits = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
        return logits

    distiller._teacher_text_inputs = _teacher_text_inputs
    distiller._splice_visual_embeds = _splice_visual_embeds
    distiller._apply_visual_switch_projection = lambda student_visual, student_inputs: student_visual
    distiller._teacher_lm_forward = _teacher_lm_forward

    row = distiller._generate_for_sample_from_student_visual(
        sample=sample,
        prompt="prompt",
        student_visual=torch.zeros(1, 2, 4),
        base_row={"teacher_tokens": [10, 11], "teacher_answer": "{}"},
        student_inputs={},
    )

    assert row["switch_logits"]["shape"] == [1, 2, 1]
    assert row["switch_logits"]["values"][0][0][0] == 6.0
    assert row["switch_logits"]["values"][0][1][0] == 7.0
    assert row["switch_logits_debug"] == {
        "prompt_input_len": 3,
        "prompt_embed_len": 7,
        "full_input_len": 5,
        "full_embed_len": 9,
        "visual_extra_prompt": 4,
        "visual_extra_full": 4,
        "answer_start_logit_index": 6,
        "teacher_tokens_len": 2,
        "switch_logits_answer_token_ids_len": 2,
        "token_identity_validation_passed": True,
    }


def test_switch_logits_generation_raises_on_visual_splice_length_mismatch(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    sample = VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world")

    def _teacher_text_inputs(text: str):
        token_ids = [101, 102, 103] if text == "prompt" else [101, 102, 103, 10, 11]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    def _splice_visual_embeds(*, teacher_inputs, projected_visual):
        seq_len = int(teacher_inputs["input_ids"].shape[1])
        embed_len = seq_len + (4 if seq_len == 3 else 5)
        return torch.zeros(1, embed_len, 4), torch.ones(1, embed_len, dtype=torch.long)

    distiller._teacher_text_inputs = _teacher_text_inputs
    distiller._splice_visual_embeds = _splice_visual_embeds
    distiller._apply_visual_switch_projection = lambda student_visual, student_inputs: student_visual
    distiller._teacher_lm_forward = lambda *, inputs_embeds, attention_mask: torch.zeros(1, 9, 8)

    with pytest.raises(ValueError, match="Switch logits visual splice length mismatch"):
        distiller._generate_for_sample_from_student_visual(
            sample=sample,
            prompt="prompt",
            student_visual=torch.zeros(1, 2, 4),
            base_row={"teacher_tokens": [10, 11], "teacher_answer": "{}"},
            student_inputs={},
        )


def test_switch_logits_generation_keeps_token_identity_validation_in_input_id_space(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    sample = VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world")

    def _teacher_text_inputs(text: str):
        token_ids = [101, 102, 103] if text == "prompt" else [101, 102, 103, 10, 99]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    def _splice_visual_embeds(*, teacher_inputs, projected_visual):
        seq_len = int(teacher_inputs["input_ids"].shape[1])
        embed_len = seq_len + 4
        return torch.zeros(1, embed_len, 4), torch.ones(1, embed_len, dtype=torch.long)

    distiller._teacher_text_inputs = _teacher_text_inputs
    distiller._splice_visual_embeds = _splice_visual_embeds
    distiller._apply_visual_switch_projection = lambda student_visual, student_inputs: student_visual
    distiller._teacher_lm_forward = lambda *, inputs_embeds, attention_mask: torch.zeros(1, 9, 8)

    with pytest.raises(ValueError, match="Switch logits token identity mismatch"):
        distiller._generate_for_sample_from_student_visual(
            sample=sample,
            prompt="prompt",
            student_visual=torch.zeros(1, 2, 4),
            base_row={"teacher_tokens": [10, 11], "teacher_answer": "{}"},
            student_inputs={},
        )


def test_paper_mode_raises_on_shape_incompatibility(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    distiller._teacher_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=12),
        model=SimpleNamespace(visual=_FakeProjector(8, 12)),
    )

    with pytest.raises(ValueError, match="Switch-KD paper path incompatible"):
        distiller._paper_path_projection(torch.zeros(1, 2, 7))


def test_teacher_projector_resolution_prefers_qwen_visual_merger():
    teacher_model = SimpleNamespace(
        visual=_FakeVisualTower(),
        model=SimpleNamespace(visual=SimpleNamespace(merger=_FakeProjector(8, 12))),
    )

    projector = get_teacher_visual_projector_or_merger(teacher_model)

    assert projector is teacher_model.model.visual.merger


def test_infer_module_input_dim_uses_qwen_merger_ln_q_normalized_shape():
    fake_merger = _FakeQwenMerger(1280)
    fake_teacher = SimpleNamespace(config=_FakeTeacherConfig(hidden_size=3584))

    inferred = _infer_module_input_dim(
        fake_merger,
        model=fake_teacher,
        module_label="teacher projector/merger",
    )

    assert inferred == 1280


def test_infer_module_input_dim_prefers_vision_hidden_size_over_text_hidden_size():
    fake_merger = _FakeOpaqueMerger()
    fake_teacher = SimpleNamespace(
        config=_FakeTeacherConfig(
            hidden_size=3584,
            vision_config=SimpleNamespace(hidden_size=1280),
        )
    )

    inferred = _infer_module_input_dim(
        fake_merger,
        model=fake_teacher,
        module_label="teacher projector/merger",
    )

    assert inferred == 1280


def test_paper_mode_rejects_full_teacher_visual_tower(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    distiller._teacher_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=12),
        visual=_FakeVisualTower(),
    )

    with pytest.raises(ValueError, match="full teacher visual tower"):
        distiller._paper_path_projection(torch.zeros(1, 2, 8))


def test_paper_mode_moves_cached_student_visual_to_teacher_projector_device_and_dtype(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    projector = _FakeProjector(8, 12)
    if torch.cuda.is_available():
        expected_device = torch.device("cuda:0")
        expected_dtype = torch.float32
        projector = projector.to(device=expected_device, dtype=expected_dtype)
    else:
        expected_device = torch.device("cpu")
        expected_dtype = torch.bfloat16
        projector = projector.to(dtype=expected_dtype)
    distiller._teacher_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=12),
        model=SimpleNamespace(visual=projector),
    )

    student_visual = torch.zeros(1, 2, 8, device="cpu", dtype=torch.float32)

    projected = distiller._paper_path_projection(student_visual)

    assert tuple(projected.shape) == (1, 2, 12)
    assert projector.last_input_device == expected_device
    assert projector.last_input_dtype == expected_dtype


def test_adapter_mode_requires_allow_fallback_adapter(tmp_path: Path):
    config_path = tmp_path / "bad_adapter.yaml"
    config_path.write_text(
        """
data:
  manifest_path: manifest.jsonl
  distill_path: distill.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: out
  adapter_dir: adapter
distillation:
  method: switch_kd
  switch_kd:
    enabled: true
    visual_switch:
      mode: adapter_to_teacher_projector
      teacher_projector: native
      allow_fallback_adapter: false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allow_fallback_adapter"):
        load_config(config_path)


def test_adapter_mode_logs_project_specific_variant(capsys, tmp_path: Path):
    config = _make_config(tmp_path)
    config.distillation.switch_kd.visual_switch.mode = "adapter_to_teacher_projector"
    config.distillation.switch_kd.visual_switch.allow_fallback_adapter = True
    config.distillation.switch_kd.visual_switch.adapter_path = "connector"
    distiller = VisualSwitchDistiller(config)
    distiller._student_model = SimpleNamespace(connector=_FakeProjector(8, 12))
    distiller._teacher_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=12),
        model=SimpleNamespace(visual=_FakeProjector(8, 12)),
    )

    component = distiller._visual_switch_projector_component()
    out = capsys.readouterr().out

    assert component is not None
    assert "This is a project-specific Switch-KD variant, not the original paper path." in out


def test_adapter_to_teacher_lm_uses_student_projector_output(tmp_path: Path):
    config = _make_config(tmp_path)
    config.distillation.switch_kd.visual_switch.mode = "adapter_to_teacher_lm"
    config.distillation.switch_kd.visual_switch.allow_fallback_adapter = True
    config.distillation.switch_kd.visual_switch.adapter_path = "connector"
    distiller = VisualSwitchDistiller(config)
    distiller._student_model = SimpleNamespace(
        connector=_FakeProjector(12, 12),
        model=SimpleNamespace(connector=_FakeProjector(12, 12)),
    )
    distiller._teacher_model = SimpleNamespace(
        connector=_FakeProjector(12, 12),
        config=SimpleNamespace(hidden_size=12),
        model=SimpleNamespace(visual=SimpleNamespace(merger=_FakeProjector(99, 12))),
    )

    projected = distiller._adapter_to_teacher_lm_projection(
        torch.zeros(1, 2, 12),
        student_inputs={},
    )

    assert tuple(projected.shape) == (1, 2, 12)


def test_extract_student_vision_hidden_states_uses_qwen_last_hidden_state():
    student_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen2_5_vl"),
        visual=_FakeQwenVisual(return_last_hidden_state=True),
    )

    hidden_states = extract_student_vision_hidden_states(
        student_model,
        student_processor=None,
        student_inputs={
            "pixel_values": torch.zeros(4, 8),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        },
    )

    assert tuple(hidden_states.shape) == (4, 8)
    assert torch.all(hidden_states == 3.0)


def test_extract_student_vision_hidden_states_raises_when_qwen_raw_states_unavailable():
    student_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen2_5_vl"),
        visual=_FakeQwenVisual(return_last_hidden_state=False),
    )

    with pytest.raises(ValueError, match="pre-merger last_hidden_state"):
        extract_student_vision_hidden_states(
            student_model,
            student_processor=None,
            student_inputs={
                "pixel_values": torch.zeros(4, 8),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            },
        )


def test_switch_logits_old_text_only_prompt_len_raises():
    row = {
        "id": "bad",
        "image": "screen.jpg",
        "teacher_tokens": list(range(477)),
        "switch_logits_prompt_len": 287,
        "switch_logits_aligned_to_answer": True,
        "switch_logits_token_identity_match": True,
        "switch_logits_answer_token_ids": list(range(477)),
        "switch_logits": {
            "indices": [[[0]]],
            "values": [[[1.0]]],
            "shape": [1, 2327, 152064],
            "vocab_size": 152064,
        },
    }

    with pytest.raises(ValueError, match="answer-only alignment"):
        _validate_switch_logits_row(
            row,
            field_name="switch_logits",
            visual_token_placeholder="<image>",
        )


def test_switch_logits_row_contains_valid_compact_payload(tmp_path: Path):
    config = _make_config(tmp_path)
    distiller = VisualSwitchDistiller(config)
    row = distiller.generate_for_sample(
        VlmSample(id="sample-1", image="screen.jpg", task="parsing", query="hello world"),
        base_row={"teacher_tokens": [1, 2], "visual_token_count": 3},
    )

    _validate_switch_logits_row(
        row,
        field_name="switch_logits",
        visual_token_placeholder="<image>",
    )
    assert {"indices", "values", "vocab_size"}.issubset(row["switch_logits"])
    assert row["switch_logits_token_identity_match"] is True
    assert row["switch_logits_answer_token_ids"] == row["teacher_tokens"]
    assert row["switch_logits"]["shape"][1] == len(row["teacher_tokens"])


def test_validate_switch_logits_row_fails_when_token_identity_flag_missing():
    row = {
        "id": "bad",
        "image": "screen.jpg",
        "teacher_tokens": [1, 2],
        "switch_logits_aligned_to_answer": True,
        "switch_logits_answer_token_ids": [1, 2],
        "switch_logits": {
            "indices": [[[0], [0]]],
            "values": [[[1.0], [1.0]]],
            "shape": [1, 2, 8],
            "vocab_size": 8,
        },
    }

    with pytest.raises(ValueError, match="switch_logits_token_identity_match"):
        _validate_switch_logits_row(
            row,
            field_name="switch_logits",
            visual_token_placeholder="<image>",
        )


def test_validate_switch_logits_row_fails_when_answer_token_ids_differ():
    row = {
        "id": "bad",
        "image": "screen.jpg",
        "teacher_tokens": [1, 2],
        "switch_logits_aligned_to_answer": True,
        "switch_logits_token_identity_match": True,
        "switch_logits_answer_token_ids": [1, 3],
        "switch_logits": {
            "indices": [[[0], [0]]],
            "values": [[[1.0], [1.0]]],
            "shape": [1, 2, 8],
            "vocab_size": 8,
        },
    }

    with pytest.raises(ValueError, match="Switch logits token identity mismatch"):
        _validate_switch_logits_row(
            row,
            field_name="switch_logits",
            visual_token_placeholder="<image>",
        )


def test_switch_logits_reads_teacher_rows_from_label_path(tmp_path: Path):
    label_path = tmp_path / "labels.jsonl"
    switch_path = tmp_path / "switch_logits.jsonl"
    label_path.write_text('{"id":"label-row","teacher_answer":"{}","teacher_tokens":[1]}\n', encoding="utf-8")
    switch_path.write_text('{"id":"switch-row","teacher_answer":"{}","teacher_tokens":[2]}\n', encoding="utf-8")
    config = _make_config(tmp_path)
    config.data.label_path = label_path
    config.data.switch_logits_path = switch_path
    config.data.teacher_logits_path = tmp_path / "legacy_teacher_logits.jsonl"

    rows = _load_switch_base_rows(config)

    assert [row["id"] for row in rows] == ["label-row"]


def test_switch_logits_does_not_use_switch_logits_as_teacher_base_when_label_missing(tmp_path: Path):
    switch_path = tmp_path / "switch_logits.jsonl"
    switch_path.write_text('{"id":"switch-row","teacher_answer":"{}","teacher_tokens":[2]}\n', encoding="utf-8")
    config = _make_config(tmp_path)
    config.data.label_path = tmp_path / "missing_labels.jsonl"
    config.data.switch_logits_path = switch_path

    assert _load_switch_base_rows(config) == []


def test_hf_vlm_has_single_distillation_block_and_preserves_visual_switch():
    config_path = Path("configs/hf_vlm.yaml")
    text = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)

    assert text.count("\ndistillation:") == 1
    assert parsed["distillation"]["switch_kd"]["visual_switch"]["mode"] == "paper"
    assert parsed["distillation"]["switch_kd"]["visual_switch"]["teacher_projector"] == "native"
    assert parsed["distillation"]["switch_kd"]["visual_switch"]["allow_fallback_adapter"] is False
