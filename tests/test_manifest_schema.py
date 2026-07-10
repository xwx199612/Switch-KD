from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vlm_distill.config_schema import (
    DataConfig,
    DistillationConfig,
    PipelineConfig,
    StudentConfig,
    TeacherConfig,
    load_config,
    resolve_label_path,
    resolve_prediction_path,
    resolve_switch_logits_path,
    resolve_teacher_logits_path,
)
from vlm_distill.data_manifest import VlmSample, validate_manifest
from vlm_distill.manifest_builder import infer_manifest_task_from_config_path
from vlm_distill.stage_teacher_precompute import _load_completed_ids, create_teacher_precompute_dataset


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_parsing_manifest_validates_without_question(tmp_path: Path):
    image_root = tmp_path / "images"
    _make_image(image_root / "screen.jpg")
    manifest = tmp_path / "screen.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "screen-1",
                "image": "screen.jpg",
                "task": "parsing",
                "query": "List all visible UI elements.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = validate_manifest(manifest, image_root=image_root)
    assert len(samples) == 1
    assert samples[0].query == "List all visible UI elements."
    assert samples[0].target_label is None


def test_load_config_accepts_legacy_target_field(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
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
  target_field: student_target
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.distillation.prompt_template == "Query: {query}\nAnswer:"
    assert config.teacher.retry_on_invalid_parsing_json is False


def test_load_config_interpolates_response_options(tmp_path: Path):
    config_path = tmp_path / "response.yaml"
    config_path.write_text(
        """
options:
  task_name: parsing
  quality: 480p
  teacher_quantization: 8bit
  student_quantization: 4bit
data:
  manifest_path: D:/TV_data/teacher_parsing/{task_name}_manifest.jsonl
  distill_path: D:/TV_data/teacher_parsing/{task_name}_teacher_labels_{label_profile}.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: outputs/{task_name}_response_{response_profile}
  adapter_dir: outputs/{task_name}_response_{response_profile}/adapter
  quantization: "{student_quantization}"
distillation:
  method: response
  prompt_template: "query: {query}\\nAnswer:"
evaluation:
  output_path: outputs/{task_name}_response_{response_profile}/eval_report.json
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.data.distill_path.as_posix() == "D:/TV_data/teacher_parsing/parsing_teacher_labels_480p_8bit.jsonl"
    assert config.student.output_dir.as_posix() == "outputs/parsing_response_480p_8bit_student_4bit"
    assert config.student.quantization == "4bit"
    assert config.distillation.prompt_template == "query: {query}\nAnswer:"


def test_load_config_interpolates_split_distillation_paths(tmp_path: Path):
    config_path = tmp_path / "switch.yaml"
    config_path.write_text(
        """
options:
  task_name: parsing
  quality: 480p
  teacher_quantization: 8bit
  student_quantization: 4bit
data:
  manifest_path: D:/TV_data/teacher_parsing/{task_name}_manifest.jsonl
  distill_path: outputs/{task_name}_switch_kd_{response_profile}.jsonl
  label_path: D:/TV_data/teacher_parsing/{task_name}_teacher_labels_{label_profile}.jsonl
  teacher_logits_path: outputs/{task_name}_teacher_logits_{label_profile}.jsonl
  switch_logits_path: outputs/{task_name}_switch_logits_{response_profile}.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: outputs/out
  adapter_dir: outputs/adapter
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert resolve_label_path(config.data).as_posix() == "D:/TV_data/teacher_parsing/parsing_teacher_labels_480p_8bit.jsonl"
    assert resolve_teacher_logits_path(config.data).as_posix() == "outputs/parsing_teacher_logits_480p_8bit.jsonl"
    assert resolve_switch_logits_path(config.data).as_posix() == "outputs/parsing_switch_logits_480p_8bit_student_4bit.jsonl"


def test_load_config_interpolates_prediction_path(tmp_path: Path):
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(
        """
options:
  task_name: parsing
  quality: 480p
  teacher_quantization: 8bit
data:
  manifest_path: D:/TV_data/teacher_parsing/{task_name}_manifest.jsonl
  distill_path: D:/TV_data/teacher_parsing/{task_name}_teacher_labels_{label_profile}.jsonl
  prediction_path: outputs/{task_name}_merged_predictions_{label_profile}.jsonl
teacher:
  model_name: mock-teacher
student:
  model_name: mock-student
  output_dir: outputs/out
  adapter_dir: outputs/adapter
  inference_model_path: outputs/student/merged
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert resolve_prediction_path(config.data).as_posix() == "outputs/parsing_merged_predictions_480p_8bit.jsonl"


def test_infer_manifest_task_from_config_path_uses_filename():
    assert infer_manifest_task_from_config_path(Path("configs/parsing_switch_kd.yaml")) == "parsing"


def test_load_completed_ids_reads_existing_ids(tmp_path: Path):
    output_path = tmp_path / "labels.jsonl"
    output_path.write_text('{"id":"row-1"}\n{"id":"row-2"}\n', encoding="utf-8")
    assert _load_completed_ids(output_path) == {"row-1", "row-2"}


def test_create_teacher_precompute_dataset_writes_elements_only_rows(tmp_path: Path):
    config = PipelineConfig(
        data=DataConfig(
            training_manifest_path=tmp_path / "manifest.jsonl",
            manifest_path=tmp_path / "manifest.jsonl",
            distill_path=tmp_path / "labels.jsonl",
            image_root=tmp_path,
        ),
        teacher=TeacherConfig(model_name="mock-teacher", backend="mock"),
        student=StudentConfig(model_name="mock-student", output_dir=tmp_path / "out", adapter_dir=tmp_path / "adapter"),
        distillation=DistillationConfig(method="switch_kd"),
    )

    sample = VlmSample(
        id="parsing-000001",
        image="screen.png",
        task="parsing",
        query="List all visible UI elements.",
        metadata={"elements": [{"text": "Home", "bbox_norm": [1, 2, 3, 4], "focused": False}]},
    )

    output_path = create_teacher_precompute_dataset(config, [sample])
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert set(row.keys()) == {"id", "image", "task", "query", "elements", "coordinate_system"}
