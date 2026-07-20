from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

import yaml

@dataclass
class DataConfig:
    training_manifest_path: Path
    distill_path: Path
    inference_manifest_path: Path | None = None
    label_path: Path | None = None
    prediction_path: Path | None = None
    teacher_logits_path: Path | None = None
    switch_logits_path: Path | None = None
    eval_path: Path | None = None
    image_root: Path = Path(".")
    training_image_dir: Path | None = None
    inference_image_dir: Path | None = None
    manifest_path: Path | None = None
    image_dir: Path | None = None
    output_dir: Path | None = None
    max_samples: int | None = None


@dataclass
class TeacherConfig:
    model_name: str
    backend: str = "mock"
    device_map: str | None = None
    attn_implementation: str = "sdpa"
    base_url: str | None = None
    api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    request_timeout: int = 120
    torch_dtype: str | None = None
    quantization: str = "none"
    temperature: float = 0.2
    max_new_tokens: int = 128
    image_resize: str = "original"
    retry_on_invalid_parsing_json: bool = False


@dataclass
class StudentConfig:
    model_name: str
    output_dir: Path
    adapter_dir: Path
    merged_model_path: Path | None = None
    deployment_artifact_path: Path | None = None
    copy_base_model_into_deployment: bool = False
    inference_adapter_path: Path | None = None
    inference_model_path: str | None = None
    device_map: str | None = "auto"
    torch_dtype: str | None = None
    attn_implementation: str = "sdpa"
    use_lora: bool = True
    load_adapter: bool = False
    merge_adapter: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    lora_layers_to_transform: list[int] | None = None
    lora_layers_pattern: str | None = None
    quantization: str = "none"
    train_multimodal_projector: bool = False
    use_projector_lora: bool = False
    projector_lora_rank: int | None = None
    projector_lora_alpha: int | None = None
    projector_lora_dropout: float | None = None
    multimodal_projector_path: str = "model.visual.merger"
    merged_artifact_mode: str = "bf16_standalone"
    allow_dequantized_projector_fallback: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ddp_find_unused_parameters: bool | None = False
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_steps: int | None = None
    log_every: int = 10
    save_every: int = 500
    mixed_precision: str = "no"
    gradient_checkpointing: bool = True
    max_length: int = 512
    image_resize: str = "original"
    freeze_vision_tower: bool = True
    mask_prompt_labels: bool = True
    quantization: str = "none"


@dataclass
class VisualSwitchConfig:
    mode: str = "paper"
    teacher_projector: str = "native"
    allow_fallback_adapter: bool = False
    adapter_path: str | None = None


@dataclass
class SwitchKDConfig:
    enabled: bool = True
    visual_switch: VisualSwitchConfig = field(default_factory=VisualSwitchConfig)


@dataclass
class DistillationConfig:
    confidence_weighting: bool = True
    min_teacher_confidence: float = 0.0
    prompt_template: str = "Query: {query}\nAnswer:"
    method: str = "sft"
    lm_loss_weight: float = 1.0
    dbild_loss_weight: float = 0.5
    vsd_loss_weight: float = 0.5
    inactive_logit_margin: float = 30.0
    kd_temperature: float = 2.0
    dbild_top_k: int = 64
    dbild_top_k_mode: str = "fixed"
    dbild_kneedle_candidate_k: int = 256
    dbild_min_top_k: int = 4
    dbild_max_top_k: int | None = None
    dbild_kl_mode: str = "symmetric"
    dbild_min_prob: float = 0.0
    teacher_logits: bool = True
    teacher_logits_field: str = "teacher_logits"
    switch_logits_field: str = "switch_logits"
    use_cached_logits: bool = True
    student_vision_path: str | None = None
    student_projector_path: str | None = None
    teacher_projector_path: str | None = None
    teacher_lm_path: str | None = None
    teacher_token_embedding_path: str | None = None
    teacher_lm_head_path: str | None = None
    switch_cache_student_visual: bool = False
    student_visual_cache_dir: Path | None = None
    keep_student_visual_cache_on_cpu: bool = True
    visual_token_placeholder: str = "<image>"
    max_cached_logits_vocab: int | None = 4096
    align_kd_logits_to_answer: bool = True
    skip_kd_on_vocab_mismatch: bool = True
    switch_kd: SwitchKDConfig = field(default_factory=SwitchKDConfig)


@dataclass
class EvaluationConfig:
    output_path: Path = Path("outputs/eval_report.json")
    metrics: list[str] = field(default_factory=lambda: ["exact_match", "token_f1"])


@dataclass
class PredictionConfig:
    debug_inference_parity: bool = False


@dataclass
class PipelineConfig:
    data: DataConfig
    teacher: TeacherConfig
    student: StudentConfig
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)


OUTPUT_ROOT_ENV_VARS = (
    "VLM_DISTILL_OUTPUT_ROOT",
    "CODEX_OUTPUT_ROOT",
)

_OFFLINE_LOGITS_WARNING = (
    "Warning: offline teacher logits config is deprecated and ignored. "
    "Online DBiLD computes logits during training."
)
_OFFLINE_LOGITS_WARNING_EMITTED = False


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw = _load_raw_config(config_path)
    raw = _apply_config_options(raw)
    return PipelineConfig(
        seed=raw.get("seed", 42),
        data=_build_data_config(raw["data"]),
        teacher=TeacherConfig(**raw["teacher"]),
        student=_build_student_config(raw["student"]),
        training=TrainingConfig(**raw.get("training", {})),
        distillation=_build_distillation_config(raw.get("distillation", {})),
        evaluation=_build_evaluation_config(raw.get("evaluation", {})),
        prediction=PredictionConfig(**raw.get("prediction", {})),
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Config extends missing file: {parent_path}")
    return _deep_merge(_load_raw_config(parent_path), raw)


def resolve_output_root() -> Path | None:
    for env_name in OUTPUT_ROOT_ENV_VARS:
        raw = os.environ.get(env_name)
        if not raw:
            raw = _read_windows_user_env(env_name)
        if raw:
            return Path(raw).expanduser()
    return None


def _read_windows_user_env(env_name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, env_name)
    except OSError:
        return None

    return value if isinstance(value, str) and value else None


def remap_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    root = resolve_output_root()
    if root is None:
        return path

    parts = path.parts
    if not parts or parts[0] != "outputs":
        return path

    return root.joinpath(*parts[1:])


def remap_output_path_string(value: str | None) -> str | None:
    if not value:
        return value
    return str(remap_output_path(Path(value)))


def _build_data_config(raw: dict[str, Any]) -> DataConfig:
    values = dict(raw)
    _warn_if_deprecated_offline_logits_config(
        values,
        deprecated_keys=("teacher_logits_path", "switch_logits_path"),
    )
    if values.get("training_manifest_path") is None and values.get("manifest_path") is not None:
        values["training_manifest_path"] = values["manifest_path"]
    if values.get("training_image_dir") is None and values.get("image_dir") is not None:
        values["training_image_dir"] = values["image_dir"]
    for key in (
        "training_manifest_path",
        "manifest_path",
        "distill_path",
        "inference_manifest_path",
        "label_path",
        "prediction_path",
        "teacher_logits_path",
        "switch_logits_path",
        "eval_path",
        "image_root",
        "training_image_dir",
        "inference_image_dir",
        "image_dir",
        "output_dir",
    ):
        if values.get(key) is not None:
            values[key] = remap_output_path(Path(values[key]))
    return DataConfig(**values)


def _build_student_config(raw: dict[str, Any]) -> StudentConfig:
    values = dict(raw)
    from .student_trainability import validate_configured_lora_targets
    configured_targets = values.get("target_modules", [])
    if not isinstance(configured_targets, list):
        raise ValueError("student.target_modules must be a list of module target names.")
    if configured_targets:
        values["target_modules"] = validate_configured_lora_targets(configured_targets)
    layers_to_transform = values.get("lora_layers_to_transform")
    if layers_to_transform is not None:
        if not isinstance(layers_to_transform, list) or not layers_to_transform:
            raise ValueError("student.lora_layers_to_transform must be null or a non-empty list.")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0
               for index in layers_to_transform):
            raise ValueError(
                "student.lora_layers_to_transform must contain only non-negative integers."
            )
        if len(set(layers_to_transform)) != len(layers_to_transform):
            raise ValueError("student.lora_layers_to_transform must not contain duplicates.")
        layers_pattern = values.get("lora_layers_pattern")
        if not isinstance(layers_pattern, str) or not layers_pattern.strip():
            raise ValueError(
                "student.lora_layers_pattern must be a non-empty string when layers are configured."
            )
        values["lora_layers_pattern"] = layers_pattern.strip()
    elif values.get("lora_layers_pattern") is not None:
        if not isinstance(values["lora_layers_pattern"], str) or not values["lora_layers_pattern"].strip():
            raise ValueError("student.lora_layers_pattern must be null or a non-empty string.")
        values["lora_layers_pattern"] = values["lora_layers_pattern"].strip()
    if not isinstance(values.get("train_multimodal_projector", False), bool):
        raise ValueError("student.train_multimodal_projector must be a boolean.")
    if not isinstance(values.get("use_projector_lora", False), bool):
        raise ValueError("student.use_projector_lora must be a boolean.")
    if not isinstance(values.get("allow_dequantized_projector_fallback", False), bool):
        raise ValueError("student.allow_dequantized_projector_fallback must be a boolean.")
    if values.get("train_multimodal_projector", False) and values.get("use_projector_lora", False):
        raise ValueError("student.train_multimodal_projector and student.use_projector_lora are mutually exclusive (A1/A2).")
    if values.get("use_projector_lora", False) and not values.get("use_lora", True):
        raise ValueError("student.use_projector_lora=true requires student.use_lora=true.")
    projector_path = values.get("multimodal_projector_path", "model.visual.merger")
    if not isinstance(projector_path, str) or not projector_path.strip():
        raise ValueError("student.multimodal_projector_path must be a non-empty dotted module path.")
    values["multimodal_projector_path"] = projector_path.strip()
    if values.get("use_projector_lora", False) and values["multimodal_projector_path"] != "model.visual.merger":
        raise ValueError("A2 projector LoRA is only supported for student.multimodal_projector_path=model.visual.merger.")
    for key in ("projector_lora_rank", "projector_lora_alpha"):
        if values.get(key) is not None and (isinstance(values[key], bool) or int(values[key]) <= 0):
            raise ValueError(f"student.{key} must be null or a positive integer.")
        if values.get(key) is not None:
            values[key] = int(values[key])
    if values.get("projector_lora_dropout") is not None:
        dropout = float(values["projector_lora_dropout"])
        if not 0 <= dropout < 1:
            raise ValueError("student.projector_lora_dropout must be null or in [0, 1).")
        values["projector_lora_dropout"] = dropout
    if values.get("use_projector_lora"):
        configured = {str(item) for item in values.get("target_modules", [])}
        if "model.visual.merger" in configured or any("deepstack_merger" in item for item in configured):
            raise ValueError("A2 must not place the projector or deepstack merger in target_modules/modules_to_save.")
    if values.get("train_multimodal_projector", False) and values.get("use_projector_lora", False):
        raise ValueError("full projector and projector LoRA are mutually exclusive.")
    artifact_mode = str(values.get("merged_artifact_mode", "bf16_standalone"))
    if artifact_mode not in {
        "mixed_4bit_bf16", "bf16_standalone", "adapter_plus_projector",
        "4bit_base_bf16_adapter",
    }:
        raise ValueError(
            "student.merged_artifact_mode must be one of: mixed_4bit_bf16, "
            "bf16_standalone, adapter_plus_projector, 4bit_base_bf16_adapter."
        )
    values["merged_artifact_mode"] = artifact_mode
    if artifact_mode == "4bit_base_bf16_adapter":
        if values.get("quantization") != "4bit":
            raise ValueError("4bit_base_bf16_adapter requires student.quantization=4bit.")
        if not values.get("use_lora", True):
            raise ValueError("4bit_base_bf16_adapter requires student.use_lora=true.")
        if values.get("multimodal_projector_path", "model.visual.merger") != "model.visual.merger":
            raise ValueError(
                "4bit_base_bf16_adapter requires student.multimodal_projector_path=model.visual.merger."
            )
        if values.get("copy_base_model_into_deployment", False):
            raise ValueError(
                "copy_base_model_into_deployment=true is unsupported: a standalone bnb 4-bit base "
                "artifact cannot be reliably saved by this Transformers/bitsandbytes stack."
            )
    for key in ("output_dir", "adapter_dir"):
        values[key] = remap_output_path(Path(values[key]))
    merged_model_path = values.get("merged_model_path")
    if merged_model_path:
        values["merged_model_path"] = remap_output_path(Path(merged_model_path))
    else:
        values["merged_model_path"] = None
    deployment_path = values.get("deployment_artifact_path")
    values["deployment_artifact_path"] = (
        remap_output_path(Path(deployment_path)) if deployment_path else None
    )
    inference_adapter_path = values.get("inference_adapter_path")
    if inference_adapter_path:
        values["inference_adapter_path"] = remap_output_path(Path(inference_adapter_path))
    else:
        values["inference_adapter_path"] = None
    if values.get("inference_model_path") is not None:
        values["inference_model_path"] = remap_output_path_string(values["inference_model_path"])
    return StudentConfig(**values)


def _build_distillation_config(raw: dict[str, Any]) -> DistillationConfig:
    values = dict(raw)
    _warn_if_deprecated_offline_logits_config(
        values,
        deprecated_keys=(
            "teacher_logits",
            "teacher_logits_field",
            "teacher_logits_mode",
            "switch_logits_field",
            "use_cached_logits",
        ),
    )
    legacy_target_field = values.pop("target_field", None)
    if legacy_target_field not in (None, "student_target", "teacher_answer"):
        raise ValueError(
            "distillation.target_field is no longer configurable. "
            "Use teacher_answer as the single training target field."
        )
    for key in (
        "student_vision_path",
        "student_projector_path",
        "teacher_projector_path",
        "teacher_lm_path",
        "teacher_token_embedding_path",
        "teacher_lm_head_path",
    ):
        if values.get(key) is not None:
            values[key] = remap_output_path_string(values[key])
    if values.get("student_visual_cache_dir") is not None:
        values["student_visual_cache_dir"] = remap_output_path(Path(values["student_visual_cache_dir"]))
    values["switch_kd"] = _build_switch_kd_config(values.get("switch_kd", {}))
    dbild_top_k_mode = values.get("dbild_top_k_mode", "fixed")
    if dbild_top_k_mode not in {"fixed", "kneedle"}:
        raise ValueError("distillation.dbild_top_k_mode must be one of: fixed, kneedle.")
    dbild_kl_mode = values.get("dbild_kl_mode", "symmetric")
    if dbild_kl_mode not in {"symmetric", "reverse", "forward"}:
        raise ValueError("distillation.dbild_kl_mode must be one of: symmetric, reverse, forward.")
    dbild_min_top_k = int(values.get("dbild_min_top_k", 4))
    if dbild_min_top_k < 1:
        raise ValueError("distillation.dbild_min_top_k must be >= 1.")
    dbild_kneedle_candidate_k = int(values.get("dbild_kneedle_candidate_k", 256))
    if dbild_kneedle_candidate_k < dbild_min_top_k:
        raise ValueError("distillation.dbild_kneedle_candidate_k must be >= distillation.dbild_min_top_k.")
    values["dbild_min_top_k"] = dbild_min_top_k
    values["dbild_kneedle_candidate_k"] = dbild_kneedle_candidate_k
    dbild_max_top_k = values.get("dbild_max_top_k")
    if dbild_max_top_k is not None:
        dbild_max_top_k = int(dbild_max_top_k)
        if dbild_max_top_k < dbild_min_top_k:
            raise ValueError("distillation.dbild_max_top_k must be >= distillation.dbild_min_top_k.")
        values["dbild_max_top_k"] = dbild_max_top_k
    return DistillationConfig(**values)


def _build_switch_kd_config(raw: Any) -> SwitchKDConfig:
    values = dict(raw or {})
    visual_switch = _build_visual_switch_config(values.get("visual_switch", {}))
    values["visual_switch"] = visual_switch
    enabled = values.get("enabled", True)
    return SwitchKDConfig(enabled=bool(enabled), visual_switch=visual_switch)


def _build_visual_switch_config(raw: Any) -> VisualSwitchConfig:
    values = dict(raw or {})
    mode = str(values.get("mode", "paper"))
    if mode not in {"paper", "adapter_to_teacher_projector", "adapter_to_teacher_lm"}:
        raise ValueError(
            "distillation.switch_kd.visual_switch.mode must be one of: "
            "paper, adapter_to_teacher_projector, adapter_to_teacher_lm."
        )
    teacher_projector = str(values.get("teacher_projector", "native"))
    if teacher_projector != "native":
        raise ValueError(
            "distillation.switch_kd.visual_switch.teacher_projector must be 'native'."
        )
    allow_fallback_adapter = bool(values.get("allow_fallback_adapter", False))
    if mode != "paper" and not allow_fallback_adapter:
        raise ValueError(
            "distillation.switch_kd.visual_switch.allow_fallback_adapter must be true "
            "when mode is adapter_to_teacher_projector or adapter_to_teacher_lm."
        )
    adapter_path = values.get("adapter_path")
    if adapter_path is not None:
        adapter_path = remap_output_path_string(str(adapter_path))
    return VisualSwitchConfig(
        mode=mode,
        teacher_projector=teacher_projector,
        allow_fallback_adapter=allow_fallback_adapter,
        adapter_path=adapter_path,
    )


def _build_evaluation_config(raw: dict[str, Any]) -> EvaluationConfig:
    values = dict(raw)
    if values.get("output_path") is not None:
        values["output_path"] = remap_output_path(Path(values["output_path"]))
    return EvaluationConfig(**values)


def _apply_config_options(raw: dict[str, Any]) -> dict[str, Any]:
    values = dict(raw)
    options_raw = values.pop("options", {})
    if not isinstance(options_raw, dict):
        raise ValueError("options must be a mapping when provided.")

    options: dict[str, str] = {
        key: str(value)
        for key, value in options_raw.items()
        if value is not None
    }
    quality = options.get("quality")
    teacher_quantization = (
        options.get("teacher_quantization")
        or options.get("teacher_label_quantization")
    )
    student_quantization = options.get("student_quantization")
    task_name = options.get("task_name", "parsing")

    if teacher_quantization:
        options.setdefault("teacher_quantization", teacher_quantization)
        options.setdefault("teacher_label_quantization", teacher_quantization)

    if quality and teacher_quantization:
        options.setdefault("label_profile", f"{quality}_{teacher_quantization}")
    if quality and teacher_quantization and student_quantization:
        options.setdefault(
            "response_profile",
            f"{quality}_{teacher_quantization}_student_{student_quantization}",
        )
    elif quality and teacher_quantization:
        options.setdefault("response_profile", f"{quality}_{teacher_quantization}")
    options.setdefault("task_name", task_name)

    return _interpolate_config_values(values, options)


def _interpolate_config_values(value: Any, options: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _interpolate_config_values(nested_value, options)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_interpolate_config_values(item, options) for item in value]
    if isinstance(value, str):
        return _replace_known_placeholders(value, options)
    return value


def _replace_known_placeholders(template: str, options: dict[str, str]) -> str:
    pattern = re.compile(r"\{([A-Za-z0-9_]+)\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return options.get(key, match.group(0))

    return pattern.sub(replace, template)


def build_prompt_context(
    *,
    query: str | None = None,
    target_label: str | None = None,
    target_type: str | None = None,
    task: str | None = None,
) -> dict[str, str]:
    query_text = query or ""
    return {
        "query": query_text,
        "question": query_text,
        "target_label": target_label or "",
        "target_type": target_type or "",
        "task": task or "",
    }


def format_prompt(
    template: str,
    *,
    query: str | None = None,
    target_label: str | None = None,
    target_type: str | None = None,
    task: str | None = None,
) -> str:
    return template.format(
        **build_prompt_context(
            query=query,
            target_label=target_label,
            target_type=target_type,
            task=task,
        )
    )


def resolve_label_path(data: DataConfig) -> Path:
    return data.label_path or data.distill_path


def resolve_prediction_path(data: DataConfig) -> Path:
    return data.prediction_path or data.distill_path


def resolve_training_manifest_path(data: DataConfig) -> Path:
    return data.training_manifest_path


def resolve_inference_manifest_path(data: DataConfig) -> Path:
    return data.inference_manifest_path or resolve_training_manifest_path(data)


def resolve_training_image_dir(data: DataConfig) -> Path | None:
    return data.training_image_dir or data.image_dir


def resolve_inference_image_dir(data: DataConfig) -> Path | None:
    return data.inference_image_dir or data.image_dir


def resolve_teacher_logits_path(data: DataConfig) -> Path:
    if data.teacher_logits_path is not None:
        return data.teacher_logits_path

    label_path = resolve_label_path(data)
    suffix = "".join(label_path.suffixes) or ".jsonl"
    stem = label_path.name[: -len(suffix)] if suffix else label_path.name
    return label_path.with_name(f"{stem}_teacher_logits{suffix}")


def resolve_switch_logits_path(data: DataConfig) -> Path:
    return data.switch_logits_path or data.distill_path


def _warn_if_deprecated_offline_logits_config(
    values: dict[str, Any],
    *,
    deprecated_keys: tuple[str, ...],
) -> None:
    global _OFFLINE_LOGITS_WARNING_EMITTED
    if _OFFLINE_LOGITS_WARNING_EMITTED:
        return
    if any(key in values and values.get(key) not in (None, False, "", []) for key in deprecated_keys):
        print(_OFFLINE_LOGITS_WARNING)
        _OFFLINE_LOGITS_WARNING_EMITTED = True
