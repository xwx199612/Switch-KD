from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urljoin

from .config_schema import (
    PipelineConfig,
    format_prompt,
    resolve_label_path,
    resolve_training_manifest_path,
)
from .data_manifest import VlmSample, read_jsonl, validate_manifest
from .device_utils import (
    batch_to_device,
    get_module_by_path,
    ensure_stage_uses_cuda,
    print_stage_model_debug,
    resolve_requested_device_map,
    select_model_input_device,
)
from .model_output_artifacts import (
    attach_parsing_sidecar_outputs,
    refresh_parsing_sidecar_reports,
)
from .model_loading import apply_attn_implementation, resolve_model_path
from .parsing_output_parser import elements_to_line_format, parse_parsing_answer


class TeacherBackend(Protocol):
    def answer(self, sample: VlmSample) -> dict:
        ...


class MockTeacher:
    def answer(self, sample: VlmSample) -> dict:
        seed = f"{sample.id}:{sample.query}:{sample.answer or ''}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        confidence = 0.55 + (int(digest[:4], 16) / 0xFFFF) * 0.4

        if sample.answer:
            teacher_answer = sample.answer
        elif sample.task == "parsing":
            elements = sample.metadata.get("elements") if isinstance(sample.metadata, dict) else None
            if isinstance(elements, list):
                teacher_answer = elements_to_line_format(
                    [element for element in elements if isinstance(element, dict)]
                )
            else:
                teacher_answer = (
                    "mock icon | 0,0,100,100 | false\n"
                    "mock settings | 100,0,200,100 | true"
                )
        else:
            teacher_answer = f"mock answer for {sample.task}"

        return {
            "teacher_answer": teacher_answer,
            "teacher_tokens": [],
            "teacher_confidence": round(confidence, 4),
            "teacher_rationale": "Mock backend used for pipeline validation.",
        }


class HuggingFaceTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._input_device = None
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVLM
            except ImportError:  # pragma: no cover - fallback for older transformers
                from transformers import AutoModelForVision2Seq as AutoModelForVLM
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers and bitsandbytes to use the Hugging Face teacher backend."
            ) from exc

        model_path = resolve_model_path(config.teacher.model_name)
        requested_device_map = resolve_requested_device_map(
            config.teacher.device_map,
            quantization=config.teacher.quantization,
            role="teacher",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        allowed_quantization = {"none", "4bit", "8bit"}

        if config.teacher.quantization not in allowed_quantization:
            raise ValueError(
                f"Unsupported teacher quantization: "
                f"{config.teacher.quantization}. "
                f"Allowed values: {sorted(allowed_quantization)}"
            )
        model_kwargs = {
            "device_map": requested_device_map,
            "trust_remote_code": True,
        }
        apply_attn_implementation(model_kwargs, config.teacher.attn_implementation)

        if config.teacher.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.teacher.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            if config.teacher.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif config.teacher.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif config.teacher.torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForVLM.from_pretrained(
            model_path,
            **model_kwargs,
            local_files_only=True,
        )
        self._input_device = select_model_input_device(
            self.model,
            preferred_modules=(getattr(self.model, "visual", None),),
            label="Teacher",
        )
        print_stage_model_debug(
            stage_label="Teacher",
            model_path=model_path,
            quantization_mode=config.teacher.quantization,
            requested_device_map=requested_device_map,
            model=self.model,
            selected_input_device=self._input_device,
        )
        ensure_stage_uses_cuda(
            stage_label="Teacher",
            requested_device_map=requested_device_map,
            model=self.model,
            selected_input_device=self._input_device,
        )

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image = _load_teacher_image(image_path, self.config.teacher.image_resize)

        prompt = _format_prompt(self.config, sample)
        answer, _generated_ids = self._generate(image=image, prompt=prompt, sample=sample)
        answer = _normalize_teacher_answer(sample, answer)

        if sample.task == "parsing" and _parsing_quality_score(answer) <= 2:
            retry_prompt = _build_parsing_retry_prompt(sample)
            retry_answer, _retry_ids = self._generate(
                image=image,
                prompt=retry_prompt,
                sample=sample,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )
            retry_answer = _normalize_teacher_answer(sample, retry_answer)
            if _parsing_quality_score(retry_answer) >= _parsing_quality_score(answer):
                answer = retry_answer

        return {
            "teacher_answer": answer.strip(),
            "teacher_tokens": self.tokenize_teacher_answer(answer.strip()),
            "teacher_confidence": 1.0,
            "teacher_rationale": "Generated by Hugging Face teacher backend.",
        }

    def tokenize_teacher_answer(self, answer: str) -> list[int]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            encoded = self.processor(text=[answer], return_tensors=None)
            input_ids = encoded["input_ids"][0]
        else:
            input_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        return [int(token_id) for token_id in input_ids]

    def decode_teacher_tokens(self, token_ids: list[int]) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        decoder = tokenizer if tokenizer is not None else self.processor
        return decoder.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _generate(
        self,
        *,
        image,
        prompt: str,
        sample: VlmSample,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
    ) -> tuple[str, list[int]]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self._input_device)

        generation_kwargs = {
            "do_sample": self.config.teacher.temperature > 0,
            "max_new_tokens": self.config.teacher.max_new_tokens,
        }
        if self.config.teacher.temperature > 0:
            generation_kwargs["temperature"] = self.config.teacher.temperature
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size is not None:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        output_ids = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if sample.task == "parsing" and _looks_degenerate_screen_output(answer):
            return "", []
        token_ids = generated_ids[0].detach().cpu().tolist() if generated_ids.shape[0] > 0 else []
        return answer, token_ids

class OpenAICompatibleTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._requests = None
        self._openai_client = None
        if not config.teacher.base_url:
            raise ValueError("teacher.base_url is required when backend='openai_compatible'.")

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image_data_url = _image_to_data_url(image_path, resize_mode=self.config.teacher.image_resize)
        prompt = _format_prompt(self.config, sample)

        payloads = [
            self._build_responses_payload(prompt, image_data_url),
            self._build_chat_payload(prompt, image_data_url),
        ]
        errors: list[str] = []
        for api_mode, payload in payloads:
            try:
                content = self._call_api(api_mode=api_mode, payload=payload)
                return {
                    "teacher_answer": content.strip(),
                    "teacher_tokens": [],
                    "teacher_confidence": 1.0,
                    "teacher_rationale": "Generated by OpenAI-compatible teacher backend.",
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{api_mode}: {exc}")
                continue

        raise RuntimeError(
            "OpenAI-compatible teacher backend failed for both responses and chat/completions. "
            + " | ".join(errors)
        )

    def _build_chat_payload(self, prompt: str, image_data_url: str) -> tuple[str, dict]:
        return (
            "chat/completions",
            {
                "model": self.config.teacher.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                "temperature": self.config.teacher.temperature,
                "max_tokens": self.config.teacher.max_new_tokens,
            },
        )

    def _build_responses_payload(self, prompt: str, image_data_url: str) -> tuple[str, dict]:
        return (
            "responses",
            {
                "model": self.config.teacher.model_name,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_data_url},
                        ],
                    }
                ],
                "temperature": self.config.teacher.temperature,
                "max_output_tokens": self.config.teacher.max_new_tokens,
            },
        )

    def _call_api(self, *, api_mode: str, payload: dict) -> str:
        if _has_requests():
            return self._call_via_requests(api_mode=api_mode, payload=payload)
        if _has_openai():
            return self._call_via_openai(api_mode=api_mode, payload=payload)
        raise RuntimeError(
            "OpenAI-compatible backend requires either `requests` or `openai` to be installed."
        )

    def _call_via_requests(self, *, api_mode: str, payload: dict) -> str:
        requests = _import_requests()
        endpoint = f"/{api_mode}"
        response = requests.post(
            _join_url(self.config.teacher.base_url, endpoint),
            headers=_auth_headers(self.config.teacher.api_key),
            json=payload,
            timeout=self.config.teacher.request_timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
        return _extract_openai_compatible_text(data, api_mode=api_mode)

    def _call_via_openai(self, *, api_mode: str, payload: dict) -> str:
        client = _openai_client(self.config.teacher.base_url, self.config.teacher.api_key)
        if api_mode == "chat/completions":
            response = client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                temperature=payload["temperature"],
                max_tokens=payload["max_tokens"],
                timeout=self.config.teacher.request_timeout,
            )
            return _extract_openai_sdk_text(response, api_mode=api_mode)
        if api_mode == "responses":
            response = client.responses.create(
                model=payload["model"],
                input=payload["input"],
                temperature=payload["temperature"],
                max_output_tokens=payload["max_output_tokens"],
                timeout=self.config.teacher.request_timeout,
            )
            return _extract_openai_sdk_text(response, api_mode=api_mode)
        raise ValueError(f"Unsupported api_mode: {api_mode}")


class OllamaTeacher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        if not config.teacher.ollama_host:
            raise ValueError("teacher.ollama_host must be set for backend='ollama'.")

    def answer(self, sample: VlmSample) -> dict:
        image_path = self.config.data.image_root / sample.image
        image_data = _image_to_base64(image_path, resize_mode=self.config.teacher.image_resize)
        prompt = _format_prompt(self.config, sample)
        payload = {
            "model": self.config.teacher.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.config.teacher.temperature,
                "num_predict": self.config.teacher.max_new_tokens,
            },
        }
        response = _import_requests().post(
            _join_url(self.config.teacher.ollama_host, "/api/chat"),
            json=payload,
            timeout=self.config.teacher.request_timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
        content = _extract_ollama_text(data)
        return {
            "teacher_answer": content.strip(),
            "teacher_tokens": [],
            "teacher_confidence": 1.0,
            "teacher_rationale": "Generated by Ollama teacher backend.",
        }


def build_teacher(config: PipelineConfig) -> TeacherBackend:
    if config.teacher.backend == "mock":
        return MockTeacher()
    if config.teacher.backend == "hf":
        return HuggingFaceTeacher(config)
    if config.teacher.backend == "openai_compatible":
        return OpenAICompatibleTeacher(config)
    if config.teacher.backend == "ollama":
        return OllamaTeacher(config)
    raise ValueError(f"Unknown teacher backend: {config.teacher.backend}")

def _format_prompt(config: PipelineConfig, sample: VlmSample) -> str:
    return format_prompt(
        config.distillation.prompt_template,
        query=sample.query,
        task=sample.task,
    )


def _target_from_existing_annotation(sample: VlmSample) -> str | None:
    elements = sample.metadata.get("elements") if isinstance(sample.metadata, dict) else None

    if sample.task == "parsing" and elements:
        if isinstance(elements, list):
            line_answer = elements_to_line_format(
                [element for element in elements if isinstance(element, dict)]
            )
            if line_answer:
                return line_answer
        return json.dumps(
            elements if isinstance(elements, dict) else {"elements": elements},
            ensure_ascii=False,
        )

    if sample.answer:
        return sample.answer

    return None

def _label_sample(
    config: PipelineConfig,
    teacher: TeacherBackend,
    sample: VlmSample,
) -> dict | None:
    existing_target = _target_from_existing_annotation(sample)

    if existing_target is not None:
        label = {
            "teacher_answer": existing_target,
            "teacher_tokens": [],
            "teacher_confidence": 1.0,
            "teacher_rationale": "Used existing manifest annotation.",
        }
    else:
        label = teacher.answer(sample)

    label["teacher_answer"] = _normalize_teacher_answer(sample, label["teacher_answer"]).strip()
    tokenizer = getattr(teacher, "tokenize_teacher_answer", None)
    if callable(tokenizer):
        label["teacher_tokens"] = tokenizer(label["teacher_answer"])

    decoder = getattr(teacher, "decode_teacher_tokens", None)
    _validate_generated_label(sample, label, decoder=decoder if callable(decoder) else None)

    if label["teacher_confidence"] < config.distillation.min_teacher_confidence:
        return None

    return {
        **_base_output_row(sample),
        **label,
    }


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed_ids: set[str] = set()
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is not None:
            completed_ids.add(str(sample_id))
    return completed_ids


def _load_teacher_image(image_path: Path, resize_mode: str):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    return _resize_teacher_image(image, resize_mode)


def _resize_teacher_image(image, resize_mode: str):
    mode = _normalize_image_resize_mode(resize_mode)
    if mode == "original":
        return image

    target_height = {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
    }[mode]
    width, height = image.size
    if height <= target_height:
        return image

    target_width = round(width * target_height / height)
    return image.resize((target_width, target_height), _pil_lanczos())


def _resized_image_bytes(image_path: Path, resize_mode: str) -> bytes:
    image = _load_teacher_image(image_path, resize_mode)
    suffix = image_path.suffix.lower()
    image_format = "PNG" if suffix == ".png" else "JPEG"
    buffer = BytesIO()
    save_kwargs = {"format": image_format}
    if image_format == "JPEG":
        save_kwargs["quality"] = 95
    image.save(buffer, **save_kwargs)
    return buffer.getvalue()


def _normalize_image_resize_mode(resize_mode: str | None) -> str:
    mode = (resize_mode or "original").lower()
    aliases = {
        "none": "original",
        "no": "original",
        "off": "original",
        "native": "original",
        "original": "original",
        "1080": "1080p",
        "1080p": "1080p",
        "720": "720p",
        "720p": "720p",
        "480": "480p",
        "480p": "480p",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported teacher.image_resize={resize_mode!r}. "
            "Use one of: original, 480p, 720p, 1080p."
        )
    return aliases[mode]


def _pil_lanczos():
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _image_to_base64(image_path: Path, *, resize_mode: str = "original") -> str:
    if _normalize_image_resize_mode(resize_mode) == "original":
        return base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_bytes = _resized_image_bytes(image_path, resize_mode)
    return base64.b64encode(image_bytes).decode("ascii")


def _image_to_data_url(image_path: Path, *, resize_mode: str = "original") -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{_image_to_base64(image_path, resize_mode=resize_mode)}"


def _join_url(base_url: str, endpoint: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _has_requests() -> bool:
    return importlib.util.find_spec("requests") is not None


def _import_requests():
    if not _has_requests():
        raise RuntimeError(
            "The selected teacher backend requires the `requests` package. "
            "Install it in the environment or use the Hugging Face backend."
        )
    import requests

    return requests


def _has_openai() -> bool:
    return importlib.util.find_spec("openai") is not None


def _openai_client(base_url: str | None, api_key: str | None):
    if not _has_openai():
        raise RuntimeError(
            "OpenAI-compatible backend can use the `openai` package, but it is not installed."
        )
    from openai import OpenAI

    kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _extract_openai_compatible_text(data: dict, *, api_mode: str) -> str:
    if isinstance(data, dict):
        if "output_text" in data and data["output_text"]:
            return str(data["output_text"])
        if api_mode == "chat/completions":
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content)
        if api_mode == "responses":
            if "output" in data:
                return _extract_openai_output_list(data["output"])
            if "content" in data:
                return _extract_openai_output_list(data["content"])
    raise RuntimeError(f"Could not parse OpenAI-compatible response payload: {data}")


def _extract_openai_sdk_text(response, *, api_mode: str) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    if api_mode == "chat/completions":
        choices = getattr(response, "choices", [])
        if choices:
            message = choices[0].message
            content = getattr(message, "content", "")
            if isinstance(content, list):
                return "".join(
                    getattr(part, "text", "") if not isinstance(part, str) else part
                    for part in content
                )
            return str(content)
    if api_mode == "responses":
        output = getattr(response, "output", None)
        if output is not None:
            return _extract_openai_output_list(output)
    raise RuntimeError(f"Could not parse OpenAI SDK response payload: {response}")


def _extract_openai_output_list(output) -> str:
    parts: list[str] = []
    for item in output:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(part))
            elif content is not None:
                parts.append(str(content))
        else:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(getattr(part, "text", part)))
            elif content is not None:
                parts.append(str(content))
    return "".join(parts)


def _extract_ollama_text(data: dict) -> str:
    message = data.get("message") or {}
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    if data.get("response") is not None:
        return str(data["response"])
    raise RuntimeError(f"Could not parse Ollama response payload: {data}")


_COMMON_TOP_TABS = {"home", "shop", "discover", "apps"}
_SCREEN_SCHEMA_WORDS = {
    "text",
    "type",
    "focused",
    "true",
    "false",
    "elements",
}
_KNOWN_APP_NAMES = {
    "netflix",
    "youtube",
    "prime video",
    "sony select",
    "music",
    "line tv",
    "spotify",
    "iqiyi",
}
_SCREEN_SCHEMA_LABELS = {
    "",
    "true",
    "false",
    "null",
    "none",
    "id",
    "ref",
    "data",
    "version",
    "text",
    "label",
    "type",
    "icon",
    "button",
    "link",
    "tab",
    "tile",
    "toggle",
    "input",
    "menu item",
    "action",
    "elements",
    "element",
    "active navigation areas",
    "active navigation area",
    "selected active navigation area index",
    "top level tabs",
    "tab label",
    "is selected",
    "tab id",
    "is currently focused",
    "has focus indicators",
    "element type",
    "tab name",
    "is enabled",
    "tab title",
    "content items",
    "navigation elements",
    "ui elements",
    "additional ui features",
    "tile text",
    "tile icon",
    "tile plus button",
    "app tiles",
    "active",
    "selected",
    "status",
    "navigation",
    "content",
    "recommended",
    "navigation button",
    "content item",
    "text label",
}
_SCREEN_SCHEMA_SUBSTRINGS = (
    "schema",
    "json",
    "active navigation",
    "toplevel",
    "selectedtab",
    "tabfocused",
    "contentitems",
    "contentitemfocused",
    "uielements",
    "actionableelements",
    "focus indicators",
    "isactive",
    "isfocused",
    "tabindex",
    "active_top_tab",
    "focused_element",
    "focus_state",
)
_ALLOWED_SCREEN_ELEMENT_TYPES = {
    "tab",
    "button",
    "app_icon",
    "app_tile",
    "menu_item",
    "tile",
    "toggle",
    "input",
    "icon",
    "link",
    "other",
    "unknown",
}


def _normalize_teacher_answer(sample: VlmSample, teacher_answer: str) -> str:
    if sample.task != "parsing":
        return teacher_answer.strip()
    canonical = _canonicalize_line_teacher_answer(teacher_answer)
    if canonical is not None:
        return canonical
    if _parse_json_object(teacher_answer) is not None:
        payload = _normalize_parsing_payload(teacher_answer)
        return _compact_json(payload)
    return teacher_answer.strip()


def _normalize_parsing_payload(teacher_answer: str) -> dict[str, object]:
    payload = _empty_parsing_payload()
    parsed = _parse_json_object(teacher_answer)

    if isinstance(parsed, dict):
        raw_elements = parsed.get("elements")
        if raw_elements is None:
            raw_elements = parsed.get("selectable_elements")
        payload["elements"] = _normalize_screen_elements(raw_elements)

    if not payload["elements"]:
        payload["elements"] = _labels_to_screen_elements(_extract_candidate_labels(teacher_answer))

    return payload


def _empty_parsing_payload() -> dict[str, object]:
    return {
        "elements": [],
    }


def _parse_json_object(text: str) -> dict | None:
    candidate = text.strip()
    if not candidate:
        return None

    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _sanitize_screen_field(value: object) -> str:
    if value is None:
        return ""
    cleaned = _clean_screen_label(str(value))
    return "" if _is_screen_schema_label(cleaned) else cleaned


def _normalize_screen_elements(raw_elements: object) -> list[dict[str, object]]:
    if not isinstance(raw_elements, list):
        return []

    elements: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_element in raw_elements:
        label: object = ""
        element_type: object = "unknown"
        focused: object = False
        if isinstance(raw_element, str):
            label = raw_element
        elif isinstance(raw_element, dict):
            label = (
                raw_element.get("text")
                or raw_element.get("label")
                or raw_element.get("name")
                or raw_element.get("title")
                or ""
            )
            element_type = raw_element.get("type") or raw_element.get("role") or "unknown"
            focused = raw_element.get("focused", raw_element.get("focus", False))
        else:
            continue

        cleaned_label = _clean_screen_label(str(label))
        if not cleaned_label or _should_drop_screen_element_label(cleaned_label):
            continue

        lowered = cleaned_label.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized_type = _normalize_screen_element_type(element_type)
        normalized_type = _repair_screen_element_type(cleaned_label, normalized_type)
        elements.append(
            {
                "text": cleaned_label,
                "type": normalized_type,
                "focused": _normalize_screen_element_focused(focused),
            }
        )

    return elements


def _extract_candidate_labels(text: str) -> list[str]:
    candidates = re.findall(r'"([^"\n]{1,80})"', text)
    labels: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _clean_screen_label(candidate)
        if not cleaned or _should_drop_screen_element_label(cleaned):
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        labels.append(cleaned)
    return labels


def _labels_to_screen_elements(labels: list[str]) -> list[dict[str, object]]:
    elements: list[dict[str, object]] = []
    for label in labels:
        normalized_type = _repair_screen_element_type(label, "unknown")
        elements.append({"text": label, "type": normalized_type, "focused": False})
    return elements


def _normalize_screen_element_type(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"

    cleaned = _clean_screen_label(value)
    if not cleaned:
        return "unknown"

    snake = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    snake = re.sub(r"_+", "_", snake)
    if not snake:
        return "unknown"
    if snake in _ALLOWED_SCREEN_ELEMENT_TYPES:
        return snake
    if snake == "unknown":
        return "other"

    tokens = [token for token in snake.split("_") if token]
    token_set = set(tokens)

    if token_set & {"app", "application"}:
        return "app_icon"
    if token_set & {"tile", "card", "carousel", "recommend", "movie", "content", "poster", "banner"}:
        return "tile"
    if "menu" in token_set:
        return "menu_item"
    if token_set & {"nav", "navigation"}:
        return "tab"
    if token_set & {"search", "search_box", "searchbar", "search_bar", "input", "text_box", "textbox", "text"}:
        return "input"
    if token_set & {"toggle", "switch"}:
        return "toggle"
    if token_set & {"icon", "setting", "settings"}:
        return "icon"
    if "link" in token_set:
        return "link"
    if token_set & {"button", "btn"}:
        return "button"
    if token_set & {"text", "label", "image"}:
        return "other"

    if "unknown" in token_set:
        return "other"

    return "other"


def _should_drop_screen_element_label(label: str) -> bool:
    lowered = label.strip().lower()
    return lowered in _SCREEN_SCHEMA_WORDS or _is_screen_schema_label(label)


def _repair_screen_element_type(label: str, normalized_type: str) -> str:
    lowered = label.strip().lower()
    heuristic_type = _infer_screen_element_type_from_text(label)

    if lowered in _COMMON_TOP_TABS and normalized_type in {"unknown", "other", "input"}:
        return "tab"
    if normalized_type == "unknown":
        if heuristic_type != "unknown":
            return heuristic_type
        return "other"
    return normalized_type


def _infer_screen_element_type_from_text(label: str) -> str:
    lowered = label.strip().lower()
    if not lowered:
        return "unknown"
    if lowered in {"home", "shop", "discover", "apps"}:
        return "tab"
    if lowered == "search":
        return "input"
    if lowered in {"details", "dismiss", "+"}:
        return "button"
    if lowered in _KNOWN_APP_NAMES:
        return "app_icon"
    if any(token in lowered for token in ("channel", "setup", "program", "labels", "adjustment", "type")):
        return "menu_item"
    if any(token in lowered for token in ("recommended", "popular", "top selling", "movie", "show")):
        return "tile"
    return "unknown"


def _normalize_screen_element_focused(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "focused", "selected", "active"}
    return bool(value)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _strip_special_tokens(text: str) -> str:
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    return text.strip()


def _canonicalize_teacher_answer(answer: str) -> str:
    stripped = _strip_special_tokens(answer)
    canonical_line = _canonicalize_line_teacher_answer(stripped)
    if canonical_line is not None:
        return canonical_line
    parsed = _parse_json_object(stripped)
    if parsed is None:
        raise ValueError("teacher_answer is not valid JSON or canonical line format")
    return _canonical_json(parsed)


def _validate_parsing_teacher_answer(answer: str) -> tuple[bool, str | None]:
    parsed = _parse_json_object(answer)
    if parsed is None:
        return False, "teacher_answer is not valid JSON"
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        return False, "teacher_answer.elements is not a list"
    for index, element in enumerate(elements):
        if isinstance(element, str):
            return False, f"teacher_answer.elements[{index}] is a string-list item"
        if not isinstance(element, dict):
            return False, f"teacher_answer.elements[{index}] is not an object"
        missing = {"text", "type", "focused"} - set(element)
        if missing:
            return False, f"teacher_answer.elements[{index}] missing {sorted(missing)}"
        if not isinstance(element.get("text"), str):
            return False, f"teacher_answer.elements[{index}].text is not a string"
        if not isinstance(element.get("type"), str):
            return False, f"teacher_answer.elements[{index}].type is not a string"
        if not isinstance(element.get("focused"), bool):
            return False, f"teacher_answer.elements[{index}].focused is not a boolean"
    return True, None


def _validate_generated_label(sample: VlmSample, label: dict, *, decoder=None) -> None:
    teacher_tokens = label.get("teacher_tokens")
    if decoder is None or not teacher_tokens:
        return

    decoded = decoder([int(token_id) for token_id in teacher_tokens])
    answer_canonical = _canonicalize_teacher_answer(str(label["teacher_answer"]))
    decoded_canonical = _canonicalize_teacher_answer(decoded)
    if decoded_canonical != answer_canonical:
        raise ValueError(f"{sample.id}: decoded teacher_tokens do not match teacher_answer")


def _clean_screen_label(value: str) -> str:
    cleaned = value.strip().strip(",.:;!?'\"`[]{}()")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _is_screen_schema_label(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _SCREEN_SCHEMA_LABELS:
        return True
    if re.fullmatch(r"\d+", lowered):
        return True
    if len(lowered) == 1 and lowered.isalpha():
        return True
    if re.fullmatch(r"[a-z]+(?:_[a-z]+)+", lowered):
        return True
    return any(token in lowered for token in _SCREEN_SCHEMA_SUBSTRINGS)


def _parsing_quality_score(answer: str) -> int:
    parsed = parse_parsing_answer(answer)
    if parsed["parse_ok"]:
        return int(parsed["element_count"]) * 2
    payload = _normalize_parsing_payload(answer)
    elements = payload.get("elements", [])
    return len(elements) * 2


def _looks_degenerate_screen_output(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    punctuation_ratio = sum(1 for char in stripped if not char.isalnum() and not char.isspace()) / max(len(stripped), 1)
    if punctuation_ratio > 0.6:
        return True
    if re.fullmatch(r"[!?.`~_\-=\s]+", stripped):
        return True
    return False


def _build_parsing_retry_prompt(sample: VlmSample) -> str:
    query = sample.query or "List all visible interactive UI elements on this screen."
    return (
        "You are parsing a GUI screenshot for a small student model distillation dataset.\n"
        f"Task: {query}\n"
        "Return one UI element per line using exactly this format:\n"
        "<label> | <x1>,<y1>,<x2>,<y2> | <focused>\n"
        "Rules:\n"
        "- Do not output JSON.\n"
        "- Do not use markdown.\n"
        "- Do not add explanations.\n"
        "- Use pixel coordinates in the original image.\n"
        "- x1,y1 is the top-left corner.\n"
        "- x2,y2 is the bottom-right corner.\n"
        "- focused must be true or false.\n"
        "Example:\n"
        "Picture | 145,238,276,292 | false\n"
        "General | 145,348,276,404 | true\n"
        "Network Settings | 705,396,807,432 | false"
    )



STRICT_TEACHER_LABEL_KEYS = {
    "id",
    "image",
    "task",
    "query",
    "teacher_answer",
    "teacher_tokens",
    "teacher_element_count",
}


@dataclass(frozen=True)
class CompletedTeacherRows:
    ids: set[str]
    valid_count: int
    invalid_count: int
    first_invalid_keys: list[str] | None


def create_teacher_precompute_dataset(config: PipelineConfig, samples: list[VlmSample] | None = None) -> Path:
    samples = samples or validate_manifest(
        resolve_training_manifest_path(config.data),
        image_root=config.data.image_root,
        max_samples=config.data.max_samples,
    )
    output_path = resolve_label_path(config.data)
    _warn_offline_teacher_logits_disabled(config)
    completed = _load_completed_teacher_rows(output_path, config=config)
    if completed.invalid_count:
        _rewrite_valid_teacher_rows(output_path, config=config)

    pending_samples = [
        sample
        for sample in samples
        if sample.id not in completed.ids
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Teacher precompute:")
    print(f"  output: {output_path}")
    print("  output_schema: id,image,task,query,teacher_answer,teacher_tokens,teacher_element_count")
    print("  offline_teacher_logits: disabled")
    print("  note: online DBiLD computes teacher/student logits during training")
    print(f"  total samples: {len(samples)}")
    print(f"  valid completed label rows: {completed.valid_count}")
    print(f"  invalid stale label rows: {completed.invalid_count}")
    print(f"  pending rows: {len(pending_samples)}")
    if completed.first_invalid_keys:
        print(f"  first invalid label row id/reason: {completed.first_invalid_keys}")

    if not pending_samples:
        refresh_parsing_sidecar_reports(output_root=output_path.parent, role="teacher")
        return output_path

    teacher = build_teacher(config)

    completed_now = 0
    with output_path.open("a", encoding="utf-8") as label_handle:
        for sample in pending_samples:
            started = time.perf_counter()
            generated = _generate_label_row(teacher, sample)
            row = _build_teacher_output_row(sample, generated)
            attach_parsing_sidecar_outputs(
                row,
                output_root=output_path.parent,
                role="teacher",
                answer_field="teacher_answer",
            )
            label_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            label_handle.flush()

            completed_now += 1
            elapsed = time.perf_counter() - started
            print(
                "[teacher-precompute] "
                f"total={len(samples)} completed={len(samples) - (len(pending_samples) - completed_now)} "
                f"pending={len(pending_samples) - completed_now} id={sample.id} "
                f"label_written=True elapsed_seconds_per_sample={elapsed:.2f}"
            )
    refresh_parsing_sidecar_reports(output_root=output_path.parent, role="teacher")
    return output_path


def create_distillation_dataset(config: PipelineConfig, samples: list[VlmSample]) -> Path:
    return create_teacher_precompute_dataset(config, samples)


def _base_output_row(sample: VlmSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "image": sample.image,
        "task": sample.task,
        "query": sample.query,
    }


def _build_teacher_output_row(sample: VlmSample, generated: dict[str, Any]) -> dict[str, Any]:
    return {
        **_base_output_row(sample),
        "teacher_answer": str(generated["teacher_answer"]),
        "teacher_tokens": [int(token_id) for token_id in generated.get("teacher_tokens", [])],
    }


def _generate_label_row(
    teacher: Any,
    sample: VlmSample,
) -> dict[str, Any]:
    label = teacher.answer(sample)
    answer = _normalize_teacher_answer(sample, str(label["teacher_answer"])).strip()
    tokenizer = getattr(teacher, "tokenize_teacher_answer", None)
    if callable(tokenizer):
        tokens = [int(token_id) for token_id in tokenizer(answer)]
    else:
        tokens = [ord(char) for char in answer]
    row = {
        "teacher_answer": answer,
        "teacher_tokens": tokens,
        "teacher_confidence": float(label.get("teacher_confidence", 1.0)),
        "teacher_rationale": label.get("teacher_rationale", "Generated by teacher backend."),
    }
    return row


def _canonicalize_line_teacher_answer(answer: str) -> str | None:
    parsed = parse_parsing_answer(answer)
    if not parsed["parse_ok"]:
        return None
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        return None
    canonical = elements_to_line_format(
        [element for element in elements if isinstance(element, dict)]
    )
    return canonical or None


def _build_teacher_forcing_inputs_and_answer_span(processor, image, prompt: str, teacher_answer: str):
    from .chat_spans import build_vlm_chat_answer_span

    span = build_vlm_chat_answer_span(processor, image, prompt, teacher_answer)
    return (
        span.prompt_inputs,
        span.full_inputs,
        span.prompt_input_ids,
        span.full_input_ids,
        span.prompt_token_len,
        span.assistant_tail_ids,
        span.answer_token_ids,
    )


def _extract_answer_token_ids_from_full_input(processor, image, prompt: str, teacher_answer: str) -> list[int]:
    _, _, _, _, _, _, answer_token_ids = _build_teacher_forcing_inputs_and_answer_span(
        processor,
        image,
        prompt,
        teacher_answer,
    )
    return answer_token_ids


def _decode_teacher_tokens(processor, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    tokenizer = getattr(processor, "tokenizer", None)
    decoder = tokenizer if tokenizer is not None else processor
    return decoder.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

def _load_completed_teacher_rows(
    path: Path,
    *,
    config: PipelineConfig,
    require_teacher_element_count: bool = False,
) -> CompletedTeacherRows:
    from .teacher_validation import build_teacher_token_decoder
    if not path.exists():
        return CompletedTeacherRows(ids=set(), valid_count=0, invalid_count=0, first_invalid_keys=None)
    completed_ids: set[str] = set()
    valid_count = 0
    invalid_count = 0
    first_invalid: list[str] | None = None
    decoder = build_teacher_token_decoder(config)
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is None:
            continue
        valid, reason = _validate_teacher_label_row(
            row,
            decode_tokens=decoder,
            require_teacher_element_count=require_teacher_element_count,
        )
        if valid:
            completed_ids.add(str(sample_id))
            valid_count += 1
        else:
            invalid_count += 1
            if first_invalid is None:
                first_invalid = [str(sample_id), str(reason)]
    return CompletedTeacherRows(
        ids=completed_ids,
        valid_count=valid_count,
        invalid_count=invalid_count,
        first_invalid_keys=first_invalid,
    )


def _rewrite_valid_teacher_rows(path: Path, *, config: PipelineConfig, require_teacher_element_count: bool = False) -> None:
    from .teacher_validation import build_teacher_token_decoder
    decoder = build_teacher_token_decoder(config)
    valid_rows = [
        row for row in read_jsonl(path)
        if _validate_teacher_label_row(
            row,
            decode_tokens=decoder,
            require_teacher_element_count=require_teacher_element_count,
        )[0]
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[teacher-precompute] pruned invalid existing rows from {path}; remaining_valid_rows={len(valid_rows)}")


def _validate_teacher_label_row(
    row: dict[str, Any],
    *,
    decode_tokens,
    require_teacher_element_count: bool = False,
) -> tuple[bool, str | None]:
    from .teacher_validation import validate_teacher_row

    valid, reason = validate_teacher_row(
        row,
        decode_tokens=decode_tokens,
    )
    if not valid:
        return valid, reason

    if str(row.get("task") or "").strip() == "parsing":
        row_keys = set(row.keys())
        if row_keys != STRICT_TEACHER_LABEL_KEYS:
            return (
                False,
                "teacher label row keys do not match strict parsing schema: "
                f"expected={sorted(STRICT_TEACHER_LABEL_KEYS)} actual={sorted(row_keys)}",
            )
        if require_teacher_element_count and row.get("teacher_element_count") is None:
            return False, "teacher_element_count is missing"

    return True, None


def _warn_offline_teacher_logits_disabled(config: PipelineConfig) -> None:
    has_deprecated_config = any(
        (
            getattr(config.data, "teacher_logits_path", None) is not None,
            getattr(config.data, "switch_logits_path", None) is not None,
            bool(getattr(config.distillation, "teacher_logits", False)),
        )
    )
    if has_deprecated_config:
        print(
            "Warning: offline teacher logits config is deprecated and ignored. "
            "Online DBiLD computes logits during training."
        )


def _format_prompt(config: PipelineConfig, sample: VlmSample) -> str:
    template = config.distillation.prompt_template

    try:
        prompt = template.format(
            query=sample.query or "",
            question=sample.query or "",
            task=sample.task,
        )
    except KeyError as exc:
        raise KeyError(
            f"Prompt template references unsupported placeholder: {exc}. "
            "Supported placeholders are: query, question, task."
        ) from exc

    if sample.task != "parsing":
        return prompt

    return (
        f"{prompt.rstrip()}\n\n"
        "Return one UI element per line using exactly this format:\n"
        "<label> | <x1>,<y1>,<x2>,<y2> | <focused>\n\n"
        "Rules:\n"
        "- Do not output JSON.\n"
        "- Do not use markdown.\n"
        "- Do not add explanations.\n"
        "- Use pixel coordinates in the original image.\n"
        "- x1,y1 is the top-left corner.\n"
        "- x2,y2 is the bottom-right corner.\n"
        "- focused must be true or false.\n\n"
        "Example:\n"
        "Picture | 145,238,276,292 | false\n"
        "General | 145,348,276,404 | true\n"
        "Network Settings | 705,396,807,432 | false"
    )


def _mock_answer(sample: VlmSample) -> str:
    if sample.answer:
        return sample.answer

    if sample.task == "parsing":
        return json.dumps(
            {
                "focused_element": "mock settings",
                "elements": [
                    {
                        "label": "mock icon",
                        "type": "app_icon",
                        "bbox": [0, 0, 100, 100],
                    },
                    {
                        "label": "mock settings",
                        "type": "button",
                        "bbox": [120, 0, 220, 100],
                    },
                ],
            },
            ensure_ascii=False,
        )

    return f"mock answer for {sample.task}"
