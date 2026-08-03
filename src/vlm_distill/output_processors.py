from __future__ import annotations

from typing import Any, Protocol

from .data_manifest import VlmSample
from .parsing_output_parser import parse_parsing_answer


class OutputProcessor(Protocol):
    mode: str

    def process(
        self,
        *,
        sample: VlmSample,
        raw_output: str,
        backend_result: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class GenericTextOutputProcessor:
    mode = "text"

    def process(
        self,
        *,
        sample: VlmSample,
        raw_output: str,
        backend_result: dict[str, Any],
    ) -> dict[str, Any]:
        del sample
        result: dict[str, Any] = {
            "usable": True,
            "raw_model_output": raw_output.strip(),
            "answer": raw_output.strip(),
            "student_answer": raw_output.strip(),
        }
        if backend_result.get("inference_debug"):
            result["inference_debug"] = backend_result["inference_debug"]
        return result


class ParsingOutputProcessor:
    mode = "parsing"

    def process(
        self,
        *,
        sample: VlmSample,
        raw_output: str,
        backend_result: dict[str, Any],
    ) -> dict[str, Any]:
        del sample
        result = {
            "raw_model_output": raw_output,
            **parse_parsing_answer(raw_output),
        }
        if backend_result.get("inference_debug"):
            result["inference_debug"] = backend_result["inference_debug"]
        return result


def build_output_processor(mode: str) -> OutputProcessor:
    normalized = str(mode).strip().lower()
    if normalized == "text":
        return GenericTextOutputProcessor()
    if normalized == "parsing":
        return ParsingOutputProcessor()
    raise ValueError(f"Unsupported output mode: {mode!r}; expected 'text' or 'parsing'.")
