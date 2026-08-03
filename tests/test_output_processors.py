from __future__ import annotations

from types import SimpleNamespace

from vlm_distill.data_manifest import VlmSample
from vlm_distill.output_processors import (
    GenericTextOutputProcessor,
    ParsingOutputProcessor,
    build_output_processor,
)
from vlm_distill.stage_teacher_precompute import _format_prompt


def _samples() -> list[VlmSample]:
    return [
        VlmSample(id="none", image="screen.png", query="List elements."),
        VlmSample(id="parsing", image="screen.png", query="List elements."),
        VlmSample(id="qa", image="screen.png", query="List elements."),
    ]


def test_generic_processor_preserves_plain_text_without_parsing(monkeypatch) -> None:
    monkeypatch.setattr(
        "vlm_distill.output_processors.parse_parsing_answer",
        lambda _raw: (_ for _ in ()).throw(AssertionError("generic mode parsed text")),
    )
    processor = GenericTextOutputProcessor()
    results = [
        processor.process(sample=sample, raw_output="  plain answer  ", backend_result={})
        for sample in _samples()
    ]

    assert all(result["answer"] == "plain answer" for result in results)
    assert all("elements" not in result for result in results)


def test_parsing_processor_is_independent_of_task_metadata() -> None:
    processor = ParsingOutputProcessor()
    raw = '{"elements":[{"text":"Home","bbox_norm":[0,0,10,10],"focused":true}]}'
    results = [
        processor.process(sample=sample, raw_output=raw, backend_result={})
        for sample in _samples()
    ]

    assert [result["elements"] for result in results] == [
        [{"text": "Home", "bbox_norm": [0, 0, 10, 10], "focused": True}]
    ] * 3
    assert all(result["parse_ok"] is True for result in results)


def test_output_processor_factory_rejects_unknown_mode() -> None:
    assert build_output_processor("text").mode == "text"
    assert build_output_processor("parsing").mode == "parsing"


def test_prompt_uses_output_mode_not_task_metadata() -> None:
    config = SimpleNamespace(
        pipeline=SimpleNamespace(output_mode="parsing"),
        distillation=SimpleNamespace(prompt_template="{output_mode}\n{query}"),
    )
    prompts = [_format_prompt(config, sample) for sample in _samples()]

    assert prompts == ["parsing\nList elements."] * 3
