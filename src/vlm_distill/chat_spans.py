from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class VlmChatAnswerSpan:
    prompt_text: str
    full_text: str
    prompt_inputs: dict[str, Any]
    full_inputs: dict[str, Any]
    prompt_input_ids: list[int]
    full_input_ids: list[int]
    prompt_token_len: int
    assistant_tail_ids: list[int]
    answer_token_ids: list[int]


def build_vlm_chat_answer_span(
    processor,
    image: Image.Image,
    prompt: str,
    answer: str,
    *,
    return_tensors: str = "pt",
    truncation: bool | None = None,
    max_length: int | None = None,
) -> VlmChatAnswerSpan:
    prompt = prompt.strip()
    answer = answer.strip()

    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    full_messages = [
        *prompt_messages,
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]

    prompt_text = _apply_chat_template(
        processor,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_text = _apply_chat_template(
        processor,
        full_messages,
        add_generation_prompt=False,
    )

    processor_kwargs: dict[str, Any] = {"return_tensors": return_tensors}
    if truncation is not None:
        processor_kwargs["truncation"] = truncation
    if max_length is not None:
        processor_kwargs["max_length"] = max_length

    prompt_inputs = _processor_call(processor, image=image, text=prompt_text, **processor_kwargs)
    full_inputs = _processor_call(processor, image=image, text=full_text, **processor_kwargs)

    prompt_input_ids = [int(token_id) for token_id in prompt_inputs["input_ids"][0].tolist()]
    full_input_ids = [int(token_id) for token_id in full_inputs["input_ids"][0].tolist()]
    prompt_token_len = len(prompt_input_ids)
    if full_input_ids[:prompt_token_len] != prompt_input_ids:
        raise ValueError("Full chat input does not preserve the prompt-only token prefix.")

    assistant_tail_ids = full_input_ids[prompt_token_len:]
    raw_answer_token_ids = _tokenize_answer_without_special_tokens(processor, answer)
    answer_token_ids = [int(token_id) for token_id in raw_answer_token_ids]
    if assistant_tail_ids[: len(answer_token_ids)] != answer_token_ids:
        raise ValueError(
            "Canonical assistant answer token span mismatch. "
            f"{_build_assistant_tail_debug_details(processor, answer_token_ids, assistant_tail_ids)}"
        )
    return VlmChatAnswerSpan(
        prompt_text=prompt_text,
        full_text=full_text,
        prompt_inputs=prompt_inputs,
        full_inputs=full_inputs,
        prompt_input_ids=prompt_input_ids,
        full_input_ids=full_input_ids,
        prompt_token_len=prompt_token_len,
        assistant_tail_ids=assistant_tail_ids,
        answer_token_ids=answer_token_ids,
    )


def _apply_chat_template(processor, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        if len(messages) == 1:
            return str(messages[0]["content"][1]["text"])
        return f"{messages[0]['content'][1]['text']}{messages[1]['content'][0]['text']}"
    return apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _processor_call(processor, *, image: Image.Image, text: str, **kwargs):
    try:
        return processor(images=[image], text=[text], **kwargs)
    except TypeError:
        return processor(text=[text], images=[image], **kwargs)


def _tokenize_answer_without_special_tokens(processor, answer: str) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    candidates = [tokenizer, processor] if tokenizer is not None else [processor]
    for candidate in candidates:
        if callable(candidate):
            try:
                encoded = candidate(answer, add_special_tokens=False)
            except TypeError:
                encoded = candidate(answer)
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return [int(token_id) for token_id in input_ids]
    raise TypeError("Processor/tokenizer does not support direct answer tokenization.")


def _decode_token_ids(processor, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    if hasattr(processor, "decode"):
        return processor.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            [token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
    return ""


def _build_assistant_tail_debug_details(processor, raw_answer_token_ids: list[int], assistant_tail_ids: list[int]) -> str:
    answer_len = len(raw_answer_token_ids)
    assistant_tail_head = assistant_tail_ids[:answer_len]
    trailing_tail_ids = assistant_tail_ids[answer_len:]
    details = {
        "decoded_raw_answer_token_ids": repr(_decode_token_ids(processor, raw_answer_token_ids)),
        "decoded_assistant_tail_ids_head": repr(_decode_token_ids(processor, assistant_tail_head)),
        "first_20_raw_answer_token_ids": raw_answer_token_ids[:20],
        "first_20_assistant_tail_ids": assistant_tail_ids[:20],
        "extra_trailing_assistant_tail_ids_after_raw_answer": trailing_tail_ids,
    }
    return ", ".join(f"{key}={value}" for key, value in details.items())
