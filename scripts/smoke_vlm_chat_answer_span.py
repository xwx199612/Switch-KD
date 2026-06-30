from __future__ import annotations

from PIL import Image
import torch

from vlm_distill.chat_spans import build_vlm_chat_answer_span
from vlm_distill.vlm_batching import build_vlm_full_answer_span_inputs


PROMPT = "List all visible interactive UI elements..."
ANSWER = '{"elements":[{"text":"Search","type":"input","focused":false}]}'
SPECIAL_TOKENS = {
    "<user>": 100001,
    "</user>": 100002,
    "<assistant>": 100003,
    "</assistant>": 100004,
    "<image>": 100005,
}


class FakeTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in SPECIAL_TOKENS.values():
                continue
            chars.append(chr(token_id))
        return "".join(chars)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            content_parts: list[str] = []
            for item in message["content"]:
                if item["type"] == "image":
                    content_parts.append("<image>")
                elif item["type"] == "text":
                    content_parts.append(item["text"])
            parts.append(f"<{role}>{''.join(content_parts)}</{role}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, *, images, text, return_tensors="pt", truncation=None, max_length=None):
        encoded = _encode_text(text[0])
        if truncation and max_length is not None:
            encoded = encoded[:max_length]
        input_ids = torch.tensor([encoded], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _encode_text(text: str) -> list[int]:
    encoded: list[int] = []
    cursor = 0
    while cursor < len(text):
        matched = False
        for token_text, token_id in SPECIAL_TOKENS.items():
            if text.startswith(token_text, cursor):
                encoded.append(token_id)
                cursor += len(token_text)
                matched = True
                break
        if matched:
            continue
        encoded.append(ord(text[cursor]))
        cursor += 1
    return encoded


def main() -> None:
    processor = FakeProcessor()
    image = Image.new("RGB", (1, 1), color="white")

    span = build_vlm_chat_answer_span(processor, image, PROMPT, ANSWER)
    if "<assistant>" not in span.prompt_text:
        raise AssertionError("Prompt text is missing the assistant generation marker.")
    if f"<user><image>{PROMPT}{ANSWER}</user>" in span.full_text:
        raise AssertionError("Answer was embedded inside the user message.")
    if f"<assistant>{ANSWER}</assistant>" not in span.full_text:
        raise AssertionError("Answer was not placed in the assistant turn.")

    encoded = build_vlm_full_answer_span_inputs(
        processor,
        image=image,
        prompt=PROMPT,
        target=ANSWER,
        max_length=4096,
        mask_prompt_labels=True,
    )
    labels = encoded.model_inputs["labels"]
    supervised_label_ids = [int(token_id) for token_id in labels[labels != -100].tolist()]
    decoded = processor.tokenizer.decode(supervised_label_ids)
    if decoded != ANSWER:
        raise AssertionError(f"Decoded supervised span mismatch: {decoded!r}")

    print("smoke_ok")
    print(f"prompt_text={span.prompt_text}")
    print(f"full_text={span.full_text}")
    print(f"decoded_supervised={decoded}")


if __name__ == "__main__":
    main()
