from __future__ import annotations

import importlib.util


_ATTN_ALIASES = {
    "sdpa": "sdpa",
    "eager": "eager",
    "fa2": "flash_attention_2",
    "flash": "flash_attention_2",
    "flash2": "flash_attention_2",
    "flash_attn_2": "flash_attention_2",
    "flash-attn-2": "flash_attention_2",
    "flash_attention_2": "flash_attention_2",
}


def resolve_attn_implementation(value: str | None) -> str:
    raw = (value or "sdpa").strip().lower()
    if raw not in _ATTN_ALIASES:
        raise ValueError(
            f"Unsupported attn_implementation={value!r}. "
            "Use one of: sdpa, eager, flash_attention_2."
        )

    resolved = _ATTN_ALIASES[raw]
    if resolved == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError(
            "attn_implementation='flash_attention_2' requires the `flash_attn` package "
            "to be installed in the active environment."
        )
    return resolved


def apply_attn_implementation(model_kwargs: dict, value: str | None) -> dict:
    model_kwargs["attn_implementation"] = resolve_attn_implementation(value)
    return model_kwargs
