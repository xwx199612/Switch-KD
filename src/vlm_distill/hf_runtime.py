from __future__ import annotations

import os


def configure_hf_offline_mode() -> None:
    """Prefer local Hugging Face caches and suppress remote metadata retries."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

