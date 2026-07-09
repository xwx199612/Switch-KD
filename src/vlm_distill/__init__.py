"""VLM distillation pipeline."""

from __future__ import annotations

import importlib

__all__ = ["__version__"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(name) from exc
    globals()[name] = module
    return module
