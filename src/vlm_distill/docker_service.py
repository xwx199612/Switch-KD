"""Single-model, single-GPU FastAPI inference service."""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .bbox_grounding_inference import BBoxGroundingInferenceEngine
from .config_schema import load_config
from .data_manifest import VlmSample
from .parsing_output_parser import COORDINATE_SYSTEM_NORMALIZED_0_1000, parse_parsing_answer
from .runtime_validation import summarize_model_precision, validate_loaded_precision
from .stage_teacher_precompute import _format_prompt, _load_teacher_image

DEFAULT_QUERY = "List all visible interactive UI elements on this screen."
inference_lock = threading.Lock()


def _runtime_load() -> tuple[Any, BBoxGroundingInferenceEngine, dict[str, Any]]:
    config_path = Path(os.environ["VLM_CONFIG_PATH"])
    config = load_config(config_path)
    engine = BBoxGroundingInferenceEngine.from_pipeline_config(config)
    summary = summarize_model_precision(engine.model)
    validate_loaded_precision(config, summary)
    return config, engine, summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    config, engine, summary = _runtime_load()
    app.state.config = config
    app.state.engine = engine
    app.state.precision_summary = summary
    app.state.config_path = str(Path(os.environ["VLM_CONFIG_PATH"]))
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(title="Switch-KD mixed-precision inference", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    config = app.state.config
    return {"status": "ready", "config_path": app.state.config_path,
            "model_path": app.state.engine.model_path,
            "student_quantization": config.student.quantization,
            "precision_summary": app.state.precision_summary}


def _infer_sync(image_bytes: bytes, query: str) -> dict[str, Any]:
    config = app.state.config
    engine = app.state.engine
    with tempfile.NamedTemporaryFile(suffix=".image") as handle:
        handle.write(image_bytes)
        handle.flush()
        image = _load_teacher_image(Path(handle.name), config.training.image_resize)
    sample = VlmSample(id="request", image="", task="parsing", query=query)
    prompt = _format_prompt(config, sample)
    raw_output = engine.generate_raw(image, prompt, config.teacher.max_new_tokens)
    parsed = parse_parsing_answer(raw_output)
    return {"raw_output": raw_output, "usable": bool(parsed.get("usable")),
            "parse_error": parsed.get("parse_error"), "elements": parsed.get("elements", []),
            "coordinate_system": COORDINATE_SYSTEM_NORMALIZED_0_1000}


@app.post("/infer")
async def infer(image: UploadFile = File(...), query: str = Form(DEFAULT_QUERY),
                request_id: str | None = Form(None)) -> dict[str, Any]:
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    started = time.perf_counter()
    try:
        content = await image.read()
        with Image.open(__import__("io").BytesIO(content)) as checked:
            checked.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    try:
        with inference_lock:
            result = await asyncio.to_thread(_infer_sync, content, query or DEFAULT_QUERY)
    except HTTPException:
        raise
    except Exception as exc:  # no traceback is returned to clients
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    result.update({"id": request_id or "request-id", "task": "parsing", "query": query or DEFAULT_QUERY,
                   "elapsed_seconds": round(time.perf_counter() - started, 3)})
    return result
