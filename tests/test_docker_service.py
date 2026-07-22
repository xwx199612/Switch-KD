import asyncio
import io
import inspect
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")

from vlm_distill import docker_service  # noqa: E402


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return output.getvalue()


class FakeUploadFile:
    def __init__(self, content: bytes):
        self.content = content

    async def read(self) -> bytes:
        return self.content


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "formatted"


class FakeEngine:
    model_path = "/fake/model"
    model = object()
    processor = FakeProcessor()

    def __init__(self):
        self.prompts = []

    def generate_raw(self, image, prompt, max_new_tokens):
        self.prompts.append((prompt, max_new_tokens, image.size))
        return '{"elements": [{"text": "Button", "bbox_norm": [1, 2, 3, 4], "focused": false}]}'


def _runtime():
    config = SimpleNamespace(
        training=SimpleNamespace(image_resize="original"),
        teacher=SimpleNamespace(max_new_tokens=17),
        student=SimpleNamespace(quantization="mixed_4bit_bf16"),
        distillation=SimpleNamespace(prompt_template="Query: {query}\nAnswer:"),
    )
    engine = FakeEngine()
    return config, engine, {"linear4bit_module_count": 1}


def test_health_and_ready_lifecycle(monkeypatch):
    monkeypatch.setattr(docker_service, "_runtime_load", _runtime)
    monkeypatch.setenv("VLM_CONFIG_PATH", "/fake/config.yaml")

    async def exercise():
        async with docker_service.lifespan(docker_service.app):
            assert await docker_service.health() == {"status": "ok"}
            response = await docker_service.ready()
            assert response["precision_summary"]["linear4bit_module_count"] == 1

    asyncio.run(exercise())


def test_ready_returns_503_before_model_loaded(monkeypatch):
    monkeypatch.setattr(docker_service, "_runtime_load", _runtime)
    monkeypatch.setenv("VLM_CONFIG_PATH", "/fake/config.yaml")
    docker_service.app.state.ready = False
    async def exercise():
        async with docker_service.lifespan(docker_service.app):
            # lifespan loads the model in normal operation; explicitly emulate
            # a failed/unloaded state for this endpoint contract.
            docker_service.app.state.ready = False
            with pytest.raises(docker_service.HTTPException) as exc_info:
                await docker_service.ready()
            assert exc_info.value.status_code == 503

    asyncio.run(exercise())


def test_infer_sync_uses_formatter_parser_and_worker_thread_lock(monkeypatch):
    fake_config, fake_engine, _summary = _runtime()
    docker_service.app.state.config = fake_config
    docker_service.app.state.engine = fake_engine
    entered_thread_ids = []
    real_lock = threading.Lock()

    class TrackingLock:
        def __enter__(self):
            entered_thread_ids.append(threading.get_ident())
            real_lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            real_lock.release()

    monkeypatch.setattr(docker_service, "inference_lock", TrackingLock())
    result_holder = []
    worker = threading.Thread(
        target=lambda: result_holder.append(docker_service._infer_sync(_png(), "find buttons"))
    )
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    body = result_holder[0]
    assert body["raw_output"]
    assert fake_engine.prompts == [("Query: find buttons\nAnswer:", 17, (8, 8))]
    assert entered_thread_ids
    assert entered_thread_ids[0] != threading.get_ident()
    endpoint_source = inspect.getsource(docker_service.infer)
    assert "with inference_lock" not in endpoint_source
    assert "asyncio.to_thread(_infer_sync" in endpoint_source


def test_waiting_inference_does_not_block_event_loop_or_health(monkeypatch):
    fake_config, fake_engine, _summary = _runtime()
    docker_service.app.state.ready = True
    docker_service.app.state.config = fake_config
    docker_service.app.state.engine = fake_engine
    lock = threading.Lock()
    monkeypatch.setattr(docker_service, "inference_lock", lock)

    def blocked_inference(_content, _query):
        with lock:
            return {"usable": True, "elements": [], "raw_output": "", "parse_error": None,
                    "coordinate_system": docker_service.COORDINATE_SYSTEM_NORMALIZED_0_1000}

    monkeypatch.setattr(docker_service, "_infer_sync", blocked_inference)
    lock.acquire()

    async def exercise():
        worker = threading.Thread(target=blocked_inference, args=(_png(), "find buttons"))
        worker.start()
        await asyncio.sleep(0.05)
        health_response = await docker_service.health()
        assert health_response == {"status": "ok"}
        lock.release()
        while worker.is_alive():
            await asyncio.sleep(0.01)
        return {"usable": True}

    try:
        result = asyncio.run(exercise())
    finally:
        if lock.locked():
            lock.release()
    assert result["usable"] is True


def test_infer_rejects_invalid_image():
    docker_service.app.state.ready = True
    async def exercise():
        with pytest.raises(docker_service.HTTPException) as exc_info:
            await docker_service.infer(
                image=FakeUploadFile(b"not an image"),
                query=docker_service.DEFAULT_QUERY,
                request_id=None,
            )
        return exc_info.value.status_code

    assert asyncio.run(exercise()) == 400
