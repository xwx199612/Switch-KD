import io
from types import SimpleNamespace

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vlm_distill import docker_service


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return output.getvalue()


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
    with TestClient(docker_service.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["precision_summary"]["linear4bit_module_count"] == 1


def test_ready_returns_503_before_model_loaded():
    docker_service.app.state.ready = False
    with TestClient(docker_service.app) as client:
        # lifespan loads the model in normal operation; explicitly emulate a
        # failed/unloaded state for this endpoint contract.
        docker_service.app.state.ready = False
        assert client.get("/ready").status_code == 503


def test_infer_uses_formatter_parser_and_lock(monkeypatch):
    fake_config, fake_engine, summary = _runtime()
    monkeypatch.setattr(docker_service, "_runtime_load", lambda: (fake_config, fake_engine, summary))
    with TestClient(docker_service.app) as client:
        response = client.post("/infer", files={"image": ("x.png", _png(), "image/png")},
                               data={"query": "find buttons", "request_id": "abc"})
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "abc"
        assert body["usable"] is True
        assert body["elements"][0]["text"] == "Button"
        assert fake_engine.prompts == [("Query: find buttons\nAnswer:", 17, (8, 8))]


def test_infer_rejects_invalid_image(monkeypatch):
    monkeypatch.setattr(docker_service, "_runtime_load", _runtime)
    with TestClient(docker_service.app) as client:
        response = client.post("/infer", files={"image": ("x.txt", b"not an image", "text/plain")})
        assert response.status_code == 400
