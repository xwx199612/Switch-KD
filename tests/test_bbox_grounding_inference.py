from types import SimpleNamespace
import sys

import torch
from PIL import Image

from vlm_distill.bbox_grounding_inference import BBoxGroundingInferenceEngine


class FakeProcessor:
    def __init__(self, reject_videos=False):
        self.reject_videos = reject_videos
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"][0]["image"] is IMAGE
        return "CHAT:" + messages[0]["content"][1]["text"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_videos and "videos" in kwargs:
            raise TypeError("videos unsupported")
        return {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones((1, 3))}

    def batch_decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return ["  {\"elements\": []}  "]


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 9, 10]])


IMAGE = Image.new("RGB", (4, 4))


def test_engine_uses_process_vision_info_and_suffix_and_greedy(monkeypatch):
    processor, model = FakeProcessor(), FakeModel()
    vision = SimpleNamespace(process_vision_info=lambda messages: (["vision"], ["video"]))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", vision)
    engine = BBoxGroundingInferenceEngine(model, processor)

    assert engine.generate_raw(IMAGE, "prompt", 17) == '{"elements": []}'
    assert processor.calls[0]["images"] == ["vision"]
    assert processor.calls[0]["videos"] == ["video"]
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 17


def test_engine_falls_back_without_qwen_utils(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)
    processor, model = FakeProcessor(reject_videos=True), FakeModel()
    assert BBoxGroundingInferenceEngine(model, processor).generate_raw(IMAGE, "p", 2)
    assert processor.calls[0]["images"] == [IMAGE]


def test_debug_parity_metadata_is_populated():
    processor, model = FakeProcessor(), FakeModel()
    engine = BBoxGroundingInferenceEngine(model, processor, debug_inference_parity=True)
    engine.generate_raw(IMAGE, "prompt", 3)
    assert engine.last_debug["input_ids_shape"] == [1, 2]
    assert engine.last_debug["generated_token_ids_hash"]
    assert engine.last_debug["raw_output_hash"]


def test_bundle_path_routes_to_high_fidelity_loader(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "deployment_config.json").write_text('{"artifact_mode":"4bit_base_bf16_adapter"}')
    expected = (object(), object())
    monkeypatch.setattr(
        "vlm_distill.bbox_grounding_inference.load_high_fidelity_adapter_deployment",
        lambda path: expected,
    )
    engine = BBoxGroundingInferenceEngine.from_cli_args(model_path=bundle)
    assert (engine.model, engine.processor) == expected


def test_normal_model_does_not_route_to_bundle_loader(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    monkeypatch.setattr(BBoxGroundingInferenceEngine, "_load", classmethod(
        lambda cls, **kwargs: cls("model", "processor")))
    engine = BBoxGroundingInferenceEngine.from_cli_args(model_path=model_path)
    assert engine.model == "model"
