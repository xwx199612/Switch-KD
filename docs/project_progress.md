# Project Progress

`tools/draw_lm_bboxes.py` now supports three bbox coordinate systems via `--coord-system`:

- `pixel`
- `normalized-1000`
- `auto`

Some Qwen-VL-style grounding outputs use normalized `0-1000` coordinates instead of original image pixels. If boxes appear too far left/up or too small, use `--coord-system auto` or `--coord-system normalized-1000`.

## Three-model comparison scripts

Two server-side comparison scripts are available for running the same image folder through three VLM checkpoints:

1. `scripts/compare_vlm_object_listing.py`
2. `scripts/compare_vlm_bbox_grounding.py`

The object-listing script compares recognition and parsing ability by asking each model to return visible object or UI-element names, saving raw outputs per image, and writing parsed text and JSON artifacts per image.

The bbox-grounding script compares visual localization ability by asking each model to return labeled elements with bounding boxes, saving raw and parsed outputs for debugging, and writing annotated image copies per model. It supports Qwen-VL-style normalized `0-1000` coordinates via `--coord-system normalized-1000` and related coordinate handling in the annotation path.

Both scripts now also support `--quantization {none,4bit,8bit}`. Quantized model loading uses `BitsAndBytesConfig`, keeps the existing one-model-at-a-time execution flow, and still calls `cleanup_model()` after each model finishes so VRAM can be released before the next checkpoint loads.

Normal inference in both comparison scripts now defaults to parse-error-resistant line-based model output instead of model-generated JSON. Raw model outputs are still saved unchanged per image, while parsed `.json` artifacts are produced by script-side line parsers for object names and bbox elements. Malformed lines are skipped instead of failing the whole image, and legacy JSON parsing is only used when explicitly requested via `--output-format json`.
