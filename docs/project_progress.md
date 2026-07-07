# Project Progress

## Current milestone

The project is currently focused on the Android TV / GUI screen understanding workflow:

```text
Screen Parsing -> UI Element Grounding -> BBox Visualization -> Student Distillation / Deployment
```

## Latest update: LM bbox annotation visualization tool

A new utility has been added for visual inspection of LM-generated UI bounding boxes:

```text
tools/draw_lm_bboxes.py
```

This tool is used to draw predicted bounding boxes on a copy of the original image without modifying the source image.

## Tool purpose

The tool supports LM outputs that contain an `elements` list, where each element may include:

```json
{
  "text": "General",
  "type": "menu_item",
  "bbox": [145, 358, 276, 412],
  "focused": true,
  "confidence": 0.98
}
```

For visualization, only the following fields are used:

```text
text
bbox
focused
```

The following fields are intentionally ignored and not drawn on the image:

```text
type
confidence
```

## Expected behavior

- The original image is never modified.
- The output image is saved as a separate annotated copy.
- Each valid bbox is drawn on the image.
- Each label shows only the UI element text.
- `focused=true` elements are drawn with a distinct highlight color.
- `focused=true` labels append a visible focus marker.
- Malformed or invalid bbox entries are skipped safely.
- LM outputs with non-JSON text before the JSON object are supported.

## Example command

```bash
python tools/draw_lm_bboxes.py \
  --image /mnt/nvme0/vlm_distill/test_images/tv_settings.jpg \
  --lm-output /mnt/nvme0/vlm_distill/test_outputs/qwen3vl_bbox.txt \
  --output /mnt/nvme0/vlm_distill/test_outputs/tv_settings_bbox_annotated.jpg
```

Optional arguments:

```bash
--font-size 18
--line-width 3
--font /path/to/font.ttf
```

## Evaluation summary

The current implementation satisfies the requested annotation requirements:

| Requirement | Status |
| --- | --- |
| Parse LM output with text before JSON | Done |
| Use only `text`, `bbox`, and `focused` | Done |
| Ignore `type` | Done |
| Ignore `confidence` | Done |
| Preserve focused information | Done |
| Use special color for focused elements | Done |
| Do not modify original image | Done |
| Save annotated copy | Done |
| Provide CLI interface | Done |
| Clamp invalid out-of-bound bboxes | Done |
| Skip malformed bboxes safely | Done |

## Notes

This tool is primarily for visual validation and debugging. It is useful for comparing the bbox quality of different prompts, teacher models, distilled models, and LM Studio GGUF deployments.

Recommended usage in the current workflow:

```text
LM / VLM prediction JSON
        ->
draw_lm_bboxes.py
        ->
annotated image
        ->
human visual inspection / prompt comparison / bbox quality analysis
```

## Next recommended steps

1. Add a small sample LM output under `examples/` for quick testing.
2. Add a smoke test that creates a temporary image and verifies that the annotated output file is produced.
3. Optionally add batch mode for drawing bbox annotations for an entire JSONL prediction file.
4. Optionally add IoU / click-point evaluation once ground-truth bbox labels are available.
