#!/usr/bin/env python3
"""Create bbox visualizations for the collected 2048-token predictions."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


EXPERIMENTS = [
    "stage1_a0_r16_attn", "stage1_a0_r32_attn",
    "stage1_a1_r16_attn_projector", "stage1_a1_r32_attn_projector",
    "stage1_a2_r16_attn_projector_lora", "stage1_a2_r32_attn_projector_lora",
    "stage1_a3_r16_attn_mlp", "stage1_a3_r32_attn_mlp",
    "stage1_a4_r16_attn_mlp_projector", "stage1_a4_r32_attn_mlp_projector",
]
DEPLOYMENTS = ("mixed_precision", "post_merge_bnb4")
BASELINES = {
    "qwen3_vl_8b_student_4bit": {
        "role": "student", "model": "Qwen3-VL-8B-Instruct", "quantization": "4bit",
    },
    "qwen3_vl_32b_teacher_4bit": {
        "role": "teacher", "model": "Qwen3-VL-32B-Instruct", "quantization": "4bit",
    },
}
NORMALIZED = "normalized_0_1000"


def font(size: int) -> ImageFont.ImageFont:
    for name in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Regular.otf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pixel_bbox(raw: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(x) for x in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in values):
        return None
    x1 = round(values[0] / 1000 * width)
    y1 = round(values[1] / 1000 * height)
    x2 = round(values[2] / 1000 * width)
    y2 = round(values[3] / 1000 * height)
    x1 = max(0, min(x1, width - 1)); x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1)); y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str,
               color: tuple[int, int, int], text_color: tuple[int, int, int],
               fnt: ImageFont.ImageFont, width: int, height: int) -> None:
    box = draw.textbbox((0, 0), label, font=fnt)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = 4
    x = max(0, min(xy[0], max(0, width - tw - 2 * pad)))
    y = xy[1]
    if y < 0:
        y = xy[1] + th + 2 * pad + 3
    y = max(0, min(y, max(0, height - th - 2 * pad)))
    draw.rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), fill=color)
    draw.text((x + pad, y + pad - box[1]), label, fill=text_color, font=fnt)


def render_group(root: Path, deployment: str, experiment: str, overwrite: bool,
                 metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    prediction = root / deployment / experiment / "student_predictions.jsonl"
    out = prediction.parent / "bbox_visualizations"
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment": experiment, "deployment": deployment,
        "prediction_file": str(prediction.resolve()), "rows": 0,
        "images_written": 0, "total_elements": 0, "focused_elements": 0,
        "invalid_bboxes": 0, "missing_images": 0, "errors": [],
    }
    if metadata:
        result.update(metadata)
        result["category"] = "baseline"
        result["max_new_tokens"] = 2048
    if not prediction.is_file():
        result["errors"].append(f"missing prediction file: {prediction}")
        (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    fnt = font(18)
    with prediction.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            result["rows"] += 1
            row_id = f"row-{line_no:06d}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                row_id = str(row.get("id") or row_id)
                source = Path(str(row.get("image") or ""))
                if not source.is_absolute():
                    source = (Path.cwd() / source).resolve()
                if not source.is_file():
                    result["missing_images"] += 1
                    raise FileNotFoundError(f"image not found: {source}")
                elements = row.get("elements")
                if not isinstance(elements, list):
                    raise ValueError("top-level elements is not a list")
                result["total_elements"] += len(elements)
                result["focused_elements"] += sum(1 for e in elements if isinstance(e, dict) and e.get("focused") is True)
                if row.get("coordinate_system", NORMALIZED) != NORMALIZED:
                    raise ValueError(f"unsupported coordinate_system: {row.get('coordinate_system')!r}")
                with Image.open(source) as original:
                    canvas = original.convert("RGB").copy()
                width, height = canvas.size
                draw = ImageDraw.Draw(canvas)
                used: list[tuple[int, int, int, int]] = []
                for index, element in enumerate(elements, 1):
                    if not isinstance(element, dict):
                        result["invalid_bboxes"] += 1
                        result["errors"].append(f"{row_id} element {index}: element is not an object")
                        continue
                    focused = element.get("focused") is True
                    bbox = pixel_bbox(element.get("bbox_norm"), width, height)
                    if bbox is None:
                        result["invalid_bboxes"] += 1
                        result["errors"].append(f"{row_id} element {index}: invalid bbox_norm {element.get('bbox_norm')!r}")
                        continue
                    # Focused: magenta/yellow high-contrast style; normal: cyan/red.
                    color = (255, 0, 220) if focused else (0, 220, 255)
                    text_color = (255, 255, 255) if focused else (0, 0, 0)
                    draw.rectangle(bbox, outline=color, width=4)
                    label = f"{index:02d} {str(element.get('text') or '<no text>')}"
                    if focused:
                        label += " [focused]"
                    draw_label(draw, (bbox[0], bbox[1] - 28), label, color, text_color, fnt, width, height)
                    used.append(bbox)
                suffix = source.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png"}:
                    suffix = ".png"
                target = out / f"{row_id}{suffix}"
                if target.exists() and not overwrite:
                    raise FileExistsError(f"output exists: {target}")
                canvas.save(target, format="PNG" if suffix == ".png" else "JPEG", quality=95)
                result["images_written"] += 1
            except Exception as exc:
                result["errors"].append(f"{row_id}: {type(exc).__name__}: {exc}")
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/lora_ablation/collected_predictions_2048"))
    parser.add_argument("--deployment", choices=DEPLOYMENTS)
    parser.add_argument("--baseline", choices=tuple(BASELINES))
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "bbox_visualizations_manifest.json"
    manifest: dict[str, Any] = {d: [] for d in DEPLOYMENTS}
    if (args.deployment or args.baseline) and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                for deployment in DEPLOYMENTS:
                    if isinstance(existing.get(deployment), list):
                        manifest[deployment] = existing[deployment]
                if isinstance(existing.get("baselines"), list):
                    manifest["baselines"] = existing["baselines"]
        except (OSError, json.JSONDecodeError):
            pass
    if args.manifest_only:
        for deployment in DEPLOYMENTS:
            manifest[deployment] = []
            for experiment in EXPERIMENTS:
                summary_path = root / deployment / experiment / "bbox_visualizations" / "summary.json"
                if not summary_path.is_file():
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                manifest[deployment].append({
                    "experiment": experiment,
                    "prediction_file": summary["prediction_file"],
                    "visualization_dir": str(summary_path.parent.resolve()),
                    "rows": summary["rows"], "images_written": summary["images_written"],
                    "total_elements": summary["total_elements"], "invalid_bboxes": summary["invalid_bboxes"],
                })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({d: len(manifest[d]) for d in DEPLOYMENTS} | {"manifest": str(manifest_path)}, ensure_ascii=False))
        return
    if args.baseline:
        experiment = args.baseline
        summary = render_group(root, "baselines", experiment, args.overwrite, BASELINES[experiment])
        entries = manifest.setdefault("baselines", [])
        entries = [item for item in entries if item.get("experiment") != experiment]
        entries.append({k: summary[k] for k in (
            "experiment", "category", "role", "model", "quantization", "max_new_tokens",
            "prediction_file", "rows", "images_written", "total_elements", "invalid_bboxes", "missing_images")
            if k in summary})
        entries[-1]["visualization_dir"] = str((root / "baselines" / experiment / "bbox_visualizations").resolve())
        manifest["baselines"] = entries
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        collection_manifest_path = root / "manifest.json"
        collection_manifest: dict[str, Any] = {}
        if collection_manifest_path.is_file():
            try:
                loaded = json.loads(collection_manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    collection_manifest = loaded
            except (OSError, json.JSONDecodeError):
                collection_manifest = {}
        collected_dir = root / "baselines" / experiment
        source_dir = Path("outputs/baselines") / f"{experiment}_2048"
        collection_entries = collection_manifest.setdefault("baselines", [])
        collection_entries = [item for item in collection_entries if item.get("experiment") != experiment]
        collection_entries.append({
            "experiment": experiment, "role": summary["role"], "model": summary["model"],
            "quantization": summary["quantization"], "max_new_tokens": 2048,
            "source": str(source_dir), "collected": str(Path("outputs/lora_ablation/collected_predictions_2048") / "baselines" / experiment),
            "prediction_file": str(Path("outputs/lora_ablation/collected_predictions_2048") / "baselines" / experiment / "student_predictions.jsonl"),
            "rows": summary["rows"],
        })
        collection_manifest["baselines"] = collection_entries
        collection_manifest_path.write_text(json.dumps(collection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"baseline": experiment, "manifest": str(manifest_path)}, ensure_ascii=False))
        return
    deployments = (args.deployment,) if args.deployment else DEPLOYMENTS
    experiments = (args.experiment,) if args.experiment else EXPERIMENTS
    for deployment in deployments:
        if not args.experiment:
            manifest[deployment] = []
        elif manifest[deployment]:
            manifest[deployment] = [item for item in manifest[deployment] if item.get("experiment") != args.experiment]
        for experiment in experiments:
            summary = render_group(root, deployment, experiment, args.overwrite)
            manifest[deployment].append({k: summary[k] for k in (
                "experiment", "prediction_file", "visualization_dir", "rows", "images_written", "total_elements", "invalid_bboxes")
                if k in summary})
            manifest[deployment][-1]["visualization_dir"] = str((root / deployment / experiment / "bbox_visualizations").resolve())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({d: len(manifest[d]) for d in DEPLOYMENTS} | {"manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
