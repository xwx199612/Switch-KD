"""Render student prediction UI elements on safe, independent image copies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .config_schema import load_config, resolve_prediction_path
from .parsing_output_parser import (
    COORDINATE_SYSTEM_NORMALIZED_0_1000,
    parse_json_like,
    parse_parsing_answer,
)

PALETTE = [(255, 64, 64), (64, 200, 255), (80, 220, 120), (255, 180, 64), (190, 100, 255), (255, 100, 190)]


def normalized_bbox_to_pixel(bbox: Iterable[float], width: int, height: int) -> list[int]:
    """Convert normalized_0_1000 coordinates to clamped, ordered pixel coordinates."""
    values = [float(value) for value in bbox]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("bbox must contain four finite numbers")
    x1, y1, x2, y2 = values
    pixels = [round(x1 / 1000 * width), round(y1 / 1000 * height), round(x2 / 1000 * width), round(y2 / 1000 * height)]
    x1, y1, x2, y2 = pixels
    x1 = max(0, min(width - 1, x1)); x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1)); y2 = max(0, min(height - 1, y2))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _bbox_value(value: Any) -> list[float] | None:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(item) for item in values) else None


def _extract_payload(value: Any) -> tuple[dict[str, Any], str]:
    if isinstance(value, dict):
        return value, str(value.get("coordinate_system") or COORDINATE_SYSTEM_NORMALIZED_0_1000)
    if not isinstance(value, str):
        raise ValueError("prediction is neither an object nor a JSON string")
    # Keep extra fields such as `type` when the repository JSON-like parser can
    # decode them. The existing parsing parser remains the fallback for safe
    # recovery of truncated model output.
    try:
        decoded = parse_json_like(value)
        if isinstance(decoded, dict):
            return decoded, str(decoded.get("coordinate_system") or COORDINATE_SYSTEM_NORMALIZED_0_1000)
    except ValueError:
        pass
    parsed = parse_parsing_answer(value)
    if parsed.get("usable"):
        return {"elements": parsed["elements"], "coordinate_system": parsed.get("coordinate_system") or COORDINATE_SYSTEM_NORMALIZED_0_1000}, COORDINATE_SYSTEM_NORMALIZED_0_1000
    raise ValueError(str(parsed.get("parse_error") or "unable to parse prediction"))


def parse_prediction_elements(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Return raw element dictionaries and explicit coordinate space from a prediction row."""
    coordinate_system = str(row.get("coordinate_system") or COORDINATE_SYSTEM_NORMALIZED_0_1000)
    candidate: Any = row.get("elements")
    if isinstance(candidate, list):
        return [item for item in candidate if isinstance(item, dict)], coordinate_system
    if isinstance(candidate, (dict, str)):
        payload, coordinate_system = _extract_payload(candidate)
        elements = payload.get("elements")
        if not isinstance(elements, list):
            raise ValueError("prediction does not contain an 'elements' list")
        return [item for item in elements if isinstance(item, dict)], coordinate_system
    for key in ("prediction", "student_answer", "raw_model_output", "raw_output"):
        if row.get(key) is not None:
            payload, coordinate_system = _extract_payload(row[key])
            elements = payload.get("elements")
            if not isinstance(elements, list):
                raise ValueError("prediction does not contain an 'elements' list")
            return [item for item in elements if isinstance(item, dict)], coordinate_system
    raise ValueError("row does not contain parsed elements or raw prediction output")


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Regular.otf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _safe_label(index: int, element: dict[str, Any], *, show_text: bool, show_type: bool, show_focused: bool, max_chars: int = 160) -> str:
    text = str(element.get("text") or "<no text>").strip() if show_text else ""
    parts = [f"#{index}"]
    if show_text:
        parts.append(text or "<no text>")
    if show_type and element.get("type"):
        parts.append(str(element["type"]))
    if show_focused and element.get("focused") is True:
        parts.append("FOCUSED")
    label = " | ".join(parts)
    return label if len(label) <= max_chars else label[: max_chars - 1] + "…"


def _output_name(sample_id: str, source: Path, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem) or "image"
    base = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', sample_id)}__{stem}__annotated"
    name = base + ".png"
    if name in used:
        suffix = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:8]
        name = f"{base}__{suffix}.png"
    used.add(name)
    return name


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color: tuple[int, int, int], font: ImageFont.ImageFont, width: int, height: int) -> None:
    box = draw.textbbox((0, 0), label, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = 4
    x = max(0, min(xy[0], max(0, width - tw - pad * 2)))
    y = xy[1]
    if y < 0 or y + th + pad * 2 > height:
        y = max(0, min(height - th - pad * 2, xy[1]))
    draw.rectangle((x, y, x + tw + pad * 2, y + th + pad * 2), fill=color)
    draw.text((x + pad, y + pad - box[1]), label, fill=(255, 255, 255), font=font)


def run_visualization(predictions: Path, output_dir: Path, *, image_root: Path = Path("."), overwrite: bool = False, max_samples: int | None = None, line_width: int = 3, font_size: int = 18, show_text: bool = True, show_type: bool = True, show_focused: bool = True, write_sidecar: bool = True) -> dict[str, int | str]:
    predictions = predictions.resolve(); output_dir = output_dir.resolve(); image_root = image_root.resolve()
    if not predictions.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions}")
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "visualization_errors.jsonl"
    used: set[str] = set(); summary = {"rows_total": 0, "images_written": 0, "rows_parse_failed": 0, "elements_total": 0, "elements_drawn": 0, "elements_skipped": 0}
    errors: list[dict[str, str]] = []
    with predictions.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if max_samples is not None and summary["rows_total"] >= max_samples:
                break
            summary["rows_total"] += 1
            sample_id = f"row-{line_number:06d}"
            source = Path("")
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row must be an object")
                sample_id = str(row.get("id") or sample_id)
                source = Path(str(row.get("image") or ""))
                if not source.is_absolute(): source = image_root / source
                source = source.resolve()
                if not source.exists(): raise FileNotFoundError(f"source image not found: {source}")
                elements, coordinate_system = parse_prediction_elements(row)
                if coordinate_system not in (COORDINATE_SYSTEM_NORMALIZED_0_1000, "normalized-1000"):
                    raise ValueError(f"unsupported coordinate_system={coordinate_system!r}; expected normalized_0_1000")
                name = _output_name(sample_id, source, used); output_path = output_dir / name
                if source == output_path.resolve(): raise ValueError("refusing to write because source and output resolve to the same path")
                if output_path.exists() and not overwrite: raise FileExistsError(f"output exists (use --overwrite): {output_path}")
                sidecar = output_path.with_suffix(".json")
                if write_sidecar and sidecar.exists() and not overwrite: raise FileExistsError(f"sidecar exists (use --overwrite): {sidecar}")
                with Image.open(source) as image:
                    canvas = image.convert("RGB").copy()
                width, height = canvas.size; draw = ImageDraw.Draw(canvas); font = _font(max(1, font_size)); metadata = []
                for index, element in enumerate(elements):
                    summary["elements_total"] += 1
                    raw_bbox = element.get("bbox_norm", element.get("bbox")); parsed_bbox = _bbox_value(raw_bbox)
                    item = {"index": index, "text": element.get("text") or "", "type": element.get("type"), "focused": element.get("focused") is True, "normalized_bbox": parsed_bbox, "pixel_bbox": None, "drawn": False, "skip_reason": None}
                    if parsed_bbox is None: item["skip_reason"] = "missing or malformed bbox"; summary["elements_skipped"] += 1; metadata.append(item); continue
                    try: pixel_bbox = normalized_bbox_to_pixel(parsed_bbox, width, height)
                    except ValueError as exc: item["skip_reason"] = str(exc); summary["elements_skipped"] += 1; metadata.append(item); continue
                    if pixel_bbox[0] == pixel_bbox[2] or pixel_bbox[1] == pixel_bbox[3]: item["skip_reason"] = "bbox has zero pixel area"; summary["elements_skipped"] += 1; metadata.append(item); continue
                    item["pixel_bbox"] = pixel_bbox; item["drawn"] = True; metadata.append(item)
                    color = PALETTE[index % len(PALETTE)]; draw.rectangle(pixel_bbox, outline=color, width=max(1, line_width))
                    label = _safe_label(index, element, show_text=show_text, show_type=show_type, show_focused=show_focused)
                    _draw_label(draw, (pixel_bbox[0], pixel_bbox[1] - font_size - 10), label, color, font, width, height); summary["elements_drawn"] += 1
                canvas.save(output_path, format="PNG"); summary["images_written"] += 1
                if write_sidecar:
                    sidecar.write_text(json.dumps({"sample_id": sample_id, "source_image": str(source), "output_image": str(output_path), "image_width": width, "image_height": height, "coordinate_space": COORDINATE_SYSTEM_NORMALIZED_0_1000, "element_count": len(elements), "drawn_element_count": sum(1 for item in metadata if item["drawn"]), "skipped_element_count": sum(1 for item in metadata if not item["drawn"]), "elements": metadata}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception as exc:  # one bad row must not abort the batch
                summary["rows_parse_failed"] += 1; errors.append({"sample_id": sample_id, "source_image": str(source), "stage": "parse_prediction", "error": f"{type(exc).__name__}: {exc}"})
    if errors:
        errors_path.write_text("".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors), encoding="utf-8")
    elif errors_path.exists() and overwrite:
        errors_path.unlink()
    summary["output_dir"] = str(output_dir); return summary


def run_from_config(config_path: Path, output_dir: Path, **kwargs: Any) -> dict[str, int | str]:
    config = load_config(config_path)
    return run_visualization(resolve_prediction_path(config.data), output_dir, image_root=config.data.image_root, max_samples=kwargs.pop("max_samples", None) if kwargs.get("max_samples") is not None else config.data.max_samples, **kwargs)
