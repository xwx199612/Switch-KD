from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parsing_output_parser import parse_parsing_answer


def write_parsing_sidecar(
    *,
    row: dict[str, Any],
    output_root: Path,
    role: str,
    raw_model_output: str,
) -> dict[str, Any]:
    if row.get("task") != "parsing":
        return {}

    json_relative = Path("json") / role / f"{row['id']}.json"
    json_path = output_root / json_relative
    json_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = parse_parsing_answer(raw_model_output)
    payload = {
        "source": role,
        "id": row.get("id"),
        "image": row.get("image"),
        "task": row.get("task"),
        "query": row.get("query"),
        "raw_model_output": raw_model_output,
        **parsed,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return parsed


def refresh_parsing_sidecar_reports(*, output_root: Path, role: str) -> dict[str, int]:
    json_dir = output_root / "json" / role
    json_dir.mkdir(parents=True, exist_ok=True)
    sidecar_files = sorted(
        path for path in json_dir.glob("*.json")
        if path.name != "parse_report.json"
    )
    failures: list[dict[str, str]] = []
    total_elements = 0
    parse_ok = 0

    for source_path in sidecar_files:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("parse_ok"):
            parse_ok += 1
            total_elements += int(payload.get("element_count", 0))
        else:
            failures.append(
                {
                    "id": str(payload.get("id") or source_path.stem),
                    "json_sidecar": str((Path("json") / role / source_path.name)).replace("\\", "/"),
                    "parse_error": str(payload.get("parse_error")),
                    "raw_preview": _build_raw_preview(str(payload.get("raw_model_output") or "")),
                }
            )

    report = {
        "total_files": len(sidecar_files),
        "parse_ok": parse_ok,
        "parse_failed": len(sidecar_files) - parse_ok,
        "total_elements": total_elements,
    }
    _write_report_files(json_dir=json_dir, report=report, failures=failures)
    return report


def _write_report_files(
    *,
    json_dir: Path,
    report: dict[str, int],
    failures: list[dict[str, str]],
) -> None:
    (json_dir / "parse_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (json_dir / "parse_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_raw_preview(raw_text: str, limit: int = 200) -> str:
    compact = " ".join(raw_text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
