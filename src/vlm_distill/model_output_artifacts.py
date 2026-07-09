from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parsing_output_parser import parse_parsing_answer


def attach_parsing_sidecar_outputs(
    row: dict[str, Any],
    *,
    output_root: Path,
    role: str,
    answer_field: str,
) -> None:
    if row.get("task") != "parsing":
        return

    answer = str(row.get(answer_field) or "").strip()
    raw_relative = Path("raw") / role / f"{row['id']}.txt"
    json_relative = Path("json") / role / f"{row['id']}.json"
    raw_path = output_root / raw_relative
    json_path = output_root / json_relative
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(answer + ("\n" if answer else ""), encoding="utf-8")

    parsed = parse_parsing_answer(answer)
    payload = {
        "source_file": str(raw_relative).replace("\\", "/"),
        **parsed,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prefix = "teacher" if role == "teacher" else "student"
    row[f"{prefix}_raw_output_path"] = str(raw_relative).replace("\\", "/")
    row[f"{prefix}_parsed_output_path"] = str(json_relative).replace("\\", "/")
    row[f"{prefix}_parse_ok"] = bool(parsed["parse_ok"])
    row[f"{prefix}_parse_error"] = parsed["parse_error"]
    row[f"{prefix}_elements"] = parsed["elements"]
    row[f"{prefix}_element_count"] = int(parsed["element_count"])
    if not parsed["parse_ok"]:
        print(
            f"Warning: {prefix} parsing output did not parse cleanly for id={row.get('id')}: "
            f"{parsed['parse_error']}"
        )


def refresh_parsing_sidecar_reports(*, output_root: Path, role: str) -> dict[str, int]:
    raw_dir = output_root / "raw" / role
    json_dir = output_root / "json" / role

    if not raw_dir.exists():
        report = {
            "total_files": 0,
            "parse_ok": 0,
            "parse_failed": 0,
            "total_elements": 0,
        }
        json_dir.mkdir(parents=True, exist_ok=True)
        _write_report_files(json_dir=json_dir, report=report, failures=[])
        return report

    json_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*.txt"))
    failures: list[dict[str, str]] = []
    total_elements = 0
    parse_ok = 0

    for source_path in raw_files:
        raw_text = source_path.read_text(encoding="utf-8")
        parsed = parse_parsing_answer(raw_text)
        raw_relative = Path("raw") / role / source_path.name
        output_payload = {
            "source_file": str(raw_relative).replace("\\", "/"),
            **parsed,
        }
        output_path = json_dir / f"{source_path.stem}.json"
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if parsed["parse_ok"]:
            parse_ok += 1
            total_elements += int(parsed["element_count"])
        else:
            failures.append(
                {
                    "source_file": str(raw_relative).replace("\\", "/"),
                    "parsed_output_file": str((Path("json") / role / output_path.name)).replace("\\", "/"),
                    "parse_error": str(parsed["parse_error"]),
                    "raw_preview": _build_raw_preview(raw_text),
                }
            )

    report = {
        "total_files": len(raw_files),
        "parse_ok": parse_ok,
        "parse_failed": len(raw_files) - parse_ok,
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
