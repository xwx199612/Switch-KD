#!/usr/bin/env python3
"""Compare JSONL predictions while preserving the project's ordered element semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {str(row["id"]): row for row in (json.loads(line) for line in handle if line.strip())}


def _usable(row: dict | None) -> object:
    if row is None:
        return None
    return row.get("usable", bool(row.get("elements")))


def compare(local: Path, docker: Path) -> dict[str, int]:
    left, right = _rows(local), _rows(docker)
    counts = {"total": len(set(left) | set(right)), "exact_match": 0,
              "parse_mismatch": 0, "element_count_mismatch": 0, "text_mismatch": 0,
              "bbox_mismatch": 0, "focused_mismatch": 0}
    for key in set(left) | set(right):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            counts["parse_mismatch"] += 1
            continue
        differences = False
        if _usable(a) != _usable(b):
            counts["parse_mismatch"] += 1
            differences = True
        ae, be = a.get("elements", []), b.get("elements", [])
        if len(ae) != len(be):
            counts["element_count_mismatch"] += 1
            differences = True
        if [x.get("text") for x in ae] != [x.get("text") for x in be]:
            counts["text_mismatch"] += 1
            differences = True
        if [x.get("bbox_norm") for x in ae] != [x.get("bbox_norm") for x in be]:
            counts["bbox_mismatch"] += 1
            differences = True
        if [x.get("focused") for x in ae] != [x.get("focused") for x in be]:
            counts["focused_mismatch"] += 1
            differences = True
        if a.get("coordinate_system") != b.get("coordinate_system"):
            counts["parse_mismatch"] += 1
            differences = True
        if not differences:
            counts["exact_match"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-predictions", type=Path, required=True)
    parser.add_argument("--docker-predictions", type=Path, required=True)
    args = parser.parse_args()
    for key, value in compare(args.local_predictions, args.docker_predictions).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
