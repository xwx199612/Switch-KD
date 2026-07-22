#!/usr/bin/env python3
"""Run the Docker inference API sequentially over a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = args.manifest.read_text(encoding="utf-8").splitlines()
    with args.output.open("w", encoding="utf-8") as handle:
        for index, line in enumerate(rows):
            if not line.strip():
                continue
            if args.max_samples is not None and index >= args.max_samples:
                break
            sample = json.loads(line)
            image_path = Path(sample["image"])
            if not image_path.is_absolute() and args.image_root is not None:
                image_path = args.image_root / image_path
            try:
                with image_path.open("rb") as image:
                    response = requests.post(
                        f"{args.base_url.rstrip('/')}/infer",
                        files={"image": (image_path.name, image, "application/octet-stream")},
                        data={"query": sample.get("query", ""), "request_id": sample["id"]},
                        timeout=600,
                    )
                try:
                    result = response.json()
                except ValueError:
                    result = {"error": response.text}
                if response.status_code >= 400:
                    result = {"id": sample["id"], "http_status": response.status_code, "error": result}
                else:
                    result["id"] = sample["id"]
            except Exception as exc:  # keep batch progress and record the failure
                result = {"id": sample["id"], "http_status": None, "error": str(exc)}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
