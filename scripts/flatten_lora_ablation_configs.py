"""Flatten the lora-ablation YAML inheritance graph.

The resolver deliberately mirrors ``vlm_distill.config_schema``: parents are
loaded relative to the child and mappings are deep-merged, while sequences
and scalar values are replaced by the child value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "lora_ablation"


class _LiteralDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.nodes.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_LiteralDumper.add_representer(str, _represent_str)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_raw(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Config inheritance cycle detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config extends missing file: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Unable to parse YAML {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    raw = dict(raw)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"Config extends must be a non-empty path: {path}")
    return _deep_merge(resolve_raw(path.parent / parent, (*stack, path)), raw)


def flatten(*, write: bool) -> list[Path]:
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    flattened: list[Path] = []
    for path in paths:
        raw = resolve_raw(path)
        if "extends" in raw:  # defensive: resolve_raw removes this key
            raise AssertionError(f"Flattened config still contains extends: {path}")
        source = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(source, dict) and "extends" in source:
            flattened.append(path)
            if write:
                path.write_text(
                    yaml.dump(raw, Dumper=_LiteralDumper, allow_unicode=True,
                              default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )
    return flattened


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="list configs that would be flattened")
    mode.add_argument("--write", action="store_true", help="write flattened configs in place")
    args = parser.parse_args()
    paths = flatten(write=args.write)
    for path in paths:
        print(path.relative_to(ROOT))
    if args.check:
        print(f"{len(paths)} config(s) require flattening")
    else:
        print(f"Flattened {len(paths)} config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
