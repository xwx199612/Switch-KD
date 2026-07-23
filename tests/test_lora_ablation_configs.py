from pathlib import Path

import yaml

from vlm_distill.config_schema import load_config


CONFIG_ROOT = Path("configs/lora_ablation")


def test_lora_ablation_configs_have_no_top_level_extends():
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert isinstance(raw, dict)
        assert "extends" not in raw, path


def test_lora_ablation_configs_load_independently(tmp_path: Path):
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        copied = tmp_path / path.name
        copied.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        load_config(copied)
