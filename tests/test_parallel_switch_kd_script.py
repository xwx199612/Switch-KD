from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path("scripts/run_parallel_switch_kd_precompute_4gpu.sh")


def test_parallel_switch_kd_script_is_deprecation_stub():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "This helper has been deprecated." in text
    assert "train_online_align_dbild" in text
    assert "offline teacher logits" in text


def test_parallel_switch_kd_script_exits_with_deprecation_message():
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=Path("."),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "deprecated" in (result.stdout + result.stderr).lower()
