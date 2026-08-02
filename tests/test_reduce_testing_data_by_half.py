import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/reduce_testing_data_by_half.py"
SPEC = importlib.util.spec_from_file_location("reduce_testing_data_by_half", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_files(directory, names):
    directory.mkdir()
    for name in names:
        (directory / name).write_text(name)


def test_even_count_keeps_half_and_odd_count_uses_ceil(tmp_path):
    even = tmp_path / "even"
    make_files(even, [f"image-{i}.png" for i in range(4)])
    selection = MODULE.select_images(even)
    assert [p.name for p in selection.kept] == ["image-0.png", "image-2.png"]
    assert len(selection.kept) == len(selection.removed) == 2

    odd = tmp_path / "odd"
    make_files(odd, [f"image-{i}.png" for i in range(5)])
    selection = MODULE.select_images(odd)
    assert len(selection.kept) == 3
    assert len(selection.removed) == 2


def test_natural_sort_and_case_insensitive_extensions(tmp_path):
    directory = tmp_path / "images"
    make_files(directory, ["image-10.PNG", "image-2.png", "image-1.jpeg", "image-3.WEBP"])
    assert [p.name for p in MODULE.image_files(directory)] == [
        "image-1.jpeg", "image-2.png", "image-3.WEBP", "image-10.PNG"
    ]


def test_non_images_are_ignored_and_dry_run_moves_nothing(tmp_path, capsys):
    directory = tmp_path / "images"
    make_files(directory, ["image-0.jpg", "image-1.jpg", "notes.txt", "data.json"])
    selection = MODULE.select_images(directory)
    MODULE.print_report(selection, directory, tmp_path / "backup")
    assert (directory / "notes.txt").exists()
    assert (directory / "data.json").exists()
    assert not (tmp_path / "backup").exists()
    assert "Found images: 2" in capsys.readouterr().out


def test_non_empty_backup_is_rejected(tmp_path):
    directory = tmp_path / "images"
    backup = tmp_path / "backup"
    make_files(directory, ["image-0.jpg", "image-1.jpg"])
    make_files(backup, ["old.jpg"])
    with pytest.raises(RuntimeError, match="non-empty"):
        MODULE.apply_selection(MODULE.select_images(directory), directory, backup)
