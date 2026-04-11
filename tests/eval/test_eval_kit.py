"""Tests for the harness-local eval kit."""

from pathlib import Path
import pytest

from scripts.prompt_eval.eval_kit import build_eval_kit


@pytest.fixture
def toybox_tmp(tmp_path: Path) -> Path:
    """Tiny ad-hoc toybox stand-in; the real toybox arrives in Task 2."""
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return foo()\n")
    return tmp_path


def test_eval_kit_has_builtin_tools(toybox_tmp: Path):
    kit = build_eval_kit(toybox_tmp)
    for name in ("read_file", "find_files", "find_def", "find_refs"):
        assert name in kit.tools, f"missing tool {name}"
        assert name in kit.callables


def test_find_def_returns_matching_file(toybox_tmp: Path):
    kit = build_eval_kit(toybox_tmp)
    rows = kit.callables["find_def"]("foo")
    assert isinstance(rows, list)
    assert any("a.py" in r["file"] for r in rows)
    assert all("line" in r for r in rows)


def test_find_refs_returns_matching_file(toybox_tmp: Path):
    kit = build_eval_kit(toybox_tmp)
    rows = kit.callables["find_refs"]("foo")
    assert isinstance(rows, list)
    # b.py calls foo()
    assert any("b.py" in r["file"] for r in rows)
