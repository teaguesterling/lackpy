"""Tests for the report generator."""

import json
from pathlib import Path

import pytest

from scripts.prompt_eval.report import (
    ReportData,
    build_report,
    consolidate_jsonls,
)


def _write(path: Path, meta: dict, rows: list[dict]) -> None:
    with path.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_consolidate_single_file(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    _write(f, {"toybox_hash": "h1"}, [
        {"model": "m1", "interpreter": "python", "variant_id": "v1",
         "intent_id": "i.1", "score": 2},
    ])
    data = consolidate_jsonls([f])
    assert isinstance(data, ReportData)
    assert len(data.rows) == 1
    assert data.toybox_hash == "h1"


def test_consolidate_refuses_hash_mismatch(tmp_path: Path):
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    _write(f1, {"toybox_hash": "h1"}, [])
    _write(f2, {"toybox_hash": "h2"}, [])
    with pytest.raises(ValueError, match="hash mismatch"):
        consolidate_jsonls([f1, f2])


def test_consolidate_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        consolidate_jsonls([tmp_path / "nope.jsonl"])


def test_build_report_contains_headings(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    _write(f, {"toybox_hash": "h1"}, [
        {"model": "m1", "interpreter": "python", "variant_id": "baseline",
         "intent_id": "py.core.01", "intent_difficulty": "core",
         "intent_text": "x", "score": 2, "duration_ms_generation": 100.0,
         "gate_passed": True, "assertion_passed": True},
    ])
    data = consolidate_jsonls([f])
    md = build_report(data)
    assert "# Prompt Eval Report" in md
    assert "## Executive summary" in md
    assert "## python" in md
    assert "m1" in md


def test_build_report_ranks_top_cells(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    rows = []
    # m1 × baseline has more score than m2 × baseline
    for i in range(5):
        rows.append({"model": "m1", "interpreter": "python",
                     "variant_id": "baseline", "intent_id": f"i{i}",
                     "score": 2, "duration_ms_generation": 50.0})
    for i in range(5):
        rows.append({"model": "m2", "interpreter": "python",
                     "variant_id": "baseline", "intent_id": f"i{i}",
                     "score": 0, "duration_ms_generation": 100.0})
    _write(f, {"toybox_hash": "h1"}, rows)
    data = consolidate_jsonls([f])
    md = build_report(data)
    # m1 should appear before m2 in the python section
    py_section = md.split("## python")[1].split("##")[0]
    assert py_section.index("m1") < py_section.index("m2")


def test_build_report_shows_failure_modes(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    rows = [
        {"model": "m1", "interpreter": "python", "variant_id": "v1",
         "intent_id": "i.1", "score": 0,
         "gate_errors": ["forbidden import statement"],
         "duration_ms_generation": 10.0},
        {"model": "m1", "interpreter": "python", "variant_id": "v1",
         "intent_id": "i.2", "score": 0,
         "gate_errors": ["forbidden import statement"],
         "duration_ms_generation": 10.0},
        {"model": "m1", "interpreter": "python", "variant_id": "v1",
         "intent_id": "i.3", "score": 1, "exec_error": "ValueError: x",
         "duration_ms_generation": 10.0},
    ]
    _write(f, {"toybox_hash": "h1"}, rows)
    data = consolidate_jsonls([f])
    md = build_report(data)
    assert "## Top failure modes" in md
    assert "forbidden import statement" in md
    assert "ValueError" in md
