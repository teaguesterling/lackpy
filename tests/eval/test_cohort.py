"""Tests for cohort helpers."""

import json
from pathlib import Path

from scripts.prompt_eval.cohort import (
    PHASE1A_MODELS,
    pick_phase1b_cohort,
)


def test_phase1a_models_nonempty_and_has_qwen():
    assert len(PHASE1A_MODELS) >= 10
    assert any("qwen" in m for m in PHASE1A_MODELS)
    assert all(":" in m for m in PHASE1A_MODELS)


def test_pick_phase1b_cohort_returns_top_n(tmp_path: Path):
    f = tmp_path / "qual.jsonl"
    lines = [json.dumps({"_meta": {"toybox_hash": "x"}})]
    # Seven models with descending scores
    for i, model in enumerate([f"m{n}" for n in range(7)]):
        for intent_id in ("py.core.01", "py.core.02", "py.core.03", "py.core.04"):
            lines.append(json.dumps({
                "model": model, "interpreter": "python",
                "variant_id": "baseline", "intent_id": intent_id,
                "score": 2 if i < 5 else 0,
                "gate_passed": True,
                "duration_ms_generation": 100.0 * (i + 1),
            }))
    f.write_text("\n".join(lines) + "\n")
    cohort = pick_phase1b_cohort(f, top_n=6, gate_floor=0.5)
    # Only m0..m4 scored 2 so they pass the floor; m5, m6 scored 0 so they miss
    # Wait — all have gate_passed=True so all pass the gate_floor. So all 7 qualify.
    # But top_n=6 so we get 6 of them.
    assert len(cohort) == 6
    # m0 has the lowest latency (100.0), m4 has 500.0, so m0 first
    assert cohort[0] == "m0"


def test_pick_phase1b_cohort_applies_gate_floor(tmp_path: Path):
    f = tmp_path / "qual.jsonl"
    lines = [json.dumps({"_meta": {"toybox_hash": "x"}})]
    # m_good: all gate-pass. m_bad: all gate-fail.
    for intent_id in ("i.1", "i.2", "i.3", "i.4"):
        lines.append(json.dumps({
            "model": "m_good", "interpreter": "python",
            "variant_id": "baseline", "intent_id": intent_id,
            "score": 1, "gate_passed": True,
            "duration_ms_generation": 100.0,
        }))
        lines.append(json.dumps({
            "model": "m_bad", "interpreter": "python",
            "variant_id": "baseline", "intent_id": intent_id,
            "score": 1, "gate_passed": False,
            "duration_ms_generation": 100.0,
        }))
    f.write_text("\n".join(lines) + "\n")
    cohort = pick_phase1b_cohort(f, top_n=10, gate_floor=0.5)
    assert "m_good" in cohort
    assert "m_bad" not in cohort


def test_phase1a_models_are_sorted_small_to_large():
    """Models should be ordered by approximate size ascending so tiny loads first."""
    # Index of qwen2.5-coder:0.5b should be before qwen2.5-coder:7b
    small_idx = PHASE1A_MODELS.index("qwen2.5-coder:0.5b") if "qwen2.5-coder:0.5b" in PHASE1A_MODELS else -1
    big_idx = PHASE1A_MODELS.index("qwen2.5-coder:7b") if "qwen2.5-coder:7b" in PHASE1A_MODELS else -1
    if small_idx >= 0 and big_idx >= 0:
        assert small_idx < big_idx
