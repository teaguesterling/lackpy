"""Tests for the query helper."""

import json
from pathlib import Path

from scripts.prompt_eval.query import summarize_jsonl


def test_summarize_empty(tmp_path: Path):
    f = tmp_path / "empty.jsonl"
    f.write_text(json.dumps({"_meta": {"toybox_hash": "x"}}) + "\n")
    summary = summarize_jsonl(f)
    assert summary["total_rows"] == 0


def test_summarize_counts_scores(tmp_path: Path):
    f = tmp_path / "x.jsonl"
    lines = [json.dumps({"_meta": {"toybox_hash": "x"}})]
    for score in [0, 1, 2, 2, 2]:
        lines.append(json.dumps({
            "model": "m1", "interpreter": "python",
            "variant_id": "baseline", "intent_id": f"i.{score}",
            "score": score,
            "duration_ms_generation": 100.0,
        }))
    f.write_text("\n".join(lines) + "\n")
    summary = summarize_jsonl(f)
    assert summary["total_rows"] == 5
    by_model = summary["by_model"]
    assert by_model["m1"]["rows"] == 5
    assert by_model["m1"]["sum_score"] == 7  # 0+1+2+2+2
    assert by_model["m1"]["pass_rate_2"] == 3 / 5
    assert by_model["m1"]["median_latency_ms"] == 100.0


def test_summarize_groups_by_variant(tmp_path: Path):
    f = tmp_path / "x.jsonl"
    lines = [json.dumps({"_meta": {"toybox_hash": "x"}})]
    lines.append(json.dumps({"model": "m1", "interpreter": "python",
                             "variant_id": "baseline", "intent_id": "a",
                             "score": 2, "duration_ms_generation": 50.0}))
    lines.append(json.dumps({"model": "m1", "interpreter": "python",
                             "variant_id": "specialized", "intent_id": "a",
                             "score": 1, "duration_ms_generation": 50.0}))
    f.write_text("\n".join(lines) + "\n")
    summary = summarize_jsonl(f)
    assert set(summary["by_variant"].keys()) == {"baseline", "specialized"}
    assert summary["by_variant"]["baseline"]["sum_score"] == 2
    assert summary["by_variant"]["specialized"]["sum_score"] == 1


def test_summarize_missing_file(tmp_path: Path):
    """Missing file returns an empty summary, not an exception."""
    f = tmp_path / "nope.jsonl"
    summary = summarize_jsonl(f)
    assert summary["total_rows"] == 0
    assert summary["by_model"] == {}


def test_summarize_skips_malformed_rows(tmp_path: Path):
    f = tmp_path / "bad.jsonl"
    f.write_text(
        json.dumps({"_meta": {"toybox_hash": "x"}}) + "\n"
        + "not-json-at-all\n"
        + json.dumps({"model": "m1", "interpreter": "python",
                      "variant_id": "baseline", "intent_id": "a",
                      "score": 2, "duration_ms_generation": 10.0}) + "\n"
    )
    summary = summarize_jsonl(f)
    assert summary["total_rows"] == 1
    assert summary["by_model"]["m1"]["rows"] == 1


def test_summarize_captures_meta(tmp_path: Path):
    f = tmp_path / "m.jsonl"
    f.write_text(json.dumps({"_meta": {"toybox_hash": "abc123",
                                         "models": ["m1", "m2"]}}) + "\n")
    summary = summarize_jsonl(f)
    assert summary["meta"] is not None
    assert summary["meta"]["toybox_hash"] == "abc123"


def test_summarize_global_median_latency(tmp_path: Path):
    f = tmp_path / "lat.jsonl"
    lines = [json.dumps({"_meta": {}})]
    for lat in [100.0, 200.0, 300.0]:
        lines.append(json.dumps({
            "model": "m1", "interpreter": "python",
            "variant_id": "baseline", "intent_id": f"i.{lat}",
            "score": 2, "duration_ms_generation": lat,
        }))
    f.write_text("\n".join(lines) + "\n")
    summary = summarize_jsonl(f)
    assert summary["global_median_ms"] == 200.0
