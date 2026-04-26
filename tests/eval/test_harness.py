"""Tests for the harness orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
pytest.importorskip("tqdm", reason="tqdm required for eval harness tests")

from scripts.prompt_eval.harness import (
    HarnessConfig,
    compute_cells,
    load_completed_keys,
    make_row,
    run_harness,
    toybox_hash,
    _read_meta_hash,
)
from scripts.prompt_eval.intents import GateResult, Intent
from scripts.prompt_eval.runner import GenerationRecord


TOYBOX = Path(__file__).parent / "fixtures" / "toybox"


def _trivial_intent(id_: str, interp: str = "python"):
    return Intent(
        id=id_,
        interpreter=interp,
        difficulty="core",
        text="trivial",
        return_shape="int",
        structural_gate=lambda p: GateResult(passed=True),
        exec_assertion=lambda x: True,
    )


def test_toybox_hash_is_64_hex():
    h = toybox_hash(TOYBOX)
    assert isinstance(h, str)
    assert len(h) == 64


def test_compute_cells_matrix_size():
    intents = [_trivial_intent("py.a"), _trivial_intent("py.b")]
    cells = compute_cells(
        models=["m1", "m2"],
        interpreters=["python"],
        variant_ids=["baseline", "specialized"],
        intents=intents,
    )
    assert len(cells) == 2 * 1 * 2 * 2  # m × i × v × intent


def test_compute_cells_model_is_outermost():
    intents = [_trivial_intent("py.a"), _trivial_intent("py.b")]
    cells = compute_cells(
        models=["m1", "m2"],
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=intents,
    )
    # All m1 cells come before any m2 cell
    m1_last = max(i for i, c in enumerate(cells) if c[0] == "m1")
    m2_first = min(i for i, c in enumerate(cells) if c[0] == "m2")
    assert m1_last < m2_first


def test_compute_cells_filters_intents_by_interpreter():
    py_intent = _trivial_intent("py.a", interp="python")
    as_intent = _trivial_intent("as.a", interp="ast-select")
    cells = compute_cells(
        models=["m1"],
        interpreters=["python", "ast-select"],
        variant_ids=["baseline"],
        intents=[py_intent, as_intent],
    )
    # Each interpreter sees only its own intents
    py_cells = [c for c in cells if c[1] == "python"]
    as_cells = [c for c in cells if c[1] == "ast-select"]
    assert len(py_cells) == 1 and py_cells[0][3].id == "py.a"
    assert len(as_cells) == 1 and as_cells[0][3].id == "as.a"


def test_load_completed_keys_empty_file(tmp_path: Path):
    f = tmp_path / "nope.jsonl"
    keys = load_completed_keys(f)
    assert keys == set()


def test_load_completed_keys_reads_meta_and_rows(tmp_path: Path):
    f = tmp_path / "x.jsonl"
    f.write_text(
        json.dumps({"_meta": {"toybox_hash": "abc"}}) + "\n"
        + json.dumps({"model": "m1", "interpreter": "python",
                      "variant_id": "baseline", "intent_id": "py.a",
                      "score": 2}) + "\n"
    )
    keys = load_completed_keys(f)
    assert ("m1", "python", "baseline", "py.a") in keys
    assert len(keys) == 1


def test_make_row_contains_expected_keys():
    intent = _trivial_intent("py.a")
    gen = GenerationRecord(
        model="m1", raw="1", tokens_eval=1, tokens_prompt=5,
        duration_ms=200.0, error=None,
    )
    row = make_row(
        model="m1", interpreter="python", variant_id="baseline",
        intent=intent, gen=gen, score=None,
    )
    for key in ("model", "interpreter", "variant_id", "intent_id",
                "raw_generation", "duration_ms_generation",
                "tokens_eval", "tokens_prompt", "gen_error"):
        assert key in row


def test_run_harness_resume_skips_completed(tmp_path: Path):
    intents = [_trivial_intent("py.a"), _trivial_intent("py.b")]
    cfg = HarnessConfig(
        output_path=tmp_path / "out.jsonl",
        models=["m1"],
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host="http://unused",
        temperature=0.1,
        timeout=10,
    )

    # Pre-seed the output with py.a already done
    cfg.output_path.write_text(
        json.dumps({"_meta": {"toybox_hash": toybox_hash(TOYBOX)}}) + "\n"
        + json.dumps({"model": "m1", "interpreter": "python",
                      "variant_id": "baseline", "intent_id": "py.a",
                      "score": 2}) + "\n"
    )

    called = []
    def fake_generate(*, client, model, system_prompt, user_message,
                       temperature, timeout, keep_alive="30m"):
        called.append((model, user_message))
        return GenerationRecord(
            model=model, raw="1", tokens_eval=1, tokens_prompt=1,
            duration_ms=1.0, error=None,
        )

    with patch("scripts.prompt_eval.harness.generate_once", side_effect=fake_generate), \
         patch("scripts.prompt_eval.harness.make_ollama_client", return_value=MagicMock()):
        run_harness(cfg)

    # Only py.b should have run
    assert len(called) == 1
    assert "trivial" in called[0][1]


def test_run_harness_writes_meta_row_on_new_file(tmp_path: Path):
    intents = [_trivial_intent("py.a")]
    cfg = HarnessConfig(
        output_path=tmp_path / "new.jsonl",
        models=["m1"],
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host="http://unused",
        temperature=0.1,
        timeout=10,
    )

    def fake_generate(*args, **kwargs):
        return GenerationRecord(
            model="m1", raw="42", tokens_eval=1, tokens_prompt=1,
            duration_ms=1.0, error=None,
        )

    with patch("scripts.prompt_eval.harness.generate_once", side_effect=fake_generate), \
         patch("scripts.prompt_eval.harness.make_ollama_client", return_value=MagicMock()):
        run_harness(cfg)

    lines = cfg.output_path.read_text().strip().splitlines()
    assert len(lines) == 2  # _meta + 1 row
    first = json.loads(lines[0])
    assert "_meta" in first
    assert first["_meta"]["toybox_hash"] == toybox_hash(TOYBOX)


def test_run_harness_refuses_resume_on_hash_mismatch(tmp_path: Path):
    """Resuming a JSONL written against a different toybox should raise."""
    import pytest

    intents = [_trivial_intent("py.a")]
    cfg = HarnessConfig(
        output_path=tmp_path / "mismatch.jsonl",
        models=["m1"],
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host="http://unused",
        temperature=0.1,
        timeout=10,
    )
    # Seed the output with a meta row whose hash is wrong
    cfg.output_path.write_text(
        json.dumps({"_meta": {"toybox_hash": "deadbeef" * 8}}) + "\n"
    )
    with patch("scripts.prompt_eval.harness.make_ollama_client", return_value=MagicMock()), \
         patch("scripts.prompt_eval.harness.generate_once") as gen:
        with pytest.raises(RuntimeError, match="toybox hash mismatch"):
            run_harness(cfg)
        gen.assert_not_called()


def test_run_harness_refuses_resume_on_missing_meta_hash(tmp_path: Path):
    """A JSONL whose _meta has no toybox_hash should fail to resume."""
    import pytest

    intents = [_trivial_intent("py.a")]
    cfg = HarnessConfig(
        output_path=tmp_path / "no_meta.jsonl",
        models=["m1"],
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host="http://unused",
        temperature=0.1,
        timeout=10,
    )
    cfg.output_path.write_text(
        json.dumps({"_meta": {"note": "no hash here"}}) + "\n"
    )
    with patch("scripts.prompt_eval.harness.make_ollama_client", return_value=MagicMock()), \
         patch("scripts.prompt_eval.harness.generate_once") as gen:
        with pytest.raises(RuntimeError, match="no toybox_hash"):
            run_harness(cfg)


def test_run_harness_allows_resume_when_hash_matches(tmp_path: Path):
    """The happy path: existing JSONL's meta hash matches current toybox."""
    intents = [_trivial_intent("py.a")]
    cfg = HarnessConfig(
        output_path=tmp_path / "ok.jsonl",
        models=["m1"],
        interpreters=["python"],
        variant_ids=["baseline"],
        intents=intents,
        toybox_dir=TOYBOX,
        ollama_host="http://unused",
        temperature=0.1,
        timeout=10,
    )
    cfg.output_path.write_text(
        json.dumps({"_meta": {"toybox_hash": toybox_hash(TOYBOX)}}) + "\n"
    )
    def fake_generate(*args, **kwargs):
        return GenerationRecord(
            model="m1", raw="42", tokens_eval=1, tokens_prompt=1,
            duration_ms=1.0, error=None,
        )
    with patch("scripts.prompt_eval.harness.generate_once", side_effect=fake_generate), \
         patch("scripts.prompt_eval.harness.make_ollama_client", return_value=MagicMock()):
        run_harness(cfg)
    assert cfg.output_path.exists()
    # Should have seeded meta + 1 row
    lines = cfg.output_path.read_text().strip().splitlines()
    assert len(lines) == 2
