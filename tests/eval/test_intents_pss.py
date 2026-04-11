"""Tests for the pss corpus."""

import pytest

from scripts.prompt_eval.intents_pss import PSS_INTENTS


def test_corpus_sizes():
    core = [i for i in PSS_INTENTS if i.difficulty == "core"]
    stretch = [i for i in PSS_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_ids_and_interpreter():
    for i in PSS_INTENTS:
        assert i.id.startswith("pss.")
        assert i.interpreter == "pss"


def test_gate_accepts_valid_sheet():
    good = ".fn#validate_token { show: body; }\n.cls#User { show: outline; }"
    for i in PSS_INTENTS:
        gr = i.structural_gate(good)
        assert gr.passed, f"{i.id} rejected a valid sheet: {gr.errors}"


def test_gate_accepts_bare_selector():
    """pss degrades gracefully to single-selector rendering when no braces are present."""
    for i in PSS_INTENTS:
        gr = i.structural_gate(".fn#validate_token")
        assert gr.passed


def test_gate_rejects_unbalanced_braces():
    bad = ".fn#validate_token { show: body;"
    for i in PSS_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_empty():
    for i in PSS_INTENTS:
        gr = i.structural_gate("   \n  ")
        assert not gr.passed
