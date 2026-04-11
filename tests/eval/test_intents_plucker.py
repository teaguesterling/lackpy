"""Tests for the plucker corpus."""

from scripts.prompt_eval.intents_plucker import PLUCKER_INTENTS


def test_corpus_sizes():
    core = [i for i in PLUCKER_INTENTS if i.difficulty == "core"]
    stretch = [i for i in PLUCKER_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_ids_and_interpreter():
    for i in PLUCKER_INTENTS:
        assert i.id.startswith("pl.")
        assert i.interpreter == "plucker"


def test_gate_accepts_source_chain():
    good = "source().find('.fn').count()"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(good)
        assert gr.passed, f"{i.id} rejected a valid chain: {gr.errors}"


def test_gate_accepts_source_with_arg():
    good = "source('**/*.py').find('.cls').names()"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(good)
        assert gr.passed, f"{i.id} rejected a valid chain with arg: {gr.errors}"


def test_gate_rejects_bare_source():
    bad = "source()"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_non_source_start():
    bad = "find('.fn').count()"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_parse_error():
    bad = "source(.find('.fn')"
    for i in PLUCKER_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed
