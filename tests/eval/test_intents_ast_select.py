"""Tests for the ast-select corpus."""

from scripts.prompt_eval.intents_ast_select import AST_SELECT_INTENTS


def test_corpus_sizes():
    core = [i for i in AST_SELECT_INTENTS if i.difficulty == "core"]
    stretch = [i for i in AST_SELECT_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_ids_are_unique_and_prefixed():
    ids = [i.id for i in AST_SELECT_INTENTS]
    assert len(set(ids)) == len(ids)
    for i in AST_SELECT_INTENTS:
        assert i.id.startswith("as.")
        assert i.interpreter == "ast-select"


def test_gate_accepts_bare_selector():
    good = ".fn#validate_token"
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate(good)
        assert gr.passed, f"{i.id} rejected a valid selector: {gr.errors}"


def test_gate_rejects_sheet():
    bad = ".fn#validate_token { show: body; }"
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_empty():
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate("")
        assert not gr.passed


def test_gate_rejects_multiline():
    bad = ".fn#a\n.fn#b"
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed


def test_gate_rejects_invalid_start():
    bad = "validate_token"
    for i in AST_SELECT_INTENTS:
        gr = i.structural_gate(bad)
        assert not gr.passed
