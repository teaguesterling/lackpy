"""Tests for the python interpreter intent corpus."""

from scripts.prompt_eval.intents import GateResult
from scripts.prompt_eval.intents_python import PYTHON_INTENTS


def test_corpus_sizes():
    core = [i for i in PYTHON_INTENTS if i.difficulty == "core"]
    stretch = [i for i in PYTHON_INTENTS if i.difficulty == "stretch"]
    assert len(core) == 8
    assert len(stretch) == 6


def test_every_intent_has_unique_id():
    ids = [i.id for i in PYTHON_INTENTS]
    assert len(set(ids)) == len(ids)
    for i in PYTHON_INTENTS:
        assert i.id.startswith("py.")


def test_every_intent_targets_python_interpreter():
    for i in PYTHON_INTENTS:
        assert i.interpreter == "python"


def test_every_intent_has_nonempty_text():
    for i in PYTHON_INTENTS:
        assert len(i.text) > 20


def test_gates_accept_known_good_programs():
    # A minimal valid program should pass every python gate
    good = "files = find_files('**/*.py')\nfiles"
    for i in PYTHON_INTENTS:
        result = i.structural_gate(good)
        assert isinstance(result, GateResult)


def test_assertions_are_callable():
    for i in PYTHON_INTENTS:
        assert callable(i.exec_assertion)
