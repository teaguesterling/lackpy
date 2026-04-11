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
        assert result.passed, f"{i.id} unexpectedly rejected known-good program: {result.errors}"


def test_gates_reject_invalid_programs():
    """The gate must reject programs that lackpy's validator forbids:
    imports, def/class, dunder access, and calls to unknown names.
    Any intent using the shared python gate should reject all of these.
    """
    bad_programs = [
        "import os\nos.listdir('.')",           # import forbidden
        "def foo():\n    return 1\nfoo()",        # def forbidden
        "class Foo:\n    pass\nFoo()",            # class forbidden
        "getattr(read_file, '__code__')",         # dunder access forbidden
        "unknown_tool('x')",                      # name not in kit
    ]
    for i in PYTHON_INTENTS:
        for bad in bad_programs:
            result = i.structural_gate(bad)
            assert not result.passed, (
                f"{i.id} accepted a program it should reject: {bad!r}"
            )
            assert result.errors, f"{i.id} rejected {bad!r} with no errors"


def test_assertions_are_callable():
    for i in PYTHON_INTENTS:
        assert callable(i.exec_assertion)
