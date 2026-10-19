"""Every builtin is classified; dangerous ones are never allowed."""

import builtins

from lackpy.lang.builtins_policy import ALLOWED, DENIED, allowed_builtin_names
from lackpy.lang.validator import validate


def test_every_builtin_is_classified():
    # A new Python release that adds a builtin must fail here, not slip in.
    assert set(dir(builtins)) <= set(ALLOWED) | set(DENIED)
    assert ALLOWED.keys() & DENIED.keys() == set()


def test_no_dangerous_name_is_allowed():
    dangerous = (
        "open eval exec compile __import__ getattr setattr delattr hasattr globals "
        "locals vars dir type super object breakpoint input exit quit BaseException "
        "SystemExit KeyboardInterrupt GeneratorExit"
    ).split()
    assert [n for n in dangerous if n in ALLOWED] == []


def test_exceptions_allowed():
    for name in ("KeyError", "ValueError", "FileNotFoundError", "UnicodeDecodeError"):
        assert name in ALLOWED


def test_denied_reasons_are_known_categories():
    known = {"reflection", "code-execution", "io", "interpreter-control",
             "type-machinery", "use-comprehension", "binary", "dunder"}
    assert set(DENIED.values()) <= known
    assert DENIED["open"] == "io"
    assert DENIED["type"] == "type-machinery"


def test_allowed_names_are_derived_from_allowed():
    assert allowed_builtin_names() == frozenset(ALLOWED)


def test_mro_denied():
    result = validate("x = KeyError.mro()")
    assert any(v.rule_id == "safety.attr-denied" for v in result.violations)


def test_sort_by_is_gone():
    result = validate("sort_by([], 'a')")
    assert not result.valid
    assert any(v.rule_id == "safety.unknown-call" for v in result.violations)


def test_allowed_construction_survives_missing_exception_group(monkeypatch):
    # ExceptionGroup exists only from Python 3.11; the lang package supports 3.10.
    from lackpy.lang.builtins_policy import _build_allowed

    monkeypatch.delattr(builtins, "ExceptionGroup")
    built = _build_allowed()
    assert "ExceptionGroup" not in built
    assert "KeyError" in built
