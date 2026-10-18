"""The runtime namespace (stepy100 Plan 2, Task 2).

The namespace is part of the sandbox boundary. A program sees only the names
that ``builtins_policy.ALLOWED`` approves, plus ``__name__``, ``__file__``,
``print`` and a ``type`` stand-in. ``LackpyRefusal`` derives from
``BaseException`` so that a program's ``except Exception`` cannot swallow it.
"""

import pytest

from lackpy.lang import builtins_policy, dictionary
from lackpy.run.namespace import build_namespace
from lackpy.run.refusal import LackpyRefusal, refuse


def _ns(program_file=None):
    return build_namespace(program_file=program_file, path_policy=None, print_fn=print)


def _exec(source, program_file=None):
    ns = _ns(program_file)
    exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
    return ns


class TestRefusal:
    def test_refusal_is_not_an_exception(self):
        # A program's `except Exception` must never catch a refusal.
        assert issubclass(LackpyRefusal, BaseException)
        assert not issubclass(LackpyRefusal, Exception)

    def test_message_without_hint(self):
        r = LackpyRefusal("type(...).mro")
        assert r.what == "type(...).mro"
        assert r.hint is None
        assert str(r) == "type(...).mro is not allowed in lackpy"

    def test_message_with_hint(self):
        r = LackpyRefusal("open", "use Path(p).read_text()")
        assert str(r) == "open is not allowed in lackpy: use Path(p).read_text()"

    def test_refuse_looks_up_the_unapproved_hint(self):
        with pytest.raises(LackpyRefusal) as info:
            refuse("open", "open")
        assert info.value.hint == dictionary.UNAPPROVED["open"]
        assert dictionary.UNAPPROVED["open"] in str(info.value)

    def test_refuse_without_hint_key(self):
        with pytest.raises(LackpyRefusal) as info:
            refuse("x")
        assert info.value.hint is None


class TestBuiltins:
    def test_builtins_hold_exactly_the_allowed_names(self):
        assert set(_ns()["__builtins__"]) == set(builtins_policy.ALLOWED)

    @pytest.mark.parametrize("name", [
        "__import__", "__build_class__", "open", "eval", "exec", "compile",
        "getattr", "type", "object", "BaseException", "SystemExit",
    ])
    def test_denied_name_is_absent(self, name):
        assert name not in _ns()["__builtins__"]

    def test_key_error_and_repr_are_available(self):
        ns = _exec("e = KeyError('k')\nr = repr([1])")
        assert isinstance(ns["e"], KeyError)
        assert ns["r"] == "[1]"

    def test_class_statement_fails(self):
        # No __build_class__: a class body cannot run.
        with pytest.raises(NameError):
            _exec("class A:\n    pass")

    def test_import_fails_closed(self):
        # Ruling A: no __import__ until Task 3.
        with pytest.raises(ImportError, match="__import__ not found"):
            _exec("import json")


class TestModuleNames:
    def test_name_is_main(self):
        assert _exec("n = __name__")["n"] == "__main__"

    def test_file_defaults_to_stdin(self):
        assert _exec("f = __file__")["f"] == "<stdin>"

    def test_file_is_the_program_path(self):
        assert _exec("f = __file__", program_file="/w/prog.py")["f"] == "/w/prog.py"

    def test_print_is_the_given_function(self):
        out = []
        ns = build_namespace(program_file=None, path_policy=None, print_fn=out.append)
        exec("print('hi')", ns)  # noqa: S102
        assert out == ["hi"]


class TestTypeStandIn:
    @pytest.mark.parametrize("value, name", [
        (3, "int"), ("s", "str"), ([], "list"), ({}, "dict"), (None, "NoneType"),
        (KeyError("k"), "KeyError"),
    ])
    def test_name_matches_cpython(self, value, name):
        ns = _ns()
        ns["v"] = value
        exec("n = type(v).__name__", ns)  # noqa: S102
        assert ns["n"] == name == type(value).__name__

    @pytest.mark.parametrize("source", [
        "type(3).mro",
        "type(3).__class__",
        "type(3).__subclasses__",
        "type(3).__dict__",
        "type(3).__base__",
    ])
    def test_other_attributes_refuse(self, source):
        with pytest.raises(LackpyRefusal):
            _exec(source)

    def test_calling_the_result_refuses(self):
        with pytest.raises(LackpyRefusal):
            _exec("type(3)('4')")

    def test_three_argument_form_refuses(self):
        with pytest.raises(LackpyRefusal):
            _exec("type('A', (), {})")

    def test_comparison_refuses_instead_of_giving_a_wrong_answer(self):
        # CPython says True; the stand-in cannot, so it must refuse (design §5.3).
        with pytest.raises(LackpyRefusal):
            _exec("type(1) == type(2)")

    def test_type_of_the_stand_in_refuses(self):
        # CPython says "type"; the stand-in would say something else.
        with pytest.raises(LackpyRefusal):
            _exec("type(type(3))")

    def test_the_stand_in_itself_exposes_no_attributes(self):
        # A plain function would leak __globals__.
        with pytest.raises(LackpyRefusal):
            _exec("type.__globals__")

    def test_identity_check_is_false(self):
        assert _exec("r = type(3) is int")["r"] is False
