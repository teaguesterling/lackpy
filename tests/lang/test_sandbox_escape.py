"""Adversarial tests for the AST-whitelist sandbox boundary (GHSA-hpcj-3c97-43jm).

These assert the validator REJECTS the dunder/private attribute-traversal escape
family *before execution*. No test executes an escape: the sentinel is a benign
marker object; we only assert ``validate(...).valid is False`` and that the
rejection reason names the attribute, so a green cannot come from an unrelated
rejection.
"""

import pytest

from lackpy.lang.validator import validate


def _attr_rejected(result) -> bool:
    """A rejection attributable to the attribute-access guard specifically."""
    return not result.valid and any(
        "attribute" in e.lower() for e in result.errors
    )


class TestDunderTraversalEscapes:
    """The confirmed escape family: reach type/module objects via dunder walk."""

    def test_rejects_confirmed_subclasses_escape(self):
        # ().__class__.__bases__[0].__subclasses__() — the reported chain.
        prog = "subs = ().__class__.__bases__[0].__subclasses__()"
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_globals_walk(self):
        # …__init__.__globals__ — the step that reaches the real module namespace.
        prog = (
            "subs = ().__class__.__bases__[0].__subclasses__()\n"
            "g = subs[0].__init__.__globals__\n"
        )
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_dunder_via_comprehension(self):
        prog = (
            "subs = ().__class__.__bases__[0].__subclasses__()\n"
            "picked = [c for c in subs if c.__name__ == 'x']\n"
        )
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_dunder_via_subscript_chain(self):
        prog = "x = ().__class__.__mro__[-1].__subclasses__()"
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_getattribute_attr(self):
        # __getattribute__ as an attribute (distinct from the string-const path).
        prog = "x = ().__getattribute__"
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_globals_on_whitelisted_builtin(self):
        # A whitelisted-looking Name (allowed builtin) walked to __globals__.
        prog = "g = sorted.__globals__"
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_dunder_inside_fstring(self):
        # FormattedValue carrying a dunder attribute-access.
        prog = "x = f'{().__class__}'"
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_bare_dunder_attribute(self):
        prog = "x = ().__class__"
        result = validate(prog)
        assert _attr_rejected(result), result.errors

    def test_rejects_verbatim_reported_chain(self):
        # The exact structure of the reported escape, with a BENIGN marker call
        # target (never executed — validation rejects it first). Pins both the
        # dunder-attribute guard AND the subscript-call default-deny in one test:
        # the final ``g['marker'](...)`` has func=Subscript.
        prog = (
            "subs = ().__class__.__bases__[0].__subclasses__()\n"
            "g = [c for c in subs if c.__name__ == 'marker'][0].__init__.__globals__\n"
            "g['marker']('SENTINEL_NOT_EXECUTED')\n"
        )
        result = validate(prog)
        assert not result.valid, result.errors
        # Rejected for the attribute walk (dunder access), not merely the call form.
        assert _attr_rejected(result), result.errors


class TestPrivateAttributeEscapes:
    """Single-underscore private attributes: broader than the advisory's literal
    ``__`` wording, closing the ``()._module``-style gadget it references."""

    def test_rejects_single_underscore_attribute(self):
        prog = "x = some_obj._module"
        result = validate(prog, allowed_names={"some_obj"})
        assert _attr_rejected(result), result.errors

    def test_rejects_private_attribute_call(self):
        prog = "x = some_obj._private_method()"
        result = validate(prog, allowed_names={"some_obj"})
        assert _attr_rejected(result), result.errors


class TestFrameAttributeEscapes:
    """Frame/generator/traceback attributes have no leading underscore, so an
    underscore-only ban would miss them. They are currently UNREACHABLE in the
    subset (no GeneratorExp / Try / Raise / sys), but we deny them explicitly as
    belt-and-suspenders. This is the named residual of a denylist approach."""

    @pytest.mark.parametrize(
        "attr",
        ["f_globals", "f_builtins", "f_locals", "gi_frame", "cr_frame", "tb_frame"],
    )
    def test_rejects_frame_attribute(self, attr):
        prog = f"x = some_obj.{attr}"
        result = validate(prog, allowed_names={"some_obj"})
        assert not result.valid, result.errors


class TestExoticCallForms:
    """Step 4 default-denies calls whose target is not a Name or method
    attribute — the previously-unchecked bypass where ``func`` is a Subscript or
    a Call result."""

    def test_rejects_call_of_subscript(self):
        result = validate("x = funcs[0]()", allowed_names={"funcs"})
        assert not result.valid, result.errors
        assert any("call target" in e.lower() for e in result.errors), result.errors

    def test_rejects_call_of_call_result(self):
        result = validate("x = get()()", allowed_names={"get"})
        assert not result.valid, result.errors
        assert any("call target" in e.lower() for e in result.errors), result.errors


class TestDunderRegardlessOfValueShape:
    """The attribute guard must fire no matter what expression the dunder is
    accessed *on* — a Tuple, a Subscript, a method-call result, etc."""

    def test_rejects_dunder_on_subscript_of_call(self):
        result = validate("x = data.items()[0].__class__", allowed_names={"data"})
        assert _attr_rejected(result), result.errors

    def test_rejects_dunder_on_method_chain(self):
        result = validate("s = get()\nx = s.encode().__class__", allowed_names={"get"})
        assert _attr_rejected(result), result.errors

    def test_rejects_globals_on_subscript(self):
        result = validate("x = lst[0].__globals__", allowed_names={"lst"})
        assert _attr_rejected(result), result.errors


class TestNoOverBlocking:
    """Legitimate attribute access on data objects must still validate — the
    guard keys on underscore prefixes / a small deny set, not all attributes."""

    def test_allows_str_split(self):
        prog = "data = read_file('f.txt')\nlines = data.split('\\n')"
        result = validate(prog, allowed_names={"read_file"})
        assert result.valid, result.errors

    def test_allows_dict_methods(self):
        prog = "d = load()\nks = d.keys()\nvs = d.values()\nits = d.items()"
        result = validate(prog, allowed_names={"load"})
        assert result.valid, result.errors

    def test_allows_str_methods(self):
        prog = "s = get()\nout = s.strip().upper().replace('a', 'b')"
        result = validate(prog, allowed_names={"get"})
        assert result.valid, result.errors

    def test_allows_get_with_default(self):
        prog = "d = load()\nv = d.get('key')"
        result = validate(prog, allowed_names={"load"})
        assert result.valid, result.errors


class TestGeneratorFrameEscapes:
    """GeneratorExp is in the subset, so a generator object CAN now be built.

    Before that change these attributes were unreachable *by construction* and
    ``DENIED_ATTRIBUTES`` was defense-in-depth. It is now the live guard: it is
    the only thing standing between a generator expression and
    ``gi_frame.f_globals``. These tests exist so pruning that list fails loudly.
    """

    @pytest.mark.parametrize("program", [
        "(x for x in xs).gi_frame",
        "(x for x in xs).gi_frame.f_globals",
        "(x for x in xs).gi_frame.f_builtins",
        "(x for x in xs).gi_code",
        "next(x for x in xs).__class__",
    ])
    def test_generator_internals_rejected(self, program):
        assert _attr_rejected(validate(program, allowed_names={"xs"}))

    @pytest.mark.parametrize("program", [
        "next(x for x in xs if x)",
        "any(x for x in xs)",
        "all(x for x in xs)",
        "sum(x for x in xs)",
        "sorted(x for x in xs)",
    ])
    def test_idiomatic_generator_forms_accepted(self, program):
        """The point of allowing GeneratorExp: `any`/`all`/`sum`/`next` are in
        ALLOWED_BUILTINS and their idiomatic argument *is* a generator, so
        forbidding the expression while permitting the functions was internally
        inconsistent -- and it penalised stronger models specifically, since they
        reach for `next(x for x in xs if ...)` precisely because it is idiomatic.
        """
        assert validate(program, allowed_names={"xs"}).valid


class TestDunderStringsAreNotASink:
    """Dunder *strings* were rejected wholesale; the sink they fed is closed anyway.

    The blanket rule stopped ``getattr(obj, "__class__")`` -> ``__bases__`` ->
    ``__subclasses__()``. Every step of that chain is independently unreachable,
    so the rule cost the ability to write or read ``__init__.py`` and to emit
    ``__main__`` guards while adding nothing. It was also bypassable by splitting
    the literal, so it never carried the weight it appeared to.

    These assert the closure directly, so removing the rule cannot quietly
    re-open a route.
    """

    @pytest.mark.parametrize("program", [
        "getattr(obj, '__class__')",
        "getattr(obj, '_' + '_class_' + '_')",          # the split-literal bypass
        "getattr(obj, f'__{name}__')",
        "globals()['__builtins__']",
        "vars(obj)['__class__']",
        "type(obj).__bases__",
        "__import__('os')",
    ])
    def test_string_mediated_reflection_rejected(self, program):
        r = validate(program, allowed_names={"obj", "name"})
        assert not r.valid, f"{program!r} unexpectedly validated"

    @pytest.mark.parametrize("program", [
        "obj.__class__",
        "obj.__class__.__bases__",
        "obj.__class__.__subclasses__()",
    ])
    def test_direct_dunder_attributes_still_rejected(self, program):
        """Blocked by the underscore prefix rule, not by any string check."""
        assert _attr_rejected(validate(program, allowed_names={"obj"}))

    def test_dunder_string_in_a_subscript_is_permitted_and_inert(self):
        """The one behaviour removal newly allows, recorded deliberately.

        ``obj["__class__"]`` is ``__getitem__`` -- a key lookup, not attribute
        access. Reaching a namespace from the result still requires
        ``.__bases__``, which the prefix rule rejects (asserted above). Kept as a
        test so a future reader sees it was considered rather than overlooked.
        """
        assert validate("obj['__class__']", allowed_names={"obj"}).valid
        assert not validate("obj['__class__'].__bases__", allowed_names={"obj"}).valid
