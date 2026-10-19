"""Typed ``params`` (stepy100 Plan 2, Task 5; design §4.1, D15).

The program's own top-level ``params = {...}`` literal holds the defaults, so
the program is plain Python on its own. Passed params replace the literal's
values before compile. Nothing binds a bare name: a program reads
``params["n"]``, exactly as it would under python3.
"""

import pytest

from lackpy.run.runner import RestrictedRunner


@pytest.fixture
def runner():
    return RestrictedRunner()


class TestOverride:
    def test_passed_value_overrides_the_default(self, runner):
        # Before Task 5 the program's own value silently won (measured): a bare
        # name `n` was injected, and the program's `n = 3` replaced it.
        result = runner.run('params = {"n": 3}\nparams["n"] * 2', params={"n": 5})
        assert result.success, result.error
        assert result.output == 10

    def test_no_params_passed_runs_the_defaults(self, runner):
        result = runner.run('params = {"n": 3}\nparams["n"] * 2')
        assert result.success, result.error
        assert result.output == 6

    def test_only_passed_keys_change(self, runner):
        result = runner.run('params = {"a": 1, "b": "x"}\nparams', params={"b": "y"})
        assert result.success, result.error
        assert result.output == {"a": 1, "b": "y"}

    def test_container_values_are_replaced(self, runner):
        result = runner.run('params = {"xs": [0], "m": {}}\nparams',
                            params={"xs": [1, (2, 3)], "m": {"k": {4}}})
        assert result.success, result.error
        assert result.output == {"xs": [1, (2, 3)], "m": {"k": {4}}}

    def test_empty_set_is_a_literal(self, runner):
        result = runner.run('params = {"s": {1}}\nparams["s"]', params={"s": set()})
        assert result.success, result.error
        assert result.output == set()

    def test_int_is_accepted_for_a_float_default(self, runner):
        result = runner.run('params = {"r": 0.5}\nparams["r"]', params={"r": 2})
        assert result.success, result.error
        assert result.output == 2

    @pytest.mark.parametrize("passed", [5, "text", [1, "a"], {"k": (1, None)}, None])
    def test_none_default_accepts_any_literal_value(self, runner, passed):
        # Review M3: a None default has no type to infer; Plan 3 docblock types tighten it.
        result = runner.run('params = {"limit": None}\nparams["limit"]', params={"limit": passed})
        assert result.success, result.error
        assert result.output == passed

    def test_none_default_still_needs_a_literal_form(self, runner):
        result = runner.run('params = {"limit": None}\nparams', params={"limit": len})
        assert not result.success
        assert "'limit'" in result.error

    def test_params_bind_no_bare_name(self, runner):
        result = runner.run('params = {"n": 3}\nn', params={"n": 5})
        assert not result.success
        assert "NameError" in result.error or "'n'" in result.error

    def test_a_param_named_like_a_builtin_does_not_replace_it(self, runner):
        # Ruling (a): the bind-time collision check goes because params bind no names.
        result = runner.run('params = {"len": "x"}\nlen("ab")', params={"len": "caller"})
        assert result.success, result.error
        assert result.output == 2


class TestErrors:
    def test_unknown_key(self, runner):
        result = runner.run('params = {"n": 3}\nparams["n"]', params={"m": 1})
        assert not result.success
        assert "'m'" in result.error

    @pytest.mark.parametrize("default, passed", [
        ("3", "5"),
        ("3", 5.0),
        ("3", True),       # bool is not int here
        ("True", 1),
        ("'a'", None),
        ("[1]", (1,)),
    ])
    def test_type_mismatch(self, runner, default, passed):
        result = runner.run(f'params = {{"n": {default}}}\nparams["n"]', params={"n": passed})
        assert not result.success
        assert "'n'" in result.error

    def test_no_params_literal(self, runner):
        result = runner.run("x = 1", params={"n": 5})
        assert not result.success
        assert "'n'" in result.error

    def test_value_not_expressible_as_a_literal(self, runner):
        # Ruling (c): the rewrite needs a literal AST node.
        result = runner.run('params = {"f": {}}\nparams', params={"f": {"g": len}})
        assert not result.success
        assert "'f'" in result.error

    def test_non_literal_default_is_an_error_when_params_are_passed(self, runner):
        result = runner.run('k = "n"\nparams = {k: 3}\nparams', params={"n": 5})
        assert not result.success
        assert "'n'" in result.error

    def test_non_dict_params_is_an_error_when_params_are_passed(self, runner):
        result = runner.run("params = dict(n=3)\nparams", params={"n": 5})
        assert not result.success
        assert "'n'" in result.error

    def test_two_params_assignments_is_an_error(self, runner):
        result = runner.run('params = {"n": 3}\nparams = {"n": 4}\nparams', params={"n": 5})
        assert not result.success
        assert "'n'" in result.error

    def test_nested_params_assignment_is_an_error(self, runner):
        program = 'params = {"n": 3}\nif True:\n    params = {"n": 4}\nparams'
        result = runner.run(program, params={"n": 5})
        assert not result.success
        assert "'n'" in result.error

    def test_function_local_params_binding_is_counted(self, runner):
        # Review M6: any binding of the name counts, so the message says "anywhere in the program".
        program = 'params = {"n": 3}\ndef f():\n    params = 2\n    return params\nparams'
        result = runner.run(program, params={"n": 5})
        assert not result.success
        assert "anywhere in the program" in result.error

    def test_odd_shapes_run_unchanged_without_passed_params(self, runner):
        result = runner.run('k = "n"\nparams = {k: 3}\nparams = dict(params)\nparams')
        assert result.success, result.error
        assert result.output == {"n": 3}
