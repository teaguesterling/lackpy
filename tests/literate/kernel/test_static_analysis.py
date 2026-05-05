"""Tests for static analysis (compile check + name resolution)."""

import pytest

from lackpy.interpreters.literate.kernel.static_analysis import (
    StaticAnalysisError,
    check_cell,
)


class TestCompileCheck:
    def test_valid_code_passes(self):
        check_cell("x = 42", known_names=set())

    def test_syntax_error_detected(self):
        with pytest.raises(StaticAnalysisError, match="syntax"):
            check_cell("x = ", known_names=set())

    def test_malformed_fstring_detected(self):
        # f'{x!q}' uses an invalid conversion specifier — caught at compile time
        with pytest.raises(StaticAnalysisError, match="syntax"):
            check_cell("print(f'{x!q}')", known_names={"x"})

    def test_multiline_code_passes(self):
        check_cell("x = 1\ny = x + 1", known_names=set())


class TestNameResolution:
    def test_defined_name_passes(self):
        check_cell("print(x)", known_names={"x", "print"})

    def test_undefined_name_detected(self):
        with pytest.raises(StaticAnalysisError, match="undefined.*'foo'"):
            check_cell("print(foo)", known_names={"print"})

    def test_builtins_always_available(self):
        check_cell("x = len([1, 2, 3])", known_names=set())

    def test_self_defining_passes(self):
        check_cell("x = 42\nprint(x)", known_names={"print"})

    def test_import_passes(self):
        check_cell("import os\nos.getcwd()", known_names=set())

    def test_for_loop_defines_target(self):
        check_cell("for i in range(10):\n    print(i)", known_names={"print"})

    def test_comprehension_variable_not_leaked(self):
        check_cell("result = [x for x in items]", known_names={"items"})

    def test_augmented_assign_requires_existing(self):
        with pytest.raises(StaticAnalysisError, match="undefined.*'counter'"):
            check_cell("counter += 1", known_names=set())

    def test_function_def_defines_name(self):
        check_cell("def helper():\n    return 42\nhelper()", known_names=set())
