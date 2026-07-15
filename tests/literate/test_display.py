"""Tests for bounded prose-interpolation rendering (L6).

The truncating display helper affects BOTH execution paths: the compiler
emits the helper call for every interpolated prose expression, and
LightweightKernel (shared by the batch interpreter and the StreamingDriver)
injects the helper into its namespace.
"""

import ast

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.interpreters.literate.display import (
    DEFAULT_DISPLAY_THRESHOLD,
    make_display,
    truncated_display,
)


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class TestTruncatedDisplayUnit:
    def test_small_list_renders_exactly_as_str(self):
        v = [1, 2, 3]
        assert truncated_display(v) == str(v)

    def test_small_value_at_threshold_untouched(self):
        v = list(range(DEFAULT_DISPLAY_THRESHOLD))
        assert truncated_display(v) == str(v)

    def test_large_list_truncates_head_count_tail(self):
        v = list(range(500))
        out = truncated_display(v)
        assert out.startswith("[0, 1, 2, ")
        assert "… n=500 …" in out
        assert out.endswith("497, 498, 499]")
        assert ", 250," not in out

    def test_large_tuple_keeps_parens(self):
        out = truncated_display(tuple(range(100)))
        assert out.startswith("(") and out.endswith(")")
        assert "… n=100 …" in out

    def test_large_dict_truncates_items(self):
        v = {i: i * 2 for i in range(100)}
        out = truncated_display(v)
        assert out.startswith("{0: 0, ")
        assert "… n=100 …" in out
        assert out.endswith("99: 198}")

    def test_large_set_truncates(self):
        out = truncated_display(set(range(100)))
        assert "… n=100 …" in out
        assert out.startswith("{") and out.endswith("}")

    def test_threshold_is_configurable(self):
        v = list(range(6))
        assert truncated_display(v, threshold=5) != str(v)
        assert "… n=6 …" in truncated_display(v, threshold=5)
        assert truncated_display(v, threshold=6) == str(v)

    def test_strings_are_never_truncated(self):
        s = "x" * 10_000
        assert truncated_display(s) == s

    def test_scalars_render_as_str(self):
        assert truncated_display(42) == "42"
        assert truncated_display(3.5) == "3.5"
        assert truncated_display(None) == "None"

    def test_make_display_binds_threshold(self):
        d = make_display(3)
        assert "… n=4 …" in d([1, 2, 3, 4])
        assert d([1, 2]) == "[1, 2]"


class TestRoundTripRefusal:
    """The truncated form is a display artifact, not data — it must not parse
    back as a binding/valid value."""

    def test_truncated_form_is_not_valid_python(self):
        form = truncated_display(list(range(500)))
        with pytest.raises(SyntaxError):
            ast.parse(f"x = {form}")

    @pytest.mark.asyncio
    async def test_truncated_form_fails_as_a_cell_binding(self, interpreter, context):
        form = truncated_display(list(range(500)))
        doc = f"```lackpy\nx = {form}\n```"
        result = await interpreter.execute(doc, context)
        assert not result.success
        assert "syntax" in (result.error or "").lower()


class TestBatchPathRendering:
    @pytest.mark.asyncio
    async def test_large_list_interpolation_truncates(self, interpreter, context):
        doc = (
            "```lackpy @hidden\nbig = list(range(500))\n```\n\n"
            "Data: {big}"
        )
        result = await interpreter.execute(doc, context)
        assert result.success, result.error
        assert "… n=500 …" in result.output
        assert "Data: [0, 1, 2," in result.output
        assert "497, 498, 499]" in result.output
        assert ", 250," not in result.output

    @pytest.mark.asyncio
    async def test_small_value_interpolation_unchanged(self, interpreter, context):
        doc = "```lackpy @hidden\nsmall = [1, 2, 3]\n```\n\nData: {small}"
        result = await interpreter.execute(doc, context)
        assert result.success, result.error
        assert "Data: [1, 2, 3]" in result.output
        assert "…" not in result.output

    @pytest.mark.asyncio
    async def test_threshold_configurable_via_context(self, tmp_path, interpreter):
        ctx = ExecutionContext(base_dir=tmp_path, config={"display_threshold": 5})
        doc = "```lackpy @hidden\nv = list(range(8))\n```\n\nV: {v}"
        result = await interpreter.execute(doc, ctx)
        assert result.success, result.error
        assert "… n=8 …" in result.output

        ctx_big = ExecutionContext(base_dir=tmp_path, config={"display_threshold": 100})
        result2 = await interpreter.execute(doc, ctx_big)
        assert result2.success, result2.error
        assert "… n=8 …" not in result2.output
        assert "V: [0, 1, 2, 3, 4, 5, 6, 7]" in result2.output

    @pytest.mark.asyncio
    async def test_format_spec_interpolation_still_works(self, interpreter, context):
        # {expr:spec} cannot be helper-wrapped; it falls back to the plain
        # f-string (a formatted value is already user-bounded).
        doc = "```lackpy @hidden\nprice = 2.5\n```\n\nPrice: {price:.2f}"
        result = await interpreter.execute(doc, context)
        assert result.success, result.error
        assert "Price: 2.50" in result.output


class TestStreamingPathRendering:
    @pytest.mark.asyncio
    async def test_driver_prose_interpolation_truncates(self):
        # The kernel injects the helper itself, so a bare StreamingDriver
        # (no _build_namespace) still renders bounded.
        from lackpy.interpreters.literate.kernel.driver import StreamingDriver
        from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel

        driver = StreamingDriver(kernel=LightweightKernel())
        await driver.feed("```lackpy @hidden\nbig = list(range(300))\n```\n\nGot {big}\n")
        await driver.flush()
        assert "… n=300 …" in driver.rendered_output
        assert ", 150," not in driver.rendered_output
