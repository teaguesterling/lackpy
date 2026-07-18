"""Integration tests for the LiterateInterpreter."""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.lang.grader import Grade
from lackpy.tools.registry import ResolvedTools
from lackpy.tools.toolbox import ToolSpec


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "hello.txt").write_text("Hello World\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    return 42\n")
    return tmp_path


class TestValidation:
    def test_valid_document(self, interpreter, context):
        doc = "Hello\n\n```lackpy\nx = 1\n```"
        result = interpreter.validate(doc, context)
        assert result.valid

    def test_invalid_document(self, interpreter, context):
        doc = "```lackpy @read\n```"
        result = interpreter.validate(doc, context)
        assert not result.valid
        assert any("requires a path" in e for e in result.errors)


class TestProseExecution:
    @pytest.mark.asyncio
    async def test_prose_only(self, interpreter, context):
        doc = "Hello World"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "Hello World" in result.output

    @pytest.mark.asyncio
    async def test_prose_with_interpolation(self, interpreter, context):
        doc = "```lackpy @hidden\nname = 'Alice'\n```\n\nHello {name}!"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "Hello Alice!" in result.output

    @pytest.mark.asyncio
    async def test_multiline_prose(self, interpreter, context):
        doc = "Line one\n\nLine two\n\nLine three"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "Line one" in result.output
        assert "Line three" in result.output


class TestCodeExecution:
    @pytest.mark.asyncio
    async def test_code_block(self, interpreter, context):
        doc = "```lackpy\nx = 2 + 2\n```\n\nResult: {x}"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "Result: 4" in result.output

    @pytest.mark.asyncio
    async def test_hidden_block(self, interpreter, context):
        doc = "```lackpy @hidden\nsetup = True\n```\n\nSetup done: {setup}"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "Setup done: True" in result.output

    @pytest.mark.asyncio
    async def test_scratch_block(self, interpreter, context):
        doc = "```lackpy @scratch\na = 10\nb = 'hello'\n```"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "[scratch:" in result.output
        assert "a=int" in result.output
        assert "b=str" in result.output


class TestFileOperations:
    @pytest.mark.asyncio
    async def test_read_annotation(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        doc = "```lackpy @read(hello.txt)\n```"
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "Hello World" in result.output

    @pytest.mark.asyncio
    async def test_write_annotation(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        doc = '```lackpy @write(output.txt)\nNew file content\n```'
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "New file content" in (workspace / "output.txt").read_text()

    @pytest.mark.asyncio
    async def test_diff_annotation(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        doc = """\
```lackpy @diff(hello.txt)
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-Hello World
+Hello Universe
```"""
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "Hello Universe" in (workspace / "hello.txt").read_text()

    @pytest.mark.asyncio
    async def test_read_file_tool(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        doc = '```lackpy\ncontent = read_file("hello.txt")\n```\n\nFile says: {content}'
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "Hello World" in result.output


class TestGatherContinue:
    @pytest.mark.asyncio
    async def test_gather_blocks_execute(self, interpreter, context):
        doc = """\
```lackpy @gather
a = 1 + 1
```

```lackpy @gather
b = 2 + 2
```

```lackpy @continue
```

a={a}, b={b}"""
        result = await interpreter.execute(doc, context)
        assert result.success
        assert "a=2" in result.output
        assert "b=4" in result.output

    @pytest.mark.asyncio
    async def test_continue_sets_metadata(self, interpreter, context):
        doc = "```lackpy @continue\n```"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert result.metadata.get("continue_requested") is True

    @pytest.mark.asyncio
    async def test_no_continue_metadata(self, interpreter, context):
        doc = "```lackpy\nx = 1\n```"
        result = await interpreter.execute(doc, context)
        assert result.success
        assert result.metadata.get("continue_requested") is False


class TestMetadata:
    @pytest.mark.asyncio
    async def test_output_format(self, interpreter, context):
        doc = "Hello"
        result = await interpreter.execute(doc, context)
        assert result.output_format == "markdown"

    @pytest.mark.asyncio
    async def test_cell_count(self, interpreter, context):
        doc = "Prose\n\n```lackpy\nx = 1\n```\n\nMore prose"
        result = await interpreter.execute(doc, context)
        assert result.metadata["cell_count"] == 3

    @pytest.mark.asyncio
    async def test_variables_captured(self, interpreter, context):
        doc = "```lackpy\nresult = 42\n```"
        result = await interpreter.execute(doc, context)
        assert result.metadata["variables"].get("result") == 42

    @pytest.mark.asyncio
    async def test_duration_tracked(self, interpreter, context):
        doc = "Hello"
        result = await interpreter.execute(doc, context)
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_variables_exclude_internals_and_tools(self, interpreter, context):
        doc = """\
```lackpy @gather
import os
items = [1, 2, 3]
for x in items:
    pass
result = sum(items)
```

```lackpy @continue
```"""
        result = await interpreter.execute(doc, context)
        assert result.success
        variables = result.metadata["variables"]
        assert "result" in variables
        assert "items" in variables
        assert "read_file" not in variables
        assert "write_file" not in variables
        assert "__builtins__" not in variables
        assert "__continue_requested__" not in variables


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_runtime_error(self, interpreter, context):
        doc = "```lackpy\nx = 1 / 0\n```"
        result = await interpreter.execute(doc, context)
        assert not result.success
        assert "division by zero" in result.error

    @pytest.mark.asyncio
    async def test_name_error(self, interpreter, context):
        doc = "```lackpy\nprint(undefined_var)\n```"
        result = await interpreter.execute(doc, context)
        assert not result.success
        assert "undefined_var" in result.error


class TestRegistration:
    def test_registered_as_literate(self):
        from lackpy.interpreters import get_interpreter
        cls = get_interpreter("literate")
        assert cls is LiterateInterpreter

    def test_system_prompt_hint(self, interpreter):
        hint = interpreter.system_prompt_hint()
        assert "literate" in hint.lower()
        assert "```lackpy" in hint
        assert "@gather" in hint


class TestFullDocument:
    @pytest.mark.asyncio
    async def test_analysis_report(self, interpreter, workspace):
        ctx = ExecutionContext(base_dir=workspace)
        doc = """\
---
echo: true
output: auto
---
# File Analysis

```lackpy @hidden
content = read_file("src/main.py")
lines = content.strip().split("\\n")
```

The file `src/main.py` has {len(lines)} lines.

```lackpy @gather
first_line = lines[0]
```

```lackpy @continue
```

The first line is: {first_line}"""

        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "2 lines" in result.output
        assert "def main():" in result.output


class TestCeilingGate:
    """The effect ceiling gate: refuse a document whose aggregate effects exceed
    the context's grade_ceiling -- statically, before any cell runs."""

    @pytest.mark.asyncio
    async def test_gate_refuses_doc_over_ceiling(self, interpreter, tmp_path):
        # A write-grade (w=3) doc under a read-only ceiling (w=1) is refused
        # before running -- the target file must NOT be created.
        ctx = ExecutionContext(base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)})
        doc = "```lackpy @write(out.py)\nvalue = 1\n```\n"
        result = await interpreter.execute(doc, ctx)
        assert not result.success
        assert "effect ceiling exceeded" in result.error
        assert "w=3" in result.error
        assert not (tmp_path / "out.py").exists()  # gate ran before the write

    @pytest.mark.asyncio
    async def test_gate_allows_doc_within_ceiling(self, interpreter, tmp_path):
        ctx = ExecutionContext(base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)})
        doc = "```lackpy @hidden\nx = 2 + 2\n```\n\nResult: {x}"
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "Result: 4" in result.output

    @pytest.mark.asyncio
    async def test_no_ceiling_means_no_gate(self, interpreter, tmp_path):
        # Without a grade_ceiling the doc runs as before (writes the file).
        ctx = ExecutionContext(base_dir=tmp_path)
        doc = "```lackpy @write(out.py)\nvalue = 1\n```\n"
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert (tmp_path / "out.py").exists()

    @pytest.mark.asyncio
    async def test_scratch_directive_allowed_under_read_ceiling(self, interpreter, workspace):
        # Review #2/#8: @scratch compiles to locals(); it must NOT be refused as
        # an escape hatch under a read-only ceiling.
        ctx = ExecutionContext(base_dir=workspace, config={"grade_ceiling": Grade(1, 1)})
        doc = "```lackpy @read(hello.txt)\n```\n\n```lackpy @scratch\nx = 1\n```"
        result = await interpreter.execute(doc, ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_injected_tool_is_graded_by_the_gate(self, interpreter, tmp_path):
        # Review #1/#3: a profile-injected write-capable tool must be graded by
        # the gate, not slip under a read-only ceiling as a pure call.
        called = []
        spec = ToolSpec(name="custom_writer", provider="python", description="w",
                        args=[], returns="bool", grade_w=3, effects_ceiling=3)
        resolved = ResolvedTools(
            tools={"custom_writer": spec},
            callables={"custom_writer": lambda p, d: called.append(p)},
            grade=Grade(3, 3), description="custom",
        )
        doc = '```lackpy\ncustom_writer("out.txt", "data")\n```'

        ctx = ExecutionContext(base_dir=tmp_path, tools=resolved,
                               config={"grade_ceiling": Grade(1, 1)})
        result = await interpreter.execute(doc, ctx)
        assert not result.success                 # refused
        assert "w=3" in result.error
        assert called == []                       # tool never ran

        ctx2 = ExecutionContext(base_dir=tmp_path, tools=resolved,
                                config={"grade_ceiling": Grade(3, 3)})
        result2 = await interpreter.execute(doc, ctx2)
        assert result2.success and called          # allowed + ran under a write ceiling

    @pytest.mark.asyncio
    async def test_syntax_error_reports_kernel_error_not_ceiling(self, interpreter, tmp_path):
        # Review #9: a cell with a Python typo under a low ceiling must surface the
        # real syntax error, not a misleading "effect ceiling exceeded".
        ctx = ExecutionContext(base_dir=tmp_path, config={"grade_ceiling": Grade(1, 1)})
        result = await interpreter.execute("```lackpy\nx = (1 +\n```", ctx)
        assert not result.success
        assert "ceiling" not in (result.error or "")
        assert "syntax" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_ceiling_accepts_a_pair_or_int(self, interpreter, tmp_path):
        # Review #6: a config/TOML-sourced ceiling (list/int) is coerced.
        for ceiling in ([1, 1], 1):
            ctx = ExecutionContext(base_dir=tmp_path, config={"grade_ceiling": ceiling})
            result = await interpreter.execute("```lackpy @write(o.py)\nv=1\n```", ctx)
            assert not result.success
            assert "ceiling" in result.error

    @pytest.mark.asyncio
    async def test_none_config_does_not_crash(self, interpreter, tmp_path):
        # Review #4: tolerate config=None (against the dict default-factory contract).
        ctx = ExecutionContext(base_dir=tmp_path, config=None)
        result = await interpreter.execute("Hello", ctx)
        assert result.success

    @staticmethod
    def _toolset(grade_w):
        spec = ToolSpec(name="reader", provider="python", description="r",
                        args=[], returns="str", grade_w=grade_w, effects_ceiling=grade_w)
        return ResolvedTools(tools={"reader": spec}, callables={"reader": lambda: "x"},
                             grade=Grade(grade_w, grade_w), description="t")

    @pytest.mark.asyncio
    async def test_ceiling_defaults_to_granted_toolset_grade(self, interpreter, tmp_path):
        # Profile -> ceiling wiring: with NO explicit ceiling, the gate caps the
        # document at the granted toolset's grade. A read-only toolset (w=1)
        # refuses a write-builtin doc; a write-capable toolset (w=3) allows it.
        write_doc = "```lackpy @write(o.py)\nv = 1\n```"

        ro = ExecutionContext(base_dir=tmp_path, tools=self._toolset(1))
        result = await interpreter.execute(write_doc, ro)
        assert not result.success and "w=3" in result.error
        assert not (tmp_path / "o.py").exists()

        rw = ExecutionContext(base_dir=tmp_path, tools=self._toolset(3))
        assert (await interpreter.execute(write_doc, rw)).success

    @pytest.mark.asyncio
    async def test_explicit_ceiling_overrides_toolset_grade(self, interpreter, tmp_path):
        # An explicit config ceiling wins over the toolset-grade default.
        ctx = ExecutionContext(base_dir=tmp_path, tools=self._toolset(3),
                               config={"grade_ceiling": Grade(1, 1)})
        result = await interpreter.execute("```lackpy @write(o.py)\nv=1\n```", ctx)
        assert not result.success


class TestWritesKeptUnderForgiveness:
    """DELIBERATE CONTRACT CHANGE (L1.2 — design conflict #2, option (c),
    decided by Teague 2026-07-17). Formerly TestTransactionalWrites: "a failed
    document rolls its writes back."

    The write journal's failure-rollback is retired together with the abort it
    served: a cell failure now reifies as a bound value + `error_reified`
    ledger entry and the run COMPLETES — file writes stand like every other
    binding (state kept), while the aggregate result still reports Left.
    (FileJournal remains an unwired component; its unit tests in
    test_journal.py are unchanged.)
    """

    OK_CEILING = {"grade_ceiling": Grade(3, 3)}  # gate present but permissive: writes allowed

    @pytest.mark.asyncio
    async def test_write_persists_when_a_later_cell_fails(self, interpreter, tmp_path):
        # WAS test_failure_removes_a_newly_created_file (rollback).
        doc = ("```lackpy @write(out.txt)\nfresh\n```\n\n"
               "```lackpy\nraise ValueError('boom')\n```")
        ctx = ExecutionContext(base_dir=tmp_path, config=self.OK_CEILING)
        result = await interpreter.execute(doc, ctx)
        assert not result.success                              # aggregate Left
        assert (tmp_path / "out.txt").read_text() == "fresh"   # write KEPT
        assert result.metadata["ledger"].query(entry_type="error_reified")

    @pytest.mark.asyncio
    async def test_overwrite_persists_when_a_later_cell_fails(self, interpreter, tmp_path):
        # WAS test_failure_restores_overwritten_file (rollback to ORIGINAL).
        (tmp_path / "keep.txt").write_text("ORIGINAL")
        doc = ("```lackpy @write(keep.txt)\nNEW\n```\n\n"
               "```lackpy\nx = 1 / 0\n```")
        ctx = ExecutionContext(base_dir=tmp_path, config=self.OK_CEILING)
        result = await interpreter.execute(doc, ctx)
        assert not result.success
        assert (tmp_path / "keep.txt").read_text() == "NEW"    # KEPT, not restored

    @pytest.mark.asyncio
    async def test_success_commits_the_write(self, interpreter, tmp_path):
        ctx = ExecutionContext(base_dir=tmp_path, config=self.OK_CEILING)
        result = await interpreter.execute("```lackpy @write(out.txt)\nkept\n```", ctx)
        assert result.success
        assert (tmp_path / "out.txt").read_text() == "kept"

    @pytest.mark.asyncio
    async def test_write_persists_without_a_ceiling(self, interpreter, tmp_path):
        # WAS test_rollback_happens_without_a_ceiling. Forgiveness, like the
        # old journal, does not depend on a ceiling or a granted toolset.
        doc = ("```lackpy @write(out.txt)\nfresh\n```\n\n"
               "```lackpy\nraise ValueError('boom')\n```")
        ctx = ExecutionContext(base_dir=tmp_path)  # no ceiling, no tools
        result = await interpreter.execute(doc, ctx)
        assert not result.success
        assert (tmp_path / "out.txt").read_text() == "fresh"

    @pytest.mark.asyncio
    async def test_diff_persists_when_a_later_cell_fails(self, interpreter, tmp_path):
        # WAS test_failure_restores_a_diffed_file (rollback to Hello World).
        (tmp_path / "hello.txt").write_text("Hello World\n")
        doc = ("```lackpy @diff(hello.txt)\n"
               "--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n"
               "-Hello World\n+Hello Universe\n```\n\n"
               "```lackpy\nraise RuntimeError('boom')\n```")
        ctx = ExecutionContext(base_dir=tmp_path, config=self.OK_CEILING)
        result = await interpreter.execute(doc, ctx)
        assert not result.success
        assert (tmp_path / "hello.txt").read_text() == "Hello Universe\n"  # KEPT


class TestInjectedToolWritesKept:
    """DELIBERATE CONTRACT CHANGE (L1.2) — formerly TestInjectedToolJournaling,
    which asserted that declared path metadata made an injected write-tool's
    effect roll back on failure. With the rollback retired, both the precise
    (declared path) and heuristic (no path) specs behave the same at this
    level: the write persists and the failure is ledgered. The metadata still
    drives effect grading/classification for the ceiling gate.
    """

    @staticmethod
    def _writer_tools(tmp_path, *, precise):
        def mywrite(path, content):
            (tmp_path / path).write_text(content)

        meta = dict(effect_kind="write", path_arg="path", path_index=0) if precise else {}
        spec = ToolSpec(name="mywrite", provider="python", description="w",
                        args=[], returns="None", grade_w=3, effects_ceiling=3, **meta)
        return ResolvedTools(tools={"mywrite": spec}, callables={"mywrite": mywrite},
                             grade=Grade(3, 3), description="w")

    DOC = '```lackpy\nmywrite("out.txt", "data")\n```\n\n```lackpy\nraise ValueError("boom")\n```'

    @pytest.mark.asyncio
    @pytest.mark.parametrize("precise", [True, False])
    async def test_injected_write_persists_on_failure(self, interpreter, tmp_path, precise):
        resolved = self._writer_tools(tmp_path, precise=precise)
        ctx = ExecutionContext(base_dir=tmp_path, tools=resolved,
                               config={"grade_ceiling": Grade(3, 3)})
        result = await interpreter.execute(self.DOC, ctx)
        assert not result.success                              # aggregate Left
        assert (tmp_path / "out.txt").read_text() == "data"    # write KEPT
        assert result.metadata["ledger"].query(entry_type="error_reified")
