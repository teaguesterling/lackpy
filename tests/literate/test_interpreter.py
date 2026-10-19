"""Integration tests for the LiterateInterpreter."""

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.lang.grader import Grade
from lackpy.tools.registry import ResolvedTools
from lackpy.tools.toolbox import ToolSpec
from lackpy.run.paths import load_policy


@pytest.fixture
def interpreter():
    return LiterateInterpreter()


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


def _pctx(base, tmp_home, **config):
    """A context whose path policy is ``base``, with a temporary home."""
    policy = load_policy(base, home=tmp_home, managed=None)
    return ExecutionContext(base_dir=base, config={"path_policy": policy, **config})


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
    async def test_read_annotation(self, interpreter, workspace, tmp_home):
        ctx = _pctx(workspace, tmp_home)
        doc = "```lackpy @read(hello.txt)\n```"
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "Hello World" in result.output

    @pytest.mark.asyncio
    async def test_write_annotation(self, interpreter, workspace, tmp_home):
        ctx = _pctx(workspace, tmp_home)
        doc = '```lackpy @write(output.txt)\nNew file content\n```'
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert "New file content" in (workspace / "output.txt").read_text()

    @pytest.mark.asyncio
    async def test_diff_annotation(self, interpreter, workspace, tmp_home):
        ctx = _pctx(workspace, tmp_home)
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
    async def test_read_into_a_name(self, interpreter, workspace, tmp_home):
        # Was test_read_file_tool (the kit's read_file): the same read through Path.
        ctx = _pctx(workspace, tmp_home)
        doc = ('```lackpy\nfrom pathlib import Path\ncontent = Path("hello.txt").read_text()\n```'
               '\n\nFile says: {content}')
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
import json
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
        assert "<compute>" in hint
        assert "<compute gather>" in hint


class TestFullDocument:
    @pytest.mark.asyncio
    async def test_analysis_report(self, interpreter, workspace, tmp_home):
        policy = load_policy(workspace, home=tmp_home, managed=None)
        ctx = ExecutionContext(base_dir=workspace, config={"path_policy": policy})
        doc = """\
---
echo: true
output: auto
---
# File Analysis

```lackpy @hidden
from pathlib import Path
content = Path("src/main.py").read_text()
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
    async def test_no_ceiling_means_no_gate(self, interpreter, tmp_path, tmp_home):
        # Without a grade_ceiling the doc runs as before (writes the file).
        ctx = _pctx(tmp_path, tmp_home)
        doc = "```lackpy @write(out.py)\nvalue = 1\n```\n"
        result = await interpreter.execute(doc, ctx)
        assert result.success
        assert (tmp_path / "out.py").exists()

    @pytest.mark.asyncio
    async def test_scratch_directive_allowed_under_read_ceiling(self, interpreter, workspace,
                                                                tmp_home):
        # Review #2/#8: @scratch (and a @read) must NOT be refused under a
        # read-only ceiling.
        ctx = _pctx(workspace, tmp_home, grade_ceiling=Grade(1, 1))
        doc = "```lackpy @read(hello.txt)\n```\n\n```lackpy @scratch\nx = 1\n```"
        result = await interpreter.execute(doc, ctx)
        assert result.success

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
    async def test_ceiling_defaults_to_granted_toolset_grade(self, interpreter, tmp_path,
                                                             tmp_home):
        # Profile -> ceiling wiring: with NO explicit ceiling, the gate caps the
        # document at the granted toolset's grade. A read-only toolset (w=1)
        # refuses a write-builtin doc; a write-capable toolset (w=3) allows it.
        write_doc = "```lackpy @write(o.py)\nv = 1\n```"

        ro = ExecutionContext(base_dir=tmp_path, tools=self._toolset(1))
        result = await interpreter.execute(write_doc, ro)
        assert not result.success and "w=3" in result.error
        assert not (tmp_path / "o.py").exists()

        rw = _pctx(tmp_path, tmp_home)
        rw.tools = self._toolset(3)
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
    async def test_write_persists_when_a_later_cell_fails(self, interpreter, tmp_path, tmp_home):
        # WAS test_failure_removes_a_newly_created_file (rollback).
        doc = ("```lackpy @write(out.txt)\nfresh\n```\n\n"
               "```lackpy\nfail = {}['boom']\n```")
        ctx = _pctx(tmp_path, tmp_home, **self.OK_CEILING)
        result = await interpreter.execute(doc, ctx)
        assert not result.success                              # aggregate Left
        assert (tmp_path / "out.txt").read_text() == "fresh"   # write KEPT
        assert result.metadata["ledger"].query(entry_type="error_reified")

    @pytest.mark.asyncio
    async def test_overwrite_persists_when_a_later_cell_fails(self, interpreter, tmp_path, tmp_home):
        # WAS test_failure_restores_overwritten_file (rollback to ORIGINAL).
        (tmp_path / "keep.txt").write_text("ORIGINAL")
        doc = ("```lackpy @write(keep.txt)\nNEW\n```\n\n"
               "```lackpy\nx = 1 / 0\n```")
        ctx = _pctx(tmp_path, tmp_home, **self.OK_CEILING)
        result = await interpreter.execute(doc, ctx)
        assert not result.success
        assert (tmp_path / "keep.txt").read_text() == "NEW"    # KEPT, not restored

    @pytest.mark.asyncio
    async def test_success_commits_the_write(self, interpreter, tmp_path, tmp_home):
        ctx = _pctx(tmp_path, tmp_home, **self.OK_CEILING)
        result = await interpreter.execute("```lackpy @write(out.txt)\nkept\n```", ctx)
        assert result.success
        assert (tmp_path / "out.txt").read_text() == "kept"

    @pytest.mark.asyncio
    async def test_write_persists_without_a_ceiling(self, interpreter, tmp_path, tmp_home):
        # WAS test_rollback_happens_without_a_ceiling. Forgiveness, like the
        # old journal, does not depend on a ceiling or a granted toolset.
        doc = ("```lackpy @write(out.txt)\nfresh\n```\n\n"
               "```lackpy\nfail = {}['boom']\n```")
        ctx = _pctx(tmp_path, tmp_home)  # no ceiling, no tools
        result = await interpreter.execute(doc, ctx)
        assert not result.success
        assert (tmp_path / "out.txt").read_text() == "fresh"

    @pytest.mark.asyncio
    async def test_diff_persists_when_a_later_cell_fails(self, interpreter, tmp_path, tmp_home):
        # WAS test_failure_restores_a_diffed_file (rollback to Hello World).
        (tmp_path / "hello.txt").write_text("Hello World\n")
        doc = ("```lackpy @diff(hello.txt)\n"
               "--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n"
               "-Hello World\n+Hello Universe\n```\n\n"
               "```lackpy\nfail = {}['boom']\n```")
        ctx = _pctx(tmp_path, tmp_home, **self.OK_CEILING)
        result = await interpreter.execute(doc, ctx)
        assert not result.success
        assert (tmp_path / "hello.txt").read_text() == "Hello Universe\n"  # KEPT



class TestRefusalEndsTheDocument:
    """D4: a refusal stops the run. It is ledgered, the run fails, the
    [kernel] note carries its text, and no later cell runs."""

    DOC = (
        "```lackpy\nbefore = 1\n```\n\n"
        "```lackpy\nfrom pathlib import Path\nt = 1\n```\n\n"
        "```lackpy\nafter = 2\nprint('AFTER RAN')\n```\n\n"
        "Tail {before}\n"
    )

    @pytest.mark.asyncio
    async def test_a_refusal_stops_the_document(self, interpreter, context):
        result = await interpreter.execute(self.DOC, context)
        assert result.success is False
        assert "not allowed in lackpy" in result.error
        assert "AFTER RAN" not in (result.output or "")
        assert "Tail" not in (result.output or "")
        assert "after" not in result.metadata["variables"]
        assert result.metadata["variables"].get("before") == 1
        ledger = result.metadata["ledger"]
        refused = ledger.query(entry_type="refused")
        assert len(refused) == 1
        assert refused[0].detail["cell_index"] == 1
        assert refused[0].detail["error_phase"] == "refused"
        assert "not allowed in lackpy" in refused[0].detail["error"]
        md = result.metadata["rendered_markdown"]
        assert "[kernel]" in md and "refused" in md and "not allowed in lackpy" in md

    @pytest.mark.asyncio
    async def test_a_refusal_is_never_reified_as_a_hole_or_error_value(
            self, interpreter, context):
        """Review Focus 4: forgiveness must not turn a refusal into a value."""
        from lackpy.interpreters.literate.kernel.forgiveness import (
            ERROR_REIFIED, HOLE_OPENED, SOURCE_UNAVAILABLE, is_forgiving)
        result = await interpreter.execute(self.DOC, context)
        ledger = result.metadata["ledger"]
        for kind in (HOLE_OPENED, ERROR_REIFIED, SOURCE_UNAVAILABLE):
            assert ledger.query(entry_type=kind) == []
        versions = result.metadata["versions"]
        for name in ("t", "Path"):
            current = versions.current(name)
            assert current is None or not is_forgiving(current.value)

    @pytest.mark.asyncio
    async def test_a_refusal_inside_try_stops_the_document(
            self, interpreter, workspace, tmp_home):
        """Gate 4: a handler cannot catch a refusal."""
        policy = load_policy(workspace, home=tmp_home, managed=None)
        ctx = ExecutionContext(base_dir=workspace, config={"path_policy": policy})
        doc = ("```lackpy\nfrom pathlib import Path\n```\n\n"
               "```lackpy\ntry:\n    t = Path('/etc/hostname').read_text()\n"
               "except KeyError:\n    t = ''\n```\n\n```lackpy\nafter = 1\n```\n")
        result = await interpreter.execute(doc, ctx)
        assert result.success is False
        assert "after" not in result.metadata["variables"]
        assert "t" not in result.metadata["variables"]
        refused = result.metadata["ledger"].query(entry_type="refused")
        assert [e.detail["cell_index"] for e in refused] == [1]

    @pytest.mark.asyncio
    async def test_a_refusal_in_a_prose_interpolation_stops_the_document(
            self, interpreter, workspace, tmp_home):
        policy = load_policy(workspace, home=tmp_home, managed=None)
        ctx = ExecutionContext(base_dir=workspace, config={"path_policy": policy})
        doc = ("```lackpy\nfrom pathlib import Path\n```\n\n"
               "Host: {Path('/etc/hostname').read_text()}\n\n"
               "```lackpy\nafter = 1\n```\n")
        result = await interpreter.execute(doc, ctx)
        assert result.success is False
        assert "/etc/hostname" in result.error
        assert "after" not in result.metadata["variables"]
        refused = result.metadata["ledger"].query(entry_type="refused")
        assert [e.detail["cell_index"] for e in refused] == [1]


class TestCellsRunThroughTheSession:
    @pytest.mark.asyncio
    async def test_host_builtins_are_not_reachable(self, interpreter, context):
        doc = "```lackpy\nt = open('/etc/hostname').read()\n```\n"
        result = await interpreter.execute(doc, context)
        assert result.success is False
        assert "Validation failed" in result.error and "open" in result.error
        assert "t" not in result.metadata["variables"]

    @pytest.mark.asyncio
    async def test_no_cwd_change(self, interpreter, context, tmp_path):
        import os
        before = os.getcwd()
        await interpreter.execute("```lackpy\nx = 1\n```\n", context)
        assert os.getcwd() == before

    @pytest.mark.asyncio
    async def test_a_path_reads_under_the_policy(self, interpreter, workspace, tmp_home):
        policy = load_policy(workspace, home=tmp_home, managed=None)
        ctx = ExecutionContext(base_dir=workspace, config={"path_policy": policy})
        doc = ("```lackpy\nfrom pathlib import Path\n"
               "text = Path('hello.txt').read_text()\n```\n\nSays: {text.strip()}\n")
        result = await interpreter.execute(doc, ctx)
        assert result.success, result.error
        assert "Says: Hello World" in result.output

    @pytest.mark.asyncio
    async def test_without_a_policy_pathlib_refuses(self, interpreter, context):
        doc = "```lackpy\nfrom pathlib import Path\n```\n"
        result = await interpreter.execute(doc, context)
        assert result.success is False
        assert result.metadata["ledger"].query(entry_type="refused")

    @pytest.mark.asyncio
    async def test_params_replace_the_defaults_of_the_params_literal(self, interpreter, tmp_path):
        # Final review I4 (design D15): passed params replace the defaults in the
        # document's one params literal; no param binds a bare name. (This test
        # replaced test_params_are_bound_as_data, which encoded bare-name binding.)
        ctx = ExecutionContext(base_dir=tmp_path, params={"n": 3, "cfg": {"k": "v"}})
        doc = ("```lackpy\nparams = {'n': 1, 'cfg': {'k': 'default'}}\n```\n\n"
               "```lackpy\ny = params['n'] * 2\n```\n\nk={params['cfg']['k']} y={y}\n")
        result = await interpreter.execute(doc, ctx)
        assert result.success, result.error
        assert "k=v y=6" in result.output

    @pytest.mark.asyncio
    async def test_a_param_that_is_not_data_is_a_configuration_error(
            self, interpreter, tmp_path):
        # After I4 the value must be refused by the literal rewrite itself: the
        # document has a params literal, and a None default skips the type check.
        ctx = ExecutionContext(base_dir=tmp_path, params={"f": print})
        result = await interpreter.execute("```lackpy\nparams = {'f': None}\n```\n", ctx)
        assert result.success is False
        assert "cannot be written as a Python literal" in result.error


class TestParamsLiteral:
    """Final review I4: the literate axis applies passed params as run_program does
    (design D15): the defaults of the document's one top-level ``params = {...}``
    literal are replaced, in whichever code cell holds it, with the same AST
    rewrite and the same errors as ``RestrictedRunner``. No param binds a bare name."""

    @staticmethod
    def _ctx(tmp_path, **params):
        return ExecutionContext(base_dir=tmp_path, params=params)

    @pytest.mark.asyncio
    async def test_the_reviewers_document_uses_the_passed_value(self, interpreter, tmp_path):
        doc = "```lackpy\nparams = {'n': 1}\nx = params['n'] * 2\nx\n```\n"
        result = await interpreter.execute(doc, self._ctx(tmp_path, n=5))
        assert result.success, result.error
        assert result.metadata["value"] == 10

    @pytest.mark.asyncio
    async def test_no_param_binds_a_bare_name(self, interpreter, tmp_path):
        doc = "```lackpy\nparams = {'n': 1}\n```\n\n```lackpy\nx = n * 2\n```\n"
        result = await interpreter.execute(doc, self._ctx(tmp_path, n=5))
        assert result.success is False
        # ``n`` is an unbound hole, never the passed 5.
        from lackpy.interpreters.literate.kernel.forgiveness import Hole
        assert isinstance(result.metadata["variables"]["n"], Hole)
        assert result.metadata["variables"]["params"] == {"n": 5}

    @pytest.mark.asyncio
    async def test_a_document_with_no_params_literal_fails_loud(self, interpreter, tmp_path):
        result = await interpreter.execute("```lackpy\nx = n * 2\n```\n", self._ctx(tmp_path, n=5))
        assert result.success is False
        assert "no top-level params" in result.error
        assert result.metadata.get("variables") is None  # nothing ran

    @pytest.mark.asyncio
    async def test_params_bound_in_two_cells_fails_loud(self, interpreter, tmp_path):
        doc = "```lackpy\nparams = {'n': 1}\n```\n\n```lackpy\nparams = {'n': 2}\n```\n"
        result = await interpreter.execute(doc, self._ctx(tmp_path, n=5))
        assert result.success is False
        assert "exactly once" in result.error

    @pytest.mark.asyncio
    async def test_an_unknown_key_fails_loud(self, interpreter, tmp_path):
        doc = "```lackpy\nparams = {'n': 1}\n```\n"
        result = await interpreter.execute(doc, self._ctx(tmp_path, m=5))
        assert result.success is False
        assert "'m' is not a key" in result.error

    @pytest.mark.asyncio
    async def test_a_value_of_another_type_fails_loud(self, interpreter, tmp_path):
        doc = "```lackpy\nparams = {'n': 1}\n```\n"
        result = await interpreter.execute(doc, self._ctx(tmp_path, n="five"))
        assert result.success is False
        assert "must be int" in result.error

    @pytest.mark.asyncio
    async def test_the_rest_of_the_cell_is_kept(self, interpreter, tmp_path):
        # Only the literal is rewritten: comments and other lines of its cell stay.
        doc = ("```lackpy\n# defaults\nparams = {'n': 1}  # the count\n"
               "x = params['n']\nx\n```\n")
        result = await interpreter.execute(doc, self._ctx(tmp_path, n=4))
        assert result.success, result.error
        assert result.metadata["value"] == 4
        rendered = result.metadata["rendered_markdown"]
        assert "# defaults" in rendered and "# the count" in rendered

    @pytest.mark.asyncio
    async def test_without_params_the_defaults_run(self, interpreter, tmp_path):
        doc = "```lackpy\nparams = {'n': 1}\nparams['n']\n```\n"
        result = await interpreter.execute(doc, ExecutionContext(base_dir=tmp_path))
        assert result.success, result.error
        assert result.metadata["value"] == 1

    def test_a_session_given_params_fails_loud(self, tmp_path):
        # A multi-round session has no whole document to hold the literal.
        from lackpy.interpreters.literate import LiterateSession
        with pytest.raises(ValueError, match="params"):
            LiterateSession(self._ctx(tmp_path, n=5))
