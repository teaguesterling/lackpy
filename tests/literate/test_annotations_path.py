"""Annotations on ``Path`` (literate stepy Task 3, design D6/D7).

* ``@read(p)`` compiles to ``print(Path(p).read_text())``.
* ``@write(p)`` compiles to ``Path(p).write_text(<body text>)``: the body is the
  file's content, as it was with the kit's ``write_file``. A path with a
  directory part makes its parent first, as ``write_file`` did.
* Each compiled cell imports ``Path``, so the code cells, joined, stay a plain
  python3 program (Review Focus 5).
* ``@diff(p)`` runs on the host side: the kernel reads ``p`` through the
  session's ``Path`` stand-in, patches in pure Python, and writes through the
  same stand-in. The policy checks both; a refusal ends the document (D4).
  No patch function is ever in a cell's namespace.
* A read-only profile is a policy deny (``read_only_policy``): ``@write`` refuses.
* Strict reactive mode withholds a dependent whose trace shows a ``Path`` write.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.interpreters.literate.compiler import compile_cell, compile_document
from lackpy.interpreters.literate.kernel.reactive import DIRTY
from lackpy.interpreters.literate.parser import Cell
from lackpy.lang import validate
from lackpy.run.paths import load_policy, read_only_policy

DIFF = (
    "```lackpy @diff(hello.txt)\n"
    "--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n"
    "-Hello World\n+Hello Universe\n```\n"
)


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "hello.txt").write_text("Hello World\n")
    return w


@pytest.fixture
def policy(ws, tmp_home):
    return load_policy(ws, home=tmp_home, managed=None)


def _ctx(ws, policy, **config):
    return ExecutionContext(base_dir=ws, config={"path_policy": policy, **config})


async def _run(ctx, doc):
    return await LiterateInterpreter().execute(doc, ctx)


def _refused(result):
    return result.metadata["ledger"].query(entry_type="refused")


# --- the compile targets -------------------------------------------------------

class TestCompileTargets:
    def test_read(self):
        cell = Cell(cell_type="read", content="", annotation_args={"path": "src/main.py"})
        assert compile_cell(cell) == (
            "from pathlib import Path\nprint(Path('src/main.py').read_text())")

    def test_write_is_the_body_text(self):
        cell = Cell(cell_type="write", content="print('hello')\nx = 1",
                    annotation_args={"path": "out.py"})
        assert compile_cell(cell) == (
            "from pathlib import Path\nPath('out.py').write_text(\"print('hello')\\nx = 1\")")

    def test_write_into_a_directory_makes_its_parent(self):
        cell = Cell(cell_type="write", content="hi", annotation_args={"path": "d/e/f.txt"})
        assert compile_cell(cell) == (
            "from pathlib import Path\n"
            "Path('d/e/f.txt').parent.mkdir(parents=True, exist_ok=True)\n"
            "Path('d/e/f.txt').write_text('hi')")

    @pytest.mark.parametrize("cell", [
        Cell(cell_type="read", content="", annotation_args={"path": "a.txt"}),
        Cell(cell_type="write", content="x\n'quoted' \"both\"", annotation_args={"path": "a.txt"}),
        Cell(cell_type="write", content="hi", annotation_args={"path": "d/a.txt"}),
        Cell(cell_type="diff", content="@@ -1 +1 @@\n-a\n+b", annotation_args={"path": "a.txt"}),
    ])
    def test_a_compiled_annotation_is_a_valid_lackpy_program(self, cell):
        verdict = validate(compile_cell(cell), allowed_names=set())
        assert verdict.valid, verdict.violations

    def test_no_kit_name_remains(self):
        doc = ("```lackpy @read(a.txt)\n```\n\n```lackpy @write(b.txt)\nx\n```\n\n" + DIFF)
        compiled = compile_document(doc)
        for kit in ("read_file", "write_file", "apply_diff"):
            assert kit not in compiled

    @pytest.mark.asyncio
    async def test_the_joined_code_cells_are_a_python3_program(self, tmp_path, ws, policy):
        # Review Focus 5: run the joined compiled cells under python3 -I in a
        # plain directory; the files and stdout match the document run.
        doc = ("```lackpy @write(out/notes.txt)\nhello\n```\n\n"
               "```lackpy @read(out/notes.txt)\n```\n\n"
               # Path is bound by the first annotation; a second import would
               # be a rebinding the validator rejects (safety.import).
               "```lackpy\nn = len(Path('out/notes.txt').read_text())\nprint(n)\n```\n")
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "prog.py").write_text(compile_document(doc))
        ran = subprocess.run([sys.executable, "-I", "prog.py"], cwd=plain,
                             capture_output=True, text=True, timeout=60)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == "hello\n5\n"
        assert (plain / "out" / "notes.txt").read_text() == "hello"

        result = await _run(_ctx(ws, policy), doc)
        assert result.success, result.error
        assert result.output == ran.stdout
        assert (ws / "out" / "notes.txt").read_text() == "hello"


# --- @read and @write under a tmp policy --------------------------------------------

class TestReadWriteSequence:
    @pytest.mark.asyncio
    async def test_two_annotations_and_a_code_import(self, ws, policy):
        # A second import of Path is a rebinding the validator rejects, so an
        # annotation imports Path only while the session has not bound it.
        doc = ("```lackpy\nfrom pathlib import Path\nname = 'hello.txt'\n```\n\n"
               "```lackpy @write(a.txt)\nA\n```\n\n"
               "```lackpy @read(a.txt)\n```\n\n"
               "```lackpy\nt = Path(name).read_text()\n```\n")
        result = await _run(_ctx(ws, policy), doc)
        assert result.success, result.error
        assert result.output == "A\n"
        assert result.metadata["variables"]["t"] == "Hello World\n"

    @pytest.mark.asyncio
    async def test_two_annotations_first(self, ws, policy):
        doc = "```lackpy @write(a.txt)\nA\n```\n\n```lackpy @read(a.txt)\n```\n\n" + DIFF
        result = await _run(_ctx(ws, policy), doc)
        assert result.success, result.error
        assert (ws / "hello.txt").read_text() == "Hello Universe\n"

    def test_the_joined_program_imports_path_once(self):
        doc = ("```lackpy @write(a.txt)\nA\n```\n\n```lackpy @read(a.txt)\n```\n\n" + DIFF
               + "\n```lackpy @read(a.txt)\n```\n")
        program = compile_document(doc)
        assert program.count("from pathlib import Path") == 1
        assert validate(program, allowed_names=set()).valid


class TestReadWrite:
    @pytest.mark.asyncio
    async def test_read_prints_the_file(self, ws, policy):
        result = await _run(_ctx(ws, policy), "```lackpy @read(hello.txt)\n```")
        assert result.success, result.error
        assert result.output == "Hello World\n\n"

    @pytest.mark.asyncio
    async def test_write_writes_the_body(self, ws, policy):
        result = await _run(_ctx(ws, policy), "```lackpy @write(out.txt)\nline 1\nline 2\n```")
        assert result.success, result.error
        assert (ws / "out.txt").read_text() == "line 1\nline 2"

    @pytest.mark.asyncio
    async def test_write_makes_a_missing_directory(self, ws, policy):
        result = await _run(_ctx(ws, policy), "```lackpy @write(new/dir/out.txt)\nx\n```")
        assert result.success, result.error
        assert (ws / "new" / "dir" / "out.txt").read_text() == "x"

    @pytest.mark.asyncio
    async def test_read_outside_the_workspace_refuses(self, ws, policy):
        result = await _run(_ctx(ws, policy),
                            "```lackpy @read(/etc/hostname)\n```\n\n```lackpy\nafter = 1\n```")
        assert not result.success
        assert [e.detail["cell_index"] for e in _refused(result)] == [0]
        assert "after" not in result.metadata["variables"]

    @pytest.mark.asyncio
    async def test_write_to_settings_refuses(self, ws, policy):
        result = await _run(_ctx(ws, policy), "```lackpy @write(.claude/settings.json)\n{}\n```")
        assert not result.success
        assert _refused(result)
        assert not (ws / ".claude" / "settings.json").exists()

    @pytest.mark.asyncio
    async def test_without_a_policy_read_refuses(self, ws):
        result = await _run(ExecutionContext(base_dir=ws), "```lackpy @read(hello.txt)\n```")
        assert not result.success
        assert _refused(result)

    @pytest.mark.asyncio
    async def test_the_writes_are_in_the_trace(self, ws, policy):
        interp = LiterateInterpreter()
        from lackpy.interpreters.literate import _build_kernel
        kernel = _build_kernel(_ctx(ws, policy))
        await interp._run_document("```lackpy @write(t.txt)\nx\n```", _ctx(ws, policy), kernel)
        tools = [e.tool for e in kernel.session.trace.entries]
        assert tools == ["Path.write_text"]


# --- @diff on the host side -----------------------------------------------------------

class TestDiff:
    @pytest.mark.asyncio
    async def test_diff_patches_the_file(self, ws, policy):
        result = await _run(_ctx(ws, policy), DIFF)
        assert result.success, result.error
        assert (ws / "hello.txt").read_text() == "Hello Universe\n"

    @pytest.mark.asyncio
    async def test_diff_reads_and_writes_through_the_stand_in(self, ws, policy):
        from lackpy.interpreters.literate import _build_kernel
        ctx = _ctx(ws, policy)
        kernel = _build_kernel(ctx)
        await LiterateInterpreter()._run_document(DIFF, ctx, kernel)
        entries = kernel.session.trace.entries
        assert [(e.tool, e.args) for e in entries] == [
            ("Path.read_text", ["hello.txt"]), ("Path.write_text", ["hello.txt"])]

    @pytest.mark.asyncio
    async def test_diff_against_an_edit_denied_path_refuses(self, ws, tmp_home):
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(
            '{"permissions": {"deny": ["Edit(hello.txt)"]}}')
        policy = load_policy(ws, home=tmp_home, managed=None)
        result = await _run(_ctx(ws, policy), DIFF + "\n```lackpy\nafter = 1\n```\n")
        assert not result.success
        assert "Edit(hello.txt)" in result.error
        assert [e.detail["cell_index"] for e in _refused(result)] == [0]
        assert "after" not in result.metadata["variables"]  # D4: the document ended
        assert (ws / "hello.txt").read_text() == "Hello World\n"

    @pytest.mark.asyncio
    async def test_diff_against_a_read_denied_path_refuses(self, ws, tmp_home):
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(
            '{"permissions": {"deny": ["Read(hello.txt)"]}}')
        policy = load_policy(ws, home=tmp_home, managed=None)
        result = await _run(_ctx(ws, policy), DIFF)
        assert not result.success
        assert "Read(hello.txt)" in result.error
        assert _refused(result)

    @pytest.mark.asyncio
    async def test_diff_outside_the_workspace_refuses(self, ws, policy):
        doc = DIFF.replace("@diff(hello.txt)", "@diff(/etc/hostname)")
        result = await _run(_ctx(ws, policy), doc)
        assert not result.success
        assert _refused(result)

    @pytest.mark.asyncio
    async def test_diff_without_a_policy_refuses(self, ws):
        result = await _run(ExecutionContext(base_dir=ws), DIFF)
        assert not result.success
        assert _refused(result)
        assert (ws / "hello.txt").read_text() == "Hello World\n"

    @pytest.mark.asyncio
    async def test_a_context_mismatch_is_a_runtime_error_not_a_refusal(self, ws, policy):
        (ws / "hello.txt").write_text("Something else\n")
        diff = ("```lackpy @diff(hello.txt)\n@@ -1,2 +1,2 @@\n Hello World\n-x\n+y\n```\n")
        result = await _run(_ctx(ws, policy), diff + "\n```lackpy\nafter = 1\n```\n")
        assert not result.success
        assert not _refused(result)
        assert result.metadata["ledger"].query(entry_type="error_reified")
        assert result.metadata["variables"]["after"] == 1  # the document went on
        assert (ws / "hello.txt").read_text() == "Something else\n"

    @pytest.mark.asyncio
    async def test_a_missing_file_is_a_runtime_error(self, ws, policy):
        (ws / "hello.txt").unlink()
        result = await _run(_ctx(ws, policy), DIFF)
        assert not result.success
        assert not _refused(result)
        assert "FileNotFoundError" in result.error

    @pytest.mark.asyncio
    async def test_no_patch_function_reaches_a_cell(self, ws, policy):
        from lackpy.interpreters.literate import _build_kernel
        ctx = _ctx(ws, policy)
        kernel = _build_kernel(ctx)
        result = await LiterateInterpreter()._run_document(
            DIFF + "\n```lackpy\nn = apply_diff\n```\n", ctx, kernel)
        assert kernel.session.bound_names() == set()
        assert result.metadata["ledger"].query(entry_type="hole_opened", name="apply_diff")


# --- the read-only profile is a policy deny ---------------------------------------------

class TestReadOnlyPolicy:
    @pytest.mark.asyncio
    async def test_write_refuses(self, ws, policy):
        result = await _run(_ctx(ws, read_only_policy(policy)),
                            "```lackpy @write(out.txt)\nx\n```\n\n```lackpy\nafter = 1\n```")
        assert not result.success
        assert "Edit(//**)" in result.error
        assert not (ws / "out.txt").exists()
        assert "after" not in result.metadata["variables"]

    @pytest.mark.asyncio
    async def test_diff_refuses_and_leaves_the_file(self, ws, policy):
        result = await _run(_ctx(ws, read_only_policy(policy)), DIFF)
        assert not result.success
        assert "Edit(//**)" in result.error
        assert (ws / "hello.txt").read_text() == "Hello World\n"

    @pytest.mark.asyncio
    async def test_read_still_works(self, ws, policy):
        result = await _run(_ctx(ws, read_only_policy(policy)), "```lackpy @read(hello.txt)\n```")
        assert result.success, result.error
        assert "Hello World" in result.output


# --- strict reactive mode reads the trace -------------------------------------------------

WRITES = {
    "write_text": "n = Path(name).write_text('hi')",
    "write_bytes": "n = Path(name).write_bytes(b'hi')",
    "mkdir": "n = Path(name).mkdir()",
}


class TestStrictModeReadsTheTrace:
    @pytest.mark.parametrize("member", sorted(WRITES))
    @pytest.mark.asyncio
    async def test_a_dependent_that_wrote_is_withheld(self, ws, policy, member):
        doc = ("```lackpy\nfrom pathlib import Path\nname = 'a.out'\n```\n\n"
               f"```lackpy\n{WRITES[member]}\n```\n\n"
               "```lackpy\nname = 'b.out'\n```\n")
        result = await _run(_ctx(ws, policy), doc)
        dirty = result.metadata["ledger"].query(entry_type=DIRTY)
        assert [e.detail["cell_index"] for e in dirty] == [1]
        assert dirty[0].detail["reexecuted"] is False
        assert dirty[0].detail["observed_effects"] == [f"Path.{member}"]
        assert "observed to perform a world effect" in dirty[0].detail["reason"]
        assert (ws / "a.out").exists()
        assert not (ws / "b.out").exists()  # never replayed

    @pytest.mark.asyncio
    async def test_permissive_mode_replays_the_write(self, ws, policy):
        doc = ("```lackpy\nfrom pathlib import Path\nname = 'a.out'\n```\n\n"
               f"```lackpy\n{WRITES['write_text']}\n```\n\n"
               "```lackpy\nname = 'b.out'\n```\n")
        result = await _run(_ctx(ws, policy, reactive_mode="permissive"), doc)
        dirty = result.metadata["ledger"].query(entry_type=DIRTY)
        assert dirty[0].detail["reexecuted"] is True
        assert dirty[0].detail["observed_effects"] == ["Path.write_text"]
        assert (ws / "b.out").read_text() == "hi"

    @pytest.mark.asyncio
    async def test_a_dependent_that_only_read_is_rerun(self, ws, policy):
        (ws / "b.txt").write_text("B")
        doc = ("```lackpy\nfrom pathlib import Path\nname = 'hello.txt'\n```\n\n"
               "```lackpy\nt = Path(name).read_text()\n```\n\n"
               "```lackpy\nname = 'b.txt'\n```\n")
        result = await _run(_ctx(ws, policy), doc)
        dirty = result.metadata["ledger"].query(entry_type=DIRTY)
        assert dirty[0].detail["reexecuted"] is True
        assert result.metadata["variables"]["t"] == "B"

    @pytest.mark.asyncio
    async def test_a_write_through_a_helper_is_seen(self, ws, policy):
        # No static classifier sees this write; the trace does.
        doc = ("```lackpy\nfrom pathlib import Path\n"
               "def save(v):\n    return Path('h.out').write_text(v)\n```\n\n"
               "```lackpy\nseed = 'one'\n```\n\n"
               "```lackpy\nres = save(seed)\n```\n\n"
               "```lackpy\nseed = 'two'\n```\n")
        result = await _run(_ctx(ws, policy), doc)
        dirty = {e.detail["cell_index"]: e
                 for e in result.metadata["ledger"].query(entry_type=DIRTY)}
        assert dirty[2].detail["reexecuted"] is False
        assert (ws / "h.out").read_text() == "one"
