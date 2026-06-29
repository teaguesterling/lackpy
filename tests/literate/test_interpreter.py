"""Integration tests for the LiterateInterpreter."""

import os
from pathlib import Path

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateInterpreter
from lackpy.lang.grader import Grade


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
