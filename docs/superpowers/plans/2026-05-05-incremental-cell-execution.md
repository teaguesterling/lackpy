# Incremental Cell Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace compile-all-exec-once with a streaming, cell-by-cell execution engine that supports pluggable recovery and produces notebook artifacts.

**Architecture:** Cells flow through a StreamingCellParser into a StreamingDriver that feeds them one at a time to a KernelInterface. The kernel does static analysis (compile + name resolution) before exec. On failure, a pluggable RecoveryHandler orchestrates fix/inspect/skip/abort actions. An ExecutionPlugin protocol provides hooks for coaching systems. The execution log serializes to .ipynb or clean markdown.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, existing lackpy parser/compiler infrastructure.

---

## File Structure

```
src/lackpy/interpreters/literate/kernel/
├── __init__.py              # Public exports
├── interface.py             # KernelInterface protocol + CellResult dataclass
├── lightweight.py           # LightweightKernel implementation (exec-into-dict)
├── static_analysis.py       # Name resolution + compile check logic
├── streaming_parser.py      # StreamingCellParser (fence detection on partial input)
├── driver.py                # StreamingDriver orchestrator
├── recovery.py              # RecoveryHandler protocol + NoRecoveryHandler + InferenceRecoveryHandler
├── plugins.py               # ExecutionPlugin protocol + PluginAdvice dataclass
└── formats.py               # to_notebook, from_notebook, render_markdown

tests/literate/kernel/
├── __init__.py
├── test_interface.py        # CellResult construction tests
├── test_lightweight.py      # LightweightKernel unit tests
├── test_static_analysis.py  # Name resolution + compile check tests
├── test_streaming_parser.py # StreamingCellParser tests
├── test_driver.py           # StreamingDriver integration tests
├── test_recovery.py         # Recovery handler protocol tests
├── test_inference_recovery.py # InferenceRecoveryHandler tests
├── test_plugins.py          # Plugin hook tests
├── test_formats.py          # Format conversion round-trip tests
└── test_e2e.py              # End-to-end integration tests

Modified:
├── src/lackpy/interpreters/literate/__init__.py  # Wrapper uses new kernel
```

---

### Task 1: KernelInterface Protocol + CellResult

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/__init__.py`
- Create: `src/lackpy/interpreters/literate/kernel/interface.py`
- Create: `tests/literate/kernel/__init__.py`
- Create: `tests/literate/kernel/test_interface.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for kernel interface types."""

from lackpy.interpreters.literate.kernel.interface import CellResult


class TestCellResult:
    def test_successful_result(self):
        result = CellResult(
            success=True,
            output="hello\n",
            error=None,
            error_phase=None,
            namespace_delta={"x": 42},
            cell_index=0,
        )
        assert result.success
        assert result.output == "hello\n"
        assert result.namespace_delta == {"x": 42}

    def test_static_failure(self):
        result = CellResult(
            success=False,
            output=None,
            error="name 'foo' is not defined",
            error_phase="static",
            namespace_delta={},
            cell_index=2,
        )
        assert not result.success
        assert result.error_phase == "static"

    def test_runtime_failure(self):
        result = CellResult(
            success=False,
            output=None,
            error="ZeroDivisionError: division by zero",
            error_phase="runtime",
            namespace_delta={},
            cell_index=1,
        )
        assert result.error_phase == "runtime"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_interface.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lackpy.interpreters.literate.kernel'`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/__init__.py`:
```python
"""Incremental cell execution kernel."""

from .interface import CellResult, KernelInterface

__all__ = ["CellResult", "KernelInterface"]
```

`src/lackpy/interpreters/literate/kernel/interface.py`:
```python
"""Kernel interface protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..parser import Cell


@dataclass
class CellResult:
    success: bool
    output: str | None
    error: str | None
    error_phase: str | None  # "static" | "runtime"
    namespace_delta: dict[str, Any]
    cell_index: int


class KernelInterface(Protocol):
    def execute_cell(self, cell: Cell, cell_index: int) -> CellResult: ...
    def inspect(self, expr: str) -> str: ...
    def get_scope(self) -> dict[str, str]: ...
    def restart(self) -> None: ...
    def get_namespace(self) -> dict[str, Any]: ...
```

`tests/literate/kernel/__init__.py`:
```python
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_interface.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/__init__.py \
        src/lackpy/interpreters/literate/kernel/interface.py \
        tests/literate/kernel/__init__.py \
        tests/literate/kernel/test_interface.py
git commit -m "feat(kernel): add KernelInterface protocol and CellResult dataclass"
```

---

### Task 2: Static Analysis Module

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/static_analysis.py`
- Create: `tests/literate/kernel/test_static_analysis.py`

- [ ] **Step 1: Write the failing tests**

```python
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
        with pytest.raises(StaticAnalysisError, match="syntax"):
            check_cell("print(f'{x:.1f if x > 0 else y:.1f}')", known_names={"x", "y"})

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_static_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/static_analysis.py`:
```python
"""Static analysis for cell pre-flight checks.

Performs two checks before executing a cell:
1. compile() - catches syntax errors, malformed f-strings
2. AST name resolution - catches undefined references
"""

from __future__ import annotations

import ast
import builtins

_BUILTINS = set(dir(builtins))


class StaticAnalysisError(Exception):
    pass


def check_cell(code: str, known_names: set[str]) -> None:
    try:
        tree = compile(code, "<cell>", "exec", ast.PyCF_ONLY_AST)
    except SyntaxError as e:
        raise StaticAnalysisError(f"syntax error: {e.msg} (line {e.lineno})") from e

    defined = _collect_definitions(tree)
    referenced = _collect_references(tree)

    available = known_names | _BUILTINS | defined
    undefined = referenced - available

    if undefined:
        names = ", ".join(f"'{n}'" for n in sorted(undefined))
        raise StaticAnalysisError(f"undefined names: {names}")


def _collect_definitions(tree: ast.Module) -> set[str]:
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                defined.update(_names_from_target(target))
        elif isinstance(node, ast.AugAssign):
            pass
        elif isinstance(node, ast.AnnAssign) and node.target:
            defined.update(_names_from_target(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.For):
            defined.update(_names_from_target(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    defined.update(_names_from_target(item.optional_vars))
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                defined.update(_names_from_target(gen.target))
    return defined


def _collect_references(tree: ast.Module) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            refs.add(node.id)
    return refs


def _names_from_target(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    elif isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in node.elts:
            names.update(_names_from_target(elt))
        return names
    elif isinstance(node, ast.Starred):
        return _names_from_target(node.value)
    return set()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_static_analysis.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/static_analysis.py \
        tests/literate/kernel/test_static_analysis.py
git commit -m "feat(kernel): static analysis with compile check and name resolution"
```

---

### Task 3: LightweightKernel

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/lightweight.py`
- Create: `tests/literate/kernel/test_lightweight.py`
- Modify: `src/lackpy/interpreters/literate/kernel/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for LightweightKernel - the exec-into-dict kernel backend."""

import pytest

from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.interpreters.literate.parser import Cell


@pytest.fixture
def kernel():
    return LightweightKernel()


class TestExecuteCell:
    def test_code_cell_executes(self, kernel):
        cell = Cell(cell_type="code", content="x = 42")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert result.namespace_delta == {"x": 42}

    def test_prose_cell_produces_output(self, kernel):
        cell = Cell(cell_type="prose", content="Hello world")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert "Hello world" in result.output

    def test_hidden_cell_no_output(self, kernel):
        cell = Cell(cell_type="hidden", content="secret = 99")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert result.output == ""
        assert result.namespace_delta == {"secret": 99}

    def test_state_persists_across_cells(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 10"), cell_index=0)
        result = kernel.execute_cell(
            Cell(cell_type="prose", content="Value: {x}"), cell_index=1
        )
        assert result.success
        assert "10" in result.output

    def test_syntax_error_caught_statically(self, kernel):
        cell = Cell(cell_type="code", content="x = ")
        result = kernel.execute_cell(cell, cell_index=0)
        assert not result.success
        assert result.error_phase == "static"

    def test_undefined_name_caught_statically(self, kernel):
        cell = Cell(cell_type="code", content="print(undefined_var)")
        result = kernel.execute_cell(cell, cell_index=0)
        assert not result.success
        assert result.error_phase == "static"

    def test_runtime_error_caught(self, kernel):
        cell = Cell(cell_type="code", content="x = 1 / 0")
        result = kernel.execute_cell(cell, cell_index=0)
        assert not result.success
        assert result.error_phase == "runtime"
        assert "ZeroDivisionError" in result.error

    def test_write_cell_calls_write_file(self, kernel):
        written = {}
        kernel._namespace["write_file"] = lambda p, c: written.update({p: c})
        cell = Cell(
            cell_type="write",
            content="print('hi')",
            annotation_args={"path": "out.py"},
        )
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert written["out.py"] == "print('hi')"

    def test_continue_cell_sets_flag(self, kernel):
        cell = Cell(cell_type="continue", content="")
        result = kernel.execute_cell(cell, cell_index=0)
        assert result.success
        assert result.namespace_delta.get("__continue_requested__") is True


class TestInspect:
    def test_inspect_variable(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = [1, 2, 3]"), cell_index=0)
        assert kernel.inspect("x") == "[1, 2, 3]"

    def test_inspect_expression(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42"), cell_index=0)
        assert kernel.inspect("x * 2") == "84"

    def test_inspect_error(self, kernel):
        result = kernel.inspect("undefined_var")
        assert "NameError" in result


class TestScope:
    def test_get_scope_empty(self, kernel):
        scope = kernel.get_scope()
        assert scope == {}

    def test_get_scope_after_execution(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42\ny = 'hi'"), cell_index=0)
        scope = kernel.get_scope()
        assert "x" in scope
        assert "int" in scope["x"]
        assert "y" in scope
        assert "str" in scope["y"]


class TestRestart:
    def test_restart_clears_state(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42"), cell_index=0)
        kernel.restart()
        scope = kernel.get_scope()
        assert "x" not in scope
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_lightweight.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/lightweight.py`:
```python
"""Lightweight kernel: exec-into-dict with static analysis."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any

from ..compiler import _COMPILERS
from ..parser import Cell
from .interface import CellResult
from .static_analysis import StaticAnalysisError, check_cell


class LightweightKernel:
    def __init__(self, namespace: dict[str, Any] | None = None) -> None:
        self._namespace: dict[str, Any] = namespace or {}
        self._initial_keys: set[str] = set(self._namespace.keys())
        if "__builtins__" not in self._namespace:
            import builtins
            self._namespace["__builtins__"] = builtins

    def execute_cell(self, cell: Cell, cell_index: int) -> CellResult:
        compiler = _COMPILERS.get(cell.cell_type)
        if compiler is None:
            return CellResult(
                success=False,
                output=None,
                error=f"Unknown cell type: {cell.cell_type}",
                error_phase="static",
                namespace_delta={},
                cell_index=cell_index,
            )

        compiled_source = compiler(cell)
        if not compiled_source.strip():
            return CellResult(
                success=True,
                output="",
                error=None,
                error_phase=None,
                namespace_delta={},
                cell_index=cell_index,
            )

        known_names = set(self._namespace.keys()) - {"__builtins__"}
        try:
            check_cell(compiled_source, known_names)
        except StaticAnalysisError as e:
            return CellResult(
                success=False,
                output=None,
                error=str(e),
                error_phase="static",
                namespace_delta={},
                cell_index=cell_index,
            )

        before_snapshot = {
            k: v for k, v in self._namespace.items()
            if k != "__builtins__"
        }

        stdout_capture = io.StringIO()
        try:
            code_obj = compile(compiled_source, "<cell>", "exec")
            with redirect_stdout(stdout_capture):
                _do_exec(code_obj, self._namespace)
        except Exception as e:
            return CellResult(
                success=False,
                output=stdout_capture.getvalue() or None,
                error=f"{type(e).__name__}: {e}",
                error_phase="runtime",
                namespace_delta={},
                cell_index=cell_index,
            )

        delta: dict[str, Any] = {}
        for k, v in self._namespace.items():
            if k == "__builtins__" or k.startswith("_"):
                continue
            if k not in before_snapshot or before_snapshot[k] is not v:
                delta[k] = v

        if cell.cell_type == "continue":
            delta["__continue_requested__"] = True

        return CellResult(
            success=True,
            output=stdout_capture.getvalue(),
            error=None,
            error_phase=None,
            namespace_delta=delta,
            cell_index=cell_index,
        )

    def inspect(self, expr: str) -> str:
        try:
            result = eval(expr, self._namespace)  # noqa: S307
            return repr(result)
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def get_scope(self) -> dict[str, str]:
        scope: dict[str, str] = {}
        for k, v in self._namespace.items():
            if k.startswith("_") or k == "__builtins__":
                continue
            if callable(v) and k in self._initial_keys:
                continue
            scope[k] = f"{type(v).__name__}: {repr(v)[:80]}"
        return scope

    def restart(self) -> None:
        tools = {k: v for k, v in self._namespace.items() if k in self._initial_keys}
        self._namespace.clear()
        self._namespace.update(tools)
        import builtins
        self._namespace["__builtins__"] = builtins

    def get_namespace(self) -> dict[str, Any]:
        return {
            k: v for k, v in self._namespace.items()
            if k != "__builtins__" and not k.startswith("_")
        }


def _do_exec(code: object, ns: dict) -> None:
    exec(code, ns)  # noqa: S102
```

Update `src/lackpy/interpreters/literate/kernel/__init__.py`:
```python
"""Incremental cell execution kernel."""

from .interface import CellResult, KernelInterface
from .lightweight import LightweightKernel

__all__ = ["CellResult", "KernelInterface", "LightweightKernel"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_lightweight.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/lightweight.py \
        src/lackpy/interpreters/literate/kernel/__init__.py \
        tests/literate/kernel/test_lightweight.py
git commit -m "feat(kernel): LightweightKernel with static analysis and exec-into-dict"
```

---

### Task 4: StreamingCellParser

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/streaming_parser.py`
- Create: `tests/literate/kernel/test_streaming_parser.py`
- Modify: `src/lackpy/interpreters/literate/kernel/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for StreamingCellParser - incremental fence detection."""

import pytest

from lackpy.interpreters.literate.kernel.streaming_parser import StreamingCellParser


@pytest.fixture
def parser():
    return StreamingCellParser()


class TestBasicParsing:
    def test_prose_only_on_flush(self, parser):
        cells = parser.feed("Hello world")
        assert cells == []
        cells = parser.flush()
        assert len(cells) == 1
        assert cells[0].cell_type == "prose"
        assert cells[0].content == "Hello world"

    def test_code_fence_yields_prose_then_code(self, parser):
        text = "Some prose\n\n```lackpy\nx = 42\n```\n"
        cells = parser.feed(text)
        assert len(cells) == 2
        assert cells[0].cell_type == "prose"
        assert "Some prose" in cells[0].content
        assert cells[1].cell_type == "code"
        assert cells[1].content == "x = 42"

    def test_hidden_annotation(self, parser):
        text = "```lackpy @hidden\nsecret = 1\n```\n"
        cells = parser.feed(text)
        assert len(cells) == 1
        assert cells[0].cell_type == "hidden"
        assert cells[0].content == "secret = 1"

    def test_write_annotation_with_path(self, parser):
        text = "```lackpy @write(out.py)\nprint('hi')\n```\n"
        cells = parser.feed(text)
        assert len(cells) == 1
        assert cells[0].cell_type == "write"
        assert cells[0].annotation_args == {"path": "out.py"}
        assert cells[0].content == "print('hi')"


class TestStreamingBehavior:
    def test_chunked_input(self, parser):
        cells1 = parser.feed("Hello\n\n```lack")
        assert cells1 == []
        cells2 = parser.feed("py\nx = 1\n``")
        assert cells2 == []
        cells3 = parser.feed("`\n")
        assert len(cells3) == 2
        assert cells3[0].cell_type == "prose"
        assert cells3[1].cell_type == "code"
        assert cells3[1].content == "x = 1"

    def test_multiple_fences(self, parser):
        text = (
            "Intro\n\n"
            "```lackpy @hidden\na = 1\n```\n\n"
            "Middle\n\n"
            "```lackpy\nb = a + 1\n```\n"
        )
        cells = parser.feed(text)
        assert len(cells) == 4
        assert cells[0].cell_type == "prose"
        assert cells[1].cell_type == "hidden"
        assert cells[2].cell_type == "prose"
        assert cells[3].cell_type == "code"

    def test_non_lackpy_fence_is_prose(self, parser):
        text = "```python\nx = 1\n```\n"
        cells = parser.feed(text)
        flush = parser.flush()
        all_cells = cells + flush
        assert len(all_cells) == 1
        assert all_cells[0].cell_type == "prose"

    def test_unclosed_fence_on_flush(self, parser):
        parser.feed("```lackpy\nx = 1\n")
        cells = parser.flush()
        assert len(cells) == 1
        assert cells[0].cell_type == "code"
        assert cells[0].content == "x = 1"


class TestFrontmatter:
    def test_frontmatter_consumed(self, parser):
        text = "---\necho: true\noutput: auto\n---\n\nHello"
        parser.feed(text)
        cells = parser.flush()
        assert len(cells) == 1
        assert cells[0].cell_type == "prose"
        assert "Hello" in cells[0].content
        assert parser.frontmatter.echo == "true"

    def test_no_frontmatter(self, parser):
        text = "Just prose"
        parser.feed(text)
        parser.flush()
        assert parser.frontmatter.echo == "true"


class TestReset:
    def test_reset_clears_state(self, parser):
        parser.feed("```lackpy\nx = 1\n")
        parser.reset()
        cells = parser.flush()
        assert cells == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_streaming_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/streaming_parser.py`:
```python
"""Streaming cell parser - detects fence boundaries in partial input."""

from __future__ import annotations

import re

from ..parser import Cell, Frontmatter, _parse_info_string, _PATH_ANNOTATIONS, _extract_path_from_body

_FENCE_OPEN = re.compile(r"^```(\S.*)?$", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"^```\s*$", re.MULTILINE)


class StreamingCellParser:
    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_fence: bool = False
        self._fence_info: str = ""
        self._fence_content_start: int = 0
        self._frontmatter: Frontmatter = Frontmatter()
        self._frontmatter_parsed: bool = False

    @property
    def frontmatter(self) -> Frontmatter:
        return self._frontmatter

    def feed(self, chunk: str) -> list[Cell]:
        self._buffer += chunk
        if not self._frontmatter_parsed:
            self._try_parse_frontmatter()
        return self._extract_cells()

    def flush(self) -> list[Cell]:
        cells: list[Cell] = []
        if self._in_fence:
            content = self._buffer[self._fence_content_start:].rstrip("\n")
            cell = self._make_fence_cell(self._fence_info, content)
            cells.append(cell)
            self._in_fence = False
        elif self._buffer.strip():
            cells.append(Cell(cell_type="prose", content=self._buffer.strip()))
        self._buffer = ""
        return cells

    def reset(self) -> None:
        self._buffer = ""
        self._in_fence = False
        self._fence_info = ""
        self._fence_content_start = 0
        self._frontmatter = Frontmatter()
        self._frontmatter_parsed = False

    def _try_parse_frontmatter(self) -> None:
        if not self._buffer.startswith("---"):
            self._frontmatter_parsed = True
            return
        lines = self._buffer.split("\n")
        if len(lines) < 2:
            return
        for i in range(1, len(lines)):
            if lines[i].rstrip() == "---":
                self._frontmatter = self._parse_fm_lines(lines[1:i])
                self._buffer = "\n".join(lines[i + 1:])
                self._frontmatter_parsed = True
                return

    def _parse_fm_lines(self, lines: list[str]) -> Frontmatter:
        fm = Frontmatter()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key == "echo":
                    fm.echo = value
                elif key == "output":
                    fm.output = value
                elif key == "interpreter":
                    fm.interpreter = value
        return fm

    def _extract_cells(self) -> list[Cell]:
        cells: list[Cell] = []
        while True:
            if self._in_fence:
                close_match = _FENCE_CLOSE.search(self._buffer, self._fence_content_start)
                if close_match is None:
                    break
                content = self._buffer[self._fence_content_start:close_match.start()].rstrip("\n")
                cell = self._make_fence_cell(self._fence_info, content)
                cells.append(cell)
                self._buffer = self._buffer[close_match.end():]
                if self._buffer.startswith("\n"):
                    self._buffer = self._buffer[1:]
                self._in_fence = False
                self._fence_content_start = 0
            else:
                open_match = _FENCE_OPEN.search(self._buffer)
                if open_match is None:
                    break
                prose_before = self._buffer[:open_match.start()].rstrip("\n")
                if prose_before.strip():
                    cells.append(Cell(cell_type="prose", content=prose_before.strip()))
                info = open_match.group(1) or ""
                if not info.startswith("lackpy"):
                    close_after = _FENCE_CLOSE.search(self._buffer, open_match.end() + 1)
                    if close_after is None:
                        self._buffer = self._buffer[open_match.start():]
                        break
                    fence_text = self._buffer[open_match.start():close_after.end()]
                    cells.append(Cell(cell_type="prose", content=fence_text))
                    self._buffer = self._buffer[close_after.end():]
                    if self._buffer.startswith("\n"):
                        self._buffer = self._buffer[1:]
                    continue
                self._fence_info = info
                self._in_fence = True
                self._buffer = self._buffer[open_match.end():]
                if self._buffer.startswith("\n"):
                    self._buffer = self._buffer[1:]
                self._fence_content_start = 0
        return cells

    def _make_fence_cell(self, info: str, content: str) -> Cell:
        cell_type, annotation_args, options, _errors = _parse_info_string(info)
        if cell_type in _PATH_ANNOTATIONS and "path" not in annotation_args:
            path, content = _extract_path_from_body(content)
            if path:
                annotation_args["path"] = path
        return Cell(
            cell_type=cell_type,
            content=content,
            annotation_args=annotation_args,
            options=options,
        )
```

Update `src/lackpy/interpreters/literate/kernel/__init__.py`:
```python
"""Incremental cell execution kernel."""

from .interface import CellResult, KernelInterface
from .lightweight import LightweightKernel
from .streaming_parser import StreamingCellParser

__all__ = ["CellResult", "KernelInterface", "LightweightKernel", "StreamingCellParser"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_streaming_parser.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/streaming_parser.py \
        src/lackpy/interpreters/literate/kernel/__init__.py \
        tests/literate/kernel/test_streaming_parser.py
git commit -m "feat(kernel): StreamingCellParser with incremental fence detection"
```

---

### Task 5: ExecutionPlugin Protocol

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/plugins.py`
- Create: `tests/literate/kernel/test_plugins.py`
- Modify: `src/lackpy/interpreters/literate/kernel/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for ExecutionPlugin protocol and PluginAdvice."""

from lackpy.interpreters.literate.kernel.plugins import (
    ExecutionPlugin,
    PluginAdvice,
    merge_advice,
)
from lackpy.interpreters.literate.kernel.interface import CellResult
from lackpy.interpreters.literate.parser import Cell


class FakePlugin:
    def __init__(self, hints: list[str] | None = None):
        self.started: list[int] = []
        self.succeeded: list[int] = []
        self.errors: list[str] = []
        self._hints = hints or []

    def on_cell_start(self, cell: Cell, index: int) -> None:
        self.started.append(index)

    def on_cell_success(self, cell: Cell, result: CellResult) -> None:
        self.succeeded.append(result.cell_index)

    def on_cell_error(self, cell: Cell, error: str, scope: dict) -> PluginAdvice:
        self.errors.append(error)
        return PluginAdvice(hints=self._hints, doc_context=[], suggestion=None)

    def on_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None:
        pass


class TestPluginAdvice:
    def test_empty_advice(self):
        advice = PluginAdvice(hints=[], doc_context=[], suggestion=None)
        assert advice.hints == []
        assert advice.suggestion is None

    def test_merge_multiple_advices(self):
        a1 = PluginAdvice(hints=["try X"], doc_context=["doc1"], suggestion="do X")
        a2 = PluginAdvice(hints=["try Y"], doc_context=["doc2"], suggestion="do Y")
        merged = merge_advice([a1, a2])
        assert merged.hints == ["try X", "try Y"]
        assert merged.doc_context == ["doc1", "doc2"]
        assert merged.suggestion == "do X"

    def test_merge_empty_list(self):
        merged = merge_advice([])
        assert merged.hints == []
        assert merged.suggestion is None


class TestPluginProtocol:
    def test_plugin_receives_start(self):
        plugin = FakePlugin()
        cell = Cell(cell_type="code", content="x = 1")
        plugin.on_cell_start(cell, 0)
        assert plugin.started == [0]

    def test_plugin_receives_error(self):
        plugin = FakePlugin(hints=["check imports"])
        cell = Cell(cell_type="code", content="x = foo")
        advice = plugin.on_cell_error(cell, "undefined 'foo'", {})
        assert advice.hints == ["check imports"]
        assert plugin.errors == ["undefined 'foo'"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_plugins.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/plugins.py`:
```python
"""Execution plugin protocol and advice types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..parser import Cell
from .interface import CellResult


@dataclass
class PluginAdvice:
    hints: list[str] = field(default_factory=list)
    doc_context: list[str] = field(default_factory=list)
    suggestion: str | None = None


class ExecutionPlugin(Protocol):
    def on_cell_start(self, cell: Cell, index: int) -> None: ...
    def on_cell_success(self, cell: Cell, result: CellResult) -> None: ...
    def on_cell_error(self, cell: Cell, error: str, scope: dict) -> PluginAdvice: ...
    def on_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None: ...


def merge_advice(advices: list[PluginAdvice]) -> PluginAdvice:
    if not advices:
        return PluginAdvice()
    hints: list[str] = []
    doc_context: list[str] = []
    suggestion: str | None = None
    for a in advices:
        hints.extend(a.hints)
        doc_context.extend(a.doc_context)
        if suggestion is None and a.suggestion is not None:
            suggestion = a.suggestion
    return PluginAdvice(hints=hints, doc_context=doc_context, suggestion=suggestion)
```

Update `src/lackpy/interpreters/literate/kernel/__init__.py`:
```python
"""Incremental cell execution kernel."""

from .interface import CellResult, KernelInterface
from .lightweight import LightweightKernel
from .plugins import ExecutionPlugin, PluginAdvice, merge_advice
from .streaming_parser import StreamingCellParser

__all__ = [
    "CellResult",
    "ExecutionPlugin",
    "KernelInterface",
    "LightweightKernel",
    "PluginAdvice",
    "StreamingCellParser",
    "merge_advice",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_plugins.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/plugins.py \
        src/lackpy/interpreters/literate/kernel/__init__.py \
        tests/literate/kernel/test_plugins.py
git commit -m "feat(kernel): ExecutionPlugin protocol and PluginAdvice"
```

---

### Task 6: RecoveryHandler Protocol + NoRecoveryHandler

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/recovery.py`
- Create: `tests/literate/kernel/test_recovery.py`
- Modify: `src/lackpy/interpreters/literate/kernel/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for recovery handler protocol."""

import pytest

from lackpy.interpreters.literate.kernel.recovery import (
    NoRecoveryHandler,
    RecoveryAction,
    RecoveryContext,
)
from lackpy.interpreters.literate.kernel.plugins import PluginAdvice
from lackpy.interpreters.literate.parser import Cell


def _make_context(**kwargs) -> RecoveryContext:
    defaults = {
        "failed_cell": Cell(cell_type="code", content="x = foo"),
        "error": "undefined names: 'foo'",
        "error_phase": "static",
        "scope": {"y": "int: 42"},
        "cell_index": 3,
        "prior_output": "Some output\n",
        "attempt": 0,
        "plugin_advice": None,
    }
    defaults.update(kwargs)
    return RecoveryContext(**defaults)


class TestRecoveryAction:
    def test_fix_action(self):
        cells = [Cell(cell_type="hidden", content="foo = 1")]
        action = RecoveryAction(kind="fix", cells=cells)
        assert action.kind == "fix"
        assert len(action.cells) == 1

    def test_inspect_action(self):
        action = RecoveryAction(kind="inspect", expr="type(x)")
        assert action.kind == "inspect"
        assert action.expr == "type(x)"

    def test_abort_action(self):
        action = RecoveryAction(kind="abort")
        assert action.kind == "abort"

    def test_skip_action(self):
        action = RecoveryAction(kind="skip")
        assert action.kind == "skip"


class TestNoRecoveryHandler:
    def test_always_aborts(self):
        handler = NoRecoveryHandler()
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "abort"

    def test_inspect_result_aborts(self):
        handler = NoRecoveryHandler()
        ctx = _make_context()
        action = handler.on_inspect_result(ctx, "42")
        assert action.kind == "abort"

    def test_max_attempts_is_zero(self):
        handler = NoRecoveryHandler()
        assert handler.max_attempts == 0


class TestRecoveryContext:
    def test_context_fields(self):
        ctx = _make_context(attempt=2)
        assert ctx.attempt == 2
        assert ctx.error == "undefined names: 'foo'"
        assert ctx.cell_index == 3

    def test_context_with_plugin_advice(self):
        advice = PluginAdvice(hints=["try importing foo"])
        ctx = _make_context(plugin_advice=advice)
        assert ctx.plugin_advice.hints == ["try importing foo"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_recovery.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/recovery.py`:
```python
"""Recovery handler protocol and built-in handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..parser import Cell
from .plugins import PluginAdvice


@dataclass
class RecoveryAction:
    kind: str  # "fix" | "inspect" | "skip" | "abort"
    cells: list[Cell] | None = None
    expr: str | None = None
    target_index: int | None = None


@dataclass
class RecoveryContext:
    failed_cell: Cell
    error: str
    error_phase: str
    scope: dict[str, str]
    cell_index: int
    prior_output: str
    attempt: int
    plugin_advice: PluginAdvice | None = None


class RecoveryHandler(Protocol):
    max_attempts: int

    def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction: ...
    def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction: ...


class NoRecoveryHandler:
    max_attempts: int = 0

    def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
        return RecoveryAction(kind="abort")

    def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
        return RecoveryAction(kind="abort")
```

Update `src/lackpy/interpreters/literate/kernel/__init__.py`:
```python
"""Incremental cell execution kernel."""

from .interface import CellResult, KernelInterface
from .lightweight import LightweightKernel
from .plugins import ExecutionPlugin, PluginAdvice, merge_advice
from .recovery import NoRecoveryHandler, RecoveryAction, RecoveryContext, RecoveryHandler
from .streaming_parser import StreamingCellParser

__all__ = [
    "CellResult",
    "ExecutionPlugin",
    "KernelInterface",
    "LightweightKernel",
    "NoRecoveryHandler",
    "PluginAdvice",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryHandler",
    "StreamingCellParser",
    "merge_advice",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_recovery.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/recovery.py \
        src/lackpy/interpreters/literate/kernel/__init__.py \
        tests/literate/kernel/test_recovery.py
git commit -m "feat(kernel): RecoveryHandler protocol and NoRecoveryHandler"
```

---

### Task 7: StreamingDriver

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/driver.py`
- Create: `tests/literate/kernel/test_driver.py`
- Modify: `src/lackpy/interpreters/literate/kernel/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for StreamingDriver - the orchestrator."""

import pytest

from lackpy.interpreters.literate.kernel.driver import CellExecutionEvent, StreamingDriver
from lackpy.interpreters.literate.kernel.lightweight import LightweightKernel
from lackpy.interpreters.literate.kernel.recovery import (
    NoRecoveryHandler,
    RecoveryAction,
    RecoveryContext,
)
from lackpy.interpreters.literate.kernel.plugins import PluginAdvice
from lackpy.interpreters.literate.parser import Cell


@pytest.fixture
def kernel():
    return LightweightKernel()


@pytest.fixture
def driver(kernel):
    return StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler())


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_prose_cell(self, driver):
        events = await driver.feed("Hello world\n")
        flush_events = await driver.flush()
        all_events = events + flush_events
        assert len(all_events) == 1
        assert all_events[0].status == "executed"
        assert "Hello world" in driver.rendered_output

    @pytest.mark.asyncio
    async def test_hidden_then_prose(self, driver):
        text = "```lackpy @hidden\nx = 42\n```\n\nValue: {x}\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        assert len(all_events) == 2
        assert all_events[0].cell.cell_type == "hidden"
        assert all_events[1].cell.cell_type == "prose"
        assert "42" in driver.rendered_output

    @pytest.mark.asyncio
    async def test_execution_log_tracks_all(self, driver):
        text = "```lackpy @hidden\na = 1\n```\n\n```lackpy\nb = 2\n```\n"
        await driver.feed(text)
        await driver.flush()
        assert len(driver.execution_log) == 2


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_static_error_aborts_with_no_recovery(self, driver):
        text = "```lackpy\nprint(undefined)\n```\n\nNever reached\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        failed = [e for e in all_events if e.status not in ("executed", "pending")]
        assert len(failed) >= 1
        assert failed[0].result.error_phase == "static"

    @pytest.mark.asyncio
    async def test_runtime_error_aborts(self, driver):
        text = "```lackpy @hidden\nx = 1\n```\n\n```lackpy\ny = 1/0\n```\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        assert all_events[0].status == "executed"
        assert all_events[1].status != "executed"


class TestRecoveryIntegration:
    @pytest.mark.asyncio
    async def test_fix_action_replaces_cell(self, kernel):
        fix_cell = Cell(cell_type="hidden", content="foo = 'fixed'")

        class FixHandler:
            max_attempts = 2

            def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
                return RecoveryAction(kind="fix", cells=[fix_cell])

            def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
                return RecoveryAction(kind="abort")

        driver = StreamingDriver(kernel=kernel, recovery=FixHandler())
        text = "```lackpy\nprint(foo)\n```\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        recovered = [e for e in all_events if e.status == "recovered"]
        assert len(recovered) == 1

    @pytest.mark.asyncio
    async def test_skip_action_continues(self, kernel):
        class SkipHandler:
            max_attempts = 1

            def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
                return RecoveryAction(kind="skip")

            def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
                return RecoveryAction(kind="abort")

        driver = StreamingDriver(kernel=kernel, recovery=SkipHandler())
        text = "```lackpy\nprint(bad)\n```\n\nAfter skip\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        skipped = [e for e in all_events if e.status == "skipped"]
        assert len(skipped) == 1
        executed = [e for e in all_events if e.status == "executed"]
        assert len(executed) >= 1

    @pytest.mark.asyncio
    async def test_inspect_action_evaluates_expr(self, kernel):
        kernel.execute_cell(Cell(cell_type="hidden", content="x = 42"), cell_index=0)

        class InspectThenFixHandler:
            max_attempts = 2

            def __init__(self):
                self.inspected = None

            def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
                return RecoveryAction(kind="inspect", expr="type(x).__name__")

            def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
                self.inspected = result
                return RecoveryAction(
                    kind="fix",
                    cells=[Cell(cell_type="code", content="y = str(x)")],
                )

        handler = InspectThenFixHandler()
        driver = StreamingDriver(kernel=kernel, recovery=handler)
        text = "```lackpy\ny = str(undefined)\n```\n"
        await driver.feed(text)
        await driver.flush()
        assert handler.inspected == "'int'"


class TestContinueSemantics:
    @pytest.mark.asyncio
    async def test_continue_pauses_execution(self, driver):
        text = "```lackpy @hidden\nx = 1\n```\n\n```lackpy @continue\n```\n\nNot reached yet\n"
        events = await driver.feed(text)
        flush_events = await driver.flush()
        all_events = events + flush_events
        continue_events = [e for e in all_events if e.status == "continue_requested"]
        assert len(continue_events) == 1
        assert "Not reached" not in driver.rendered_output

    @pytest.mark.asyncio
    async def test_generation_increments_on_continue(self, driver):
        text1 = "```lackpy @hidden\nx = 1\n```\n\n```lackpy @continue\n```\n"
        await driver.feed(text1)
        await driver.flush()
        assert driver.generation == 0

        driver.resume()
        text2 = "```lackpy\ny = x + 1\n```\n"
        await driver.feed(text2)
        await driver.flush()
        assert driver.generation == 1


class TestPluginNotifications:
    @pytest.mark.asyncio
    async def test_plugins_notified_on_success(self, kernel):
        class TrackingPlugin:
            def __init__(self):
                self.started = []
                self.succeeded = []

            def on_cell_start(self, cell, index):
                self.started.append(index)

            def on_cell_success(self, cell, result):
                self.succeeded.append(result.cell_index)

            def on_cell_error(self, cell, error, scope):
                return PluginAdvice()

            def on_recovery_result(self, cell, success, attempt):
                pass

        plugin = TrackingPlugin()
        driver = StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler(), plugins=[plugin])
        text = "```lackpy @hidden\nx = 1\n```\n"
        await driver.feed(text)
        await driver.flush()
        assert plugin.started == [0]
        assert plugin.succeeded == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_driver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/driver.py`:
```python
"""Streaming driver - orchestrates parse, kernel, recovery, plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..parser import Cell
from .interface import CellResult
from .lightweight import LightweightKernel
from .plugins import PluginAdvice, merge_advice
from .recovery import NoRecoveryHandler, RecoveryAction, RecoveryContext, RecoveryHandler
from .streaming_parser import StreamingCellParser


@dataclass
class CellExecutionEvent:
    cell: Cell
    cell_index: int
    result: CellResult | None
    status: str
    recovery_attempts: int = 0
    generation: int = 0


class StreamingDriver:
    def __init__(
        self,
        kernel: LightweightKernel,
        recovery: RecoveryHandler | None = None,
        plugins: list[Any] | None = None,
    ) -> None:
        self._kernel = kernel
        self._recovery = recovery or NoRecoveryHandler()
        self._plugins: list[Any] = plugins or []
        self._parser = StreamingCellParser()
        self._log: list[CellExecutionEvent] = []
        self._output_parts: list[str] = []
        self._cell_counter: int = 0
        self._generation: int = 0
        self._interrupted: bool = False
        self._continue_requested: bool = False

    @property
    def execution_log(self) -> list[CellExecutionEvent]:
        return list(self._log)

    @property
    def rendered_output(self) -> str:
        return "".join(self._output_parts)

    @property
    def generation(self) -> int:
        return self._generation

    def interrupt(self) -> None:
        self._interrupted = True

    def resume(self) -> None:
        self._continue_requested = False
        self._interrupted = False
        self._generation += 1
        self._parser.reset()

    async def feed(self, chunk: str) -> list[CellExecutionEvent]:
        if self._interrupted or self._continue_requested:
            return []
        cells = self._parser.feed(chunk)
        return await self._execute_cells(cells)

    async def flush(self) -> list[CellExecutionEvent]:
        if self._interrupted or self._continue_requested:
            return []
        cells = self._parser.flush()
        return await self._execute_cells(cells)

    async def _execute_cells(self, cells: list[Cell]) -> list[CellExecutionEvent]:
        events: list[CellExecutionEvent] = []
        for cell in cells:
            if self._interrupted or self._continue_requested:
                event = CellExecutionEvent(
                    cell=cell,
                    cell_index=self._cell_counter,
                    result=None,
                    status="pending",
                    generation=self._generation,
                )
                self._log.append(event)
                events.append(event)
                self._cell_counter += 1
                continue

            index = self._cell_counter
            self._cell_counter += 1

            self._notify_start(cell, index)
            result = self._kernel.execute_cell(cell, index)

            if result.success:
                if result.namespace_delta.get("__continue_requested__"):
                    event = CellExecutionEvent(
                        cell=cell,
                        cell_index=index,
                        result=result,
                        status="continue_requested",
                        generation=self._generation,
                    )
                    self._log.append(event)
                    events.append(event)
                    self._continue_requested = True
                    continue

                if result.output:
                    self._output_parts.append(result.output)
                self._notify_success(cell, result)
                event = CellExecutionEvent(
                    cell=cell,
                    cell_index=index,
                    result=result,
                    status="executed",
                    generation=self._generation,
                )
                self._log.append(event)
                events.append(event)
            else:
                event = await self._handle_failure(cell, index, result)
                self._log.append(event)
                events.append(event)
                if event.status not in ("recovered", "skipped"):
                    self._interrupted = True

        return events

    async def _handle_failure(
        self, cell: Cell, index: int, result: CellResult
    ) -> CellExecutionEvent:
        advice = self._collect_advice(cell, result.error or "")
        attempt = 0

        while attempt <= self._recovery.max_attempts:
            ctx = RecoveryContext(
                failed_cell=cell,
                error=result.error or "",
                error_phase=result.error_phase or "runtime",
                scope=self._kernel.get_scope(),
                cell_index=index,
                prior_output=self.rendered_output,
                attempt=attempt,
                plugin_advice=advice,
            )

            action = self._recovery.on_cell_error(ctx)

            if action.kind == "abort":
                self._notify_recovery_result(cell, False, attempt)
                return CellExecutionEvent(
                    cell=cell, cell_index=index, result=result,
                    status="aborted", recovery_attempts=attempt,
                    generation=self._generation,
                )

            if action.kind == "skip":
                self._notify_recovery_result(cell, False, attempt)
                return CellExecutionEvent(
                    cell=cell, cell_index=index, result=result,
                    status="skipped", recovery_attempts=attempt,
                    generation=self._generation,
                )

            if action.kind == "inspect":
                inspect_result = self._kernel.inspect(action.expr or "None")
                action = self._recovery.on_inspect_result(ctx, inspect_result)
                if action.kind != "fix":
                    attempt += 1
                    continue

            if action.kind == "fix" and action.cells:
                fix_succeeded = True
                fix_results: list[CellResult] = []
                for fix_cell in action.cells:
                    fix_result = self._kernel.execute_cell(fix_cell, index)
                    fix_results.append(fix_result)
                    if not fix_result.success:
                        fix_succeeded = False
                        result = fix_result
                        break
                    if fix_result.output:
                        self._output_parts.append(fix_result.output)

                if fix_succeeded:
                    self._notify_recovery_result(cell, True, attempt)
                    final_result = fix_results[-1] if fix_results else result
                    return CellExecutionEvent(
                        cell=action.cells[-1], cell_index=index,
                        result=final_result, status="recovered",
                        recovery_attempts=attempt + 1,
                        generation=self._generation,
                    )

            attempt += 1

        self._notify_recovery_result(cell, False, attempt)
        return CellExecutionEvent(
            cell=cell, cell_index=index, result=result,
            status="aborted", recovery_attempts=attempt,
            generation=self._generation,
        )

    def _notify_start(self, cell: Cell, index: int) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_cell_start(cell, index)
            except Exception:
                pass

    def _notify_success(self, cell: Cell, result: CellResult) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_cell_success(cell, result)
            except Exception:
                pass

    def _notify_recovery_result(self, cell: Cell, success: bool, attempt: int) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_recovery_result(cell, success, attempt)
            except Exception:
                pass

    def _collect_advice(self, cell: Cell, error: str) -> PluginAdvice:
        advices: list[PluginAdvice] = []
        scope = self._kernel.get_scope()
        for plugin in self._plugins:
            try:
                advice = plugin.on_cell_error(cell, error, scope)
                if advice:
                    advices.append(advice)
            except Exception:
                pass
        return merge_advice(advices)
```

Update `src/lackpy/interpreters/literate/kernel/__init__.py`:
```python
"""Incremental cell execution kernel."""

from .driver import CellExecutionEvent, StreamingDriver
from .interface import CellResult, KernelInterface
from .lightweight import LightweightKernel
from .plugins import ExecutionPlugin, PluginAdvice, merge_advice
from .recovery import NoRecoveryHandler, RecoveryAction, RecoveryContext, RecoveryHandler
from .streaming_parser import StreamingCellParser

__all__ = [
    "CellExecutionEvent",
    "CellResult",
    "ExecutionPlugin",
    "KernelInterface",
    "LightweightKernel",
    "NoRecoveryHandler",
    "PluginAdvice",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryHandler",
    "StreamingCellParser",
    "StreamingDriver",
    "merge_advice",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_driver.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/driver.py \
        src/lackpy/interpreters/literate/kernel/__init__.py \
        tests/literate/kernel/test_driver.py
git commit -m "feat(kernel): StreamingDriver with recovery and plugin orchestration"
```

---

### Task 8: Format Converters

**Files:**
- Create: `src/lackpy/interpreters/literate/kernel/formats.py`
- Create: `tests/literate/kernel/test_formats.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for format converters - markdown / Cell / .ipynb."""

import pytest

from lackpy.interpreters.literate.kernel.driver import CellExecutionEvent
from lackpy.interpreters.literate.kernel.formats import (
    from_notebook,
    render_markdown,
    to_notebook,
)
from lackpy.interpreters.literate.kernel.interface import CellResult
from lackpy.interpreters.literate.parser import Cell, Frontmatter


def _event(cell: Cell, index: int, output: str = "", status: str = "executed") -> CellExecutionEvent:
    return CellExecutionEvent(
        cell=cell,
        cell_index=index,
        result=CellResult(
            success=True, output=output, error=None,
            error_phase=None, namespace_delta={}, cell_index=index,
        ),
        status=status,
        generation=0,
    )


class TestToNotebook:
    def test_basic_structure(self):
        log = [
            _event(Cell(cell_type="prose", content="Hello"), 0, output="Hello\n"),
            _event(Cell(cell_type="hidden", content="x = 1"), 1),
        ]
        nb = to_notebook(log, Frontmatter())
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) == 2

    def test_prose_becomes_markdown_cell(self):
        log = [_event(Cell(cell_type="prose", content="# Title"), 0)]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert nb["cells"][0]["source"] == "# Title"

    def test_code_becomes_code_cell(self):
        log = [_event(Cell(cell_type="code", content="x = 42"), 0)]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["cell_type"] == "code"
        assert nb["cells"][0]["source"] == "x = 42"

    def test_hidden_becomes_code_with_metadata(self):
        log = [_event(Cell(cell_type="hidden", content="y = 1"), 0)]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["cell_type"] == "code"
        assert nb["cells"][0]["metadata"]["lackpy"]["cell_type"] == "hidden"

    def test_output_captured(self):
        log = [_event(Cell(cell_type="code", content="print(42)"), 0, output="42\n")]
        nb = to_notebook(log, Frontmatter())
        assert nb["cells"][0]["outputs"][0]["text"] == "42\n"

    def test_frontmatter_in_metadata(self):
        fm = Frontmatter(echo="true", output="auto")
        nb = to_notebook([], fm)
        assert nb["metadata"]["lackpy"]["frontmatter"]["echo"] == "true"


class TestFromNotebook:
    def test_roundtrip(self):
        log = [
            _event(Cell(cell_type="prose", content="Hello"), 0),
            _event(Cell(cell_type="hidden", content="x = 1"), 1),
            _event(Cell(cell_type="code", content="print(x)"), 2, output="1\n"),
        ]
        fm = Frontmatter(echo="true")
        nb = to_notebook(log, fm)
        recovered_fm, recovered_cells = from_notebook(nb)
        assert recovered_fm.echo == "true"
        assert len(recovered_cells) == 3
        assert recovered_cells[0].cell_type == "prose"
        assert recovered_cells[0].content == "Hello"
        assert recovered_cells[1].cell_type == "hidden"
        assert recovered_cells[2].cell_type == "code"


class TestRenderMarkdown:
    def test_prose_passthrough(self):
        log = [_event(Cell(cell_type="prose", content="Hello world"), 0)]
        md = render_markdown(log, Frontmatter())
        assert "Hello world" in md

    def test_code_in_fence(self):
        log = [_event(Cell(cell_type="code", content="x = 42"), 0)]
        md = render_markdown(log, Frontmatter())
        assert "```lackpy" in md
        assert "x = 42" in md

    def test_hidden_annotated(self):
        log = [_event(Cell(cell_type="hidden", content="y = 1"), 0)]
        md = render_markdown(log, Frontmatter())
        assert "```lackpy @hidden" in md

    def test_write_annotated_with_path(self):
        cell = Cell(cell_type="write", content="code", annotation_args={"path": "out.py"})
        log = [_event(cell, 0)]
        md = render_markdown(log, Frontmatter())
        assert "```lackpy @write(out.py)" in md

    def test_frontmatter_included(self):
        fm = Frontmatter(echo="false", output="json")
        log = [_event(Cell(cell_type="prose", content="Hi"), 0)]
        md = render_markdown(log, fm)
        assert "---" in md
        assert "echo: false" in md

    def test_skipped_cells_omitted(self):
        log = [
            _event(Cell(cell_type="code", content="bad"), 0, status="skipped"),
            _event(Cell(cell_type="prose", content="Good"), 1),
        ]
        md = render_markdown(log, Frontmatter())
        assert "bad" not in md
        assert "Good" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_formats.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/lackpy/interpreters/literate/kernel/formats.py`:
```python
"""Format converters: execution log / .ipynb / markdown."""

from __future__ import annotations

from typing import Any

from ..parser import Cell, Frontmatter
from .driver import CellExecutionEvent

_CODE_CELL_TYPES = {"code", "hidden", "gather", "scratch", "continue", "read", "write", "diff"}


def to_notebook(log: list[CellExecutionEvent], frontmatter: Frontmatter) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for event in log:
        cell = event.cell
        nb_cell: dict[str, Any] = {
            "metadata": {
                "lackpy": {
                    "cell_type": cell.cell_type,
                    "annotation_args": cell.annotation_args,
                    "status": event.status,
                    "recovery_attempts": event.recovery_attempts,
                    "generation": event.generation,
                }
            },
            "outputs": [],
        }

        if cell.cell_type == "prose":
            nb_cell["cell_type"] = "markdown"
            nb_cell["source"] = cell.content
        else:
            nb_cell["cell_type"] = "code"
            nb_cell["source"] = cell.content
            nb_cell["execution_count"] = event.cell_index + 1
            output_text = event.result.output if event.result and event.result.output else ""
            if output_text:
                nb_cell["outputs"].append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output_text,
                })

        cells.append(nb_cell)

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "lackpy": {
                "frontmatter": {
                    "echo": frontmatter.echo,
                    "output": frontmatter.output,
                    "interpreter": frontmatter.interpreter,
                },
            },
            "kernelspec": {
                "display_name": "Lackpy Literate",
                "language": "python",
                "name": "lackpy-literate",
            },
        },
        "cells": cells,
    }


def from_notebook(nb: dict[str, Any]) -> tuple[Frontmatter, list[Cell]]:
    fm_data = nb.get("metadata", {}).get("lackpy", {}).get("frontmatter", {})
    frontmatter = Frontmatter(
        echo=fm_data.get("echo", "true"),
        output=fm_data.get("output", "auto"),
        interpreter=fm_data.get("interpreter", "python"),
    )

    cells: list[Cell] = []
    for nb_cell in nb.get("cells", []):
        lackpy_meta = nb_cell.get("metadata", {}).get("lackpy", {})
        cell_type = lackpy_meta.get("cell_type", "code")
        annotation_args = lackpy_meta.get("annotation_args", {})

        if cell_type == "prose" or nb_cell.get("cell_type") == "markdown":
            cell_type = "prose"

        cells.append(Cell(
            cell_type=cell_type,
            content=nb_cell.get("source", ""),
            annotation_args=annotation_args,
        ))

    return frontmatter, cells


def render_markdown(log: list[CellExecutionEvent], frontmatter: Frontmatter) -> str:
    parts: list[str] = []

    has_non_default_fm = (
        frontmatter.echo != "true"
        or frontmatter.output != "auto"
        or frontmatter.interpreter != "python"
    )
    if has_non_default_fm:
        parts.append("---")
        if frontmatter.echo != "true":
            parts.append(f"echo: {frontmatter.echo}")
        if frontmatter.output != "auto":
            parts.append(f"output: {frontmatter.output}")
        if frontmatter.interpreter != "python":
            parts.append(f"interpreter: {frontmatter.interpreter}")
        parts.append("---")
        parts.append("")

    for event in log:
        if event.status == "skipped":
            continue

        cell = event.cell
        if cell.cell_type == "prose":
            parts.append(cell.content)
            parts.append("")
        elif cell.cell_type == "code":
            parts.append("```lackpy")
            parts.append(cell.content)
            parts.append("```")
            parts.append("")
        elif cell.cell_type in _CODE_CELL_TYPES:
            annotation = f"@{cell.cell_type}"
            if cell.annotation_args.get("path"):
                annotation += f"({cell.annotation_args['path']})"
            parts.append(f"```lackpy {annotation}")
            parts.append(cell.content)
            parts.append("```")
            parts.append("")

    return "\n".join(parts).rstrip("\n") + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_formats.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/formats.py \
        tests/literate/kernel/test_formats.py
git commit -m "feat(kernel): format converters for .ipynb and markdown round-trip"
```

---

### Task 9: InferenceRecoveryHandler

**Files:**
- Modify: `src/lackpy/interpreters/literate/kernel/recovery.py`
- Create: `tests/literate/kernel/test_inference_recovery.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for InferenceRecoveryHandler - model-based recovery."""

import pytest

from lackpy.interpreters.literate.kernel.recovery import (
    InferenceRecoveryHandler,
    RecoveryAction,
    RecoveryContext,
)
from lackpy.interpreters.literate.kernel.plugins import PluginAdvice
from lackpy.interpreters.literate.parser import Cell


def _make_context(**kwargs) -> RecoveryContext:
    defaults = {
        "failed_cell": Cell(cell_type="prose", content="Value: {x:.1f if x > 0 else y:.1f}"),
        "error": "Invalid format specifier",
        "error_phase": "runtime",
        "scope": {"x": "float: 3.14", "y": "float: 2.71"},
        "cell_index": 5,
        "prior_output": "Previous output\n",
        "attempt": 0,
        "plugin_advice": None,
    }
    defaults.update(kwargs)
    return RecoveryContext(**defaults)


class TestInferenceRecoveryHandler:
    def test_fix_response_parsed(self):
        async def fake_infer(prompt: str) -> str:
            return "```lackpy @hidden\nval = x if x > 0 else y\n```\n\nValue: {val:.1f}"

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "fix"
        assert len(action.cells) == 2
        assert action.cells[0].cell_type == "hidden"
        assert action.cells[1].cell_type == "prose"

    def test_scratch_response_becomes_inspect(self):
        async def fake_infer(prompt: str) -> str:
            return "```lackpy @scratch\ntype(x)\n```"

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "inspect"
        assert "type(x)" in action.expr

    def test_empty_response_aborts(self):
        async def fake_infer(prompt: str) -> str:
            return ""

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "abort"

    def test_plugin_advice_included_in_prompt(self):
        prompts_seen: list[str] = []

        async def fake_infer(prompt: str) -> str:
            prompts_seen.append(prompt)
            return "```lackpy @hidden\nfoo = 1\n```"

        advice = PluginAdvice(hints=["Try importing the module"], doc_context=["see docs/x.md"])
        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=2)
        ctx = _make_context(plugin_advice=advice)
        handler.on_cell_error(ctx)
        assert "Try importing the module" in prompts_seen[0]

    def test_inspect_result_fed_back(self):
        call_count = [0]

        async def fake_infer(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "```lackpy @scratch\ntype(x)\n```"
            return "```lackpy @hidden\nval = float(x)\n```\n\nValue: {val:.1f}"

        handler = InferenceRecoveryHandler(infer_fn=fake_infer, max_attempts=3)
        ctx = _make_context()
        action = handler.on_cell_error(ctx)
        assert action.kind == "inspect"

        action = handler.on_inspect_result(ctx, "'float'")
        assert action.kind == "fix"
        assert len(action.cells) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/literate/kernel/test_inference_recovery.py -v`
Expected: FAIL with `ImportError: cannot import name 'InferenceRecoveryHandler'`

- [ ] **Step 3: Write minimal implementation**

Add to end of `src/lackpy/interpreters/literate/kernel/recovery.py`:

```python
import asyncio
from collections.abc import Awaitable, Callable

from .streaming_parser import StreamingCellParser as _RecoveryParser

InferFn = Callable[[str], Awaitable[str]]


class InferenceRecoveryHandler:
    def __init__(self, infer_fn: InferFn, max_attempts: int = 2) -> None:
        self.max_attempts = max_attempts
        self._infer_fn = infer_fn

    def on_cell_error(self, ctx: RecoveryContext) -> RecoveryAction:
        prompt = self._build_prompt(ctx)
        response = self._call_infer(prompt)
        return self._parse_response(response)

    def on_inspect_result(self, ctx: RecoveryContext, result: str) -> RecoveryAction:
        prompt = self._build_inspect_followup(ctx, result)
        response = self._call_infer(prompt)
        return self._parse_response(response)

    def _call_infer(self, prompt: str) -> str:
        try:
            return asyncio.run(self._infer_fn(prompt))
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._infer_fn(prompt))
                return future.result()
        except Exception:
            return ""

    def _build_prompt(self, ctx: RecoveryContext) -> str:
        parts = [
            "A cell failed during execution.\n",
            f"Cell type: {ctx.failed_cell.cell_type}",
            f"Cell content:\n{ctx.failed_cell.content}\n",
            f"Error ({ctx.error_phase}): {ctx.error}\n",
            "Variables in scope:",
        ]
        for name, summary in ctx.scope.items():
            parts.append(f"  {name} = {summary}")

        if ctx.plugin_advice:
            if ctx.plugin_advice.hints:
                parts.append("\nCoaching hints:")
                for hint in ctx.plugin_advice.hints:
                    parts.append(f"  - {hint}")
            if ctx.plugin_advice.doc_context:
                parts.append("\nRelevant documentation:")
                for doc in ctx.plugin_advice.doc_context:
                    parts.append(f"  {doc}")

        parts.append(
            "\nFix this cell. Return replacement cells as a literate document fragment. "
            "You may add @hidden blocks before the cell to pre-compute values. "
            "Use @scratch if you need to inspect a value first."
        )
        return "\n".join(parts)

    def _build_inspect_followup(self, ctx: RecoveryContext, result: str) -> str:
        return (
            f"Inspection result: {result}\n\n"
            f"Original error: {ctx.error}\n"
            f"Original cell:\n{ctx.failed_cell.content}\n\n"
            "Now provide the fix as a literate document fragment."
        )

    def _parse_response(self, response: str) -> RecoveryAction:
        if not response.strip():
            return RecoveryAction(kind="abort")

        parser = _RecoveryParser()
        cells = parser.feed(response)
        cells.extend(parser.flush())

        if not cells:
            return RecoveryAction(kind="abort")

        if len(cells) == 1 and cells[0].cell_type == "scratch":
            return RecoveryAction(kind="inspect", expr=cells[0].content.strip())

        return RecoveryAction(kind="fix", cells=cells)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/literate/kernel/test_inference_recovery.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/kernel/recovery.py \
        tests/literate/kernel/test_inference_recovery.py
git commit -m "feat(kernel): InferenceRecoveryHandler with model-based cell recovery"
```

---

### Task 10: Backward-Compatible LiterateInterpreter Wrapper

**Files:**
- Modify: `src/lackpy/interpreters/literate/__init__.py`
- Test: `tests/literate/test_interpreter.py` (existing tests must still pass)

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `pytest tests/literate/test_interpreter.py -v`
Expected: PASS (all existing tests)

- [ ] **Step 2: Refactor execute() to use kernel**

Replace the body of `src/lackpy/interpreters/literate/__init__.py`:

```python
"""Literate programming interpreter for lackpy.

Executes markdown documents with embedded ```lackpy code blocks.
The compilation pipeline: parse markdown -> cell sequence -> execute
cell-by-cell via kernel -> captured stdout IS the rendered document.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from ..base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from .kernel import LightweightKernel, NoRecoveryHandler, StreamingDriver
from .parser import parse
from .prompt import LITERATE_HINT
from .tools import make_tool_namespace


class LiterateInterpreter:
    """Literate programming interpreter.

    Takes a markdown document with ```lackpy code blocks, compiles it
    to Python, and executes it. The captured stdout is the rendered
    document - prose becomes print() calls, code executes and its
    output joins the stream.
    """

    name = "literate"
    description = "Literate programming - markdown with embedded lackpy code blocks"

    def system_prompt_hint(self) -> str:
        return LITERATE_HINT

    def validate(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterValidationResult:
        """Validate a literate document by parsing it."""
        result = parse(program)
        if result.errors:
            return InterpreterValidationResult(
                valid=False,
                errors=result.errors,
            )
        return InterpreterValidationResult(valid=True)

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterExecutionResult:
        """Execute a literate document and return the rendered output."""
        start = time.perf_counter()

        parsed = parse(program)
        if parsed.errors:
            return InterpreterExecutionResult(
                success=False,
                error="Parse errors: " + "; ".join(parsed.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        namespace = _build_namespace(context)
        kernel = LightweightKernel(namespace=namespace)
        driver = StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler())

        prev_cwd = os.getcwd()
        if context.base_dir:
            os.chdir(context.base_dir)

        try:
            for cell in parsed.cells:
                events = await driver._execute_cells([cell])
                for event in events:
                    if event.status == "continue_requested":
                        break
                    if event.status not in ("executed", "recovered"):
                        error_msg = event.result.error if event.result else "Unknown error"
                        return InterpreterExecutionResult(
                            success=False,
                            error=error_msg,
                            output_format="text",
                            duration_ms=(time.perf_counter() - start) * 1000,
                        )
        finally:
            os.chdir(prev_cwd)

        elapsed = (time.perf_counter() - start) * 1000
        variables = kernel.get_namespace()
        continue_requested = any(
            e.status == "continue_requested" for e in driver.execution_log
        )

        return InterpreterExecutionResult(
            success=True,
            output=driver.rendered_output,
            output_format="markdown",
            duration_ms=elapsed,
            metadata={
                "variables": variables,
                "continue_requested": continue_requested,
                "cell_count": len(parsed.cells),
                "frontmatter": {
                    "echo": parsed.frontmatter.echo,
                    "output": parsed.frontmatter.output,
                    "interpreter": parsed.frontmatter.interpreter,
                },
            },
        )


_INTERNAL_NAMES = frozenset({
    "read_file", "write_file", "apply_diff",
    "search_content", "run_command", "run_tests",
    "__literate_continue__", "__builtins__",
})


def _build_namespace(context: ExecutionContext) -> dict[str, Any]:
    """Build the execution namespace for a literate document."""
    ns: dict[str, Any] = {}
    ns.update(make_tool_namespace(context.base_dir))

    import builtins as _builtins_mod
    ns["__builtins__"] = _builtins_mod

    if context.kit:
        for name, fn in context.kit.callables.items():
            ns[name] = fn

    if context.params:
        ns.update(context.params)

    return ns
```

- [ ] **Step 3: Run existing tests to verify backward compat**

Run: `pytest tests/literate/test_interpreter.py -v`
Expected: PASS (all existing tests still pass)

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/literate/ -v`
Expected: PASS (all tests — both old and new)

- [ ] **Step 5: Commit**

```bash
git add src/lackpy/interpreters/literate/__init__.py
git commit -m "refactor(literate): use kernel-based execution in LiterateInterpreter"
```

---

### Task 11: End-to-End Integration Test

**Files:**
- Create: `tests/literate/kernel/test_e2e.py`

- [ ] **Step 1: Write integration tests**

```python
"""End-to-end tests: full documents through StreamingDriver."""

import os

import pytest

from lackpy.interpreters.literate.kernel import (
    LightweightKernel,
    NoRecoveryHandler,
    StreamingDriver,
)
from lackpy.interpreters.literate.kernel.formats import from_notebook, render_markdown, to_notebook
from lackpy.interpreters.literate.parser import Frontmatter


@pytest.fixture
def driver():
    kernel = LightweightKernel()
    return StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler())


class TestFullDocumentExecution:
    @pytest.mark.asyncio
    async def test_analysis_document(self, driver):
        doc = (
            "---\necho: true\n---\n\n# Report\n\n"
            "```lackpy @hidden\ndata = [1, 2, 3, 4, 5]\ntotal = sum(data)\n```\n\n"
            "The dataset has {len(data)} items totaling {total}.\n\n"
            "```lackpy\nmean = total / len(data)\nprint(f\"Mean: {mean:.1f}\")\n```\n\n"
            "Average value: {mean:.1f}\n"
        )
        for line in doc.split("\n"):
            await driver.feed(line + "\n")
        await driver.flush()

        output = driver.rendered_output
        assert "5 items" in output
        assert "totaling 15" in output
        assert "Mean: 3.0" in output
        assert "Average value: 3.0" in output

    @pytest.mark.asyncio
    async def test_document_with_write(self, tmp_path, driver):
        os.chdir(tmp_path)
        from lackpy.interpreters.literate.tools import write_file
        driver._kernel._namespace["write_file"] = write_file

        doc = (
            "```lackpy @write(hello.py)\nprint(\"hello world\")\n```\n\n"
            "```lackpy @hidden\nimport os\nexists = os.path.exists(\"hello.py\")\n```\n\n"
            "File created: {exists}\n"
        )
        await driver.feed(doc)
        await driver.flush()

        assert "True" in driver.rendered_output
        assert (tmp_path / "hello.py").exists()


class TestNotebookRoundTrip:
    @pytest.mark.asyncio
    async def test_execute_export_reimport(self, driver):
        doc = "```lackpy @hidden\nx = 42\n```\n\nThe answer: {x}\n"
        await driver.feed(doc)
        await driver.flush()

        fm = Frontmatter()
        nb = to_notebook(driver.execution_log, fm)
        recovered_fm, recovered_cells = from_notebook(nb)

        assert len(recovered_cells) == 2
        assert recovered_cells[0].cell_type == "hidden"
        assert recovered_cells[1].cell_type == "prose"

        kernel2 = LightweightKernel()
        driver2 = StreamingDriver(kernel=kernel2, recovery=NoRecoveryHandler())
        for cell in recovered_cells:
            await driver2._execute_cells([cell])
        assert "42" in driver2.rendered_output


class TestMarkdownRoundTrip:
    @pytest.mark.asyncio
    async def test_render_clean_markdown(self, driver):
        doc = "```lackpy @hidden\nx = 1\n```\n\nValue: {x}\n"
        await driver.feed(doc)
        await driver.flush()

        md = render_markdown(driver.execution_log, Frontmatter())
        assert "```lackpy @hidden" in md
        assert "x = 1" in md
        assert "Value: {x}" in md
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/literate/kernel/test_e2e.py -v`
Expected: PASS (all integration tests)

- [ ] **Step 3: Run complete test suite**

Run: `pytest tests/literate/ -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
git add tests/literate/kernel/test_e2e.py
git commit -m "test(kernel): end-to-end integration tests for streaming execution"
```
