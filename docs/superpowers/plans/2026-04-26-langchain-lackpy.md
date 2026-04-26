# langchain-lackpy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thin bridge package exposing lackpy's safe, graded program execution to the langchain/langgraph ecosystem.

**Architecture:** `packages/langchain-lackpy/` in the lackpy monorepo. `LackpyToolkit` (BaseToolkit) is the single entry point — it produces individual tool wrappers, a delegate tool, and a LangGraph node. An ArgSpec-to-Python-type mapping utility is added to lackpy core.

**Tech Stack:** `langchain-core>=0.3`, `lackpy>=0.9`, optional `langgraph>=0.3`. Pydantic v2 (via langchain-core). pytest + pytest-asyncio for tests.

---

## File Structure

```
packages/langchain-lackpy/
  langchain_lackpy/
    __init__.py          — re-exports LackpyToolkit
    toolkit.py           — LackpyToolkit (BaseToolkit subclass)
    _tool_wrapper.py     — LackpyToolWrapper (BaseTool for individual tools)
    _delegate.py         — LackpyDelegateTool (BaseTool wrapping delegate())
    _node.py             — as_node() factory returning async LangGraph node fn
    _schema.py           — args_schema_from_argspecs() Pydantic model builder
  tests/
    conftest.py          — shared fixtures: mock service, toolbox, resolved kit
    test_schema.py       — ArgSpec → Pydantic model tests
    test_tool_wrapper.py — individual tool wrapping tests
    test_delegate.py     — delegate tool tests
    test_toolkit.py      — toolkit construction and get_tools/as_delegate/as_node
    test_node.py         — LangGraph node tests
  pyproject.toml
  README.md
```

Also modified in lackpy core:
- `src/lackpy/kit/toolbox.py` — add `ARGSPEC_TYPE_MAP` and `resolve_python_type()` utility
- `tests/kit/test_toolbox.py` — tests for the new utility

---

### Task 1: Package scaffold and pyproject.toml

**Files:**
- Create: `packages/langchain-lackpy/pyproject.toml`
- Create: `packages/langchain-lackpy/langchain_lackpy/__init__.py`
- Create: `packages/langchain-lackpy/README.md`

- [ ] **Step 1: Create the package directory structure**

```bash
mkdir -p packages/langchain-lackpy/langchain_lackpy
mkdir -p packages/langchain-lackpy/tests
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "langchain-lackpy"
version = "0.1.0"
description = "LangChain integration for lackpy — safe, graded, restricted Python execution."
requires-python = ">=3.10"
dependencies = [
    "langchain-core>=0.3,<1.0",
    "lackpy>=0.9",
]

[project.optional-dependencies]
graph = ["langgraph>=0.3"]
dev = ["pytest>=8", "pytest-asyncio>=0.24"]

[tool.hatch.build.targets.wheel]
packages = ["langchain_lackpy"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Write the initial __init__.py**

```python
"""LangChain integration for lackpy."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write README.md**

```markdown
# langchain-lackpy

LangChain integration for [lackpy](https://github.com/teague/lackpy) — safe, graded, restricted Python execution as a LangChain tool.

## Installation

```bash
pip install langchain-lackpy
```

For LangGraph support:

```bash
pip install langchain-lackpy[graph]
```

## Quick Start

```python
from lackpy import LackpyService
from langchain_lackpy import LackpyToolkit

svc = LackpyService(workspace=Path("."))
toolkit = LackpyToolkit(service=svc, kit=["read_file", "find_files"])

# Individual tools for ReAct agents
tools = toolkit.get_tools()

# Single delegate tool (safe PythonREPLTool replacement)
delegate = toolkit.as_delegate()
```
```

- [ ] **Step 5: Commit**

```bash
git add packages/langchain-lackpy/
git commit -m "scaffold: langchain-lackpy package structure"
```

---

### Task 2: ArgSpec type mapping in lackpy core

**Files:**
- Modify: `src/lackpy/kit/toolbox.py`
- Create: `tests/kit/test_type_map.py`

- [ ] **Step 1: Write the failing test**

Create `tests/kit/test_type_map.py`:

```python
"""Tests for ArgSpec type string → Python type mapping."""

from __future__ import annotations

import pytest
from typing import Any

from lackpy.kit.toolbox import ARGSPEC_TYPE_MAP, resolve_python_type


class TestArgspecTypeMap:
    def test_str_maps_to_str(self):
        assert resolve_python_type("str") is str

    def test_int_maps_to_int(self):
        assert resolve_python_type("int") is int

    def test_float_maps_to_float(self):
        assert resolve_python_type("float") is float

    def test_bool_maps_to_bool(self):
        assert resolve_python_type("bool") is bool

    def test_dict_maps_to_dict(self):
        assert resolve_python_type("dict") is dict

    def test_list_maps_to_list(self):
        assert resolve_python_type("list") is list

    def test_any_maps_to_any(self):
        assert resolve_python_type("Any") is Any

    def test_unknown_falls_back_to_any(self):
        assert resolve_python_type("SomeCustomType") is Any

    def test_list_str_falls_back_to_any(self):
        assert resolve_python_type("list[str]") is Any

    def test_map_contains_all_base_types(self):
        expected = {"str", "int", "float", "bool", "dict", "list", "Any"}
        assert set(ARGSPEC_TYPE_MAP.keys()) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/kit/test_type_map.py -v`
Expected: FAIL with `ImportError: cannot import name 'ARGSPEC_TYPE_MAP'`

- [ ] **Step 3: Implement the type mapping**

Add to the end of `src/lackpy/kit/toolbox.py`, after the `Toolbox` class:

```python
from typing import Any as _Any

ARGSPEC_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
    "Any": _Any,
}


def resolve_python_type(type_str: str) -> type:
    """Map an ArgSpec type string to a Python type.

    Returns ``Any`` for unrecognized type strings (e.g. ``"list[str]"``).
    """
    return ARGSPEC_TYPE_MAP.get(type_str, _Any)
```

Also add to the existing `from typing import Any` import at the top — use `_Any` alias to avoid shadowing the existing `Any` usage. Alternatively, add the import inline as shown.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/kit/test_type_map.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Export from lackpy package**

Add to `src/lackpy/__init__.py` imports:

```python
from .kit.toolbox import Toolbox, ToolSpec, ArgSpec, ARGSPEC_TYPE_MAP, resolve_python_type
```

And add `"ARGSPEC_TYPE_MAP", "resolve_python_type"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add src/lackpy/kit/toolbox.py src/lackpy/__init__.py tests/kit/test_type_map.py
git commit -m "feat(kit): ArgSpec type string → Python type mapping utility"
```

---

### Task 3: Schema builder (_schema.py)

**Files:**
- Create: `packages/langchain-lackpy/langchain_lackpy/_schema.py`
- Create: `packages/langchain-lackpy/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

Create `packages/langchain-lackpy/tests/conftest.py`:

```python
"""Shared fixtures for langchain-lackpy tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest

from lackpy.kit.toolbox import ArgSpec, ToolSpec, Toolbox
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


@pytest.fixture
def sample_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="read_file", provider="builtin",
            description="Read file contents",
            args=[ArgSpec(name="path", type="str", description="File path")],
            returns="str", grade_w=1, effects_ceiling=1,
        ),
        ToolSpec(
            name="write_file", provider="builtin",
            description="Write content to a file",
            args=[
                ArgSpec(name="path", type="str", description="File path"),
                ArgSpec(name="content", type="str", description="Content to write"),
            ],
            returns="bool", grade_w=3, effects_ceiling=3,
        ),
    ]


@pytest.fixture
def mock_toolbox(sample_specs) -> Toolbox:
    toolbox = Toolbox()
    for spec in sample_specs:
        toolbox.register_tool(spec)
    toolbox._providers["builtin"] = MagicMock(
        name="builtin",
        resolve=lambda spec: lambda **kw: f"mock:{spec.name}({kw})",
    )
    return toolbox


@pytest.fixture
def resolved_kit(mock_toolbox, sample_specs) -> ResolvedKit:
    from lackpy.kit.registry import resolve_kit
    return resolve_kit(["read_file", "write_file"], mock_toolbox)


@pytest.fixture
def mock_service(mock_toolbox, tmp_path) -> MagicMock:
    svc = MagicMock()
    svc.toolbox = mock_toolbox
    svc.workspace = tmp_path
    svc.delegate = AsyncMock(return_value={
        "success": True,
        "output": "file contents here",
        "error": None,
        "program": "x = read_file('test.txt')",
        "grade": {"w": 1, "d": 1},
        "trace": [],
        "files_read": ["test.txt"],
        "files_modified": [],
        "generation_time_ms": 100,
        "execution_time_ms": 50,
        "total_time_ms": 150,
    })
    return svc
```

Create `packages/langchain-lackpy/tests/test_schema.py`:

```python
"""Tests for ArgSpec → Pydantic args_schema conversion."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lackpy.kit.toolbox import ArgSpec

from langchain_lackpy._schema import args_schema_from_argspecs


class TestArgsSchemaFromArgspecs:
    def test_single_str_arg(self):
        specs = [ArgSpec(name="path", type="str", description="File path")]
        model = args_schema_from_argspecs("read_file", specs)
        assert issubclass(model, BaseModel)
        fields = model.model_fields
        assert "path" in fields
        assert fields["path"].annotation is str

    def test_multiple_args(self):
        specs = [
            ArgSpec(name="path", type="str", description="File path"),
            ArgSpec(name="content", type="str", description="Content"),
        ]
        model = args_schema_from_argspecs("write_file", specs)
        assert set(model.model_fields.keys()) == {"path", "content"}

    def test_int_arg_type(self):
        specs = [ArgSpec(name="count", type="int", description="Number of items")]
        model = args_schema_from_argspecs("my_tool", specs)
        assert model.model_fields["count"].annotation is int

    def test_any_arg_type(self):
        from typing import Any
        specs = [ArgSpec(name="data", type="Any", description="Arbitrary data")]
        model = args_schema_from_argspecs("my_tool", specs)
        assert model.model_fields["data"].annotation is Any

    def test_unknown_type_falls_back_to_any(self):
        from typing import Any
        specs = [ArgSpec(name="data", type="SomeCustomType", description="Custom")]
        model = args_schema_from_argspecs("my_tool", specs)
        assert model.model_fields["data"].annotation is Any

    def test_field_descriptions_set(self):
        specs = [ArgSpec(name="path", type="str", description="The file path")]
        model = args_schema_from_argspecs("read_file", specs)
        assert model.model_fields["path"].description == "The file path"

    def test_model_name_derived_from_tool(self):
        specs = [ArgSpec(name="x", type="str")]
        model = args_schema_from_argspecs("read_file", specs)
        assert model.__name__ == "ReadFileInput"

    def test_empty_args_produces_empty_model(self):
        model = args_schema_from_argspecs("no_args_tool", [])
        assert len(model.model_fields) == 0

    def test_model_validates_input(self):
        specs = [ArgSpec(name="count", type="int", description="N")]
        model = args_schema_from_argspecs("my_tool", specs)
        instance = model(count=5)
        assert instance.count == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pip install -e ".[dev]" -q && PYTHONPATH=../../src pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_lackpy._schema'`

Note: For all subsequent test runs in this package, use `PYTHONPATH=../../src` to ensure lackpy is importable, or install lackpy in editable mode as well.

- [ ] **Step 3: Implement _schema.py**

Create `packages/langchain-lackpy/langchain_lackpy/_schema.py`:

```python
"""ArgSpec list → Pydantic BaseModel conversion."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic import create_model as _create_model

from lackpy.kit.toolbox import ArgSpec, resolve_python_type


def args_schema_from_argspecs(tool_name: str, argspecs: list[ArgSpec]) -> type[BaseModel]:
    """Build a Pydantic model from a list of ArgSpecs.

    The model name is derived from the tool name: ``read_file`` becomes
    ``ReadFileInput``.
    """
    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Input"

    field_definitions: dict[str, Any] = {}
    for arg in argspecs:
        python_type = resolve_python_type(arg.type)
        field_definitions[arg.name] = (
            python_type,
            Field(description=arg.description or None),
        )

    return _create_model(model_name, **field_definitions)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_schema.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/langchain-lackpy/langchain_lackpy/_schema.py packages/langchain-lackpy/tests/conftest.py packages/langchain-lackpy/tests/test_schema.py
git commit -m "feat(langchain): ArgSpec → Pydantic args_schema builder"
```

---

### Task 4: Individual tool wrapper (_tool_wrapper.py)

**Files:**
- Create: `packages/langchain-lackpy/langchain_lackpy/_tool_wrapper.py`
- Create: `packages/langchain-lackpy/tests/test_tool_wrapper.py`

- [ ] **Step 1: Write the failing test**

Create `packages/langchain-lackpy/tests/test_tool_wrapper.py`:

```python
"""Tests for individual ToolSpec → BaseTool wrapping."""

from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock

from langchain_core.tools import BaseTool

from lackpy.kit.toolbox import ArgSpec, ToolSpec

from langchain_lackpy._tool_wrapper import LackpyToolWrapper


@pytest.fixture
def read_file_spec() -> ToolSpec:
    return ToolSpec(
        name="read_file", provider="builtin",
        description="Read file contents",
        args=[ArgSpec(name="path", type="str", description="File path")],
        returns="str", grade_w=1, effects_ceiling=1,
    )


@pytest.fixture
def read_file_callable():
    return MagicMock(return_value="hello world")


class TestLackpyToolWrapper:
    def test_is_base_tool(self, read_file_spec, read_file_callable):
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        assert isinstance(tool, BaseTool)

    def test_name_from_spec(self, read_file_spec, read_file_callable):
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        assert tool.name == "read_file"

    def test_description_from_spec(self, read_file_spec, read_file_callable):
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        assert "Read file contents" in tool.description

    def test_args_schema_has_correct_fields(self, read_file_spec, read_file_callable):
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        assert "path" in tool.args_schema.model_fields
        assert tool.args_schema.model_fields["path"].annotation is str

    def test_metadata_carries_provider(self, read_file_spec, read_file_callable):
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        assert tool.metadata["provider"] == "builtin"

    def test_metadata_carries_grade(self, read_file_spec, read_file_callable):
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        assert tool.metadata["grade_w"] == 1
        assert tool.metadata["effects_ceiling"] == 1

    def test_run_delegates_to_callable(self, read_file_spec, read_file_callable):
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        result = tool._run(path="/tmp/test.txt")
        read_file_callable.assert_called_once_with(path="/tmp/test.txt")

    def test_run_returns_string(self, read_file_spec, read_file_callable):
        read_file_callable.return_value = 42
        tool = LackpyToolWrapper.from_spec(read_file_spec, read_file_callable)
        result = tool._run(path="/tmp/test.txt")
        assert result == "42"

    def test_multi_arg_tool(self):
        spec = ToolSpec(
            name="write_file", provider="builtin",
            description="Write content to a file",
            args=[
                ArgSpec(name="path", type="str", description="File path"),
                ArgSpec(name="content", type="str", description="Content"),
            ],
            returns="bool", grade_w=3, effects_ceiling=3,
        )
        fn = MagicMock(return_value=True)
        tool = LackpyToolWrapper.from_spec(spec, fn)
        tool._run(path="/tmp/out.txt", content="hello")
        fn.assert_called_once_with(path="/tmp/out.txt", content="hello")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_tool_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_lackpy._tool_wrapper'`

- [ ] **Step 3: Implement _tool_wrapper.py**

Create `packages/langchain-lackpy/langchain_lackpy/_tool_wrapper.py`:

```python
"""Individual ToolSpec → BaseTool wrapper."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from lackpy.kit.toolbox import ToolSpec

from ._schema import args_schema_from_argspecs


class LackpyToolWrapper(BaseTool):
    """A langchain BaseTool wrapping a single lackpy ToolSpec and its callable."""

    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] = BaseModel
    metadata: dict[str, Any] = {}
    _callable: Callable[..., Any]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_spec(cls, spec: ToolSpec, callable_fn: Callable[..., Any]) -> LackpyToolWrapper:
        schema = args_schema_from_argspecs(spec.name, spec.args)
        description = f"{spec.description} [provider={spec.provider}, grade_w={spec.grade_w}]"
        instance = cls(
            name=spec.name,
            description=description,
            args_schema=schema,
            metadata={
                "provider": spec.provider,
                "grade_w": spec.grade_w,
                "effects_ceiling": spec.effects_ceiling,
            },
        )
        instance._callable = callable_fn
        return instance

    def _run(self, **kwargs: Any) -> str:
        result = self._callable(**kwargs)
        return str(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_tool_wrapper.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/langchain-lackpy/langchain_lackpy/_tool_wrapper.py packages/langchain-lackpy/tests/test_tool_wrapper.py
git commit -m "feat(langchain): LackpyToolWrapper — ToolSpec as BaseTool"
```

---

### Task 5: Delegate tool (_delegate.py)

**Files:**
- Create: `packages/langchain-lackpy/langchain_lackpy/_delegate.py`
- Create: `packages/langchain-lackpy/tests/test_delegate.py`

- [ ] **Step 1: Write the failing test**

Create `packages/langchain-lackpy/tests/test_delegate.py`:

```python
"""Tests for the delegate BaseTool wrapping LackpyService.delegate()."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.tools import BaseTool, ToolException

from langchain_lackpy._delegate import LackpyDelegateTool


@pytest.fixture
def delegate_tool(mock_service) -> LackpyDelegateTool:
    return LackpyDelegateTool.create(
        service=mock_service,
        kit_config=["read_file", "write_file"],
        resolved_description="  read_file(path) -> str: Read file contents\n  write_file(path, content) -> bool: Write content",
    )


class TestDelegateToolConstruction:
    def test_is_base_tool(self, delegate_tool):
        assert isinstance(delegate_tool, BaseTool)

    def test_default_name(self, delegate_tool):
        assert delegate_tool.name == "lackpy_delegate"

    def test_custom_name(self, mock_service):
        tool = LackpyDelegateTool.create(
            service=mock_service,
            kit_config=["read_file"],
            resolved_description="read_file(path) -> str",
            name="lackpy_filesystem",
        )
        assert tool.name == "lackpy_filesystem"

    def test_auto_generated_description(self, delegate_tool):
        assert "read_file" in delegate_tool.description
        assert "validated" in delegate_tool.description.lower() or "safe" in delegate_tool.description.lower()

    def test_custom_description(self, mock_service):
        tool = LackpyDelegateTool.create(
            service=mock_service,
            kit_config=["read_file"],
            resolved_description="",
            description="My custom description",
        )
        assert tool.description == "My custom description"

    def test_args_schema_has_intent(self, delegate_tool):
        fields = delegate_tool.args_schema.model_fields
        assert "intent" in fields
        assert fields["intent"].annotation is str


class TestDelegateToolExecution:
    @pytest.mark.asyncio
    async def test_arun_returns_output_on_success(self, delegate_tool, mock_service):
        result = await delegate_tool._arun(intent="read test.txt")
        assert result == "file contents here"
        mock_service.delegate.assert_called_once()

    @pytest.mark.asyncio
    async def test_arun_returns_error_on_failure(self, delegate_tool, mock_service):
        mock_service.delegate.return_value = {
            "success": False,
            "output": None,
            "error": "Tool 'read_file' raised: FileNotFoundError",
        }
        result = await delegate_tool._arun(intent="read missing.txt")
        assert "FileNotFoundError" in result

    @pytest.mark.asyncio
    async def test_arun_passes_kit_config(self, delegate_tool, mock_service):
        await delegate_tool._arun(intent="do something")
        call_kwargs = mock_service.delegate.call_args[1]
        assert call_kwargs["kit"] == ["read_file", "write_file"]

    def test_run_raises_not_implemented(self, delegate_tool):
        with pytest.raises(NotImplementedError):
            delegate_tool._run(intent="read test.txt")

    @pytest.mark.asyncio
    async def test_arun_raises_tool_exception_on_infrastructure_error(self, delegate_tool, mock_service):
        mock_service.delegate.side_effect = KeyError("Unknown tool: 'bad_tool'")
        with pytest.raises(ToolException, match="bad_tool"):
            await delegate_tool._arun(intent="do something")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_delegate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_lackpy._delegate'`

- [ ] **Step 3: Implement _delegate.py**

Create `packages/langchain-lackpy/langchain_lackpy/_delegate.py`:

```python
"""Delegate BaseTool wrapping LackpyService.delegate()."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field


class DelegateInput(BaseModel):
    intent: str = Field(description="Natural language description of what to accomplish")


class LackpyDelegateTool(BaseTool):
    """A langchain tool that delegates to lackpy's generate-then-execute pipeline."""

    name: str = "lackpy_delegate"
    description: str = ""
    args_schema: type[BaseModel] = DelegateInput
    _service: Any
    _kit_config: Any

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
        cls,
        service: Any,
        kit_config: Any,
        resolved_description: str,
        name: str = "lackpy_delegate",
        description: str | None = None,
    ) -> LackpyDelegateTool:
        if description is None:
            description = (
                "Safely compose operations into a validated, restricted Python program. "
                "Programs are graded for safety before execution.\n"
                f"Available operations:\n{resolved_description}"
            )
        instance = cls(name=name, description=description)
        instance._service = service
        instance._kit_config = kit_config
        return instance

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("LackpyDelegateTool is async-only. Use ainvoke().")

    async def _arun(self, intent: str, **kwargs: Any) -> str:
        try:
            result = await self._service.delegate(intent=intent, kit=self._kit_config)
        except Exception as exc:
            raise ToolException(str(exc)) from exc

        if result.get("success"):
            output = result.get("output")
            return str(output) if output is not None else ""
        return result.get("error") or "Unknown error"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_delegate.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/langchain-lackpy/langchain_lackpy/_delegate.py packages/langchain-lackpy/tests/test_delegate.py
git commit -m "feat(langchain): LackpyDelegateTool — safe PythonREPLTool replacement"
```

---

### Task 6: LackpyToolkit (toolkit.py)

**Files:**
- Create: `packages/langchain-lackpy/langchain_lackpy/toolkit.py`
- Create: `packages/langchain-lackpy/tests/test_toolkit.py`

- [ ] **Step 1: Write the failing test**

Create `packages/langchain-lackpy/tests/test_toolkit.py`:

```python
"""Tests for LackpyToolkit — the primary entry point."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_lackpy.toolkit import LackpyToolkit


class TestToolkitConstruction:
    def test_is_base_toolkit(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file", "write_file"])
        assert isinstance(toolkit, BaseToolkit)

    def test_accepts_list_kit(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file"])
        assert toolkit._resolved_kit is not None

    def test_accepts_string_kit(self, mock_service, tmp_path):
        kit_dir = tmp_path / ".lackpy" / "kits"
        kit_dir.mkdir(parents=True)
        (kit_dir / "test.kit").write_text("read_file\nwrite_file\n")
        toolkit = LackpyToolkit(service=mock_service, kit="test", kits_dir=kit_dir)
        assert "read_file" in toolkit._resolved_kit.tools

    def test_accepts_dict_kit(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit={"reader": "read_file"})
        assert "reader" in toolkit._resolved_kit.tools


class TestToolkitFromConfig:
    def test_from_config_creates_toolkit(self, tmp_path, mock_toolbox):
        with patch("langchain_lackpy.toolkit.LackpyService") as MockSvc:
            mock_svc = MagicMock()
            mock_svc.toolbox = mock_toolbox
            MockSvc.return_value = mock_svc
            toolkit = LackpyToolkit.from_config(
                workspace=tmp_path,
                kit=["read_file"],
            )
            assert isinstance(toolkit, LackpyToolkit)


class TestGetTools:
    def test_returns_list_of_base_tools(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file", "write_file"])
        tools = toolkit.get_tools()
        assert len(tools) == 2
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_tool_names_match_kit(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file", "write_file"])
        tools = toolkit.get_tools()
        names = {t.name for t in tools}
        assert names == {"read_file", "write_file"}


class TestAsDelegate:
    def test_returns_base_tool(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file"])
        delegate = toolkit.as_delegate()
        assert isinstance(delegate, BaseTool)

    def test_default_name(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file"])
        delegate = toolkit.as_delegate()
        assert delegate.name == "lackpy_delegate"

    def test_custom_name(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file"])
        delegate = toolkit.as_delegate(name="my_tool")
        assert delegate.name == "my_tool"

    def test_description_mentions_tools(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file"])
        delegate = toolkit.as_delegate()
        assert "read_file" in delegate.description


class TestAsNode:
    def test_returns_callable(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file"])
        node = toolkit.as_node()
        assert callable(node)

    @pytest.mark.asyncio
    async def test_node_calls_delegate(self, mock_service):
        toolkit = LackpyToolkit(service=mock_service, kit=["read_file"])
        node = toolkit.as_node()
        result = await node({"intent": "read test.txt"})
        assert "lackpy_result" in result
        mock_service.delegate.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_toolkit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_lackpy.toolkit'`

- [ ] **Step 3: Implement toolkit.py**

Create `packages/langchain-lackpy/langchain_lackpy/toolkit.py`:

```python
"""LackpyToolkit — primary entry point for langchain-lackpy integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field, PrivateAttr

from lackpy import LackpyService
from lackpy.config import load_config
from lackpy.kit.registry import ResolvedKit, resolve_kit

from ._delegate import LackpyDelegateTool
from ._tool_wrapper import LackpyToolWrapper


class LackpyToolkit(BaseToolkit):
    """Wraps a lackpy kit as a langchain BaseToolkit.

    All langchain-lackpy integration flows through this class.
    Construct it with a service and kit, then call get_tools(),
    as_delegate(), or as_node() to get langchain-compatible objects.
    """

    _service: Any = PrivateAttr()
    _kit_config: Any = PrivateAttr()
    _resolved_kit: ResolvedKit = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        service: LackpyService,
        kit: str | list[str] | dict | None = None,
        kits_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._service = service
        self._kit_config = kit
        self._resolved_kit = resolve_kit(kit, service.toolbox, kits_dir=kits_dir)

    @classmethod
    def from_config(
        cls,
        workspace: Path,
        kit: str | list[str] | dict | None = None,
        kits_dir: Path | None = None,
    ) -> LackpyToolkit:
        config = load_config(workspace)
        service = LackpyService(workspace=workspace, config=config)
        return cls(service=service, kit=kit, kits_dir=kits_dir)

    def get_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for name, spec in self._resolved_kit.tools.items():
            callable_fn = self._resolved_kit.callables[name]
            tools.append(LackpyToolWrapper.from_spec(spec, callable_fn))
        return tools

    def as_delegate(
        self,
        name: str = "lackpy_delegate",
        description: str | None = None,
    ) -> BaseTool:
        return LackpyDelegateTool.create(
            service=self._service,
            kit_config=self._kit_config,
            resolved_description=self._resolved_kit.description,
            name=name,
            description=description,
        )

    def as_node(
        self,
        intent_key: str = "intent",
        result_key: str = "lackpy_result",
    ) -> Callable[..., Any]:
        from ._node import make_node
        return make_node(
            service=self._service,
            kit_config=self._kit_config,
            intent_key=intent_key,
            result_key=result_key,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_toolkit.py -v`
Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/langchain-lackpy/langchain_lackpy/toolkit.py packages/langchain-lackpy/tests/test_toolkit.py
git commit -m "feat(langchain): LackpyToolkit — primary entry point"
```

---

### Task 7: LangGraph node factory (_node.py)

**Files:**
- Create: `packages/langchain-lackpy/langchain_lackpy/_node.py`
- Create: `packages/langchain-lackpy/tests/test_node.py`

- [ ] **Step 1: Write the failing test**

Create `packages/langchain-lackpy/tests/test_node.py`:

```python
"""Tests for the LangGraph node factory."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_lackpy._node import make_node


class TestMakeNode:
    def test_returns_callable(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="intent",
            result_key="lackpy_result",
        )
        assert callable(node)

    @pytest.mark.asyncio
    async def test_reads_intent_from_state(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="intent",
            result_key="lackpy_result",
        )
        result = await node({"intent": "read config.yaml"})
        mock_service.delegate.assert_called_once()
        call_kwargs = mock_service.delegate.call_args[1]
        assert call_kwargs["intent"] == "read config.yaml"

    @pytest.mark.asyncio
    async def test_passes_kit_config(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file", "write_file"],
            intent_key="intent",
            result_key="result",
        )
        await node({"intent": "do something"})
        call_kwargs = mock_service.delegate.call_args[1]
        assert call_kwargs["kit"] == ["read_file", "write_file"]

    @pytest.mark.asyncio
    async def test_returns_result_under_correct_key(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="task",
            result_key="output",
        )
        result = await node({"task": "read test.txt"})
        assert "output" in result
        assert result["output"]["success"] is True

    @pytest.mark.asyncio
    async def test_custom_keys(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="my_intent",
            result_key="my_result",
        )
        result = await node({"my_intent": "hello"})
        assert "my_result" in result

    @pytest.mark.asyncio
    async def test_missing_intent_key_raises(self, mock_service):
        node = make_node(
            service=mock_service,
            kit_config=["read_file"],
            intent_key="intent",
            result_key="result",
        )
        with pytest.raises(KeyError, match="intent"):
            await node({"wrong_key": "hello"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_node.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_lackpy._node'`

- [ ] **Step 3: Implement _node.py**

Create `packages/langchain-lackpy/langchain_lackpy/_node.py`:

```python
"""LangGraph node factory for lackpy delegate."""

from __future__ import annotations

from typing import Any, Callable


def make_node(
    service: Any,
    kit_config: Any,
    intent_key: str = "intent",
    result_key: str = "lackpy_result",
) -> Callable[..., Any]:
    """Return an async function shaped for StateGraph.add_node().

    The returned function reads ``state[intent_key]``, calls
    ``service.delegate()``, and returns ``{result_key: result}``.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        intent = state[intent_key]
        result = await service.delegate(intent=intent, kit=kit_config)
        return {result_key: result}

    return node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_node.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/langchain-lackpy/langchain_lackpy/_node.py packages/langchain-lackpy/tests/test_node.py
git commit -m "feat(langchain): basic_lackpy_node via as_node() factory"
```

---

### Task 8: Package __init__.py and public API

**Files:**
- Modify: `packages/langchain-lackpy/langchain_lackpy/__init__.py`

- [ ] **Step 1: Write the final __init__.py**

Update `packages/langchain-lackpy/langchain_lackpy/__init__.py`:

```python
"""LangChain integration for lackpy."""

__version__ = "0.1.0"

from .toolkit import LackpyToolkit
from ._tool_wrapper import LackpyToolWrapper
from ._delegate import LackpyDelegateTool

__all__ = [
    "LackpyToolkit",
    "LackpyToolWrapper",
    "LackpyDelegateTool",
]
```

- [ ] **Step 2: Verify import works**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src python -c "from langchain_lackpy import LackpyToolkit, LackpyToolWrapper, LackpyDelegateTool; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run the full test suite**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/ -v`
Expected: all tests PASS (schema: 9, tool_wrapper: 9, delegate: 10, toolkit: 12, node: 6 = 46 total)

- [ ] **Step 4: Commit**

```bash
git add packages/langchain-lackpy/langchain_lackpy/__init__.py
git commit -m "feat(langchain): public API — export LackpyToolkit, LackpyToolWrapper, LackpyDelegateTool"
```

---

### Task 9: Integration smoke test

**Files:**
- Create: `packages/langchain-lackpy/tests/test_integration.py`

This test uses a real `LackpyService` with builtin tools (no inference provider needed — we use `_program_override` to skip generation).

- [ ] **Step 1: Write the integration test**

Create `packages/langchain-lackpy/tests/test_integration.py`:

```python
"""Integration test: real LackpyService through langchain tool interface."""

from __future__ import annotations

import pytest
from pathlib import Path

from langchain_lackpy import LackpyToolkit


@pytest.fixture
def real_service(tmp_path):
    from lackpy import LackpyService
    svc = LackpyService(workspace=tmp_path)
    test_file = tmp_path / "hello.txt"
    test_file.write_text("hello world")
    return svc


class TestEndToEnd:
    def test_get_tools_returns_builtin_tools(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file", "find_files"])
        tools = toolkit.get_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"read_file", "find_files"}

    def test_individual_tool_invocation(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        tools = toolkit.get_tools()
        read_tool = tools[0]
        test_file = real_service.workspace / "hello.txt"
        result = read_tool.invoke({"path": str(test_file)})
        assert "hello world" in result

    def test_as_delegate_returns_tool(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        delegate = toolkit.as_delegate()
        assert delegate.name == "lackpy_delegate"
        assert "read_file" in delegate.description
```

- [ ] **Step 2: Run integration tests**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/test_integration.py -v`
Expected: all 3 tests PASS

- [ ] **Step 3: Run the complete suite one final time**

Run: `cd packages/langchain-lackpy && PYTHONPATH=../../src pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add packages/langchain-lackpy/tests/test_integration.py
git commit -m "test(langchain): end-to-end integration with real LackpyService"
```

---

### Task 10: Verify lackpy core tests still pass

**Files:** None (verification only)

- [ ] **Step 1: Run lackpy's full test suite**

Run: `PYTHONPATH=src pytest tests/ -q --tb=line`
Expected: 473+ passed, 0 failures (the new `test_type_map.py` adds ~10 tests)

- [ ] **Step 2: Verify nothing was broken**

If any tests fail, investigate and fix before proceeding.

- [ ] **Step 3: Final commit if any fixes needed**

Only if Step 2 required changes.
