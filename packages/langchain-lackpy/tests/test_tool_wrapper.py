"""Tests for individual ToolSpec → BaseTool wrapping."""

from __future__ import annotations

import pytest
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
        tool._run(path="/tmp/test.txt")
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
