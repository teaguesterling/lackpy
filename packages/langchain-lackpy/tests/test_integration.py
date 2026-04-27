"""Integration test: real LackpyService through langchain tool interface."""

from __future__ import annotations

import pytest
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_lackpy import LackpyToolkit, LackpyToolWrapper, LackpyDelegateTool


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
        test_file = real_service._workspace / "hello.txt"
        result = read_tool.invoke({"path": str(test_file)})
        assert "hello world" in result

    def test_as_delegate_returns_tool(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        delegate = toolkit.as_delegate()
        assert delegate.name == "lackpy_delegate"
        assert "read_file" in delegate.description


class TestDictKit:
    def test_dict_kit_renames_tools(self, real_service):
        toolkit = LackpyToolkit(
            service=real_service,
            kit={"reader": "read_file", "finder": "find_files"},
        )
        tools = toolkit.get_tools()
        names = {t.name for t in tools}
        assert names == {"reader", "finder"}

    def test_dict_kit_tools_still_work(self, real_service):
        toolkit = LackpyToolkit(
            service=real_service,
            kit={"reader": "read_file"},
        )
        tools = toolkit.get_tools()
        reader = tools[0]
        test_file = real_service._workspace / "hello.txt"
        result = reader.invoke({"path": str(test_file)})
        assert "hello world" in result

    def test_dict_kit_delegate_description_mentions_underlying_tool(self, real_service):
        toolkit = LackpyToolkit(
            service=real_service,
            kit={"reader": "read_file"},
        )
        delegate = toolkit.as_delegate()
        assert "read_file" in delegate.description


class TestToolMetadata:
    def test_tool_carries_provider(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        tool = toolkit.get_tools()[0]
        assert "provider" in tool.metadata
        assert tool.metadata["provider"] == "builtin"

    def test_tool_carries_grade(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        tool = toolkit.get_tools()[0]
        assert "grade_w" in tool.metadata
        assert isinstance(tool.metadata["grade_w"], (int, float))

    def test_tool_carries_effects_ceiling(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        tool = toolkit.get_tools()[0]
        assert "effects_ceiling" in tool.metadata

    def test_tool_has_pydantic_args_schema(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        tool = toolkit.get_tools()[0]
        assert issubclass(tool.args_schema, BaseModel)
        field_names = set(tool.args_schema.model_fields.keys())
        assert "path" in field_names

    def test_filter_tools_by_grade(self, real_service):
        toolkit = LackpyToolkit(
            service=real_service,
            kit=["read_file", "find_files", "write_file"],
        )
        tools = toolkit.get_tools()
        safe_tools = [t for t in tools if t.metadata["grade_w"] <= 1]
        unsafe_names = {t.name for t in tools} - {t.name for t in safe_tools}
        for name in unsafe_names:
            tool = next(t for t in tools if t.name == name)
            assert tool.metadata["grade_w"] > 1


class TestTypeInheritance:
    def test_wrapper_is_base_tool(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        tool = toolkit.get_tools()[0]
        assert isinstance(tool, BaseTool)
        assert isinstance(tool, LackpyToolWrapper)

    def test_delegate_is_base_tool(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        delegate = toolkit.as_delegate()
        assert isinstance(delegate, BaseTool)
        assert isinstance(delegate, LackpyDelegateTool)

    def test_node_is_callable(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        node = toolkit.as_node()
        assert callable(node)


class TestCustomDelegateName:
    def test_multiple_delegates_have_distinct_names(self, real_service):
        file_tk = LackpyToolkit(service=real_service, kit=["read_file", "find_files"])
        write_tk = LackpyToolkit(service=real_service, kit=["write_file"])
        file_delegate = file_tk.as_delegate(name="lackpy_files")
        write_delegate = write_tk.as_delegate(name="lackpy_writer")
        assert file_delegate.name == "lackpy_files"
        assert write_delegate.name == "lackpy_writer"
        assert file_delegate.name != write_delegate.name
