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
        test_file = real_service._workspace / "hello.txt"
        result = read_tool.invoke({"path": str(test_file)})
        assert "hello world" in result

    def test_as_delegate_returns_tool(self, real_service):
        toolkit = LackpyToolkit(service=real_service, kit=["read_file"])
        delegate = toolkit.as_delegate()
        assert delegate.name == "lackpy_delegate"
        assert "read_file" in delegate.description
