"""Tests for the delegate BaseTool wrapping LackpyService.delegate()."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

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
        assert "safe" in delegate_tool.description.lower() or "validated" in delegate_tool.description.lower()

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
