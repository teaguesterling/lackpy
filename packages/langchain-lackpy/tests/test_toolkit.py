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
