"""Tests for MCP server tools (config, provider_list, language_spec)."""

import pytest
from pathlib import Path

from lackpy.mcp import server


@pytest.fixture(autouse=True)
def reset_service():
    """Reset the module-level service singleton between tests."""
    server._service = None
    server._workspace = None
    yield
    server._service = None
    server._workspace = None


@pytest.fixture
def workspace(tmp_path):
    server.set_workspace(tmp_path)
    return tmp_path


def test_set_workspace(tmp_path):
    server.set_workspace(tmp_path)
    assert server._workspace == tmp_path


def test_get_service_uses_workspace(workspace):
    svc = server._get_service()
    assert svc._workspace == workspace


def test_config_tool(workspace):
    result = server.config()
    assert isinstance(result, dict)
    assert "inference_order" in result
    assert "config_dir" in result


def test_provider_list_tool(workspace):
    result = server.provider_list()
    assert isinstance(result, list)
    assert len(result) >= 2  # templates + rules at minimum


def test_language_spec_tool(workspace):
    result = server.language_spec()
    assert isinstance(result, dict)
    assert "allowed_nodes" in result
    assert "allowed_builtins" in result
