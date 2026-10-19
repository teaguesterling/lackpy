"""Tests for MCP server tools (config, provider_list, language_spec)."""

import pytest
from pathlib import Path

pytest.importorskip("mcp", reason="mcp required for MCP server tests")

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


def test_run_program_output_is_json_serializable(workspace):
    # Parked item 2: a non-data last value (a Path stand-in, a module proxy, a type
    # view) reaches the host as text, never as a live object.
    import asyncio
    import json
    # (The validator rejects a bare type(3) or json; the runner tests cover those.)
    for program in ["from pathlib import Path\nPath('a')", "from collections import Counter\nCounter('ab')"]:
        result = asyncio.run(server.run_program(program))
        assert result["success"], result["error"]
        assert isinstance(result["output"], str)
        json.dumps(result)
    assert asyncio.run(server.run_program("from pathlib import Path\nPath('a')"))["output"] == repr(Path("a"))


def test_run_program_takes_a_plain_program_without_a_profile(workspace):
    import asyncio
    (workspace / "t.txt").write_text("abc")
    result = asyncio.run(server.run_program("from pathlib import Path\nlen(Path('t.txt').read_text())"))
    assert result["success"], result["error"]
    assert result["output"] == 3
    assert [s["tool"] for s in result["trace"]] == ["Path.read_text"]
