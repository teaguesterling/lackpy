"""RFC 0002 §11 — the MCP surface as an orchestrator contract.

Actions are annotated tools; read-only data is exposed as resources, not tools; the
unwired `sandbox` param is gone.
"""
import asyncio
import inspect

import pytest

pytest.importorskip("mcp", reason="mcp required for MCP server tests")

from lackpy.mcp import server


def _tools():
    return asyncio.run(server.mcp.list_tools())


def _resources():
    return {str(r.uri) for r in asyncio.run(server.mcp.list_resources())}


def test_data_getters_are_resources_not_tools():
    tool_names = {t.name for t in _tools()}
    res = _resources()
    for name, uri in [("config", "lackpy://config"), ("language_spec", "lackpy://language-spec"),
                      ("toolbox_list", "lackpy://toolbox"), ("provider_list", "lackpy://providers"),
                      ("profile_list", "lackpy://profiles")]:
        assert uri in res, f"{uri} should be a resource"
        assert name not in tool_names, f"{name} should not be a tool"


def test_tools_are_actions_only():
    names = {t.name for t in _tools()}
    assert {"delegate", "generate", "run_program", "validate", "create",
            "profile_create"} <= names
    assert len(names) <= 10  # was 14 flat tools


def test_executing_tools_are_marked_destructive_others_readonly():
    d = {t.name: t for t in _tools()}
    assert d["delegate"].annotations.readOnlyHint is False
    assert d["delegate"].annotations.destructiveHint is True
    assert d["run_program"].annotations.destructiveHint is True
    assert d["validate"].annotations.readOnlyHint is True
    assert d["generate"].annotations.readOnlyHint is True


def test_unwired_sandbox_param_removed():
    for fn in (server.delegate, server.run_program):
        assert "sandbox" not in inspect.signature(fn).parameters


def test_resource_functions_still_directly_callable(tmp_path):
    # Tests/back-compat call these as plain functions; FastMCP keeps them callable.
    server.set_workspace(tmp_path)
    assert isinstance(server.config(), dict)
    assert isinstance(server.toolbox_list(), list)
    server._service = None
