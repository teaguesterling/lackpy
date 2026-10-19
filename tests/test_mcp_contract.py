"""RFC 0002 §11 — the MCP surface as an orchestrator contract.

Actions are annotated tools; read-only data is exposed as resources, not tools; the
unwired `sandbox` param is gone.

stepy100 Plan 2, Task 7: no kit in the agent's view. No tool takes `extra_tools`,
`profile` is a profile name only, and the tool-set tools and resources are gone
(`profile_create`, `profile_info`, `docs_index`, `resolve_doc`, `lackpy://toolbox`,
`lackpy://profiles`).
"""
import asyncio
import inspect
import typing

import pytest

pytest.importorskip("mcp", reason="mcp required for MCP server tests")

from lackpy.mcp import server

ACTIONS = {"delegate", "generate", "run_program", "validate", "create"}


def _tools():
    return asyncio.run(server.mcp.list_tools())


def _resources():
    return {str(r.uri) for r in asyncio.run(server.mcp.list_resources())}


def test_data_getters_are_resources_not_tools():
    tool_names = {t.name for t in _tools()}
    res = _resources()
    for name, uri in [("config", "lackpy://config"), ("language_spec", "lackpy://language-spec"),
                      ("provider_list", "lackpy://providers")]:
        assert uri in res, f"{uri} should be a resource"
        assert name not in tool_names, f"{name} should not be a tool"


def test_tools_are_actions_only():
    # Retired kit model: profile_create, profile_info, docs_index and resolve_doc were tools.
    assert {t.name for t in _tools()} == ACTIONS


def test_no_tool_set_resources():
    # Retired kit model: lackpy://toolbox and lackpy://profiles listed tools and kit files.
    assert _resources() == {"lackpy://config", "lackpy://providers", "lackpy://language-spec"}


def test_no_tool_takes_extra_tools():
    for t in _tools():
        assert "extra_tools" not in t.inputSchema.get("properties", {}), t.name


def test_profile_is_a_name_only():
    for name in ACTIONS:
        fn = getattr(server, name)
        hints = typing.get_type_hints(fn)
        if "profile" in inspect.signature(fn).parameters:
            assert hints["profile"] == (str | None), name


def test_descriptions_do_not_offer_a_tool_set():
    for t in _tools():
        text = (t.description or "").lower()
        for phrase in ("tool set", "tool names", "list of tool", "alias", "extra_tools", "kit"):
            assert phrase not in text, f"{t.name}: {phrase!r}"


def test_run_program_description_names_pathlib():
    d = {t.name: t for t in _tools()}
    assert "pathlib.Path" in d["run_program"].description
    assert "pathlib.Path" in d["delegate"].description


def test_executing_tools_are_marked_destructive_others_readonly():
    d = {t.name: t for t in _tools()}
    assert d["delegate"].annotations.readOnlyHint is False
    assert d["delegate"].annotations.destructiveHint is True
    assert d["run_program"].annotations.destructiveHint is True
    assert d["validate"].annotations.readOnlyHint is True
    assert d["generate"].annotations.readOnlyHint is True


def test_delegate_result_has_no_grade(monkeypatch):
    # The grade was the join of the kit's tool grades; the lean result drops it.
    class _Svc:
        async def delegate(self, intent, **kw):
            return {"success": True, "output": 1, "grade": {"w": 1, "d": 1}, "trace": []}

    monkeypatch.setattr(server, "_service", _Svc())
    result = asyncio.run(server.delegate("x"))
    assert "grade" not in result
    assert result["success"] is True


def test_unwired_sandbox_param_removed():
    for fn in (server.delegate, server.run_program):
        assert "sandbox" not in inspect.signature(fn).parameters


def test_resource_functions_still_directly_callable(tmp_path):
    # Tests/back-compat call these as plain functions; FastMCP keeps them callable.
    server.set_workspace(tmp_path)
    assert isinstance(server.config(), dict)
    assert isinstance(server.provider_list(), list)
    server._service = None


def test_run_program_returns_the_programs_stdout(tmp_path):
    # Ruling (Task 7 fix round, I1): a heredoc replacement that prints nothing is not one.
    server.set_workspace(tmp_path)
    try:
        result = asyncio.run(server.run_program("print('hi')\nprint('there')"))
        assert result["success"], result["error"]
        assert result["stdout"] == "hi\nthere\n"
    finally:
        server._service = None


def test_delegate_returns_the_programs_stdout(monkeypatch):
    class _Svc:
        async def delegate(self, intent, **kw):
            return {"success": True, "output": None, "stdout": "hi\n", "trace": []}

    monkeypatch.setattr(server, "_service", _Svc())
    assert asyncio.run(server.delegate("x"))["stdout"] == "hi\n"


def test_configured_literate_profile_runs(tmp_path):
    # Literate stepy Task 4: the axis runs (review C1 refused it with
    # LITERATE_PENDING). The escape document is gate 1 in tests/literate/test_gate.py.
    from lackpy.run.paths import load_policy
    from lackpy.service import LackpyService
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    ws = tmp_path / "ws"
    (ws / ".lackpy").mkdir(parents=True)
    (ws / ".lackpy" / "config.toml").write_text('[profiles.lit]\nexecution = "literate"\n')
    server._service = LackpyService(
        workspace=ws, path_policy_loader=lambda w: load_policy(w, home=home, managed=None))
    try:
        doc = "```lackpy\nx = 6 * 7\nx\n```\n\nX={x}"
        result = asyncio.run(server.run_program(doc, profile="lit"))
        assert result["success"], result["error"]
        assert "X=42" in result["stdout"]
        assert result["output"] == 42
    finally:
        server._service = None
