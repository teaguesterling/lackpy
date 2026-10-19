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


# --- Final review I2, I3, I4, M1, M9 and Rec 5: results through FastMCP's serializer ---

import json  # noqa: E402


def _strict_json(text):
    def no_constant(name):
        raise ValueError(f"non-standard JSON constant {name}")
    return json.loads(text, parse_constant=no_constant)


def _call(name, args):
    """Call a tool through FastMCP, as a client does, and parse its JSON strictly."""
    import asyncio
    result = asyncio.run(server.mcp.call_tool(name, args))
    content = result[0] if isinstance(result, tuple) else result
    return _strict_json(content[0].text)



def test_hash_of_a_binary_file_round_trips(workspace):
    # I2, the reviewer's reproduction: the read_bytes trace entry held raw bytes.
    (workspace / "img.bin").write_bytes(bytes(range(256)) * 4)
    program = "import hashlib\nfrom pathlib import Path\nhashlib.sha256(Path('img.bin').read_bytes()).hexdigest()"
    out = _call("run_program", {"program": program})
    assert out["success"], out["error"]
    assert len(out["output"]) == 64
    (entry,) = out["trace"]
    assert entry["tool"] == "Path.read_bytes"
    assert entry["result"].endswith("... (1024 bytes)")


def test_edge_values_round_trip(workspace):
    # Rec 5: bytes, nan, inf, tuple keys and nested containers.
    cases = {
        'b"\\xff"': repr(b"\xff"),
        'float("nan")': "nan",
        'float("inf")': "inf",
        '-float("inf")': "-inf",
        '{(1, 2): b"a"}': {"(1, 2)": "b'a'"},
        '[(1, b"ok"), {"a": {None: [float("nan")]}}, {3}]': [[1, "b'ok'"], {"a": {"None": ["nan"]}}, [3]],
        '{1: "a", True: 2.5}': {"1": 2.5},
    }
    for program, expected in cases.items():
        out = _call("run_program", {"program": program})
        assert out["success"], (program, out["error"])
        assert out["output"] == expected, program


def test_validate_reports_hints(workspace):
    # I3: MCP validate carries each violation's hint.
    out = _call("validate", {"program": 'open("a.txt").read()'})
    assert out["valid"] is False
    hinted = [v for v in out["violations"] if v["hint"]]
    assert hinted and "Path(p).read_text()" in hinted[0]["hint"]
    assert [v["message"] for v in out["violations"]] == out["errors"]


def test_string_rules_resolve(workspace):
    # M1: rules is list[str] in the schema; a string raised TypeError.
    out = _call("validate", {"program": "for i in []:\n    pass", "rules": ["no_loops"]})
    assert out["valid"] is False
    out = _call("run_program", {"program": "x = 1", "rules": ["no_such_rule"]})
    assert out["success"] is False
    assert "no_such_rule" in out["error"]


def test_a_refusal_while_the_policy_loads_is_a_failed_result(workspace):
    # I4: the refusal escaped FastMCP's tool error handling as a BaseException.
    (workspace / "loop").symlink_to(workspace / "loop")
    (workspace / ".claude").mkdir()
    (workspace / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"deny": ["Read(./loop/secret)"]}}))
    out = _call("run_program", {"program": "x = 1\nx"})
    assert out["success"] is False
    assert "Read(./loop/secret)" in out["error"]
