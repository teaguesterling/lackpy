"""Tests for lackpyctl mcp CLI (init subcommand)."""

import json
import pytest
from pathlib import Path

from lackpy.mcp.cli import mcp_init


class TestMcpInit:
    def test_creates_mcp_json_when_absent(self, tmp_path):
        result = mcp_init(workspace=tmp_path)
        assert result == 0
        mcp_file = tmp_path / ".mcp.json"
        assert mcp_file.exists()
        data = json.loads(mcp_file.read_text())
        assert "lackpy" in data["mcpServers"]
        assert data["mcpServers"]["lackpy"]["command"] == "lackpy-mcp"

    def test_includes_workspace_in_args(self, tmp_path):
        mcp_init(workspace=tmp_path)
        data = json.loads((tmp_path / ".mcp.json").read_text())
        args = data["mcpServers"]["lackpy"]["args"]
        assert "--workspace" in args
        assert str(tmp_path) in args

    def test_preserves_existing_entries(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "other-tool": {"command": "other", "args": []}
            }
        }))
        mcp_init(workspace=tmp_path)
        data = json.loads(mcp_file.read_text())
        assert "other-tool" in data["mcpServers"]
        assert "lackpy" in data["mcpServers"]

    def test_refuses_overwrite_without_force(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "lackpy": {"command": "old", "args": []}
            }
        }))
        result = mcp_init(workspace=tmp_path, force=False)
        assert result == 1
        # Original entry unchanged
        data = json.loads(mcp_file.read_text())
        assert data["mcpServers"]["lackpy"]["command"] == "old"

    def test_overwrites_with_force(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "lackpy": {"command": "old", "args": []}
            }
        }))
        result = mcp_init(workspace=tmp_path, force=True)
        assert result == 0
        data = json.loads(mcp_file.read_text())
        assert data["mcpServers"]["lackpy"]["command"] == "lackpy-mcp"

    def test_custom_name(self, tmp_path):
        mcp_init(workspace=tmp_path, name="my-lackpy")
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "my-lackpy" in data["mcpServers"]
        assert "lackpy" not in data["mcpServers"]

    def test_rejects_invalid_json(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text("not valid json {{{")
        result = mcp_init(workspace=tmp_path)
        assert result == 1

    def test_rejects_non_object_json(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text("[]")
        result = mcp_init(workspace=tmp_path)
        assert result == 1


class TestMcpEntryPoints:
    """The external-consumer launch surface (woollama spawns lackpy's MCP server)."""

    def test_lackpy_mcp_routes_to_server(self):
        # `lackpy mcp --help` must reach mcp_main (exit 0), not the runner's parser
        # (which would reject 'mcp' as unrecognized → exit 2).
        import pytest
        from lackpy.cli import main
        with pytest.raises(SystemExit) as e:
            main(["mcp", "--help"])
        assert e.value.code == 0

    def test_mcp_main_parses_workspace(self, monkeypatch):
        from lackpy.mcp import cli as mcpcli
        seen = {}

        def fake_serve(ws):
            seen["ws"] = ws
            return 0

        monkeypatch.setattr(mcpcli, "mcp_serve", fake_serve)
        rc = mcpcli.mcp_main(["--workspace", "/tmp/ws"])
        assert rc == 0 and str(seen["ws"]) == "/tmp/ws"

    def test_generated_config_uses_dedicated_entrypoint(self, tmp_path):
        from lackpy.mcp.cli import mcp_init
        mcp_init(tmp_path)
        import json
        data = json.loads((tmp_path / ".mcp.json").read_text())
        entry = data["mcpServers"]["lackpy"]
        assert entry["command"] == "lackpy-mcp"        # decoupled, spawnable as-is
        assert "serve" not in entry["args"]            # no subcommand needed
