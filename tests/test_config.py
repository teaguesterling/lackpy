"""Tests for configuration loading."""

import pytest
from pathlib import Path

from lackpy.config import load_config, LackpyConfig


@pytest.fixture
def config_dir(tmp_path):
    config_file = tmp_path / ".lackpy" / "config.toml"
    config_file.parent.mkdir()
    config_file.write_text('''
[inference]
order = ["templates", "rules", "ollama-local"]

[inference.providers.ollama-local]
plugin = "ollama"
host = "http://localhost:11434"
model = "qwen2.5-coder:1.5b"

[kit]
default = "debug"

[sandbox]
enabled = false
timeout_seconds = 60
''')
    return tmp_path


def test_load_config(config_dir):
    cfg = load_config(config_dir)
    assert cfg.inference_order == ["templates", "rules", "ollama-local"]
    assert cfg.kit_default == "debug"
    assert cfg.sandbox_enabled is False


def test_load_config_defaults(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.inference_order is not None
    assert cfg.kit_default == "debug"


def test_provider_config(config_dir):
    cfg = load_config(config_dir)
    ollama_cfg = cfg.inference_providers.get("ollama-local")
    assert ollama_cfg is not None
    assert ollama_cfg["host"] == "http://localhost:11434"


def test_load_config_parses_source_sections(tmp_path):
    # load_config must read the tool-source sections (not just LackpyConfig(...)).
    cfg_file = tmp_path / ".lackpy" / "config.toml"
    cfg_file.parent.mkdir()
    cfg_file.write_text('''
[[tools]]
name = "count_lines"
provider = "python"
module = "my_tools"
function = "count_lines"

[mcp_servers.fs]
transport = "stdio"
command = "mcp-server-filesystem"
args = ["--root", "."]

[mcp]
host_configs = ["~/.cursor/mcp.json", "/etc/lackpy/host.json"]

[[virtual_tools]]
name = "notify"
description = "harness notification"
returns = "bool"
''')
    cfg = load_config(tmp_path)
    assert [t["name"] for t in cfg.tools] == ["count_lines"]
    assert cfg.mcp_servers["fs"]["command"] == "mcp-server-filesystem"
    assert cfg.mcp_host_configs == ["~/.cursor/mcp.json", "/etc/lackpy/host.json"]
    assert [v["name"] for v in cfg.virtual_tools] == ["notify"]


def test_source_sections_default_empty(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.tools == []
    assert cfg.mcp_servers == {}
    assert cfg.mcp_host_configs == []
    assert cfg.virtual_tools == []
