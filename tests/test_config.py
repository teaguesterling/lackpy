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
