"""Host MCP config ingestion ([mcp].host_configs) — RFC 0002 §3.4, 3b."""

import json
import sys
from pathlib import Path

import pytest

from lackpy.sources.mcp import mcp_available

pytestmark = pytest.mark.skipif(not mcp_available(), reason="mcp SDK not installed")

FIXTURE = str(Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")


def _host_json(tmp_path: Path) -> Path:
    cfg = {"mcpServers": {"echo": {"command": sys.executable, "args": [FIXTURE]}}}
    p = tmp_path / "host.json"
    p.write_text(json.dumps(cfg))
    return p


def test_load_host_servers_normalizes_mcpservers(tmp_path):
    from lackpy.sources.mcp.host_config import load_host_servers

    specs = load_host_servers(str(_host_json(tmp_path)))
    assert len(specs) == 1
    assert specs[0].server_id == "echo"
    assert specs[0].transport == "stdio"
    assert specs[0].command == sys.executable
    assert specs[0].args == [FIXTURE]


def test_load_host_servers_http_entry(tmp_path):
    from lackpy.sources.mcp.host_config import load_host_servers

    p = tmp_path / "h.json"
    p.write_text(json.dumps({"mcpServers": {"remote": {"url": "http://localhost:9/mcp"}}}))
    specs = load_host_servers(str(p))
    assert specs[0].transport == "http"
    assert specs[0].url == "http://localhost:9/mcp"


async def test_service_ingests_host_config_end_to_end(tmp_path):
    from lackpy.config import LackpyConfig
    from lackpy.service import LackpyService

    cfg = LackpyConfig(mcp_host_configs=[str(_host_json(tmp_path))])
    svc = LackpyService(workspace=tmp_path, config=cfg)
    try:
        names = {t["name"] for t in svc.toolbox_list()}
        assert {"echo", "add"} <= names
        res = await svc.run_program("r = echo(text='hi')\nr", kit=["echo"])
        assert res.success, res.error
        assert res.output == "hi"
    finally:
        svc.close()


async def test_malformed_host_config_is_skipped_not_fatal(tmp_path):
    from lackpy.config import LackpyConfig
    from lackpy.service import LackpyService

    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json")
    # service init must not raise; builtins still present.
    svc = LackpyService(workspace=tmp_path, config=LackpyConfig(mcp_host_configs=[str(bad)]))
    try:
        assert "read_file" in {t["name"] for t in svc.toolbox_list()}
    finally:
        svc.close()
