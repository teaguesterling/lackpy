"""Offline tests for scripts/eval/ste_usability/run_lackpy.py.

The device is never called: the woollama route goes to a local stub server
that answers like the OpenAI-compatible gateway.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "ste_usability"
sys.path.insert(0, str(HERE))
import run_lackpy  # noqa: E402

pytest.importorskip("woollama.core")

PROGRAM = 'from pathlib import Path\n\nprint(Path("notes.md").read_text().count("Status"))\n'


@pytest.fixture
def gateway(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            reply = f"```python\n{PROGRAM}```"
            out = json.dumps({"id": "x", "object": "chat.completion", "model": body["model"],
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(run_lackpy, "BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    yield seen
    srv.shutdown()


def test_parse_args():
    run_dir, label, route = run_lackpy.parse_args(["R", "q30", "tiiny-fast/Qwen/Qwen3-30B-A3B-Instruct"])
    assert (run_dir, label, route) == (Path("R"), "q30", "tiiny-fast/Qwen/Qwen3-30B-A3B-Instruct")
    assert run_lackpy.model_for(route) == "openai/tiiny-fast/Qwen/Qwen3-30B-A3B-Instruct"


@pytest.mark.parametrize("argv", [[], ["R", "L"], ["R", "L", "default"],
                                  ["R", "L", "openai/tiiny-fast/x"], ["R", "L", "r", "extra"]])
def test_parse_args_refuses(argv):
    with pytest.raises(SystemExit):
        run_lackpy.parse_args(argv)


def test_thinking_off_reaches_the_params(monkeypatch, tmp_path):
    monkeypatch.setenv("THINKING", "off")
    cfg = run_lackpy.make_config(tmp_path, "tiiny-fast/m")
    params = cfg.inference_providers["woollama"]["params"]
    assert params["chat_template_kwargs"] == {"enable_thinking": False}


def test_saves_the_generated_program(gateway, tmp_path):
    from lackpy.infer.prompt import build_system_prompt
    task = {"id": "patch-02", "goal": "Count the Status lines in notes.md."}
    run_lackpy.main([str(tmp_path), "lbl", "tiiny-fast/m"], tasks=[task])
    out = tmp_path / "lbl" / "r1"
    assert (out / "patch-02.py").read_text() == PROGRAM
    meta = json.loads((out / "patch-02.meta.json").read_text())
    assert meta["ok"] and meta["provider"] == "woollama" and meta["correction_strategy"] is None
    path, body = gateway[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "tiiny-fast/m"
    assert body["messages"][0] == {"role": "system", "content": build_system_prompt("")}
    assert body["messages"][1] == {"role": "user", "content": task["goal"]}


def test_failed_generation_writes_an_error_not_a_program(tmp_path):
    class _Broken:
        async def generate(self, goal, model=None):
            raise RuntimeError("All 1 providers failed")

    task = {"id": "def-01", "goal": "x"}
    run_lackpy.main([str(tmp_path), "lbl", "tiiny-fast/m"], tasks=[task],
                    service_factory=lambda ws: _Broken())
    out = tmp_path / "lbl" / "r1"
    assert not (out / "def-01.py").exists()
    assert "providers failed" in (out / "def-01.error.txt").read_text()
    assert json.loads((out / "def-01.meta.json").read_text())["ok"] is False
