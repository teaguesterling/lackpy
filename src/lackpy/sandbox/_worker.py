"""Subprocess worker harness -- runs inside the nsjail sandbox.

The parent process writes a request to <io_dir>/request.json,
launches this module inside the jail, and reads the result
from <io_dir>/result.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def write_request(io_dir: Path, request: dict[str, Any]) -> None:
    (io_dir / "request.json").write_text(json.dumps(request))


def read_request(io_dir: Path) -> dict[str, Any]:
    path = io_dir / "request.json"
    if not path.exists():
        raise FileNotFoundError(f"No request file at {path}")
    return json.loads(path.read_text())


def write_result(io_dir: Path, result: dict[str, Any]) -> None:
    (io_dir / "result.json").write_text(json.dumps(result))


def read_result(io_dir: Path) -> dict[str, Any]:
    path = io_dir / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"No result file at {path}")
    return json.loads(path.read_text())


def build_tool_namespace(
    embedded_sources: dict[str, str],
    bridge_client: Any | None,
) -> dict[str, Any]:
    """Build a namespace of callable tools from embedded sources and bridge proxy."""
    namespace: dict[str, Any] = {}

    for name, source in embedded_sources.items():
        local_ns: dict[str, Any] = {}
        compiled = compile(source, f"<embedded:{name}>", "exec")
        exec(compiled, local_ns)  # noqa: S102
        for k, v in local_ns.items():
            if callable(v) and not k.startswith("_"):
                namespace[name] = v
                break

    if bridge_client is not None:
        tool_names = bridge_client.list_tools()
        for tool_name in tool_names:
            def _make_proxy(tn: str) -> Any:
                def proxy(*args: Any, **kwargs: Any) -> Any:
                    return bridge_client.call(tn, *args, **kwargs)
                return proxy
            namespace[tool_name] = _make_proxy(tool_name)

    return namespace


def main() -> None:
    """Entry point when run as `python -m lackpy.sandbox._worker <io_dir>`."""
    io_dir = Path(sys.argv[1])
    request = read_request(io_dir)

    program = request["program"]
    embedded_sources = request.get("embedded_sources", {})
    bridge_socket = request.get("bridge_socket")
    base_dir = request.get("base_dir", ".")

    client = None
    if bridge_socket:
        from lackpy.sandbox.bridge import bridge_client as _bridge_client
        client = _bridge_client(Path(bridge_socket))

    tool_ns = build_tool_namespace(embedded_sources, client)

    import os
    os.chdir(base_dir)

    start = time.perf_counter()
    try:
        local_ns = dict(tool_ns)
        compiled_code = compile(program, "<sandbox>", "exec")
        exec(compiled_code, {"__builtins__": {}}, local_ns)  # noqa: S102
        output = local_ns.get("__result__")
        duration_ms = (time.perf_counter() - start) * 1000
        write_result(io_dir, {
            "success": True,
            "output": str(output) if output is not None else None,
            "output_format": "text" if output is not None else "none",
            "error": None,
            "duration_ms": duration_ms,
            "metadata": {},
        })
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        write_result(io_dir, {
            "success": False,
            "output": None,
            "output_format": "none",
            "error": f"{type(e).__name__}: {e}",
            "duration_ms": duration_ms,
            "metadata": {},
        })


if __name__ == "__main__":
    main()
