"""Dedicated-thread MCP client.

All MCP I/O runs on one background thread + event loop owned by this client, so
sessions open once and are reused across delegate calls and CLI invocations, and
the bridge has a stable loop to marshal onto (RFC 0002 §3, §5; loop-ownership
decision: dedicated client loop thread).

The ``mcp`` SDK is imported lazily so importing this module never requires the
optional dependency.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future as CFuture
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ...run.bridge import AsyncBridge


@dataclass
class McpServerSpec:
    """Connection spec for one MCP server."""

    server_id: str
    transport: str = "stdio"  # "stdio" | "http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None


class McpClient:
    """Owns a background thread + event loop for all MCP I/O.

    Each server gets a long-lived *manager task* that enters and exits its
    session context in the SAME task (avoiding anyio's cross-task cancel-scope
    error), keeping the session alive until shutdown.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._shutdown_event: asyncio.Event | None = None
        self._serve_tasks: list[asyncio.Task] = []
        self._sessions: dict[str, Any] = {}
        self._started = False
        self.bridge = AsyncBridge()

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(target=self._thread_main, name="lackpy-mcp", daemon=True)
        self._thread.start()
        self._ready.wait()
        self._started = True

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._shutdown_event = asyncio.Event()
        self.bridge.loop = self._loop
        self._ready.set()
        self._loop.run_forever()
        # After stop(): drain anything still pending, then close.
        pending = asyncio.all_tasks(self._loop)
        for t in pending:
            t.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    def shutdown(self) -> None:
        if not self._started or self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._drain(), self._loop)
            fut.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._started = False

    async def _drain(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._serve_tasks:
            await asyncio.gather(*self._serve_tasks, return_exceptions=True)

    # ---------- connect / discover ----------

    def connect(self, spec: McpServerSpec, timeout: float = 30.0) -> list[Any]:
        """Open a session to ``spec`` and return its discovered tools.

        Blocking; safe to call from the service's (sync) init on any thread.
        """
        if not self._started:
            self.start()
        ready: CFuture = CFuture()

        def _schedule() -> None:
            task = self._loop.create_task(self._serve(spec, ready))  # type: ignore[union-attr]
            self._serve_tasks.append(task)

        self._loop.call_soon_threadsafe(_schedule)  # type: ignore[union-attr]
        return ready.result(timeout)

    async def _serve(self, spec: McpServerSpec, ready: CFuture) -> None:
        from mcp import ClientSession

        try:
            async with self._open_transport(spec) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._sessions[spec.server_id] = session
                    tools = (await session.list_tools()).tools
                    ready.set_result(tools)
                    await self._shutdown_event.wait()  # hold the context open in THIS task
        except Exception as e:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(e)
        finally:
            self._sessions.pop(spec.server_id, None)

    @asynccontextmanager
    async def _open_transport(self, spec: McpServerSpec):
        if spec.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=spec.command, args=spec.args, env=spec.env, cwd=spec.cwd,
            )
            async with stdio_client(params) as (read, write):
                yield read, write
        elif spec.transport == "http":
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(spec.url, headers=spec.headers) as (read, write, _):
                yield read, write
        else:
            raise ValueError(f"Unknown MCP transport: {spec.transport!r}")

    # ---------- call ----------

    def call(self, server_id: str, name: str, arguments: dict[str, Any] | None,
             timeout: float | None = None) -> Any:
        """Invoke an MCP tool and return its unwrapped result (sync; for proxies)."""
        return self.bridge.call_sync(self._call(server_id, name, arguments), timeout)

    async def _call(self, server_id: str, name: str, arguments: dict[str, Any] | None) -> Any:
        session = self._sessions.get(server_id)
        if session is None:
            raise RuntimeError(f"MCP server {server_id!r} is not connected")
        result = await session.call_tool(name, arguments or {})
        if result.isError:
            raise RuntimeError(_format_error(result))
        return _unwrap(result)


def _unwrap(result: Any) -> Any:
    """Normalize a CallToolResult to a plain Python value.

    structured output → its value (unwrapping fastmcp's ``{"result": ...}``);
    a single text block → its string; multiple → list of strings.
    """
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        if set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    blocks = result.content or []
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
    if len(texts) == 1:
        return texts[0]
    if texts:
        return texts
    return None


def _format_error(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
    return "; ".join(texts) or "MCP tool returned an error"
