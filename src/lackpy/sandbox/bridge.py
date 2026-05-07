"""Tool bridge: multiprocessing.managers-based IPC for bridged tools."""

from __future__ import annotations

import os
import tempfile
import threading
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, Callable


class _ToolDispatcher:
    """Server-side dispatcher that holds tool callables."""

    def __init__(self, callables: dict[str, Callable[..., Any]]) -> None:
        self._callables = callables

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._callables:
            raise KeyError(f"No bridged tool named '{name}'")
        return self._callables[name](*args, **kwargs)

    def list_tools(self) -> list[str]:
        return list(self._callables.keys())


class _BridgeManager(BaseManager):
    pass


class ToolBridgeManager:
    """Manages a multiprocessing.managers server for bridged tool calls.

    Created per-invocation, torn down after sandbox exits.
    """

    def __init__(
        self,
        callables: dict[str, Callable[..., Any]],
        socket_dir: Path | None = None,
    ) -> None:
        self._callables = callables
        self._owns_socket_dir = socket_dir is None
        self._socket_dir = socket_dir or Path(tempfile.mkdtemp(prefix="lackpy_bridge_"))
        self._socket_path = self._socket_dir / "bridge.sock"
        self._authkey = os.urandom(32)
        self._dispatcher = _ToolDispatcher(callables)
        self._server: Any | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def authkey(self) -> bytes:
        return self._authkey

    def start(self) -> None:
        dispatcher = self._dispatcher

        class ServerManager(_BridgeManager):
            pass

        ServerManager.register("get_dispatcher", callable=lambda: dispatcher)
        address = str(self._socket_path)
        mgr = ServerManager(address=address, authkey=self._authkey)
        # get_server() binds the socket synchronously — file exists after this call
        self._server = mgr.get_server()
        server = self._server

        def _serve() -> None:
            try:
                server.serve_forever()
            except SystemExit:
                # serve_forever() calls sys.exit(0) after stop_event is set;
                # swallow it so the daemon thread exits cleanly without noise.
                pass

        self._server_thread = threading.Thread(target=_serve, daemon=True)
        self._server_thread.start()
        # Brief yield so serve_forever() can set stop_event before we return
        self._server_thread.join(timeout=0)

    def stop(self) -> None:
        if self._server is not None:
            # serve_forever() sets stop_event at startup; poll briefly for it
            import time as _time
            for _ in range(20):
                if hasattr(self._server, "stop_event"):
                    break
                _time.sleep(0.01)
            stop_event = getattr(self._server, "stop_event", None)
            if stop_event is not None:
                stop_event.set()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
        if self._socket_path.exists():
            self._socket_path.unlink()
        if self._owns_socket_dir and self._socket_dir.exists():
            try:
                self._socket_dir.rmdir()
            except OSError:
                pass

    def __enter__(self) -> ToolBridgeManager:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class _BridgeClient:
    """Client-side proxy that connects to the bridge manager."""

    def __init__(self, dispatcher_proxy: Any) -> None:
        self._dispatcher = dispatcher_proxy

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self._dispatcher.call(name, *args, **kwargs)

    def list_tools(self) -> list[str]:
        return self._dispatcher.list_tools()


def bridge_client(socket_path: Path, authkey: bytes) -> _BridgeClient:
    """Connect to a running ToolBridgeManager and return a client."""

    class ClientManager(_BridgeManager):
        pass

    ClientManager.register("get_dispatcher")
    mgr = ClientManager(address=str(socket_path), authkey=authkey)
    mgr.connect()
    dispatcher = mgr.get_dispatcher()
    return _BridgeClient(dispatcher)
