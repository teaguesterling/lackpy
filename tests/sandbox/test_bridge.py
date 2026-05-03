"""Tests for ToolBridgeManager lifecycle."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestToolBridgeManager:
    def test_creates_socket_path(self):
        from lackpy.sandbox.bridge import ToolBridgeManager
        mgr = ToolBridgeManager(callables={"read_file": lambda path: "contents"})
        assert mgr.socket_path is not None

    def test_start_creates_socket_file(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager
        fn = MagicMock(return_value="hello")
        mgr = ToolBridgeManager(callables={"test_tool": fn}, socket_dir=tmp_path)
        mgr.start()
        try:
            assert mgr.socket_path.exists()
        finally:
            mgr.stop()

    def test_stop_cleans_up(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager
        fn = MagicMock(return_value="hello")
        mgr = ToolBridgeManager(callables={"test_tool": fn}, socket_dir=tmp_path)
        mgr.start()
        sock = mgr.socket_path
        mgr.stop()
        assert not sock.exists()

    def test_context_manager(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager
        fn = MagicMock(return_value="hello")
        with ToolBridgeManager(callables={"test_tool": fn}, socket_dir=tmp_path) as mgr:
            assert mgr.socket_path.exists()
        assert not mgr.socket_path.exists()

    def test_client_can_call_tool(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager, bridge_client
        def greet(name: str) -> str:
            return f"hello {name}"
        with ToolBridgeManager(callables={"greet": greet}, socket_dir=tmp_path) as mgr:
            client = bridge_client(mgr.socket_path)
            result = client.call("greet", "world")
            assert result == "hello world"

    def test_client_unknown_tool_raises(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager, bridge_client
        with ToolBridgeManager(callables={}, socket_dir=tmp_path) as mgr:
            client = bridge_client(mgr.socket_path)
            with pytest.raises(KeyError):
                client.call("nonexistent")

    def test_no_callables_no_error(self, tmp_path):
        from lackpy.sandbox.bridge import ToolBridgeManager
        mgr = ToolBridgeManager(callables={}, socket_dir=tmp_path)
        mgr.start()
        mgr.stop()
