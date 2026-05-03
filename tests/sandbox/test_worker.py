"""Tests for the subprocess worker harness serialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestWorkerRequestSerialization:
    def test_write_and_read_request(self):
        from lackpy.sandbox._worker import write_request, read_request
        with tempfile.TemporaryDirectory() as td:
            io_dir = Path(td)
            request = {
                "program": "result = read_file('test.txt')",
                "interpreter": "python",
                "params": {"x": 1},
                "embedded_sources": {"read_file": "def read_file(path): return open(path).read()"},
                "bridge_socket": None,
                "base_dir": "/workspace",
            }
            write_request(io_dir, request)
            loaded = read_request(io_dir)
            assert loaded["program"] == request["program"]
            assert loaded["embedded_sources"]["read_file"] == request["embedded_sources"]["read_file"]

    def test_read_request_missing_file_raises(self):
        from lackpy.sandbox._worker import read_request
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                read_request(Path(td))


class TestWorkerResultSerialization:
    def test_write_and_read_result(self):
        from lackpy.sandbox._worker import write_result, read_result
        with tempfile.TemporaryDirectory() as td:
            io_dir = Path(td)
            result = {
                "success": True,
                "output": "hello world",
                "output_format": "text",
                "error": None,
                "duration_ms": 42.5,
                "metadata": {},
            }
            write_result(io_dir, result)
            loaded = read_result(io_dir)
            assert loaded["success"] is True
            assert loaded["output"] == "hello world"
            assert loaded["duration_ms"] == 42.5

    def test_read_result_missing_file_raises(self):
        from lackpy.sandbox._worker import read_result
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                read_result(Path(td))


class TestWorkerEmbeddedSources:
    def test_build_namespace_from_sources(self):
        from lackpy.sandbox._worker import build_tool_namespace
        sources = {
            "greet": "def greet(name):\n    return f'hello {name}'\n",
        }
        ns = build_tool_namespace(sources, bridge_client=None)
        assert "greet" in ns
        assert ns["greet"]("world") == "hello world"

    def test_build_namespace_empty(self):
        from lackpy.sandbox._worker import build_tool_namespace
        ns = build_tool_namespace({}, bridge_client=None)
        assert ns == {}
