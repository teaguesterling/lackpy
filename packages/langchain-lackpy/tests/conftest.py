"""Shared fixtures for langchain-lackpy tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest

from lackpy.kit.toolbox import ArgSpec, ToolSpec, Toolbox
from lackpy.kit.registry import ResolvedKit
from lackpy.lang.grader import Grade


@pytest.fixture
def sample_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="read_file", provider="builtin",
            description="Read file contents",
            args=[ArgSpec(name="path", type="str", description="File path")],
            returns="str", grade_w=1, effects_ceiling=1,
        ),
        ToolSpec(
            name="write_file", provider="builtin",
            description="Write content to a file",
            args=[
                ArgSpec(name="path", type="str", description="File path"),
                ArgSpec(name="content", type="str", description="Content to write"),
            ],
            returns="bool", grade_w=3, effects_ceiling=3,
        ),
    ]


@pytest.fixture
def mock_toolbox(sample_specs) -> Toolbox:
    toolbox = Toolbox()
    for spec in sample_specs:
        toolbox.register_tool(spec)
    toolbox._providers["builtin"] = MagicMock(
        name="builtin",
        resolve=lambda spec: lambda **kw: f"mock:{spec.name}({kw})",
    )
    return toolbox


@pytest.fixture
def resolved_kit(mock_toolbox, sample_specs) -> ResolvedKit:
    from lackpy.kit.registry import resolve_kit
    return resolve_kit(["read_file", "write_file"], mock_toolbox)


@pytest.fixture
def mock_service(mock_toolbox, tmp_path) -> MagicMock:
    svc = MagicMock()
    svc.toolbox = mock_toolbox
    svc.workspace = tmp_path
    svc.delegate = AsyncMock(return_value={
        "success": True,
        "output": "file contents here",
        "error": None,
        "program": "x = read_file('test.txt')",
        "grade": {"w": 1, "d": 1},
        "trace": [],
        "files_read": ["test.txt"],
        "files_modified": [],
        "generation_time_ms": 100,
        "execution_time_ms": 50,
        "total_time_ms": 150,
    })
    return svc
