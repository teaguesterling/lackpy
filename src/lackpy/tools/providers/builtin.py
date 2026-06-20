"""Built-in tool provider — tools implemented inside lackpy.

Retained for direct ``register_provider(BuiltinProvider())`` use (notably in tests).
The implementations themselves live in ``lackpy.sources.builtins`` (the single
source of truth shared with the shipped ``default_tools.toml``).
"""

from __future__ import annotations

from typing import Any, Callable

from ...sources.builtins import edit_file, find_files, read_file, write_file
from ..toolbox import ToolSpec


class BuiltinProvider:
    @property
    def name(self) -> str:
        return "builtin"

    def available(self) -> bool:
        return True

    def resolve(self, tool_spec: ToolSpec) -> Callable[..., Any]:
        implementations = {
            "read_file": read_file,
            "find_files": find_files,
            "write_file": write_file,
            "edit_file": edit_file,
        }
        fn = implementations.get(tool_spec.name)
        if fn is None:
            raise KeyError(f"No builtin implementation for '{tool_spec.name}'")
        return fn
