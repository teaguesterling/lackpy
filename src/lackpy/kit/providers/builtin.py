"""Built-in tool provider — tools implemented inside lackpy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..toolbox import ToolSpec


class BuiltinProvider:
    @property
    def name(self) -> str:
        return "builtin"

    def available(self) -> bool:
        return True

    def resolve(self, tool_spec: ToolSpec) -> Callable[..., Any]:
        implementations = {
            "read": _builtin_read,
            "glob": _builtin_glob,
            "write": _builtin_write,
            "edit": _builtin_edit,
        }
        fn = implementations.get(tool_spec.name)
        if fn is None:
            raise KeyError(f"No builtin implementation for '{tool_spec.name}'")
        return fn


def _builtin_read(path: str) -> str:
    return Path(path).read_text()


def _builtin_glob(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


def _builtin_write(path: str, content: str) -> bool:
    Path(path).write_text(content)
    return True


def _builtin_edit(path: str, old_str: str, new_str: str) -> bool:
    p = Path(path)
    text = p.read_text()
    if old_str not in text:
        return False
    p.write_text(text.replace(old_str, new_str, 1))
    return True
