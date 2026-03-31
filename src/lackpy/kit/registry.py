"""Kit resolution: name/list/dict/None -> ResolvedKit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..lang.grader import Grade, compute_grade
from .toolbox import Toolbox, ToolSpec


@dataclass
class ResolvedKit:
    """A fully resolved kit with callables ready for execution.

    Attributes:
        tools: Mapping of tool name (or alias) to spec.
        callables: Mapping of tool name (or alias) to callable implementation.
        grade: Aggregate security grade (join of tool grades).
        description: Formatted namespace description for inference prompts.
    """

    tools: dict[str, ToolSpec]
    callables: dict[str, Callable[..., Any]]
    grade: Grade
    description: str


def resolve_kit(
    kit: str | list[str] | dict[str, str | dict] | None,
    toolbox: Toolbox,
    kits_dir: Path | None = None,
) -> ResolvedKit:
    """Resolve a kit specification into a ResolvedKit ready for execution.

    Accepts four forms for ``kit``:

    - ``str``: name of a ``.kit`` file in ``kits_dir``
    - ``list[str]``: explicit list of tool names
    - ``dict``: alias-to-tool mapping; values may be a tool name string or a
      dict with a ``"tool"`` key
    - ``None``: not yet supported (raises NotImplementedError)

    Args:
        kit: Kit specification — name, list of names, dict mapping, or None.
        toolbox: The Toolbox instance from which to resolve tools.
        kits_dir: Directory containing ``.kit`` files. Defaults to
            ``.lackpy/kits`` relative to cwd.

    Returns:
        A ResolvedKit with tools, callables, grade, and description populated.

    Raises:
        NotImplementedError: If ``kit`` is None (Quartermaster not implemented).
        FileNotFoundError: If a named kit file is not found in ``kits_dir``.
        KeyError: If a tool name is not registered in the toolbox.
        TypeError: If ``kit`` is an unsupported type or contains an unsupported entry type.
    """
    if kit is None:
        raise NotImplementedError(
            "Quartermaster (automatic kit selection) is not yet implemented. "
            "Specify a kit name, tool list, or tool mapping."
        )
    if isinstance(kit, str):
        tool_names = _load_kit_file(kit, kits_dir)
        return _resolve_tool_names(tool_names, tool_names, toolbox)
    elif isinstance(kit, list):
        return _resolve_tool_names(kit, kit, toolbox)
    elif isinstance(kit, dict):
        return _resolve_dict_kit(kit, toolbox)
    else:
        raise TypeError(f"Unsupported kit type: {type(kit)}")


def _load_kit_file(name: str, kits_dir: Path | None) -> list[str]:
    if kits_dir is None:
        kits_dir = Path(".lackpy/kits")
    kit_file = kits_dir / f"{name}.kit"
    if not kit_file.exists():
        raise FileNotFoundError(f"Kit file not found: {kit_file}")
    text = kit_file.read_text()
    lines = text.strip().split("\n")
    in_frontmatter = False
    tool_names = []
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if stripped and not stripped.startswith("#"):
            tool_names.append(stripped)
    return tool_names


def _resolve_tool_names(tool_names: list[str], alias_names: list[str], toolbox: Toolbox) -> ResolvedKit:
    tools: dict[str, ToolSpec] = {}
    callables: dict[str, Callable] = {}
    for alias, name in zip(alias_names, tool_names):
        if name not in toolbox.tools:
            raise KeyError(f"Unknown tool: '{name}'")
        spec = toolbox.tools[name]
        tools[alias] = spec
        callables[alias] = toolbox.resolve(name)
    grade_input = {
        n: {"grade_w": s.grade_w, "effects_ceiling": s.effects_ceiling}
        for n, s in tools.items()
    }
    grade = compute_grade(grade_input)
    description = toolbox.format_description(tool_names)
    return ResolvedKit(tools=tools, callables=callables, grade=grade, description=description)


def _resolve_dict_kit(kit: dict[str, str | dict], toolbox: Toolbox) -> ResolvedKit:
    alias_names = []
    tool_names = []
    for alias, value in kit.items():
        if isinstance(value, str):
            alias_names.append(alias)
            tool_names.append(value)
        elif isinstance(value, dict):
            actual_name = value.get("tool", alias)
            alias_names.append(alias)
            tool_names.append(actual_name)
        else:
            raise TypeError(f"Unsupported kit entry for '{alias}': {type(value)}")
    return _resolve_tool_names(tool_names, alias_names, toolbox)
