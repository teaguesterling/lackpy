"""Kit resolution: name/list/dict/None -> ResolvedKit."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    docs: list[str] = field(default_factory=list)


def resolve_kit(
    kit: str | list[str] | dict[str, str | dict] | None,
    toolbox: Toolbox,
    kits_dir: Path | None = None,
    extra_tools: list[str] | None = None,
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
        extra_tools: Additional tool names to merge into the resolved kit.
            Duplicates of tools already in the kit are silently ignored.

    Returns:
        A ResolvedKit with tools, callables, grade, and description populated.

    Raises:
        NotImplementedError: If ``kit`` is None (Quartermaster not implemented).
        FileNotFoundError: If a named kit file is not found in ``kits_dir``.
        KeyError: If a tool name is not registered in the toolbox.
        TypeError: If ``kit`` is an unsupported type or contains an unsupported entry type.
    """
    if kit is None:
        if extra_tools:
            kit = []
        else:
            raise NotImplementedError(
                "Quartermaster (automatic kit selection) is not yet implemented. "
                "Specify a kit name, tool list, or tool mapping."
            )
    if isinstance(kit, str) and kit == "none":
        resolved = _resolve_tool_names([], [], toolbox)
    elif isinstance(kit, str):
        meta = _load_kit_file(kit, kits_dir)
        resolved = _resolve_tool_names(meta.tool_names, meta.tool_names, toolbox)
        resolved.docs = meta.docs
    elif isinstance(kit, list):
        resolved = _resolve_tool_names(kit, kit, toolbox)
    elif isinstance(kit, dict):
        resolved = _resolve_dict_kit(kit, toolbox)
    else:
        raise TypeError(f"Unsupported kit type: {type(kit)}")

    if extra_tools:
        resolved = _merge_extra_tools(resolved, extra_tools, toolbox)
    return resolved


@dataclass
class KitFileMetadata:
    tool_names: list[str]
    docs: list[str] = field(default_factory=list)


def _load_kit_file(name: str, kits_dir: Path | None) -> KitFileMetadata:
    if kits_dir is None:
        kits_dir = Path(".lackpy/kits")
    kit_file = kits_dir / f"{name}.kit"
    if not kit_file.exists():
        raise FileNotFoundError(f"Kit file not found: {kit_file}")
    text = kit_file.read_text()
    lines = text.strip().split("\n")
    in_frontmatter = False
    tool_names = []
    docs: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            if stripped.startswith("docs:"):
                value = stripped[5:].strip()
                if value:
                    docs.append(value)
            continue
        if stripped and not stripped.startswith("#"):
            tool_names.append(stripped)
    return KitFileMetadata(tool_names=tool_names, docs=docs)


def _resolve_tool_names(tool_names: list[str], alias_names: list[str], toolbox: Toolbox) -> ResolvedKit:
    tools: dict[str, ToolSpec] = {}
    callables: dict[str, Callable] = {}
    tool_docs: list[str] = []
    for alias, name in zip(alias_names, tool_names):
        if name not in toolbox.tools:
            raise KeyError(f"Unknown tool: '{name}'")
        spec = toolbox.tools[name]
        tools[alias] = spec
        callables[alias] = toolbox.resolve(name)
        if spec.docs:
            tool_docs.append(spec.docs)
    grade_input = {
        n: {"grade_w": s.grade_w, "effects_ceiling": s.effects_ceiling}
        for n, s in tools.items()
    }
    grade = compute_grade(grade_input)
    description = toolbox.format_description(tool_names)
    return ResolvedKit(tools=tools, callables=callables, grade=grade, description=description, docs=tool_docs)


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


def _merge_extra_tools(resolved: ResolvedKit, extra_tools: list[str], toolbox: Toolbox) -> ResolvedKit:
    new_names = [n for n in extra_tools if n not in resolved.tools]
    if not new_names:
        return resolved
    extra = _resolve_tool_names(new_names, new_names, toolbox)
    merged_tools = {**resolved.tools, **extra.tools}
    merged_callables = {**resolved.callables, **extra.callables}
    grade_input = {
        n: {"grade_w": s.grade_w, "effects_ceiling": s.effects_ceiling}
        for n, s in merged_tools.items()
    }
    grade = compute_grade(grade_input)
    description = toolbox.format_description(list(merged_tools.keys()))
    merged_docs = list(dict.fromkeys(resolved.docs + extra.docs))
    return ResolvedKit(tools=merged_tools, callables=merged_callables, grade=grade, description=description, docs=merged_docs)
