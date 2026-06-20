"""Kit resolution: name/list/dict/None -> ResolvedTools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..lang.grader import Grade, compute_grade
from .toolbox import Toolbox, ToolSpec


@dataclass
class ResolvedTools:
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


def resolve_tools(
    selection: str | list[str] | dict[str, str | dict] | None,
    toolbox: Toolbox,
    kits_dir: Path | None = None,
    extra_tools: list[str] | None = None,
) -> ResolvedTools:
    """Resolve a kit specification into a ResolvedTools ready for execution.

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
        A ResolvedTools with tools, callables, grade, and description populated.

    Raises:
        NotImplementedError: If ``kit`` is None (Quartermaster not implemented).
        FileNotFoundError: If a named kit file is not found in ``kits_dir``.
        KeyError: If a tool name is not registered in the toolbox.
        TypeError: If ``kit`` is an unsupported type or contains an unsupported entry type.
    """
    if selection is None:
        if extra_tools:
            selection = []
        else:
            raise NotImplementedError(
                "Quartermaster (automatic profile selection) is not yet implemented. "
                "Specify a profile name, tool list, or tool mapping."
            )
    if isinstance(selection, str) and selection == "none":
        resolved = _resolve_tool_names([], [], toolbox)
    elif isinstance(selection, str):
        meta = _load_tools_file(selection, kits_dir)
        resolved = _resolve_tool_names(meta.tool_names, meta.tool_names, toolbox)
        resolved.docs = meta.docs
    elif isinstance(selection, list):
        resolved = _resolve_tool_names(selection, selection, toolbox)
    elif isinstance(selection, dict):
        resolved = _resolve_dict_tools(selection, toolbox)
    else:
        raise TypeError(f"Unsupported profile selection type: {type(selection)}")

    if extra_tools:
        resolved = _merge_extra_tools(resolved, extra_tools, toolbox)
    return resolved


@dataclass
class ToolsFileMetadata:
    tool_names: list[str]
    docs: list[str] = field(default_factory=list)


def _load_tools_file(name: str, kits_dir: Path | None) -> ToolsFileMetadata:
    if kits_dir is None:
        kits_dir = Path(".lackpy/kits")
    # Prefer .profile; fall back to legacy .kit (both are listed by profile_list, so
    # both must resolve).
    tools_file = kits_dir / f"{name}.profile"
    if not tools_file.exists():
        tools_file = kits_dir / f"{name}.kit"
    if not tools_file.exists():
        raise FileNotFoundError(
            f"Profile/tool-set file not found: {kits_dir / f'{name}.profile'} (or .kit)")
    text = tools_file.read_text()
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
    return ToolsFileMetadata(tool_names=tool_names, docs=docs)


def _resolve_tool_names(tool_names: list[str], alias_names: list[str], toolbox: Toolbox) -> ResolvedTools:
    tools: dict[str, ToolSpec] = {}
    callables: dict[str, Callable] = {}
    tool_docs: list[str] = []
    for alias, name in zip(alias_names, tool_names):
        if name not in toolbox.tools:
            available = sorted(toolbox.tools)
            shown = ", ".join(available[:20]) + (" …" if len(available) > 20 else "")
            raise KeyError(
                f"Unknown tool {name!r}: no configured source provides it. Tools come "
                f"from builtins, config [[tools]], or an MCP server — confirm the kit's "
                f"names match a configured source. Available now: {shown or '(none)'}"
            )
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
    return ResolvedTools(tools=tools, callables=callables, grade=grade, description=description, docs=tool_docs)


def _resolve_dict_tools(mapping: dict[str, str | dict], toolbox: Toolbox) -> ResolvedTools:
    alias_names = []
    tool_names = []
    for alias, value in mapping.items():
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


def _merge_extra_tools(resolved: ResolvedTools, extra_tools: list[str], toolbox: Toolbox) -> ResolvedTools:
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
    return ResolvedTools(tools=merged_tools, callables=merged_callables, grade=grade, description=description, docs=merged_docs)
