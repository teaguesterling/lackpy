"""Create Lackey .py files from generated programs."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def create_lackey_source(
    program: str, name: str, tools: list[str],
    params: dict[str, dict[str, Any]] | None = None,
    returns: str | None = None,
    creation_log: list[dict[str, Any]] | None = None,
    description: str | None = None,
) -> str:
    """Generate a Lackey class source file from a lackpy program."""
    params = params or {}
    lines: list[str] = []

    imports = ["Lackey", "Tool"]
    if creation_log:
        imports.extend(["Log", "System", "User", "Assistant"])
    lines.append(f"from lackpy.lackey import {', '.join(imports)}")
    lines.append("")
    lines.append("")
    lines.append(f"class {name}(Lackey):")

    if description:
        lines.append(f'    """{description}"""')
        lines.append("")

    for tool in tools:
        lines.append(f"    {tool} = Tool()")
    if tools:
        lines.append("")

    for param_name, spec in params.items():
        ptype = spec.get("type", "str")
        default = spec.get("default")
        if default is not None:
            lines.append(f"    {param_name}: {ptype} = {_repr_default(default)}")
        else:
            lines.append(f"    {param_name}: {ptype}")
    if params:
        lines.append("")

    if returns:
        lines.append(f"    returns: {returns}")
        lines.append("")

    if creation_log:
        lines.append("    creation_log = Log([")
        for msg in creation_log:
            role = msg["role"]
            content_repr = repr(msg["content"])
            if role == "system":
                lines.append(f"        System({content_repr}),")
            elif role == "user":
                lines.append(f"        User({content_repr}),")
            elif role == "assistant":
                accepted = msg.get("accepted", True)
                errors = msg.get("errors")
                strategy = msg.get("strategy")
                parts = [content_repr, f"accepted={accepted!r}"]
                if errors:
                    parts.append(f"errors={errors!r}")
                if strategy:
                    parts.append(f"strategy={strategy!r}")
                lines.append(f"        Assistant({', '.join(parts)}),")
        lines.append("    ])")
        lines.append("")

    run_body = _add_self_prefix(program, tools, set(params.keys()))
    return_annotation = f" -> {returns}" if returns else ""
    lines.append(f"    def run(self){return_annotation}:")

    for body_line in run_body.splitlines():
        if body_line.strip():
            lines.append(f"        {body_line}")
        else:
            lines.append("")

    lines.append("")
    return "\n".join(lines)


def _repr_default(value: Any) -> str:
    """Format a default value, using double quotes for strings."""
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return repr(value)


def save_lackey(program: str, name: str, tools: list[str], output_dir: Path, **kwargs: Any) -> Path:
    """Create and save a Lackey file."""
    source = create_lackey_source(program, name, tools, **kwargs)
    filename = _class_name_to_filename(name)
    path = output_dir / filename
    path.write_text(source)
    return path


def _add_self_prefix(program: str, tools: list[str], params: set[str]) -> str:
    """Rewrite bare tool calls and param references to use self. prefix."""
    tree = ast.parse(program)
    names_to_prefix = set(tools) | params
    tree = _SelfPrefixer(names_to_prefix).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class _SelfPrefixer:
    def __init__(self, names: set[str]) -> None:
        self._names = names

    def visit(self, node: ast.AST) -> ast.AST:
        for child_name, child_node in ast.iter_fields(node):
            if isinstance(child_node, list):
                for i, item in enumerate(child_node):
                    if isinstance(item, ast.AST):
                        child_node[i] = self.visit(item)
            elif isinstance(child_node, ast.AST):
                setattr(node, child_name, self.visit(child_node))

        if isinstance(node, ast.Name) and node.id in self._names:
            if isinstance(node.ctx, ast.Store):
                return node
            return ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=node.id, ctx=node.ctx,
            )
        return node


def _class_name_to_filename(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    s = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", s)
    return s.lower() + ".py"
