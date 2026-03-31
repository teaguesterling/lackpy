"""Parse Lackey .py files into structured LackeyInfo."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extractor import extract_run_source, rewrite_self_to_plain


@dataclass
class LackeyInfo:
    """Parsed metadata from a Lackey file."""
    name: str
    description: str
    class_name: str
    tools: list[str]
    params: dict[str, dict[str, Any]]
    returns: str | None
    run_body: str
    has_creation_log: bool
    path: Path


_RESERVED = {"returns", "creation_log"}


def parse_lackey(path: Path) -> LackeyInfo:
    """Parse a Lackey .py file and extract metadata."""
    source = path.read_text()
    tree = ast.parse(source)

    class_node = _find_lackey_class(tree)
    if class_node is None:
        raise ValueError(f"No Lackey subclass found in {path}")

    tools = _extract_tools(class_node)
    params = _extract_params(class_node, tools)
    returns = _extract_returns(class_node)
    description = ast.get_docstring(class_node) or ""

    has_creation_log = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "creation_log" for t in node.targets)
        for node in class_node.body
    )

    run_source = extract_run_source(source, class_node.name)
    run_body = rewrite_self_to_plain(run_source)

    return LackeyInfo(
        name=path.stem, description=description, class_name=class_node.name,
        tools=tools, params=params, returns=returns, run_body=run_body,
        has_creation_log=has_creation_log, path=path,
    )


def _find_lackey_class(tree: ast.Module) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Lackey":
                    return node
    return None


def _extract_tools(class_node: ast.ClassDef) -> list[str]:
    tools = []
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_tool_call(node.value):
                    tools.append(target.id)
    return tools


def _is_tool_call(node: ast.expr) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Tool")


def _extract_params(class_node: ast.ClassDef, tools: list[str]) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name in _RESERVED or name in tools or name.startswith("_"):
                continue
            spec: dict[str, Any] = {"type": ast.unparse(node.annotation)}
            if node.value is not None:
                try:
                    spec["default"] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    spec["default"] = ast.unparse(node.value)
            params[name] = spec
    return params


def _extract_returns(class_node: ast.ClassDef) -> str | None:
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "returns":
            return ast.unparse(node.annotation)
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            if node.returns:
                return ast.unparse(node.returns)
    return None
