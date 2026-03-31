"""Extract run() body from Lackey classes and rewrite self.x -> x."""

from __future__ import annotations

import ast
import textwrap


def extract_run_source(source: str, class_name: str) -> str:
    """Extract the run() method body from a Lackey class as source code.

    Args:
        source: Full Python source of the file.
        class_name: Name of the Lackey subclass.

    Returns:
        Source code of the run() body (without the def line).

    Raises:
        ValueError: If the class or run() method is not found.
    """
    tree = ast.parse(source)

    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_node = node
            break

    if class_node is None:
        raise ValueError(f"Class '{class_name}' not found in source")

    run_node = None
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            run_node = node
            break

    if run_node is None:
        raise ValueError(f"No run() method found in class '{class_name}'")

    lines = source.splitlines()
    body_start = run_node.body[0].lineno - 1
    body_end = run_node.end_lineno

    body_lines = lines[body_start:body_end]
    body_source = textwrap.dedent("\n".join(body_lines))

    return body_source


def rewrite_self_to_plain(code: str) -> str:
    """Rewrite self.x references to plain x in lackpy code.

    Transforms:
        self.read(path)  ->  read(path)
        self.pattern     ->  pattern
    """
    tree = ast.parse(code)
    tree = _SelfRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class _SelfRewriter(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.Name(id=node.attr, ctx=node.ctx)
        return node
