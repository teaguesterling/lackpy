"""Deterministic cleanup — safe AST-level transforms for common model mistakes."""

from __future__ import annotations

import ast
import re


_IMPORT_RE = re.compile(r"^[ \t]*(import |from )\S", re.MULTILINE)


def _strip_import_lines(text: str) -> str:
    """Remove import and from-import lines from source text."""
    lines = text.splitlines()
    kept = [line for line in lines if not re.match(r"^[ \t]*(import |from )\S", line)]
    return "\n".join(kept)


class _OpenRewriter(ast.NodeTransformer):
    """Rewrite open(path).read() -> read(path) and open(path).readlines() -> read(path).splitlines()."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # Match: open(expr).read() or open(expr).readlines()
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "open"
            and len(node.func.value.args) >= 1
            and not node.func.value.keywords
        ):
            path_arg = node.func.value.args[0]
            method = node.func.attr

            if method == "read":
                # open(path).read() -> read(path)
                new_node = ast.Call(
                    func=ast.Name(id="read", ctx=ast.Load()),
                    args=[path_arg],
                    keywords=[],
                )
                return ast.copy_location(new_node, node)

            if method == "readlines":
                # open(path).readlines() -> read(path).splitlines()
                read_call = ast.Call(
                    func=ast.Name(id="read", ctx=ast.Load()),
                    args=[path_arg],
                    keywords=[],
                )
                new_node = ast.Call(
                    func=ast.Attribute(
                        value=read_call,
                        attr="splitlines",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
                return ast.copy_location(new_node, node)

        return node


class _OsPathRewriter(ast.NodeTransformer):
    """Rewrite os.path.basename(x) and os.path.join(a, b) to stdlib-free equivalents."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        # Match os.path.basename(x) -> x.rsplit('/', 1)[-1]
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "basename"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "path"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
            and len(node.args) == 1
            and not node.keywords
        ):
            x = node.args[0]
            rsplit_call = ast.Call(
                func=ast.Attribute(value=x, attr="rsplit", ctx=ast.Load()),
                args=[ast.Constant(value="/"), ast.Constant(value=1)],
                keywords=[],
            )
            new_node = ast.Subscript(
                value=rsplit_call,
                slice=ast.Constant(value=-1),
                ctx=ast.Load(),
            )
            return ast.copy_location(new_node, node)

        # Match os.path.join(a, b) -> f"{a}/{b}"
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "path"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
            and len(node.args) == 2
            and not node.keywords
        ):
            a, b = node.args
            new_node = ast.JoinedStr(
                values=[
                    ast.FormattedValue(value=a, conversion=-1, format_spec=None),
                    ast.Constant(value="/"),
                    ast.FormattedValue(value=b, conversion=-1, format_spec=None),
                ]
            )
            return ast.copy_location(new_node, node)

        return node


def deterministic_cleanup(program: str) -> str:
    """Apply deterministic cleanup transforms to a generated program.

    Steps:
    1. Strip import/from-import lines (text-level).
    2. Parse remaining code as AST.
    3. Apply _OpenRewriter and _OsPathRewriter transforms.
    4. Return ast.unparse() result.

    If any AST step fails (SyntaxError, unparse failure), returns the
    text-cleaned version instead.

    Args:
        program: Raw Python source string, possibly containing imports and
            unsupported stdlib calls.

    Returns:
        Cleaned Python source string.
    """
    text_cleaned = _strip_import_lines(program).strip()

    try:
        tree = ast.parse(text_cleaned)
    except SyntaxError:
        return text_cleaned

    try:
        tree = _OpenRewriter().visit(tree)
        tree = _OsPathRewriter().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return text_cleaned
