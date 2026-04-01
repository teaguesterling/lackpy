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


class _WithOpenRewriter(ast.NodeTransformer):
    """Rewrite `with open(path) as f: body` to inline read() calls.

    Transforms patterns like:
        with open(f, 'r') as fh:
            content = fh.read()
        →  content = read(f)

        with open(f) as fh:
            lines = fh.readlines()
        →  lines = read(f).splitlines()
    """

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)

        if len(node.items) != 1:
            return node

        item = node.items[0]
        # Must be: with open(...) as <name>
        if (
            not isinstance(item.context_expr, ast.Call)
            or not isinstance(item.context_expr.func, ast.Name)
            or item.context_expr.func.id != "open"
            or item.optional_vars is None
            or not isinstance(item.optional_vars, ast.Name)
            or not item.context_expr.args
        ):
            return node

        path_arg = item.context_expr.args[0]
        file_var = item.optional_vars.id

        # Rewrite body statements that use the file variable
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            rewritten = self._rewrite_file_usage(stmt, file_var, path_arg)
            if rewritten is not None:
                new_body.append(rewritten)
            else:
                # Can't rewrite this statement — keep the whole with block
                return node

        return new_body

    def _rewrite_file_usage(
        self, stmt: ast.stmt, file_var: str, path_arg: ast.expr,
    ) -> ast.stmt | None:
        """Try to rewrite a single statement that uses the file handle.

        Returns rewritten statement, or None if it can't be rewritten.
        """
        # Match: target = fh.read()
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == file_var
        ):
            method = stmt.value.func.attr
            if method == "read":
                # target = read(path)
                new_value = ast.Call(
                    func=ast.Name(id="read", ctx=ast.Load()),
                    args=[path_arg],
                    keywords=[],
                )
                new_stmt = ast.Assign(
                    targets=stmt.targets, value=new_value,
                    lineno=stmt.lineno, col_offset=stmt.col_offset,
                )
                return ast.copy_location(new_stmt, stmt)

            if method == "readlines":
                # target = read(path).splitlines()
                read_call = ast.Call(
                    func=ast.Name(id="read", ctx=ast.Load()),
                    args=[path_arg],
                    keywords=[],
                )
                new_value = ast.Call(
                    func=ast.Attribute(
                        value=read_call, attr="splitlines", ctx=ast.Load(),
                    ),
                    args=[], keywords=[],
                )
                new_stmt = ast.Assign(
                    targets=stmt.targets, value=new_value,
                    lineno=stmt.lineno, col_offset=stmt.col_offset,
                )
                return ast.copy_location(new_stmt, stmt)

        # Match: for line in fh: ... (iterate lines)
        if (
            isinstance(stmt, ast.For)
            and isinstance(stmt.iter, ast.Name)
            and stmt.iter.id == file_var
        ):
            # for line in fh → for line in read(path).splitlines()
            read_call = ast.Call(
                func=ast.Name(id="read", ctx=ast.Load()),
                args=[path_arg], keywords=[],
            )
            stmt.iter = ast.Call(
                func=ast.Attribute(
                    value=read_call, attr="splitlines", ctx=ast.Load(),
                ),
                args=[], keywords=[],
            )
            return stmt

        return None


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
        tree = _WithOpenRewriter().visit(tree)
        tree = _OpenRewriter().visit(tree)
        tree = _OsPathRewriter().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return text_cleaned
