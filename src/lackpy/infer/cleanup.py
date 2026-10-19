"""Deterministic cleanup — safe AST-level transforms for common model mistakes.

A lackpy program imports from the dictionary (stepy100), so cleanup keeps
approved imports and strips the others (by AST node). ``open(p).read()`` becomes
``Path(p).read_text()``, and ``from pathlib import Path`` is added when it is
missing. Source is re-printed (``ast.unparse``, which drops comments) only when
a rewrite changed something.
"""

from __future__ import annotations

import ast
import re

from ..lang.dictionary import MODULES

_IMPORT_RE = re.compile(r"^[ \t]*(?:import|from)[ \t]+([\w.]+)")


def _approved(module: str | None) -> bool:
    return bool(module) and module.split(".")[0] in MODULES


def _strip_import_lines(text: str) -> str:
    """Remove top-level imports of modules the dictionary does not approve.

    The program is parsed, and only the line ranges of ``Import`` /
    ``ImportFrom`` nodes are removed, so a string literal that holds an
    import-looking line is never touched (fix round 1, review I1). A node that
    imports an approved module too (``import os, json``), or that shares a line
    with another statement, is kept for the validator to report. Only a program
    that does not parse falls back to matching lines as text.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        kept = [line for line in text.splitlines()
                if not ((m := _IMPORT_RE.match(line)) and not _approved(m.group(1)))]
        return "\n".join(kept)

    def lines(node: ast.stmt) -> set[int]:
        return set(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    drop: set[int] = set()
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Import):
            unapproved = not any(_approved(a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            unapproved = node.level > 0 or not _approved(node.module)
        else:
            continue
        others = set().union(*(lines(n) for j, n in enumerate(tree.body) if j != i))
        if unapproved and not (lines(node) & others):
            drop |= lines(node)
    if not drop:
        return text
    return "\n".join(line for n, line in enumerate(text.splitlines(), start=1)
                     if n not in drop)


def _text_open(call: ast.expr) -> ast.expr | None:
    """The path of ``open(p)`` / ``open(p, "r")`` (text read), else None.

    The only keyword kept is ``encoding``; any other argument means the call is
    not a plain text read, so it is not rewritten.
    """
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "open" and 1 <= len(call.args) <= 2):
        return None
    if len(call.args) == 2:
        mode = call.args[1]
        if not (isinstance(mode, ast.Constant) and mode.value in ("r", "rt")):
            return None
    if any(k.arg != "encoding" for k in call.keywords):
        return None
    return call.args[0]


def _read_text(call: ast.Call, method: str) -> ast.expr | None:
    """``Path(p).read_text()`` for ``.read()``; None for any other method.

    ``.readlines()`` and file iteration are not rewritten: ``str.splitlines``
    also splits on ``\\x0c``, ``\\x85`` and others, where file lines split on
    ``\\n`` only (fix round 1, review M1). The validator reports them.
    """
    if method != "read":
        return None
    return ast.Call(
        func=ast.Attribute(
            value=ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[call.args[0]], keywords=[]),
            attr="read_text", ctx=ast.Load()),
        args=[], keywords=list(call.keywords))


class _OpenRewriter(ast.NodeTransformer):
    """Rewrite open(p).read() -> Path(p).read_text()."""

    changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and not node.args and not node.keywords
                and _text_open(node.func.value) is not None):
            new_node = _read_text(node.func.value, node.func.attr)
            if new_node is not None:
                self.changed = True
                return ast.copy_location(new_node, node)
        return node


class _WithOpenRewriter(ast.NodeTransformer):
    """Rewrite `with open(path) as f: target = f.read()` to a Path read.

        with open(f, 'r') as fh:
            content = fh.read()
        →  content = Path(f).read_text()

    The body may use the handle once only, in that one ``.read()``. A second
    read (``''`` under python3), ``next(fh)``, ``fh.close()`` or iteration keeps
    the with block (fix round 1, review M1, M2).
    """

    changed = False

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)

        if len(node.items) != 1:
            return node
        item = node.items[0]
        if (_text_open(item.context_expr) is None
                or not isinstance(item.optional_vars, ast.Name)):
            return node

        file_var = item.optional_vars.id
        uses = sum(isinstance(n, ast.Name) and n.id == file_var
                   for stmt in node.body for n in ast.walk(stmt))
        if uses != 1:
            return node
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            if not any(isinstance(n, ast.Name) and n.id == file_var for n in ast.walk(stmt)):
                new_body.append(stmt)
                continue
            rewritten = self._rewrite_read(stmt, file_var, item.context_expr)
            if rewritten is None:
                return node  # the handle is used some other way — keep the with block
            new_body.append(rewritten)
        self.changed = True
        return new_body

    def _rewrite_read(self, stmt: ast.stmt, file_var: str, call: ast.Call) -> ast.stmt | None:
        """``target = fh.read()`` -> ``target = Path(p).read_text()``, else None."""
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.value, ast.Call)
            and not stmt.value.args and not stmt.value.keywords
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == file_var
        ):
            new_value = _read_text(call, stmt.value.func.attr)
            if new_value is not None:
                return ast.copy_location(ast.Assign(targets=stmt.targets, value=new_value), stmt)
        return None


def _has_path_import(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.ImportFrom) and n.module == "pathlib" and n.level == 0
               and any(a.name == "Path" and a.asname is None for a in n.names)
               for n in tree.body)


class _OsPathRewriter(ast.NodeTransformer):
    """Rewrite os.path.basename(x) and os.path.join(a, b) to stdlib-free equivalents."""

    changed = False

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
            self.changed = True
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
            self.changed = True
            return ast.copy_location(new_node, node)

        return node


def deterministic_cleanup(program: str) -> str:
    """Apply deterministic cleanup transforms to a generated program.

    Steps:
    1. Strip top-level imports of modules the dictionary does not approve
       (by AST node line range; text matching only for a SyntaxError).
    2. Parse the rest as AST.
    3. Apply the with-open, open and os.path rewrites. When an open rewrite
       fired and ``from pathlib import Path`` is missing, add it first.
    4. Return ``ast.unparse()`` of the tree if a rewrite fired, otherwise the
       text from step 1 unchanged (comments, the intent block and waivers kept).

    If any AST step fails (SyntaxError, unparse failure), returns the
    text-cleaned version instead.

    Args:
        program: Raw Python source string, possibly containing unapproved
            imports and unsupported stdlib calls.

    Returns:
        Cleaned Python source string.
    """
    text_cleaned = _strip_import_lines(program).strip()

    try:
        tree = ast.parse(text_cleaned)
    except SyntaxError:
        return text_cleaned

    try:
        with_open, open_ = _WithOpenRewriter(), _OpenRewriter()
        os_path = _OsPathRewriter()
        tree = os_path.visit(open_.visit(with_open.visit(tree)))
        if not (with_open.changed or open_.changed or os_path.changed):
            return text_cleaned
        if (with_open.changed or open_.changed) and not _has_path_import(tree):
            tree.body.insert(0, ast.ImportFrom(
                module="pathlib", names=[ast.alias(name="Path")], level=0))
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return text_cleaned
