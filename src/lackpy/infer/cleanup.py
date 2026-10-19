"""Deterministic cleanup — safe AST-level transforms for common model mistakes.

A lackpy program imports from the dictionary (stepy100), so cleanup keeps
approved imports and strips the others. ``open(p).read()`` becomes
``Path(p).read_text()``, and ``from pathlib import Path`` is added when it is
missing. Source is re-printed (``ast.unparse``, which drops comments) only when
a rewrite changed something.
"""

from __future__ import annotations

import ast
import re

from ..lang.dictionary import MODULES

_IMPORT_RE = re.compile(r"^[ \t]*(?:import|from)[ \t]+([\w.]+)")


def _strip_import_lines(text: str) -> str:
    """Remove import lines whose module the dictionary does not approve."""
    kept = []
    for line in text.splitlines():
        m = _IMPORT_RE.match(line)
        if m and m.group(1).split(".")[0] not in MODULES:
            continue
        kept.append(line)
    return "\n".join(kept)


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
    """``Path(p).read_text()`` for ``.read()``, plus ``.splitlines(keepends=True)``
    for ``.readlines()`` (readlines keeps the line ends). None for other methods."""
    if method not in ("read", "readlines"):
        return None
    path_arg = call.args[0]
    node: ast.expr = ast.Call(
        func=ast.Attribute(
            value=ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[path_arg], keywords=[]),
            attr="read_text", ctx=ast.Load()),
        args=[], keywords=list(call.keywords))
    if method == "readlines":
        node = ast.Call(func=ast.Attribute(value=node, attr="splitlines", ctx=ast.Load()),
                        args=[], keywords=[ast.keyword(arg="keepends", value=ast.Constant(True))])
    return node


class _OpenRewriter(ast.NodeTransformer):
    """Rewrite open(p).read() -> Path(p).read_text() and
    open(p).readlines() -> Path(p).read_text().splitlines(keepends=True)."""

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
    """Rewrite `with open(path) as f: body` to Path reads.

    Transforms patterns like:
        with open(f, 'r') as fh:
            content = fh.read()
        →  content = Path(f).read_text()

        with open(f) as fh:
            lines = fh.readlines()
        →  lines = Path(f).read_text().splitlines(keepends=True)

        with open(f) as fh:
            for line in fh: ...
        →  for line in Path(f).read_text().splitlines(keepends=True): ...
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

        call = item.context_expr
        file_var = item.optional_vars.id
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            rewritten = self._rewrite_file_usage(stmt, file_var, call)
            if rewritten is None:
                return node  # can't rewrite this statement — keep the whole with block
            new_body.append(rewritten)
        self.changed = True
        return new_body

    def _rewrite_file_usage(self, stmt: ast.stmt, file_var: str,
                            call: ast.Call) -> ast.stmt | None:
        """Rewrite one statement that uses the file handle, or return None."""
        # Match: target = fh.read() / fh.readlines()
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
            if new_value is None:
                return None
            return ast.copy_location(ast.Assign(targets=stmt.targets, value=new_value), stmt)

        # Match: for line in fh: ... (iterate lines, ends kept)
        if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Name) and stmt.iter.id == file_var:
            stmt.iter = _read_text(call, "readlines")
            return stmt

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
    1. Strip import lines of modules the dictionary does not approve (text-level).
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
