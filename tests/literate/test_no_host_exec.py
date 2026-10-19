"""Anti-goal (literate stepy Task 2, design gate 5): no code in the literate
interpreter execs or evals document code. Every cell, interpolation and
inspection goes through ``CellSession``.

The check is static: no call to the ``exec``, ``eval`` or ``compile``
builtins anywhere in ``src/lackpy/interpreters/literate/``, except the
parse-only ``compile`` calls listed here (``ast.PyCF_ONLY_AST``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import lackpy.interpreters.literate as literate

ROOT = Path(literate.__file__).parent

#: (file relative to ROOT, function) of each allowed parse-only compile call.
PARSE_ONLY_COMPILES = {("kernel/static_analysis.py", "check_cell")}


def _calls():
    """(file, innermost enclosing function, Call node) for each exec/eval/compile call."""
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        stack = ["<module>"]
        found = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "compile"}:
                    found.append((rel, stack[-1], node))
                self.generic_visit(node)

        Visitor().visit(ast.parse(path.read_text()))
        yield from found


def test_no_exec_eval_or_compile_outside_cellsession():
    found = {(f, fn, node.func.id) for f, fn, node in _calls()}
    allowed = {(f, fn, "compile") for f, fn in PARSE_ONLY_COMPILES}
    assert found - allowed == set()


def test_the_allowed_compiles_are_parse_only():
    for f, fn, node in _calls():
        if node.func.id == "compile":
            flags = [ast.unparse(a) for a in node.args[3:]] + [
                ast.unparse(k.value) for k in node.keywords if k.arg == "flags"]
            assert "ast.PyCF_ONLY_AST" in flags, (f, fn)
