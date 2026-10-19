"""Anti-goal (literate stepy Task 2, design gate 5): no code in the literate
interpreter execs or evals document code. Every cell, interpolation and
inspection goes through ``CellSession``.

The check is static, over ``src/lackpy/interpreters/literate/``:

* no reference to the ``exec``, ``eval``, ``compile``, ``__import__`` or
  ``__builtins__`` names, called or aliased (``e = eval``), and no attribute
  of that name (``builtins.exec``); ``re.compile`` and the parse-only
  ``compile`` calls listed here (``ast.PyCF_ONLY_AST``) are allowed;
* no import of a module that runs or builds code or processes
  (``importlib``, ``runpy``, ``subprocess``, ``builtins``, ``code``, ...);
* no ``types.FunctionType`` / ``types.CodeType`` / ``os.system`` / ``os.exec*``.

``tools.py`` is the one exception: it is unwired (no src module imports it,
checked below) and literate stepy Task 3 deletes it (D7).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import lackpy.interpreters.literate as literate

ROOT = Path(literate.__file__).parent
SRC = ROOT.parents[1]  # src/lackpy (a namespace package: no __file__)

#: (file relative to ROOT, function) of each allowed parse-only compile call.
PARSE_ONLY_COMPILES = {("kernel/static_analysis.py", "check_cell")}

#: Unwired until Task 3 deletes it (D7).
UNWIRED = {"tools.py"}

#: (file, what) references that run no code, with the reason.
ALLOWED = {
    # dir(builtins) only: the names check_cell resolves against.
    ("kernel/static_analysis.py", "import builtins"),
    # Reads the packaged tool_effects.toml; Task 3 deletes both (D7).
    ("effects.py", "from importlib.resources import"),
}

_NAMES = {"exec", "eval", "compile", "__import__", "__builtins__"}
_MODULES = {"importlib", "runpy", "subprocess", "builtins", "code", "codeop",
            "ctypes", "marshal", "multiprocessing", "pty"}
_ATTRS = {("types", "FunctionType"), ("types", "CodeType"), ("os", "system"),
          ("os", "popen"), ("os", "fork"), ("os", "spawnv")}


def _findings():
    """(file, innermost function, what) for each forbidden reference."""
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in UNWIRED:
            continue
        stack = ["<module>"]
        found = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "compile":
                    flags = [ast.unparse(a) for a in node.args[3:]] + [
                        ast.unparse(k.value) for k in node.keywords if k.arg == "flags"]
                    if (rel, stack[-1]) in PARSE_ONLY_COMPILES and "ast.PyCF_ONLY_AST" in flags:
                        for a in node.args:  # visit the arguments, not the allowed name
                            self.visit(a)
                        return
                self.generic_visit(node)

            def visit_Name(self, node):
                if node.id in _NAMES:
                    found.append((rel, stack[-1], node.id))

            def visit_Attribute(self, node):
                base = node.value.id if isinstance(node.value, ast.Name) else None
                if node.attr in _NAMES and not (node.attr == "compile" and base == "re"):
                    found.append((rel, stack[-1], f"{base}.{node.attr}"))
                if (base, node.attr) in _ATTRS or (base == "os" and node.attr.startswith("exec")):
                    found.append((rel, stack[-1], f"{base}.{node.attr}"))
                self.generic_visit(node)

            def visit_Import(self, node):
                for a in node.names:
                    if a.name.split(".")[0] in _MODULES:
                        found.append((rel, stack[-1], f"import {a.name}"))

            def visit_ImportFrom(self, node):
                mod = (node.module or "").split(".")[0]
                if node.level == 0 and mod in _MODULES:
                    found.append((rel, stack[-1], f"from {node.module} import"))
                for a in node.names:
                    if a.name in _NAMES:
                        found.append((rel, stack[-1], f"from {node.module} import {a.name}"))

        Visitor().visit(ast.parse(path.read_text()))
        yield from found


def test_no_code_running_reference_outside_cellsession():
    assert [f for f in _findings() if (f[0], f[2]) not in ALLOWED] == []


def test_the_scan_catches_the_shapes_it_names(tmp_path, monkeypatch):
    """The scan is not vacuous: each named shape is found."""
    pkg = tmp_path / "lit"
    pkg.mkdir()
    (pkg / "bad.py").write_text(
        "import builtins, importlib\nimport subprocess as sp\nfrom runpy import run_path\n"
        "import types, os, re\n"
        "def a(src):\n    e = eval\n    builtins.exec(src)\n    exec(src)\n"
        "    compile(src, 'x', 'exec')\n    re.compile('ok')\n"
        "    types.FunctionType\n    os.system('x')\n    os.execv('x', [])\n"
        "    __import__('os')\n")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", pkg)
    what = {w for _, _, w in _findings()}
    assert {"eval", "builtins.exec", "exec", "compile", "__import__", "import builtins",
            "import importlib", "import subprocess", "from runpy import",
            "types.FunctionType", "os.system", "os.execv"} <= what
    assert "re.compile" not in what


def test_the_unwired_files_have_no_importer_in_src():
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        assert "literate.tools" not in text and "make_tool_namespace" not in text \
            or path.name in UNWIRED, path
