"""Anti-goal (literate stepy Task 2, design gate 5): no code in the literate
interpreter execs or evals document code. Every cell, interpolation and
inspection goes through ``CellSession``.

The check is static, over every file under ``src/lackpy/interpreters/literate/``
(ruling: an import allowlist, not a denylist of call shapes):

* **Imports.** Every module imported, by ``import``, ``from ... import``,
  ``importlib.import_module(...)`` or ``__import__(...)``, must be in
  :data:`ALLOWED_MODULES` (stdlib data and text modules, and ``lackpy`` itself).
  The never-allowed modules (:data:`NEVER_ALLOWED`) cannot be added there.
  A dynamic import call is never allowed.
* **Code-running names.** No ``exec``, ``eval`` or ``compile`` (bare, as an
  attribute or as an alias), and no ``__import__`` or ``__builtins__`` name. A
  ``.compile`` attribute is allowed only on a module imported as ``re``.
* **Dunder attributes.** No ``x.__dunder__`` attribute except ``__name__``
  and ``__init__`` (``super().__init__``): no ``f.__code__``, ``x.__globals__``, ``x.__subclasses__`` ...

An exception is LINE-scoped, never file-scoped: it names the file and the exact
source text of the one node it excuses, with a reason. Each exception must
match exactly one finding, so a stale or widened exception fails too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lackpy.interpreters.literate as literate

ROOT = Path(literate.__file__).parent
SRC = ROOT.parents[1]  # src/lackpy (a namespace package: no __file__)

#: Modules the literate package may import (top-level name; ``lackpy`` covers
#: every ``lackpy.*`` and relative import).
ALLOWED_MODULES = frozenset({
    "__future__", "ast", "collections", "contextlib", "dataclasses",
    "functools", "json", "markdown_it", "re", "time", "typing", "uuid",
    "lackpy",
})

#: Never allowed (ruling). A needed use is a line-scoped EXCEPTION instead.
NEVER_ALLOWED = frozenset({
    "sys", "os", "builtins", "types", "importlib", "code", "subprocess", "pickle",
    "pdb", "runpy", "timeit", "asyncio", "ctypes", "multiprocessing",
})

#: (file relative to ROOT, exact source text of the node) -> reason.
EXCEPTIONS = {
    ("kernel/static_analysis.py", "import builtins"):
        "dir(builtins) only: the builtin names check_cell resolves against",
    ("kernel/static_analysis.py", 'compile(code, "<cell>", "exec", ast.PyCF_ONLY_AST)'):
        "parse only (ast.PyCF_ONLY_AST): no code object is made",
    ("kernel/recovery.py", "import asyncio"):
        "runs the host's own inference coroutine (a model call), never document code",
    ("kernel/recovery.py", "import concurrent.futures"):
        "a thread for that inference coroutine when a loop is already running",
}

_CODE_NAMES = {"exec", "eval", "compile", "__import__", "__builtins__"}
_DUNDERS_OK = {"__name__", "__init__"}  # super().__init__(...) in persistence.py


def check(root: Path) -> list[tuple[str, str, str]]:
    """``(file, source text, what)`` for each finding under ``root``, before exceptions."""
    findings: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        text = path.read_text()
        tree = ast.parse(text)
        re_aliases = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                      for a in n.names if a.name == "re"}

        def add(node: ast.AST, what: str) -> None:
            findings.append((rel, ast.get_source_segment(text, node) or "", what))

        def module_ok(name: str) -> bool:
            top = name.split(".")[0]
            return top in ALLOWED_MODULES and top not in NEVER_ALLOWED

        # A called code name is reported with its whole call's text, so an
        # exception names the exact call.
        called: set[int] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in _CODE_NAMES):
                called.add(id(node.func))
                add(node, node.func.id)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if not module_ok(a.name):
                        add(node, f"import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and not module_ok(node.module or ""):
                    add(node, f"from {node.module} import")
            elif isinstance(node, ast.Name) and node.id in _CODE_NAMES and id(node) not in called:
                add(node, node.id)
            elif isinstance(node, ast.Attribute):
                base = node.value.id if isinstance(node.value, ast.Name) else None
                if node.attr in _CODE_NAMES and not (node.attr == "compile" and base in re_aliases):
                    add(node, f"{base}.{node.attr}")
                elif node.attr == "import_module":
                    add(node, f"{base}.import_module")
                elif (node.attr.startswith("__") and node.attr.endswith("__")
                      and node.attr not in _DUNDERS_OK):
                    add(node, f".{node.attr}")
    return findings


def _unexcused(root: Path, exceptions: dict) -> tuple[list, list]:
    found = check(root)
    used: dict[tuple[str, str], int] = {}
    unexcused = []
    for f in found:
        key = (f[0], f[1])
        if key in exceptions:
            used[key] = used.get(key, 0) + 1
        else:
            unexcused.append(f)
    bad_exceptions = [k for k in exceptions if used.get(k, 0) != 1]
    return unexcused, bad_exceptions


def test_no_code_running_reference_outside_cellsession():
    unexcused, bad_exceptions = _unexcused(ROOT, EXCEPTIONS)
    assert unexcused == []
    assert bad_exceptions == []  # each exception excuses exactly one node


def test_the_allowlist_holds_no_never_allowed_module():
    assert ALLOWED_MODULES & NEVER_ALLOWED == set()


def test_every_exception_has_a_reason():
    assert all(reason.strip() for reason in EXCEPTIONS.values())


def test_the_tool_layer_is_gone():
    """D7: the kit tools and their grade table are deleted, and nothing imports them."""
    assert not (ROOT / "tools.py").exists()
    assert not (ROOT / "tool_effects.toml").exists()
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        assert "literate.tools" not in text and "make_tool_namespace" not in text, path
        assert "from .tools import" not in text and "from ..tools import" not in text, path


PLANTED = {
    "getattr builtins": "import builtins\ngetattr(builtins, 'ex'+'ec')('1')\n",
    "__builtins__ subscript": "__builtins__['exec']('1')\n",
    "compile attr": "import ast\nast.compile\nx.compile('1','x','exec')\n",
    "code.InteractiveInterpreter": "import code\ncode.InteractiveInterpreter().runsource('1')\n",
    "from code import": "from code import InteractiveInterpreter\n",
    "sys.modules builtins getattr": "import sys\ngetattr(sys.modules['builtins'], 'ex'+'ec')('1')\n",
    "vars(sys.modules[...])": "import sys\nvars(sys.modules['builtins'])['ev'+'al']('1')\n",
    "from os import system": "from os import system\nsystem('id')\n",
    "import os as o": "import os as o\no.system('id')\n",
    "from types import FunctionType": "from types import FunctionType, CodeType\n",
    "os.posix_spawn/spawnl": "import os\nos.posix_spawn('x',[],{})\nos.spawnl(0,'x')\n",
    "timeit": "import timeit\ntimeit.timeit('1')\n",
    "pickle": "import pickle\npickle.loads(b'')\n",
    "asyncio subprocess": "import asyncio\nasyncio.create_subprocess_shell('id')\n",
    "pdb": "import pdb\npdb.run('1')\n",
    "func.__code__.replace": "def f(): pass\ntype(f)(f.__code__.replace(), {})\n",
    "operator attrgetter on builtins":
        "import operator, sys\noperator.attrgetter('ex'+'ec')(sys.modules['builtins'])\n",
    "importlib.import_module": "import importlib\nimportlib.import_module('os')\n",
    "__import__ call": "__import__('os')\n",
}


def test_every_planted_shape_is_caught(tmp_path):
    """The re-review's shapes (rr4), and the dynamic imports: the same
    checker, run over a planted file, catches each one."""
    missed = []
    for name, shape in PLANTED.items():
        d = tmp_path / name.replace(" ", "_").replace("/", "_")
        d.mkdir()
        (d / "m.py").write_text(shape)
        if not check(d):
            missed.append(name)
    assert missed == []


def test_an_exception_excuses_one_line_not_a_file(tmp_path):
    (tmp_path / "m.py").write_text("import sys\nimport ast\nx = ast.parse('1').__class__\n")
    unexcused, bad = _unexcused(tmp_path, {("m.py", "import sys"): "planted"})
    assert bad == []
    assert [w for *_, w in unexcused] == [".__class__"]
    # A second, identical line is not covered by the one exception.
    (tmp_path / "m.py").write_text("import sys\nimport sys\n")
    _, bad = _unexcused(tmp_path, {("m.py", "import sys"): "planted"})
    assert bad == [("m.py", "import sys")]


def test_re_compile_is_not_code_compile(tmp_path):
    (tmp_path / "m.py").write_text("import re\nR = re.compile('x')\n")
    assert check(tmp_path) == []
    (tmp_path / "m.py").write_text("import ast as re2\nR = re2.compile('x')\n")
    assert [w for *_, w in check(tmp_path)] == ["re2.compile"]
