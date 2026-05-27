"""Guard: lackpy.lang is a pure leaf (RFC 0001).

It must never import 'upward' into the runtime, generator, policy, kit, service, or the
concrete interpreters — that upward edge is exactly what would make lackpy-lang
un-extractable. If this goes red, an upward import slipped in: move that code out of lang/
or invert the dependency. (Same role as workstream B's producer-driven contract test:
a tripwire on a boundary that's otherwise easy to erode silently.)
"""
import ast
from pathlib import Path

import lackpy.lang

LANG_DIR = Path(lackpy.lang.__file__).parent
FORBIDDEN = {"run", "kit", "policy", "infer", "service", "lackey", "prompts",
             "interpreters", "mcp"}


def _imported_modules(py: Path):
    tree = ast.parse(py.read_text(), filename=str(py))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield (node.module or ""), node.level
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0


def test_lang_is_a_leaf_no_upward_imports():
    violations = []
    for py in LANG_DIR.rglob("*.py"):
        for module, level in _imported_modules(py):
            parts = module.split(".") if module else []
            # absolute:  from lackpy.<forbidden> ...
            if len(parts) >= 2 and parts[0] == "lackpy" and parts[1] in FORBIDDEN:
                violations.append((py.name, module))
            # relative escape: from ..<forbidden> ...  (level>=2 leaves lang/)
            elif level >= 2 and parts and parts[0] in FORBIDDEN:
                violations.append((py.name, "." * level + module))
    assert not violations, (
        f"lackpy.lang must be a pure leaf, but found upward imports: {violations}"
    )
