"""AST validation pipeline for lackpy programs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from . import rules_safety  # noqa: F401  (registers the rules into RULES)
from . import rules_style_basic  # noqa: F401
from . import rules_style_def  # noqa: F401
from . import rules_style_try  # noqa: F401
from . import rules_style_shape  # noqa: F401
from .dictionary import MODULES
from .registry import RULES, Context, Violation, Waiver
from .waivers import apply_waivers


@dataclass
class ValidationResult:
    """Result of validating a lackpy program.

    Attributes:
        valid: Whether the program passed all checks.
        errors: List of validation error messages.
        calls: Function names called in the program. The list is advisory: best-effort
            named calls of builtins and allowed names, for the kibitzer pre-check. It is
            not a soundness boundary. A tool reached through a rebound name or passed
            as an argument (``key=read_file``) is not in it. Sound effect derivation
            is Plan 3's job.
        variables: Variable names assigned in the program.
        violations: The failed checks, each with its rule id.
            ``errors == [v.message for v in violations]``.
        waivers: The valid ``# ste:`` waivers in the program. ``used`` is True
            when a waiver removed a style violation, False when it matched none.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    waivers: list[Waiver] = field(default_factory=list)


def _collect_bindings(tree: ast.Module) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Read the approved top-level imports into (module_bindings, member_bindings).

    Only the module's own body counts. ``safety.import`` rejects an import
    anywhere else, and every unapproved form, so neither gives a binding.
    """
    modules: dict[str, str] = {}
    members: dict[str, tuple[str, str]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                if a.name in MODULES:
                    modules[a.asname or a.name] = a.name
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module in MODULES:
            for a in stmt.names:
                if a.name in MODULES[stmt.module]:
                    members[a.asname or a.name] = (stmt.module, a.name)
    return modules, members


def validate(
    program: str,
    allowed_names: set[str] | None = None,
    extra_rules: list | None = None,
) -> ValidationResult:
    """Validate a lackpy program string against the language grammar and namespace.

    Parses the program, builds one rule Context, and runs every registered
    rule in ``registry.RULES`` in order, then any extra custom rules.

    Args:
        program: The lackpy program source code to validate.
        allowed_names: Set of callable names permitted beyond the core builtins.
            Typically the tool names from a resolved kit.
        extra_rules: Additional rule callables ``(ast.Module) -> list[str]``
            applied after the built-in checks.

    Returns:
        A ValidationResult with valid=True if no errors were found.
    """
    if allowed_names is None:
        allowed_names = set()

    # safety.parse: no Context exists without a tree, so validate() emits it here.
    try:
        tree = ast.parse(program)
    except SyntaxError as e:
        v = Violation("safety.parse", e.lineno, f"Parse error: {e.msg} (line {e.lineno})")
        return ValidationResult(valid=False, errors=[v.message], violations=[v])

    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    module_bindings, member_bindings = _collect_bindings(tree)
    defined_functions: dict[str, int] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            defined_functions.setdefault(stmt.name, stmt.lineno)
    ctx = Context(
        tree=tree,
        source=program,
        parents=parents,
        allowed_names=frozenset(allowed_names),
        defined_functions=defined_functions,
        module_bindings=module_bindings,
        member_bindings=member_bindings,
    )

    violations: list[Violation] = []
    for rule in RULES:
        violations.extend(rule.check(ctx))

    # Waivers: a ``# ste:`` comment removes a style violation on its line.
    violations, waivers, meta = apply_waivers(violations, program)
    violations.extend(meta)

    # safety.parse, compile step: some programs pass every rule and still fail to
    # compile (two starred targets, a repeated parameter or keyword). Compile only a
    # program with no other violation, so the messages of a rejected program do not change.
    if not violations:
        try:
            compile(tree, "<lackpy>", "exec")
        except SyntaxError as e:
            violations.append(Violation("safety.parse", e.lineno,
                                        f"Compile error: {e.msg} (line {e.lineno})"))

    # Custom rules: callables (ast.Module) -> list[str].
    if extra_rules:
        for extra in extra_rules:
            violations.extend(Violation("extra", None, msg) for msg in extra(tree))

    # ``calls`` drives the kibitzer planned-tool-call pre-check, so it holds only
    # named (Name-target) calls. Method calls (``data.split(...)``) are left out on
    # purpose: recording ``split``/``items`` there would surface them as ungranted
    # "tools". Calls to import bindings (``Path(...)``) are left out for the same reason.
    # A call above its import is kept: at run time it reaches something else.
    # Calls to a program def are left out too (the kibitzer pre-check wants tools only).
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and not (node.func.id in member_bindings
                      and rules_safety.bound_before(tree, node.func.id, node))
             and not (node.func.id in defined_functions
                      and rules_style_def.defined_before(tree, node.func.id, node))]

    variables: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.append(target.id)

    return ValidationResult(
        valid=len(violations) == 0,
        errors=[v.message for v in violations],
        calls=calls,
        variables=variables,
        violations=violations,
        waivers=waivers,
    )
