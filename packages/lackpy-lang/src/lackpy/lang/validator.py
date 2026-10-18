"""AST validation pipeline for lackpy programs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from . import rules_safety  # noqa: F401  (registers the rules into RULES)
from .registry import RULES, Context, Violation, Waiver


@dataclass
class ValidationResult:
    """Result of validating a lackpy program.

    Attributes:
        valid: Whether the program passed all checks.
        errors: List of validation error messages.
        calls: Function names called in the program.
        variables: Variable names assigned in the program.
        violations: The failed checks, each with its rule id.
            ``errors == [v.message for v in violations]``.
        waivers: The ``# ste:`` waivers found in the program. Nothing parses
            waivers yet, so this list is empty for now.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    waivers: list[Waiver] = field(default_factory=list)


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

    ctx = Context(
        tree=tree,
        source=program,
        parents=parents,
        allowed_names=frozenset(allowed_names),
        defined_functions={},
        module_bindings={},
        member_bindings={},
    )

    violations: list[Violation] = []
    for rule in RULES:
        violations.extend(rule.check(ctx))

    # Custom rules: callables (ast.Module) -> list[str].
    if extra_rules:
        for extra in extra_rules:
            violations.extend(Violation("extra", None, msg) for msg in extra(tree))

    # ``calls`` drives the kibitzer planned-tool-call pre-check, so it holds only
    # named (Name-target) calls. Method calls (``data.split(...)``) are left out on
    # purpose: recording ``split``/``items`` there would surface them as ungranted
    # "tools".
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]

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
    )
