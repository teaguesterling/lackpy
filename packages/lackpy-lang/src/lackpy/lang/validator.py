"""AST validation pipeline for lackpy programs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .grammar import (
    ALLOWED_NODES,
    FORBIDDEN_NAMES,
    ALLOWED_BUILTINS,
    DENIED_ATTRIBUTES,
)


@dataclass
class ValidationResult:
    """Result of validating a lackpy program.

    Attributes:
        valid: Whether the program passed all checks.
        errors: List of validation error messages.
        calls: Function names called in the program.
        variables: Variable names assigned in the program.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)


def validate(
    program: str,
    allowed_names: set[str] | None = None,
    extra_rules: list | None = None,
) -> ValidationResult:
    """Validate a lackpy program string against the language grammar and namespace.

    Parses the program and runs a multi-step pipeline: AST node allowlist,
    forbidden name check, namespace check, for-loop iter check, dunder string
    check, and any extra custom rules.

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

    all_allowed_calls = ALLOWED_BUILTINS | allowed_names
    errors: list[str] = []
    calls: list[str] = []
    variables: list[str] = []

    # Step 1: Parse
    try:
        tree = ast.parse(program)
    except SyntaxError as e:
        return ValidationResult(
            valid=False,
            errors=[f"Parse error: {e.msg} (line {e.lineno})"],
        )

    # Build parent map for lambda restriction check (step 5.5)
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node

    # Step 2: Node walk — reject disallowed node types
    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in ALLOWED_NODES:
            errors.append(
                f"Forbidden AST node: {node_type.__name__}"
                + (f" at line {node.lineno}" if hasattr(node, "lineno") else "")
            )

    # Step 3: Name check — reject forbidden names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"Forbidden name: '{node.id}' at line {node.lineno}")

    # Step 3.5: Attribute check — the sandbox boundary (GHSA-hpcj-3c97-43jm).
    #
    # ast.Attribute is allowed (data objects legitimately expose methods like
    # ``.split``/``.items``/``.get``), but attribute *names* are restricted so an
    # attribute chain cannot walk from a benign value to a type or module object:
    #   - any name starting with "_" is denied — this covers the entire dunder
    #     traversal escape family (``__class__``/``__bases__``/``__subclasses__``/
    #     ``__globals__``/``__mro__``/``__init__``/``__getattribute__``/…) as well
    #     as single-underscore private gadgets (e.g. ``()._module``).
    #   - a small set of non-underscore frame/generator internals (``f_globals``,
    #     ``gi_frame``, …) is denied explicitly (see DENIED_ATTRIBUTES).
    # This is an allow-by-shape / deny-by-name rule: it does not enumerate every
    # safe attribute (that set is open-ended for arbitrary data), so it is a
    # denylist on the dangerous shape. The known-complete escape surface in this
    # subset is underscore-prefixed names; the explicit frame set covers the only
    # non-underscore internals that expose a namespace.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_") or attr in DENIED_ATTRIBUTES:
                errors.append(
                    f"Forbidden attribute access: '.{attr}' at line {node.lineno} "
                    f"(private/dunder/internal attribute access is not permitted)"
                )

    # Step 4: Namespace check — reject calls to unknown / unresolvable functions.
    #
    # Every ast.Call is inspected (not only ``func is ast.Name``): a call whose
    # target is an attribute (``data.split(...)``) is permitted only because the
    # attribute name already passed Step 3.5; a call whose target is a Subscript
    # or another Call/expression is default-denied here — those exotic call forms
    # are not part of the intended subset and were an unchecked bypass before.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            calls.append(name)
            if name not in all_allowed_calls and name not in FORBIDDEN_NAMES:
                errors.append(
                    f"Unknown function: '{name}' at line {node.lineno} "
                    f"(not in kit or builtins)"
                )
        elif isinstance(func, ast.Attribute):
            # Method call on a value; Step 3.5 already vetted the attribute name.
            # Intentionally NOT recorded in ``calls``: that list drives the
            # kibitzer planned-tool-call pre-check, which must see only named
            # (Name-target) tool/builtin calls — recording method names like
            # ``split``/``items`` there would surface them as ungranted "tools".
            pass
        else:
            # Call of a subscript / call result / other expression — default-deny.
            errors.append(
                f"Unsupported call target ({type(func).__name__}) at line "
                f"{node.lineno}: only named functions and method calls are allowed"
            )

    # Step 5: For-loop check — must iterate over call result or variable
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            it = node.iter
            if not isinstance(it, (ast.Call, ast.Name)):
                errors.append(
                    f"For-loop at line {node.lineno} must iterate over "
                    f"a function call or variable"
                )

    # Step 5.5: Lambda check — only allowed as key= keyword argument value
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            parent = parent_map.get(id(node))
            if not (isinstance(parent, ast.keyword) and parent.arg == "key"):
                errors.append(
                    f"Lambda at line {node.lineno} is only allowed as key= argument "
                    f"(e.g., key=lambda x: x['field'])"
                )

    # Step 6: String literal check — no dunder strings
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                errors.append(
                    f"String containing '__' at line {node.lineno}: "
                    f"dunder access via string is forbidden"
                )

    # Step 7: Custom rules
    if extra_rules:
        for rule in extra_rules:
            rule_errors = rule(tree)
            errors.extend(rule_errors)

    # Collect assigned variable names
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.append(target.id)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        calls=calls,
        variables=variables,
    )
