"""The rule registry: every validator check is a rule with an id, a tier and examples.

A rule module (for example ``rules_safety``) registers its checks with the
``register`` decorator. ``validate()`` builds one ``Context`` per program and
runs ``RULES`` in registration order.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Literal

Tier = Literal["safety", "style"]


@dataclass(frozen=True)
class Violation:
    """One failed check. ``message`` is the text that ``ValidationResult.errors`` shows."""

    rule_id: str
    line: int | None
    message: str
    hint: str | None = None


@dataclass(frozen=True)
class Waiver:
    """A ``# ste: <rule-id>, <why>`` comment that waives a style rule on one line."""

    rule_id: str
    line: int
    why: str
    used: bool = False  # True when the waiver removed a violation


@dataclass
class Context:
    """What a rule check reads. ``validate()`` builds it once per program."""

    tree: ast.Module
    source: str
    parents: dict[int, ast.AST]  # id(child) -> parent
    allowed_names: frozenset[str]  # caller-supplied (today: kit tool names)
    defined_functions: dict[str, int]  # top-level def name -> line
    module_bindings: dict[str, str]  # local name -> dictionary module ("json", "pathlib", ...)
    member_bindings: dict[str, tuple[str, str]]  # local name -> (module, member), from `from m import x`


@dataclass(frozen=True)
class Rule:
    """A registered check, with the text and examples that document it."""

    id: str
    tier: Tier
    summary: str
    rationale: str
    bad: str
    good: str
    check: Callable[[Context], list[Violation]]


RULES: list[Rule] = []


def register(id: str, tier: Tier, summary: str, rationale: str, bad: str, good: str) -> Callable:
    """Register the decorated check as a rule. A duplicate id raises ValueError."""

    def decorator(check: Callable[[Context], list[Violation]]) -> Callable[[Context], list[Violation]]:
        if any(r.id == id for r in RULES):
            raise ValueError(f"duplicate rule id: {id!r}")
        RULES.append(Rule(id=id, tier=tier, summary=summary, rationale=rationale,
                          bad=bad, good=good, check=check))
        return check

    return decorator


def rule_by_id(rule_id: str) -> Rule:
    """Return the registered rule with this id. An unknown id raises KeyError."""
    for r in RULES:
        if r.id == rule_id:
            return r
    raise KeyError(rule_id)
