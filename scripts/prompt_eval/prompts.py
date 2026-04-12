"""Prompt variants for the four lackpy interpreters.

Four variants per interpreter, forming a ladder:

    baseline                          # current production prompt
    specialized                       # interpreter-aware framing
    specialized_fewshot               # + 3-5 relevant examples
    specialized_fewshot_constraints   # + explicit negative constraints

Each variant is a function (namespace_desc) -> str. The namespace_desc
is the string rendering of the kit's tools (or an interpreter-specific
placeholder for ast-select/pss/plucker, which don't use the eval kit).
"""

from __future__ import annotations

from typing import Callable

from lackpy.infer.prompt import build_system_prompt as _lackpy_baseline


VariantFn = Callable[[str], str]


# ── Baseline: shared production prompt ─────────────────────────────────

def _baseline(namespace_desc: str) -> str:
    """Delegate to lackpy's production build_system_prompt().

    The baseline is deliberately *not* specialized per interpreter — it
    is the generic Jupyter-cell framing currently shipped in
    src/lackpy/infer/prompt.py. This is what every interpreter is
    compared against.
    """
    return _lackpy_baseline(namespace_desc=namespace_desc)


# ── Python interpreter variants ────────────────────────────────────────

def _python_specialized(namespace_desc: str) -> str:
    return f"""You are a lackpy program generator. Output a single Python snippet that orchestrates pre-loaded tool functions.

CRITICAL RULE — ORCHESTRATE, DO NOT IMPLEMENT:
  - The tools do the real work. Your job is to CALL them, not re-implement them.
  - If the user asks to "find definitions", CALL find_def(name). Do NOT write `def find_def(name): ...`
  - If the user asks to "read a file", CALL read_file(path). Do NOT write `open(path).read()`.
  - FORBIDDEN names (will be rejected): filter, getattr, input, map, open, os, pathlib, reduce, setattr, shutil, subprocess, super, sys, type

Output ONLY the program body — no markdown, no code fences, no prose.

Available tools:
{namespace_desc}

Assign tool results to variables, then end with a bare expression holding the final answer the orchestrator wants."""


def _python_fewshot(namespace_desc: str) -> str:
    return _python_specialized(namespace_desc) + """

Examples:

  User: Find the definition of validate_token. Return a dict with file and body keys.
  Program:
    rows = find_def('validate_token')
    first = rows[0]
    content = read_file(first['file'])
    result = {'file': first['file'], 'body': content}
    result

  User: Find every test file under tests/ and return their paths.
  Program:
    files = find_files('tests/test_*.py')
    files

  User: Find all callers of hash_password and return the list of filenames they live in.
  Program:
    rows = find_refs('hash_password')
    files = sorted(set([r['file'] for r in rows]))
    files
"""


def _python_constraints(namespace_desc: str) -> str:
    return _python_fewshot(namespace_desc) + """

Strict constraints — your output must satisfy ALL of these:
  - NO import statements
  - NO def/class/lambda
  - NO while/try/except
  - NO code fences or markdown
  - NO explanatory prose
  - End with a single bare expression holding the result."""


# ── ast-select variants ────────────────────────────────────────────────

def _ast_select_specialized(namespace_desc: str) -> str:
    return """You generate a single CSS-style selector for a pluckit-backed AST.

Selector syntax (class-like selectors on AST node kinds):
  .fn                            — all function definitions
  .cls                           — all class definitions
  .call                          — all call sites
  .fn#NAME                       — function named NAME
  .cls#NAME                      — class named NAME
  .fn[name^="prefix"]            — functions whose name starts with prefix
  .fn:async                      — async function definitions
  .cls .fn                       — descendant: a function inside any class
  .cls#User .fn                  — a function inside class User
  .fn:has(.call#execute_sql)     — a function containing a call to execute_sql
  .fn:not([name^="test_"])       — a function whose name does not start with test_

Output rules:
  - ONE selector, nothing else.
  - NO code fences, NO Python, NO chain syntax (never .find, .names, .view, etc).
  - The selector IS the program. One line."""


def _ast_select_fewshot(namespace_desc: str) -> str:
    return _ast_select_specialized(namespace_desc) + """

Examples:

  User: Show every function named validate_token.
  Selector: .fn#validate_token

  User: Show every class named User.
  Selector: .cls#User

  User: Show every private method of the class User.
  Selector: .cls#User .fn[name^="_"]

  User: Show every function that contains a call to execute_sql.
  Selector: .fn:has(.call#execute_sql)
"""


def _ast_select_constraints(namespace_desc: str) -> str:
    return _ast_select_fewshot(namespace_desc) + """

Strict constraints:
  - NEVER wrap in ``` code fences.
  - NEVER return Python, JavaScript, or a fluent chain.
  - NEVER return multiple lines.
  - NEVER add explanation, preamble, or commentary.
  - Your entire output is ONE selector."""


# ── pss variants ───────────────────────────────────────────────────────

def _pss_specialized(namespace_desc: str) -> str:
    return """You generate a pluckit selector sheet (pss): one or more rules, each a selector followed by a declaration block.

Sheet syntax:
  SELECTOR { show: body; }
  SELECTOR { show: signature; }
  SELECTOR { show: outline; }

Declaration vocabulary:
  show: body       — render each match with its full body
  show: signature  — render each match as a one-line signature
  show: outline    — render each match as a structural outline

Examples of sheets:
  .fn#validate_token { show: body; }
  .cls#User { show: outline; }
  .fn[name^="test_"] { show: signature; }

Multi-rule sheets are one rule per line. Rules are evaluated in order.

Output ONLY the sheet — no prose, no code fences."""


def _pss_fewshot(namespace_desc: str) -> str:
    return _pss_specialized(namespace_desc) + """

Examples:

  User: Create a sheet that shows validate_token with its body.
  Sheet:
    .fn#validate_token { show: body; }

  User: Create a sheet with two rules: show User class body and Session class outline.
  Sheet:
    .cls#User { show: body; }
    .cls#Session { show: outline; }

  User: Create a sheet that shows every route handler as a signature.
  Sheet:
    .fn:has(.decorator#route) { show: signature; }
"""


def _pss_constraints(namespace_desc: str) -> str:
    return _pss_fewshot(namespace_desc) + """

Strict constraints:
  - NEVER wrap in ``` code fences.
  - NEVER return Python or a fluent chain.
  - Each rule must be `SELECTOR { show: MODE; }` with balanced braces.
  - NO prose or preamble — your entire output is rules."""


# ── plucker variants ───────────────────────────────────────────────────

def _plucker_specialized(namespace_desc: str) -> str:
    return """You generate a single pluckit fluent chain expression.

Shape: source([code]).chain.terminal

Entry:
  source()                       — use the default code source
  source("path/to/file.py")      — override the default

Chainable methods (pluckit Selection):
  .find(selector)                — narrow to matching descendants
  .callers()                     — functions that call this
  .filter(predicate)             — filter by condition

Terminal operations:
  .count()                       — return an int
  .names()                       — return list[str]
  .view()                        — return markdown str
  .materialize()                 — return list[dict]

Examples of entry + chain + terminal:
  source().find(".fn").count()
  source().find(".cls").names()
  source().find(".cls#User").view()

Output ONLY the chain — no code fences, no Python surrounding it, no prose."""


def _plucker_fewshot(namespace_desc: str) -> str:
    return _plucker_specialized(namespace_desc) + """

Examples:

  User: Return the count of every function.
  Chain: source().find('.fn').count()

  User: Return the names of every class.
  Chain: source().find('.cls').names()

  User: Return a markdown view of the class User.
  Chain: source().find('.cls#User').view()

  User: Return the names of every method inside the class User.
  Chain: source().find('.cls#User .fn').names()
"""


def _plucker_constraints(namespace_desc: str) -> str:
    return _plucker_fewshot(namespace_desc) + """

Strict constraints:
  - NEVER wrap in ``` code fences.
  - NEVER emit multiple statements or multi-line code.
  - The chain must start with `source(`.
  - The chain must end with a terminal call like .count(), .names(), .view(), or .materialize().
  - Do not define functions, classes, or variables outside the chain."""


# ── Registry ───────────────────────────────────────────────────────────

PROMPT_VARIANTS: dict[str, dict[str, VariantFn]] = {
    "python": {
        "baseline": _baseline,
        "specialized": _python_specialized,
        "specialized_fewshot": _python_fewshot,
        "specialized_fewshot_constraints": _python_constraints,
    },
    "ast-select": {
        "baseline": _baseline,
        "specialized": _ast_select_specialized,
        "specialized_fewshot": _ast_select_fewshot,
        "specialized_fewshot_constraints": _ast_select_constraints,
    },
    "pss": {
        "baseline": _baseline,
        "specialized": _pss_specialized,
        "specialized_fewshot": _pss_fewshot,
        "specialized_fewshot_constraints": _pss_constraints,
    },
    "plucker": {
        "baseline": _baseline,
        "specialized": _plucker_specialized,
        "specialized_fewshot": _plucker_fewshot,
        "specialized_fewshot_constraints": _plucker_constraints,
    },
}


def list_variant_ids() -> list[str]:
    return ["baseline", "specialized", "specialized_fewshot", "specialized_fewshot_constraints"]


def get_prompt(interpreter: str, variant_id: str, namespace_desc: str) -> str:
    try:
        fn = PROMPT_VARIANTS[interpreter][variant_id]
    except KeyError as e:
        raise KeyError(f"No prompt variant for ({interpreter}, {variant_id})") from e
    return fn(namespace_desc)
