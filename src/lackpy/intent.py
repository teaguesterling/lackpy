"""The ``# /// intent`` block of a program or document (spec E5, E6).

The block follows PEP 723: ``# /// intent`` ... ``# ///``, with TOML inside.
"""

from __future__ import annotations

import ast
import io
import tokenize
import tomllib
from dataclasses import dataclass, replace
from typing import Any

from .interpreters.literate import _SOURCE_CELLS
from .interpreters.literate.parser import first_line, parse as parse_document
from .run.runner import _params_literal

_OPEN = "# /// intent"
_CLOSE = "# ///"
_KEYS = {"goal", "reads", "writes", "checks", "output", "params"}
_OUTPUTS = ("none", "printed", "value")
_PARAM_TYPES = ("str", "int", "float", "bool", "list[str]")


class IntentError(ValueError):
    """The intent block is malformed."""


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    doc: str
    default: Any = None


@dataclass(frozen=True)
class Intent:
    goal: str = ""
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] | None = None  # None: the block has no `writes` key
    checks: tuple[str, ...] = ()
    output: str = "none"
    params: tuple[Param, ...] = ()


def _strings(key: str, value: Any) -> tuple[str, ...]:
    if not (isinstance(value, list) and all(type(item) is str for item in value)):
        raise IntentError(f"intent block: '{key}' must be a list of strings.")
    return tuple(value)


def _toml_text(lines: list[str]) -> str:
    return "\n".join(line[2:] if line.startswith("# ") else line[1:] for line in lines)


def _leading_comments(source: str) -> list[str]:
    """The lines before the first code line: a shebang, blank lines and comments."""
    lines = []
    for line in source.splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            break
        lines.append(line.rstrip())
    return lines


def _late_marker(source: str, after: int) -> int | None:
    """The line of the first ``# /// intent`` comment after line ``after``, or None.

    Comments only, by the tokenizer: a marker inside a string does not count.
    Source that does not tokenize gives None; its syntax error is reported
    when it is validated.
    """
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and tok.start[0] > after \
                    and tok.string.strip() == _OPEN:
                return tok.start[0]
    except (tokenize.TokenError, SyntaxError):
        pass
    return None


def parse_intent(source: str) -> Intent | None:
    """Parse the intent block in the leading comments of ``source``; None when there is none.

    A marker comment after the leading comments is an ``IntentError`` (final
    review 9): the block must be at the top, and is never silently ignored.
    """
    leading = _leading_comments(source)
    late = _late_marker(source, len(leading))
    if late is not None:
        raise IntentError(f"intent block: the '# /// intent' marker at line {late} comes "
                          "after code. Put the block at the top of the program, before any code.")
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for text in leading:
        if current is None:
            if text == _OPEN:
                current = []
            elif text.strip() == _OPEN:
                raise IntentError("intent block: the '# /// intent' marker must start at column 0.")
        elif text == _CLOSE:
            blocks.append(current)
            current = None
        else:
            current.append(text)
    if current is not None:
        raise IntentError("intent block: the closing '# ///' is missing.")
    if not blocks:
        return None
    if len(blocks) > 1:
        raise IntentError("intent block: a second block is not allowed. Keep one block.")
    try:
        data = tomllib.loads(_toml_text(blocks[0]))
    except tomllib.TOMLDecodeError as exc:
        raise IntentError(f"intent block: the TOML is not valid: {exc}") from exc
    for key in data:
        if key not in _KEYS:
            raise IntentError(
                f"intent block: unknown key '{key}'. Allowed keys: {', '.join(sorted(_KEYS))}.")
    goal = data.get("goal", "")
    if type(goal) is not str:
        raise IntentError("intent block: 'goal' must be a string.")
    output = data.get("output", "none")
    if output not in _OUTPUTS:
        raise IntentError(
            f"intent block: output {output!r} is not allowed. Use one of: {', '.join(_OUTPUTS)}.")
    params = []
    table = data.get("params", {})
    if not isinstance(table, dict):
        raise IntentError("intent block: 'params' must be a table.")
    for name, entry in table.items():
        if not (isinstance(entry, dict) and set(entry) == {"type", "doc"}
                and type(entry["doc"]) is str):
            raise IntentError(
                f"intent block: param '{name}' must be {{ type = ..., doc = ... }}.")
        if entry["type"] not in _PARAM_TYPES:
            raise IntentError(
                f"intent block: param '{name}' has type {entry['type']!r}. "
                f"Use one of: {', '.join(_PARAM_TYPES)}.")
        params.append(Param(name, entry["type"], entry["doc"], None))
    return Intent(
        goal=goal,
        reads=_strings("reads", data.get("reads", [])),
        writes=_strings("writes", data["writes"]) if "writes" in data else None,
        checks=_strings("checks", data.get("checks", [])),
        output=output,
        params=tuple(params),
    )


def program_intent(program: str) -> Intent | None:
    """The block at the top of a program: only a shebang, blank lines and comments may precede it."""
    return parse_intent(program)


def document_intent(document: str) -> Intent | None:
    """The block in the first code cell of a document.

    A marker comment in a later cell of program text is an ``IntentError``
    that names the cell (final review 9). The body of a ``@write`` cell is
    file text, so a marker there does not count.
    """
    cells = parse_document(document).cells
    first = next((i for i, cell in enumerate(cells) if cell.cell_type != "prose"), None)
    if first is None:
        return None
    for index in range(first + 1, len(cells)):
        cell = cells[index]
        if cell.cell_type in _SOURCE_CELLS and _late_marker(cell.content, 0) is not None:
            raise IntentError(f"intent block: the block is in cell {index} "
                              f"({first_line(cell)}). Put it in the first code cell.")
    return program_intent(cells[first].content)


def _matches(declared: str, value: Any) -> bool:
    if declared == "list[str]":
        return isinstance(value, list) and all(type(item) is str for item in value)
    # Exact types: a float param needs a float default, never an int or a bool
    # (Task 8 fix round 1), because the runner checks a value against its default's type.
    return type(value) is {"str": str, "int": int, "float": float, "bool": bool}[declared]


def _no_literal_message(tree: ast.Module) -> str:
    for stmt in tree.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) \
                and stmt.target.id == "params":
            return ("The 'params' assignment has an annotation. "
                    "Write params = {...} with no annotation.")
        if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "params" for t in stmt.targets):
            return ("The 'params' dict literal must hold only literal values, "
                    "with no ** entries and no calls.")
    return ("The intent block has [params], but the program has no top-level "
            "'params' dict literal. Add one, or remove [params].")


def check_params(intent: Intent | None, program: str) -> tuple[Intent | None, list[str]]:
    """Check ``[params]`` against the ``params`` literal of ``program``, both ways.

    Returns a new Intent whose Params carry their defaults, and the error texts
    (empty when the checks pass). The Intent passed in is not changed. A program
    with no block is valid: it returns (None, []).
    """
    if intent is None:
        return None, []
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        return intent, [f"The program does not parse, so its params cannot be checked: {exc.msg}."]
    literal = _params_literal(tree)
    if literal is None:
        return intent, [_no_literal_message(tree)] if intent.params else []
    defaults = {k.value: ast.literal_eval(v) for k, v in zip(literal.keys, literal.values)}
    declared = {p.name for p in intent.params}
    errors = [f"The 'params' literal has key {key!r}, which [params] does not describe. "
              "Add it to [params]." for key in defaults if key not in declared]
    filled = []
    for param in intent.params:
        if param.name not in defaults:
            errors.append(f"[params] describes {param.name!r}, but the 'params' literal "
                          "has no such key. Add it to the literal.")
            filled.append(param)
            continue
        filled.append(replace(param, default=defaults[param.name]))
        if not _matches(param.type, defaults[param.name]):
            errors.append(f"The default of param {param.name!r} is {defaults[param.name]!r}, "
                          f"but [params] declares type {param.type}.")
    return replace(intent, params=tuple(filled)), errors
