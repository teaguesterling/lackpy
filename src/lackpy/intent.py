"""The ``# /// intent`` block of a program or document (spec E5, E6).

The block follows PEP 723: ``# /// intent`` ... ``# ///``, with TOML inside.
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from typing import Any

from .interpreters.literate.parser import parse as parse_document
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


def parse_intent(source: str) -> Intent | None:
    """Parse the intent block in ``source``; None when there is none."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in source.splitlines():
        text = line.rstrip()
        if current is None:
            if text == _OPEN:
                current = []
        elif text == _CLOSE:
            blocks.append(current)
            current = None
        elif text.startswith("#"):
            current.append(text)
        else:
            raise IntentError("intent block: a line without '#' comes before the closing '# ///'.")
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
    for number, line in enumerate(program.splitlines()):
        text = line.strip()
        if text == _OPEN:
            return parse_intent(program)
        if text and not text.startswith("#"):
            return None
        if text.startswith("#!") and number != 0:
            return None
    return None


def document_intent(document: str) -> Intent | None:
    """The block in the first code cell of a document."""
    for cell in parse_document(document).cells:
        if cell.cell_type != "prose":
            return program_intent(cell.content)
    return None


def _matches(declared: str, value: Any) -> bool:
    if declared == "list[str]":
        return isinstance(value, list) and all(type(item) is str for item in value)
    return type(value) is {"str": str, "int": int, "float": float, "bool": bool}[declared]


def check_params(intent: Intent | None, program: str) -> list[str]:
    """Check ``[params]`` against the ``params`` literal of ``program``, both ways.

    Returns error texts, empty when the checks pass. Fills each ``Param.default``.
    A program with no block is valid.
    """
    if intent is None:
        return []
    try:
        literal = _params_literal(ast.parse(program))
    except SyntaxError as exc:
        return [f"The program does not parse, so its params cannot be checked: {exc.msg}."]
    if literal is None:
        if intent.params:
            return ["The intent block has [params], but the program has no top-level "
                    "'params' dict literal. Add one, or remove [params]."]
        return []
    defaults = {k.value: ast.literal_eval(v) for k, v in zip(literal.keys, literal.values)}
    declared = {p.name for p in intent.params}
    errors = [f"The 'params' literal has key {key!r}, which [params] does not describe. "
              "Add it to [params]." for key in defaults if key not in declared]
    for param in intent.params:
        if param.name not in defaults:
            errors.append(f"[params] describes {param.name!r}, but the 'params' literal "
                          "has no such key. Add it to the literal.")
            continue
        object.__setattr__(param, "default", defaults[param.name])
        if not _matches(param.type, defaults[param.name]):
            errors.append(f"The default of param {param.name!r} is {defaults[param.name]!r}, "
                          f"but [params] declares type {param.type}.")
    return errors
