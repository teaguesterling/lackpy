"""The ``stepy`` command (spec E8-E11): run a program under its intent block.

``stepy script.py [options]``, ``stepy -c 'program text' [options]`` and a
``#!/usr/bin/env stepy`` first line all run a program. Every option after the
script belongs to it: stepy's own (``--help``, ``--check``, ``--timeout``,
``--profile``, ``--workspace``) and one ``--<key>`` per ``[params]`` key.
Before a run, the intent block, the ``[params]`` match, the program and the
declared writes are checked; an error there exits 2 and nothing runs.

stdout carries only what the program printed. A failure writes ``stepy: `` and
the error to stderr. Exit codes: 0 success; 1 failure or refusal; 2 usage or
validation error, nothing ran; 124 timeout.
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .intent import Intent, IntentError, Param, check_params, program_intent
from .run.refusal import LackpyRefusal

_OWN = ("--help", "--check", "--timeout", "--profile", "--workspace")
_USAGE = ("usage: stepy SCRIPT.py [--help] [--check] [--timeout SECONDS] [--profile NAME] "
          "[--workspace DIR] [--<param> VALUE ...]\n"
          "       stepy -c 'program text' [options]")
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_UNSAFE = frozenset("*?[]")  # a filled value may not widen a glob (Task 7 review)


class _Usage(Exception):
    """A usage or validation error: exit 2, and nothing ran."""


@dataclass
class _Args:
    values: dict[str, Any] = field(default_factory=dict)  # only the params given
    help: bool = False
    check: bool = False
    timeout: float | None = None
    profile: str | None = None
    workspace: Path | None = None


def _options(intent: Intent | None) -> dict[str, tuple[Param, bool]]:
    """Each parameter option and what it sets: ``--key`` and, for a bool, ``--no-key``.

    A key whose option is one of stepy's own, or another key's, is an IntentError.
    """
    options: dict[str, tuple[Param, bool]] = {}
    for param in intent.params if intent else ():
        names = [(f"--{param.name}", True)]
        if param.type == "bool":
            names.append((f"--no-{param.name}", False))
        for name, value in names:
            if name in _OWN:
                raise IntentError(f"intent block: param '{param.name}' gives the option {name}, "
                                  "which stepy uses. Rename the param.")
            if name in options:
                raise IntentError(f"intent block: params '{options[name][0].name}' and "
                                  f"'{param.name}' both give the option {name}. Rename one.")
            options[name] = (param, value)
    return options


def _convert(name: str, param: Param, text: str) -> Any:
    try:
        return {"int": int, "float": float}.get(param.type, str)(text)
    except ValueError:
        raise _Usage(f"{name} takes a value of type {param.type}, not {text!r}.") from None


def _parse(intent: Intent | None, argv: list[str]) -> _Args:
    """The options after the script: no positional arguments, no prefix matching."""
    options = _options(intent)
    args, seen = _Args(), set()
    items = iter(argv)
    for arg in items:
        if arg in _OWN or arg in options:
            param = options[arg][0] if arg in options else None
            key = param.name if param else arg
            if key in seen and not (param and param.type == "list[str]"):
                raise _Usage(f"{arg} is given twice. Give each option once.")
            seen.add(key)
        if arg in ("--help", "--check"):
            setattr(args, arg[2:], True)
            continue
        if arg not in _OWN and arg not in options:
            if not arg.startswith("-"):
                raise _Usage(f"{arg!r} is a positional argument, and stepy takes none. "
                             "Give each parameter as --<key> VALUE.")
            known = ", ".join([*_OWN, *options])
            raise _Usage(f"{arg}: this script has no such option. Its options are: {known}.")
        param, flag = options.get(arg, (None, True))
        if param is not None and param.type == "bool":
            args.values[param.name] = flag
            continue
        text = next(items, None)
        if text is None or text.startswith("--"):
            raise _Usage(f"{arg} needs a value.")
        if arg == "--timeout":
            try:
                args.timeout = float(text)
            except ValueError:
                raise _Usage(f"--timeout takes a number of seconds, not {text!r}.") from None
        elif arg == "--profile":
            args.profile = text
        elif arg == "--workspace":
            args.workspace = Path(text)
        elif param.type == "list[str]":
            args.values.setdefault(param.name, []).append(text)
        else:
            args.values[param.name] = _convert(arg, param, text)
    return args


def _fill(glob: str, intent: Intent, values: dict[str, Any]) -> str:
    """``glob`` with each ``{name}`` filled in from the final parameter values (spec E7)."""
    params = {p.name: p for p in intent.params}

    def value(match: re.Match) -> str:
        name = match.group(1)
        if name not in params:
            raise _Usage(f"the declared writes glob {glob!r} names {{{name}}}, which is not a "
                         "parameter. Add it to [params], or remove it from writes.")
        if params[name].type not in ("str", "int", "float"):
            raise _Usage(f"the declared writes glob {glob!r} names {{{name}}}, a "
                         f"{params[name].type} parameter. Only a str, int or float can fill a glob.")
        text = str(values[name])
        reason = None
        if text in ("", ".", ".."):
            reason = "it is empty" if not text else f"it is {text!r}"
        elif ".." in re.split(r"[/\\]", text):
            reason = "it has a '..' segment"
        elif _UNSAFE & set(text):
            reason = f"it has the glob character {sorted(_UNSAFE & set(text))[0]!r}"
        if reason:
            raise _Usage(f"--{name} {text!r} cannot fill the declared writes glob {glob!r}: "
                         f"{reason}. Give a plain name.")
        return text

    return _PLACEHOLDER.sub(value, glob)


def _help(intent: Intent | None, script: str) -> str:
    if intent is None:
        return f"{script}: this script declares nothing, so it may write nothing.\n\n{_USAGE}\n"
    lines = [intent.goal or f"{script}: the intent block has no goal.", ""]
    if intent.params:
        lines.append("parameters:")
        for p in intent.params:
            lines.append(f"  --{p.name} {p.type} (default: {p.default!r}) {p.doc}")
    lines.append("writes: declared, enforced")
    lines += [f"  {g}" for g in intent.writes or ("none: no write is allowed",)]
    lines.append("reads: declared, not checked")
    lines += [f"  {g}" for g in intent.reads or ("none",)]
    lines.append("checks:")
    lines += [f"  {c}" for c in intent.checks or ("none",)]
    lines += ["", "--timeout SECONDS is the time limit for the whole run.", _USAGE]
    return "\n".join(lines) + "\n"


def _source(argv: list[str]) -> tuple[str, str, list[str]]:
    """The script's name, the program text, and the options after it."""
    if not argv:
        raise _Usage("give a script to run.")
    if argv[0] == "-c":
        if len(argv) < 2:
            raise _Usage("-c needs the program text.")
        return "-c", argv[1], argv[2:]
    path = Path(argv[0])
    if path.suffix == ".md":
        raise _Usage(f"{path}: running a .md document is not available yet. Use a .py program.")
    if path.suffix != ".py":
        raise _Usage(f"{path}: stepy runs a .py program or a .md document, "
                     f"not a file with the extension {path.suffix or '(none)'!r}.")
    if not path.is_file():
        raise _Usage(f"{path}: the file does not exist.")
    return str(path), path.read_text(), argv[1:]


def _prepare(service: Any, intent: Intent, program: str, args: _Args) -> tuple[dict, list[str]]:
    """The checks before a run, shared with ``--check``: the program, then the declared writes.

    Returns the final parameter values and the declared writes. Raises ``_Usage``.
    """
    from .service import validation_error
    validation = service.validate(program, profile=args.profile)
    if not validation.valid:
        raise _Usage(validation_error(validation))
    values = {p.name: p.default for p in intent.params} if intent else {}
    values.update(args.values)
    globs = [_fill(g, intent, values) for g in intent.writes or ()] if intent else []
    from .run.paths import with_write_globs
    try:
        policy = service._load_path_policy(read_only=False)
    except (Exception, LackpyRefusal):
        return values, globs  # the run reports a policy that cannot be loaded
    try:
        with_write_globs(policy, globs)
    except LackpyRefusal as e:  # a glob that leaves the workspace (Task 7 review)
        raise _Usage(f"refused: {e.what}.") from None
    return values, globs


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--help"]:
        print(_USAGE)
        return 0
    try:
        script, program, rest = _source(argv)
        intent, errors = check_params(program_intent(program), program)
        if errors:
            raise _Usage(" ".join(errors))
        args = _parse(intent, rest)
        if args.help:
            sys.stdout.write(_help(intent, script))
            return 0
        from .service import LackpyService
        service = LackpyService(workspace=args.workspace or Path.cwd())
        values, globs = _prepare(service, intent, program, args)
    except (_Usage, IntentError) as e:
        print(f"stepy: {e}", file=sys.stderr)
        if not argv:
            print(_USAGE, file=sys.stderr)
        return 2
    if args.check:
        return 0
    result = asyncio.run(service.run_program(
        program, profile=args.profile, params=values or None, timeout=args.timeout,
        declared_writes=globs))
    sys.stdout.write(result.printed)
    sys.stdout.flush()
    if result.success:
        return 0
    print(f"stepy: {result.error}", file=sys.stderr)
    return 124 if result.error_phase == "timeout" else 1
