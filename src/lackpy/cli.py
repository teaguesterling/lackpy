"""Command-line interface for lackpy."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .profiles import ProfileError
from .tools.toolbox import UnknownToolError



# Failures a caller should receive as an envelope rather than a traceback.
# RuntimeError is "all providers failed"; FileNotFoundError is an unresolvable
# profile (`--profile log` where log is a tool, or the default 'debug' profile
# that does not ship); UnknownToolError is an unknown tool name. All are ordinary
# misuse, and a caller parsing stdout should not have to scrape a stack trace to
# learn which.
#
# UnknownToolError rather than bare KeyError on purpose: this wraps the whole
# generate/delegate call, so catching KeyError here would turn any internal dict
# miss -- in inference, tool resolution, execution -- into an envelope reading
# {"error": "'somekey'"} with the traceback thrown away. A bug would be reported
# to the user as ordinary misuse. Anything added here must be a type raised
# deliberately for a user-facing condition, never a builtin an implementation
# error can also raise.
_ENVELOPE_ERRORS = (RuntimeError, FileNotFoundError, UnknownToolError, NotImplementedError,
                    ProfileError)


_TIMEOUT_HELP = ("The run's time limit in seconds (default: [sandbox] timeout_seconds, 120); "
                 "over the profile's max_timeout it is refused")

_TOOLS_REMOVED = "--tools was removed with the tool kits in lackpy stepy100; use pathlib.Path"


def _parse_params(param_list: list[str] | None) -> dict[str, str]:
    """Parse --param key=value flags into a dict."""
    if not param_list:
        return {}
    params = {}
    for p in param_list:
        key, _, value = p.partition("=")
        params[key.strip()] = value.strip()
    return params


def _coerce_params(raw: dict[str, str], defaults: dict[str, Any]) -> dict[str, Any]:
    """Type each ``--param`` text like the program's default (review I6).

    The runner compares a passed value's type with its default's. A key whose
    default is a ``str`` keeps the raw text, so ``--param name=5`` gives
    ``"5"``. Every other value is read with ``ast.literal_eval`` (``5``,
    ``2.5``, ``True``, ``[1, 2]``), and text that is not a literal stays a
    string (``--param p=src/*.py``).
    """
    out: dict[str, Any] = {}
    for key, text in raw.items():
        if type(defaults.get(key)) is str:
            out[key] = text
            continue
        try:
            out[key] = ast.literal_eval(text)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            out[key] = text
    return out


def _program_defaults(content: str) -> dict[str, Any]:
    """The defaults in a program's top-level ``params = {...}`` literal, or {}."""
    from .run.runner import _params_literal
    try:
        literal = _params_literal(ast.parse(content))
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return {}
    if literal is None:
        return {}
    return {k.value: ast.literal_eval(v) for k, v in zip(literal.keys, literal.values)}  # type: ignore[union-attr]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lackpy",
        description="lackpy — run Lackey files and delegate natural-language programs",
        epilog=(
            "Configuration & management:\n"
            "  Use lackpyctl for workspace init, profile/toolbox/template management,\n"
            "  and MCP server. Run 'lackpyctl --help' for details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="Workspace directory (default: cwd)",
    )
    parser.add_argument("-c", dest="intent", default=None, help="One-shot intent")
    parser.add_argument("--create", action="store_true", default=False, help="Save as Lackey file")
    parser.add_argument("--generate", action="store_true", default=False, help="Generate without running")
    parser.add_argument("--name", default=None, help="Class name for --create")
    parser.add_argument("--profile", default=None,
                        help="Profile name: a [profiles.<name>] table (model, mode, rules)")
    parser.add_argument("--param", action="append", default=None, help="Parameter: key=value (repeatable)")
    parser.add_argument("--validate", action="store_true", default=False, help="Validate without running")
    parser.add_argument("--mode", default=None, help="Inference mode: 1-shot, spm (default: from config or legacy)")
    parser.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                        help=_TIMEOUT_HELP)

    return parser


async def _run_file(svc: Any, path: Path, profile: str | None, params: dict[str, str],
                    sandbox: Any, timeout: float | None = None) -> dict[str, Any]:
    """Run a file — a Lackey file, or a plain program."""
    content = path.read_text()
    if "Lackey" in content and "def run" in content:
        defaults = {n: spec.get("default") for n, spec in svc.parse_lackey(path).params.items()}
        return await svc.run_lackey(path, params=_coerce_params(params, defaults), sandbox=sandbox,
                                    timeout=timeout)
    exec_result = await svc.run_program(content, profile=profile,
                                        params=_coerce_params(params, _program_defaults(content)),
                                        timeout=timeout)
    return {"success": exec_result.success, "output": exec_result.effective_output,
            "stdout": exec_result.stdout, "error": exec_result.error,
            "error_phase": exec_result.error_phase}


def _file_entrypoint(raw_args: list[str]) -> int:
    """Handle `lackpy script.py [--profile ...] [--param k=v] [--validate] [--workspace ...]` invocation."""
    ep = argparse.ArgumentParser(prog="lackpy", add_help=False)
    ep.add_argument("file")
    ep.add_argument("--profile", default=None)
    ep.add_argument("--param", action="append", default=None)
    ep.add_argument("--validate", action="store_true", default=False)
    ep.add_argument("--workspace", type=Path, default=None)
    ep.add_argument("--sandbox", default=None)
    ep.add_argument("--timeout", type=float, default=None)
    args, _ = ep.parse_known_args(raw_args)
    if any(a == "--tools" or a.startswith("--tools=") for a in raw_args):
        ep.error(_TOOLS_REMOVED)

    path = Path(args.file)
    if not path.exists():
        print(f"lackpy: file not found: {args.file}", file=sys.stderr)
        return 1

    workspace = args.workspace or Path.cwd()
    profile = args.profile
    params = _parse_params(args.param)

    from .service import LackpyService
    svc = LackpyService(workspace=workspace)

    if args.validate:
        program = path.read_text()
        result = svc.validate(program, profile=profile)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    result_dict = asyncio.run(_run_file(svc, path, profile, params, args.sandbox, args.timeout))
    print(json.dumps(result_dict, indent=2, default=str))
    return 0 if result_dict.get("success") else 1


def main(argv: list[str] | None = None) -> int:
    # Pre-parse: detect bare file argument before argparse sees it.
    # If the first non-flag token ends with .py (and isn't a flag), treat it as a file.
    raw_args = argv if argv is not None else sys.argv[1:]
    # `lackpy mcp [--workspace …]` → start the MCP server on stdio. Matches an external
    # consumer's `command: "lackpy", args: ["mcp"]` (e.g. woollama's mcp.json); the
    # dedicated `lackpy-mcp` console script is the canonical, decoupled form.
    if raw_args and raw_args[0] == "mcp":
        from .mcp.cli import mcp_main
        return mcp_main(raw_args[1:])
    first_positional = next(
        (a for a in raw_args if not a.startswith("-")), None
    )
    if first_positional and not any(a == '-c' for a in raw_args) and (first_positional.endswith(".py") or "/" in first_positional or first_positional.startswith(".")):
        # Looks like a file path — use the file entrypoint.
        # Skip when -c is present: -c explicitly declares an intent, which
        # may contain file paths as part of the natural language (issue #3).
        return _file_entrypoint(raw_args)

    parser = build_parser()
    args = parser.parse_args(argv)

    workspace = args.workspace or Path.cwd()

    # --validate + -c → validate the code string
    if args.validate and args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        result = svc.validate(args.intent, profile=args.profile)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    # Runner-style interface (-c flag)
    if args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        profile = args.profile
        mode = getattr(args, 'mode', None)

        if args.create:
            gen = asyncio.run(svc.generate(args.intent, profile=profile, mode=mode))
            path = asyncio.run(svc.create_lackey(
                program=gen.program, name=args.name or "Generated",
                tools=[],
                creation_log=[
                    {"role": "user", "content": args.intent},
                    {"role": "assistant", "content": gen.program, "accepted": True},
                ],
            ))
            print(f"Created {path}")
            return 0

        if args.generate:
            try:
                gen = asyncio.run(svc.generate(args.intent, profile=profile, mode=mode))
            except _ENVELOPE_ERRORS as e:
                # STDERR here, unlike the delegate path below. --generate's stdout
                # is program SOURCE, not an envelope, so `lackpy -c … --generate >
                # out.py` must not leave JSON in out.py. The delegate rationale --
                # one stream carries every outcome -- only holds where that stream
                # is already JSON. Here the artifact is on stdout, diagnostics on
                # stderr, status in the exit code, and an empty stdout with exit 1
                # is itself the unambiguous "no program" signal.
                print(json.dumps({"success": False, "error": str(e)}, indent=2),
                      file=sys.stderr)
                return 1
            print(gen.program)
            return 0

        # Default: delegate (generate + run)
        try:
            result = asyncio.run(svc.delegate(args.intent, profile=profile, mode=mode,
                                              timeout=args.timeout))
        except _ENVELOPE_ERRORS as e:
            # A caller parses one stream. Emitting the success envelope on stdout
            # and this one on stderr means anybody reading stdout records a failed
            # generation as empty output with no error -- a well-formed null result
            # indistinguishable from "the model produced nothing", which is a
            # different and much rarer thing. The exit code still says 1.
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["success"] else 1

    # Stdin: read program and run if stdin is not a tty
    if not sys.stdin.isatty():
        program = sys.stdin.read()
        if program.strip():
            from .service import LackpyService
            svc = LackpyService(workspace=workspace)
            exec_result = asyncio.run(svc.run_program(program, profile=args.profile,
                                                      timeout=args.timeout))
            out_dict = {"success": exec_result.success, "output": exec_result.effective_output,
                        "stdout": exec_result.stdout, "error": exec_result.error,
                        "error_phase": exec_result.error_phase}
            print(json.dumps(out_dict, indent=2, default=str))
            return 0 if exec_result.success else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
