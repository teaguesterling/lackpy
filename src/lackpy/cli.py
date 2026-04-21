"""Command-line interface for lackpy."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any



def _parse_kit(kit_str: str) -> list[str]:
    """Parse --kit argument as a list of tool names.

    Always returns a list. Use 'lackpyctl kit info <name>' to query
    predefined kits by name instead.
    """
    return [k.strip() for k in kit_str.split(",")]


def _parse_tools(tools_str: str) -> list[str]:
    """Parse --tools argument as a comma-separated list of tool names."""
    return [t.strip() for t in tools_str.split(",") if t.strip()]


def _parse_params(param_list: list[str] | None) -> dict[str, str]:
    """Parse --param key=value flags into a dict."""
    if not param_list:
        return {}
    params = {}
    for p in param_list:
        key, _, value = p.partition("=")
        params[key.strip()] = value.strip()
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lackpy",
        description="lackpy — run Lackey files and delegate natural-language programs",
        epilog=(
            "Configuration & management:\n"
            "  Use lackpyctl for workspace init, kit/toolbox/template management,\n"
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
    parser.add_argument("--kit", default=None, help="Kit name, comma-separated list, or @file")
    parser.add_argument("--tools", default=None, help="Extra tool names (comma-separated) to add to the kit")
    parser.add_argument("--param", action="append", default=None, help="Parameter: key=value (repeatable)")
    parser.add_argument("--validate", action="store_true", default=False, help="Validate without running")
    parser.add_argument("--mode", default=None, help="Inference mode: 1-shot, spm (default: from config or legacy)")

    return parser


async def _run_file(svc: Any, path: Path, kit: list[str] | None, params: dict[str, str], sandbox: Any,
                    extra_tools: list[str] | None = None) -> dict[str, Any]:
    """Run a file — Lackey file or plain program (with --kit or --tools)."""
    content = path.read_text()
    if "Lackey" in content and "def run" in content:
        return await svc.run_lackey(path, params=params, sandbox=sandbox)
    elif kit or extra_tools:
        exec_result = await svc.run_program(content, kit=kit, params=params, extra_tools=extra_tools)
        return {"success": exec_result.success, "output": exec_result.output, "error": exec_result.error}
    else:
        return {"success": False, "error": "Specify --kit or --tools for plain program files, or use a Lackey file."}


def _file_entrypoint(raw_args: list[str]) -> int:
    """Handle `lackpy script.py [--kit ...] [--param k=v] [--validate] [--workspace ...]` invocation."""
    ep = argparse.ArgumentParser(prog="lackpy", add_help=False)
    ep.add_argument("file")
    ep.add_argument("--kit", default=None)
    ep.add_argument("--tools", default=None)
    ep.add_argument("--param", action="append", default=None)
    ep.add_argument("--validate", action="store_true", default=False)
    ep.add_argument("--workspace", type=Path, default=None)
    ep.add_argument("--sandbox", default=None)
    args, _ = ep.parse_known_args(raw_args)

    path = Path(args.file)
    if not path.exists():
        print(f"lackpy: file not found: {args.file}", file=sys.stderr)
        return 1

    workspace = args.workspace or Path.cwd()
    kit = _parse_kit(args.kit) if args.kit else None
    extra_tools = _parse_tools(args.tools) if args.tools else None
    params = _parse_params(args.param)

    from .service import LackpyService
    svc = LackpyService(workspace=workspace)

    if args.validate:
        program = path.read_text()
        result = svc.validate(program, kit=kit, extra_tools=extra_tools)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    result_dict = asyncio.run(_run_file(svc, path, kit, params, args.sandbox, extra_tools=extra_tools))
    print(json.dumps(result_dict, indent=2, default=str))
    return 0 if result_dict.get("success") else 1


def main(argv: list[str] | None = None) -> int:
    # Pre-parse: detect bare file argument before argparse sees it.
    # If the first non-flag token ends with .py (and isn't a flag), treat it as a file.
    raw_args = argv if argv is not None else sys.argv[1:]
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

    extra_tools = _parse_tools(args.tools) if args.tools else None

    # --validate + -c → validate the code string
    if args.validate and args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        kit = _parse_kit(args.kit) if args.kit else None
        result = svc.validate(args.intent, kit=kit, extra_tools=extra_tools)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    # Runner-style interface (-c flag)
    if args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        kit = _parse_kit(args.kit) if args.kit else None
        mode = getattr(args, 'mode', None)

        if args.create:
            gen = asyncio.run(svc.generate(args.intent, kit=kit, mode=mode, extra_tools=extra_tools))
            tools = kit if isinstance(kit, list) else []
            if extra_tools:
                tools = tools + extra_tools
            path = asyncio.run(svc.create_lackey(
                program=gen.program, name=args.name or "Generated",
                tools=tools,
                creation_log=[
                    {"role": "user", "content": args.intent},
                    {"role": "assistant", "content": gen.program, "accepted": True},
                ],
            ))
            print(f"Created {path}")
            return 0

        if args.generate:
            try:
                gen = asyncio.run(svc.generate(args.intent, kit=kit, mode=mode, extra_tools=extra_tools))
            except RuntimeError as e:
                print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
                return 1
            print(gen.program)
            return 0

        # Default: delegate (generate + run)
        try:
            result = asyncio.run(svc.delegate(args.intent, kit=kit, mode=mode, extra_tools=extra_tools))
        except RuntimeError as e:
            print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["success"] else 1

    # Stdin: read program and run if stdin is not a tty
    if not sys.stdin.isatty():
        program = sys.stdin.read()
        if program.strip():
            from .service import LackpyService
            svc = LackpyService(workspace=workspace)
            kit = _parse_kit(args.kit) if args.kit else None
            exec_result = asyncio.run(svc.run_program(program, kit=kit, extra_tools=extra_tools))
            out_dict = {"success": exec_result.success, "output": exec_result.output, "error": exec_result.error}
            print(json.dumps(out_dict, indent=2, default=str))
            return 0 if exec_result.success else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
