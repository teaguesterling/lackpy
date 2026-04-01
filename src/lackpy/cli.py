"""Command-line interface for lackpy."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


# Subcommands that have been migrated to lackpyctl
_CTL_COMMANDS = frozenset({"init", "status", "spec", "kit", "toolbox", "template"})


def _parse_kit(kit_str: str) -> list[str]:
    """Parse --kit argument as a list of tool names.

    Always returns a list. Use 'lackpyctl kit info <name>' to query
    predefined kits by name instead.
    """
    return [k.strip() for k in kit_str.split(",")]


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
    parser.add_argument("--param", action="append", default=None, help="Parameter: key=value (repeatable)")
    parser.add_argument("--validate", action="store_true", default=False, help="Validate without running")

    subparsers = parser.add_subparsers(dest="command")

    # delegate
    delegate_p = subparsers.add_parser("delegate", help="[deprecated: use -c] Generate and run a program from intent")
    delegate_p.add_argument("intent", help="Natural language intent")
    delegate_p.add_argument("--kit", default=None, help="Kit name, comma-separated list, or @file")
    delegate_p.add_argument("--sandbox", default=None, help="Sandbox profile")

    # generate
    generate_p = subparsers.add_parser("generate", help="[deprecated: use -c --generate] Generate a program from intent without running")
    generate_p.add_argument("intent", help="Natural language intent")
    generate_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")

    # run
    run_p = subparsers.add_parser("run", help="[deprecated: use lackpy file.py] Run a program file")
    run_p.add_argument("file", help="Program file to run")
    run_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")

    # create
    create_p = subparsers.add_parser("create", help="[deprecated: use -c --create] Validate and save a program as a template")
    create_p.add_argument("file", help="Program file to save as template")
    create_p.add_argument("--name", required=True, help="Template name")
    create_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")
    create_p.add_argument("--pattern", default=None, help="Intent pattern regex")

    # validate (subcommand — deprecated in favour of --validate flag)
    validate_p = subparsers.add_parser("validate", help="[deprecated: use --validate] Validate a program without running")
    validate_p.add_argument("file", help="Program file to validate")
    validate_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")

    # spec
    subparsers.add_parser("spec", help="[deprecated: use lackpyctl spec] Print language spec")

    # status
    subparsers.add_parser("status", help="[deprecated: use lackpyctl status] Show lackpy status and configuration")

    # kit
    kit_p = subparsers.add_parser("kit", help="[deprecated: use lackpyctl kit] Manage kits")
    kit_sub = kit_p.add_subparsers(dest="kit_command")

    kit_sub.add_parser("list", help="List available kits")

    kit_info_p = kit_sub.add_parser("info", help="Show kit info")
    kit_info_p.add_argument("name", help="Kit name or comma-separated tools")
    kit_info_p.add_argument("--tools", nargs="+", default=None, help="Tool names")

    kit_create_p = kit_sub.add_parser("create", help="Create a new kit")
    kit_create_p.add_argument("name", help="Kit name")
    kit_create_p.add_argument("--tools", nargs="+", required=True, help="Tool names to include")
    kit_create_p.add_argument("--description", default=None, help="Kit description")

    # toolbox
    toolbox_p = subparsers.add_parser("toolbox", help="[deprecated: use lackpyctl toolbox] Manage toolbox")
    toolbox_sub = toolbox_p.add_subparsers(dest="toolbox_command")

    toolbox_sub.add_parser("list", help="List all registered tools")

    toolbox_show_p = toolbox_sub.add_parser("show", help="Show tool details")
    toolbox_show_p.add_argument("name", help="Tool name")

    # template
    template_p = subparsers.add_parser("template", help="[deprecated: use lackpyctl template] Manage templates")
    template_sub = template_p.add_subparsers(dest="template_command")

    template_sub.add_parser("list", help="List available templates")

    template_test_p = template_sub.add_parser("test", help="Test a template")
    template_test_p.add_argument("name", help="Template name")

    # init
    init_p = subparsers.add_parser("init", help="[deprecated: use lackpyctl init] Initialize .lackpy workspace")
    init_p.add_argument("--ollama-model", default="qwen2.5-coder:1.5b", help="Default Ollama model")
    init_p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")

    return parser


def _init_config(workspace: Path, ollama_model: str, ollama_url: str = "http://localhost:11434") -> None:
    config_dir = workspace / ".lackpy"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "templates").mkdir(exist_ok=True)
    (config_dir / "kits").mkdir(exist_ok=True)
    config_file = config_dir / "config.toml"
    if config_file.exists():
        print(f"Config already exists at {config_file}", file=sys.stderr)
        return
    config_file.write_text(f"""\
[inference]
order = ["templates", "rules", "ollama-local"]

[inference.providers.ollama-local]
plugin = "ollama"
host = "{ollama_url}"
model = "{ollama_model}"

[kit]
default = "debug"

[sandbox]
enabled = false
timeout_seconds = 120
memory_mb = 512
""")
    print(f"Initialized lackpy workspace at {config_dir}")


async def _run_file(svc: Any, path: Path, kit: list[str] | None, params: dict[str, str], sandbox: Any) -> dict[str, Any]:
    """Run a file — Lackey file or plain program (with --kit)."""
    content = path.read_text()
    if "Lackey" in content and "def run" in content:
        return await svc.run_lackey(path, params=params, sandbox=sandbox)
    elif kit:
        exec_result = await svc.run_program(content, kit=kit, params=params)
        return {"success": exec_result.success, "output": exec_result.output, "error": exec_result.error}
    else:
        return {"success": False, "error": "Specify --kit for plain program files, or use a Lackey file."}


def _file_entrypoint(raw_args: list[str]) -> int:
    """Handle `lackpy script.py [--kit ...] [--param k=v] [--validate] [--workspace ...]` invocation."""
    ep = argparse.ArgumentParser(prog="lackpy", add_help=False)
    ep.add_argument("file")
    ep.add_argument("--kit", default=None)
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
    params = _parse_params(args.param)

    from .service import LackpyService
    svc = LackpyService(workspace=workspace)

    if args.validate:
        program = path.read_text()
        result = svc.validate(program, kit=kit)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    result_dict = asyncio.run(_run_file(svc, path, kit, params, args.sandbox))
    print(json.dumps(result_dict, indent=2, default=str))
    return 0 if result_dict.get("success") else 1


def main(argv: list[str] | None = None) -> int:
    # Pre-parse: detect bare file argument before argparse sees it.
    # If the first non-flag token ends with .py (and isn't a flag), treat it as a file.
    raw_args = argv if argv is not None else sys.argv[1:]
    first_positional = next(
        (a for a in raw_args if not a.startswith("-")), None
    )
    if first_positional and (first_positional.endswith(".py") or "/" in first_positional or first_positional.startswith(".")):
        # Looks like a file path — use the file entrypoint
        return _file_entrypoint(raw_args)

    # Also detect stdin piping (non-tty stdin with no other args)
    if not raw_args and not sys.stdin.isatty():
        raw_args_for_stdin: list[str] = []
        # Fall through to the parser which will print help; stdin handling below
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    workspace = args.workspace or Path.cwd()

    # --validate + -c → validate the code string
    if args.validate and args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        kit = _parse_kit(args.kit) if args.kit else None
        result = svc.validate(args.intent, kit=kit)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    # Runner-style interface (-c flag)
    if args.intent:
        from .service import LackpyService
        svc = LackpyService(workspace=workspace)
        kit = _parse_kit(args.kit) if args.kit else None

        if args.create:
            gen = asyncio.run(svc.generate(args.intent, kit=kit))
            tools = kit if isinstance(kit, list) else []
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
                gen = asyncio.run(svc.generate(args.intent, kit=kit))
            except RuntimeError as e:
                print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
                return 1
            print(gen.program)
            return 0

        # Default: delegate (generate + run)
        try:
            result = asyncio.run(svc.delegate(args.intent, kit=kit))
        except RuntimeError as e:
            print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["success"] else 1

    # Stdin: read program and run if stdin is not a tty and no command given
    if args.command is None and not sys.stdin.isatty():
        program = sys.stdin.read()
        if program.strip():
            from .service import LackpyService
            svc = LackpyService(workspace=workspace)
            kit = _parse_kit(args.kit) if args.kit else None
            exec_result = asyncio.run(svc.run_program(program, kit=kit))
            out_dict = {"success": exec_result.success, "output": exec_result.output, "error": exec_result.error}
            print(json.dumps(out_dict, indent=2, default=str))
            return 0 if exec_result.success else 1

    if args.command is None:
        parser.print_help()
        return 0

    # Deprecation warnings for management commands now in lackpyctl
    if args.command in _CTL_COMMANDS:
        print(
            f"lackpy: '{args.command}' has moved to lackpyctl. "
            f"Run: lackpyctl {args.command}",
            file=sys.stderr,
        )

    if args.command == "init":
        _init_config(workspace, args.ollama_model, args.ollama_url)
        return 0

    if args.command == "spec":
        from .lang.spec import get_spec
        print(json.dumps(get_spec(), indent=2))
        return 0

    from .service import LackpyService
    svc = LackpyService(workspace=workspace)

    if args.command == "status":
        config = svc._config
        info = {
            "workspace": str(workspace),
            "config_dir": str(config.config_dir),
            "inference_order": config.inference_order,
            "kit_default": config.kit_default,
            "sandbox_enabled": config.sandbox_enabled,
            "tools": len(svc.toolbox.tools),
        }
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "toolbox":
        if args.toolbox_command == "list":
            tools = svc.toolbox_list()
            print(json.dumps(tools, indent=2))
        elif args.toolbox_command == "show":
            tools = svc.toolbox_list()
            match = [t for t in tools if t["name"] == args.name]
            if not match:
                print(f"Tool '{args.name}' not found", file=sys.stderr)
                return 1
            print(json.dumps(match[0], indent=2))
        else:
            print("Usage: lackpyctl toolbox {list|show}", file=sys.stderr)
            return 1
        return 0

    if args.command == "kit":
        if args.kit_command == "list":
            kits = svc.kit_list()
            print(json.dumps(kits, indent=2))
        elif args.kit_command == "info":
            kit = _parse_kit(args.name) if args.tools is None else args.tools
            info = svc.kit_info(kit)
            print(json.dumps(info, indent=2))
        elif args.kit_command == "create":
            result = svc.kit_create(args.name, args.tools, args.description)
            print(json.dumps(result, indent=2))
        else:
            print("Usage: lackpyctl kit {list|info|create}", file=sys.stderr)
            return 1
        return 0

    if args.command == "template":
        if args.template_command == "list":
            templates_dir = svc._config.config_dir / "templates"
            if not templates_dir.exists():
                print("[]")
            else:
                tmpls = [{"name": p.stem, "path": str(p)} for p in sorted(templates_dir.glob("*.tmpl"))]
                print(json.dumps(tmpls, indent=2))
        elif args.template_command == "test":
            print(f"Testing template '{args.name}' not yet implemented", file=sys.stderr)
            return 1
        else:
            print("Usage: lackpyctl template {list|test}", file=sys.stderr)
            return 1
        return 0

    kit = _parse_kit(args.kit) if getattr(args, "kit", None) else None

    # Deprecation warnings for runner subcommands now superseded by flags/file args
    if args.command in {"delegate", "generate", "run", "create", "validate"}:
        _deprecated_runner_hint = {
            "delegate": "lackpy -c '<intent>'",
            "generate": "lackpy -c '<intent>' --generate",
            "run": "lackpy <file.py>",
            "create": "lackpy -c '<intent>' --create --name <Name>",
            "validate": "lackpy --validate <file.py>",
        }
        print(
            f"lackpy: subcommand '{args.command}' is deprecated. "
            f"Use: {_deprecated_runner_hint[args.command]}",
            file=sys.stderr,
        )

    if args.command == "validate":
        program = Path(args.file).read_text()
        result = svc.validate(program, kit=kit)
        out_v: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out_v, indent=2))
        return 0 if result.valid else 1

    if args.command == "run":
        program = Path(args.file).read_text()
        result = asyncio.run(svc.run_program(program, kit=kit))
        out_r = {"success": result.success, "output": result.output, "error": result.error}
        print(json.dumps(out_r, indent=2))
        return 0 if result.success else 1

    if args.command == "generate":
        try:
            result = asyncio.run(svc.generate(args.intent, kit=kit))
        except RuntimeError as e:
            print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
            return 1
        print(result.program)
        return 0

    if args.command == "create":
        program = Path(args.file).read_text()
        result = asyncio.run(svc.create(program, kit=kit, name=args.name, pattern=args.pattern))
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    if args.command == "delegate":
        sandbox = getattr(args, "sandbox", None)
        try:
            result = asyncio.run(svc.delegate(args.intent, kit=kit, sandbox=sandbox))
        except RuntimeError as e:
            print(json.dumps({"success": False, "error": str(e)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
