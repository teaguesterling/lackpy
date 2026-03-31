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

    Always returns a list. Use 'lackpy kit info <name>' to query
    predefined kits by name instead.
    """
    return [k.strip() for k in kit_str.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lackpy",
        description="lackpy — micro-inferencer for restricted Python programs",
    )
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="Workspace directory (default: cwd)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # delegate
    delegate_p = subparsers.add_parser("delegate", help="Generate and run a program from intent")
    delegate_p.add_argument("intent", help="Natural language intent")
    delegate_p.add_argument("--kit", default=None, help="Kit name, comma-separated list, or @file")
    delegate_p.add_argument("--sandbox", default=None, help="Sandbox profile")

    # generate
    generate_p = subparsers.add_parser("generate", help="Generate a program from intent without running")
    generate_p.add_argument("intent", help="Natural language intent")
    generate_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")

    # run
    run_p = subparsers.add_parser("run", help="Run a program file")
    run_p.add_argument("file", help="Program file to run")
    run_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")

    # create
    create_p = subparsers.add_parser("create", help="Validate and save a program as a template")
    create_p.add_argument("file", help="Program file to save as template")
    create_p.add_argument("--name", required=True, help="Template name")
    create_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")
    create_p.add_argument("--pattern", default=None, help="Intent pattern regex")

    # validate
    validate_p = subparsers.add_parser("validate", help="Validate a program without running")
    validate_p.add_argument("file", help="Program file to validate")
    validate_p.add_argument("--kit", default=None, help="Kit name or comma-separated list")

    # spec
    subparsers.add_parser("spec", help="Print language spec")

    # status
    subparsers.add_parser("status", help="Show lackpy status and configuration")

    # kit
    kit_p = subparsers.add_parser("kit", help="Manage kits")
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
    toolbox_p = subparsers.add_parser("toolbox", help="Manage toolbox")
    toolbox_sub = toolbox_p.add_subparsers(dest="toolbox_command")

    toolbox_sub.add_parser("list", help="List all registered tools")

    toolbox_show_p = toolbox_sub.add_parser("show", help="Show tool details")
    toolbox_show_p.add_argument("name", help="Tool name")

    # template
    template_p = subparsers.add_parser("template", help="Manage templates")
    template_sub = template_p.add_subparsers(dest="template_command")

    template_sub.add_parser("list", help="List available templates")

    template_test_p = template_sub.add_parser("test", help="Test a template")
    template_test_p.add_argument("name", help="Template name")

    # init
    init_p = subparsers.add_parser("init", help="Initialize .lackpy workspace")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    workspace = args.workspace or Path.cwd()

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
            print("Usage: lackpy toolbox {list|show}", file=sys.stderr)
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
            print("Usage: lackpy kit {list|info|create}", file=sys.stderr)
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
            print("Usage: lackpy template {list|test}", file=sys.stderr)
            return 1
        return 0

    kit = _parse_kit(args.kit) if getattr(args, "kit", None) else None

    if args.command == "validate":
        program = Path(args.file).read_text()
        result = svc.validate(program, kit=kit)
        out: dict[str, Any] = {"valid": result.valid, "errors": result.errors, "calls": list(result.calls)}
        print(json.dumps(out, indent=2))
        return 0 if result.valid else 1

    if args.command == "run":
        program = Path(args.file).read_text()
        result = asyncio.run(svc.run_program(program, kit=kit))
        out = {"success": result.success, "output": result.output, "error": result.error}
        print(json.dumps(out, indent=2))
        return 0 if result.success else 1

    if args.command == "generate":
        result = asyncio.run(svc.generate(args.intent, kit=kit))
        print(result.program)
        return 0

    if args.command == "create":
        program = Path(args.file).read_text()
        result = asyncio.run(svc.create(program, kit=kit, name=args.name, pattern=args.pattern))
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    if args.command == "delegate":
        sandbox = getattr(args, "sandbox", None)
        result = asyncio.run(svc.delegate(args.intent, kit=kit, sandbox=sandbox))
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
