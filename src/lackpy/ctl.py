"""Management CLI for lackpy (lackpyctl)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_profile(profile_str: str) -> str | list[str]:
    """Parse a profile argument: comma-separated → tool list; bare → profile name."""
    parts = [k.strip() for k in profile_str.split(",")]
    return parts if len(parts) > 1 else parts[0]


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
order = ["templates", "rules", "local"]

# Model calls go through woollama's model-management core: one provider routes to
# any woollama-known backend via a "<provider>/<model>" string (ollama/…,
# anthropic/…, openai/…). The model choice is per-machine — set it here.
[inference.providers.local]
plugin = "woollama"
model = "ollama/{ollama_model}"
base_url = "{ollama_url}/v1"

[profile]
default = "debug"

[sandbox]
enabled = false
timeout_seconds = 120
memory_mb = 512
""")
    print(f"Initialized lackpy workspace at {config_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lackpyctl",
        description="lackpyctl — manager for lackpy workspaces, profiles, toolboxes, and templates",
    )
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="Workspace directory (default: cwd)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # init
    init_p = subparsers.add_parser("init", help="Initialize .lackpy workspace")
    init_p.add_argument("--ollama-model", default="qwen2.5-coder:1.5b", help="Default Ollama model")
    init_p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")

    # status
    subparsers.add_parser("status", help="Show lackpy status and configuration")

    # spec
    subparsers.add_parser("spec", help="Print language spec")

    # profile
    profile_p = subparsers.add_parser("profile", help="Manage profiles")
    profile_sub = profile_p.add_subparsers(dest="profile_command")

    profile_sub.add_parser("list", help="List available profiles")

    profile_info_p = profile_sub.add_parser("info", help="Show profile info")
    profile_info_p.add_argument("name", help="Profile name or comma-separated tools")
    profile_info_p.add_argument("--tools", nargs="+", default=None, help="Tool names")

    profile_create_p = profile_sub.add_parser("create", help="Create a new profile / tool-set")
    profile_create_p.add_argument("name", help="Profile name")
    profile_create_p.add_argument("--tools", nargs="+", required=True, help="Tool names to include")
    profile_create_p.add_argument("--description", default=None, help="Profile description")

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

    # provider (placeholder)
    provider_p = subparsers.add_parser("provider", help="Manage inference providers")
    provider_sub = provider_p.add_subparsers(dest="provider_command")

    provider_sub.add_parser("list", help="List configured providers")

    provider_add_p = provider_sub.add_parser("add", help="Add a provider")
    provider_add_p.add_argument("name", help="Provider name")

    provider_show_p = provider_sub.add_parser("show", help="Show provider details")
    provider_show_p.add_argument("name", help="Provider name")

    # mcp
    mcp_p = subparsers.add_parser("mcp", help="MCP server management")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command")

    mcp_sub.add_parser("serve", help="Start the MCP server (stdio transport)")

    mcp_init_p = mcp_sub.add_parser("init", help="Add lackpy to .mcp.json")
    mcp_init_p.add_argument("--name", default="lackpy", help="Server name in .mcp.json (default: lackpy)")
    mcp_init_p.add_argument("--force", action="store_true", default=False, help="Overwrite existing entry")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workspace = args.workspace or Path.cwd()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "init":
        _init_config(workspace, args.ollama_model, args.ollama_url)
        return 0

    if args.command == "spec":
        from .lang.spec import get_spec
        print(json.dumps(get_spec(), indent=2))
        return 0

    if args.command == "mcp":
        from .mcp.cli import mcp_init, mcp_serve
        if args.mcp_command == "serve":
            return mcp_serve(workspace)
        elif args.mcp_command == "init":
            return mcp_init(
                workspace=workspace,
                name=args.name,
                force=args.force,
            )
        else:
            print("Usage: lackpyctl mcp {serve|init}", file=sys.stderr)
            return 1

    from .service import LackpyService
    svc = LackpyService(workspace=workspace)

    if args.command == "status":
        config = svc.get_config()
        print(json.dumps(config, indent=2))
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

    if args.command == "profile":
        if args.profile_command == "list":
            profiles = svc.profile_list()
            print(json.dumps(profiles, indent=2))
        elif args.profile_command == "info":
            profile = _parse_profile(args.name) if args.tools is None else args.tools
            info = svc.profile_info(profile)
            print(json.dumps(info, indent=2))
        elif args.profile_command == "create":
            result = svc.profile_create(args.name, args.tools, args.description)
            print(json.dumps(result, indent=2))
        else:
            print("Usage: lackpyctl profile {list|info|create}", file=sys.stderr)
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

    if args.command == "provider":
        print("Provider management not yet implemented", file=sys.stderr)
        return 1

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
