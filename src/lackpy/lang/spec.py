"""The lackpy language spec as structured data, exposable as an MCP resource."""

from .grammar import ALLOWED_NODES, FORBIDDEN_NODES, FORBIDDEN_NAMES, ALLOWED_BUILTINS


def format_spec() -> str:
    lines = ["# lackpy Language Specification", ""]
    lines.append("## Allowed AST Nodes")
    for node in sorted(n.__name__ for n in ALLOWED_NODES):
        lines.append(f"  - {node}")
    lines.append("")
    lines.append("## Forbidden AST Nodes")
    for node in sorted(n.__name__ for n in FORBIDDEN_NODES):
        lines.append(f"  - {node}")
    lines.append("")
    lines.append("## Forbidden Names")
    for name in sorted(FORBIDDEN_NAMES):
        lines.append(f"  - {name}")
    lines.append("")
    lines.append("## Allowed Builtins")
    for name in sorted(ALLOWED_BUILTINS):
        lines.append(f"  - {name}")
    return "\n".join(lines)
