"""The lackpy language spec as structured data, exposable as an MCP resource."""

import sys

from .builtins_policy import ALLOWED
from .dictionary import MODULES, PATH_MEMBERS, UNAPPROVED
from .grammar import ALLOWED_NODES, FORBIDDEN_NODES, FORBIDDEN_NAMES, ALLOWED_BUILTINS
from .validator import validate as _validate  # noqa: F401  (importing it registers every rule)
from .registry import RULES


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


def get_spec() -> dict:
    """Return the language spec as structured data."""
    return {
        "allowed_nodes": sorted(n.__name__ for n in ALLOWED_NODES),
        "forbidden_nodes": sorted(n.__name__ for n in FORBIDDEN_NODES),
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "allowed_builtins": sorted(ALLOWED_BUILTINS),
        "rules": [{"id": r.id, "tier": r.tier, "summary": r.summary} for r in RULES],
    }


_PREFACE = """\
This guide defines the lackpy language. A lackpy program is a plain Python 3 program. It runs unchanged under CPython with only the standard library, from the workspace directory, and gives the same output there as inside lackpy. Inside lackpy it can only do less: a sandbox refusal is an error, never a different result. Every rule has one of two tiers. A *safety* rule protects the sandbox, and nothing waives it. A *style* rule keeps programs short and uniform, and a waiver can lift it. A waiver is a comment of the form `# ste: <rule-id>, <why>`. The why is required. A waiver covers one statement: put it at the end of the statement's first line, or on the line directly above. For a compound statement, a waiver covers the header only, not the body. A waiver names style rules only. A waiver that names a safety rule gives `meta.waiver-safety`. The builtin exception classes in Part 2 come from Python {version}, the interpreter that generated this file.
"""


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def format_style_guide() -> str:
    """Render the style guide (Parts 1 and 2) from the registry and the dictionary."""
    v = sys.version_info
    lines = ["# lackpy Style Guide", "",
             _PREFACE.format(version=f"{v.major}.{v.minor}.{v.micro}").rstrip(), "",
             "## Part 1. Program rules", ""]
    for r in RULES:
        lines += [f"### {r.id}", "", r.summary, "", f"**Tier:** {r.tier}", "",
                  f"**Why:** {r.rationale}", "",
                  "**Not approved:**", "", "```python", r.bad.rstrip("\n"), "```", "",
                  "**Approved:**", "", "```python", r.good.rstrip("\n"), "```", ""]
    functions = sorted(n for n, k in ALLOWED.items() if k == "function")
    exceptions = sorted(n for n, k in ALLOWED.items() if k == "exception")
    lines += ["## Part 2. Dictionary", "", "### Approved builtins", ""]
    lines += _table(["Kind", "Names"], [
        ["Functions and types", ", ".join(f"`{n}`" for n in functions)],
        ["Exception classes", ", ".join(f"`{n}`" for n in exceptions)],
    ]) + [""]
    lines += ["### Approved modules and members", ""]
    lines += _table(["Module", "Members"], [
        [f"`{m}`", ", ".join(f"`{x}`" for x in sorted(members))] for m, members in MODULES.items()
    ]) + [""]
    lines += ["### Path members", "",
             ", ".join(f"`{x}`" for x in sorted(PATH_MEMBERS)), ""]
    lines += ["### Unapproved words", ""]
    lines += _table(["Word", "Approved alternative"],
                    [[f"`{w}`", hint] for w, hint in UNAPPROVED.items()]) + [""]
    lines += ["Part 3 (intent rules) arrives with Plan 3.", ""]
    return "\n".join(lines)
