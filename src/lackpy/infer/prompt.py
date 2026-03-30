"""System prompt construction for inference providers."""

from __future__ import annotations

from ..lang.grammar import ALLOWED_BUILTINS

_TEMPLATE = """\
You are a Jupyter notebook cell generator. Write a single cell \
using ONLY the pre-loaded kernel namespace below.

Output ONLY the cell contents — no markdown, no explanation, no code fences.

Assign tool results to variables and reuse them. Never call the same function twice \
when you can reuse a variable.

Kernel namespace:
{namespace_desc}

Builtins: {builtins_list}
{params_section}\
Not available: import, def, class, while, try/except, lambda, open

The cell's last expression is displayed as output."""


def build_system_prompt(namespace_desc: str, params_desc: str | None = None) -> str:
    builtins_list = ", ".join(sorted(ALLOWED_BUILTINS))
    params_section = ""
    if params_desc:
        params_section = (
            f"\nPre-set variables (already defined, use directly):\n"
            f"{params_desc}\n\n"
        )
    return _TEMPLATE.format(
        namespace_desc=namespace_desc,
        builtins_list=builtins_list,
        params_section=params_section,
    )


def format_params_description(params: dict) -> str:
    lines = []
    for name, value in params.items():
        if isinstance(value, dict) and "value" in value:
            ptype = value.get("type", type(value["value"]).__name__)
            desc = value.get("description", "")
            lines.append(f"  {name}: {ptype}" + (f" — {desc}" if desc else ""))
        else:
            ptype = type(value).__name__
            lines.append(f"  {name}: {ptype}")
    return "\n".join(lines)
