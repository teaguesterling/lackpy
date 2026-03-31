"""System prompt construction for inference providers."""

from __future__ import annotations

from ..lang.grammar import ALLOWED_BUILTINS

_TEMPLATE = """\
You are a Jupyter notebook cell generator. Write a single cell \
using ONLY the pre-loaded kernel namespace below.

Output ONLY the cell contents — no markdown, no explanation, no code fences.

Assign tool results to variables and reuse them. Never call the same function twice \
when you can reuse a variable.

You orchestrate tools to find and modify existing code. \
Use read(path) to get file contents.

Kernel namespace:
{namespace_desc}

Builtins: {builtins_list}
{params_section}"""


def build_system_prompt(namespace_desc: str, params_desc: str | None = None) -> str:
    """Build the system prompt for inference providers.

    Constructs the full instruction prompt from the tool namespace description,
    allowed builtins, and optional parameter variable descriptions.

    Args:
        namespace_desc: Formatted string of available tools and their signatures.
        params_desc: Optional description of pre-set parameter variables; inserted
            into the prompt when provided.

    Returns:
        The complete system prompt string ready to send to an inference provider.
    """
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
    """Format a parameters dict into a human-readable description string.

    Each parameter is rendered as ``name: type`` optionally followed by a
    description when the value is a metadata dict.

    Args:
        params: Mapping of parameter name to either a raw value or a metadata
            dict with ``"value"``, optional ``"type"``, and optional
            ``"description"`` keys.

    Returns:
        A newline-joined string of parameter descriptions.
    """
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
