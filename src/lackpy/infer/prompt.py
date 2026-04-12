"""System prompt construction for inference providers."""

from __future__ import annotations

from typing import Any

from ..lang.grammar import ALLOWED_BUILTINS
from .retrieval import Example, format_examples_for_prompt, retrieve_examples

_TEMPLATE = """\
You are a Jupyter notebook cell generator. Write a single cell \
using ONLY the pre-loaded kernel namespace below.

Output ONLY the cell contents — no markdown, no explanation, no code fences.

Assign tool results to variables and reuse them. Never call the same function twice \
when you can reuse a variable.

You orchestrate tools to find and modify existing code. \
Use read_file(path) to get file contents.

Kernel namespace:
{namespace_desc}

Builtins: {builtins_list}
{params_section}{examples_section}"""

_SPECIALIZED_TEMPLATE = """\
{interpreter_hint}

{namespace_desc}

Builtins: {builtins_list}
{params_section}{examples_section}"""


def build_system_prompt(
    namespace_desc: str,
    params_desc: str | None = None,
    intent: str | None = None,
    example_pool: list[Example] | None = None,
    n_examples: int = 6,
    interpreter: Any = None,
) -> str:
    """Build the system prompt for inference providers.

    When ``interpreter`` is provided and has a ``system_prompt_hint()``
    method, the prompt uses interpreter-specialized framing instead of
    the generic Jupyter-cell template. This typically produces 3–5×
    higher pass rates on local models (empirically validated across
    qwen2.5 0.5b–7b, smollm2, and granite models).

    When no interpreter is provided, falls back to the original
    Jupyter-cell framing for backward compatibility.

    Args:
        namespace_desc: Formatted string of available tools and their signatures.
        params_desc: Optional description of pre-set parameter variables.
        intent: Natural language intent. Required for example retrieval.
        example_pool: Candidate examples to retrieve from. If None or empty,
            no examples section is added.
        n_examples: Maximum number of examples to include. Default 6 — tested
            as the sweet spot for qwen2.5-coder models on structured output.
        interpreter: An interpreter instance (or any object with a
            ``system_prompt_hint()`` method). When present and the method
            exists, the prompt uses interpreter-specialized framing.

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

    examples_section = ""
    if intent and example_pool:
        selected = retrieve_examples(intent, example_pool, n=n_examples)
        if selected:
            examples_section = "\n" + format_examples_for_prompt(selected) + "\n"

    # Use interpreter-specialized framing when available
    hint = None
    if interpreter is not None:
        hint_fn = getattr(interpreter, "system_prompt_hint", None)
        if hint_fn is not None:
            hint = hint_fn()

    if hint:
        return _SPECIALIZED_TEMPLATE.format(
            interpreter_hint=hint,
            namespace_desc=namespace_desc,
            builtins_list=builtins_list,
            params_section=params_section,
            examples_section=examples_section,
        )

    return _TEMPLATE.format(
        namespace_desc=namespace_desc,
        builtins_list=builtins_list,
        params_section=params_section,
        examples_section=examples_section,
    )


def collect_example_pool(tool_specs: list) -> list[Example]:
    """Gather all examples from a list of ToolSpec objects into a flat pool.

    Each tool may contribute zero or more examples. Returns an empty list if
    none are defined. Example dicts are converted to Example objects; entries
    missing required fields are skipped.
    """
    pool: list[Example] = []
    for spec in tool_specs:
        for ex in getattr(spec, "examples", None) or []:
            intent = ex.get("intent", "")
            code = ex.get("code", "")
            if not intent or not code:
                continue
            tags = set(ex.get("tags", []))
            pool.append(Example(intent=intent, code=code, tags=tags))
    return pool


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
