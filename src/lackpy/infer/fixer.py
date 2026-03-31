"""Fresh fixer prompt for code correction strategy."""

from __future__ import annotations

from ..lang.grammar import ALLOWED_BUILTINS

_FIXER_TEMPLATE = """\
You are fixing code for a restricted Python environment. \
The code below was generated but contains errors.

Rewrite the code using ONLY these functions:
{namespace_desc}

Builtins: {builtins_list}

Output ONLY the fixed code — no explanation, no markdown fences."""


def build_fixer_prompt(namespace_desc: str) -> str:
    """Build the system prompt for the fixer inference role.

    Args:
        namespace_desc: Formatted string of available tools and their signatures.

    Returns:
        The complete fixer system prompt string.
    """
    builtins_list = ", ".join(sorted(ALLOWED_BUILTINS))
    return _FIXER_TEMPLATE.format(
        namespace_desc=namespace_desc,
        builtins_list=builtins_list,
    )


def build_fixer_messages(
    intent: str,
    broken_program: str,
    errors: str,
    namespace_desc: str,
) -> list[dict]:
    """Build the [system, user] message list for a fixer inference call.

    Args:
        intent: The original user intent / task description.
        broken_program: The broken Python source that needs correction.
        errors: Error output or validation messages from the broken program.
        namespace_desc: Formatted string of available tools and their signatures.

    Returns:
        A list of two message dicts: the system prompt and the user message.
    """
    system_prompt = build_fixer_prompt(namespace_desc)
    user_content = (
        f"Intent: {intent}\n\n"
        f"Broken code:\n{broken_program}\n\n"
        f"Errors:\n{errors}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
