"""Fresh fixer prompt for code correction strategy."""

from __future__ import annotations

from ..lang.spec import format_preprompt

_FIXER_TEMPLATE = """\
You are fixing a lackpy program. The validator rejected the code below; its errors follow it.
A lackpy program is plain Python 3 that follows the lesson below.

{lesson}

Rewrite the program so that it does the intent and breaks no rule.
Output ONLY the fixed code — no explanation, no markdown fences."""


def build_fixer_prompt(namespace_desc: str) -> str:
    """Build the system prompt for the fixer inference role.

    The prompt teaches the lackpy language through ``format_preprompt()``
    (stepy100 Task 8). ``namespace_desc`` is not taught: no run binds a tool
    since the kit was removed.

    Args:
        namespace_desc: Unused; kept for callers' signatures.

    Returns:
        The complete fixer system prompt string.
    """
    return _FIXER_TEMPLATE.format(lesson=format_preprompt().rstrip())


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
        namespace_desc: Unused; see :func:`build_fixer_prompt`.

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
