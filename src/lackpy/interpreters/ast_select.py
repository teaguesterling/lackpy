"""AST selector interpreter — pluckit-backed CSS selector evaluation.

The program is a single CSS selector string like ``.fn:async`` or
``.class#Bar .fn#foo``. Execution resolves the selector against a
configured code source (via pluckit) and renders the matches as markdown
with the selector as the heading and each match's body in a code block.

This interpreter exists so that a small model can be trained / prompted
exclusively on the selector language without being distracted by output
formatting, display choices, or declaration vocabulary. The inferencer
writes selectors; the interpreter handles the rest.

Output format:

.. code-block:: markdown

    # `.class#Bar .fn#foo`

    ## `C/Bar F/foo` — src/my_module.py:12-18

    ```python
    def foo(data):
        ...
    ```

- ``mode="full"`` (default): full body in code block
- ``mode="brief"``: one-line-per-match using pluckit's ``peek`` field

``qualified_name`` becomes the per-match identifier, giving downstream
inferencers a unique way to reference a specific match when a selector
has multiple hits. The selector itself remains the portable reference
because selectors compose; the qualified name is for disambiguation
inside the output, not for passing around.
"""

from __future__ import annotations

import time
from pathlib import Path

from .base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)


class AstSelectInterpreter:
    """Evaluate a CSS selector against a pluckit-backed code source.

    Configuration (via ``context.config``):

    - ``mode``: ``"full"`` (default) or ``"brief"``. ``full`` renders
      each match's body in a code block; ``brief`` renders one line per
      match using pluckit's ``peek`` field (typically the signature line).
    - ``code``: the source the selector evaluates against. A glob
      pattern, file path, or pluckit table reference. If omitted, the
      interpreter uses ``context.base_dir`` as a broad ``**/*`` glob.
    - ``plugins``: additional pluckit plugins to load (rare; the default
      Code plugin is always loaded).
    """

    name = "ast-select"
    description = "CSS selectors evaluated via pluckit, rendered as markdown"

    def system_prompt_hint(self) -> str:
        """Interpreter-specialized prompt fragment for ast-select.

        The selector syntax reference alone drives 93% pass rates on
        capable models — few-shot examples don't improve and sometimes
        hurt. Keep this lean.
        """
        return (
            "You generate a single CSS-style selector for querying source code ASTs.\n"
            "\n"
            "Selector syntax:\n"
            "  .fn                         — all function definitions\n"
            "  .cls                        — all class definitions\n"
            "  .fn#NAME                    — function named NAME\n"
            "  .cls#NAME                   — class named NAME\n"
            '  .fn[name^="prefix"]         — functions whose name starts with prefix\n'
            "  .fn:async                   — async functions\n"
            "  .cls .fn                    — methods inside any class\n"
            "  .cls#User .fn               — methods inside class User\n"
            "  .fn:has(.call#execute_sql)  — functions containing a call to execute_sql\n"
            '  .fn:not([name^="test_"])    — functions not starting with test_\n'
            "\n"
            "Output: ONE selector, ONE line, nothing else.\n"
            "No code fences, no Python, no chain syntax, no explanation."
        )

    def validate(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterValidationResult:
        """Basic sanity checks on the selector program.

        v1 validation is lightweight: check the program is non-empty and
        contains no brace-delimited declaration blocks (that would mean
        the inferencer produced a selector sheet, which belongs to the
        ``pss`` interpreter, not this one). Full selector-grammar
        validation is deferred to pluckit at execution time.
        """
        if not program or not program.strip():
            return InterpreterValidationResult(
                valid=False,
                errors=["empty program"],
            )

        stripped = program.strip()

        # ast-select accepts only bare selectors. Declaration blocks
        # belong to the pss interpreter.
        if "{" in stripped or "}" in stripped:
            return InterpreterValidationResult(
                valid=False,
                errors=[
                    "ast-select expects a bare CSS selector, not a selector sheet. "
                    "Use the 'pss' interpreter for selectors with declaration blocks."
                ],
            )

        # Crude sanity: selectors shouldn't span multiple non-empty lines.
        lines = [ln for ln in stripped.splitlines() if ln.strip()]
        if len(lines) > 1:
            return InterpreterValidationResult(
                valid=False,
                errors=[
                    f"ast-select expects a single-line selector; got {len(lines)} lines. "
                    "Use the 'pss' interpreter for multi-rule sheets."
                ],
            )

        return InterpreterValidationResult(valid=True)

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterExecutionResult:
        """Resolve the selector via pluckit and render matches as markdown.

        Delegates to pluckit's ``Plucker.view()`` which handles rendering,
        dedenting, and heading formatting. The ``mode`` config key maps to
        a pss declaration: ``full`` → ``show: body``, ``brief`` →
        ``show: signature``.
        """
        start = time.perf_counter()

        validation = self.validate(program, context)
        if not validation.valid:
            return InterpreterExecutionResult(
                success=False,
                error="Validation failed: " + "; ".join(validation.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        selector = program.strip()
        mode = context.config.get("mode", "full")
        if mode not in ("full", "brief"):
            return InterpreterExecutionResult(
                success=False,
                error=f"unknown mode: {mode!r}; expected 'full' or 'brief'",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            from pluckit.plucker import Plucker
            from pluckit.plugins import AstViewer
        except ImportError:
            return InterpreterExecutionResult(
                success=False,
                error="ast-select requires pluckit to be installed (pip install ast-pluckit)",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        code = context.config.get("code")
        if not code:
            code = str(Path(context.base_dir))

        # Map mode to pss declaration syntax for pluckit's view()
        show = "body" if mode == "full" else "signature"
        query = f"{selector} {{ show: {show}; }}"

        try:
            pluck = Plucker(code=code, plugins=[AstViewer])
            view = pluck.view(query)
            output = str(view)
            match_count = len(view)
        except Exception as e:
            return InterpreterExecutionResult(
                success=False,
                error=f"pluckit execution failed: {type(e).__name__}: {e}",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        return InterpreterExecutionResult(
            success=True,
            output=output,
            output_format="markdown",
            duration_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "match_count": match_count,
                "selector": selector,
                "mode": mode,
            },
        )
