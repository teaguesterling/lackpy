"""Pluckit Source Selection (pss) interpreter — multi-rule selector sheets.

The program is a CSS-style selector sheet: one or more rules, each a
selector optionally followed by a declaration block that controls how
the match is rendered. The interpreter delegates to pluckit's
``AstViewer`` plugin and returns the combined markdown.

.. code-block:: css

    .fn#authenticate { show: body; }
    .class#User      { show: outline; }
    .fn[name^=test_] { show: signature; }

This interpreter exists so that an inferencer can be prompted
exclusively on *view composition* — picking selectors, choosing display
modes, structuring a curated view — without mixing in the raw-query
task that :mod:`lackpy.interpreters.ast_select` handles.

pss and ast-select share a backend (``pluck.view()``), but they are
separate plugins because the cognitive task of the inferencer generating
each one is different. An inferencer targeting pss produces a document
with declaration vocabulary and structural decisions; an inferencer
targeting ast-select produces a single bare selector with no display
decisions. Keeping them separate lets each be prompted and retrieval-
augmented for exactly one target.

Output format is pluckit's native viewer output: file path as an H1
heading per match, followed by a language-tagged code block. The exact
rendering is whatever the ``AstViewer`` plugin produces — the pss
interpreter does not reformat.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)


class PssInterpreter:
    """Multi-rule selector sheet interpreter backed by pluckit's AstViewer.

    Configuration (via ``context.config``):

    - ``code``: source the selectors evaluate against. Glob pattern,
      file path, or pluckit table reference. Required — unlike
      ast-select, pss does not default to a broad glob because view
      composition is usually intentional about scope.
    - ``format``: rendering format passed to ``pluck.view()``. Default
      ``"markdown"``. Other values depend on what pluckit's AstViewer
      supports at the current version.
    - ``plugins``: additional pluckit plugins to load beyond the
      default Code and AstViewer plugins. Rare.
    """

    name = "pss"
    description = (
        "Pluckit selector sheets (multi-rule selector + declaration blocks) "
        "rendered as markdown"
    )

    def system_prompt_hint(self) -> str:
        """Interpreter-specialized prompt fragment for pss."""
        return (
            "You generate a pluckit selector sheet: one or more rules, "
            "each a selector followed by a declaration block.\n"
            "\n"
            "Sheet syntax:\n"
            "  SELECTOR { show: body; }\n"
            "  SELECTOR { show: signature; }\n"
            "  SELECTOR { show: outline; }\n"
            "  SELECTOR { show: 10; }\n"
            "\n"
            "Multi-rule sheets are one rule per line.\n"
            "\n"
            "Selector syntax:\n"
            "  .fn                         — all function definitions\n"
            "  .cls                        — all class definitions\n"
            "  .fn#NAME                    — function named NAME\n"
            "  .cls#NAME                   — class named NAME\n"
            '  .fn[name^="prefix"]         — functions whose name starts with prefix\n'
            '  .fn[name*="substr"]         — functions whose name contains substr\n'
            "  .fn:async                   — async functions\n"
            "  .fn:exported                — public functions\n"
            "  .fn:decorated               — decorated functions\n"
            "  .fn:long(50)                — functions longer than 50 lines\n"
            "  .cls .fn                    — methods inside any class\n"
            "  .cls#User .fn               — methods inside class User\n"
            "  .fn:has(.call#execute)      — functions containing a call to execute\n"
            "  .fn:not(:async)             — non-async functions\n"
            "\n"
            "Output ONLY the sheet — no prose, no code fences."
        )

    def validate(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterValidationResult:
        """Lightweight validation: non-empty, balanced braces.

        Full grammar validation is deferred to pluckit's AstViewer at
        execution time. Here we only check that the program is
        non-empty and has balanced curly braces (so an obviously
        malformed sheet fails fast without spinning up pluckit).
        """
        if not program or not program.strip():
            return InterpreterValidationResult(
                valid=False,
                errors=["empty program"],
            )

        # A pss program is allowed to be a single bare selector with no
        # declarations (it then behaves identically to ast-select). That
        # means we don't require a { } block. But if there's an opening
        # brace, there must be a matching close.
        open_count = program.count("{")
        close_count = program.count("}")
        if open_count != close_count:
            return InterpreterValidationResult(
                valid=False,
                errors=[
                    f"unbalanced braces: {open_count} opening, {close_count} closing"
                ],
            )

        return InterpreterValidationResult(valid=True)

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterExecutionResult:
        """Evaluate the sheet via pluckit's AstViewer and return markdown."""
        start = time.perf_counter()

        validation = self.validate(program, context)
        if not validation.valid:
            return InterpreterExecutionResult(
                success=False,
                error="Validation failed: " + "; ".join(validation.errors),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        code = context.config.get("code")
        if not code:
            # Unlike ast-select, pss does not provide a broad default
            # glob. View composition is intentional about what it covers.
            return InterpreterExecutionResult(
                success=False,
                error=(
                    "pss requires a 'code' config value (glob, file path, or "
                    "pluckit table reference) pointing at the source to query"
                ),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        fmt = context.config.get("format", "markdown")

        try:
            from pluckit import Plucker, AstViewer
        except ImportError:
            return InterpreterExecutionResult(
                success=False,
                error=(
                    "pss requires pluckit to be installed "
                    "(pip install ast-pluckit)"
                ),
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        extra_plugins = list(context.config.get("plugins", []))

        try:
            pluck = Plucker(code=code, plugins=[AstViewer, *extra_plugins])
            output = pluck.view(program, format=fmt)
        except Exception as e:
            return InterpreterExecutionResult(
                success=False,
                error=f"pluckit view failed: {type(e).__name__}: {e}",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        return InterpreterExecutionResult(
            success=True,
            output=output,
            output_format=fmt,
            duration_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "code": code,
                "format": fmt,
                "rule_count": self._count_rules(program),
            },
        )

    def _count_rules(self, program: str) -> int:
        """Rough count of rules in the sheet for metadata purposes.

        Counts non-empty logical blocks separated by ``}`` followed by
        whitespace. Not grammar-aware — a rough indicator for
        observability, not a validation check.
        """
        if "{" not in program:
            # Bare selector with no declaration block — one rule
            return 1
        return max(1, program.count("}"))
