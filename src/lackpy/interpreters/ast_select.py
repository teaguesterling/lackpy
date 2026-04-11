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
from typing import Any

from .base import (
    ExecutionContext,
    Interpreter,
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
        """Resolve the selector via pluckit and render matches as markdown."""
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
            from pluckit import Plucker
        except ImportError:
            return InterpreterExecutionResult(
                success=False,
                error="ast-select requires pluckit to be installed (pip install ast-pluckit)",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        code = context.config.get("code")
        if not code:
            # Default to a broad glob under base_dir. Pluckit handles
            # multi-language detection across file extensions.
            code = str(Path(context.base_dir) / "**" / "*")

        try:
            pluck = Plucker(code=code)
            selection = pluck.find(selector)
            rows = selection.materialize()
        except Exception as e:
            return InterpreterExecutionResult(
                success=False,
                error=f"pluckit execution failed: {type(e).__name__}: {e}",
                output_format="none",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        if not rows:
            # Empty match set → empty output. Mirrors grep semantics.
            return InterpreterExecutionResult(
                success=True,
                output="",
                output_format="markdown",
                duration_ms=(time.perf_counter() - start) * 1000,
                metadata={"match_count": 0, "selector": selector, "mode": mode},
            )

        # Fetch bodies only when we need them (mode=full).
        bodies: list[str] = []
        if mode == "full":
            try:
                bodies = selection.text()
            except Exception as e:
                return InterpreterExecutionResult(
                    success=False,
                    error=f"pluckit text() failed: {type(e).__name__}: {e}",
                    output_format="none",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            if len(bodies) != len(rows):
                return InterpreterExecutionResult(
                    success=False,
                    error=(
                        f"pluckit returned inconsistent row/text counts: "
                        f"{len(rows)} rows vs {len(bodies)} texts"
                    ),
                    output_format="none",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

        if mode == "brief":
            output = self._render_brief(selector, rows, context.base_dir)
        else:
            output = self._render_full(selector, rows, bodies, context.base_dir)

        return InterpreterExecutionResult(
            success=True,
            output=output,
            output_format="markdown",
            duration_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "match_count": len(rows),
                "selector": selector,
                "mode": mode,
            },
        )

    # ── Rendering ─────────────────────────────────────────────────────

    def _render_full(
        self,
        selector: str,
        rows: list[dict[str, Any]],
        bodies: list[str],
        base_dir: Path,
    ) -> str:
        """Render the full-body markdown for one or more matches."""
        lines: list[str] = [f"# `{selector}`", ""]

        for row, body in zip(rows, bodies):
            location = self._format_location(row, base_dir)
            qname = row.get("qualified_name") or row.get("name") or ""
            heading = self._match_heading(qname, location)
            lines.append(f"## {heading}")
            lines.append("")
            lang = row.get("language") or "text"
            lines.append(f"```{lang}")
            lines.append(self._dedent_body(body))
            lines.append("```")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_brief(
        self,
        selector: str,
        rows: list[dict[str, Any]],
        base_dir: Path,
    ) -> str:
        """Render the one-line-per-match brief format using peek."""
        lines: list[str] = [f"# `{selector}`", ""]

        for row in rows:
            location = self._format_location(row, base_dir, include_end=False)
            peek = (row.get("peek") or "").strip()
            lines.append(f"- `{location}` — {peek}")

        lines.append("")
        return "\n".join(lines)

    def _match_heading(self, qname: str, location: str) -> str:
        """Build a per-match subheading combining qualified name and location.

        Prefers ``qualified_name location`` when both are present;
        falls back to location alone if the qualified name is missing.
        """
        if qname:
            return f"`{qname}` — {location}"
        return f"`{location}`"

    def _format_location(
        self,
        row: dict[str, Any],
        base_dir: Path,
        include_end: bool = True,
    ) -> str:
        """Format a row's file path and line range, relative to base_dir."""
        file_path = row.get("file_path") or ""
        start = row.get("start_line") or 0
        end = row.get("end_line") or start

        try:
            rel = str(Path(file_path).resolve().relative_to(Path(base_dir).resolve()))
        except (ValueError, OSError):
            rel = file_path

        if include_end and end and end != start:
            return f"{rel}:{start}-{end}"
        return f"{rel}:{start}"

    def _dedent_body(self, body: str) -> str:
        """Strip common leading whitespace from a multi-line source body.

        Pluckit returns source text with its original indentation (methods
        inside classes have 4-space indent, etc.). For display in code
        blocks, it reads more naturally with the common prefix removed.
        """
        if not body:
            return body
        lines = body.splitlines()
        # Find the minimum indent of non-empty lines
        indents = [
            len(line) - len(line.lstrip())
            for line in lines
            if line.strip()
        ]
        if not indents:
            return body
        common = min(indents)
        if common == 0:
            return body
        return "\n".join(
            line[common:] if len(line) >= common else line
            for line in lines
        )
