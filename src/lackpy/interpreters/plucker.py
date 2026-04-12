"""Plucker interpreter — fluent chain expressions over pluckit.

The program is a jQuery-style fluent chain built on pluckit's Plucker
and Selection classes:

.. code-block:: python

    p = source("src/**/*.py")
    fns = p.find(".fn:async")
    fns.names()

Or the same chain in a single expression:

.. code-block:: python

    source("src/**/*.py").find(".fn:async").names()

Implementation: a thin wrapper over :class:`PythonInterpreter`. Lackpy's
restricted validator already allows method calls on tool return values
without requiring the method names to be in ``allowed_names`` — only
the entry point needs to be registered. The plucker interpreter builds
a kit with a single entry point, ``source(code)``, that returns a real
:class:`pluckit.Plucker` instance. Everything after the first call is
normal Python attribute access on live pluckit objects.

Output format is ``"python"`` because the chain's terminal operation
determines the return type: ``.names()`` gives a list of strings,
``.materialize()`` gives a list of dicts, ``.view()`` gives markdown,
``.count()`` gives an int. The interpreter preserves whatever pluckit
returns without coercing to a single shape.

Tracing: only the initial ``source()`` call appears in the execution
trace; subsequent method calls are not traced because they happen
inside runtime Python object dispatch rather than through the kit's
instrumented tool wrappers. Fluent chains are opaque to lackpy's
tracing by design — the final result is the interesting artifact, not
the intermediate steps.

This interpreter exists so an inferencer can target pluckit's fluent
API as its own cognitive task, separately from restricted Python
(:class:`PythonInterpreter`) or selector rendering (``ast-select``,
``pss``). Each interpreter targets a distinct output shape:

- python → arbitrary tool composition via bare function calls
- ast-select → bare CSS selector, rendered as markdown
- pss → multi-rule selector sheet, rendered as markdown
- plucker → fluent chain expression, returns pluckit's native type
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..kit.registry import ResolvedKit
from ..kit.toolbox import ArgSpec, ToolSpec
from ..lang.grader import Grade
from .base import (
    ExecutionContext,
    InterpreterExecutionResult,
    InterpreterValidationResult,
)
from .python import PythonInterpreter


class PluckerInterpreter:
    """Fluent chain interpreter backed by pluckit's Plucker + Selection classes.

    Configuration (via ``context.config``):

    - ``code``: default source passed to ``source()`` when the program
      calls it with no argument. A glob pattern, file path, or pluckit
      table reference. If the program calls ``source(code)`` with an
      explicit argument, that overrides the default.
    - ``plugins``: additional pluckit plugins to load beyond the
      default Code and AstViewer plugins.
    """

    name = "plucker"
    description = (
        "Fluent chain expressions over pluckit's Plucker/Selection classes"
    )

    def system_prompt_hint(self) -> str:
        """Interpreter-specialized prompt fragment for plucker."""
        return (
            "You generate a single pluckit fluent chain expression.\n"
            "\n"
            "Shape: source().find(selector).terminal()\n"
            "\n"
            "Entry:\n"
            "  source()              — use the default code source\n"
            '  source("path.py")     — override the source\n'
            "\n"
            "Chain methods:\n"
            "  .find(selector)       — narrow to matching descendants\n"
            "  .callers()            — functions that call this\n"
            "  .filter(predicate)    — filter by condition\n"
            "\n"
            "Terminal operations:\n"
            "  .count()              — return int\n"
            "  .names()              — return list[str]\n"
            "  .view()               — return markdown\n"
            "  .materialize()        — return list[dict]\n"
            "\n"
            "Output ONLY the chain — no code fences, no surrounding Python, no prose."
        )

    def __init__(self) -> None:
        self._python = PythonInterpreter()

    def validate(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterValidationResult:
        """Validate using the restricted Python grammar with a plucker kit.

        The plucker kit exposes ``source()`` as the sole bare callable.
        Any chain rooted in ``source(...)`` passes validation because
        the restricted validator allows method calls on the result of
        allowed-name calls without checking the method names.
        """
        plucker_ctx = self._with_plucker_kit(context)
        return self._python.validate(program, plucker_ctx)

    async def execute(
        self,
        program: str,
        context: ExecutionContext,
    ) -> InterpreterExecutionResult:
        """Execute a fluent chain against a real pluckit Plucker.

        Delegates to :meth:`PythonInterpreter.execute` with a kit whose
        ``source`` callable constructs a live :class:`pluckit.Plucker`.
        The result is whatever the chain's terminal operation returns.
        """
        plucker_ctx = self._with_plucker_kit(context)
        result = await self._python.execute(program, plucker_ctx)
        # Annotate the metadata so downstream consumers can tell a
        # plucker result apart from a raw python result.
        if result.metadata is None:
            result.metadata = {}
        result.metadata["interpreter"] = "plucker"
        return result

    def _with_plucker_kit(self, context: ExecutionContext) -> ExecutionContext:
        """Return a new context with a plucker-specific kit installed.

        The kit is built fresh on each call so the default ``code``
        from ``context.config`` is captured in the ``source`` closure.
        """
        default_code = context.config.get("code")
        extra_plugins = list(context.config.get("plugins", []))
        kit = _build_plucker_kit(
            default_code=default_code,
            extra_plugins=extra_plugins,
        )
        return dataclasses.replace(context, kit=kit)


def _build_plucker_kit(
    default_code: str | None,
    extra_plugins: list,
) -> ResolvedKit:
    """Construct a ResolvedKit whose ``source`` returns a live Plucker.

    The ``source`` callable is a closure over the default code and the
    extra plugins list, so each plucker interpreter invocation gets a
    kit that reflects its current context.
    """
    def source(code: str | None = None) -> Any:
        try:
            from pluckit import Plucker, AstViewer
        except ImportError as e:
            raise ImportError(
                "plucker interpreter requires pluckit to be installed "
                "(pip install ast-pluckit)"
            ) from e
        effective = code if code is not None else default_code
        if effective is None:
            raise ValueError(
                "source() requires a 'code' argument when no default is "
                "set via context.config['code']"
            )
        plugins = [AstViewer, *extra_plugins]
        return Plucker(code=effective, plugins=plugins)

    tools = {
        "source": ToolSpec(
            name="source",
            provider="builtin",
            description=(
                "Create a pluckit Plucker over a source glob, file path, or "
                "table reference. Returns a Plucker whose methods (find, "
                "view, source, etc.) produce Selections you can chain."
            ),
            args=[
                ArgSpec(
                    name="code",
                    type="str",
                    description=(
                        "Source specification: glob pattern, file path, or "
                        "pluckit table reference. Optional if the interpreter "
                        "context provides a default."
                    ),
                ),
            ],
            returns="Plucker",
            grade_w=1,
            effects_ceiling=1,
        ),
    }

    return ResolvedKit(
        tools=tools,
        callables={"source": source},
        grade=Grade(w=1, d=1),
        description=(
            "source(code) -> Plucker: Create a pluckit Plucker over source "
            "code. Chain .find(), .view(), .callers(), and other methods on "
            "the returned Plucker and its Selection results."
        ),
    )
