"""v1 restricted runner: AST compile + restricted exec with traced namespace."""

from __future__ import annotations

import ast
import builtins as _builtins_mod
from typing import Any, Callable

from ..lang.grammar import ALLOWED_BUILTINS
from .base import ExecutionResult
from .trace import Trace, make_traced


def _sort_by(items, key, reverse=False):
    """Sort dicts by key or objects by attribute.

    The attribute name is caller data, so this is a place where a string becomes
    an attribute lookup -- the same shape as str.format. The validator's Step 3.5
    prefix rule cannot see it, because the name never appears as an ast.Attribute.
    Underscore keys are refused here instead. Dict subscripting is untouched: a
    mapping key is data, not a namespace.
    """
    def _get(x):
        if isinstance(x, dict):
            return x[key]
        if isinstance(key, str) and key.startswith("_"):
            raise ValueError(
                f"sort_by: refusing private/dunder attribute {key!r}; "
                f"attribute names starting with '_' are not reachable from a "
                f"lackpy program"
            )
        return getattr(x, key)

    return sorted(items, key=_get, reverse=reverse)


class RestrictedRunner:
    """Execute validated lackpy programs via AST compilation.

    Programs are compiled from validated ASTs and run with empty
    ``__builtins__`` and a controlled namespace. Each tool call is
    wrapped to record timing and results in the trace.
    """

    def run(self, program: str, namespace: dict[str, Callable],
            params: dict[str, Any] | None = None,
            kibitzer_session: Any = None) -> ExecutionResult:
        """Compile and execute a lackpy program in a restricted namespace.

        The program's last expression (if any) is captured as the output.
        All other top-level assignments are returned in ``variables``.

        Args:
            program: The lackpy program source to execute.
            namespace: Mapping of tool names to callables, injected into the
                execution globals alongside allowed builtins.
            params: Named parameter values to inject into the namespace.
            kibitzer_session: Optional KibitzerSession for per-call tracking.

        Returns:
            An ExecutionResult with success, output, trace, and variables.
            Returns a failed result on parse errors or runtime exceptions.
        """
        trace = Trace()
        param_names = set(params.keys()) if params else set()

        # Capture print() into a per-run buffer rather than letting it escape to
        # the process stdout. Binding the buffer here (not redirecting the global
        # sys.stdout) keeps capture thread-safe under the threaded execution path
        # and lets the caller recover a value when a generated program ends in
        # ``print(x)`` instead of a bare last expression.
        captured: list[str] = []

        def _capturing_print(*args: Any, sep: str = " ", end: str = "\n",
                             file: Any = None, flush: bool = False) -> None:
            captured.append(sep.join(str(a) for a in args) + end)

        traced_ns: dict[str, Any] = {}
        for name, fn in namespace.items():
            traced_ns[name] = make_traced(name, fn, trace, kibitzer_session=kibitzer_session)

        for name in ALLOWED_BUILTINS:
            if name == "sort_by":
                traced_ns[name] = _sort_by
            elif name == "print":
                traced_ns[name] = _capturing_print
            else:
                traced_ns[name] = getattr(_builtins_mod, name)

        if params:
            for name, value in params.items():
                traced_ns[name] = value

        try:
            tree = ast.parse(program)
        except SyntaxError as e:
            return ExecutionResult(success=False, error=f"Parse error: {e.msg} (line {e.lineno})", trace=trace)

        has_result = tree.body and isinstance(tree.body[-1], ast.Expr)
        if has_result:
            last_expr = tree.body[-1]
            tree.body[-1] = ast.Assign(
                targets=[ast.Name(id="__result__", ctx=ast.Store())],
                value=last_expr.value,
                lineno=last_expr.lineno,
                col_offset=last_expr.col_offset,
                end_lineno=last_expr.end_lineno,
                end_col_offset=last_expr.end_col_offset,
            )
            ast.fix_missing_locations(tree)

        code = compile(tree, "<lackpy>", "exec")
        exec_globals: dict[str, Any] = {"__builtins__": {}}
        exec_globals.update(traced_ns)

        try:
            _run_validated_code(code, exec_globals)
        except Exception as e:
            return ExecutionResult(success=False, error=str(e), trace=trace,
                                   stdout="".join(captured))

        output = exec_globals.get("__result__")
        variables = {
            k: v for k, v in exec_globals.items()
            if k not in traced_ns and not k.startswith("_") and k not in param_names
        }

        return ExecutionResult(success=True, output=output, trace=trace,
                               variables=variables, stdout="".join(captured))


def _run_validated_code(code: object, globals_dict: dict[str, Any]) -> None:
    exec(code, globals_dict)  # noqa: S102
