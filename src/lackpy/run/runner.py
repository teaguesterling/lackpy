"""Restricted runner: AST compile, then exec in the runtime namespace."""

from __future__ import annotations

import ast
from typing import Any, Callable

from .base import ExecutionResult
from .namespace import build_namespace
from .paths import PathPolicy
from .refusal import LackpyRefusal
from .trace import Trace, make_traced


class RestrictedRunner:
    """Execute validated lackpy programs via AST compilation.

    Programs run in the namespace that :func:`build_namespace` builds: the
    ``builtins_policy.ALLOWED`` builtins and nothing else. Kit tools, when a
    kit namespace is given, are wrapped to record timing and results in the
    trace. The runner never raises for anything a program does.
    """

    def run(self, program: str, namespace: dict[str, Callable] | None = None,
            params: dict[str, Any] | None = None,
            kibitzer_session: Any = None, *,
            program_file: str | None = None,
            path_policy: PathPolicy | None = None) -> ExecutionResult:
        """Compile and execute a lackpy program in a restricted namespace.

        The program's last expression (if any) is captured as the output.
        All other top-level assignments are returned in ``variables``.

        Args:
            program: The lackpy program source to execute.
            namespace: Optional kit tools (name -> callable), bound as traced
                globals. Kept for backward compatibility until stepy100 Task 7.
            params: Named parameter values to inject into the namespace.
            kibitzer_session: Optional KibitzerSession for per-call tracking.
            program_file: The program's path, for ``__file__``.
            path_policy: Passed to :func:`build_namespace`; None leaves
                ``pathlib`` refusing.

        Returns:
            An ExecutionResult with success, output, trace, and variables.
            A parse, compile or runtime failure, a ``LackpyRefusal``, a
            ``RecursionError`` or a ``MemoryError`` gives a failed result.
            ``KeyboardInterrupt`` and ``SystemExit`` propagate: the host owns them.
        """
        trace = Trace()
        # Capture print() into a per-run buffer rather than letting it escape to
        # the process stdout. Binding the buffer here (not redirecting the global
        # sys.stdout) keeps capture thread-safe under the threaded execution path
        # and lets the caller recover a value when a generated program ends in
        # ``print(x)`` instead of a bare last expression.
        captured: list[str] = []

        def _capturing_print(*args: Any, sep: str = " ", end: str = "\n",
                             file: Any = None, flush: bool = False) -> None:
            captured.append(sep.join(str(a) for a in args) + end)

        try:
            return self._run(program, namespace or {}, params or {}, kibitzer_session,
                             program_file, path_policy, trace, captured, _capturing_print)
        except (Exception, LackpyRefusal) as e:
            # Exception covers RecursionError and MemoryError (design §8, gate 4).
            return ExecutionResult(success=False, error=_error_text(e),
                                   trace=trace, stdout="".join(captured))

    def _run(self, program: str, namespace: dict[str, Callable], params: dict[str, Any],
             kibitzer_session: Any, program_file: str | None, path_policy: Any,
             trace: Trace, captured: list[str],
             print_fn: Callable[..., None]) -> ExecutionResult:
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

        try:
            code = compile(tree, "<lackpy>", "exec")
        except SyntaxError as e:
            # validate() compiles too, but any drift between them must not raise out of run().
            return ExecutionResult(success=False, error=f"Compile error: {e.msg} (line {e.lineno})",
                                   trace=trace)

        base = build_namespace(program_file=program_file, path_policy=path_policy,
                               print_fn=print_fn)
        # A tool or param must not replace a namespace name, and the namespace
        # must not silently replace the caller's value: refuse the run instead.
        reserved = set(base) | set(base["__builtins__"])
        for kind, names in (("kit tool", namespace), ("param", params)):
            clash = sorted(set(names) & reserved)
            if clash:
                return ExecutionResult(
                    success=False, trace=trace,
                    error=f"{kind} name {clash[0]!r} collides with a lackpy namespace name",
                )
        exec_globals: dict[str, Any] = {
            name: make_traced(name, fn, trace, kibitzer_session=kibitzer_session)
            for name, fn in namespace.items()
        }
        exec_globals.update(params)
        exec_globals.update(base)
        bound = set(exec_globals)

        _run_validated_code(code, exec_globals)

        output = exec_globals.get("__result__")
        variables = {
            k: v for k, v in exec_globals.items()
            if k not in bound and not k.startswith("_")
        }

        return ExecutionResult(success=True, output=output, trace=trace,
                               variables=variables, stdout="".join(captured))


def _error_text(e: BaseException) -> str:
    """Return ``str(e)``, or a fixed text when ``str(e)`` itself raises.

    ``str(KeyError(n))`` raises ``ValueError`` for an int over 4,300 digits;
    the runner must still return a result (review I1).
    """
    try:
        return str(e)
    except (Exception, LackpyRefusal):
        return f"{type(e).__name__} (its message could not be converted to text)"


def _run_validated_code(code: object, globals_dict: dict[str, Any]) -> None:
    exec(code, globals_dict)  # noqa: S102
