"""Restricted runner: AST compile, then exec in the runtime namespace."""

from __future__ import annotations

import ast
import types
from typing import Any, Callable

from .base import ExecutionResult
from .modules import ModuleProxy
from .namespace import build_namespace
from .paths import PathPolicy
from .paths.proxy import is_path_stand_in
from .refusal import LackpyRefusal
from .trace import Trace, error_text, make_traced


class RestrictedRunner:
    """Execute validated lackpy programs via AST compilation.

    Programs run in the namespace that :func:`build_namespace` builds: the
    ``builtins_policy.ALLOWED`` builtins and nothing else. Kit tools, when a
    kit namespace is given, are wrapped to record timing and results in the
    trace. The runner never raises for anything a program does.

    Params (design §4.1, D15): the program's one top-level
    ``params = {"name": <literal>, ...}`` assignment holds the defaults. Passed
    params replace that dict's values in the AST before compile, so the program
    reads ``params["name"]`` as it would under python3. No param binds a bare
    name. With no params passed, the program runs unchanged.
    """

    def run(self, program: str, namespace: dict[str, Callable] | None = None,
            params: dict[str, Any] | None = None,
            kibitzer_session: Any = None, *,
            program_file: str | None = None,
            path_policy: PathPolicy | None = None) -> ExecutionResult:
        """Compile and execute a lackpy program in a restricted namespace.

        The program's last expression (if any) is captured as the output.
        The other top-level names that hold data are returned in
        ``variables``: callables, classes, modules, module proxies and ``Path``
        stand-ins are left out (design §8, gate 7).

        Args:
            program: The lackpy program source to execute.
            namespace: Optional kit tools (name -> callable), bound as traced
                globals. Kept for backward compatibility until stepy100 Task 7.
            params: Values that replace the defaults in the program's
                ``params`` literal. An unknown key, a value of another type
                (``int`` is accepted for a ``float`` default; ``bool`` is not
                ``int``), a value with no literal form, or a program without the
                literal gives a failed result that names the key.
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
            return ExecutionResult(success=False, error=error_text(e),
                                   trace=trace, stdout="".join(captured))

    def _run(self, program: str, namespace: dict[str, Callable], params: dict[str, Any],
             kibitzer_session: Any, program_file: str | None, path_policy: Any,
             trace: Trace, captured: list[str],
             print_fn: Callable[..., None]) -> ExecutionResult:
        try:
            tree = ast.parse(program)
        except SyntaxError as e:
            return ExecutionResult(success=False, error=f"Parse error: {e.msg} (line {e.lineno})", trace=trace)

        if params:
            params_error = _apply_params(tree, params)
            if params_error is not None:
                return ExecutionResult(success=False, error=params_error, trace=trace)

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
                               print_fn=print_fn, trace=trace)
        # A kit tool must not replace a namespace name, and the namespace must
        # not silently replace the caller's tool: refuse the run instead.
        reserved = set(base) | set(base["__builtins__"])
        clash = sorted(set(namespace) & reserved)
        if clash:
            return ExecutionResult(
                success=False, trace=trace,
                error=f"kit tool name {clash[0]!r} collides with a lackpy namespace name",
            )
        exec_globals: dict[str, Any] = {
            name: make_traced(name, fn, trace, kibitzer_session=kibitzer_session)
            for name, fn in namespace.items()
        }
        exec_globals.update(base)
        bound = set(exec_globals)

        _run_validated_code(code, exec_globals)

        output = exec_globals.get("__result__")
        variables = {
            k: v for k, v in exec_globals.items()
            if k not in bound and not k.startswith("_") and _is_data(v)
        }

        return ExecutionResult(success=True, output=output, trace=trace,
                               variables=variables, stdout="".join(captured))


def _is_data(value: Any) -> bool:
    """Whether a program variable is data, for ``variables`` (gate 7).

    Only ``type(value)`` is inspected, never an attribute of ``value``: the
    ``type`` stand-in and its views refuse every attribute, ``__class__`` too.
    ``callable()`` drops functions, classes, the ``type`` stand-in and its
    views; the rest drops modules, module proxies and ``Path`` stand-ins.
    """
    kind = type(value)
    if callable(value) or issubclass(kind, (type, types.ModuleType, ModuleProxy)):
        return False
    return not is_path_stand_in(value)


class _NotLiteral(Exception):
    pass


_SCALARS = (str, bytes, int, float, complex, bool, type(None))


def _literal_node(value: Any) -> ast.expr:
    """Return an AST literal node for ``value``; raise ``_NotLiteral`` if it has none (ruling c)."""
    kind = type(value)
    if kind in _SCALARS:
        return ast.Constant(value=value)
    if kind is list:
        return ast.List(elts=[_literal_node(v) for v in value], ctx=ast.Load())
    if kind is tuple:
        return ast.Tuple(elts=[_literal_node(v) for v in value], ctx=ast.Load())
    if kind is set:
        return ast.Set(elts=[_literal_node(v) for v in value])
    if kind is dict:
        return ast.Dict(keys=[_literal_node(k) for k in value],
                        values=[_literal_node(v) for v in value.values()])
    raise _NotLiteral(kind.__name__)


def _binds_params(node: ast.AST) -> bool:
    """Whether ``node`` binds (or deletes) the name ``params``."""
    if isinstance(node, ast.Name):
        return node.id == "params" and not isinstance(node.ctx, ast.Load)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)):
        return node.name == "params"
    if isinstance(node, ast.alias):
        return (node.asname or node.name.split(".")[0]) == "params"
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return "params" in node.names
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return node.name == "params"
    if isinstance(node, ast.MatchMapping):
        return node.rest == "params"
    return False


def _params_literal(tree: ast.Module) -> ast.Dict | None:
    """Return the dict of the top-level ``params = {"name": <literal>, ...}``, or None."""
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "params"):
            literal = stmt.value
            if not isinstance(literal, ast.Dict):
                return None
            for key, value in zip(literal.keys, literal.values):
                if not (isinstance(key, ast.Constant) and type(key.value) is str):
                    return None  # also a `**mapping` entry, whose key is None
                try:
                    ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    return None
            return literal
    return None


def _apply_params(tree: ast.Module, params: dict[str, Any]) -> str | None:
    """Replace the defaults in the program's ``params`` literal; return an error text or None."""
    keys = ", ".join(repr(k) for k in params)
    bindings = sum(1 for node in ast.walk(tree) if _binds_params(node))
    if bindings == 0:
        return (f"params {keys} were passed, but the program has no top-level "
                f"params = {{...}} literal to hold them")
    literal = _params_literal(tree)
    if bindings > 1 or literal is None:
        return (f"params {keys} were passed, but the program does not bind params exactly once, "
                f'as a top-level params = {{"name": <literal>, ...}}')
    for key, value in params.items():
        slots = [i for i, k in enumerate(literal.keys) if k.value == key]  # type: ignore[union-attr]
        if not slots:
            return f"param {key!r} is not a key of the program's params literal"
        default = ast.literal_eval(literal.values[slots[-1]])
        expected = type(default)
        if not (type(value) is expected or (expected is float and type(value) is int)):
            return f"param {key!r} must be {expected.__name__} like its default, not {type(value).__name__}"
        try:
            for i in slots:
                literal.values[i] = ast.copy_location(_literal_node(value), literal.values[i])
        except (_NotLiteral, RecursionError):
            return f"param {key!r} has a value that cannot be written as a Python literal"
    ast.fix_missing_locations(tree)
    return None


def _run_validated_code(code: object, globals_dict: dict[str, Any]) -> None:
    exec(code, globals_dict)  # noqa: S102
