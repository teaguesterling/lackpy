"""Compile parsed cells into Python source code.

Each cell type maps to a compiler function in _COMPILERS. The kernel
calls these via _COMPILERS[cell.cell_type](cell) to get Python source,
which it then passes through static_analysis.check_cell() and runs
through the document's CellSession (validated, restricted namespace).

Compilation rules:
  prose    → segments (prose_segments), not code: text, and one-line
             {expr} sources the kernel evaluates with CellSession.eval_expr.
             compile_cell gives the expressions, one per line, as analysis
             source only.
  code     → pass through
  @hidden  → pass through (same as code; visibility is the driver's concern)
  @gather  → pass through
  @continue → sentinel function call that the kernel detects
  @read    → print(read_file(path))
  @write   → write_file(path, content)
  @diff    → apply_diff(path, diff_text)
  @scratch → the body; the kernel prints the new-name summary

The _split_interpolation() function handles prose interpolation by
tracking brace depth and string context to correctly parse nested
expressions like {chr(10).join([f"...{x}..." for x in items])}.
"""

from __future__ import annotations

import re
import ast
from collections.abc import Callable
from dataclasses import dataclass

from .parser import Cell, ParseResult

CONTINUE_SENTINEL = "__literate_continue__()"

#: Call names the compiler emits that are runtime-internal plumbing, not tool
#: calls the writer planned. Callers that enumerate a compiled document's
#: called names for policy (e.g. kibitzer mode validation) must filter these.
LITERATE_RUNTIME_INTERNALS: frozenset[str] = frozenset({"__literate_continue__"})

_INTERPOLATION_START = re.compile(r"\{[A-Za-z_]")


def _split_interpolation(content: str) -> list[tuple[str, bool]]:
    """Split prose into (text, is_expression) parts with proper brace matching.

    Handles nested braces so expressions like {chr(10).join([f"...{x}..."])}
    are captured as a single expression rather than splitting at the first }.
    Doubled braces ({{ and }}) produce literal { and } characters.
    """
    parts: list[tuple[str, bool]] = []
    i = 0
    n = len(content)
    literal_start = 0

    while i < n:
        if content[i] == "{" and i + 1 < n:
            if content[i + 1] == "{":
                # {{ → literal {
                parts.append((content[literal_start:i] + "{", False))
                i += 2
                literal_start = i
                continue
            if content[i + 1].isalpha() or content[i + 1] == "_":
                if i > literal_start:
                    parts.append((content[literal_start:i], False))
                depth = 1
                j = i + 1
                in_string: str | None = None
                while j < n and depth > 0:
                    ch = content[j]
                    if in_string:
                        if ch == "\\" and j + 1 < n:
                            j += 2
                            continue
                        if ch == in_string:
                            in_string = None
                    elif ch in ('"', "'"):
                        in_string = ch
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                    j += 1
                if depth == 0:
                    parts.append((content[i + 1 : j - 1], True))
                    literal_start = j
                    i = j
                else:
                    i += 1
            else:
                i += 1
        elif content[i] == "}" and i + 1 < n and content[i + 1] == "}":
            # }} → literal }
            parts.append((content[literal_start:i] + "}", False))
            i += 2
            literal_start = i
        else:
            i += 1

    if literal_start < n:
        parts.append((content[literal_start:], False))

    # Merge adjacent literal parts (produced by escape sequences)
    merged: list[tuple[str, bool]] = []
    for text, is_expr in parts:
        if not is_expr and merged and not merged[-1][1]:
            merged[-1] = (merged[-1][0] + text, False)
        else:
            merged.append((text, is_expr))

    return merged


@dataclass(frozen=True)
class ProseSegment:
    """One piece of a prose cell (design D3): text, or an expression.

    ``text`` is rendered as it is. ``expr`` is one line of lackpy source that
    the kernel evaluates with ``CellSession.eval_expr``. When ``display`` is
    True the value goes through the bounded display (``display.py``); when
    False, ``expr`` is an f-string (the interpolation had a conversion or a
    format spec, as in ``{total:.2f}``) whose value is already the text.
    """

    text: str | None = None
    expr: str | None = None
    display: bool = True


def _expr_segment(text: str) -> ProseSegment:
    """The segment for one ``{text}`` interpolation.

    ``text`` is parsed the way Python parses the inside of ``f"{...}"``. A
    plain expression becomes its own one-line source; one with a conversion
    (``!r``), a format spec (``:.2f``) or a ``=`` becomes the one-line
    f-string, so Python formats it inside the session. Text that does not
    parse is kept as it is: the kernel's static check reports the syntax error.
    """
    for quote in ('"""', "'''"):
        try:
            node = ast.parse(f"f{quote}{{{text}}}{quote}", mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.JoinedStr):
            continue
        values = node.values
        if (len(values) == 1 and isinstance(values[0], ast.FormattedValue)
                and values[0].conversion == -1 and values[0].format_spec is None):
            return ProseSegment(expr=ast.unparse(values[0].value), display=True)
        return ProseSegment(expr=ast.unparse(node), display=False)
    return ProseSegment(expr=text, display=True)


def prose_segments(cell: Cell) -> list[ProseSegment]:
    """Split a prose cell into text and expression segments (design D3).

    Whitespace-only prose has no segments (it renders as a blank line). Prose
    with no ``{name...`` interpolation start is one text segment, as it is.
    """
    content = cell.content
    if not content.strip():
        return []
    if not _INTERPOLATION_START.search(content):
        return [ProseSegment(text=content)]
    return [_expr_segment(text) if is_expr else ProseSegment(text=text)
            for text, is_expr in _split_interpolation(content)]


def _compile_prose(cell: Cell) -> str:
    """The prose cell's expressions, one per line: ANALYSIS source only.

    Static analysis (references for holes and the dependency graph, the
    effect gate) reads it. Nothing executes it: the kernel renders prose from
    :func:`prose_segments`, evaluating each expression in the session.
    """
    return "\n".join(seg.expr for seg in prose_segments(cell) if seg.expr is not None)


def _compile_code(cell: Cell) -> str:
    return cell.content


def _compile_hidden(cell: Cell) -> str:
    return cell.content


def _compile_gather(cell: Cell) -> str:
    return cell.content


def _compile_continue(cell: Cell) -> str:
    return CONTINUE_SENTINEL


def _compile_read(cell: Cell) -> str:
    path = cell.annotation_args.get("path", "")
    return f"print(read_file({repr(path)}))"


def _compile_write(cell: Cell) -> str:
    path = cell.annotation_args.get("path", "")
    return f"write_file({repr(path)}, {repr(cell.content)})"


def _compile_diff(cell: Cell) -> str:
    path = cell.annotation_args.get("path", "")
    return f"apply_diff({repr(path)}, {repr(cell.content)})"


def _compile_scratch(cell: Cell) -> str:
    # The body only: the kernel runs it in the session and writes the
    # ``[scratch: name=type, ...]`` summary on the host side.
    return cell.content


_COMPILERS: dict[str, Callable[[Cell], str]] = {
    "prose": _compile_prose,
    "code": _compile_code,
    "hidden": _compile_hidden,
    "gather": _compile_gather,
    "continue": _compile_continue,
    "read": _compile_read,
    "write": _compile_write,
    "diff": _compile_diff,
    "scratch": _compile_scratch,
}


def compile_cell(cell: Cell) -> str:
    """Compile a single cell to Python source.

    Raises ``ValueError`` on an unknown cell type (the parser only emits known
    types, so this is a guard, not an expected path). Used by the effect-ceiling
    gate to classify cells one at a time."""
    compiler = _COMPILERS.get(cell.cell_type)
    if compiler is None:
        raise ValueError(f"Unknown cell type: {cell.cell_type}")
    return compiler(cell)


def compile_cells(parse_result: ParseResult) -> str:
    """Compile a parsed document into a single Python program."""
    parts: list[str] = []
    for cell in parse_result.cells:
        compiler = _COMPILERS.get(cell.cell_type)
        if compiler is None:
            raise ValueError(f"Unknown cell type: {cell.cell_type}")
        compiled = compiler(cell)
        if compiled.strip():
            parts.append(compiled)
    return "\n".join(parts)


def compile_document(document: str) -> str:
    """Parse and compile a literate document to Python in one step."""
    from .parser import parse
    result = parse(document)
    if result.errors:
        error_msg = "; ".join(result.errors)
        raise ValueError(f"Parse errors: {error_msg}")
    return compile_cells(result)
