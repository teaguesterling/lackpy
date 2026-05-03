"""Compile a cell sequence into a single Python program.

Each cell type has its own compilation rule:
- prose → print(f"...") with {expr} interpolation
- code → pass through
- @hidden → pass through (no print wrapping)
- @gather → pass through
- @continue → emit sentinel that executor recognizes as pause point
- @read(path) → print(read_file(path))
- @write(path) → write_file(path, content)
- @diff(path) → apply_diff(path, diff_text)
- @scratch → capture new variables, emit summary
"""

from __future__ import annotations

import re
from textwrap import indent

from .parser import Cell, ParseResult

CONTINUE_SENTINEL = "__literate_continue__()"

_INTERPOLATION = re.compile(r"\{([^}]+)\}")


def _compile_prose(cell: Cell) -> str:
    content = cell.content
    if not content.strip():
        return "print()"

    has_interpolation = _INTERPOLATION.search(content)
    if has_interpolation:
        return f"print(f{repr(content)})"
    return f"print({repr(content)})"


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
    body = cell.content
    escaped_body = indent(body, "    ") if body else "    pass"
    return (
        "_scratch_names_before = set(locals().keys())\n"
        f"if True:\n{escaped_body}\n"
        '_scratch_names_after = set(locals().keys()) - _scratch_names_before - {"_scratch_names_before"}\n'
        "_scratch_parts = []\n"
        "for _sn in sorted(_scratch_names_after):\n"
        "    _sv = locals()[_sn]\n"
        '    _scratch_parts.append(f"{_sn}={type(_sv).__name__}")\n'
        'print(f"[scratch: {\', \'.join(_scratch_parts)}]")\n'
        "del _scratch_names_before, _scratch_names_after, _scratch_parts"
    )


_COMPILERS: dict[str, callable] = {
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
