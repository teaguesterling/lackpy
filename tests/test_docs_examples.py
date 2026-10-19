"""Every lackpy program shown in the user docs must pass the validator.

The test reads each ```python fence in ``DOC_FILES`` and sorts it by the
fence's ``title`` attribute (pymdownx.highlight renders it as a caption):

- No title: a lackpy program. ``validate(src, allowed_names=set())`` must
  return valid, as ``LackpyService.run_program`` does before it runs.
- ``title="Refused..."``: a program that the docs show *because* lackpy
  refuses it (the validator rejects it, or the PathPolicy refuses its I/O at
  run time). It is skipped by the validator check. Instead, it must fail
  ``validate``, or fail at run time with a refusal ("... is not allowed in
  lackpy") in a workspace that has ``REFUSAL_CONFIG``, a ``.env`` and a
  ``notes.md``. Use it sparingly: one fence per refusal that a page explains.
- ``title="Python API..."``: host code that calls the ``lackpy`` library
  (``LackpyService``, ``asyncio``). It is not a lackpy program, so it is
  skipped by the validator check.

Every fence, of every kind, must parse as Python.

``docs/concepts/literate.md`` also holds literate documents, in ````markdown
fences (four backticks, so the ```lackpy cells nest). Each document is parsed with the literate
parser, and its code cells are validated one by one as cumulative programs: cell
N is valid when cells 1..N, joined as ``compile_cells`` joins them, validate as one lackpy program.
A document titled ``Refused...`` must really be refused: a cell fails validation, or
the run ends in a refusal. A document that holds no code cell does not count.

The tutorial walks through the pre-prompt's worked examples, so it must hold
each ``PREPROMPT_EXAMPLES`` program verbatim; a change to the pre-prompt fails
here until the tutorial follows it.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from lackpy.lang import validate
from lackpy.lang.spec import PREPROMPT_EXAMPLES

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [
    "README.md",
    "docs/index.md",
    "docs/getting-started.md",
    "docs/tutorial.md",
    "docs/concepts/language-spec.md",
    "docs/concepts/literate.md",
    "docs/concepts/paths.md",
    "docs/concepts/profiles.md",
    "docs/concepts/policy-layer.md",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs" / "tools").glob("*.md")),
]

_FENCE = re.compile(r"^(?P<indent>[ \t]*)```python(?P<info>[^\n]*)\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
                    re.M | re.S)
_TITLE = re.compile(r'title="([^"]*)"')


def _fences():
    out = []
    for rel in DOC_FILES:
        text = (ROOT / rel).read_text()
        for m in _FENCE.finditer(text):
            title = _TITLE.search(m.group("info"))
            kind = "program"
            if title and title.group(1).startswith("Refused"):
                kind = "refused"
            elif title and title.group(1).startswith("Python API"):
                kind = "host"
            line = text.count("\n", 0, m.start()) + 1
            out.append((f"{rel}:{line}", kind, textwrap.dedent(m.group("body"))))
    return out


FENCES = _fences()


def test_the_docs_hold_programs():
    # Absence of signal is not success: a broken fence regex would find nothing.
    kinds = [k for _, k, _ in FENCES]
    assert kinds.count("program") >= 15
    files = {where.split(":")[0] for where, k, _ in FENCES if k == "program"}
    assert {"README.md", "docs/tutorial.md", "docs/concepts/paths.md"} <= files


@pytest.mark.parametrize("where,kind,src", FENCES, ids=[w for w, _, _ in FENCES])
def test_fence_parses(where, kind, src):
    ast.parse(src)


@pytest.mark.parametrize("where,src", [(w, s) for w, k, s in FENCES if k == "program"],
                         ids=[w for w, k, _ in FENCES if k == "program"])
def test_program_fence_validates(where, src):
    result = validate(src, allowed_names=set())
    assert result.valid, f"{where}: {result.errors}"


REFUSAL_CONFIG = '[paths]\ndeny = ["Read(.env)"]\n'  # the configuration paths.md shows


@pytest.mark.parametrize("where,src", [(w, s) for w, k, s in FENCES if k == "refused"],
                         ids=[w for w, k, _ in FENCES if k == "refused"])
def test_refused_fence_is_refused(where, src, tmp_path):
    # A stale "Refused" marker must not hide a program that lackpy now accepts.
    if not validate(src, allowed_names=set()).valid:
        return
    import asyncio

    from lackpy.service import LackpyService

    (tmp_path / ".lackpy").mkdir()
    (tmp_path / ".lackpy" / "config.toml").write_text(REFUSAL_CONFIG)
    (tmp_path / ".env").write_text("SECRET=1\n")
    (tmp_path / "notes.md").write_text("Status: draft\n")
    result = asyncio.run(LackpyService(workspace=tmp_path).run_program(src))
    assert not result.success, f"{where}: marked Refused, but it ran"
    assert "is not allowed in lackpy" in (result.error or ""), f"{where}: failed, not refused: {result.error}"


@pytest.mark.parametrize("title,program", PREPROMPT_EXAMPLES, ids=[t for t, _ in PREPROMPT_EXAMPLES])
def test_tutorial_walks_through_each_preprompt_example(title, program):
    programs = [s for w, k, s in FENCES if w.startswith("docs/tutorial.md:") and k == "program"]
    assert program in programs, f"the tutorial lacks the pre-prompt example {title!r} verbatim"


def test_cli_reference_teaches_no_kit_flags():
    # Final review M6: --tools and kit-form --profile values now fail.
    text = (ROOT / "docs" / "reference" / "cli.md").read_text()
    fences = re.findall(r"^```bash\n(.*?)^```", text, re.M | re.S)
    assert fences
    for body in fences:
        assert "--tools" not in body
        assert "lackpyctl profile" not in body
        for value in re.findall(r"--profile ([^\s\]]+)", body):
            assert value in ("NAME", "fast"), value  # a [profiles.<name>] table, never a kit


def test_cascade_default_examples_are_valid_programs():
    # Final review M6: the cascade's few-shot examples taught squackit kit calls.
    from lackpy.infer.providers.cascade import _DEFAULT_EXAMPLES

    examples: list[list[str]] = []
    for line in _DEFAULT_EXAMPLES.splitlines():
        if " -> " in line:
            examples.append([line.split(" -> ", 1)[1]])
        else:
            examples[-1].append(line)
    assert len(examples) >= 5
    for lines in examples:
        program = "\n".join(lines)
        result = validate(program, allowed_names=set())
        assert result.valid, (program, result.errors)


_DOC_FENCE = re.compile(r"^````markdown(?P<info>[^\n]*)\n(?P<body>.*?)^````[ \t]*$", re.M | re.S)
LITERATE_PAGE = "docs/concepts/literate.md"


def _literate_docs():
    text = (ROOT / LITERATE_PAGE).read_text()
    out = []
    for m in _DOC_FENCE.finditer(text):
        title = _TITLE.search(m.group("info"))
        kind = "refused" if title and title.group(1).startswith("Refused") else "document"
        out.append((f"{LITERATE_PAGE}:{text.count(chr(10), 0, m.start()) + 1}", kind, m.group("body")))
    return out


LITERATE_DOCS = _literate_docs()


def _first_invalid_cell(document: str):
    """The first code cell whose cumulative program does not validate, else None.

    The cumulative program of cell N is ``compile_cells`` over the code cells 1..N:
    the same join the kernel and the gate-3 equivalence test use.
    """
    from lackpy.interpreters.literate.compiler import compile_cells
    from lackpy.interpreters.literate.parser import ParseResult, parse

    parsed = parse(document)
    assert not parsed.errors, parsed.errors
    code = [c for c in parsed.cells if c.cell_type != "prose"]
    for n in range(1, len(code) + 1):
        program = compile_cells(ParseResult(frontmatter=parsed.frontmatter, cells=code[:n]))
        result = validate(program, allowed_names=set())
        if not result.valid:
            return n, result.errors
    return None


def test_the_literate_page_holds_documents():
    # Absence of signal is not success: a broken fence regex would find nothing.
    kinds = [k for _, k, _ in LITERATE_DOCS]
    assert kinds.count("document") >= 5
    assert kinds.count("refused") >= 3


@pytest.mark.parametrize("where,src", [(w, s) for w, k, s in LITERATE_DOCS if k == "document"],
                         ids=[w for w, k, _ in LITERATE_DOCS if k == "document"])
def test_literate_document_cells_validate_cumulatively(where, src):
    from lackpy.interpreters.literate.parser import parse

    assert any(c.cell_type != "prose" for c in parse(src).cells), f"{where}: no code cell"
    assert _first_invalid_cell(src) is None, f"{where}: {_first_invalid_cell(src)}"


@pytest.mark.parametrize("where,src", [(w, s) for w, k, s in LITERATE_DOCS if k == "refused"],
                         ids=[w for w, k, _ in LITERATE_DOCS if k == "refused"])
def test_refused_literate_document_is_refused(where, src, tmp_path):
    # A stale "Refused" marker must not hide a document that lackpy now accepts.
    if _first_invalid_cell(src) is not None:
        return
    import asyncio

    from lackpy.profiles import Profile
    from lackpy.run.paths import load_policy
    from lackpy.service import LackpyService

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.md").write_text("# Notes\n")
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    service = LackpyService(workspace=ws)
    service._load_path_policy = lambda read_only=False: load_policy(ws, home=home, managed=None)
    result = asyncio.run(service.run_program(src, profile=Profile(execution="literate")))
    assert not result.success, f"{where}: marked Refused, but it ran"
    assert "is not allowed in lackpy" in (result.error or ""), f"{where}: failed, not refused: {result.error}"
