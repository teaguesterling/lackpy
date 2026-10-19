"""Every lackpy program shown in the user docs must pass the validator.

The test reads each ```python fence in ``DOC_FILES`` and sorts it by the
fence's ``title`` attribute (pymdownx.highlight renders it as a caption):

- No title: a lackpy program. ``validate(src, allowed_names=set())`` must
  return valid, as ``LackpyService.run_program`` does before it runs.
- ``title="Refused..."``: a program that the docs show *because* lackpy
  refuses it (the validator rejects it, or the PathPolicy refuses its I/O at
  run time). It is skipped by the validator check. Use it sparingly: one fence
  per refusal that a page explains.
- ``title="Python API..."``: host code that calls the ``lackpy`` library
  (``LackpyService``, ``asyncio``). It is not a lackpy program, so it is
  skipped by the validator check.

Every fence, of every kind, must parse as Python.

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


@pytest.mark.parametrize("title,program", PREPROMPT_EXAMPLES, ids=[t for t, _ in PREPROMPT_EXAMPLES])
def test_tutorial_walks_through_each_preprompt_example(title, program):
    programs = [s for w, k, s in FENCES if w.startswith("docs/tutorial.md:") and k == "program"]
    assert program in programs, f"the tutorial lacks the pre-prompt example {title!r} verbatim"
