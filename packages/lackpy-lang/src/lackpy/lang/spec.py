"""The lackpy language spec as structured data, exposable as an MCP resource."""

from .builtins_policy import ALLOWED
from .dictionary import MODULES, PATH_MEMBERS, UNAPPROVED
from .grammar import ALLOWED_NODES, FORBIDDEN_NODES, FORBIDDEN_NAMES, ALLOWED_BUILTINS
from .validator import validate as _validate  # noqa: F401  (importing it registers every rule)
from .registry import RULES
from .rules_safety import LIMIT_SIGNALS


def format_spec() -> str:
    lines = ["# lackpy Language Specification", ""]
    lines.append("## Allowed AST Nodes")
    for node in sorted(n.__name__ for n in ALLOWED_NODES):
        lines.append(f"  - {node}")
    lines.append("")
    lines.append("## Forbidden AST Nodes")
    for node in sorted(n.__name__ for n in FORBIDDEN_NODES):
        lines.append(f"  - {node}")
    lines.append("")
    lines.append("## Forbidden Names")
    for name in sorted(FORBIDDEN_NAMES):
        lines.append(f"  - {name}")
    lines.append("")
    lines.append("## Allowed Builtins")
    for name in sorted(ALLOWED_BUILTINS):
        lines.append(f"  - {name}")
    return "\n".join(lines)


def get_spec() -> dict:
    """Return the language spec as structured data."""
    return {
        "allowed_nodes": sorted(n.__name__ for n in ALLOWED_NODES),
        "forbidden_nodes": sorted(n.__name__ for n in FORBIDDEN_NODES),
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "allowed_builtins": sorted(ALLOWED_BUILTINS),
        "rules": [{"id": r.id, "tier": r.tier, "summary": r.summary} for r in RULES],
    }


_PREFACE = """\
A lackpy program is a plain Python 3 program. It runs unchanged under CPython with only the standard library, from the workspace directory. It gives the same output there as inside lackpy. Inside lackpy it can only do less. A sandbox refusal is an error, never a different result.

Every rule has one of two tiers. A *safety* rule protects the sandbox, and nothing waives it. A *style* rule keeps programs short and uniform. A waiver can lift a style rule.

A waiver is a comment of the form `# ste: <rule-id>, <why>`. The why is required. A waiver covers one statement. Put it at the end of the statement's first line, or on the line directly above. For a compound statement, a waiver covers the header only, not the body.

A waiver names style rules only. A waiver that names a safety rule gives `meta.waiver-safety`. A misplaced or malformed waiver gives `meta.waiver-unknown`.
"""


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def format_style_guide() -> str:
    """Render the style guide (Parts 1 and 2) from the registry and the dictionary."""
    lines = ["# lackpy Style Guide", "",
             _PREFACE.rstrip(), "",
             "## Part 1. Program rules", ""]
    for r in RULES:
        lines += [f"### {r.id}", "", r.summary, "", f"**Tier:** {r.tier}", "",
                  f"**Why:** {r.rationale}", "",
                  "**Not approved:**", "", "```python", r.bad.rstrip("\n"), "```", "",
                  "**Approved:**", "", "```python", r.good.rstrip("\n"), "```", ""]
    functions = sorted(n for n, k in ALLOWED.items() if k == "function")
    uncatchable = ", ".join(f"`{n}`" for n in sorted(LIMIT_SIGNALS | {"Exception"}))
    lines += ["## Part 2. Dictionary", "", "### Approved builtins", ""]
    lines += _table(["Kind", "Names"], [
        ["Functions and types", ", ".join(f"`{n}`" for n in functions)],
        ["Exception classes", "Every built-in subclass of `Exception`."],
    ]) + [""]
    lines += [f"`safety.except-base` forbids catching {uncatchable}.", ""]
    lines += ["### Approved modules and members", ""]
    lines += _table(["Module", "Members"], [
        [f"`{m}`", ", ".join(f"`{x}`" for x in sorted(members))] for m, members in MODULES.items()
    ]) + [""]
    lines += ["### Path members", "",
             ", ".join(f"`{x}`" for x in sorted(PATH_MEMBERS)), "",
             "The `/` operator joins paths, for example `Path(d) / name`.", ""]
    lines += ["### Unapproved words", ""]
    lines += _table(["Word", "Approved alternative"],
                    [[f"`{w}`", hint] for w, hint in UNAPPROVED.items()]) + [""]
    lines += ["Part 3 (intent rules) arrives with Plan 3.", ""]
    return "\n".join(lines)


# --- the pre-prompt -------------------------------------------------------------
#
# A short teaching text that goes BEFORE the full guide. The guide is a reference;
# this names the habits a model brings from ordinary Python and gives the approved
# form for each. The order of HABITS is measured, not guessed: it follows the
# violation counts of Qwen3-30B-A3B-Instruct and haiku on the 30-task usability
# eval of 2026-10-18 (scripts/eval/ste_usability/FINDINGS.md, finding 4).

HABITS: list[tuple[str, str]] = [
    ("`except Exception:` or a bare `except:`",
     "name the error you expect: `except KeyError:`"),
    ("`exit()` or `sys.exit()` to stop on a problem",
     "state the condition: `assert cond, \"why it must hold\"`"),
    ("`raise SomeError(...)` when a check fails",
     "`assert cond, \"why it must hold\"`"),
    ("`open(p)` or `with open(p) as f:`",
     "`text = Path(p).read_text()`"),
    ("`json.load(f)` / `json.dump(obj, f)`",
     "`data = json.loads(text)`, and `Path(p).write_text(json.dumps(obj))`"),
    ("`finally:` or `else:` on a `try`",
     "put cleanup code after the try block; lackpy has no finally"),
    ("`while`, `class`, `global`, `lambda` stored in a name",
     "a for loop, a dict, a returned value, or a top-level def"),
]

PREPROMPT_EXAMPLES: list[tuple[str, str]] = [
    ("Patch one occurrence in a file", '''from pathlib import Path

path = Path("notes.md")
text = path.read_text()
assert text.count("Status: draft") == 1, "the status line occurs once"
path.write_text(text.replace("Status: draft", "Status: final"))
'''),
    ("Read JSON lines, skip bad ones, count", '''import json
from collections import Counter
from pathlib import Path

counts = Counter()
text = Path("data/rows.jsonl").read_text()
for line in text.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    counts[row["status"]] += 1
for status in sorted(counts):
    print(status, counts[status])
'''),
    ("Scan files with a helper function", '''from pathlib import Path


def todo_count(text):
    return text.count("TODO")


paths = Path("src").glob("*.cpp")
for p in sorted(paths):
    text = p.read_text()
    print(p.name, todo_count(text))
'''),
]

_PREPROMPT_HEAD = """\
# Writing lackpy

A lackpy program is plain Python 3. It uses only the standard library, and it
does file I/O through `pathlib.Path`. A validator checks every program before it
runs and rejects any rule it breaks. The full style guide follows this page.

Approved imports: `pathlib` (`Path`), `json`, `re`, `collections`, `math`,
`statistics`, `hashlib`, `datetime`. Nothing else.

## Habits to replace

These come from ordinary Python. The validator rejects each one.
"""

_PREPROMPT_SHAPE = """\
## Shape

- In a chain, the I/O call comes first: `Path(p).read_text().strip()`.
- A call nests at most 3 deep: `print(len(sorted(x)))` is fine.
- A `try` names its error, has at most 3 statements, and its handler ends in
  `continue`, `return`, `pass`, or assignments to every name the body assigns.
- A `def` is top level, has at most 12 statements, and is defined before use.
- Every `assert` has a message.
"""


def format_preprompt() -> str:
    """A one-page lesson on lackpy, to put before the full guide."""
    lines = [_PREPROMPT_HEAD]
    lines += _table(["Do not write", "Write instead"], [[a, b] for a, b in HABITS])
    lines += ["", _PREPROMPT_SHAPE, "## Worked examples", ""]
    for title, program in PREPROMPT_EXAMPLES:
        lines += [f"### {title}", "", "```python", program.rstrip(), "```", ""]
    return "\n".join(lines)
