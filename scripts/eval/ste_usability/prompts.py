"""Build the prompts every model gets: the guide, the skeleton, the files, the task."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from tasks import TASKS  # noqa: E402

SKELETON = '''# /// intent
# goal   = "<the task, one sentence>"
# reads  = []
# writes = []
# output = "none"          # none | printed | value
# ///

# imports. Approved: pathlib (Path), json, re, collections, math,
#          statistics, hashlib, datetime

# implementation
'''


def _workspace() -> str:
    parts = []
    for p in sorted((HERE / "fixtures").rglob("*")):
        if p.is_file():
            rel = p.relative_to(HERE / "fixtures")
            parts.append(f"--- {rel} ---\n{p.read_text()}")
    return "\n".join(parts)


def task_prompt(task: dict) -> str:
    guide = (REPO / "docs/reference/style-guide.md").read_text()
    pre = ""
    if os.environ.get("PROMPT_VARIANT") == "pre":  # the measured pre-prompt, before the guide
        from lackpy.lang.spec import format_preprompt
        pre = f"=== LESSON ===\n{format_preprompt()}\n=== END LESSON ===\n\n"
    return f"""You write lackpy programs. A lackpy program is plain Python 3 that follows the
lackpy style guide below. The validator rejects any program that breaks a rule.

{pre}=== STYLE GUIDE ===
{guide}
=== END STYLE GUIDE ===

Start from this skeleton and fill in the intent block:

{SKELETON}
The program runs with the workspace below as its current directory.

=== WORKSPACE FILES ===
{_workspace()}
=== END WORKSPACE FILES ===

TASK: {task['goal']}

Reply with the complete program in one ```python fenced block and nothing else.
"""


def repair_prompt(task: dict, program: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return task_prompt(task) + f"""
Your previous program was rejected by the validator:

```python
{program}
```

Validator errors:
{listed}

Fix every error. Reply with the complete corrected program in one ```python
fenced block and nothing else.
"""


def extract(reply: str) -> str:
    """The longest fenced block; an unclosed fence runs to the end of the reply."""
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    if m:
        return max(m, key=len).rstrip() + "\n"
    opened = re.search(r"```(?:python|py)?\s*\n(.*)\Z", reply, re.S)
    return (opened.group(1) if opened else reply).rstrip() + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        (out / f"{t['id']}.md").write_text(task_prompt(t))
    print(f"wrote {len(TASKS)} prompts to {out}")
