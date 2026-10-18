"""``# ste: <rule-id>, <why>`` waivers.

A waiver removes a ``style.*`` violation on its own line, or on the next line
when the comment stands alone. Safety rules are never waivable. Comments are
read with ``tokenize``, so a string that contains ``# ste:`` is not a waiver.
"""

from __future__ import annotations

import dataclasses
import io
import re
import tokenize

from .registry import RULES, Violation, Waiver

_WAIVER = re.compile(r"#\s*ste:(.*)$", re.DOTALL)


def scan_waivers(source: str) -> tuple[list[Waiver], list[Violation]]:
    """Return (valid waivers, ``meta.waiver-*`` violations) for ``source``."""
    tiers = {r.id: r.tier for r in RULES}
    waivers: list[Waiver] = []
    meta: list[Violation] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return waivers, meta
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        m = _WAIVER.match(tok.string)
        if not m:
            continue
        line = tok.start[0]
        rule_id, _, why = (part.strip() for part in m.group(1).partition(","))
        form = "Expected '# ste: <rule-id>, <why>'"
        if not rule_id or not why:
            meta.append(Violation("meta.waiver-unknown", line,
                                  f"Malformed waiver at line {line}", hint=form))
        elif tiers.get(rule_id) == "safety":
            meta.append(Violation("meta.waiver-safety", line,
                                  f"Waiver at line {line} names safety rule {rule_id}; safety rules cannot be waived"))
        elif rule_id not in tiers:
            meta.append(Violation("meta.waiver-unknown", line,
                                  f"Waiver at line {line} names unknown rule {rule_id}", hint=form))
        else:
            waivers.append(Waiver(rule_id, line, why))
    return waivers, meta


def parse_waivers(source: str) -> list[Waiver]:
    """Return the valid ``# ste:`` waivers in ``source``."""
    return scan_waivers(source)[0]


def apply_waivers(
    violations: list[Violation], waivers: list[Waiver], source: str
) -> tuple[list[Violation], list[Waiver]]:
    """Drop the violations that a waiver covers. Mark each waiver ``used`` or not."""
    lines = source.splitlines()
    covered: dict[tuple[str, int], int] = {}  # (rule id, violation line) -> waiver index
    for i, w in enumerate(waivers):
        covered[(w.rule_id, w.line)] = i
        if lines[w.line - 1].lstrip().startswith("#"):  # standalone: covers the next line too
            covered.setdefault((w.rule_id, w.line + 1), i)
    used: set[int] = set()
    kept = []
    for v in violations:
        i = covered.get((v.rule_id, v.line))
        if i is None:
            kept.append(v)
        else:
            used.add(i)
    return kept, [dataclasses.replace(w, used=i in used) for i, w in enumerate(waivers)]
