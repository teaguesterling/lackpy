"""``# ste: <rule-id>, <why>`` waivers.

A waiver attaches to one statement: a trailing comment on the statement's first
line, or a standalone comment on the line directly above it. It removes the
``style.*`` violations of its rule anywhere in that statement. For a compound
statement it covers the header lines only, not the body. A ``# ste:`` comment
anywhere else is ``meta.waiver-unknown`` (misplaced). Safety rules are never waivable. Comments are
read with ``tokenize``, so a string that contains ``# ste:`` is not a waiver.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import re
import tokenize

from .registry import RULES, Violation, Waiver

_WAIVER = re.compile(r"#\s*ste:(.*)$", re.DOTALL)


_NESTED = re.compile(r"#\s*ste:")
_FORM = "Expected '# ste: <rule-id>, <why>'"
_PLACE = "Put the waiver on the statement's first line or on the line above"


def _statements(tree: ast.AST) -> list[tuple[int, int, int]]:
    """(first line, header end line, full end line) of every statement."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            body = getattr(node, "body", None)
            header_end = node.end_lineno
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                header_end = max(node.lineno, body[0].lineno - 1)
            out.append((node.lineno, header_end, node.end_lineno))
    return out


def _statement_of(stmts: list[tuple[int, int, int]], line: int) -> int | None:
    """Index of the innermost statement whose header contains ``line``."""
    for pick in (lambda s: s[0] <= line <= s[1], lambda s: s[0] <= line <= s[2]):
        hits = [i for i, s in enumerate(stmts) if pick(s)]
        if hits:
            return max(hits, key=lambda i: (stmts[i][0], -stmts[i][2]))
    return None


def _scan(source: str) -> tuple[list[tuple[Waiver, set[int]]], list[Violation], list[tuple[int, int, int]]]:
    """Return ((waiver, attached statement indexes), meta violations, statements)."""
    tiers = {r.id: r.tier for r in RULES}
    found: list[tuple[Waiver, set[int]]] = []
    meta: list[Violation] = []
    try:
        tree = ast.parse(source)
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return found, meta, []
    stmts = _statements(tree)
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        m = _WAIVER.match(tok.string)
        if not m:
            continue
        line = tok.start[0]
        rule_id, _, why = (part.strip() for part in m.group(1).partition(","))
        if not rule_id or not why or _NESTED.search(why):
            meta.append(Violation("meta.waiver-unknown", line,
                                  f"Malformed waiver at line {line}", hint=_FORM))
        elif tiers.get(rule_id) == "safety":
            meta.append(Violation("meta.waiver-safety", line,
                                  f"Waiver at line {line} names safety rule {rule_id}; safety rules cannot be waived"))
        elif rule_id not in tiers:
            meta.append(Violation("meta.waiver-unknown", line,
                                  f"Waiver at line {line} names unknown rule {rule_id}", hint=_FORM))
        else:
            standalone = tok.line[:tok.start[1]].strip() == ""
            target = line + 1 if standalone else line
            attached = {i for i, s in enumerate(stmts) if s[0] == target}
            if attached:
                found.append((Waiver(rule_id, line, why), attached))
            else:
                meta.append(Violation("meta.waiver-unknown", line,
                                      f"Waiver at line {line} is misplaced", hint=_PLACE))
    return found, meta, stmts


def scan_waivers(source: str) -> tuple[list[Waiver], list[Violation]]:
    """Return (valid, placed waivers, ``meta.waiver-*`` violations) for ``source``."""
    found, meta, _ = _scan(source)
    return [w for w, _ in found], meta


def parse_waivers(source: str) -> list[Waiver]:
    """Return the valid, placed ``# ste:`` waivers in ``source``."""
    return scan_waivers(source)[0]


def apply_waivers(
    violations: list[Violation], source: str
) -> tuple[list[Violation], list[Waiver], list[Violation]]:
    """Return (kept violations, waivers marked ``used`` or not, ``meta`` violations)."""
    found, meta, stmts = _scan(source)
    used: set[int] = set()
    kept = []
    for v in violations:
        idx = _statement_of(stmts, v.line) if v.line is not None else None
        hit = [k for k, (w, att) in enumerate(found) if w.rule_id == v.rule_id and idx in att]
        if hit:
            used.update(hit)
        else:
            kept.append(v)
    return kept, [dataclasses.replace(w, used=k in used) for k, (w, _) in enumerate(found)], meta
