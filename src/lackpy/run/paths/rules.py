"""Path rules and the ``PathPolicy`` (stepy100 Plan 2, Task 4).

This module is part of the sandbox boundary: ``PathPolicy.check`` decides every
file access a program makes through the ``Path`` stand-in.

The rule language is Claude Code's ``Read(...)`` and ``Edit(...)`` permission
rules, from https://code.claude.com/docs/en/permissions (fetched 2026-10-18).

Semantics:

1. Deny is checked first, then allow. A deny from any source wins. A ``Read``
   deny blocks reads and writes. An ``Edit`` deny blocks writes. A ``Read``
   allow grants reads only; an ``Edit`` allow grants writes only (stricter
   reading: upstream does not say that one implies the other).
2. Anchors: ``//p`` is absolute. ``~/p`` is under home. ``/p`` is relative to
   the rule's ``anchor`` (the workspace for workspace settings, ``~/.claude``
   for user settings, the managed settings directory for managed settings).
   ``p`` and ``./p`` are relative to the workspace.
3. Globs: ``*`` matches within one path segment; ``**`` as a whole segment
   matches zero or more segments (so ``d/**`` also matches ``d``). A bare
   filename (one segment, no ``/``, such as ``.env``) matches at any depth:
   for a deny, anywhere on the filesystem; for an allow, at any depth under the
   workspace (stricter reading). A single-segment directory pattern such as
   ``src/**`` matches at any depth for a deny, but only ``<workspace>/src`` for
   an allow. A deny also matches every path under a path it matches.
4. With no matching allow, read and write are allowed inside the workspace and
   the extra directories, and refused everywhere else.
5. Symlinks: an allow must hold for both the requested path (made absolute and
   normalized) and its ``resolve()``d target. A deny matches either. A write
   refuses when the final component is a symlink, and when any directory
   component is a symlink whose target is not writable.
7. Policy sources are never writable (security rulings, Task 4 fix rounds 1-2):
   any path with a ``.claude`` or ``.lackpy`` component anywhere (so a later
   run rooted in a subdirectory or an extra directory cannot be widened),
   ``<home>/.claude.json``, the managed settings file and its
   ``managed-settings.d``. This is checked
   first, on the lexical and resolved forms, case-insensitively; no rule or
   default root lifts it. A write there would widen the policy of every later run.
6. Matching is case-sensitive. Upstream does not document case sensitivity;
   case-sensitive is the stricter choice on a case-insensitive filesystem for
   allows, and is how Linux paths compare.

Stricter readings where the brief is silent: a bare ``Read`` or ``Edit`` (no
parentheses) deny matches every path, and a bare allow is ignored. A
malformed ``Read(...)`` or ``Edit(...)`` raises ``ValueError`` rather than
being dropped, because a dropped deny loosens the policy.

The check and the I/O that follows it are two system calls: a concurrent
process that swaps a symlink between them is not defended against.
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Iterable, Literal

from ..refusal import LackpyRefusal

_RULE_RE = re.compile(r"^(Read|Edit)(?:\((.*)\))?$", re.DOTALL)
_TOOL_RE = re.compile(r"^[A-Za-z_][\w-]*")


def managed_settings_path(platform: str) -> Path:
    """The managed settings file for ``platform`` (a ``sys.platform`` value)."""
    if platform == "darwin":
        return Path("/Library/Application Support/ClaudeCode/managed-settings.json")
    if platform == "win32":
        # Not the legacy C:\ProgramData path.
        return PureWindowsPath(r"C:\Program Files\ClaudeCode\managed-settings.json")  # type: ignore[return-value]
    return Path("/etc/claude-code/managed-settings.json")  # Linux and WSL


MANAGED_SETTINGS: Path = managed_settings_path(sys.platform)

_GLOB_CHARS = frozenset("*?[")
_PROTECTED_PARTS = frozenset({".claude", ".lackpy"})  # compared casefolded


@dataclass(frozen=True)
class PathRule:
    """One ``Read(...)`` or ``Edit(...)`` rule.

    ``home`` is the directory ``~/p`` patterns are under. It is an addition to
    the brief's fields, so that tests never depend on the real home.
    """

    kind: Literal["read", "edit"]
    effect: Literal["allow", "deny"]
    pattern: str
    anchor: Path
    source: str
    home: Path | None = None

    def text(self) -> str:
        return f"{self.kind.capitalize()}({self.pattern})"


def parse_rule(text: str, *, effect: Literal["allow", "deny", "ask"], anchor: Path,
               source: str, home: Path | None = None) -> PathRule | None:
    """Parse one permission rule string.

    Returns None for a rule of any other tool (``Write(...)`` included: Claude
    Code accepts it but never consults it). ``ask`` gives ``effect="deny"``.
    Raises ``ValueError`` for a malformed ``Read``/``Edit`` rule.
    """
    text = text.strip()
    match = _RULE_RE.match(text)
    if match is None:
        tool = _TOOL_RE.match(text)
        if tool and tool.group(0) in ("Read", "Edit"):
            raise ValueError(f"malformed rule {text!r} in {source}")
        return None
    tool, pattern = match.group(1), match.group(2)
    final_effect: Literal["allow", "deny"] = "allow" if effect == "allow" else "deny"
    if pattern is None:
        if final_effect == "allow":
            return None  # stricter reading: a bare allow would open the whole filesystem
        pattern = "//**"
    elif not pattern.strip():
        raise ValueError(f"malformed rule {text!r} in {source}: empty pattern")
    return PathRule(kind="read" if tool == "Read" else "edit", effect=final_effect,
                    pattern=pattern.strip(), anchor=Path(anchor), source=source,
                    home=Path(home) if home is not None else None)


def _norm(path: Path | str) -> str:
    """Absolute, lexically normalized, as a POSIX string."""
    return posixpath.normpath(os.path.abspath(os.fspath(path)))


def _real(path: Path | str) -> str:
    try:
        return os.fspath(Path(path).resolve())
    except (OSError, RuntimeError) as e:  # symlink loop
        raise LackpyRefusal(f"access to {path} (it cannot be resolved: {e})") from None


def _bases(path: Path) -> set[str]:
    """The lexical and the resolved form of a base directory."""
    return {_norm(path), _real(path)}


@lru_cache(maxsize=1024)
def _segment_re(segment: str) -> re.Pattern[str]:
    return re.compile(fnmatch.translate(segment))  # case-sensitive (item 6)


def _match(pattern: tuple[str, ...], parts: tuple[str, ...], prefix: bool) -> bool:
    """Match segments. With ``prefix``, a match of any leading run of ``parts`` counts."""

    @lru_cache(maxsize=None)
    def go(i: int, j: int) -> bool:
        if i == len(pattern):
            return j == len(parts) or prefix
        segment = pattern[i]
        if segment == "**":
            return any(go(i + 1, k) for k in range(j, len(parts) + 1))
        if j == len(parts):
            return False
        return bool(_segment_re(segment).fullmatch(parts[j])) and go(i + 1, j + 1)

    return go(0, 0)


def _segments(path: str) -> tuple[str, ...]:
    return tuple(p for p in path.split("/") if p)


def _compile(rule: PathRule, workspace: Path) -> list[tuple[str, ...]]:
    """The rule as absolute segment patterns, one per base form."""
    p = rule.pattern.replace("\\", "/") if os.sep == "\\" else rule.pattern
    deny = rule.effect == "deny"
    if p.startswith("//"):
        bases, rest = {"/"}, p[2:]
    elif p == "~" or p.startswith("~/"):
        bases, rest = _bases(rule.home if rule.home is not None else Path.home()), p[2:]
    elif p.startswith("/"):
        bases, rest = _bases(rule.anchor), p[1:]
    elif p.startswith("./"):
        bases, rest = _bases(workspace), p[2:]
    else:
        stripped = p.rstrip("/")
        segs = _segments(stripped)
        floating = len(segs) == 1 or (len(segs) == 2 and segs[1] == "**")
        if floating and deny:
            return [("**",) + segs]
        bases, rest = _bases(workspace), p
        if floating:
            rest = p if len(segs) == 2 else "**/" + stripped
    patterns = []
    for base in bases:
        joined = posixpath.normpath(posixpath.join(base, rest)) if rest else base
        patterns.append(_segments(joined))
    if deny:
        # Review M1: a symlinked literal prefix (~/.ssh -> R/keys) must not let
        # the target be read directly. Add the resolved prefix as another base.
        for segs in list(patterns):
            cut = next((i for i, seg in enumerate(segs) if _GLOB_CHARS & set(seg)), len(segs))
            if cut == 0:
                continue
            resolved = _segments(_real("/" + "/".join(segs[:cut]))) + segs[cut:]
            if resolved not in patterns:
                patterns.append(resolved)
    return patterns


class PathPolicy:
    """Decides each file access a program makes. See the module docstring."""

    def __init__(self, workspace: Path, rules: list[PathRule],
                 extra_dirs: list[Path] = [],  # noqa: B006 (never mutated)
                 *, protected: Iterable[Path] = ()) -> None:
        """``protected`` adds write-protected paths to the built-in ones.

        Built in: ``<workspace>/.claude``, ``<workspace>/.lackpy``,
        ``Path.home()/.claude``, ``MANAGED_SETTINGS`` and its
        ``managed-settings.d``. ``load_policy`` adds its own home and managed path.
        """
        self.workspace = Path(_norm(workspace))
        self.rules = list(rules)
        self.extra_dirs = [Path(_norm(self.workspace / d)) for d in extra_dirs]
        self._compiled = [(r, _compile(r, self.workspace)) for r in self.rules]
        roots = [self.workspace, *self.extra_dirs]
        self._roots_lexical = {b for r in roots for b in _bases(r)}
        self._roots_real = {_real(r) for r in roots}
        managed = Path(os.fspath(MANAGED_SETTINGS))
        builtin = [self.workspace / ".claude", self.workspace / ".lackpy", Path.home() / ".claude",
                   Path.home() / ".claude.json", managed, managed.parent / "managed-settings.d"]
        self.protected = [Path(_norm(self.workspace / p)) for p in (*builtin, *protected)]
        self._protected = {b.casefold() for p in self.protected for b in _bases(p)}

    def absolute(self, path: Path | str) -> Path:
        """``path`` joined to the workspace when relative, not normalized."""
        return self.workspace / path

    def _rules_match(self, path: str, kinds: tuple[str, ...], effect: str) -> PathRule | None:
        parts = _segments(path)
        for rule, patterns in self._compiled:
            if rule.effect != effect or rule.kind not in kinds:
                continue
            if any(_match(pat, parts, prefix=(effect == "deny")) for pat in patterns):
                return rule
        return None

    @staticmethod
    def _under(path: str, roots: set[str]) -> bool:
        return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)

    def _contains_a_root(self, path: str) -> bool:
        return any(self._under(root, {path}) for root in self._roots_lexical)

    def _decide(self, raw: Path, op: Literal["read", "write"]) -> str | None:
        """Return the refusal reason, or None when allowed."""
        lexical, real = _norm(raw), _real(raw)
        if op == "write":
            for form in (lexical, real):
                folded = form.casefold()
                if self._under(folded, self._protected) or _PROTECTED_PARTS & set(_segments(folded)):
                    return "it is a permission settings location, which lackpy never writes"
        deny_kinds = ("read",) if op == "read" else ("read", "edit")
        for form in (lexical, real):
            rule = self._rules_match(form, deny_kinds, "deny")
            if rule is not None:
                return f"denied by {rule.text()} in {rule.source}"
        allow_kind = ("read",) if op == "read" else ("edit",)
        for form, roots in ((lexical, self._roots_lexical), (real, self._roots_real)):
            if not self._under(form, roots) and self._rules_match(form, allow_kind, "allow") is None:
                where = "it" if form == lexical else f"its target {form}"
                return f"{where} is outside the workspace and no allow rule matches"
        return None

    def check(self, path: Path, op: Literal["read", "write"]) -> None:
        """Raise ``LackpyRefusal`` unless ``op`` on ``path`` is allowed."""
        if op not in ("read", "write"):
            raise ValueError(f"op must be 'read' or 'write', not {op!r}")
        raw = self.absolute(path)
        reason = self._decide(raw, op)
        if reason is None and op == "write":
            if raw.is_symlink():
                reason = "the final component is a symlink"
            else:
                for ancestor in reversed(raw.parents):
                    if self._contains_a_root(_norm(ancestor)):
                        continue  # at or above a root: the roots themselves were resolved
                    if ancestor.is_symlink() and self._decide(Path(_real(ancestor)), "write"):
                        reason = f"directory {ancestor} is a symlink to a place that is not writable"
                        break
        if reason is not None:
            verb = "reading" if op == "read" else "writing"
            raise LackpyRefusal(f"{verb} {_norm(raw)} ({reason})")

    def describe(self) -> list[str]:
        """One line per effective rule, and one per default directory."""
        lines = [f"allow read, write: {d} (default directory)"
                 for d in (self.workspace, *self.extra_dirs)]
        lines += [f"never write: {p} (permission settings)" for p in self.protected]
        for rule, patterns in self._compiled:
            ops = "read" if rule.kind == "read" else "write"
            if rule.kind == "read" and rule.effect == "deny":
                ops = "read, write"
            shown = " | ".join("/" + "/".join(p) for p in sorted(set(patterns)))
            lines.append(f"{rule.effect} {ops}: {rule.text()} -> {shown} (from {rule.source})")
        return lines
