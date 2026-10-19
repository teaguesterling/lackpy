"""Load a ``PathPolicy`` from Claude Code settings and ``.lackpy/config.toml``.

Sources, all unioned (Teague's ruling 4, 2026-10-18: managed settings included):

- Claude Code's managed settings (https://code.claude.com/docs/en/managed-settings,
  fetched 2026-10-18): ``MANAGED_SETTINGS``, then every ``*.json`` in the
  ``managed-settings.d/`` directory beside it, in alphabetical order. Hidden
  files are skipped. ``/p`` rules there are relative to that directory
  (a choice: the docs anchor ``/p`` for user settings at ``~/.claude``, the
  settings file's directory, and this follows the same shape).
- ``home/.claude/settings.json`` (``/p`` relative to ``home/.claude``).
- ``workspace/.claude/settings.json`` and ``settings.local.json`` (``/p``
  relative to the workspace).
- ``workspace/.lackpy/config.toml``, table ``[paths]``: ``allow``, ``deny``,
  ``additional_directories``. Any other key in ``[paths]`` is an error, so a
  misspelt ``deny`` is never silently ignored.

When a managed source sets ``allowManagedPermissionRulesOnly: true`` (at the
top level or inside ``permissions``; either is accepted, the stricter
reading), only managed sources apply: their rules and their additional
directories. The user, project, local and ``[paths]`` additional directories
are ignored too (security ruling, Task 4 fix round 1, reversing the brief's
design call: managed-only means the organization fixes the reachable roots).

Every settings location read here is passed to the policy as write-protected.

A missing file is skipped. A malformed file raises ``ValueError`` naming it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .rules import MANAGED_SETTINGS, PathPolicy, PathRule, managed_settings_path, parse_rule

__all__ = ["MANAGED_SETTINGS", "load_policy", "managed_settings_path"]


class _Collected:
    def __init__(self) -> None:
        self.rules: list[PathRule] = []
        self.dirs: list[Path] = []
        self._seen: set[tuple[str, str, str]] = set()

    def add_rules(self, texts: list[str], effect: str, *, anchor: Path, source: str,
                  home: Path, dedupe: bool = False) -> None:
        for text in texts:
            key = (effect, text.strip(), str(anchor))
            if dedupe and key in self._seen:
                continue
            self._seen.add(key)
            try:
                rule = parse_rule(text, effect=effect, anchor=anchor, source=source,  # type: ignore[arg-type]
                                  home=home)
            except ValueError as e:
                raise ValueError(f"{source}: {e}") from None
            if rule is not None:
                self.rules.append(rule)


def _string_list(value: Any, where: str, source: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{source}: {where} must be a list of strings")
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise ValueError(f"{path}: malformed settings file: {e}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: settings must be a JSON object")
    return data


def _permissions(data: dict[str, Any], path: Path) -> dict[str, list[str]]:
    perms = data.get("permissions", {})
    if not isinstance(perms, dict):
        raise ValueError(f"{path}: permissions must be an object")
    out = {}
    for key in ("allow", "deny", "ask", "additionalDirectories"):
        out[key] = _string_list(perms.get(key, []), f"permissions.{key}", path)
    return out


def _managed_only(data: dict[str, Any], path: Path) -> bool:
    perms = data.get("permissions", {})
    flags = [data.get("allowManagedPermissionRulesOnly", False),
             perms.get("allowManagedPermissionRulesOnly", False) if isinstance(perms, dict) else False]
    for flag in flags:
        if not isinstance(flag, bool):
            raise ValueError(f"{path}: allowManagedPermissionRulesOnly must be true or false")
    return any(flags)


def _directory(text: str, workspace: Path, home: Path) -> Path:
    if text == "~" or text.startswith("~/"):
        return home / text[2:]
    return workspace / text  # an absolute text replaces the workspace


def _read_paths_table(path: Path) -> dict[str, list[str]] | None:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise ValueError(f"{path}: malformed config file: {e}") from None
    if "paths" not in data:
        return None
    table = data["paths"]
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [paths] must be a table")
    unknown = set(table) - {"allow", "deny", "additional_directories"}
    if unknown:
        raise ValueError(f"{path}: unknown key(s) in [paths]: {', '.join(sorted(unknown))}")
    return {key: _string_list(table.get(key, []), f"paths.{key}", path)
            for key in ("allow", "deny", "additional_directories")}


def load_policy(workspace: Path, *, home: Path = Path.home(),
                managed: Path | None = MANAGED_SETTINGS) -> PathPolicy:
    """Build the ``PathPolicy`` for ``workspace`` from every settings source."""
    workspace, home = Path(workspace), Path(home)
    managed_set = _Collected()
    others = _Collected()
    managed_only = False

    if managed is not None:
        managed = Path(managed)
        dropins = managed.parent / "managed-settings.d"
        files = [managed]
        if dropins.is_dir():
            files += sorted(p for p in dropins.glob("*.json") if not p.name.startswith("."))
        for path in files:
            data = _read_json(path)
            if data is None:
                continue
            perms = _permissions(data, path)
            managed_only = managed_only or _managed_only(data, path)
            for effect in ("deny", "ask", "allow"):
                managed_set.add_rules(perms[effect], effect, anchor=managed.parent,
                                      source=str(path), home=home, dedupe=True)
            managed_set.dirs += [_directory(d, workspace, home) for d in perms["additionalDirectories"]]

    for path, anchor in ((home / ".claude" / "settings.json", home / ".claude"),
                         (workspace / ".claude" / "settings.json", workspace),
                         (workspace / ".claude" / "settings.local.json", workspace)):
        data = _read_json(path)
        if data is None:
            continue
        perms = _permissions(data, path)
        for effect in ("deny", "ask", "allow"):
            others.add_rules(perms[effect], effect, anchor=anchor, source=str(path), home=home)
        others.dirs += [_directory(d, workspace, home) for d in perms["additionalDirectories"]]

    config = workspace / ".lackpy" / "config.toml"
    table = _read_paths_table(config)
    if table is not None:
        for effect in ("deny", "allow"):
            others.add_rules(table[effect], effect, anchor=workspace, source=str(config), home=home)
        others.dirs += [_directory(d, workspace, home) for d in table["additional_directories"]]

    rules = managed_set.rules + ([] if managed_only else others.rules)
    dirs = managed_set.dirs + ([] if managed_only else others.dirs)
    protected = [home / ".claude", home / ".claude.json"]
    if managed is not None:
        protected += [managed, managed.parent / "managed-settings.d"]
    return PathPolicy(workspace, rules, extra_dirs=dirs, protected=protected)
