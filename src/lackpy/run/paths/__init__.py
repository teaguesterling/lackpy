"""File access for lackpy programs: the ``PathPolicy`` and the ``Path`` stand-in (stepy100 Plan 2, Task 4).

This package is part of the sandbox boundary. ``rules`` parses Claude-style
``Read(...)``/``Edit(...)`` permission rules and decides each access;
``sources`` loads them from settings files; ``proxy`` is the ``Path``
stand-in a program gets from ``pathlib``.
"""

from .proxy import make_path_class
from .rules import PathPolicy, PathRule, parse_rule, read_only_policy
from .sources import MANAGED_SETTINGS, load_policy

__all__ = ["MANAGED_SETTINGS", "PathPolicy", "PathRule", "load_policy", "make_path_class", "parse_rule",
           "read_only_policy"]
